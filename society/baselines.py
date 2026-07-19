"""Faithful baseline memory backends for comparison against `society.ltm.SharedMemory`.

Each class here is a drop-in, duck-typed replacement for `SharedMemory`: same
constructor keyword shape, same `remember`/`recall`/`forget`/`revise`/
`all_entries`/`export`/`restore`/`stats` methods and return shapes. None of
them get our normalize gate or consensus merge -- that machinery is the
contribution under test, so giving it to a baseline would invalidate the
comparison. Every store here is a plain in-memory list of rows with
embeddings stored as plain Python lists (no numpy); ranking is real cosine
similarity, computed in pure Python, from vectors returned by `embed_fn`.

`make_memory(kind, embed_fn, llm=None, **kw)` is the factory experiment
runners should call to select a backend by name.
"""

import math
import re
import uuid
from datetime import datetime, timezone

from society.ltm import SharedMemory

_DEFAULT_IMPORTANCE = 5


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ==========================================================================
# 1) GenerativeAgentsMemory -- Park et al. 2023, "Generative Agents: Interactive
#    Simulacra of Human Behavior" (memory stream)
# ==========================================================================


class GenerativeAgentsMemory:
    """Per-agent private memory stream (Park et al. 2023).

    Fidelity note: faithful to the paper's core mechanism -- every agent has
    its own private stream, and `remember` stores a SEPARATE row for each
    owner in the call's owner set (a runtime `remember(agent_id, ...)` call
    has owner set {agent_id}); there is no consensus/dedup step, so the same
    shared scene recorded by N agents produces N independent rows, which is
    exactly the footprint-inflation behavior this baseline is meant to
    exhibit. Retrieval uses the paper's three-term score
    (recency + importance + relevance), min-max normalized per component
    then weighted-summed with weights RECENCY_W/IMPORTANCE_W/RELEVANCE_W
    (default 1.0 each, matching the paper's un-tuned equal weighting).
    Importance is scored once at insert time via a single `llm.chat` call
    (paper: "poignancy" 1-10 rating) and cached on the row; recency uses an
    exponential decay over a synthetic access-tick counter that advances on
    every `recall` (paper uses wall-clock hours since last retrieval -- we
    substitute a monotonic tick counter since the simulator has no wall
    clock, which is the charitable, deterministic reading for testing).
    Simplification left out: the paper's separate "reflection" tree
    (higher-level synthesized memories) is not implemented -- reflections
    would themselves be additional stream entries produced via an LLM
    summarization pass, which is orthogonal to the retrieval-scoring
    mechanism this adapter is faithful to and is not needed to compare
    against consensus-compressed storage.
    """

    RECENCY_W = 1.0
    IMPORTANCE_W = 1.0
    RELEVANCE_W = 1.0
    DECAY = 0.1

    def __init__(self, embed_fn, llm=None, *, top_k: int = 5, collection_name=None, **kwargs):
        self._embed_fn = embed_fn
        self._llm = llm
        self._rows = {}  # id -> row dict
        self._clock = 0  # monotonic access-tick counter

    async def _score_importance(self, text: str) -> int:
        if self._llm is None:
            return _DEFAULT_IMPORTANCE
        prompt = (
            "On a scale of 1 to 10, where 1 is mundane and 10 is extremely "
            "important/poignant, rate the importance of this memory. Reply "
            f"with only the integer.\n\nMemory: {text}"
        )
        reply = await self._llm.chat(prompt, system=None, bucket="importance")
        match = re.search(r"-?\d+", reply or "")
        if not match:
            return _DEFAULT_IMPORTANCE
        try:
            val = int(match.group())
            return max(1, min(10, val))
        except (ValueError, TypeError):
            return _DEFAULT_IMPORTANCE

    async def remember(
        self,
        agent_id: str,
        text: str,
        tick: int = 0,
        source: str = "runtime",
        story_order=None,
        story_time=None,
    ) -> list[dict]:
        owners = [agent_id]
        embedding = (await self._embed_fn([text]))[0]
        importance = await self._score_importance(text)
        results = []
        for owner in owners:
            row_id = uuid.uuid4().hex
            self._clock += 1
            row = {
                "id": row_id,
                "text": text,
                "owner": owner,
                "embedding": list(embedding),
                "meta": {
                    "created_at": _now_iso(),
                    "source": source,
                    "tick": tick,
                    "importance": importance,
                    "last_access": self._clock,
                },
            }
            if story_order is not None:
                row["meta"]["story_order"] = story_order
            if story_time is not None:
                row["meta"]["story_time"] = story_time
            self._rows[row_id] = row
            results.append({"id": row_id, "text": text, "merged": False, "owners": [owner]})
        return results

    async def recall(self, agent_id: str, query: str, top_k: int = 5) -> list[dict]:
        candidates = [r for r in self._rows.values() if r["owner"] == agent_id]
        if not candidates:
            return []
        q_emb = (await self._embed_fn([query]))[0]

        self._clock += 1  # one "current time" tick shared by all candidates
        now = self._clock
        raw = []
        for row in candidates:
            ticks_since = now - row["meta"]["last_access"]
            recency = math.exp(-self.DECAY * ticks_since)
            importance = row["meta"]["importance"] / 10.0
            relevance = _cosine(q_emb, row["embedding"])
            raw.append((row, recency, importance, relevance))

        def _norm(vals):
            lo, hi = min(vals), max(vals)
            if hi - lo < 1e-12:
                return [1.0 for _ in vals]
            return [(v - lo) / (hi - lo) for v in vals]

        recencies = _norm([r[1] for r in raw])
        importances = _norm([r[2] for r in raw])
        relevances = _norm([r[3] for r in raw])

        scored = []
        for (row, _, _, _), rec_n, imp_n, rel_n in zip(raw, recencies, importances, relevances):
            score = (
                self.RECENCY_W * rec_n
                + self.IMPORTANCE_W * imp_n
                + self.RELEVANCE_W * rel_n
            )
            scored.append((score, row))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[:top_k]
        for _, row in top:
            row["meta"]["last_access"] = now
        return [{"id": row["id"], "text": row["text"]} for _, row in top]

    def forget(self, agent_id: str, memory_id: str) -> bool:
        row = self._rows.get(memory_id)
        if row is None or row["owner"] != agent_id:
            return False
        del self._rows[memory_id]
        return True

    async def revise(self, agent_id: str, memory_id: str, new_text: str, tick: int = 0) -> list[dict]:
        self.forget(agent_id, memory_id)
        return await self.remember(agent_id, new_text, tick=tick)

    def all_entries(self) -> list[dict]:
        return [
            {"id": r["id"], "text": r["text"], "owners": [r["owner"]], "meta": r["meta"]}
            for r in self._rows.values()
        ]

    def export(self) -> list[dict]:
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "owners": [r["owner"]],
                "meta": dict(r["meta"]),
                "embedding": list(r["embedding"]),
            }
            for r in self._rows.values()
        ]

    async def restore(self, entries: list[dict]) -> None:
        if not entries:
            return
        missing_idx = [i for i, e in enumerate(entries) if not e.get("embedding")]
        computed = {}
        if missing_idx:
            texts = [entries[i]["text"] for i in missing_idx]
            vectors = await self._embed_fn(texts)
            for i, vec in zip(missing_idx, vectors):
                computed[i] = list(vec)

        # Advance the monotonic access-tick clock past the highest
        # `last_access` being restored so a subsequent `recall` never
        # computes a large-negative `ticks_since` (which would overflow
        # `math.exp`) against a fresh instance's clock starting at 0.
        self._clock = max(
            [self._clock]
            + [int(e.get("meta", {}).get("last_access", 0) or 0) for e in entries]
        )

        for i, entry in enumerate(entries):
            owners = entry.get("owners", [])
            owner = owners[0] if owners else None
            meta = dict(entry.get("meta", {}) or {})
            meta.setdefault("importance", _DEFAULT_IMPORTANCE)
            meta.setdefault("last_access", self._clock)
            self._rows[entry["id"]] = {
                "id": entry["id"],
                "text": entry["text"],
                "owner": owner,
                "embedding": entry.get("embedding") or computed[i],
                "meta": meta,
            }

    def stats(self) -> dict:
        entries = self.all_entries()
        total = len(entries)
        shared = sum(1 for e in entries if len(e["owners"]) >= 2)
        ratio = (shared / total) if total else 0.0
        return {"total": total, "shared": shared, "ratio": ratio}


# ==========================================================================
# 2) GMemory -- Zhang et al. 2025 hierarchical graph memory
# ==========================================================================


class GMemory:
    """Single shared store with a hierarchical tier tag (G-Memory, 2025).

    Fidelity note: faithful to the paper's central design choice that memory
    is a SHARED graph rather than private per-agent streams -- every
    `remember` call appends a row visible to any agent's `recall` (default
    `owner_scope=False`, i.e. cross-agent retrieval; pass `owner_scope=True`
    to `recall` to restrict to the calling agent's own writes, exposed for
    completeness but not the default since G-Memory's whole point is shared
    retrieval). Each row carries a `tier` tag ("interaction" for raw
    `remember` observations, "insight"/"query" reserved for LLM-distilled
    tiers) mirroring the paper's insight/query/interaction hierarchy, but
    only the "interaction" tier is populated by plain `remember` -- the
    paper's higher tiers are produced by a separate distillation pass over
    accumulated interactions, which is out of scope for a like-for-like
    comparison of the raw-storage mechanism. No atomic splitting and no
    dedup: the full observation text is stored as one row, owner={agent_id},
    appended every call even if a prior call stored identical text.
    `stats()`'s `shared` counts rows whose owner set has length >=2; since
    there is no cross-insert owner merge, `shared` stays ~0 under normal
    runtime use (this baseline shares the STORE across agents, not the
    per-entry OWNER SET) -- that is the faithful, expected reading, not a
    bug.
    """

    def __init__(self, embed_fn, llm=None, *, top_k: int = 5, collection_name=None, **kwargs):
        self._embed_fn = embed_fn
        self._llm = llm
        self._rows = {}

    async def remember(
        self,
        agent_id: str,
        text: str,
        tick: int = 0,
        source: str = "runtime",
        story_order=None,
        story_time=None,
    ) -> list[dict]:
        embedding = (await self._embed_fn([text]))[0]
        row_id = uuid.uuid4().hex
        meta = {
            "created_at": _now_iso(),
            "source": source,
            "tick": tick,
            "tier": "interaction",
        }
        if story_order is not None:
            meta["story_order"] = story_order
        if story_time is not None:
            meta["story_time"] = story_time
        self._rows[row_id] = {
            "id": row_id,
            "text": text,
            "owners": [agent_id],
            "embedding": list(embedding),
            "meta": meta,
        }
        return [{"id": row_id, "text": text, "merged": False, "owners": [agent_id]}]

    async def recall(
        self, agent_id: str, query: str, top_k: int = 5, owner_scope: bool = False
    ) -> list[dict]:
        rows = list(self._rows.values())
        if owner_scope:
            rows = [r for r in rows if agent_id in r["owners"]]
        if not rows:
            return []
        q_emb = (await self._embed_fn([query]))[0]
        scored = [(_cosine(q_emb, r["embedding"]), r) for r in rows]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[:top_k]
        return [{"id": r["id"], "text": r["text"]} for _, r in top]

    def forget(self, agent_id: str, memory_id: str) -> bool:
        row = self._rows.get(memory_id)
        if row is None or agent_id not in row["owners"]:
            return False
        row["owners"] = [o for o in row["owners"] if o != agent_id]
        if not row["owners"]:
            del self._rows[memory_id]
        return True

    async def revise(self, agent_id: str, memory_id: str, new_text: str, tick: int = 0) -> list[dict]:
        self.forget(agent_id, memory_id)
        return await self.remember(agent_id, new_text, tick=tick)

    def all_entries(self) -> list[dict]:
        return [
            {"id": r["id"], "text": r["text"], "owners": list(r["owners"]), "meta": r["meta"]}
            for r in self._rows.values()
        ]

    def export(self) -> list[dict]:
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "owners": list(r["owners"]),
                "meta": dict(r["meta"]),
                "embedding": list(r["embedding"]),
            }
            for r in self._rows.values()
        ]

    async def restore(self, entries: list[dict]) -> None:
        if not entries:
            return
        missing_idx = [i for i, e in enumerate(entries) if not e.get("embedding")]
        computed = {}
        if missing_idx:
            texts = [entries[i]["text"] for i in missing_idx]
            vectors = await self._embed_fn(texts)
            for i, vec in zip(missing_idx, vectors):
                computed[i] = list(vec)

        for i, entry in enumerate(entries):
            meta = dict(entry.get("meta", {}) or {})
            meta.setdefault("tier", "interaction")
            self._rows[entry["id"]] = {
                "id": entry["id"],
                "text": entry["text"],
                "owners": list(entry.get("owners", [])),
                "embedding": entry.get("embedding") or computed[i],
                "meta": meta,
            }

    def stats(self) -> dict:
        entries = self.all_entries()
        total = len(entries)
        shared = sum(1 for e in entries if len(e["owners"]) >= 2)
        ratio = (shared / total) if total else 0.0
        return {"total": total, "shared": shared, "ratio": ratio}


# ==========================================================================
# 3) CollaborativeMemory -- access-controlled shared fragments (2025)
# ==========================================================================


class CollaborativeMemory:
    """Shared fragment store gated by a per-fragment read ACL (2025).

    Fidelity note: faithful to the paper's core mechanism -- a shared pool
    of immutable fragments, each carrying an access-control list of agents
    permitted to READ it (initialized to the writer, {agent_id}), plus
    provenance (source, tick, created_at). This is the axis that
    distinguishes the baseline from our consensus store: we deduplicate
    COPIES across agents, this baseline instead gates READS on a single
    copy -- `recall` filters candidates to fragments whose ACL contains
    `agent_id` *before* ranking by cosine relevance, so an agent with no
    grant sees nothing regardless of relevance. No merge/dedup of duplicate
    fragment text: two agents writing identical text produce two fragments.
    `grant(memory_id, agent_id)` extends a fragment's ACL (the paper's
    sharing/delegation primitive) and is the only way `stats()`'s `shared`
    (ACL size >= 2) becomes nonzero -- under plain runtime `remember` calls
    ACLs start single-owner, so `shared` ~0 until an explicit grant, which
    is documented as expected rather than a gap. `forget(agent_id, id)`
    mirrors `SharedMemory.forget`: it revokes read access for `agent_id`
    and deletes the fragment once its ACL is empty.
    """

    def __init__(self, embed_fn, llm=None, *, top_k: int = 5, collection_name=None, **kwargs):
        self._embed_fn = embed_fn
        self._llm = llm
        self._rows = {}

    async def remember(
        self,
        agent_id: str,
        text: str,
        tick: int = 0,
        source: str = "runtime",
        story_order=None,
        story_time=None,
    ) -> list[dict]:
        embedding = (await self._embed_fn([text]))[0]
        row_id = uuid.uuid4().hex
        meta = {
            "created_at": _now_iso(),
            "source": source,
            "tick": tick,
        }
        if story_order is not None:
            meta["story_order"] = story_order
        if story_time is not None:
            meta["story_time"] = story_time
        self._rows[row_id] = {
            "id": row_id,
            "text": text,
            "acl": {agent_id},
            "embedding": list(embedding),
            "meta": meta,
        }
        return [{"id": row_id, "text": text, "merged": False, "owners": [agent_id]}]

    async def recall(self, agent_id: str, query: str, top_k: int = 5) -> list[dict]:
        rows = [r for r in self._rows.values() if agent_id in r["acl"]]
        if not rows:
            return []
        q_emb = (await self._embed_fn([query]))[0]
        scored = [(_cosine(q_emb, r["embedding"]), r) for r in rows]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        top = scored[:top_k]
        return [{"id": r["id"], "text": r["text"]} for _, r in top]

    def grant(self, memory_id: str, agent_id: str) -> bool:
        """Extend a fragment's read ACL to include `agent_id`. Returns False
        if the fragment doesn't exist."""
        row = self._rows.get(memory_id)
        if row is None:
            return False
        row["acl"].add(agent_id)
        return True

    def forget(self, agent_id: str, memory_id: str) -> bool:
        """Revoke agent_id's read access; delete the fragment once its ACL
        is empty. Returns False if the fragment doesn't exist or agent_id
        was never a reader."""
        row = self._rows.get(memory_id)
        if row is None or agent_id not in row["acl"]:
            return False
        row["acl"].discard(agent_id)
        if not row["acl"]:
            del self._rows[memory_id]
        return True

    async def revise(self, agent_id: str, memory_id: str, new_text: str, tick: int = 0) -> list[dict]:
        self.forget(agent_id, memory_id)
        return await self.remember(agent_id, new_text, tick=tick)

    def all_entries(self) -> list[dict]:
        return [
            {"id": r["id"], "text": r["text"], "owners": sorted(r["acl"]), "meta": r["meta"]}
            for r in self._rows.values()
        ]

    def export(self) -> list[dict]:
        return [
            {
                "id": r["id"],
                "text": r["text"],
                "owners": sorted(r["acl"]),
                "meta": dict(r["meta"]),
                "embedding": list(r["embedding"]),
            }
            for r in self._rows.values()
        ]

    async def restore(self, entries: list[dict]) -> None:
        if not entries:
            return
        missing_idx = [i for i, e in enumerate(entries) if not e.get("embedding")]
        computed = {}
        if missing_idx:
            texts = [entries[i]["text"] for i in missing_idx]
            vectors = await self._embed_fn(texts)
            for i, vec in zip(missing_idx, vectors):
                computed[i] = list(vec)

        for i, entry in enumerate(entries):
            self._rows[entry["id"]] = {
                "id": entry["id"],
                "text": entry["text"],
                "acl": set(entry.get("owners", [])),
                "embedding": entry.get("embedding") or computed[i],
                "meta": dict(entry.get("meta", {}) or {}),
            }

    def stats(self) -> dict:
        entries = self.all_entries()
        total = len(entries)
        shared = sum(1 for e in entries if len(e["owners"]) >= 2)
        ratio = (shared / total) if total else 0.0
        return {"total": total, "shared": shared, "ratio": ratio}


# ==========================================================================
# factory
# ==========================================================================

_REGISTRY = {
    "consensus": SharedMemory,
    "generative_agents": GenerativeAgentsMemory,
    "g_memory": GMemory,
    "collaborative": CollaborativeMemory,
}


def make_memory(kind: str, embed_fn, llm=None, **kw):
    """Return an instance of the memory backend named by `kind`.

    kind in {"consensus", "generative_agents", "g_memory", "collaborative"}.
    Raises ValueError for an unknown kind.
    """
    try:
        cls = _REGISTRY[kind]
    except KeyError:
        raise ValueError(
            f"unknown memory kind {kind!r}; expected one of {sorted(_REGISTRY)}"
        )
    return cls(embed_fn, llm=llm, **kw)
