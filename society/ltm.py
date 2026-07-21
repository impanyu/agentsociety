import json
import re
import uuid
from datetime import datetime, timezone

import chromadb

from society.textlen import count_tokens, truncate_to_tokens

_TERMINATORS = "。！？；.;"
_CONNECTIVES = ("然后", "并且", "而且", "同时", "接着", " and ", " then ")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SharedMemory:
    """Shared long-term memory with a normalize gate and consensus-based insert."""

    def __init__(
        self,
        embed_fn,
        llm=None,
        *,
        max_chars: int | None = None,
        max_tokens: int = 50,
        sim_threshold: float = 0.86,
        top_k: int = 5,
        collection_name: str | None = None,
    ):
        """
        Initialize SharedMemory.

        Args:
            embed_fn: async callable(list[str]) -> list[vector].
            llm: object with async .chat(prompt, system=None, bucket=...) -> str, or None.
            max_chars: DEPRECATED, ignored. Kept only so existing call sites
                passing `max_chars=` don't crash. Length is now measured in
                tokens (see `max_tokens`) since char counts are not uniform
                across languages (e.g. Chinese runs ~1 token/char while
                English runs several chars/token).
            max_tokens: max length of an atomic memory entry, in o200k_base
                tokens, before hard truncation.
            sim_threshold: cosine similarity (1 - distance) required to consider two
                entries candidates for consensus merge.
            top_k: default number of nearest neighbors to consider/return.
            collection_name: name of the underlying chroma collection. Default
                None auto-generates a unique name — chromadb's default clients
                share one in-process store, so same-name collections in the same
                process would silently share data.
        """
        self._embed_fn = embed_fn
        self._llm = llm
        self.max_chars = max_chars  # deprecated, unused; retained for compatibility
        self.max_tokens = max_tokens
        self.sim_threshold = sim_threshold
        self.top_k = top_k
        if collection_name is None:
            collection_name = f"agent_society_ltm_{uuid.uuid4().hex[:8]}"
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    # ------------------------------------------------------------------
    # normalize gate
    # ------------------------------------------------------------------

    def _needs_normalize(self, text: str) -> bool:
        if count_tokens(text) > self.max_tokens:
            return True
        terminator_count = sum(text.count(ch) for ch in _TERMINATORS)
        if terminator_count > 1:
            return True
        if any(conn in text for conn in _CONNECTIVES):
            return True
        return False

    def _fallback_split(self, text: str) -> list[str]:
        parts = re.split(f"[{re.escape(_TERMINATORS)}!]", text)
        entries = [p.strip() for p in parts if p.strip()]
        return [truncate_to_tokens(e, self.max_tokens) for e in entries]

    async def _normalize(self, text: str) -> list[str]:
        """Split text into atomic memory strings, calling the LLM only when needed."""
        if not self._needs_normalize(text):
            return [text]

        if self._llm is None:
            return self._fallback_split(text)

        prompt = (
            "Split the following memory into a JSON array of short, atomic, "
            "independent memory statements. Reply with ONLY the JSON array. "
            f"Each atomic statement should be at most ~{self.max_tokens} tokens "
            "(one complete event).\n\n"
            f"Memory: {text}"
        )
        reply = await self._llm.chat(prompt, system=None, bucket="normalize")

        match = re.search(r"\[.*\]", reply, re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
                entries = [str(e).strip() for e in parsed if str(e).strip()]
                if entries:
                    return [truncate_to_tokens(e, self.max_tokens) for e in entries]
            except (ValueError, TypeError):
                pass

        return self._fallback_split(text)

    # ------------------------------------------------------------------
    # consensus insert
    # ------------------------------------------------------------------

    async def _consensus_insert(
        self,
        owners: list[str],
        text: str,
        tick: int,
        source: str,
        story_order: int | None = None,
        story_time: str | None = None,
    ) -> dict:
        embedding = (await self._embed_fn([text]))[0]

        candidates = []
        if self._collection.count() > 0:
            results = self._collection.query(
                query_embeddings=[embedding],
                n_results=min(self.top_k, self._collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            ids = results["ids"][0]
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                sim = 1 - dist
                if sim >= self.sim_threshold:
                    candidates.append({"id": cid, "text": doc, "meta": meta, "sim": sim})

        match_idx = -1
        if candidates and self._llm is not None:
            lines = "\n".join(
                f"{i}: {c['text']}" for i, c in enumerate(candidates)
            )
            prompt = (
                f"New memory: {text}\n\nExisting candidate memories:\n{lines}\n\n"
                "Which existing candidate (by index) is semantically equivalent to "
                "the new memory? Reply with only the index number, or -1 if none match."
            )
            reply = await self._llm.chat(prompt, system=None, bucket="consensus")
            m = re.search(r"-?\d+", reply or "")
            if m:
                idx = int(m.group())
                if 0 <= idx < len(candidates):
                    match_idx = idx

        new_owners = sorted(set(owners))

        if match_idx == -1:
            new_id = uuid.uuid4().hex
            metadata = {
                "owners": json.dumps(new_owners),
                "created_at": _now_iso(),
                "source": source,
                "tick": tick,
            }
            for o in new_owners:
                metadata[f"owner_{o}"] = True
            if story_order is not None:
                metadata["story_order"] = story_order
            if story_time is not None:
                metadata["story_time"] = story_time
            self._collection.add(
                ids=[new_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
            )
            return {"id": new_id, "text": text, "merged": False, "owners": new_owners}

        candidate = candidates[match_idx]
        cid = candidate["id"]
        meta = candidate["meta"]
        existing_owners = set(json.loads(meta.get("owners", "[]")))
        merged_owners = sorted(existing_owners | set(new_owners))

        keep_text = candidate["text"]
        update_kwargs = {}
        if len(text) < len(candidate["text"]):
            keep_text = text
            update_kwargs["documents"] = [keep_text]
            update_kwargs["embeddings"] = [embedding]

        update_metadata = {
            "owners": json.dumps(merged_owners),
            "tick": tick,
        }
        for o in new_owners:
            update_metadata[f"owner_{o}"] = True

        existing_story_order = meta.get("story_order")
        existing_story_time = meta.get("story_time")
        eff_existing = (
            existing_story_order if existing_story_order is not None else float("inf")
        )
        eff_new = story_order if story_order is not None else float("inf")
        if eff_existing != float("inf") or eff_new != float("inf"):
            if eff_new < eff_existing:
                kept_story_order, kept_story_time = story_order, story_time
            else:
                kept_story_order, kept_story_time = (
                    existing_story_order,
                    existing_story_time,
                )
            update_metadata["story_order"] = kept_story_order
            if kept_story_time is not None:
                update_metadata["story_time"] = kept_story_time

        update_kwargs["metadatas"] = [update_metadata]
        self._collection.update(ids=[cid], **update_kwargs)

        return {"id": cid, "text": keep_text, "merged": True, "owners": merged_owners}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def remember(
        self,
        agent_id: str,
        text: str,
        tick: int = 0,
        source: str = "runtime",
        story_order: int | None = None,
        story_time: str | None = None,
    ) -> list[dict]:
        """Normalize text into atomic entries and consensus-insert each one.

        story_order/story_time (when given) are attached to every atomic
        entry produced from `text` -- they describe when in the story's
        timeline this memory happened, not when it was recorded (`tick`).
        """
        entries = await self._normalize(text)
        results = []
        for entry in entries:
            results.append(
                await self._consensus_insert(
                    [agent_id],
                    entry,
                    tick,
                    source,
                    story_order=story_order,
                    story_time=story_time,
                )
            )
        return results

    async def remember_atomic(
        self,
        owners: list[str],
        text: str,
        tick: int = 0,
        source: str = "sediment",
        story_order: int | None = None,
        story_time: str | None = None,
    ) -> dict | None:
        """Deposit a PRE-ATOMIZED fragment (already one complete event) owned by
        `owners` (list[str], >=1). Skips the normalize/split gate (the caller has
        already atomized), applies ONLY the token cap, then consensus-inserts with
        the full owner set. Returns the insert result dict, or None if text is
        empty/whitespace after stripping."""
        text = text.strip()
        if not text:
            return None
        if not owners:
            raise ValueError("remember_atomic requires at least one owner")

        text = truncate_to_tokens(text, self.max_tokens)
        return await self._consensus_insert(
            sorted(set(owners)),
            text,
            tick,
            source,
            story_order=story_order,
            story_time=story_time,
        )

    async def recall(self, agent_id: str, query: str, top_k: int = 5) -> list[dict]:
        """Return the top_k entries owned by agent_id, ranked by similarity to query."""
        if self._collection.count() == 0:
            return []
        embedding = (await self._embed_fn([query]))[0]
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, self._collection.count()),
            where={f"owner_{agent_id}": True},
            include=["documents", "metadatas"],
        )
        docs = results["documents"][0]
        ids = results["ids"][0]
        return [{"id": i, "text": d} for i, d in zip(ids, docs)]

    def forget(self, agent_id: str, memory_id: str) -> bool:
        """Remove agent_id from the entry's owners; delete the entry if it becomes
        ownerless. Returns False if the entry doesn't exist or agent isn't an owner."""
        got = self._collection.get(ids=[memory_id], include=["metadatas"])
        if not got["ids"]:
            return False
        meta = got["metadatas"][0]
        owners = json.loads(meta.get("owners", "[]"))
        if agent_id not in owners:
            return False
        owners = [o for o in owners if o != agent_id]
        if not owners:
            self._collection.delete(ids=[memory_id])
        else:
            self._collection.update(
                ids=[memory_id],
                metadatas=[
                    {
                        "owners": json.dumps(owners),
                        f"owner_{agent_id}": None,
                    }
                ],
            )
        return True

    async def revise(
        self, agent_id: str, memory_id: str, new_text: str, tick: int = 0
    ) -> list[dict]:
        """Forget the old entry (for agent_id) then remember the new text."""
        self.forget(agent_id, memory_id)
        return await self.remember(agent_id, new_text, tick=tick)

    def all_entries(self) -> list[dict]:
        """Return every stored entry as {"id", "text", "owners", "meta"}."""
        if self._collection.count() == 0:
            return []
        got = self._collection.get(include=["documents", "metadatas"])
        entries = []
        for eid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
            owners = sorted(json.loads(meta.get("owners", "[]")))
            entries.append({"id": eid, "text": doc, "owners": owners, "meta": meta})
        return entries

    def export(self) -> list[dict]:
        """Holographic export: every entry with its embedding, for checkpointing.

        Returns a list of {"id", "text", "owners", "meta", "embedding"}, where
        "meta" is {"created_at", "source", "tick"} (plus "story_order" /
        "story_time" when the entry carries them) and "embedding" is the raw
        vector (list[float]) fetched straight from the collection.
        """
        if self._collection.count() == 0:
            return []
        got = self._collection.get(include=["documents", "metadatas", "embeddings"])
        entries = []
        embeddings = got.get("embeddings")
        for i, (eid, doc, meta) in enumerate(
            zip(got["ids"], got["documents"], got["metadatas"])
        ):
            owners = sorted(json.loads(meta.get("owners", "[]")))
            embedding = None
            if embeddings is not None:
                vec = embeddings[i]
                if vec is not None:
                    embedding = [float(x) for x in vec]
            meta_out = {
                "created_at": meta.get("created_at"),
                "source": meta.get("source"),
                "tick": meta.get("tick"),
            }
            if meta.get("story_order") is not None:
                meta_out["story_order"] = meta.get("story_order")
            if meta.get("story_time") is not None:
                meta_out["story_time"] = meta.get("story_time")
            entries.append(
                {
                    "id": eid,
                    "text": doc,
                    "owners": owners,
                    "meta": meta_out,
                    "embedding": embedding,
                }
            )
        return entries

    async def restore(self, entries: list[dict]) -> None:
        """Re-populate the collection from `export()`'s output, preserving
        ids/owners/meta/embeddings exactly (no re-normalization, no
        consensus merge, no new embed calls for entries that already carry
        an embedding). Entries missing an embedding recompute it via
        `_embed_fn` (kept async to support that path)."""
        if not entries:
            return

        missing_idx = [i for i, e in enumerate(entries) if not e.get("embedding")]
        computed = {}
        if missing_idx:
            texts = [entries[i]["text"] for i in missing_idx]
            vectors = await self._embed_fn(texts)
            for i, vec in zip(missing_idx, vectors):
                computed[i] = list(vec)

        ids = []
        docs = []
        embeddings = []
        metadatas = []
        for i, entry in enumerate(entries):
            owners = entry.get("owners", [])
            meta = entry.get("meta", {}) or {}
            metadata = {"owners": json.dumps(list(owners))}
            for key in ("created_at", "source", "tick", "story_order", "story_time"):
                value = meta.get(key)
                if value is not None:
                    metadata[key] = value
            for owner in owners:
                metadata[f"owner_{owner}"] = True

            ids.append(entry["id"])
            docs.append(entry["text"])
            embeddings.append(entry.get("embedding") or computed[i])
            metadatas.append(metadata)

        self._collection.add(
            ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas
        )

    def stats(self) -> dict:
        """Return {"total", "shared", "ratio"} where shared = entries with >=2 owners."""
        entries = self.all_entries()
        total = len(entries)
        shared = sum(1 for e in entries if len(e["owners"]) >= 2)
        ratio = (shared / total) if total else 0.0
        return {"total": total, "shared": shared, "ratio": ratio}
