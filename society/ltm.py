import json
import re
import uuid
from datetime import datetime, timezone

import chromadb

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
        max_chars: int = 80,
        sim_threshold: float = 0.86,
        top_k: int = 5,
        collection_name: str = "agent_society_ltm",
    ):
        """
        Initialize SharedMemory.

        Args:
            embed_fn: async callable(list[str]) -> list[vector].
            llm: object with async .chat(prompt, system=None, bucket=...) -> str, or None.
            max_chars: max length of an atomic memory entry before hard truncation.
            sim_threshold: cosine similarity (1 - distance) required to consider two
                entries candidates for consensus merge.
            top_k: default number of nearest neighbors to consider/return.
            collection_name: name of the underlying chroma collection.
        """
        self._embed_fn = embed_fn
        self._llm = llm
        self.max_chars = max_chars
        self.sim_threshold = sim_threshold
        self.top_k = top_k
        self._client = chromadb.Client()
        self._collection = self._client.get_or_create_collection(
            name=collection_name, metadata={"hnsw:space": "cosine"}
        )

    # ------------------------------------------------------------------
    # normalize gate
    # ------------------------------------------------------------------

    def _needs_normalize(self, text: str) -> bool:
        if len(text) > self.max_chars:
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
        return [e[: self.max_chars] for e in entries]

    async def _normalize(self, text: str) -> list[str]:
        """Split text into atomic memory strings, calling the LLM only when needed."""
        if not self._needs_normalize(text):
            return [text]

        if self._llm is None:
            return self._fallback_split(text)

        prompt = (
            "Split the following memory into a JSON array of short, atomic, "
            "independent memory statements. Reply with ONLY the JSON array.\n\n"
            f"Memory: {text}"
        )
        reply = await self._llm.chat(prompt, system=None, bucket="normalize")

        match = re.search(r"\[.*\]", reply, re.S)
        if match:
            try:
                parsed = json.loads(match.group(0))
                entries = [str(e).strip() for e in parsed if str(e).strip()]
                if entries:
                    return [e[: self.max_chars] for e in entries]
            except (ValueError, TypeError):
                pass

        return self._fallback_split(text)

    # ------------------------------------------------------------------
    # consensus insert
    # ------------------------------------------------------------------

    async def _consensus_insert(
        self, agent_id: str, text: str, tick: int, source: str
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

        if match_idx == -1:
            new_id = uuid.uuid4().hex
            metadata = {
                "owners": json.dumps([agent_id]),
                f"owner_{agent_id}": True,
                "created_at": _now_iso(),
                "source": source,
                "tick": tick,
            }
            self._collection.add(
                ids=[new_id],
                documents=[text],
                embeddings=[embedding],
                metadatas=[metadata],
            )
            return {"id": new_id, "text": text, "merged": False, "owners": [agent_id]}

        candidate = candidates[match_idx]
        cid = candidate["id"]
        meta = candidate["meta"]
        owners = set(json.loads(meta.get("owners", "[]")))
        owners.add(agent_id)
        owners = sorted(owners)

        keep_text = candidate["text"]
        update_kwargs = {}
        if len(text) < len(candidate["text"]):
            keep_text = text
            update_kwargs["documents"] = [keep_text]
            update_kwargs["embeddings"] = [embedding]

        update_kwargs["metadatas"] = [
            {
                "owners": json.dumps(owners),
                f"owner_{agent_id}": True,
                "tick": tick,
            }
        ]
        self._collection.update(ids=[cid], **update_kwargs)

        return {"id": cid, "text": keep_text, "merged": True, "owners": owners}

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    async def remember(
        self, agent_id: str, text: str, tick: int = 0, source: str = "runtime"
    ) -> list[dict]:
        """Normalize text into atomic entries and consensus-insert each one."""
        entries = await self._normalize(text)
        results = []
        for entry in entries:
            results.append(
                await self._consensus_insert(agent_id, entry, tick, source)
            )
        return results

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

    def stats(self) -> dict:
        """Return {"total", "shared", "ratio"} where shared = entries with >=2 owners."""
        entries = self.all_entries()
        total = len(entries)
        shared = sum(1 for e in entries if len(e["owners"]) >= 2)
        ratio = (shared / total) if total else 0.0
        return {"total": total, "shared": shared, "ratio": ratio}
