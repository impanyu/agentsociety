import asyncio
import json
import math

_STRATEGIES = {"fifo", "relevance", "hybrid"}


def _pair_text(action: dict, result: dict) -> str:
    """Stable, deterministic text representation of an (action, result)
    pair, used as the embedding input for relevance/hybrid eviction."""
    return json.dumps(action, ensure_ascii=False) + " || " + json.dumps(result, ensure_ascii=False)


def _cosine(a, b) -> float:
    """Cosine similarity of two vectors, clamped to [0.0, 1.0].

    Embeddings from the same model are typically non-negative cosine, so
    clamping negative values to 0.0 (rather than remapping via (cos+1)/2)
    keeps "0" meaning "unrelated" instead of "opposite". Guards zero-norm
    vectors (returns 0.0) so an all-zero fallback embedding never blows up.
    """
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, dot / (norm_a * norm_b))


def _recency(pos: int, n: int) -> float:
    """Normalized recency for an item at index `pos` of `n` items,
    oldest=0.0 .. newest=1.0. With a single item, defined as 1.0."""
    if n <= 1:
        return 1.0
    return pos / (n - 1)


class FifoCache:
    """Cache storing (action, result) pairs, most-recent-last, with a
    pluggable eviction strategy for when appending would exceed `maxlen`:

      - "fifo" (default): drop the oldest pair. Unchanged legacy behavior;
        never needs an embed_fn.
      - "relevance": drop the existing pair least similar (embedding
        cosine) to the new pair being appended.
      - "hybrid": drop the pair with the lowest
        `alpha*recency + (1-alpha)*relevance` combined score.

    The new pair being appended is always kept -- the strategy only
    decides which *existing* pair (if any) gets evicted to make room.
    """

    def __init__(self, maxlen: int = 20, strategy: str = "fifo", embed_fn=None, alpha: float = 0.5):
        """
        Args:
            maxlen: Max number of (action, result) pairs kept.
            strategy: One of "fifo", "relevance", "hybrid".
            embed_fn: async callable(list[str]) -> list[vector]. Required
                (raises ValueError otherwise) when strategy is not "fifo".
            alpha: Recency weight for "hybrid" (relevance weight is
                1 - alpha). Unused by "fifo"/"relevance".
        """
        if strategy not in _STRATEGIES:
            raise ValueError(f"unknown cache strategy: {strategy!r}")
        if strategy != "fifo" and embed_fn is None:
            raise ValueError(f"cache strategy {strategy!r} requires an embed_fn")

        self._maxlen = maxlen
        self._strategy = strategy
        self._embed_fn = embed_fn
        self._alpha = alpha
        # Oldest-first list of {"action", "result", "embedding"} dicts.
        # "embedding" is None until lazily computed.
        self._items: list[dict] = []

    async def append(self, action: dict, result: dict) -> None:
        """Add an (action, result) pair, evicting per `strategy` if full.

        fifo needs no embedding at all. relevance/hybrid embed the new
        pair's text; if the cache is full, any existing pair missing an
        embedding is lazily (batch-)embedded first, then the victim is
        chosen by the strategy's score and removed. The new pair is
        always kept, so the window never exceeds `maxlen`.
        """
        if self._strategy == "fifo":
            if len(self._items) >= self._maxlen:
                self._items.pop(0)
            self._items.append({"action": action, "result": result, "embedding": None})
            return

        new_embedding = (await self._embed_fn([_pair_text(action, result)]))[0]

        if len(self._items) >= self._maxlen:
            await self._ensure_embeddings()
            victim = self._select_victim(new_embedding)
            self._items.pop(victim)

        self._items.append({"action": action, "result": result, "embedding": new_embedding})

    async def _ensure_embeddings(self) -> None:
        """Lazily batch-embed any existing items whose embedding is still
        None (e.g. items loaded via `restore_items`)."""
        missing = [i for i, it in enumerate(self._items) if it["embedding"] is None]
        if not missing:
            return
        texts = [_pair_text(self._items[i]["action"], self._items[i]["result"]) for i in missing]
        vectors = await self._embed_fn(texts)
        for i, vector in zip(missing, vectors):
            self._items[i]["embedding"] = vector

    def _select_victim(self, new_embedding) -> int:
        """Index of the existing item to evict for "relevance"/"hybrid".

        Scores are computed oldest-to-newest and the victim is the first
        (i.e. oldest) strict minimum encountered, so ties are broken by
        dropping the oldest tied pair -- a stable, FIFO-like tiebreak.
        """
        n = len(self._items)
        best_idx = 0
        best_score = None
        for pos, it in enumerate(self._items):
            relevance = _cosine(it["embedding"], new_embedding)
            if self._strategy == "relevance":
                score = relevance
            else:  # hybrid
                recency = _recency(pos, n)
                score = self._alpha * recency + (1 - self._alpha) * relevance
            if best_score is None or score < best_score:
                best_score = score
                best_idx = pos
        return best_idx

    def items(self) -> list[tuple[dict, dict]]:
        """Return all items as list of (action, result) tuples."""
        return [(it["action"], it["result"]) for it in self._items]

    def restore_items(self, items) -> None:
        """Sync, non-evicting: set the window contents directly from a
        list of (action, result) pairs (e.g. from a checkpoint). Skips
        eviction and embedding entirely -- embeddings are left None and
        computed lazily on the next relevance/hybrid append."""
        self._items = [{"action": a, "result": r, "embedding": None} for a, r in items]

    def __len__(self):
        """Return the number of items in cache."""
        return len(self._items)


class GoalStack:
    """Stack of goals where index 0 = bottom = most fundamental."""

    def __init__(self):
        self._stack = []

    def push(self, text: str) -> None:
        """Push a goal onto the top of the stack."""
        self._stack.append(text)

    def pop(self) -> str | None:
        """Pop and return the top goal, or None if empty."""
        if self._stack:
            return self._stack.pop()
        return None

    def replace(self, text: str) -> None:
        """Replace the top goal; if empty, same as push."""
        if self._stack:
            self._stack[-1] = text
        else:
            self._stack.append(text)

    def peek(self) -> str | None:
        """Return the top goal without removing it, or None if empty."""
        if self._stack:
            return self._stack[-1]
        return None

    def items(self) -> list[str]:
        """Return all goals as list [bottom, ..., top]."""
        return list(self._stack)

    def empty(self) -> bool:
        """Check if the stack is empty."""
        return len(self._stack) == 0


class StatusRegister:
    """Dict-like status register with public/private keys."""

    def __init__(self, initial: dict | None = None, private_keys: set | None = None):
        """Initialize status register.

        Args:
            initial: Initial dict of key-value pairs
            private_keys: Set of keys to treat as private.
                         Defaults to {"mood"} if None.
                         "location" is always public (never in private_keys).
        """
        self._data = dict(initial) if initial else {}

        # Set default private keys if not provided
        if private_keys is None:
            self._private_keys = {"mood"}
        else:
            self._private_keys = private_keys.copy()

        # "location" is always public (remove from private if present)
        self._private_keys.discard("location")

    def set(self, key: str, value) -> None:
        """Set a key-value pair."""
        self._data[key] = value

    def remove(self, key: str) -> None:
        """Remove a key from the register."""
        if key in self._data:
            del self._data[key]

    def get(self, key: str, default=None):
        """Get a value by key, with optional default."""
        return self._data.get(key, default)

    def public_view(self) -> dict:
        """Return dict of public keys only (always includes 'location' if set)."""
        public = {k: v for k, v in self._data.items() if k not in self._private_keys}
        return public

    def all(self) -> dict:
        """Return all keys as dict."""
        return dict(self._data)


class STM:
    """Short-term memory combining FIFO cache, goal stack, status, and inbox."""

    def __init__(
        self,
        fifo_size: int = 20,
        status: dict | None = None,
        private_keys: set | None = None,
        goals: list[str] | None = None,
        cache_strategy: str = "fifo",
        cache_alpha: float = 0.5,
        cache_embed_fn=None,
    ):
        """Initialize STM.

        Args:
            fifo_size: Max size of FIFO cache
            status: Initial status dict
            private_keys: Set of private status keys
            goals: List of goals [bottom, ..., top]
            cache_strategy: FIFO eviction strategy: "fifo", "relevance", or
                "hybrid" (see `FifoCache`).
            cache_alpha: Recency weight for the "hybrid" strategy.
            cache_embed_fn: async callable(list[str]) -> list[vector], used
                by the FIFO cache for "relevance"/"hybrid" scoring.
                Required (FifoCache raises ValueError otherwise) unless
                cache_strategy is "fifo".
        """
        self.fifo = FifoCache(
            maxlen=fifo_size,
            strategy=cache_strategy,
            embed_fn=cache_embed_fn,
            alpha=cache_alpha,
        )
        self.status = StatusRegister(initial=status, private_keys=private_keys)
        self.goals = GoalStack()
        self.inbox = asyncio.Queue()

        # Initialize goals from bottom to top
        if goals:
            for goal in goals:
                self.goals.push(goal)

    def inbox_items(self) -> list:
        """Return the current inbox contents (front-to-back) without consuming.

        Reads asyncio.Queue's internal `_queue` deque directly; this is a
        single-process, single-consumer read-only peek used by both
        Agent.build_view (inbox_head) and Kernel.execute's peek_inbox
        handler, so there's one place that knows about this internal.
        """
        return list(getattr(self.inbox, "_queue", []))
