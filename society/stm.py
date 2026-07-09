import asyncio
from collections import deque


class FifoCache:
    """FIFO cache storing (action, result) pairs, most-recent-last."""

    def __init__(self, maxlen: int = 20):
        self._cache = deque(maxlen=maxlen)

    def append(self, action: dict, result: dict) -> None:
        """Add an (action, result) pair."""
        self._cache.append((action, result))

    def items(self) -> list[tuple[dict, dict]]:
        """Return all items as list of (action, result) tuples."""
        return list(self._cache)

    def __len__(self):
        """Return the number of items in cache."""
        return len(self._cache)


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
        goals: list[str] | None = None
    ):
        """Initialize STM.

        Args:
            fifo_size: Max size of FIFO cache
            status: Initial status dict
            private_keys: Set of private status keys
            goals: List of goals [bottom, ..., top]
        """
        self.fifo = FifoCache(maxlen=fifo_size)
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
