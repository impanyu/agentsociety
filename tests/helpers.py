import hashlib
import random
from collections import defaultdict

from society.llm import BudgetExceeded


class FakeLLM:
    """In-memory stand-in for LLMClient; duck-types chat/usage."""

    def __init__(self, responses=None, fn=None, raise_after=None):
        """
        Initialize FakeLLM.

        Args:
            responses: Optional list of scripted responses, popped in order
                (front of the list first) as chat() is called.
            fn: Optional callable(prompt, system) -> str used to compute a
                response when responses is exhausted or not provided.
            raise_after: Optional int. Once more than this many chat() calls
                have been attempted (across all agents/buckets sharing this
                client), every subsequent call raises
                `society.llm.BudgetExceeded`, simulating a real LLMClient
                whose max_calls/max_tokens budget has been exhausted.
                None (default) disables this -- chat() never raises.
        """
        self._responses = list(responses) if responses else []
        self._fn = fn
        self.raise_after = raise_after
        self._call_count = 0
        self.calls = []
        self._usage = defaultdict(lambda: {"calls": 0, "tokens": 0})

    async def chat(self, prompt, system=None, bucket="decide") -> str:
        """
        Record the call and return the next scripted or computed response.

        Args:
            prompt: User prompt content.
            system: Optional system message content.
            bucket: Usage bucket to tally this call under.

        Returns:
            Response string.

        Raises:
            BudgetExceeded: once more than `raise_after` calls have been
                attempted, if `raise_after` was set.
        """
        self._call_count += 1
        if self.raise_after is not None and self._call_count > self.raise_after:
            raise BudgetExceeded(f"fake budget exceeded after {self.raise_after} calls")

        self.calls.append((bucket, prompt, system))

        if self._responses:
            reply = self._responses.pop(0)
        elif self._fn is not None:
            reply = self._fn(prompt, system)
        else:
            reply = ""

        self._usage[bucket]["calls"] += 1
        self._usage["_total"]["calls"] += 1

        return reply

    def usage(self) -> dict:
        """
        Get accumulated usage.

        Returns:
            Dict mapping bucket name -> {"calls": int, "tokens": int}, plus
            a "_total" key with the sum across all buckets.
        """
        return {bucket: dict(counts) for bucket, counts in self._usage.items()}


def fake_embed(texts: list[str]) -> list[list[float]]:
    """
    Deterministic fake embedding: seeds a random.Random from the md5 of each
    text so identical text always yields an identical 8-dim vector.

    Args:
        texts: List of input strings.

    Returns:
        List of 8-dim float vectors, one per input text.
    """
    vectors = []
    for text in texts:
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        rng = random.Random(digest)
        vectors.append([rng.uniform(-1.0, 1.0) for _ in range(8)])
    return vectors


async def afake_embed(texts: list[str]) -> list[list[float]]:
    """Async wrapper around fake_embed."""
    return fake_embed(texts)
