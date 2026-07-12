import asyncio
from collections import defaultdict


class BudgetExceeded(Exception):
    """Raised when a configured call/token budget would be exceeded."""


class LLMClient:
    """Async chat-completion client with concurrency limiting, budget
    enforcement, retry-with-backoff, and per-bucket usage accounting."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        chat_model: str,
        *,
        max_concurrency: int = 16,
        max_calls: int | None = None,
        max_tokens: int | None = None,
        transport=None,
        retries: int = 3,
        backoff_base: float = 0.5,
    ):
        """
        Initialize LLMClient.

        Args:
            api_key: API key for the backend.
            base_url: Base URL of the chat-completions API.
            chat_model: Model name to send in requests.
            max_concurrency: Max in-flight requests (semaphore size).
            max_calls: Optional cap on total number of calls across all buckets.
            max_tokens: Optional cap on total tokens across all buckets.
            transport: Optional async callable(payload_dict) -> response_dict,
                used in place of the default httpx-based transport (for tests).
            retries: Number of attempts on transport exceptions before raising.
            backoff_base: Base seconds for exponential backoff (backoff_base * 2**n).
        """
        self.api_key = api_key
        self.base_url = base_url
        self.chat_model = chat_model
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self.max_concurrency = max_concurrency
        self.retries = retries
        self.backoff_base = backoff_base
        self._transport = transport or self._default_transport
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._usage = defaultdict(lambda: {"calls": 0, "tokens": 0})

    async def _default_transport(self, payload: dict) -> dict:
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            return response.json()

    def _check_budget(self) -> None:
        total = self._usage["_total"]
        if self.max_calls is not None and total["calls"] >= self.max_calls:
            raise BudgetExceeded(f"max_calls={self.max_calls} exceeded")
        if self.max_tokens is not None and total["tokens"] >= self.max_tokens:
            raise BudgetExceeded(f"max_tokens={self.max_tokens} exceeded")

    async def chat(self, prompt: str, system: str | None = None, bucket: str = "decide") -> str:
        """
        Send a chat completion request.

        Args:
            prompt: User prompt content.
            system: Optional system message content.
            bucket: Usage bucket to tally this call under.

        Returns:
            Assistant reply content string.

        Raises:
            BudgetExceeded: if max_calls/max_tokens would be exceeded before the call.
        """
        self._check_budget()

        messages = []
        if system is not None:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.chat_model, "messages": messages}

        async with self._semaphore:
            response = await self._call_with_retry(payload)

        content = response["choices"][0]["message"]["content"]
        tokens = response.get("usage", {}).get("total_tokens", 0)

        self._usage[bucket]["calls"] += 1
        self._usage[bucket]["tokens"] += tokens
        self._usage["_total"]["calls"] += 1
        self._usage["_total"]["tokens"] += tokens

        return content

    async def _call_with_retry(self, payload: dict) -> dict:
        attempt = 0
        while True:
            try:
                return await self._transport(payload)
            except Exception:
                attempt += 1
                if attempt >= self.retries:
                    raise
                await asyncio.sleep(self.backoff_base * (2 ** (attempt - 1)))

    def usage(self) -> dict:
        """
        Get accumulated usage.

        Returns:
            Dict mapping bucket name -> {"calls": int, "tokens": int}, plus
            a "_total" key with the sum across all buckets.
        """
        return {bucket: dict(counts) for bucket, counts in self._usage.items()}
