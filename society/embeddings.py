import asyncio


class EmbeddingClient:
    """Async client for computing text embeddings."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        embed_model: str,
        transport=None,
        retries: int = 5,
        backoff_base: float = 1.0,
    ):
        """
        Initialize EmbeddingClient.

        Args:
            api_key: API key for the backend.
            base_url: Base URL of the embeddings API.
            embed_model: Model name to send in requests.
            transport: Optional async callable(payload_dict) -> response_dict,
                used in place of the default httpx-based transport (for tests).
            retries: Number of attempts on transport exceptions before raising.
                Long sedimentation runs make thousands of embed calls; a single
                transient 5xx/timeout must not abort the whole run.
            backoff_base: Base seconds for exponential backoff (backoff_base * 2**n).
        """
        self.api_key = api_key
        self.base_url = base_url
        self.embed_model = embed_model
        self._transport = transport or self._default_transport
        self.retries = retries
        self.backoff_base = backoff_base

    async def _default_transport(self, payload: dict) -> dict:
        import httpx

        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            return response.json()

    async def _call_with_retry(self, payload: dict) -> dict:
        attempt = 0
        while True:
            try:
                return await self._transport(payload)
            except asyncio.CancelledError:
                raise
            except Exception:
                attempt += 1
                if attempt >= self.retries:
                    raise
                await asyncio.sleep(self.backoff_base * (2 ** (attempt - 1)))

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Compute embeddings for a list of texts.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors, one per input text, in order.
        """
        payload = {"model": self.embed_model, "input": texts}
        response = await self._call_with_retry(payload)
        return [item["embedding"] for item in response["data"]]
