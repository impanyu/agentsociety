class EmbeddingClient:
    """Async client for computing text embeddings."""

    def __init__(self, api_key: str, base_url: str, embed_model: str, transport=None):
        """
        Initialize EmbeddingClient.

        Args:
            api_key: API key for the backend.
            base_url: Base URL of the embeddings API.
            embed_model: Model name to send in requests.
            transport: Optional async callable(payload_dict) -> response_dict,
                used in place of the default httpx-based transport (for tests).
        """
        self.api_key = api_key
        self.base_url = base_url
        self.embed_model = embed_model
        self._transport = transport or self._default_transport

    async def _default_transport(self, payload: dict) -> dict:
        import httpx

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            response.raise_for_status()
            return response.json()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Compute embeddings for a list of texts.

        Args:
            texts: List of input strings.

        Returns:
            List of embedding vectors, one per input text, in order.
        """
        payload = {"model": self.embed_model, "input": texts}
        response = await self._transport(payload)
        return [item["embedding"] for item in response["data"]]
