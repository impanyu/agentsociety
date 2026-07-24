import asyncio

import pytest

from society.embeddings import EmbeddingClient


def _ok_response(texts):
    return {"data": [{"embedding": [float(len(t))]} for t in texts]}


async def test_embed_returns_vectors_in_order():
    async def transport(payload):
        return _ok_response(payload["input"])

    c = EmbeddingClient("k", "u", "m", transport=transport)
    out = await c.embed(["a", "bb", "ccc"])
    assert out == [[1.0], [2.0], [3.0]]


async def test_embed_retries_transient_failure_then_succeeds():
    calls = {"n": 0}

    async def flaky(payload):
        calls["n"] += 1
        if calls["n"] < 3:                      # fail twice, succeed on 3rd
            raise RuntimeError("500 Internal Server Error")
        return _ok_response(payload["input"])

    # tiny backoff so the test is fast
    c = EmbeddingClient("k", "u", "m", transport=flaky, retries=5, backoff_base=0.0)
    out = await c.embed(["x"])
    assert out == [[1.0]]
    assert calls["n"] == 3                       # retried, did not abort


async def test_embed_raises_after_exhausting_retries():
    async def always_fail(payload):
        raise RuntimeError("persistent 500")

    c = EmbeddingClient("k", "u", "m", transport=always_fail, retries=3, backoff_base=0.0)
    with pytest.raises(RuntimeError):
        await c.embed(["x"])


async def test_embed_does_not_retry_cancellation():
    calls = {"n": 0}

    async def cancel(payload):
        calls["n"] += 1
        raise asyncio.CancelledError()

    c = EmbeddingClient("k", "u", "m", transport=cancel, retries=5, backoff_base=0.0)
    with pytest.raises(asyncio.CancelledError):
        await c.embed(["x"])
    assert calls["n"] == 1                       # cancellation propagates immediately
