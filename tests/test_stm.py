import asyncio
import math

import pytest

from society.stm import FifoCache, GoalStack, StatusRegister, STM


def make_fake_embed(vectors: dict):
    """Deterministic fake async embed_fn: looks up each text by substring
    match against `vectors` keys (action names embedded via json.dumps end
    up containing the key), falling back to the zero vector."""

    async def _embed(texts):
        out = []
        for t in texts:
            for key, vec in vectors.items():
                if key in t:
                    out.append(vec)
                    break
            else:
                out.append((0.0, 0.0))
        return out

    return _embed


async def test_fifo_evicts_oldest_pair():
    f = FifoCache(maxlen=2)
    await f.append({"name": "a1"}, {"ok": True})
    await f.append({"name": "a2"}, {"ok": True})
    await f.append({"name": "a3"}, {"ok": True})
    assert len(f) == 2
    assert [a["name"] for a, _ in f.items()] == ["a2", "a3"]


def test_fifo_default_strategy_needs_no_embed_fn():
    # Must not raise -- fifo is the default and never requires embed_fn.
    f = FifoCache(maxlen=2)
    assert len(f) == 0


def test_relevance_strategy_requires_embed_fn():
    with pytest.raises(ValueError):
        FifoCache(maxlen=2, strategy="relevance")


def test_hybrid_strategy_requires_embed_fn():
    with pytest.raises(ValueError):
        FifoCache(maxlen=2, strategy="hybrid")


async def test_relevance_evicts_least_similar_pair():
    # new = (1.0, 0.0); p1 cos=0.9 (very relevant), p2 cos=0.0 (orthogonal,
    # least relevant), p3 cos=0.1 (mildly relevant).
    vectors = {
        "p1": (0.9, math.sqrt(0.19)),
        "p2": (0.0, 1.0),
        "p3": (0.1, math.sqrt(0.99)),
        "new": (1.0, 0.0),
    }
    f = FifoCache(maxlen=3, strategy="relevance", embed_fn=make_fake_embed(vectors))
    await f.append({"name": "p1"}, {})
    await f.append({"name": "p2"}, {})
    await f.append({"name": "p3"}, {})
    await f.append({"name": "new"}, {})

    names = [a["name"] for a, _ in f.items()]
    assert len(f) == 3
    assert "p2" not in names  # least relevant -> evicted
    assert "p1" in names      # most relevant -> survives
    assert "new" in names     # new pair always kept


async def test_hybrid_disagrees_with_fifo_and_pure_relevance():
    # p1 oldest + very relevant (rec=0.0, rel=0.9) -> fifo would evict this
    # p2 middle + mildly relevant (rec=0.5, rel=0.1) -> hybrid(alpha=0.5) evicts this
    # p3 newest + irrelevant (rec=1.0, rel=0.0) -> pure relevance would evict this
    vectors = {
        "p1": (0.9, math.sqrt(0.19)),
        "p2": (0.1, math.sqrt(0.99)),
        "p3": (0.0, 1.0),
        "new": (1.0, 0.0),
    }
    f = FifoCache(maxlen=3, strategy="hybrid", embed_fn=make_fake_embed(vectors), alpha=0.5)
    await f.append({"name": "p1"}, {})
    await f.append({"name": "p2"}, {})
    await f.append({"name": "p3"}, {})
    await f.append({"name": "new"}, {})

    names = [a["name"] for a, _ in f.items()]
    assert "p2" not in names            # hybrid's argmin(0.5*rec+0.5*rel)
    assert "p1" in names and "p3" in names and "new" in names


async def test_hybrid_alpha_one_behaves_like_recency_fifo_victim():
    vectors = {
        "p1": (0.9, math.sqrt(0.19)),
        "p2": (0.1, math.sqrt(0.99)),
        "p3": (0.0, 1.0),
        "new": (1.0, 0.0),
    }
    f = FifoCache(maxlen=3, strategy="hybrid", embed_fn=make_fake_embed(vectors), alpha=1.0)
    await f.append({"name": "p1"}, {})
    await f.append({"name": "p2"}, {})
    await f.append({"name": "p3"}, {})
    await f.append({"name": "new"}, {})

    names = [a["name"] for a, _ in f.items()]
    assert "p1" not in names  # oldest evicted, same victim as fifo


async def test_hybrid_alpha_zero_behaves_like_pure_relevance_victim():
    vectors = {
        "p1": (0.9, math.sqrt(0.19)),
        "p2": (0.1, math.sqrt(0.99)),
        "p3": (0.0, 1.0),
        "new": (1.0, 0.0),
    }
    f = FifoCache(maxlen=3, strategy="hybrid", embed_fn=make_fake_embed(vectors), alpha=0.0)
    await f.append({"name": "p1"}, {})
    await f.append({"name": "p2"}, {})
    await f.append({"name": "p3"}, {})
    await f.append({"name": "new"}, {})

    names = [a["name"] for a, _ in f.items()]
    assert "p3" not in names  # least relevant evicted, same victim as pure relevance


async def test_restore_items_sets_contents_without_eviction_then_lazy_embeds():
    vectors = {"p1": (1.0, 0.0), "p2": (0.0, 1.0), "new": (1.0, 0.0)}
    f = FifoCache(maxlen=2, strategy="relevance", embed_fn=make_fake_embed(vectors))

    f.restore_items([({"name": "p1"}, {}), ({"name": "p2"}, {})])
    assert len(f) == 2
    assert [a["name"] for a, _ in f.items()] == ["p1", "p2"]

    # A subsequent relevance append must not crash -- it lazily embeds the
    # restored (embedding=None) items before choosing a victim.
    await f.append({"name": "new"}, {})
    assert len(f) == 2
    names = [a["name"] for a, _ in f.items()]
    assert "new" in names


def test_goal_stack_bottom_is_fundamental():
    g = GoalStack()
    g.push("fundamental"); g.push("immediate")
    assert g.items() == ["fundamental", "immediate"]
    assert g.peek() == "immediate"
    assert g.pop() == "immediate"
    assert g.pop() == "fundamental"
    assert g.pop() is None and g.empty()

def test_goal_replace_top():
    g = GoalStack(); g.push("a"); g.replace("b")
    assert g.items() == ["b"]
    g2 = GoalStack(); g2.replace("x")           # empty → push
    assert g2.items() == ["x"]

def test_status_public_private():
    s = StatusRegister({"mood": "sad", "appearance": "tall", "location": "hall"})
    assert "mood" not in s.public_view()
    assert s.public_view() == {"appearance": "tall", "location": "hall"}
    assert s.get("mood") == "sad"
    s2 = StatusRegister({"mood": "happy"}, private_keys=set())  # override: mood public
    assert s2.public_view() == {"mood": "happy"}

def test_stm_wiring_and_initial_goals():
    stm = STM(fifo_size=3, status={"location": "hall"}, goals=["deep", "top"])
    assert stm.goals.items() == ["deep", "top"]
    assert isinstance(stm.inbox, asyncio.Queue)
    assert stm.status.get("location") == "hall"


async def test_stm_threads_cache_strategy_and_embed_fn_to_fifo():
    vectors = {"a": (1.0, 0.0), "b": (0.0, 1.0)}
    stm = STM(fifo_size=2, cache_strategy="relevance", cache_embed_fn=make_fake_embed(vectors))
    await stm.fifo.append({"name": "a"}, {})
    await stm.fifo.append({"name": "b"}, {})
    assert len(stm.fifo) == 2


def test_stm_default_cache_strategy_is_fifo_and_needs_no_embed_fn():
    # Must not raise even without cache_embed_fn -- default strategy is fifo.
    stm = STM(fifo_size=2)
    assert len(stm.fifo) == 0
