import pytest

from society.ltm import SharedMemory
from society.baselines import (
    GenerativeAgentsMemory,
    GMemory,
    CollaborativeMemory,
    make_memory,
)
from tests.helpers import FakeLLM, afake_embed


# ----------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------

TEXT_A = "关羽千里走单骑"
TEXT_B = "刘备三顾茅庐"


def _stats_shape_ok(stats):
    assert set(stats.keys()) == {"total", "shared", "ratio"}
    assert isinstance(stats["total"], int)
    assert isinstance(stats["shared"], int)
    assert isinstance(stats["ratio"], float)


# ========================================================================
# GenerativeAgentsMemory
# ========================================================================


async def test_ga_remember_then_recall_returns_text():
    m = GenerativeAgentsMemory(afake_embed)
    await m.remember("guanyu", TEXT_A)
    results = await m.recall("guanyu", TEXT_A, top_k=5)
    assert any(r["text"] == TEXT_A for r in results)
    assert set(results[0].keys()) == {"id", "text"}


async def test_ga_recall_ranks_by_relevance():
    m = GenerativeAgentsMemory(afake_embed)
    await m.remember("guanyu", TEXT_A)
    await m.remember("guanyu", TEXT_B)
    results = await m.recall("guanyu", TEXT_A, top_k=2)
    assert results[0]["text"] == TEXT_A


async def test_ga_export_restore_roundtrip():
    m = GenerativeAgentsMemory(afake_embed)
    await m.remember("guanyu", TEXT_A)
    exported = m.export()
    assert len(exported) == 1
    assert exported[0]["embedding"] is not None
    assert exported[0]["text"] == TEXT_A
    assert exported[0]["owners"] == ["guanyu"]

    m2 = GenerativeAgentsMemory(afake_embed)
    await m2.restore(exported)
    entries = m2.all_entries()
    assert len(entries) == 1
    assert entries[0]["id"] == exported[0]["id"]
    assert entries[0]["text"] == TEXT_A
    assert entries[0]["owners"] == ["guanyu"]


async def test_ga_stats_shape():
    m = GenerativeAgentsMemory(afake_embed)
    await m.remember("guanyu", TEXT_A)
    _stats_shape_ok(m.stats())


async def test_ga_forget_removes_row():
    m = GenerativeAgentsMemory(afake_embed)
    results = await m.remember("guanyu", TEXT_A)
    memory_id = results[0]["id"]
    assert m.forget("guanyu", memory_id) is True
    assert m.all_entries() == []
    # forgetting again (already gone) returns False
    assert m.forget("guanyu", memory_id) is False


async def test_ga_duplication_across_owners_and_private_recall():
    m = GenerativeAgentsMemory(afake_embed)
    await m.remember("guanyu", TEXT_A)
    await m.remember("liubei", TEXT_A)

    entries = m.all_entries()
    assert len(entries) == 2  # duplicated, one row per owner -- no dedup

    guanyu_results = await m.recall("guanyu", TEXT_A, top_k=5)
    assert len(guanyu_results) == 1
    assert all(r["text"] == TEXT_A for r in guanyu_results)

    liubei_results = await m.recall("liubei", TEXT_A, top_k=5)
    assert len(liubei_results) == 1


async def test_ga_importance_uses_llm_when_available():
    llm = FakeLLM(responses=["8"])
    m = GenerativeAgentsMemory(afake_embed, llm=llm)
    await m.remember("guanyu", TEXT_A)
    entries = m.all_entries()
    assert entries[0]["meta"]["importance"] == 8


async def test_ga_importance_defaults_without_llm():
    m = GenerativeAgentsMemory(afake_embed, llm=None)
    await m.remember("guanyu", TEXT_A)
    entries = m.all_entries()
    assert entries[0]["meta"]["importance"] == 5


# ========================================================================
# GMemory
# ========================================================================


async def test_gmemory_remember_then_recall_returns_text():
    m = GMemory(afake_embed)
    await m.remember("guanyu", TEXT_A)
    results = await m.recall("guanyu", TEXT_A, top_k=5)
    assert any(r["text"] == TEXT_A for r in results)
    assert set(results[0].keys()) == {"id", "text"}


async def test_gmemory_recall_ranks_by_relevance():
    m = GMemory(afake_embed)
    await m.remember("guanyu", TEXT_A)
    await m.remember("guanyu", TEXT_B)
    results = await m.recall("guanyu", TEXT_A, top_k=2)
    assert results[0]["text"] == TEXT_A


async def test_gmemory_export_restore_roundtrip():
    m = GMemory(afake_embed)
    await m.remember("guanyu", TEXT_A)
    exported = m.export()
    assert len(exported) == 1
    assert exported[0]["embedding"] is not None

    m2 = GMemory(afake_embed)
    await m2.restore(exported)
    entries = m2.all_entries()
    assert len(entries) == 1
    assert entries[0]["text"] == TEXT_A
    assert entries[0]["owners"] == ["guanyu"]


async def test_gmemory_stats_shape():
    m = GMemory(afake_embed)
    await m.remember("guanyu", TEXT_A)
    _stats_shape_ok(m.stats())


async def test_gmemory_forget_removes_ownership():
    m = GMemory(afake_embed)
    results = await m.remember("guanyu", TEXT_A)
    memory_id = results[0]["id"]
    assert m.forget("guanyu", memory_id) is True
    assert m.all_entries() == []


async def test_gmemory_recall_is_shared_across_agents():
    m = GMemory(afake_embed)
    await m.remember("guanyu", TEXT_A)
    # agent B never wrote anything but can retrieve A's shared entry
    results = await m.recall("liubei", TEXT_A, top_k=5)
    assert any(r["text"] == TEXT_A for r in results)


async def test_gmemory_no_dedup_on_duplicate_text():
    m = GMemory(afake_embed)
    await m.remember("guanyu", TEXT_A)
    await m.remember("liubei", TEXT_A)
    entries = m.all_entries()
    assert len(entries) == 2  # appended, no merge


# ========================================================================
# CollaborativeMemory
# ========================================================================


async def test_collab_remember_then_recall_returns_text():
    m = CollaborativeMemory(afake_embed)
    await m.remember("guanyu", TEXT_A)
    results = await m.recall("guanyu", TEXT_A, top_k=5)
    assert any(r["text"] == TEXT_A for r in results)
    assert set(results[0].keys()) == {"id", "text"}


async def test_collab_recall_ranks_by_relevance():
    m = CollaborativeMemory(afake_embed)
    await m.remember("guanyu", TEXT_A)
    await m.remember("guanyu", TEXT_B)
    results = await m.recall("guanyu", TEXT_A, top_k=2)
    assert results[0]["text"] == TEXT_A


async def test_collab_export_restore_roundtrip():
    m = CollaborativeMemory(afake_embed)
    await m.remember("guanyu", TEXT_A)
    exported = m.export()
    assert len(exported) == 1
    assert exported[0]["embedding"] is not None
    assert exported[0]["owners"] == ["guanyu"]

    m2 = CollaborativeMemory(afake_embed)
    await m2.restore(exported)
    entries = m2.all_entries()
    assert len(entries) == 1
    assert entries[0]["text"] == TEXT_A
    assert entries[0]["owners"] == ["guanyu"]


async def test_collab_stats_shape():
    m = CollaborativeMemory(afake_embed)
    await m.remember("guanyu", TEXT_A)
    _stats_shape_ok(m.stats())


async def test_collab_acl_gates_recall_then_grant_allows():
    m = CollaborativeMemory(afake_embed)
    results = await m.remember("guanyu", TEXT_A)
    memory_id = results[0]["id"]

    # B has no read access yet
    b_results = await m.recall("liubei", TEXT_A, top_k=5)
    assert b_results == []

    # grant read access to B
    assert m.grant(memory_id, "liubei") is True
    b_results = await m.recall("liubei", TEXT_A, top_k=5)
    assert any(r["text"] == TEXT_A for r in b_results)

    stats = m.stats()
    assert stats["shared"] == 1  # ACL now has 2 members


async def test_collab_forget_removes_acl_membership():
    m = CollaborativeMemory(afake_embed)
    results = await m.remember("guanyu", TEXT_A)
    memory_id = results[0]["id"]
    m.grant(memory_id, "liubei")

    assert m.forget("guanyu", memory_id) is True
    entries = m.all_entries()
    assert len(entries) == 1
    assert entries[0]["owners"] == ["liubei"]

    # removing the last reader deletes the fragment entirely
    assert m.forget("liubei", memory_id) is True
    assert m.all_entries() == []


async def test_collab_forget_unknown_agent_returns_false():
    m = CollaborativeMemory(afake_embed)
    results = await m.remember("guanyu", TEXT_A)
    memory_id = results[0]["id"]
    assert m.forget("liubei", memory_id) is False


# ========================================================================
# factory
# ========================================================================


def test_make_memory_returns_correct_classes():
    assert isinstance(make_memory("consensus", afake_embed), SharedMemory)
    assert isinstance(
        make_memory("generative_agents", afake_embed), GenerativeAgentsMemory
    )
    assert isinstance(make_memory("g_memory", afake_embed), GMemory)
    assert isinstance(make_memory("collaborative", afake_embed), CollaborativeMemory)


def test_make_memory_unknown_kind_raises():
    with pytest.raises(ValueError):
        make_memory("nonexistent", afake_embed)


def test_make_memory_ignores_extra_kwargs():
    # existing call sites do SharedMemory(embed_fn, llm, max_chars=...) -- adapters
    # must accept and ignore kwargs they don't use.
    m = make_memory(
        "generative_agents", afake_embed, llm=None, max_chars=80, sim_threshold=0.86
    )
    assert isinstance(m, GenerativeAgentsMemory)
