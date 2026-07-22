from society.ltm import SharedMemory
from tests.helpers import FakeLLM, afake_embed


def mem(llm=None, **kw):
    import uuid
    return SharedMemory(afake_embed, llm=llm,
                        collection_name=f"t_{uuid.uuid4().hex[:8]}", **kw)


async def test_short_atomic_text_skips_llm_and_stores():
    llm = FakeLLM()
    m = mem(llm)
    out = await m.remember("alice", "黛玉葬花")
    assert len(out) == 1 and out[0]["merged"] is False
    assert llm.calls == []                      # gate not triggered
    assert m.stats() == {"total": 1, "shared": 0, "ratio": 0.0}

async def test_normalize_splits_long_or_compound(monkeypatch):
    llm = FakeLLM(responses=['["宝玉挨打", "贾政动怒"]'])
    m = mem(llm)
    out = await m.remember("alice", "宝玉挨打了，然后贾政大发雷霆并且惊动了贾母。")
    texts = {e["text"] for e in out}
    assert texts == {"宝玉挨打", "贾政动怒"}
    assert any(c[0] == "normalize" for c in llm.calls) or llm.calls  # one normalize call

async def test_normalize_fallback_without_llm():
    m = mem(None)
    entries = await m._normalize("句子一。句子二。")
    assert entries == ["句子一", "句子二"]

async def test_consensus_merges_equivalent_keeps_shorter_and_unions_owners():
    # identical text → identical fake embedding → sim 1.0 → candidate; llm says index 0
    llm = FakeLLM(responses=["0"])
    m = mem(llm)
    await m.remember("alice", "国王死于春天")
    out = await m.remember("bob", "国王死于春天")     # same text: equal length → keep existing
    assert out[0]["merged"] is True
    entries = m.all_entries()
    assert len(entries) == 1 and set(entries[0]["owners"]) == {"alice", "bob"}
    assert m.stats()["shared"] == 1 and m.stats()["ratio"] == 1.0

async def test_consensus_not_equivalent_adds_new():
    llm = FakeLLM(responses=["-1"])
    m = mem(llm)
    await m.remember("alice", "国王死于春天")
    await m.remember("bob", "国王死于春天")           # candidates found but llm says -1
    assert m.stats()["total"] == 2 and m.stats()["shared"] == 0

async def test_recall_owner_filtered():
    m = mem(None)
    await m.remember("alice", "花园着火")
    await m.remember("bob", "厨房进贼")
    got = await m.recall("alice", "花园着火", top_k=5)
    assert [e["text"] for e in got] == ["花园着火"]

async def test_forget_and_shared_survival():
    llm = FakeLLM(responses=["0"])
    m = mem(llm)
    await m.remember("alice", "国王死于春天")
    await m.remember("bob", "国王死于春天")
    (entry,) = m.all_entries()
    assert m.forget("alice", entry["id"]) is True
    (e2,) = m.all_entries()                       # bob still owns → survives
    assert e2["owners"] == ["bob"]
    assert m.forget("bob", e2["id"]) is True
    assert m.all_entries() == []                  # empty owners → deleted

async def test_revise_is_forget_plus_consensus_insert():
    m = mem(None)
    stored = await m.remember("alice", "旧的记忆内容")
    out = await m.revise("alice", stored[0]["id"], "新记忆")
    entries = m.all_entries()
    assert len(entries) == 1 and entries[0]["text"] == "新记忆"


# ----------------------------------------------------------------------
# token-based length cap (max_tokens replaces max_chars for length logic)
# ----------------------------------------------------------------------

from society.textlen import count_tokens


async def test_default_max_tokens_is_50():
    m = mem(None)
    assert m.max_tokens == 50


async def test_short_single_clause_input_not_normalized_even_if_long_in_chars():
    # A single short clause under the token cap, with no terminators/connectives,
    # must be stored as-is even though max_chars (deprecated) would have split it.
    m = mem(None, max_tokens=50)
    text = "黛玉葬花"
    assert m._needs_normalize(text) is False
    out = await m.remember("alice", text)
    assert len(out) == 1
    assert out[0]["text"] == text


async def test_over_50_token_input_triggers_normalize():
    m = mem(None, max_tokens=50)
    long_en = "the quick brown fox jumps over the lazy dog and keeps running " * 5
    assert count_tokens(long_en) > 50
    assert m._needs_normalize(long_en) is True


async def test_deposited_memory_capped_at_max_tokens_english():
    m = mem(None, max_tokens=50)
    long_en = "the quick brown fox jumps over the lazy dog and keeps running " * 5
    out = await m.remember("alice", long_en)
    for entry in out:
        assert count_tokens(entry["text"]) <= 50


async def test_deposited_memory_capped_at_max_tokens_chinese():
    m = mem(None, max_tokens=50)
    long_zh = "宝玉挨打了" * 30
    out = await m.remember("alice", long_zh)
    for entry in out:
        assert count_tokens(entry["text"]) <= 50


async def test_max_chars_kwarg_accepted_but_ignored():
    # Deprecated kwarg must not crash and must not affect the token-based gate.
    m = mem(None, max_chars=5, max_tokens=50)
    text = "黛玉葬花"  # far more than 5 chars, well under 50 tokens
    assert m._needs_normalize(text) is False


# ----------------------------------------------------------------------
# remember_atomic: multi-owner atomic deposit
# ----------------------------------------------------------------------


async def test_remember_atomic_stores_multiple_owners():
    m = mem(None)
    out = await m.remember_atomic(["a", "b", "c"], "刘备与张飞结拜")
    assert out is not None and out["merged"] is False
    (entry,) = m.all_entries()
    assert set(entry["owners"]) == {"a", "b", "c"}
    # each owner_* flag independently supports recall
    got_a = await m.recall("a", "刘备与张飞结拜")
    got_b = await m.recall("b", "刘备与张飞结拜")
    assert [e["text"] for e in got_a] == ["刘备与张飞结拜"]
    assert [e["text"] for e in got_b] == ["刘备与张飞结拜"]


async def test_remember_atomic_merge_unions_owners():
    # LLM says the new fragment is equivalent to candidate index 0 -> merge path.
    llm = FakeLLM(responses=["0"])
    m = mem(llm)
    await m.remember_atomic(["a", "b", "c"], "国王死于春天")
    out = await m.remember_atomic(["d"], "国王死于春天")
    assert out["merged"] is True
    (entry,) = m.all_entries()
    assert set(entry["owners"]) == {"a", "b", "c", "d"}
    # The merge must PRESERVE the pre-existing owners' recall flags (not just the
    # "owners" list): every owner, old and new, must still recall the merged entry.
    for owner in ("a", "b", "c", "d"):
        hits = await m.recall(owner, "国王死于春天", top_k=5)
        assert any(h["text"] == entry["text"] for h in hits), f"owner {owner} lost recall after merge"


async def test_remember_atomic_applies_token_cap():
    m = mem(None, max_tokens=50)
    long_en = "the quick brown fox jumps over the lazy dog and keeps running " * 5
    assert count_tokens(long_en) > 50
    out = await m.remember_atomic(["a"], long_en)
    assert count_tokens(out["text"]) <= 50


async def test_remember_atomic_empty_text_returns_none():
    m = mem(None)
    out = await m.remember_atomic(["a"], "   ")
    assert out is None
    assert m.all_entries() == []


async def test_remember_atomic_empty_owners_raises():
    m = mem(None)
    try:
        await m.remember_atomic([], "some fragment")
    except ValueError:
        pass
    else:
        assert False, "expected ValueError"
    assert m.all_entries() == []


async def test_forget_survives_on_multi_owner_atomic_entry():
    m = mem(None)
    out = await m.remember_atomic(["a", "b", "c"], "刘备与张飞结拜")
    memory_id = out["id"]
    assert m.forget("a", memory_id) is True
    (entry,) = m.all_entries()
    assert set(entry["owners"]) == {"b", "c"}
    assert m.forget("b", memory_id) is True
    assert m.forget("c", memory_id) is True
    assert m.all_entries() == []


# ----------------------------------------------------------------------
# affiliated_memories: related-memory graph
# ----------------------------------------------------------------------


async def test_remember_atomic_stores_affiliated_and_excludes_self():
    m = mem(None)
    out = await m.remember_atomic(["a"], "刘备与张飞结拜")
    fake_related_id = "some-other-id"
    out2 = await m.remember_atomic(
        ["b"], "关羽温酒斩华雄", affiliated=[out["id"], fake_related_id]
    )
    assert out2["id"] not in out2["affiliated"]  # excludes self
    (entry,) = [e for e in m.all_entries() if e["id"] == out2["id"]]
    assert set(entry["affiliated"]) == {out["id"], fake_related_id}

    exported = m.export()
    (exp_entry,) = [e for e in exported if e["id"] == out2["id"]]
    assert set(exp_entry["affiliated"]) == {out["id"], fake_related_id}


async def test_remember_atomic_affiliated_defaults_to_empty():
    m = mem(None)
    out = await m.remember_atomic(["a"], "刘备与张飞结拜")
    assert out["affiliated"] == []
    (entry,) = m.all_entries()
    assert entry["affiliated"] == []


async def test_consensus_merge_unions_affiliated():
    llm = FakeLLM(responses=["0"])
    m = mem(llm)
    out1 = await m.remember_atomic(["a"], "国王死于春天", affiliated=["x"])
    out2 = await m.remember_atomic(["b"], "国王死于春天", affiliated=["y"])
    assert out2["merged"] is True
    (entry,) = m.all_entries()
    assert set(entry["affiliated"]) == {"x", "y"}
    assert entry["id"] not in entry["affiliated"]  # never affiliates to self
    assert out1["id"] == out2["id"] == entry["id"]


async def test_link_group_symmetric_pairwise():
    m = mem(None)
    a = (await m.remember_atomic(["u"], "甲事件"))["id"]
    b = (await m.remember_atomic(["u"], "乙事件"))["id"]
    c = (await m.remember_atomic(["u"], "丙事件"))["id"]
    m.link_group([a, b, c])
    by_id = {e["id"]: e for e in m.all_entries()}
    assert set(by_id[a]["affiliated"]) == {b, c}
    assert set(by_id[b]["affiliated"]) == {a, c}
    assert set(by_id[c]["affiliated"]) == {a, b}
    for eid in (a, b, c):
        assert eid not in by_id[eid]["affiliated"]


async def test_link_group_skips_missing_ids():
    m = mem(None)
    a = (await m.remember_atomic(["u"], "甲事件"))["id"]
    b = (await m.remember_atomic(["u"], "乙事件"))["id"]
    m.link_group([a, b, "does-not-exist"])
    by_id = {e["id"]: e for e in m.all_entries()}
    assert set(by_id[a]["affiliated"]) == {b}
    assert set(by_id[b]["affiliated"]) == {a}


async def test_add_remove_get_affiliations():
    m = mem(None)
    a = (await m.remember_atomic(["u"], "甲事件"))["id"]
    b = (await m.remember_atomic(["u"], "乙事件"))["id"]
    c = (await m.remember_atomic(["u"], "丙事件"))["id"]

    assert m.add_affiliations(a, [b, c, a]) is True  # self excluded
    assert m.get_affiliations(a) == sorted([b, c])

    assert m.remove_affiliations(a, [b]) is True
    assert m.get_affiliations(a) == [c]

    assert m.add_affiliations("missing-id", [b]) is False
    assert m.remove_affiliations("missing-id", [b]) is False
    assert m.get_affiliations("missing-id") == []


async def test_export_restore_roundtrips_affiliated():
    m = mem(None)
    a = (await m.remember_atomic(["u"], "甲事件"))["id"]
    b = (await m.remember_atomic(["u"], "乙事件"))["id"]
    m.link_group([a, b])

    exported = m.export()
    m2 = mem(None)
    await m2.restore(exported)

    by_id = {e["id"]: e for e in m2.all_entries()}
    assert set(by_id[a]["affiliated"]) == {b}
    assert set(by_id[b]["affiliated"]) == {a}
