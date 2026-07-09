import json, pytest
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
