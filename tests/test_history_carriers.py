"""Task F4: info-carrier document chaining in the sedimentation pipeline.

`_sediment_carriers` turns each `registry["carriers"]` entry into a CHAIN of
sentence memory-entries: one extraction LLM call per carrier returns its
content as a JSON array of sentence strings, each deposited via
`SharedMemory.remember_atomic([carrier_id], sentence, source="document",
readable=True, story_order=...)`, then chained i -> i+1 via
`SharedMemory.add_affiliations` (directional, NOT the symmetric
`link_group`). This is purely ADDITIVE to the LTM -- it must never touch
`_assemble_history_scenario` or `_write_corpora`.

No real API calls anywhere -- FakeLLM (scripted per-carrier, routed on the
"[carrier-extract]" marker baked into the prompt) + the async fake embed.
"""

import copy
import json
import os
import uuid

from society.history_extract import _sediment_carriers, extract_history
from society.ltm import SharedMemory
from tests.helpers import FakeLLM, afake_embed

REGISTRY_ONE_CARRIER = {
    "characters": [{"id": "alice", "name": "甲", "aliases": [], "profile": ""}],
    "locations": [{"id": "loc1", "name": "集市", "aliases": [], "profile": ""}],
    "carriers": [{"id": "letter1", "name": "一封信", "profile": "写给甲的信"}],
}


def _registry(*carrier_dicts) -> dict:
    reg = copy.deepcopy(REGISTRY_ONE_CARRIER)
    if carrier_dicts:
        reg["carriers"] = [dict(c) for c in carrier_dicts]
    return reg


def _new_shared(llm) -> SharedMemory:
    return SharedMemory(afake_embed, llm, collection_name=f"t_{uuid.uuid4().hex[:8]}", max_tokens=64)


def _routed_extract(by_carrier: dict[str, str], default: str = "[]"):
    """Routes the fake LLM by which carrier name/id appears in the
    "[carrier-extract]" prompt; anything else (e.g. a stray consensus
    "Which existing candidate" call) falls back to `default`."""

    def fake(prompt, system=None):
        if "[carrier-extract]" in prompt:
            for key, resp in by_carrier.items():
                if key in prompt:
                    return resp
        return default

    return fake


# ----------------------------------------------------------------------
# 1. Basic case: one carrier, 3-sentence JSON array -> 3 entries, each
#    owned by the carrier, each readable=True.
# ----------------------------------------------------------------------


async def test_sediment_carriers_deposits_three_entries_owned_and_readable():
    registry = _registry()
    sentences = ["一封信的第一句", "一封信的第二句", "一封信的第三句"]
    llm = FakeLLM(fn=_routed_extract({"一封信": json.dumps(sentences, ensure_ascii=False)}))
    shared = _new_shared(llm)
    warnings: list[str] = []

    await _sediment_carriers(
        llm, shared, registry, "SOURCE TEXT", "zh", "", warnings, story_order_base=1000
    )

    entries = shared.all_entries()
    assert len(entries) == 3
    texts = {e["text"] for e in entries}
    assert texts == set(sentences)
    for e in entries:
        assert e["owners"] == ["letter1"]
        assert e["readable"] is True
        assert e["meta"]["source"] == "document"
    assert warnings == []


# ----------------------------------------------------------------------
# 2. CHAIN DIRECTION: entry0 -> entry1 -> entry2, forward only (via
#    get_affiliations), never symmetric (unlike link_group).
# ----------------------------------------------------------------------


async def test_chain_direction_is_forward_only_not_symmetric():
    registry = _registry()
    sentences = ["第一句", "第二句", "第三句"]
    llm = FakeLLM(fn=_routed_extract({"一封信": json.dumps(sentences, ensure_ascii=False)}))
    shared = _new_shared(llm)
    warnings: list[str] = []

    await _sediment_carriers(
        llm, shared, registry, "SOURCE TEXT", "zh", "", warnings, story_order_base=1000
    )

    entries = {e["text"]: e for e in shared.all_entries()}
    e0, e1, e2 = entries["第一句"], entries["第二句"], entries["第三句"]

    assert shared.get_affiliations(e0["id"]) == [e1["id"]]
    assert shared.get_affiliations(e1["id"]) == [e2["id"]]
    # Last sentence in the chain has no next.
    assert shared.get_affiliations(e2["id"]) == []

    # Directional, not symmetric: e1 is NOT affiliated back to e0, and e2 is
    # NOT affiliated back to e1 (would be, under link_group's pairwise
    # symmetric linking, but add_affiliations is one-directional).
    assert e0["id"] not in shared.get_affiliations(e1["id"])
    assert e1["id"] not in shared.get_affiliations(e2["id"])

    # story_order is monotonic in chain/reading order, starting at the base.
    order_by_text = {t: e["meta"]["story_order"] for t, e in entries.items()}
    assert order_by_text["第一句"] < order_by_text["第二句"] < order_by_text["第三句"]
    assert order_by_text["第一句"] >= 1000


# ----------------------------------------------------------------------
# 3. Empty content / parse failure -> warning, no entries, no crash.
# ----------------------------------------------------------------------


async def test_empty_array_content_skips_carrier_with_warning():
    registry = _registry()
    llm = FakeLLM(fn=_routed_extract({"一封信": "[]"}))
    shared = _new_shared(llm)
    warnings: list[str] = []

    await _sediment_carriers(
        llm, shared, registry, "SOURCE TEXT", "zh", "", warnings, story_order_base=1000
    )

    assert shared.all_entries() == []
    assert any("letter1" in w and "no usable sentences" in w for w in warnings)


async def test_parse_failure_skips_carrier_with_warning_no_crash():
    registry = _registry()
    llm = FakeLLM(fn=_routed_extract({"一封信": "not json at all"}))
    shared = _new_shared(llm)
    warnings: list[str] = []

    await _sediment_carriers(
        llm, shared, registry, "SOURCE TEXT", "zh", "", warnings, story_order_base=1000
    )

    assert shared.all_entries() == []
    assert any("letter1" in w and "valid JSON" in w for w in warnings)


async def test_non_array_json_content_skips_carrier_with_warning():
    registry = _registry()
    llm = FakeLLM(fn=_routed_extract({"一封信": json.dumps({"not": "an array"})}))
    shared = _new_shared(llm)
    warnings: list[str] = []

    await _sediment_carriers(
        llm, shared, registry, "SOURCE TEXT", "zh", "", warnings, story_order_base=1000
    )

    assert shared.all_entries() == []
    assert any("letter1" in w for w in warnings)


# ----------------------------------------------------------------------
# 4. Multi-carrier independence: each carrier gets its own chain; carrier
#    A's entries are never chained to carrier B's.
# ----------------------------------------------------------------------


async def test_multiple_carriers_independent_chains():
    registry = _registry(
        {"id": "letter1", "name": "甲的信", "profile": "第一封信"},
        {"id": "diary1", "name": "乙的日记", "profile": "第二份文档"},
    )

    def fake(prompt, system=None):
        if "[carrier-extract]" in prompt:
            if "letter1" in prompt or "甲的信" in prompt:
                return json.dumps(["信件第一句", "信件第二句"], ensure_ascii=False)
            if "diary1" in prompt or "乙的日记" in prompt:
                return json.dumps(["日记第一句", "日记第二句"], ensure_ascii=False)
        return "[]"

    llm = FakeLLM(fn=fake)
    shared = _new_shared(llm)
    warnings: list[str] = []

    await _sediment_carriers(
        llm, shared, registry, "SOURCE TEXT", "zh", "", warnings, story_order_base=1000
    )

    entries = shared.all_entries()
    assert len(entries) == 4
    by_text = {e["text"]: e for e in entries}

    assert by_text["信件第一句"]["owners"] == ["letter1"]
    assert by_text["信件第二句"]["owners"] == ["letter1"]
    assert by_text["日记第一句"]["owners"] == ["diary1"]
    assert by_text["日记第二句"]["owners"] == ["diary1"]

    letter_id0 = by_text["信件第一句"]["id"]
    letter_id1 = by_text["信件第二句"]["id"]
    diary_id0 = by_text["日记第一句"]["id"]
    diary_id1 = by_text["日记第二句"]["id"]

    assert shared.get_affiliations(letter_id0) == [letter_id1]
    assert shared.get_affiliations(diary_id0) == [diary_id1]

    # Independence: no cross-carrier affiliation in either direction.
    assert diary_id0 not in shared.get_affiliations(letter_id0)
    assert diary_id1 not in shared.get_affiliations(letter_id1)
    assert letter_id0 not in shared.get_affiliations(diary_id0)
    assert letter_id1 not in shared.get_affiliations(diary_id1)


# ----------------------------------------------------------------------
# 5. Carriers-empty registry -> no-op (no LLM calls, no entries).
# ----------------------------------------------------------------------


async def test_carriers_empty_registry_is_noop():
    registry = _registry()
    registry["carriers"] = []
    llm = FakeLLM(fn=lambda prompt, system=None: (_ for _ in ()).throw(
        AssertionError("no LLM call should happen for an empty carriers registry")
    ))
    shared = _new_shared(llm)
    warnings: list[str] = []

    await _sediment_carriers(
        llm, shared, registry, "SOURCE TEXT", "zh", "", warnings, story_order_base=1000
    )

    assert shared.all_entries() == []
    assert warnings == []
    assert llm.calls == []


async def test_carriers_missing_key_is_noop():
    registry = _registry()
    del registry["carriers"]
    llm = FakeLLM(fn=lambda prompt, system=None: "[]")
    shared = _new_shared(llm)
    warnings: list[str] = []

    await _sediment_carriers(
        llm, shared, registry, "SOURCE TEXT", "zh", "", warnings, story_order_base=1000
    )

    assert shared.all_entries() == []
    assert llm.calls == []


# ----------------------------------------------------------------------
# 6. End-to-end wiring: extract_history includes carrier chains in the
#    exported LTM (deposited after Pass 2, before export), while assembly's
#    info_carrier agent build and corpus writing are unaffected.
# ----------------------------------------------------------------------

E2E_TEXT = ("甲乙" * 90) + ("乙甲" * 85)

E2E_REGISTRY = {
    "characters": [
        {"id": "alice", "name": "甲", "aliases": [], "profile": ""},
        {"id": "bob", "name": "乙", "aliases": [], "profile": ""},
    ],
    "locations": [{"id": "loc1", "name": "集市", "aliases": [], "profile": ""}],
    "carriers": [{"id": "letter1", "name": "一封信", "profile": "甲写给乙的信"}],
}

SUMMARY_RESPONSE = json.dumps(
    {"summary": "s", "characters": ["甲", "乙"], "locations": ["集市"]}, ensure_ascii=False
)
REGISTRY_RESPONSE = json.dumps(E2E_REGISTRY, ensure_ascii=False)
ROSTER_RESPONSE = json.dumps(
    {"characters": ["alice", "bob"], "state_updates": [], "story_time": "t1"}, ensure_ascii=False
)


async def test_extract_history_end_to_end_includes_carrier_chain_in_ltm_export(tmp_path):
    assert 300 < len(E2E_TEXT) < 400  # chunk_chars=200 below forces exactly 2 chunks

    kickoff_response = json.dumps(
        [{"to": ["alice"], "kind": "system", "content": "start"}], ensure_ascii=False
    )
    carrier_sentences = ["信中第一句", "信中第二句"]

    def fake(prompt, system=None):
        if "摘要" in prompt:
            return SUMMARY_RESPONSE
        if "注册表" in prompt:
            return REGISTRY_RESPONSE
        if "出场" in prompt:
            return ROSTER_RESPONSE
        if "[atomize]" in prompt:
            return json.dumps([["alice and bob crossed paths"]], ensure_ascii=False)
        if "[assign]" in prompt:
            return json.dumps([["alice", "bob"]], ensure_ascii=False)
        if "[carrier-extract]" in prompt:
            return json.dumps(carrier_sentences, ensure_ascii=False)
        if "起始" in prompt:
            return kickoff_response
        if "Which existing candidate" in prompt:
            return "-1"
        return "[]"

    llm = FakeLLM(fn=fake)
    out = str(tmp_path / "out.yaml")

    cfg = await extract_history(E2E_TEXT, llm, out, embed_fn=afake_embed, chunk_chars=200)

    # Assembly/corpus wiring is untouched: the carrier still becomes an
    # info_carrier agent with corpus/RetrievalBrain, exactly as before.
    carrier_agents = [a for a in cfg["agents"] if a["kind"] == "info_carrier"]
    assert len(carrier_agents) == 1
    assert carrier_agents[0]["id"] == "letter1"
    assert carrier_agents[0]["corpus"] == "corpora/letter1.txt"
    assert os.path.exists(os.path.join(os.path.dirname(out), "corpora", "letter1.txt"))

    # The carrier's chained entries are present in the exported holographic
    # LTM, additively alongside the event memories.
    with open(out + ".ltm.json", encoding="utf-8") as f:
        exported = json.load(f)

    letter_entries = [e for e in exported if e["owners"] == ["letter1"]]
    assert len(letter_entries) == 2
    texts = {e["text"] for e in letter_entries}
    assert texts == set(carrier_sentences)
    for e in letter_entries:
        assert e["readable"] is True

    by_text = {e["text"]: e for e in letter_entries}
    s0, s1 = by_text["信中第一句"], by_text["信中第二句"]
    assert s1["id"] in s0["affiliated"]
    assert s0["id"] not in s1["affiliated"]

    # Event memories from Pass 2 are still there too (purely additive).
    event_entries = [e for e in exported if e["owners"] != ["letter1"]]
    assert any("alice and bob crossed paths" in e["text"] for e in event_entries)
