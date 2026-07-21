"""Task B2: atomize -> assign-owners history-sedimentation pipeline
(`detail="atomic"`, now the DEFAULT of `extract_history`).

Replaces per-character extraction with: (1) a roster call for state only,
(2) one atomize call breaking the chunk into complete, self-contained,
source-language fragments, (3) one assign call attributing each fragment to
its owner role id(s) (falling back to the reserved `narrator` role). Each
fragment is deposited exactly once via `SharedMemory.remember_atomic`,
however many owners it has.

No real API calls anywhere -- FakeLLM (scripted per-call, routed on the
distinguishing marker text baked into each new prompt) + the async fake
embed throughout.
"""

import copy
import json
import uuid

import pytest

from society.history_extract import (
    NARRATOR_ID,
    _RegistryResolvers,
    _ensure_narrator_role,
    _process_chunk_atomic,
    _run_sediment_pass_atomic,
    extract_history,
)
from society.ltm import SharedMemory
from society.textlen import count_tokens
from tests.helpers import FakeLLM, afake_embed

REGISTRY = {
    "characters": [
        {"id": "alice", "name": "甲", "aliases": [], "profile": ""},
        {"id": "bob", "name": "乙", "aliases": [], "profile": ""},
    ],
    "locations": [{"id": "loc1", "name": "集市", "aliases": [], "profile": ""}],
    "carriers": [],
}


def _registry() -> dict:
    return copy.deepcopy(REGISTRY)


def _new_shared(llm) -> SharedMemory:
    return SharedMemory(afake_embed, llm, collection_name=f"t_{uuid.uuid4().hex[:8]}")


def _chunk(text: str, flat_idx: int = 0, title: str | None = "第一回") -> dict:
    return {"text": text, "title": title, "flat_idx": flat_idx}


ROSTER_RESPONSE = json.dumps(
    {
        "characters": ["alice", "bob"],
        "state_updates": [{"id": "alice", "location": "loc1", "alive": True}],
        "story_time": "t1",
    },
    ensure_ascii=False,
)


def _routed_fake(atomize_response: str, assign_response: str, roster_response: str = ROSTER_RESPONSE):
    def fake(prompt, system=None):
        if "[atomize]" in prompt:
            return atomize_response
        if "[assign]" in prompt:
            return assign_response
        if "出场" in prompt:
            return roster_response
        return "[]"

    return fake


# ----------------------------------------------------------------------
# 1. _process_chunk_atomic: single-owner, multi-owner, narrator fallback.
# ----------------------------------------------------------------------


async def test_process_chunk_atomic_single_multi_and_narrator_fallback():
    registry = _registry()
    _ensure_narrator_role(registry)  # this chunk-level unit exercises resolver directly
    resolvers = _RegistryResolvers(registry)
    warnings: list[str] = []

    atomize_response = json.dumps(
        [
            "alice went to the market",
            "alice and bob met at the market",
            "the market was crowded",
        ],
        ensure_ascii=False,
    )
    assign_response = json.dumps([["alice"], ["alice", "bob"], []], ensure_ascii=False)
    llm = FakeLLM(fn=_routed_fake(atomize_response, assign_response))

    result = await _process_chunk_atomic(llm, _chunk("正文"), registry, resolvers, "", warnings)

    assert result["flat_idx"] == 0
    assert result["story_time"] == "t1"
    assert result["state_updates"] == [{"id": "alice", "location": "loc1", "alive": True}]

    deposits = result["deposits"]
    assert deposits[0] == ("alice went to the market", ["alice"])
    assert deposits[1][0] == "alice and bob met at the market"
    assert sorted(deposits[1][1]) == ["alice", "bob"]
    assert deposits[2] == ("the market was crowded", [NARRATOR_ID])
    assert warnings == []


# ----------------------------------------------------------------------
# 2. End-to-end via _run_sediment_pass_atomic: deposits land in SharedMemory
#    with the right owners; narrator role auto-added to the registry.
# ----------------------------------------------------------------------


async def test_run_sediment_pass_atomic_deposits_with_owners_and_adds_narrator_role():
    registry = _registry()
    assert not any(loc.get("id") == NARRATOR_ID for loc in registry["locations"])

    atomize_response = json.dumps(
        ["alice bought bread", "alice and bob argued over bread", "birds flew overhead"],
        ensure_ascii=False,
    )
    assign_response = json.dumps([["alice"], ["alice", "bob"], []], ensure_ascii=False)
    llm = FakeLLM(fn=_routed_fake(atomize_response, assign_response))
    shared = _new_shared(llm)
    warnings: list[str] = []

    state = await _run_sediment_pass_atomic(llm, shared, [_chunk("正文")], registry, "", warnings)

    # Narrator role added to the registry (mutated in place) -- and exactly once.
    narrator_entries = [loc for loc in registry["locations"] if loc["id"] == NARRATOR_ID]
    assert len(narrator_entries) == 1
    assert narrator_entries[0]["name"] == "Narrator/旁白"

    entries = shared.all_entries()
    assert len(entries) == 3
    by_text = {e["text"]: e for e in entries}
    assert by_text["alice bought bread"]["owners"] == ["alice"]
    assert sorted(by_text["alice and bob argued over bread"]["owners"]) == ["alice", "bob"]
    assert by_text["birds flew overhead"]["owners"] == [NARRATOR_ID]

    # State from the roster call still resolves normally.
    assert state["alice"]["location"] == "loc1"
    assert warnings == []


# ----------------------------------------------------------------------
# 3. Two chunks: deposits stay in story order (flat_idx*1000+i), even
#    though the read-only LLM phase is gathered concurrently across chunks.
# ----------------------------------------------------------------------


async def test_two_chunks_deposit_order_matches_story_order():
    registry = _registry()

    def fake(prompt, system=None):
        if "[atomize]" in prompt:
            if "ALPHA" in prompt:
                return json.dumps(["alpha fragment one", "alpha fragment two"], ensure_ascii=False)
            return json.dumps(["beta fragment one"], ensure_ascii=False)
        if "[assign]" in prompt:
            if "alpha fragment one" in prompt:
                return json.dumps([["alice"], ["bob"]], ensure_ascii=False)
            return json.dumps([["alice"]], ensure_ascii=False)
        if "出场" in prompt:
            return json.dumps(
                {"characters": [], "state_updates": [], "story_time": None}, ensure_ascii=False
            )
        return "[]"

    llm = FakeLLM(fn=fake)
    shared = _new_shared(llm)
    warnings: list[str] = []

    chunks = [_chunk("ALPHA chunk text", flat_idx=0, title=None), _chunk("BETA chunk text", flat_idx=1, title=None)]
    await _run_sediment_pass_atomic(llm, shared, chunks, registry, "", warnings)

    entries = sorted(shared.all_entries(), key=lambda e: e["meta"]["story_order"])
    assert [e["meta"]["story_order"] for e in entries] == [0, 1, 1000]
    assert [e["text"] for e in entries] == [
        "alpha fragment one",
        "alpha fragment two",
        "beta fragment one",
    ]
    assert warnings == []


# ----------------------------------------------------------------------
# 4. Language passthrough: English atomize output is stored verbatim (no
#    forced translation) -- the pipeline just passes text through.
# ----------------------------------------------------------------------


async def test_english_fragments_stored_verbatim_no_translation():
    registry = _registry()
    english_fragment = "Napoleon crossed the river at dawn."
    atomize_response = json.dumps([english_fragment], ensure_ascii=False)
    assign_response = json.dumps([["alice"]], ensure_ascii=False)
    llm = FakeLLM(fn=_routed_fake(atomize_response, assign_response))
    shared = _new_shared(llm)
    warnings: list[str] = []

    await _run_sediment_pass_atomic(llm, shared, [_chunk("Some English source text.")], registry, "", warnings)

    entries = shared.all_entries()
    assert len(entries) == 1
    assert entries[0]["text"] == english_fragment


# ----------------------------------------------------------------------
# 5. Token cap: an over-length fragment is stored truncated (enforced by
#    `remember_atomic`, but this proves the atomic pipeline actually routes
#    through it instead of e.g. the normalize-split `remember` path).
# ----------------------------------------------------------------------


async def test_overlength_fragment_stored_truncated():
    registry = _registry()
    long_fragment = "word " * 120  # well over the default 50-token cap
    assert count_tokens(long_fragment) > 50

    atomize_response = json.dumps([long_fragment], ensure_ascii=False)
    assign_response = json.dumps([["alice"]], ensure_ascii=False)
    llm = FakeLLM(fn=_routed_fake(atomize_response, assign_response))
    shared = _new_shared(llm)
    warnings: list[str] = []

    await _run_sediment_pass_atomic(llm, shared, [_chunk("正文")], registry, "", warnings)

    entries = shared.all_entries()
    assert len(entries) == 1
    assert count_tokens(entries[0]["text"]) <= 50
    assert entries[0]["text"] != long_fragment.strip()


# ----------------------------------------------------------------------
# 6. Assign length-mismatch and unknown-owner-id degrade gracefully.
# ----------------------------------------------------------------------


async def test_assign_length_mismatch_pads_defensively_with_warning():
    registry = _registry()
    atomize_response = json.dumps(["fragment one", "fragment two", "fragment three"], ensure_ascii=False)
    # Only ONE owner list for THREE fragments.
    assign_response = json.dumps([["alice"]], ensure_ascii=False)
    llm = FakeLLM(fn=_routed_fake(atomize_response, assign_response))
    shared = _new_shared(llm)
    warnings: list[str] = []

    await _run_sediment_pass_atomic(llm, shared, [_chunk("正文")], registry, "", warnings)

    entries = shared.all_entries()
    assert len(entries) == 3
    by_text = {e["text"]: e for e in entries}
    assert by_text["fragment one"]["owners"] == ["alice"]
    # Padded fragments fall back to narrator, not a crash.
    assert by_text["fragment two"]["owners"] == [NARRATOR_ID]
    assert by_text["fragment three"]["owners"] == [NARRATOR_ID]
    assert any("padding/truncating" in w for w in warnings)


async def test_assign_unknown_owner_id_dropped_with_warning_not_crash():
    registry = _registry()
    atomize_response = json.dumps(["fragment with a bogus owner"], ensure_ascii=False)
    assign_response = json.dumps([["ghost_id"]], ensure_ascii=False)
    llm = FakeLLM(fn=_routed_fake(atomize_response, assign_response))
    shared = _new_shared(llm)
    warnings: list[str] = []

    await _run_sediment_pass_atomic(llm, shared, [_chunk("正文")], registry, "", warnings)

    entries = shared.all_entries()
    assert len(entries) == 1
    # Unknown ref dropped -> no valid owners left -> narrator fallback.
    assert entries[0]["owners"] == [NARRATOR_ID]
    assert any("unknown id" in w and "ghost_id" in w for w in warnings)


async def test_assign_call_invalid_json_falls_back_to_narrator_for_whole_chunk():
    registry = _registry()
    atomize_response = json.dumps(["fragment one", "fragment two"], ensure_ascii=False)
    assign_response = "not json at all"
    llm = FakeLLM(fn=_routed_fake(atomize_response, assign_response))
    shared = _new_shared(llm)
    warnings: list[str] = []

    await _run_sediment_pass_atomic(llm, shared, [_chunk("正文")], registry, "", warnings)

    entries = shared.all_entries()
    assert len(entries) == 2
    assert all(e["owners"] == [NARRATOR_ID] for e in entries)
    assert any("assign pass failed" in w for w in warnings)


# ----------------------------------------------------------------------
# 7. Atomize failure skips the chunk's memories but keeps roster state.
# ----------------------------------------------------------------------


async def test_atomize_failure_skips_memories_but_keeps_roster_state():
    registry = _registry()

    def fake(prompt, system=None):
        if "出场" in prompt:
            return ROSTER_RESPONSE
        if "[atomize]" in prompt:
            return "not json at all"
        return "[]"

    llm = FakeLLM(fn=fake)
    shared = _new_shared(llm)
    warnings: list[str] = []

    state = await _run_sediment_pass_atomic(llm, shared, [_chunk("正文")], registry, "", warnings)

    assert shared.all_entries() == []
    assert state["alice"]["location"] == "loc1"
    assert any("atomize pass failed" in w for w in warnings)


# ----------------------------------------------------------------------
# 8. Roster failure still proceeds with atomize/assign (state best-effort).
# ----------------------------------------------------------------------


async def test_roster_failure_still_proceeds_with_atomize_assign():
    registry = _registry()

    def fake(prompt, system=None):
        if "出场" in prompt:
            return "not json at all"
        if "[atomize]" in prompt:
            return json.dumps(["alice did something"], ensure_ascii=False)
        if "[assign]" in prompt:
            return json.dumps([["alice"]], ensure_ascii=False)
        return "[]"

    llm = FakeLLM(fn=fake)
    shared = _new_shared(llm)
    warnings: list[str] = []

    state = await _run_sediment_pass_atomic(
        llm, shared, [_chunk("正文", title="第一回")], registry, "", warnings
    )

    entries = shared.all_entries()
    assert len(entries) == 1
    assert entries[0]["text"] == "alice did something"
    assert entries[0]["meta"]["story_time"] == "【第一回】"
    # No state_updates were recorded (roster failed) -> empty state table.
    assert state == {}
    assert any("roster" in w and "chunk 0" in w for w in warnings)


# ----------------------------------------------------------------------
# 9. registry_only gate still returns after Pass 1, with the new atomic
#    default -- Pass 2 (and hence atomize/assign) never runs.
# ----------------------------------------------------------------------


SUMMARY_RESPONSE = json.dumps(
    {"summary": "s", "characters": ["甲"], "locations": ["集市"]}, ensure_ascii=False
)
REGISTRY_RESPONSE = json.dumps(REGISTRY, ensure_ascii=False)


async def test_registry_only_gate_with_default_atomic_detail(tmp_path):
    def fake(prompt, system=None):
        if "摘要" in prompt:
            return SUMMARY_RESPONSE
        if "注册表" in prompt:
            return REGISTRY_RESPONSE
        raise AssertionError(f"Pass 2 must not run under registry_only: {prompt[:50]!r}")

    llm = FakeLLM(fn=fake)
    out = str(tmp_path / "out.yaml")

    result = await extract_history(
        "正文" * 50, llm, out, embed_fn=afake_embed, chunk_chars=200, registry_only=True
    )

    assert result["registry"]["characters"][0]["id"] == "alice"
    import os

    assert os.path.exists(out + ".registry.json")


# ----------------------------------------------------------------------
# 10. Full extract_history pipeline with the DEFAULT detail (no detail=
#     kwarg passed) exercises the atomic path end-to-end and proves no
#     per-character (沉淀/补漏) extraction call remains in it.
# ----------------------------------------------------------------------


async def test_extract_history_default_detail_is_atomic_end_to_end(tmp_path):
    text = ("甲乙" * 90) + ("乙甲" * 85)
    assert 300 < len(text) < 400  # chunk_chars=200 below forces exactly 2 chunks

    kickoff_response = json.dumps(
        [{"to": ["alice"], "kind": "system", "content": "start"}], ensure_ascii=False
    )

    def make_fake():
        def fake(prompt, system=None):
            if "摘要" in prompt:
                return SUMMARY_RESPONSE
            if "注册表" in prompt:
                return REGISTRY_RESPONSE
            if "出场" in prompt:
                return ROSTER_RESPONSE
            if "[atomize]" in prompt:
                return json.dumps(["alice and bob crossed paths"], ensure_ascii=False)
            if "[assign]" in prompt:
                return json.dumps([["alice", "bob"]], ensure_ascii=False)
            if "起始" in prompt:
                return kickoff_response
            if "Which existing candidate" in prompt:
                return "0"
            return "[]"

        return fake

    llm = FakeLLM(fn=make_fake())
    out = str(tmp_path / "out.yaml")

    cfg = await extract_history(text, llm, out, embed_fn=afake_embed, chunk_chars=200)

    # No detail= was passed -- this exercises the new default end-to-end.
    for _, prompt, _ in llm.calls:
        assert "沉淀" not in prompt
        assert "补漏" not in prompt

    import os

    assert os.path.exists(out)
    assert os.path.exists(out + ".ltm.json")

    agent_ids = {a["id"] for a in cfg["agents"]}
    assert NARRATOR_ID in agent_ids
    narrator_agent = next(a for a in cfg["agents"] if a["id"] == NARRATOR_ID)
    assert narrator_agent["kind"] == "environment"
