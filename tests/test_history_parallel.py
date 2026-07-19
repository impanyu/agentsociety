"""Task 3a: per-character extraction calls in the exhaustive history-
sedimentation pipeline (`_process_chunk_exhaustive`) must run CONCURRENTLY
via asyncio.gather instead of sequentially, while preserving:

  1. Deterministic `appearing_ids` order for per_char_entries / story_order
     (NOT completion order).
  2. Per-character error isolation (invalid JSON, or a genuine post-retry
     exception from llm.chat) -- same skip+warning behaviour, one bad cid
     never aborts the chunk or affects siblings.
  3. Identical final output vs. the old sequential behaviour.

These tests use direct duck-typed fake LLM clients (not tests.helpers.FakeLLM,
whose chat() never yields control back to the event loop and so can't prove
concurrency) with real coordination via asyncio.Event -- no wall-clock
sleeps, so nothing here can flake on timing.
"""

import asyncio
import json
import uuid

from society.history_extract import _process_chunk_exhaustive, _RegistryResolvers
from society.ltm import SharedMemory
from tests.helpers import afake_embed

REGISTRY = {
    "characters": [
        {"id": "alice", "name": "甲", "aliases": [], "profile": ""},
        {"id": "bob", "name": "乙", "aliases": [], "profile": ""},
        {"id": "carol", "name": "丙", "aliases": [], "profile": ""},
    ],
    "locations": [{"id": "loc1", "name": "集市", "aliases": [], "profile": ""}],
    "carriers": [],
}

CHUNK = {"text": "甲乙丙的故事" * 5, "title": "第一回", "flat_idx": 0}

ROSTER_RESPONSE = json.dumps(
    {"characters": ["alice", "bob", "carol"], "state_updates": [], "story_time": "t1"},
    ensure_ascii=False,
)


def _cid_in_prompt(prompt: str) -> str | None:
    for cid in ("alice", "bob", "carol"):
        if f"id={cid}" in prompt:
            return cid
    return None


def _new_shared(llm) -> SharedMemory:
    return SharedMemory(afake_embed, llm, collection_name=f"t_{uuid.uuid4().hex[:8]}")


# ----------------------------------------------------------------------
# 1. Concurrency proof
# ----------------------------------------------------------------------


class BarrierLLM:
    """Roster call answers immediately; each per-character (沉淀) call blocks
    until `n_expected` such calls are simultaneously in-flight, then all are
    released together. If the calls were issued sequentially, the first one
    would block forever (only 1 ever in-flight) -- caught via wait_for."""

    def __init__(self, n_expected: int):
        self.n_expected = n_expected
        self.in_flight = 0
        self.max_in_flight = 0
        self._arrived = 0
        self._release = asyncio.Event()

    async def chat(self, prompt, system=None, bucket="extract"):
        if "出场" in prompt:
            return ROSTER_RESPONSE
        if "沉淀" in prompt:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
            self._arrived += 1
            if self._arrived >= self.n_expected:
                self._release.set()
            else:
                await self._release.wait()
            self.in_flight -= 1
            cid = _cid_in_prompt(prompt)
            return json.dumps([f"【t1】{cid} did something"], ensure_ascii=False)
        if "补漏" in prompt:
            return json.dumps({}, ensure_ascii=False)
        return "[]"


async def test_per_character_extract_calls_run_concurrently():
    llm = BarrierLLM(n_expected=3)
    shared = _new_shared(llm)
    resolvers = _RegistryResolvers(REGISTRY)
    warnings: list[str] = []
    raw_state_updates: list[tuple[int, dict]] = []

    # If the implementation is sequential, the first 沉淀 call deadlocks
    # (waits for 3 in-flight, only ever reaches 1) -- wait_for turns that
    # hang into a fast, deterministic test failure instead of a hung suite.
    await asyncio.wait_for(
        _process_chunk_exhaustive(
            llm,
            shared,
            CHUNK,
            REGISTRY,
            resolvers,
            "",
            warnings,
            0,
            raw_state_updates,
        ),
        timeout=5,
    )

    assert llm.max_in_flight >= 2
    assert warnings == []
    entries = shared.all_entries()
    assert len(entries) == 3


# ----------------------------------------------------------------------
# 2. Output-unchanged: assembly order follows appearing_ids, not completion
#    order.
# ----------------------------------------------------------------------


class CompletionOrderLLM:
    """Per-character calls deliberately resolve out of appearing_ids order
    (bob completes first with zero yields, then carol after one yield, then
    alice after three yields) -- proves per_char_entries/story_order
    assembly uses input order, not completion order."""

    def __init__(self):
        self.completion_order: list[str] = []

    async def chat(self, prompt, system=None, bucket="extract"):
        if "出场" in prompt:
            return ROSTER_RESPONSE
        if "沉淀" in prompt:
            cid = _cid_in_prompt(prompt)
            yields = {"alice": 3, "bob": 0, "carol": 1}[cid]
            for _ in range(yields):
                await asyncio.sleep(0)
            self.completion_order.append(cid)
            return json.dumps([f"【t1】{cid} fact"], ensure_ascii=False)
        if "补漏" in prompt:
            return json.dumps({}, ensure_ascii=False)
        return "[]"


async def test_output_and_order_unchanged_despite_completion_order():
    llm = CompletionOrderLLM()
    shared = _new_shared(llm)
    resolvers = _RegistryResolvers(REGISTRY)
    warnings: list[str] = []
    raw_state_updates: list[tuple[int, dict]] = []

    await _process_chunk_exhaustive(
        llm, shared, CHUNK, REGISTRY, resolvers, "", warnings, 0, raw_state_updates
    )

    # Sanity check that completion order really was shuffled by the fake --
    # otherwise this test wouldn't be exercising anything interesting.
    assert llm.completion_order != ["alice", "bob", "carol"]

    assert warnings == []
    entries = sorted(shared.all_entries(), key=lambda e: e["meta"]["story_order"])
    assert [e["owners"] for e in entries] == [["alice"], ["bob"], ["carol"]]
    assert [e["meta"]["story_order"] for e in entries] == [0, 1, 2]
    assert [e["text"] for e in entries] == [
        "【t1】alice fact",
        "【t1】bob fact",
        "【t1】carol fact",
    ]


# ----------------------------------------------------------------------
# 3. Error isolation: invalid JSON from one character
# ----------------------------------------------------------------------


class InvalidJsonForOneLLM:
    def __init__(self, bad_cid: str):
        self.bad_cid = bad_cid

    async def chat(self, prompt, system=None, bucket="extract"):
        if "出场" in prompt:
            return ROSTER_RESPONSE
        if "沉淀" in prompt:
            cid = _cid_in_prompt(prompt)
            if cid == self.bad_cid:
                return "not json at all"
            return json.dumps([f"【t1】{cid} fact"], ensure_ascii=False)
        if "补漏" in prompt:
            return json.dumps({}, ensure_ascii=False)
        return "[]"


async def test_invalid_json_for_one_character_skipped_others_unaffected():
    llm = InvalidJsonForOneLLM(bad_cid="bob")
    shared = _new_shared(llm)
    resolvers = _RegistryResolvers(REGISTRY)
    warnings: list[str] = []
    raw_state_updates: list[tuple[int, dict]] = []

    await _process_chunk_exhaustive(
        llm, shared, CHUNK, REGISTRY, resolvers, "", warnings, 0, raw_state_updates
    )

    assert len(warnings) == 1
    assert "failed for 'bob'" in warnings[0]
    assert "chunk 0" in warnings[0]

    entries = shared.all_entries()
    owners = sorted(o for e in entries for o in e["owners"])
    assert owners == ["alice", "carol"]


# ----------------------------------------------------------------------
# 4. Error isolation: a genuine (post-retry) exception from llm.chat itself
# ----------------------------------------------------------------------


class RaisingForOneLLM:
    """Simulates llm.chat raising after exhausting its internal retries for
    exactly one character; siblings must still complete normally and the
    chunk must not abort."""

    def __init__(self, raising_cid: str):
        self.raising_cid = raising_cid

    async def chat(self, prompt, system=None, bucket="extract"):
        if "出场" in prompt:
            return ROSTER_RESPONSE
        if "沉淀" in prompt:
            cid = _cid_in_prompt(prompt)
            if cid == self.raising_cid:
                raise RuntimeError("simulated post-retry transport failure")
            return json.dumps([f"【t1】{cid} fact"], ensure_ascii=False)
        if "补漏" in prompt:
            return json.dumps({}, ensure_ascii=False)
        return "[]"


async def test_exception_from_one_character_degrades_to_skip_not_abort():
    llm = RaisingForOneLLM(raising_cid="carol")
    shared = _new_shared(llm)
    resolvers = _RegistryResolvers(REGISTRY)
    warnings: list[str] = []
    raw_state_updates: list[tuple[int, dict]] = []

    # Must NOT raise -- the whole point is that one cid's exception degrades
    # to a skip+warning instead of propagating and aborting the chunk.
    await _process_chunk_exhaustive(
        llm, shared, CHUNK, REGISTRY, resolvers, "", warnings, 0, raw_state_updates
    )

    assert len(warnings) == 1
    assert "failed for 'carol'" in warnings[0]
    assert "simulated post-retry transport failure" in warnings[0]

    entries = shared.all_entries()
    owners = sorted(o for e in entries for o in e["owners"])
    assert owners == ["alice", "bob"]
