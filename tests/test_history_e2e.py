"""End-to-end test for Task H4: the full history-sedimentation loop --
extract_history (2-chunk micro-novel, FakeLLM) -> build_society (restores
the sedimented LTM, no seed replay) -> kernel.run() for a handful of ticks,
driven by a scripted brain for the surviving character. No network calls
anywhere (FakeLLM + afake_embed throughout).
"""

import json
import os

from society.events import EventLog
from society.history_extract import extract_history
from society.scenario import build_society, load_scenario
from tests.helpers import FakeLLM, afake_embed

# Same trick as test_history_extract.py: the FakeLLM routes purely on the
# marker word present in each stage's prompt, so chunk *content* doesn't
# matter, only its length (chunk_chars=200 below forces exactly 2 chunks).
TEXT = ("董卓" * 90) + ("曹操" * 85)
assert 300 < len(TEXT) < 400

REGISTRY = {
    "characters": [
        {"id": "dongzhuo", "name": "董卓", "aliases": ["太师"], "profile": "汉末权臣,把持朝政"},
        {"id": "caocao", "name": "曹操", "aliases": ["孟德", "丞相"], "profile": "魏武帝"},
    ],
    "locations": [
        {"id": "luoyang", "name": "洛阳", "profile": "汉末京师"},
    ],
    "carriers": [],
}

SUMMARY_RESPONSE = json.dumps(
    {"summary": "董卓与曹操的一段往事", "characters": ["董卓", "曹操"], "locations": ["洛阳"]},
    ensure_ascii=False,
)
REGISTRY_RESPONSE = json.dumps(REGISTRY, ensure_ascii=False)

# The overlapping fact both characters "remember" in both chunks -> a
# single multi-owner consensus entry (identical text -> identical fake
# embedding -> the scripted equivalence answer "0" merges every insert
# into the same entry).
OVERLAP_FACT = "【初平三年】董卓死于吕布之手"

KICKOFF_RESPONSE = json.dumps(
    [{"to": ["caocao"], "kind": "system", "content": "后传伊始,风声再起"}], ensure_ascii=False
)


def _sediment_response(chunk_idx: int) -> str:
    memories = {"dongzhuo": [OVERLAP_FACT], "caocao": [OVERLAP_FACT]}
    if chunk_idx == 0:
        state_updates = [
            {"id": "dongzhuo", "location": "luoyang", "alive": True},
            {"id": "caocao", "location": "luoyang", "alive": True},
        ]
    else:
        # Final chunk: dongzhuo dies, caocao survives.
        state_updates = [
            {"id": "dongzhuo", "location": "luoyang", "alive": False},
            {"id": "caocao", "location": "luoyang", "alive": True},
        ]
    return json.dumps(
        {"memories": memories, "state_updates": state_updates, "story_time": "初平三年"},
        ensure_ascii=False,
    )


def make_extraction_fake():
    """Routes extract_history's registry+sediment+kickoff calls, plus
    SharedMemory's internal consensus-equivalence check ("0" always merges
    into the first/only candidate -- the point of this fixture)."""
    counter = {"沉淀": 0}

    def fake(prompt, system=None):
        if "摘要" in prompt:
            return SUMMARY_RESPONSE
        if "注册表" in prompt:
            return REGISTRY_RESPONSE
        if "沉淀" in prompt:
            idx = counter["沉淀"]
            counter["沉淀"] += 1
            return _sediment_response(idx)
        if "起始" in prompt:
            return KICKOFF_RESPONSE
        if "Which existing candidate" in prompt:
            return "0"
        return "[]"

    return fake


async def test_history_e2e_full_pipeline(tmp_path):
    out = str(tmp_path / "sanguo_sequel.yaml")
    extraction_llm = FakeLLM(fn=make_extraction_fake())

    await extract_history(TEXT, extraction_llm, out, embed_fn=afake_embed, chunk_chars=200)

    # ------------------------------------------------------------------
    # 1. extract_history wrote yaml + .ltm.json + registry.json.
    # ------------------------------------------------------------------
    ltm_path = out + ".ltm.json"
    registry_path = out + ".registry.json"
    assert os.path.exists(out)
    assert os.path.exists(ltm_path)
    assert os.path.exists(registry_path)

    loaded = load_scenario(out)
    by_id = {a["id"]: a for a in loaded["agents"]}
    assert by_id["dongzhuo"]["archived"] is True
    assert by_id["caocao"].get("archived", False) is False
    assert by_id["caocao"]["goals"] == []
    assert loaded["ltm_file"] == os.path.basename(ltm_path)

    # ------------------------------------------------------------------
    # 2. build_society: sediment restored (multi-owner entry present),
    #    dead agent archived, alive agent starts with an empty goal stack.
    # ------------------------------------------------------------------
    decide_calls = {"n": 0}
    captured = {}

    def run_fake(prompt, system=None):
        if system is not None:
            # A decide() call from an LLMBrain always carries its system
            # prompt; the only "llm"-brain agent left eligible in this
            # scenario is the surviving caocao (dongzhuo is archived).
            n = decide_calls["n"]
            decide_calls["n"] += 1
            if n == 0:
                captured["saw_goal_hint"] = "goal_hint" in prompt
                return json.dumps({"action": "noop", "params": {}}, ensure_ascii=False)
            if n == 1:
                return json.dumps(
                    {"action": "push_goal", "params": {"text": "光复汉室,匡扶天下"}},
                    ensure_ascii=False,
                )
            return json.dumps({"action": "noop", "params": {}}, ensure_ascii=False)
        if "Which existing candidate" in prompt:
            return "0"
        return "[]"

    run_llm = FakeLLM(fn=run_fake)
    event_log = EventLog(None)
    k = await build_society(loaded, llm=run_llm, embed_fn=afake_embed, event_log=event_log)

    entries = k.shared_memory.all_entries()
    multi_owner = [e for e in entries if len(e["owners"]) >= 2]
    assert multi_owner, "expected a multi-owner consensus entry from the overlapping fact"
    assert set(multi_owner[0]["owners"]) == {"caocao", "dongzhuo"}

    assert k.agents["dongzhuo"].archived is True
    assert k.agents["caocao"].archived is False
    assert k.agents["caocao"].stm.goals.empty()

    # ------------------------------------------------------------------
    # 3. run 5 ticks: kickoff wakes caocao; its scripted brain sees
    #    goal_hint on the first decide, then pushes a goal, then noops.
    #    The archived agent never decides (zero action events).
    # ------------------------------------------------------------------
    await k.run(max_ticks=5)

    assert captured.get("saw_goal_hint") is True
    assert decide_calls["n"] >= 2
    assert not k.agents["caocao"].stm.goals.empty()

    events = event_log.all()
    dongzhuo_actions = [e for e in events if e["kind"] == "action" and e["agent"] == "dongzhuo"]
    assert dongzhuo_actions == []

    caocao_actions = [e for e in events if e["kind"] == "action" and e["agent"] == "caocao"]
    assert len(caocao_actions) >= 2
    assert caocao_actions[0]["action"]["name"] == "noop"
    assert caocao_actions[1]["action"]["name"] == "push_goal"

    # ------------------------------------------------------------------
    # 4. stats snapshot: the merged consensus entry counts as "shared".
    # ------------------------------------------------------------------
    snap = k.metrics.snapshot(k.tick)
    assert snap["consensus_ratio"]["shared"] >= 1
