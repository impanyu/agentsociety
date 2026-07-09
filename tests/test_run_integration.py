import json
import os

import yaml

from society.run import run_scenario
from tests.helpers import FakeLLM, afake_embed

SCEN = {
    "scenario": "smoke", "language": "zh",
    "defaults": {"stats_interval": 10, "distance": 3},
    "agents": [
        {"id": "hall", "kind": "environment", "brain": "rule", "profile": "hall"},
        {"id": "study", "kind": "environment", "brain": "rule", "profile": "study"},
        {"id": "book", "kind": "info_carrier", "brain": "retrieval",
         "status": {"location": "hall"}, "corpus": "corpora/smoke.txt"},
        {"id": "amy", "kind": "character", "brain": "llm", "profile": "amy",
         "status": {"location": "hall"}, "goals": ["chat"]},
        {"id": "ben", "kind": "character", "brain": "llm", "profile": "ben",
         "status": {"location": "hall"}},
        {"id": "cid", "kind": "character", "brain": "llm", "profile": "cid",
         "status": {"location": "hall"}},
    ],
    "map": {"default_distance": 3},
    "kickoff": [{"to": ["amy"], "kind": "system", "content": "开始"}],
}


async def test_smoke_run_produces_all_outputs(tmp_path):
    scen_dir = tmp_path / "scen"
    (scen_dir / "corpora").mkdir(parents=True)
    (scen_dir / "corpora" / "smoke.txt").write_text("规则手册第一条。", encoding="utf-8")
    spath = scen_dir / "smoke.yaml"
    spath.write_text(yaml.safe_dump(SCEN, allow_unicode=True), encoding="utf-8")

    seq = {"amy": ['{"action": "observe", "params": {"target": "hall"}}',
                   '{"action": "say", "params": {"targets": ["ben", "cid"], "content": "大家好"}}',
                   '{"action": "remember", "params": {"text": "我在大厅打了招呼"}}',
                   '{"action": "pop_goal", "params": {}}'],
           "ben": ['{"action": "pop_message", "params": {}}',
                   '{"action": "say", "params": {"targets": ["amy"], "content": "你好"}}'],
           "cid": ['{"action": "pop_message", "params": {}}',
                   '{"action": "gesture", "params": {"targets": ["amy"], "description": "挥手"}}']}

    def fn(prompt, system=None):
        # Route by matching the agent id against the system prompt (each
        # LLMBrain's system prompt starts with its own profile text, which
        # here is just the agent id). Pop the next scripted response for
        # that agent, or idle with "wait" once its script is exhausted.
        for aid, responses in seq.items():
            if system and aid in system:
                if responses:
                    return responses.pop(0)
                return '{"action": "wait", "params": {}}'
        return '{"action": "wait", "params": {}}'

    llm = FakeLLM(fn=fn)

    out = str(tmp_path / "run1")
    summary = await run_scenario(str(spath), ticks=30, out_dir=out, llm=llm, embed_fn=afake_embed)
    assert os.path.exists(f"{out}/events.jsonl")
    assert os.path.exists(f"{out}/transcripts/amy.md")
    assert os.path.exists(f"{out}/config_snapshot.yaml")
    assert os.path.exists(f"{out}/llm_usage.json")
    assert summary["ticks_run"] >= 10

    stats = sorted(os.listdir(f"{out}/stats"))
    assert any(s.startswith("tick_") for s in stats)
    snap = json.load(open(f"{out}/stats/{stats[-1]}", encoding="utf-8"))
    und = snap["comm_graph"]["undirected"]
    assert und.get("amy|ben") and und.get("amy|cid")        # both directions counted

    amy_md = open(f"{out}/transcripts/amy.md", encoding="utf-8").read()
    assert "remember" in amy_md and "[tick" in amy_md
