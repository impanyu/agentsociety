import json
import uuid

import pytest
import yaml

from society.events import EventLog
from society.ltm import SharedMemory
from society.scenario import build_society, load_scenario
from tests.helpers import FakeLLM, afake_embed


class _CountingEmbed:
    """Wraps afake_embed and counts how many times it's invoked."""

    def __init__(self):
        self.calls = 0

    async def __call__(self, texts):
        self.calls += 1
        return await afake_embed(texts)


def mem(llm=None, **kw):
    return SharedMemory(
        afake_embed, llm=llm, collection_name=f"t_{uuid.uuid4().hex[:8]}", **kw
    )


# ----------------------------------------------------------------------
# 1. story fields stored + exported/restored
# ----------------------------------------------------------------------

async def test_story_fields_stored_and_exported():
    embed = _CountingEmbed()
    m = SharedMemory(embed, collection_name=f"t_{uuid.uuid4().hex[:8]}")

    await m.remember(
        "guanyu", "关羽挂印封金离开曹营", story_order=3005, story_time="建安五年"
    )

    entries = m.all_entries()
    assert len(entries) == 1
    meta = entries[0]["meta"]
    assert meta["story_order"] == 3005
    assert meta["story_time"] == "建安五年"

    exported = m.export()
    assert len(exported) == 1
    assert exported[0]["meta"]["story_order"] == 3005
    assert exported[0]["meta"]["story_time"] == "建安五年"

    calls_before_restore = embed.calls
    embed2 = _CountingEmbed()
    m2 = SharedMemory(embed2, collection_name=f"t_{uuid.uuid4().hex[:8]}")
    await m2.restore(exported)
    assert embed2.calls == 0  # entries carry embeddings -> no recompute
    assert embed.calls == calls_before_restore  # export() itself made no embed calls

    restored_entries = m2.all_entries()
    assert len(restored_entries) == 1
    assert restored_entries[0]["meta"]["story_order"] == 3005
    assert restored_entries[0]["meta"]["story_time"] == "建安五年"


# ----------------------------------------------------------------------
# 2. consensus merge keeps the smaller story_order
# ----------------------------------------------------------------------

async def test_merge_keeps_smaller_story_order():
    llm = FakeLLM(responses=["0"])
    m = mem(llm)

    await m.remember("liubei", "桃园结义", story_order=2000, story_time="建安三年")
    await m.remember("guanyu", "桃园结义", story_order=1000, story_time="初平二年")

    entries = m.all_entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["meta"]["story_order"] == 1000
    assert entry["meta"]["story_time"] == "初平二年"
    assert set(entry["owners"]) == {"liubei", "guanyu"}

    # reverse insertion order (smaller story_order first) also ends at 1000
    llm2 = FakeLLM(responses=["0"])
    m2 = mem(llm2)
    await m2.remember("guanyu", "桃园结义", story_order=1000, story_time="初平二年")
    await m2.remember("liubei", "桃园结义", story_order=2000, story_time="建安三年")

    entries2 = m2.all_entries()
    assert len(entries2) == 1
    entry2 = entries2[0]
    assert entry2["meta"]["story_order"] == 1000
    assert set(entry2["owners"]) == {"liubei", "guanyu"}


# ----------------------------------------------------------------------
# 3. ltm_file: build_society restores a sediment file and skips seeds
# ----------------------------------------------------------------------

SEDIMENT_TEXT = "关羽千里走单骑"


def _write_ltm_scenario(tmp_path, ltm_filename="sediment.ltm.json"):
    scenario_dict = {
        "scenario": "ltm_reuse_test",
        "language": "zh",
        "defaults": {"stats_interval": 100, "distance": 3},
        "agents": [
            {"id": "hall", "kind": "environment", "brain": "rule", "profile": "大厅"},
            {
                "id": "guanyu",
                "kind": "character",
                "brain": "llm",
                "profile": "关羽",
                "status": {"location": "hall"},
                "seed_memories": ["不该出现"],
            },
        ],
        "map": {"default_distance": 3},
        "ltm_file": ltm_filename,
    }
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(yaml.safe_dump(scenario_dict, allow_unicode=True), encoding="utf-8")
    return scenario_path


async def _write_sediment_file(tmp_path, filename="sediment.ltm.json"):
    embedding = (await afake_embed([SEDIMENT_TEXT]))[0]
    sediment = [
        {
            "id": uuid.uuid4().hex,
            "text": SEDIMENT_TEXT,
            "owners": ["guanyu"],
            "meta": {
                "created_at": "2026-07-11T00:00:00+00:00",
                "source": "history",
                "tick": 0,
                "story_order": 1,
                "story_time": "建安五年",
            },
            "embedding": [float(x) for x in embedding],
        }
    ]
    (tmp_path / filename).write_text(json.dumps(sediment, ensure_ascii=False), encoding="utf-8")


async def test_ltm_file_restore_skips_seeds(tmp_path):
    scenario_path = _write_ltm_scenario(tmp_path)
    await _write_sediment_file(tmp_path)

    cfg = load_scenario(str(scenario_path))

    embed = _CountingEmbed()
    llm = FakeLLM()
    k = await build_society(cfg, llm=llm, embed_fn=embed, event_log=EventLog(None))

    entries = k.shared_memory.all_entries()
    assert len(entries) == 1
    assert entries[0]["text"] == SEDIMENT_TEXT
    assert entries[0]["owners"] == ["guanyu"]
    assert entries[0]["meta"]["story_order"] == 1
    assert not any("不该出现" in e["text"] for e in entries)
    assert embed.calls == 0  # sediment entry carried its embedding -> zero recompute


async def test_ltm_file_missing_raises(tmp_path):
    scenario_dict = {
        "scenario": "ltm_missing_test",
        "language": "zh",
        "agents": [
            {"id": "hall", "kind": "environment", "brain": "rule", "profile": "大厅"},
        ],
        "ltm_file": "does_not_exist.ltm.json",
    }
    scenario_path = tmp_path / "scenario.yaml"
    scenario_path.write_text(yaml.safe_dump(scenario_dict, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError):
        load_scenario(str(scenario_path))
