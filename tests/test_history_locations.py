"""Task H5: location invariants + exhaustive per-character sedimentation.

Covers the deterministic id/name/alias resolver, the 补注册 registry-
augmentation call, assembly's I1-I3 hard invariants (I1/I3 auto-fixed then
sanity-checked; I2 raises), I4's memory-attribution filter, chapter-aware
chunking, and the exhaustive Pass 2 pipeline (出场 roster -> per-character
沉淀 -> 补漏 coverage audit).
"""

import json
import uuid

import pytest

from society.history_extract import (
    _RegistryResolvers,
    _assemble_history_scenario,
    _chunk_history_text,
    _dedupe_role_ids,
    _is_ascii_id,
    _run_sediment_pass,
    extract_history,
)
from society.ltm import SharedMemory
from tests.helpers import FakeLLM, afake_embed

# ----------------------------------------------------------------------
# Resolver: alias -> canonical id
# ----------------------------------------------------------------------


def test_resolver_resolves_id_name_and_alias_for_locations():
    registry = {
        "characters": [],
        "locations": [{"id": "changan", "name": "长安", "aliases": ["西京"]}],
        "carriers": [],
    }
    resolvers = _RegistryResolvers(registry)
    assert resolvers.resolve_location("changan") == "changan"
    assert resolvers.resolve_location("长安") == "changan"
    assert resolvers.resolve_location("西京") == "changan"
    assert resolvers.resolve_location("nonexistent") is None


def test_resolver_classify_distinguishes_kinds():
    registry = {
        "characters": [{"id": "caocao", "name": "曹操", "aliases": ["孟德"]}],
        "locations": [{"id": "xuchang", "name": "许昌", "aliases": []}],
        "carriers": [{"id": "letter1", "name": "一封信", "aliases": []}],
    }
    resolvers = _RegistryResolvers(registry)
    assert resolvers.classify("孟德") == ("caocao", "character")
    assert resolvers.classify("xuchang") == ("xuchang", "location")
    assert resolvers.classify("letter1") == ("letter1", "carrier")
    assert resolvers.classify("ghost") is None


# ----------------------------------------------------------------------
# 补注册 registry augmentation
# ----------------------------------------------------------------------


async def test_registry_augment_resolves_unregistered_location():
    registry = {
        "characters": [{"id": "caocao", "name": "曹操", "aliases": []}],
        "locations": [{"id": "xuchang", "name": "许昌", "aliases": []}],
        "carriers": [],
    }

    def fake(prompt, system=None):
        if "补注册" in prompt:
            return json.dumps(
                {"locations": [{"id": "xuzhou", "name": "徐州", "aliases": [], "profile": "州郡"}]},
                ensure_ascii=False,
            )
        if "沉淀" in prompt:
            return json.dumps(
                {
                    "memories": {"caocao": ["【建安五年】曹操南下"]},
                    "state_updates": [{"id": "caocao", "location": "徐州", "alive": True}],
                    "story_time": "建安五年",
                },
                ensure_ascii=False,
            )
        return "[]"

    llm = FakeLLM(fn=fake)
    shared = SharedMemory(afake_embed, llm, collection_name=f"t_{uuid.uuid4().hex[:8]}")
    warnings = []

    state = await _run_sediment_pass(llm, shared, ["dummy chunk text"], registry, "", warnings)

    assert state["caocao"]["location"] == "xuzhou"
    assert any(loc["id"] == "xuzhou" for loc in registry["locations"])
    assert not any("remained unresolved" in w for w in warnings)


async def test_unresolved_location_dropped_keeps_previous_with_warning():
    registry = {
        "characters": [{"id": "caocao", "name": "曹操", "aliases": []}],
        "locations": [{"id": "xuchang", "name": "许昌", "aliases": []}],
        "carriers": [],
    }

    def fake(prompt, system=None):
        if "补注册" in prompt:
            return "not json at all"  # forces the augment call to fail
        if "沉淀" in prompt:
            if "第1块" in prompt:
                return json.dumps(
                    {
                        "memories": {"caocao": ["【时间】曹操到许昌"]},
                        "state_updates": [{"id": "caocao", "location": "xuchang", "alive": True}],
                        "story_time": "t1",
                    },
                    ensure_ascii=False,
                )
            return json.dumps(
                {
                    "memories": {},
                    "state_updates": [{"id": "caocao", "location": "不存在之地", "alive": True}],
                    "story_time": "t2",
                },
                ensure_ascii=False,
            )
        return "[]"

    llm = FakeLLM(fn=fake)
    shared = SharedMemory(afake_embed, llm, collection_name=f"t_{uuid.uuid4().hex[:8]}")
    warnings = []

    state = await _run_sediment_pass(llm, shared, ["chunk one", "chunk two"], registry, "", warnings)

    # location update dropped -- previous location (xuchang) kept, not a
    # dangling/placeholder value.
    assert state["caocao"]["location"] == "xuchang"
    assert any("remained unresolved after registry augmentation" in w for w in warnings)
    assert not any(loc["id"] == "不存在之地" for loc in registry["locations"])


# ----------------------------------------------------------------------
# _dedupe_role_ids: Pass-1 duplicate-id bug (real example: 红楼's 大观楼 and
# 缀锦阁 both got id "daguanlou"; 三国's has "hulaoguan" x2) -- distinct
# entities sharing an already-ascii id must be disambiguated, while genuine
# same-id-same-name duplicates collapse to one entry, all without ever
# letting `_assemble_history_scenario` see two roles with the same id
# (which would otherwise raise "duplicate agent id").
# ----------------------------------------------------------------------


def test_dedupe_role_ids_renames_distinct_locations_sharing_ascii_id():
    registry = {
        "characters": [],
        "locations": [
            {"id": "daguanlou", "name": "大观楼", "aliases": []},
            {"id": "daguanlou", "name": "缀锦阁", "aliases": []},
        ],
        "carriers": [],
    }
    warnings = []

    _dedupe_role_ids(registry, warnings)

    locs = registry["locations"]
    assert len(locs) == 2
    ids = [loc["id"] for loc in locs]
    assert len(set(ids)) == 2  # globally unique now
    assert ids[0] == "daguanlou"  # first entry keeps the original id
    renamed = locs[1]
    assert renamed["id"] != "daguanlou"
    assert "daguanlou" in renamed["aliases"]  # old id preserved as an alias
    assert any("collided" in w for w in warnings)


def test_dedupe_role_ids_collapses_same_id_same_name_duplicate():
    registry = {
        "characters": [],
        "locations": [
            {"id": "hulaoguan", "name": "虎牢关", "aliases": ["虎牢"]},
            {"id": "hulaoguan", "name": "虎牢关", "aliases": ["汜水关"]},
        ],
        "carriers": [],
    }
    warnings = []

    _dedupe_role_ids(registry, warnings)

    locs = registry["locations"]
    assert len(locs) == 1  # collapsed to a single entry, not two agents
    assert locs[0]["id"] == "hulaoguan"
    assert set(locs[0]["aliases"]) == {"虎牢", "汜水关"}  # aliases merged
    assert any("dropped duplicate" in w for w in warnings)


def test_dedupe_role_ids_disambiguates_character_and_location_sharing_id():
    registry = {
        "characters": [{"id": "lvbu", "name": "吕布", "aliases": []}],
        "locations": [{"id": "lvbu", "name": "吕布祠", "aliases": []}],
        "carriers": [],
    }
    warnings = []

    _dedupe_role_ids(registry, warnings)

    char_id = registry["characters"][0]["id"]
    loc = registry["locations"][0]
    assert char_id == "lvbu"  # characters are processed first, keep priority
    assert loc["id"] != "lvbu"  # location renamed to stay globally unique
    assert "lvbu" in loc["aliases"]
    assert any("collided" in w for w in warnings)


def test_dedupe_role_ids_leaves_clean_registry_unchanged():
    registry = {
        "characters": [{"id": "caocao", "name": "曹操", "aliases": ["孟德"]}],
        "locations": [{"id": "xuchang", "name": "许昌", "aliases": []}],
        "carriers": [{"id": "letter1", "name": "一封信", "aliases": []}],
    }
    warnings = []

    _dedupe_role_ids(registry, warnings)

    assert registry["characters"][0]["id"] == "caocao"
    assert registry["locations"][0]["id"] == "xuchang"
    assert registry["carriers"][0]["id"] == "letter1"
    assert warnings == []


async def test_extract_history_atomic_assembles_despite_duplicate_ascii_location_ids(tmp_path):
    """Regression for the crash this task fixes: a Pass-1 registry with two
    locations sharing the ascii id "daguanlou" used to survive all the way
    to assembly (only non-ascii ids were auto-fixed) and blow up with
    `ValueError: duplicate agent id: daguanlou`."""
    out = str(tmp_path / "honglou.yaml")
    registry = {
        "characters": [{"id": "daiyu", "name": "黛玉", "aliases": []}],
        "locations": [
            {"id": "daguanlou", "name": "大观楼", "aliases": []},
            {"id": "daguanlou", "name": "缀锦阁", "aliases": []},
        ],
        "carriers": [],
    }

    def fake(prompt, system=None):
        if "[atomize]" in prompt:
            return json.dumps(["黛玉在大观楼读书"], ensure_ascii=False)
        if "[assign]" in prompt:
            return json.dumps([["daiyu"]], ensure_ascii=False)
        if "出场" in prompt:
            return json.dumps(
                {
                    "state_updates": [{"id": "daiyu", "location": "大观楼", "alive": True}],
                    "story_time": "t",
                },
                ensure_ascii=False,
            )
        if "起始" in prompt:
            return json.dumps(
                [{"to": ["daiyu"], "kind": "system", "content": "后传伊始"}], ensure_ascii=False
            )
        return "[]"

    llm = FakeLLM(fn=fake)

    # Must not raise ValueError: duplicate agent id.
    cfg = await extract_history(
        "黛玉" * 20, llm, out, embed_fn=afake_embed, registry=registry, detail="atomic"
    )

    agent_ids = [a["id"] for a in cfg["agents"]]
    assert len(agent_ids) == len(set(agent_ids))  # globally unique across all agent kinds

    env_ids = {a["id"] for a in cfg["agents"] if a["kind"] == "environment"}
    assert "daguanlou" in env_ids
    assert len(env_ids) >= 2  # 大观楼 kept id, 缀锦阁 renamed -- neither dropped


# ----------------------------------------------------------------------
# I1: colliding environment names/aliases -> auto-merge
# ----------------------------------------------------------------------


def test_i1_merges_colliding_environment_names_and_remaps_references():
    registry = {
        "characters": [{"id": "a", "name": "甲", "aliases": []}],
        "locations": [
            {"id": "changan", "name": "长安", "aliases": []},
            {"id": "xijing", "name": "西京", "aliases": ["长安"]},  # alias collides with changan
        ],
        "carriers": [],
    }
    state = {"a": {"location": "xijing", "alive": True}}
    warnings = []

    cfg, _carriers = _assemble_history_scenario(
        registry=registry, state=state, scenario_name="s", language="zh", warnings=warnings
    )

    env_agents = [a for a in cfg["agents"] if a["kind"] == "environment"]
    assert len(env_agents) == 1
    assert env_agents[0]["id"] == "changan"

    char_agent = next(a for a in cfg["agents"] if a["id"] == "a")
    assert char_agent["status"]["location"] == "changan"
    assert any("I1" in w for w in warnings)


# ----------------------------------------------------------------------
# I2: character location must reference a defined environment -- raises
# ----------------------------------------------------------------------


def test_i2_raises_on_dangling_character_location():
    registry = {
        "characters": [{"id": "a", "name": "甲", "aliases": []}],
        "locations": [{"id": "changan", "name": "长安", "aliases": []}],
        "carriers": [],
    }
    state = {"a": {"location": "nonexistent_env", "alive": True}}

    with pytest.raises(ValueError, match="I2"):
        _assemble_history_scenario(
            registry=registry, state=state, scenario_name="s", language="zh", warnings=[]
        )


# ----------------------------------------------------------------------
# I3: non-ascii environment id -> auto-slug, Chinese name preserved
# ----------------------------------------------------------------------


def test_i3_slugs_non_ascii_environment_id_and_remaps_references():
    registry = {
        "characters": [{"id": "a", "name": "甲", "aliases": []}],
        "locations": [{"id": "长安", "name": "长安", "aliases": []}],
        "carriers": [],
    }
    state = {"a": {"location": "长安", "alive": True}}
    warnings = []

    cfg, _carriers = _assemble_history_scenario(
        registry=registry, state=state, scenario_name="s", language="zh", warnings=warnings
    )

    env_agents = [a for a in cfg["agents"] if a["kind"] == "environment"]
    assert len(env_agents) == 1
    env = env_agents[0]
    assert _is_ascii_id(env["id"])
    assert env["name"] == "长安"

    char_agent = next(a for a in cfg["agents"] if a["id"] == "a")
    assert char_agent["status"]["location"] == env["id"]
    assert any("I3" in w for w in warnings)


# ----------------------------------------------------------------------
# I4: memory attributed to a known-but-not-character id -> skip + warn
# ----------------------------------------------------------------------


async def test_i4_memory_to_location_id_skipped_with_warning():
    registry = {
        "characters": [{"id": "caocao", "name": "曹操", "aliases": []}],
        "locations": [{"id": "xuchang", "name": "许昌", "aliases": []}],
        "carriers": [],
    }

    def fake(prompt, system=None):
        if "沉淀" in prompt:
            return json.dumps(
                {
                    "memories": {"xuchang": ["【时间】许昌城墙高耸"]},
                    "state_updates": [],
                    "story_time": "t",
                },
                ensure_ascii=False,
            )
        return "[]"

    llm = FakeLLM(fn=fake)
    shared = SharedMemory(afake_embed, llm, collection_name=f"t_{uuid.uuid4().hex[:8]}")
    warnings = []

    await _run_sediment_pass(llm, shared, ["chunk"], registry, "", warnings)

    assert shared.all_entries() == []
    assert any("non-character id" in w for w in warnings)


# ----------------------------------------------------------------------
# Chapter-aware chunking
# ----------------------------------------------------------------------


def test_chapter_chunking_splits_by_huimu_headings():
    text = (
        "第一回　甲出场\n" + ("正文甲" * 30) + "\n"
        "第二回　乙出场\n" + ("正文乙" * 30) + "\n"
        "第三回　丙出场\n" + ("正文丙" * 30) + "\n"
    )

    chunks = _chunk_history_text(text)

    assert len(chunks) == 3
    assert [c["chapter_idx"] for c in chunks] == [0, 1, 2]
    assert [c["seq"] for c in chunks] == [0, 0, 0]
    assert [c["flat_idx"] * 1000 for c in chunks] == [0, 1000, 2000]
    assert chunks[0]["title"].startswith("第一回")
    assert chunks[1]["title"].startswith("第二回")
    assert chunks[2]["title"].startswith("第三回")


def test_chapter_chunking_falls_back_when_no_huimu_headings():
    text = "没有回目标题的一段普通文本。" * 10
    chunks = _chunk_history_text(text)
    assert len(chunks) == 1
    assert chunks[0]["title"] is None


# ----------------------------------------------------------------------
# Exhaustive mode: roster (出场) -> per-character (沉淀) -> audit (补漏)
# ----------------------------------------------------------------------

EXHAUSTIVE_REGISTRY = {
    "characters": [
        {"id": "alice", "name": "甲", "aliases": ["A甲"], "profile": "..."},
        {"id": "bob", "name": "乙", "aliases": ["B乙"], "profile": "..."},
    ],
    "locations": [{"id": "loc1", "name": "集市", "aliases": ["市集"], "profile": "..."}],
    "carriers": [],
}

EXHAUSTIVE_TEXT = (
    "第一回　甲乙初遇\n" + ("甲乙相遇于集市" * 20) + "\n"
    "第二回　甲乙再会\n" + ("甲乙重逢于市集" * 20) + "\n"
)


def _make_exhaustive_fake():
    counts = {"出场": 0, "沉淀": 0, "补漏": 0, "起始": 0}
    per_char_seq = {"alice": 0, "bob": 0}

    def fake(prompt, system=None):
        if "摘要" in prompt or "注册表" in prompt:
            raise AssertionError("pass 1 should be skipped when a registry is supplied")
        if "出场" in prompt:
            counts["出场"] += 1
            return json.dumps(
                {
                    "characters": ["alice", "bob"],
                    "state_updates": [
                        {"id": "alice", "location": "loc1", "alive": True},
                        {"id": "bob", "location": "loc1", "alive": True},
                    ],
                    "story_time": f"chunk{counts['出场']}",
                },
                ensure_ascii=False,
            )
        if "沉淀" in prompt:
            counts["沉淀"] += 1
            if "id=alice" in prompt:
                per_char_seq["alice"] += 1
                return json.dumps([f"【时间】甲第{per_char_seq['alice']}次行动"], ensure_ascii=False)
            per_char_seq["bob"] += 1
            return json.dumps([f"【时间】乙第{per_char_seq['bob']}次行动"], ensure_ascii=False)
        if "补漏" in prompt:
            counts["补漏"] += 1
            return json.dumps({"alice": [f"【时间】甲遗漏的事{counts['补漏']}"]}, ensure_ascii=False)
        if "起始" in prompt:
            counts["起始"] += 1
            return json.dumps(
                [{"to": ["alice"], "kind": "system", "content": "后传伊始"}], ensure_ascii=False
            )
        if "Which existing candidate" in prompt:
            return "-1"  # never merge -- keep counts exact and predictable
        return "[]"

    return fake, counts


async def test_exhaustive_mode_per_character_calls_and_coverage_audit(tmp_path):
    out = str(tmp_path / "novel.yaml")
    fake, counts = _make_exhaustive_fake()
    llm = FakeLLM(fn=fake)

    cfg = await extract_history(
        EXHAUSTIVE_TEXT,
        llm,
        out,
        embed_fn=afake_embed,
        registry=EXHAUSTIVE_REGISTRY,
        detail="exhaustive",
        coverage_rounds=1,
    )

    # 2 chapters -> 2 roster calls; 2 characters appearing in each -> 4
    # per-character 沉淀 calls; 1 coverage round per chunk -> 2 补漏 calls.
    assert counts["出场"] == 2
    assert counts["沉淀"] == 4
    assert counts["补漏"] == 2
    assert counts["起始"] == 1

    ltm_path = out + ".ltm.json"
    with open(ltm_path, "r", encoding="utf-8") as f:
        exported = json.load(f)

    alice_entries = [e for e in exported if e["owners"] == ["alice"]]
    bob_entries = [e for e in exported if e["owners"] == ["bob"]]
    # alice: 1 base fact/chunk * 2 chunks + 1 audit-added fact/chunk * 2 chunks
    assert len(alice_entries) == 4
    # bob: 1 base fact/chunk * 2 chunks, no audit facts attributed to bob
    assert len(bob_entries) == 2
    assert len(exported) == 6

    by_id = {a["id"]: a for a in cfg["agents"]}
    assert by_id["alice"]["status"]["location"] == "loc1"
    assert by_id["bob"]["status"]["location"] == "loc1"


async def test_exhaustive_mode_coverage_rounds_zero_disables_audit(tmp_path):
    out = str(tmp_path / "novel.yaml")
    fake, counts = _make_exhaustive_fake()
    llm = FakeLLM(fn=fake)

    await extract_history(
        EXHAUSTIVE_TEXT,
        llm,
        out,
        embed_fn=afake_embed,
        registry=EXHAUSTIVE_REGISTRY,
        detail="exhaustive",
        coverage_rounds=0,
    )

    assert counts["补漏"] == 0
    assert counts["沉淀"] == 4


# ----------------------------------------------------------------------
# Kickoff targets resolved through the character resolver
# ----------------------------------------------------------------------


async def test_kickoff_targets_resolved_through_character_resolver(tmp_path):
    out = str(tmp_path / "out.yaml")
    registry = {
        "characters": [{"id": "caocao", "name": "曹操", "aliases": ["孟德"]}],
        "locations": [{"id": "xuchang", "name": "许昌", "aliases": []}],
        "carriers": [],
    }

    def fake(prompt, system=None):
        if "沉淀" in prompt:
            return json.dumps(
                {
                    "memories": {"caocao": ["【时间】曹操驻守许昌"]},
                    "state_updates": [{"id": "caocao", "location": "xuchang", "alive": True}],
                    "story_time": "t",
                },
                ensure_ascii=False,
            )
        if "起始" in prompt:
            # LLM used the alias instead of the canonical id.
            return json.dumps(
                [{"to": ["孟德"], "kind": "system", "content": "后传伊始"}], ensure_ascii=False
            )
        return "[]"

    llm = FakeLLM(fn=fake)
    cfg = await extract_history(
        "曹操" * 10, llm, out, embed_fn=afake_embed, registry=registry, detail="fast"
    )

    assert cfg["kickoff"] == [{"to": ["caocao"], "kind": "system", "content": "后传伊始"}]
