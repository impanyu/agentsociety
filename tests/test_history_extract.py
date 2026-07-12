import json
import uuid

from society.events import EventLog
from society.extract import _build_parser
from society.history_extract import extract_history
from society.ltm import SharedMemory
from society.scenario import build_society, load_scenario
from tests.helpers import FakeLLM, afake_embed

# A ~350-char text with no real semantic content -- the FakeLLM routes purely
# on the marker word present in each stage's prompt, so the chunk *content*
# doesn't matter, only its length (chunk_chars=200 below forces exactly 2
# chunks out of it).
TEXT = ("曹操" * 90) + ("关羽" * 85)
assert 300 < len(TEXT) < 400

REGISTRY = {
    "characters": [
        {"id": "caocao", "name": "曹操", "aliases": ["孟德", "丞相"], "profile": "魏武帝"},
        {"id": "guanyu", "name": "关羽", "aliases": ["云长"], "profile": "蜀汉名将"},
    ],
    "locations": [
        {"id": "xuchang", "name": "许昌", "profile": "曹操治下都城"},
    ],
    "carriers": [],
}

SUMMARY_RESPONSE = json.dumps(
    {"summary": "曹操与关羽的一段往事", "characters": ["曹操", "关羽"], "locations": ["许昌"]},
    ensure_ascii=False,
)

REGISTRY_RESPONSE = json.dumps(REGISTRY, ensure_ascii=False)


def _sediment_response(chunk_no: int, extra_memories=None, extra_updates=None, story_time=None):
    memories = {"caocao": ["【建安五年】曹操驻守许昌"], "guanyu": ["【建安五年】关羽护送嫂嫂北上"]}
    if extra_memories:
        memories.update(extra_memories)
    state_updates = [
        {"id": "caocao", "location": "xuchang", "alive": True},
        {"id": "guanyu", "location": "xuchang", "alive": True},
    ]
    if extra_updates:
        state_updates = extra_updates
    return json.dumps(
        {
            "memories": memories,
            "state_updates": state_updates,
            "story_time": story_time or f"建安{chunk_no}年",
        },
        ensure_ascii=False,
    )


KICKOFF_RESPONSE = json.dumps(
    [{"to": ["guanyu"], "kind": "system", "content": "后传伊始，风声再起"}], ensure_ascii=False
)


def make_fake(sediment_fn=None):
    """sediment_fn(chunk_idx:int, prompt:str) -> response str; defaults to
    _sediment_response(chunk_idx)."""
    call_counter = {"沉淀": 0}

    def fake(prompt, system=None):
        if "摘要" in prompt:
            return SUMMARY_RESPONSE
        if "注册表" in prompt:
            return REGISTRY_RESPONSE
        if "沉淀" in prompt:
            idx = call_counter["沉淀"]
            call_counter["沉淀"] += 1
            if sediment_fn is not None:
                return sediment_fn(idx, prompt)
            return _sediment_response(idx)
        if "起始" in prompt:
            return KICKOFF_RESPONSE
        # SharedMemory's internal "consensus"/"normalize" calls don't carry
        # any of the extractor's markers; equivalence checks always match
        # candidate 0 (used by the consensus-merge test).
        if "Which existing candidate" in prompt:
            return "0"
        return "[]"

    return fake


# ----------------------------------------------------------------------
# 1. registry pass merges chunk summaries and writes aliases; registry_only
# ----------------------------------------------------------------------


async def test_registry_pass_merges_and_writes(tmp_path):
    out = str(tmp_path / "sanguo.yaml")
    llm = FakeLLM(fn=make_fake())

    result = await extract_history(
        TEXT, llm, out, embed_fn=afake_embed, chunk_chars=200, registry_only=True
    )

    assert result["registry"]["characters"][0]["id"] == "caocao"
    assert "孟德" in result["registry"]["characters"][0]["aliases"]
    assert result["registry"]["carriers"] == []

    registry_path = out + ".registry.json"
    with open(registry_path, "r", encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk == result["registry"]

    # registry_only stops before Pass 2 -- no ltm.json / scenario yaml yet.
    import os

    assert not os.path.exists(out)
    assert not os.path.exists(out + ".ltm.json")

    # Two chunks -> two 摘要 calls + one 注册表 merge call.
    markers = ["摘要" if "摘要" in p else ("注册表" if "注册表" in p else "?") for _, p, _ in llm.calls]
    assert markers.count("摘要") == 2
    assert markers.count("注册表") == 1


# ----------------------------------------------------------------------
# 2. closed-world attribution + story_order + consensus merge
# ----------------------------------------------------------------------


async def test_sediment_closed_world_and_story_order(tmp_path):
    out = str(tmp_path / "sanguo.yaml")

    def sediment_fn(idx, prompt):
        if idx == 0:
            # chunk 0: two memories for caocao (story_order 0,1), one for an
            # unknown id "luxun" (not in registry -> must be skipped+warned).
            return json.dumps(
                {
                    "memories": {
                        "caocao": ["【建安五年】曹操驻守许昌", "【建安五年】曹操大宴群臣"],
                        "luxun": ["【建安五年】鲁迅路过许昌"],
                    },
                    "state_updates": [{"id": "caocao", "location": "xuchang", "alive": True}],
                    "story_time": "建安五年",
                },
                ensure_ascii=False,
            )
        # chunk 1: one memory for guanyu (story_order 1000).
        return json.dumps(
            {
                "memories": {"guanyu": ["【建安五年】关羽护送嫂嫂北上"]},
                "state_updates": [{"id": "guanyu", "location": "xuchang", "alive": True}],
                "story_time": "建安五年",
            },
            ensure_ascii=False,
        )

    llm = FakeLLM(fn=make_fake(sediment_fn))
    shared = SharedMemory(afake_embed, llm, collection_name=f"t_{uuid.uuid4().hex[:8]}")

    from society.history_extract import _run_sediment_pass, _chunk_text

    chunks = _chunk_text(TEXT, chunk_chars=200, overlap=50)
    assert len(chunks) == 2

    warnings = []
    state = await _run_sediment_pass(llm, shared, chunks, REGISTRY, "", warnings)

    assert any("luxun" in w for w in warnings)
    entries = shared.all_entries()
    assert len(entries) == 3  # 2 caocao memories + 1 guanyu memory, luxun's skipped
    caocao_orders = sorted(
        e["meta"]["story_order"] for e in shared.all_entries() if "caocao" in e["owners"]
    )
    assert caocao_orders == [0, 1]
    guanyu_orders = [
        e["meta"]["story_order"] for e in shared.all_entries() if "guanyu" in e["owners"]
    ]
    assert guanyu_orders == [1000]
    assert state["caocao"]["location"] == "xuchang" and state["caocao"]["alive"] is True
    assert state["guanyu"]["location"] == "xuchang" and state["guanyu"]["alive"] is True


async def test_sediment_consensus_merges_across_chunks(tmp_path):
    """Two chunks both report the same fact for caocao -> equivalence call
    (scripted to answer "0") merges them into a single shared entry whose
    story_order is the smaller of the two."""
    SAME_TEXT = "【建安五年】曹操与关羽结为兄弟"

    def sediment_fn(idx, prompt):
        return json.dumps(
            {
                "memories": {"caocao": [SAME_TEXT]},
                "state_updates": [{"id": "caocao", "location": "xuchang", "alive": True}],
                "story_time": "建安五年",
            },
            ensure_ascii=False,
        )

    llm = FakeLLM(fn=make_fake(sediment_fn))
    shared = SharedMemory(afake_embed, llm, collection_name=f"t_{uuid.uuid4().hex[:8]}")

    from society.history_extract import _run_sediment_pass, _chunk_text

    chunks = _chunk_text(TEXT, chunk_chars=200, overlap=50)
    assert len(chunks) == 2

    warnings = []
    await _run_sediment_pass(llm, shared, chunks, REGISTRY, "", warnings)

    entries = shared.all_entries()
    assert len(entries) == 1
    assert entries[0]["meta"]["story_order"] == 0  # chunk0's story_order (smaller) wins
    assert set(entries[0]["owners"]) == {"caocao"}


# ----------------------------------------------------------------------
# 3. assembly: alive/dead, goals=[], ltm_file, load_scenario, build_society
# ----------------------------------------------------------------------


async def test_assembly_alive_dead_and_ltm_file(tmp_path):
    out = str(tmp_path / "sanguo.yaml")

    def sediment_fn(idx, prompt):
        if idx == 0:
            return _sediment_response(0)
        # Final chunk marks caocao dead.
        return json.dumps(
            {
                "memories": {"guanyu": ["【建安六年】关羽再战"]},
                "state_updates": [
                    {"id": "caocao", "location": "xuchang", "alive": False},
                    {"id": "guanyu", "location": "xuchang", "alive": True},
                ],
                "story_time": "建安六年",
            },
            ensure_ascii=False,
        )

    llm = FakeLLM(fn=make_fake(sediment_fn))

    cfg = await extract_history(TEXT, llm, out, embed_fn=afake_embed, chunk_chars=200)

    import os

    assert os.path.exists(out)
    ltm_path = out + ".ltm.json"
    assert os.path.exists(ltm_path)

    loaded = load_scenario(out)
    assert loaded["ltm_file"] == os.path.basename(ltm_path)

    by_id = {a["id"]: a for a in loaded["agents"]}
    assert by_id["caocao"]["archived"] is True
    assert by_id["guanyu"].get("archived", False) is False
    for aid in ("caocao", "guanyu"):
        assert by_id[aid]["goals"] == []
        assert by_id[aid]["kind"] == "character"
        assert by_id[aid]["brain"] == "llm"
    assert by_id["xuchang"]["kind"] == "environment"

    assert cfg["_warnings"] == [] or isinstance(cfg["_warnings"], list)

    # build_society restores the sediment (no seed replay) and skips seeds.
    k = await build_society(
        loaded, llm=FakeLLM(fn=make_fake()), embed_fn=afake_embed, event_log=EventLog(None)
    )
    entries = k.shared_memory.all_entries()
    with open(ltm_path, "r", encoding="utf-8") as f:
        exported = json.load(f)
    assert len(entries) == len(exported)
    assert len(entries) > 0


# ----------------------------------------------------------------------
# 4. --registry reuse skips Pass 1
# ----------------------------------------------------------------------


async def test_registry_reuse_skips_pass1(tmp_path):
    out = str(tmp_path / "sanguo.yaml")
    llm = FakeLLM(fn=make_fake())

    cfg = await extract_history(
        TEXT, llm, out, embed_fn=afake_embed, chunk_chars=200, registry=REGISTRY
    )

    markers_hit = {"摘要": False, "注册表": False}
    for _, prompt, _ in llm.calls:
        if "摘要" in prompt:
            markers_hit["摘要"] = True
        if "注册表" in prompt:
            markers_hit["注册表"] = True
    assert markers_hit == {"摘要": False, "注册表": False}
    assert cfg["registry"] == REGISTRY  # reused verbatim, not re-derived

    import os

    # No registry file is (re)written when an existing registry is reused.
    assert not os.path.exists(out + ".registry.json")


# ----------------------------------------------------------------------
# 5. CLI parses --mode history --model x --registry-only (no network)
# ----------------------------------------------------------------------


def test_cli_parses_history_mode_flags():
    parser = _build_parser()
    args = parser.parse_args(
        [
            "--input",
            "book.txt",
            "--output",
            "out.yaml",
            "--mode",
            "history",
            "--model",
            "gpt-4o",
            "--registry-only",
            "--registry",
            "reg.json",
        ]
    )
    assert args.mode == "history"
    assert args.model == "gpt-4o"
    assert args.registry_only is True
    assert args.registry == "reg.json"

    # Defaults for snapshot-mode invocation are unaffected.
    args2 = parser.parse_args(["--input", "book.txt", "--output", "out.yaml"])
    assert args2.mode == "snapshot"
    assert args2.model is None
    assert args2.registry_only is False
    assert args2.registry is None
