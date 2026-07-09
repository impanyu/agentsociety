import json
import os
import uuid

from society.metrics import Metrics
from society.ltm import SharedMemory
from tests.helpers import FakeLLM, afake_embed


async def test_snapshot_contents(tmp_path):
    shared = SharedMemory(
        afake_embed,
        llm=FakeLLM(responses=["0"]),
        collection_name=f"t_{uuid.uuid4().hex[:8]}",
    )
    await shared.remember("alice", "国王死于春天")
    await shared.remember("bob", "国王死于春天")  # merges → shared entry
    await shared.remember("bob", "厨房进贼")
    m = Metrics({}, shared, str(tmp_path), interval=10)
    m.on_message("a", "b", "say")
    m.on_message("a", "b", "say")
    m.on_message("b", "a", "gesture")
    m.on_message("a", "b", "system")  # ignored
    snap = m.snapshot(10)
    assert snap["consensus_ratio"] == {"total": 2, "shared": 1, "ratio": 0.5}
    assert snap["comm_graph"]["directed"] == {"a->b": 2, "b->a": 1}
    assert snap["comm_graph"]["undirected"] == {"a|b": 3}
    owners = snap["consensus_owners"]
    assert len(owners) == 1 and set(owners[0]["owners"]) == {"alice", "bob"}
    assert os.path.exists(tmp_path / "stats" / "tick_000010.json")
    on_disk = json.load(open(tmp_path / "stats" / "tick_000010.json", encoding="utf-8"))
    assert on_disk["tick"] == 10


async def test_maybe_snapshot_interval(tmp_path):
    m = Metrics({}, None, str(tmp_path), interval=10)
    assert m.maybe_snapshot(0) is None
    assert m.maybe_snapshot(7) is None
    assert m.maybe_snapshot(10) is not None
    # shared_memory None → consensus fields zeroed
    snap = m.snapshot(20)
    assert snap["consensus_ratio"]["total"] == 0 and snap["consensus_owners"] == []
