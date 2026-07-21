"""Checkpoint persistence: holographic snapshot + restore of a running Kernel.

A checkpoint captures everything needed to resume a run bit-for-bit
(scenario config, tick, event sequence counter, every agent's STM
runtime state, presence, in-flight pending messages, the full LTM
including embeddings, and metrics' directed communication counts) so
`restore_society` can rebuild a Kernel that continues exactly where
`save_checkpoint` left off -- seeds and kickoff messages are NOT
replayed, since the checkpoint already reflects their effects.
"""

import json
import os

from society.actions import Message
from society.kernel import Kernel
from society.ltm import SharedMemory
from society.metrics import Metrics
from society.scenario import build_agents_and_map

CHECKPOINT_VERSION = 1


def _agent_state(agent) -> dict:
    return {
        "fifo": [[action, result] for action, result in agent.stm.fifo.items()],
        "goals": agent.stm.goals.items(),
        "status": agent.stm.status.all(),
        "private_keys": sorted(agent.stm.status._private_keys),
        "inbox": [m.to_dict() for m in agent.stm.inbox_items()],
        "waiting_until": agent.waiting_until,
        "transit": agent.transit,
        "archived": agent.archived,
    }


def _build_checkpoint_dict(kernel) -> dict:
    agents_state = {aid: _agent_state(a) for aid, a in kernel.agents.items()}
    presence = {loc: sorted(ids) for loc, ids in kernel.presence.items()}
    pending = [m.to_dict() for m in kernel._pending]
    ltm = kernel.shared_memory.export() if kernel.shared_memory is not None else []

    metrics_directed = {}
    if kernel.metrics is not None:
        directed_edges = getattr(kernel.metrics, "_directed_edges", None)
        if directed_edges is not None:
            metrics_directed = dict(directed_edges)

    return {
        "version": CHECKPOINT_VERSION,
        "tick": kernel.tick,
        "event_seq": kernel.event_log._seq_counter,
        "scenario": kernel.scenario_cfg,
        "agents": agents_state,
        "presence": presence,
        "pending": pending,
        "ltm": ltm,
        "metrics_directed": metrics_directed,
    }


def save_checkpoint(kernel, path: str) -> None:
    """Atomically write `kernel`'s full state as JSON to `path`.

    Writes to `path + ".tmp"` first, then `os.replace`s it into place, so a
    crash mid-write never leaves a corrupt/partial checkpoint at `path`.
    """
    ckpt = _build_checkpoint_dict(kernel)

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(ckpt, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_checkpoint(path: str) -> dict:
    """Load a checkpoint JSON file written by `save_checkpoint`."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def restore_society(ckpt: dict, *, llm, embed_fn, event_log, out_dir=None) -> Kernel:
    """Rebuild a Kernel from a checkpoint dict and overwrite its runtime
    state to match the checkpoint exactly.

    Agents/brains/map are (re)built via `build_agents_and_map` -- the same
    helper `build_society` uses for a fresh run -- but seed memories and
    kickoff messages are deliberately NOT replayed (the checkpoint's LTM
    export and per-agent inbox/pending state already reflect their
    effects; replaying them would duplicate work).
    """
    cfg = ckpt["scenario"]
    agents, worldmap, defaults, _seed_specs = build_agents_and_map(cfg, llm=llm, embed_fn=embed_fn)

    memory_max_tokens = defaults.get(
        "memory_max_tokens", defaults.get("memory_max_chars", 50)
    )
    stats_interval = defaults.get("stats_interval", 10)

    shared = SharedMemory(embed_fn, llm, max_tokens=memory_max_tokens)
    metrics = Metrics(agents, shared, out_dir, interval=stats_interval)

    kernel = Kernel(
        agents,
        worldmap,
        event_log,
        shared_memory=shared,
        llm=llm,
        metrics=metrics,
        config={"language": cfg.get("language", "zh")},
    )
    kernel.scenario_cfg = cfg
    kernel.tick = ckpt["tick"]

    for aid, state in ckpt.get("agents", {}).items():
        agent = agents.get(aid)
        if agent is None:
            continue

        agent.stm.fifo.restore_items(
            [(action, result) for action, result in state.get("fifo", [])]
        )

        # build_agents_and_map already seeded goals/status from the
        # scenario's initial config (same as a fresh build_society run) --
        # replace that seed wholesale with the checkpoint's snapshot rather
        # than merging/pushing on top of it.
        agent.stm.goals._stack = list(state.get("goals", []))

        agent.stm.status._data = dict(state.get("status") or {})
        agent.stm.status._private_keys = set(state.get("private_keys", []))

        # archived is already set from the scenario config by
        # build_agents_and_map above; the checkpoint value is authoritative
        # in case it was ever toggled at runtime, so re-apply it explicitly
        # (a no-op in the common case).
        agent.archived = state.get("archived", agent.archived)

        for msg_dict in state.get("inbox", []):
            agent.stm.inbox.put_nowait(Message(**msg_dict))

        agent.waiting_until = state.get("waiting_until")
        agent.transit = state.get("transit")

    kernel.presence = {
        loc: set(ids) for loc, ids in ckpt.get("presence", {}).items()
    }

    kernel._pending = [Message(**d) for d in ckpt.get("pending", [])]

    await shared.restore(ckpt.get("ltm", []))

    directed = ckpt.get("metrics_directed") or {}
    for edge, count in directed.items():
        metrics._directed_edges[edge] = count

    return kernel
