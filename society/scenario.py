import os
import uuid

import yaml

from society.actions import Message
from society.agent import Agent
from society.brains import LLMBrain, RetrievalBrain, RuleBrain
from society.kernel import Kernel
from society.ltm import SharedMemory
from society.metrics import Metrics
from society.stm import STM
from society.worldmap import WorldMap

_BRAIN_KINDS = {"llm", "rule", "retrieval"}


def load_scenario(path: str) -> dict:
    """Load and validate a scenario YAML file.

    Raises ValueError on:
      - an agent missing "id" or "kind"
      - duplicate agent ids
      - an unknown "brain" value
      - a "private_status_keys" field that isn't a list of strings
      - a character/info_carrier whose initial status.location does not
        reference an environment agent defined in the file
      - a map edge endpoint that isn't a defined environment id

    Stores the scenario file's directory under cfg["_dir"] so that
    build_society can resolve relative paths (e.g. info_carrier corpus
    files) against the scenario file rather than the process cwd.
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    agents = cfg.get("agents", [])
    seen_ids = set()
    env_ids = set()

    for a in agents:
        if "id" not in a:
            raise ValueError("agent missing required 'id' field")
        if "kind" not in a:
            raise ValueError(f"agent {a['id']!r} missing required 'kind' field")

        aid = a["id"]
        if aid in seen_ids:
            raise ValueError(f"duplicate agent id: {aid}")
        seen_ids.add(aid)

        brain = a.get("brain")
        if brain not in _BRAIN_KINDS:
            raise ValueError(f"agent {aid!r}: unknown brain {brain!r}")

        private_keys = a.get("private_status_keys")
        if private_keys is not None:
            if not isinstance(private_keys, list) or not all(
                isinstance(k, str) for k in private_keys
            ):
                raise ValueError(
                    f"agent {aid!r}: private_status_keys must be a list of strings"
                )

        if a["kind"] == "environment":
            env_ids.add(aid)

    for a in agents:
        if a["kind"] in ("character", "info_carrier"):
            loc = (a.get("status") or {}).get("location")
            if loc is not None and loc not in env_ids:
                raise ValueError(
                    f"agent {a['id']!r}: initial location {loc!r} is not an "
                    "environment agent defined in this scenario"
                )

    map_cfg = cfg.get("map") or {}
    for edge in map_cfg.get("edges", []):
        a_id, b_id = edge[0], edge[1]
        if a_id not in env_ids or b_id not in env_ids:
            raise ValueError(
                f"map edge {edge!r} references an id that isn't an environment agent"
            )

    cfg["_dir"] = os.path.dirname(os.path.abspath(path))
    return cfg


def _make_brain(a: dict, *, llm, language: str, scenario_dir: str):
    kind_name = a["brain"]
    if kind_name == "rule":
        return RuleBrain()
    if kind_name == "llm":
        return LLMBrain(llm, profile=a.get("profile", ""), language=language)
    # retrieval
    corpus_text = ""
    corpus_path = a.get("corpus")
    if corpus_path:
        full_path = os.path.join(scenario_dir, corpus_path)
        with open(full_path, "r", encoding="utf-8") as f:
            corpus_text = f.read()
    return RetrievalBrain(corpus_text)


async def build_society(
    cfg: dict,
    *,
    llm,
    embed_fn,
    event_log,
    out_dir=None,
    metrics_interval=None,
) -> Kernel:
    """Build a fully-wired Kernel from a loaded scenario dict.

    Creates all agents (brain selected per agent's "brain" field), a
    WorldMap from cfg["map"], a SharedMemory seeded with each agent's
    seed_memories, and a Metrics instance. Kickoff messages are queued
    with tick_sent=-1 before the kernel is returned so they are already
    pending on kernel.send() (visible to recipients once the caller calls
    kernel.run(), per the tick-0 delivery semantics in Kernel.run()).
    """
    defaults = cfg.get("defaults", {}) or {}
    language = cfg.get("language", "zh")
    scenario_dir = cfg.get("_dir", ".")

    fifo_size = defaults.get("fifo_size", 20)
    memory_max_chars = defaults.get("memory_max_chars", 80)
    distance = defaults.get("distance", 20)
    stats_interval = defaults.get("stats_interval", 10)

    agents: dict[str, Agent] = {}
    env_ids = []
    seed_specs = []  # (agent_id, [texts])

    for a in cfg.get("agents", []):
        brain = _make_brain(a, llm=llm, language=language, scenario_dir=scenario_dir)
        stm = STM(
            fifo_size=fifo_size,
            status=a.get("status"),
            private_keys=set(a["private_status_keys"]) if a.get("private_status_keys") else None,
            goals=a.get("goals"),
        )
        agent = Agent(
            a["id"],
            a["kind"],
            brain,
            stm,
            portable=a.get("portable", False),
            holder=a.get("holder"),
            profile=a.get("profile", ""),
        )
        agents[a["id"]] = agent
        if a["kind"] == "environment":
            env_ids.append(a["id"])
        if a.get("seed_memories"):
            seed_specs.append((a["id"], a["seed_memories"]))

    map_cfg = cfg.get("map") or {}
    edges = [tuple(e) for e in map_cfg.get("edges", [])]
    worldmap = WorldMap(
        env_ids,
        edges=edges,
        default_distance=map_cfg.get("default_distance", distance),
    )

    shared = SharedMemory(embed_fn, llm, max_chars=memory_max_chars)

    interval = metrics_interval if metrics_interval is not None else stats_interval
    metrics = Metrics(agents, shared, out_dir, interval=interval)

    kernel = Kernel(agents, worldmap, event_log, shared_memory=shared, llm=llm, metrics=metrics)

    for agent_id, texts in seed_specs:
        for text in texts:
            await shared.remember(agent_id, text, tick=0, source="scenario_seed")

    for kick in cfg.get("kickoff", []):
        msg = Message(
            id=str(uuid.uuid4()),
            sender="scenario",
            recipients=list(kick["to"]),
            kind=kick["kind"],
            content=kick["content"],
            tick_sent=-1,
        )
        kernel.send(msg)

    return kernel
