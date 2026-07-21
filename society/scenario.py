import json
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
      - a top-level "ltm_file" that doesn't exist relative to the
        scenario file's directory

    Stores the scenario file's directory under cfg["_dir"] so that
    build_society can resolve relative paths (e.g. info_carrier corpus
    files, ltm_file) against the scenario file rather than the process cwd.
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

        name = a.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError(f"agent {aid!r}: name must be a string")

        archived = a.get("archived")
        if archived is not None and not isinstance(archived, bool):
            raise ValueError(f"agent {aid!r}: archived must be a bool")

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

    ltm_file = cfg.get("ltm_file")
    if ltm_file is not None:
        full_ltm_path = os.path.join(cfg["_dir"], ltm_file)
        if not os.path.isfile(full_ltm_path):
            raise ValueError(f"ltm_file not found: {full_ltm_path!r}")

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


def build_agents_and_map(cfg: dict, *, llm, embed_fn=None) -> tuple[dict, "WorldMap", dict, list]:
    """Build the agents dict + WorldMap from a loaded scenario dict.

    This is the brain/agent-construction core shared by `build_society`
    (fresh run) and `society.persistence.restore_society` (resume from a
    checkpoint) -- kept in exactly one place so the two paths can never
    drift apart on how a scenario's agents/brains/map are constructed.

    `embed_fn` (the same async embedding function used for SharedMemory)
    is threaded into each agent's STM as `cache_embed_fn`, so the STM
    cache's "relevance"/"hybrid" eviction strategies (config knobs
    `cache_strategy`/`cache_alpha` in `defaults`) have an embedder to
    score with. It's optional (None) since it's only required when a
    non-"fifo" cache_strategy is configured.

    Returns (agents, worldmap, defaults, seed_specs) where `defaults` is
    the resolved `cfg["defaults"]` dict and `seed_specs` is the
    `[(agent_id, [seed_memory_texts]), ...]` list -- callers that must NOT
    replay seed memories (i.e. resume) simply ignore `seed_specs`.
    """
    defaults = cfg.get("defaults", {}) or {}
    language = cfg.get("language", "zh")
    scenario_dir = cfg.get("_dir", ".")

    fifo_size = defaults.get("fifo_size", 20)
    distance = defaults.get("distance", 20)
    cache_strategy = defaults.get("cache_strategy", "fifo")
    cache_alpha = defaults.get("cache_alpha", 0.5)

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
            cache_strategy=cache_strategy,
            cache_alpha=cache_alpha,
            cache_embed_fn=embed_fn,
        )
        agent = Agent(
            a["id"],
            a["kind"],
            brain,
            stm,
            portable=a.get("portable", False),
            holder=a.get("holder"),
            profile=a.get("profile", ""),
            name=a.get("name"),
            archived=a.get("archived", False),
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

    return agents, worldmap, defaults, seed_specs


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
    WorldMap from cfg["map"], a SharedMemory, and a Metrics instance.
    Kickoff messages are queued with tick_sent=-1 before the kernel is
    returned so they are already pending on kernel.send() (visible to
    recipients once the caller calls kernel.run(), per the tick-0 delivery
    semantics in Kernel.run()).

    If cfg has a top-level "ltm_file" (path relative to the scenario
    file's directory, validated to exist by load_scenario), the
    SharedMemory is populated via `shared.restore()` from that holographic
    dump instead -- seed_memories are NOT replayed in that case, since the
    ltm_file already reflects (a superset of) their effect. Otherwise each
    agent's seed_memories are seeded as before.
    """
    agents, worldmap, defaults, seed_specs = build_agents_and_map(cfg, llm=llm, embed_fn=embed_fn)

    memory_max_tokens = defaults.get(
        "memory_max_tokens", defaults.get("memory_max_chars", 50)
    )
    stats_interval = defaults.get("stats_interval", 10)

    shared = SharedMemory(embed_fn, llm, max_tokens=memory_max_tokens)

    interval = metrics_interval if metrics_interval is not None else stats_interval
    metrics = Metrics(agents, shared, out_dir, interval=interval)

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

    ltm_file = cfg.get("ltm_file")
    if ltm_file is not None:
        ltm_path = os.path.join(cfg.get("_dir", "."), ltm_file)
        with open(ltm_path, "r", encoding="utf-8") as f:
            await shared.restore(json.load(f))
    else:
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
