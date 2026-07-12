"""Novel -> scenario extractor (Task 13).

Turns free-form narrative text into a standard AgentSociety scenario YAML
(plus info_carrier corpus files) via a staged LLM pipeline:

    characters -> locations + map -> info_carriers -> seed memories -> kickoff

Each stage is exactly one `llm.chat(bucket="extract")` call per chunk (a
single chunk for short text). Long text is split into overlapping chunks
and each stage's per-chunk results are merged/deduped by id/name.

Every stage's prompt contains a Chinese "marker" word so that test doubles
(and, incidentally, human reviewers) can identify which stage a prompt
belongs to: 角色 (characters), 地点 (locations), 信息载体 (info carriers),
记忆 (memories), 起始 (kickoff).
"""

import argparse
import asyncio
import json
import os

import yaml

from society.llm import LLMClient
from society.scenario import load_scenario

CHUNK_CHARS = 8000
CHUNK_OVERLAP = 500


def _chunk_text(text: str, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split `text` into overlapping chunks.

    Returns a single-element list (the whole text) when it already fits
    in one chunk.
    """
    if len(text) <= chunk_chars:
        return [text]

    chunks = []
    start = 0
    n = len(text)
    step = max(1, chunk_chars - overlap)
    while start < n:
        end = min(start + chunk_chars, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start += step
    return chunks


def _extract_json_block(raw: str):
    """Tolerantly pull a JSON array/object out of `raw` and parse it.

    Tries the whole string first, then falls back to locating the first
    "[...]" or "{...}" span (whichever appears first) and parsing that.
    Returns the parsed object, or raises ValueError if nothing parses.
    """
    raw = raw.strip()
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        pass

    starts = [i for i in (raw.find("["), raw.find("{")) if i != -1]
    if not starts:
        raise ValueError("no JSON block found in response")
    start = min(starts)
    opener = raw[start]
    closer = "]" if opener == "[" else "}"
    end = raw.rfind(closer)
    if end == -1 or end < start:
        raise ValueError("no matching closing bracket found in response")

    candidate = raw[start : end + 1]
    return json.loads(candidate)


_STAGE_PROMPTS = {
    "zh": {
        "characters": (
            "请通读下面的小说片段,提取其中出现的【角色】人物列表。"
            "以 JSON 数组返回,每个元素形如:"
            '{{"id": "拼音或英文短id", "name": "角色姓名", "profile": "一句话人物简介", '
            '"status": {{"location": "该角色当前所在地点的id", "mood": "当前心情"}}, '
            '"goals": ["角色的目标1", "角色的目标2"]}}。'
            "只输出 JSON,不要输出任何解释文字。\n\n提示信息:{hints}\n\n正文:\n{chunk}"
        ),
        "locations": (
            "请通读下面的小说片段,提取其中出现的【地点】以及地点之间的连接关系,"
            "构建一张简单的地图。"
            '以 JSON 对象返回,形如:{{"locations": [{{"id": "地点id", "profile": "地点简介"}}], '
            '"edges": [["地点id1", "地点id2", 距离数字]]}}。'
            "只输出 JSON,不要输出任何解释文字。\n\n提示信息:{hints}\n\n正文:\n{chunk}"
        ),
        "carriers": (
            "请通读下面的小说片段,找出其中出现的【信息载体】"
            "(例如信件、日记、诗稿、题字、书籍等可被阅读的物品)。"
            '以 JSON 数组返回,每个元素形如:{{"id": "短id", "profile": "简介", '
            '"location": "所在位置的id", "excerpt": "该载体内容的原文摘录"}}。'
            "只输出 JSON,不要输出任何解释文字。\n\n提示信息:{hints}\n\n正文:\n{chunk}"
        ),
        "memories": (
            "请通读下面的小说片段,为每个人物提取若干条应作为其【记忆】"
            "种子的关键往事或印象。"
            '以 JSON 对象返回,键为人物id,值为字符串数组,形如:'
            '{{"人物id": ["记忆1", "记忆2"]}}。'
            "已知人物列表:{character_ids}。"
            "只输出 JSON,不要输出任何解释文字。\n\n提示信息:{hints}\n\n正文:\n{chunk}"
        ),
        "kickoff": (
            "请通读下面的小说片段,设计若干条【起始】事件,用于在模拟开始时"
            "推动人物行动(例如一条系统消息告知某人发生了什么)。"
            '以 JSON 数组返回,每个元素形如:{{"to": ["人物id"], "kind": "system", '
            '"content": "事件内容描述"}}。'
            "已知人物列表:{character_ids}。"
            "只输出 JSON,不要输出任何解释文字。\n\n提示信息:{hints}\n\n正文:\n{chunk}"
        ),
    },
}

# English variant keeps the zh marker words in a bilingual header so the
# marker-based routing (used by tests and any downstream tooling) keeps
# working regardless of `language`.
_STAGE_PROMPTS["en"] = {
    stage: "[stage: %s]\n" % {
        "characters": "角色 / characters",
        "locations": "地点 / locations",
        "carriers": "信息载体 / info carriers",
        "memories": "记忆 / memories",
        "kickoff": "起始 / kickoff",
    }[stage]
    + (
        {
            "characters": (
                "Read the novel excerpt below and extract the list of characters. "
                'Return a JSON array, each item shaped like: {{"id": "short id", '
                '"name": "character name", "profile": "one-line profile", '
                '"status": {{"location": "id of where they currently are", "mood": "current mood"}}, '
                '"goals": ["goal 1", "goal 2"]}}. Output JSON only, no explanation.'
            ),
            "locations": (
                "Read the novel excerpt below and extract the locations that appear, "
                "plus the connections between them, forming a simple map. Return a "
                'JSON object shaped like: {{"locations": [{{"id": "location id", '
                '"profile": "short description"}}], "edges": [["loc id 1", "loc id 2", '
                'distance_number]]}}. Output JSON only, no explanation.'
            ),
            "carriers": (
                "Read the novel excerpt below and find any info carriers "
                "(letters, diaries, poems, inscriptions, books, or other readable "
                'items). Return a JSON array, each item shaped like: {{"id": "short id", '
                '"profile": "short description", "location": "location id", '
                '"excerpt": "verbatim excerpt of its content"}}. Output JSON only, no '
                "explanation."
            ),
            "memories": (
                "Read the novel excerpt below and, for each character, extract a "
                "handful of key past events or impressions to seed as their "
                'memories. Return a JSON object keyed by character id, each value a '
                'string array, e.g. {{"character_id": ["memory 1", "memory 2"]}}. '
                "Known characters: {character_ids}. Output JSON only, no explanation."
            ),
            "kickoff": (
                "Read the novel excerpt below and design a handful of kickoff "
                "events to set characters in motion at the start of the simulation "
                "(e.g. a system message telling a character what just happened). "
                'Return a JSON array, each item shaped like: {{"to": ["character id"], '
                '"kind": "system", "content": "description of the event"}}. Known '
                "characters: {character_ids}. Output JSON only, no explanation."
            ),
        }[stage]
    )
    + "\n\nHints: {hints}\n\nText:\n{chunk}"
    for stage in ["characters", "locations", "carriers", "memories", "kickoff"]
}


def _build_prompt(stage: str, language: str, *, chunk: str, hints: str, character_ids=None) -> str:
    templates = _STAGE_PROMPTS.get(language, _STAGE_PROMPTS["zh"])
    template = templates[stage]
    ids_str = ", ".join(character_ids) if character_ids else ""
    return template.format(chunk=chunk, hints=hints or "", character_ids=ids_str)


_STAGE_MARKERS = {
    "characters": "角色",
    "locations": "地点",
    "carriers": "信息载体",
    "memories": "记忆",
    "kickoff": "起始",
}

_STAGE_LABELS = {
    "characters": "characters",
    "locations": "locations",
    "carriers": "carriers",
    "memories": "memories",
    "kickoff": "kickoff",
}


async def _run_stage(
    llm,
    stage: str,
    language: str,
    chunks: list[str],
    hints: str,
    warnings: list[str],
    *,
    character_ids=None,
    empty_value,
):
    """Run one pipeline stage across all chunks; merge, tolerant-parse,
    retry-once-on-failure, else skip with a warning.

    Returns the merged parsed result for this stage (or `empty_value` if
    the stage is skipped entirely).
    """
    per_chunk_results = []

    for chunk in chunks:
        prompt = _build_prompt(stage, language, chunk=chunk, hints=hints, character_ids=character_ids)
        parsed = None
        last_error = None
        for _attempt in range(2):  # first try + one retry
            try:
                raw = await llm.chat(prompt, bucket="extract")
                parsed = _extract_json_block(raw)
                break
            except (ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                continue

        if parsed is None:
            warnings.append(
                f"stage '{_STAGE_LABELS[stage]}' ({_STAGE_MARKERS[stage]}) failed to "
                f"produce valid JSON after retry ({last_error}); chunk skipped"
            )
            continue

        per_chunk_results.append(parsed)

    if not per_chunk_results:
        return empty_value

    return per_chunk_results


def _merge_list_by_id(chunks_results: list[list[dict]]) -> list[dict]:
    merged = {}
    order = []
    for result in chunks_results:
        for item in result:
            key = item.get("id") or item.get("name")
            if key is None:
                continue
            if key not in merged:
                order.append(key)
            merged[key] = {**merged.get(key, {}), **item}
    return [merged[k] for k in order]


def _merge_locations(chunks_results: list[dict]) -> dict:
    locs = {}
    loc_order = []
    edges = []
    seen_edges = set()
    for result in chunks_results:
        for loc in result.get("locations", []):
            lid = loc.get("id")
            if lid is None:
                continue
            if lid not in locs:
                loc_order.append(lid)
            locs[lid] = {**locs.get(lid, {}), **loc}
        for edge in result.get("edges", []):
            key = (edge[0], edge[1])
            if key in seen_edges or (key[1], key[0]) in seen_edges:
                continue
            seen_edges.add(key)
            edges.append(list(edge))
    return {"locations": [locs[k] for k in loc_order], "edges": edges}


def _merge_memories(chunks_results: list[dict]) -> dict:
    merged: dict[str, list[str]] = {}
    for result in chunks_results:
        for char_id, mems in result.items():
            bucket = merged.setdefault(char_id, [])
            for m in mems:
                if m not in bucket:
                    bucket.append(m)
    return merged


def _merge_kickoff(chunks_results: list[list[dict]]) -> list[dict]:
    merged = []
    seen = set()
    for result in chunks_results:
        for item in result:
            key = (tuple(item.get("to", [])), item.get("kind"), item.get("content"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    return merged


async def extract_scenario(
    text: str,
    llm,
    out_yaml: str,
    *,
    max_agents: int = 15,
    language: str = "zh",
    hints: str = "",
) -> dict:
    """Extract a standard scenario YAML from free-form novel text.

    Runs a five-stage LLM pipeline (characters, locations+map, info
    carriers, seed memories, kickoff events), assembles the results into
    the standard scenario format, writes `out_yaml` (and any info_carrier
    corpora next to it, under `corpora/<id>.txt`), validates the result
    with `load_scenario`, and returns the assembled cfg dict (which always
    has a `_warnings` list, possibly empty).

    A LOCATIONS stage that fails entirely (no chunk produces valid JSON)
    is unrecoverable -- character `status.location` references would
    dangle -- so it raises ValueError. Other stage failures degrade
    gracefully: the stage's contribution is simply omitted, and a warning
    is recorded in `cfg["_warnings"]`.
    """
    warnings: list[str] = []
    chunks = _chunk_text(text)

    # Stage 1: characters
    char_chunks = await _run_stage(
        llm, "characters", language, chunks, hints, warnings, empty_value=[]
    )
    characters = _merge_list_by_id(char_chunks) if char_chunks else []
    character_ids = [c["id"] for c in characters if "id" in c]

    # Stage 2: locations + map (unrecoverable failure)
    loc_chunks = await _run_stage(
        llm, "locations", language, chunks, hints, warnings, empty_value=None
    )
    if loc_chunks is None:
        raise ValueError("locations stage failed for all chunks; scenario would be unusable")
    loc_result = _merge_locations(loc_chunks)
    locations = loc_result["locations"]
    edges = loc_result["edges"]

    # Stage 3: info carriers (gracefully skippable)
    carrier_chunks = await _run_stage(
        llm,
        "carriers",
        language,
        chunks,
        hints,
        warnings,
        character_ids=character_ids,
        empty_value=[],
    )
    carriers = _merge_list_by_id(carrier_chunks) if carrier_chunks else []

    # Stage 4: seed memories (gracefully skippable)
    mem_chunks = await _run_stage(
        llm,
        "memories",
        language,
        chunks,
        hints,
        warnings,
        character_ids=character_ids,
        empty_value=None,
    )
    memories = _merge_memories(mem_chunks) if mem_chunks else {}

    # Stage 5: kickoff (gracefully skippable)
    kickoff_chunks = await _run_stage(
        llm,
        "kickoff",
        language,
        chunks,
        hints,
        warnings,
        character_ids=character_ids,
        empty_value=[],
    )
    kickoff = _merge_kickoff(kickoff_chunks) if kickoff_chunks else []

    out_dir = os.path.dirname(os.path.abspath(out_yaml))
    corpora_dir = os.path.join(out_dir, "corpora")
    scenario_name = os.path.splitext(os.path.basename(out_yaml))[0] or "extracted"

    cfg = _assemble_scenario(
        characters=characters,
        locations=locations,
        edges=edges,
        carriers=carriers,
        memories=memories,
        kickoff=kickoff,
        max_agents=max_agents,
        language=language,
        scenario_name=scenario_name,
        warnings=warnings,
    )

    os.makedirs(out_dir, exist_ok=True)
    _write_scenario(cfg, out_yaml, carriers, corpora_dir)

    # Validate loadability; internal inconsistencies were already patched
    # in _assemble_scenario (placeholder locations), so this should pass.
    load_scenario(out_yaml)

    cfg["_warnings"] = warnings
    return cfg


def _id_as_name(agent_id: str) -> dict:
    """Environments/carriers only carry an "id" + "profile" out of the
    extraction stages (no separate "name" field). When the id itself is a
    non-ascii display string (e.g. a Chinese location/carrier name rather
    than a pinyin/ascii short id), reuse it as the "name" too so the
    kernel's alias map (Fix 1a) and view enrichment (Fix 1b) have a name
    to surface -- harmless no-op for ascii ids.

    Returns {"name": agent_id} or {} (spread into a dict literal).
    """
    if not agent_id.isascii():
        return {"name": agent_id}
    return {}


def _assemble_scenario(
    *,
    characters,
    locations,
    edges,
    carriers,
    memories,
    kickoff,
    max_agents,
    language,
    scenario_name,
    warnings,
) -> dict:
    location_ids = {loc["id"] for loc in locations if "id" in loc}

    agents = []

    for c in characters:
        status = dict(c.get("status") or {})
        loc = status.get("location")
        if loc is not None and loc not in location_ids:
            # Extraction inconsistency: patch in a placeholder location so
            # the scenario stays loadable.
            locations.append({"id": loc, "profile": "(自动补全的地点)"})
            location_ids.add(loc)
        agent = {
            "id": c["id"],
            "kind": "character",
            "brain": "llm",
            "profile": c.get("profile", ""),
            "status": status,
            "goals": c.get("goals", []),
            "seed_memories": memories.get(c["id"], []),
        }
        # The characters stage already extracts a display "name" -- carry
        # it into the assembled agent dict so the kernel's name->id alias
        # resolution (Fix 1a) has something to key off of.
        char_name = c.get("name")
        if char_name:
            agent["name"] = char_name
        agents.append(agent)

    location_agents = [
        {
            "id": loc["id"],
            "kind": "environment",
            "brain": "rule",
            "profile": loc.get("profile", ""),
            **_id_as_name(loc["id"]),
        }
        for loc in locations
        if "id" in loc
    ]

    carrier_agents = []
    for carrier in carriers:
        cid = carrier.get("id")
        if cid is None:
            continue
        cloc = carrier.get("location")
        if cloc is not None and cloc not in location_ids:
            locations.append({"id": cloc, "profile": "(自动补全的地点)"})
            location_agents.append(
                {
                    "id": cloc,
                    "kind": "environment",
                    "brain": "rule",
                    "profile": "(自动补全的地点)",
                    **_id_as_name(cloc),
                }
            )
            location_ids.add(cloc)
        carrier_agents.append(
            {
                "id": cid,
                "kind": "info_carrier",
                "brain": "retrieval",
                "profile": carrier.get("profile", ""),
                "status": {"location": cloc} if cloc is not None else {},
                "corpus": f"corpora/{cid}.txt",
                **_id_as_name(cid),
            }
        )

    # Cap total agents at max_agents; characters are prioritized, then
    # locations (needed for any surviving character references), then
    # carriers.
    all_agents = agents + location_agents + carrier_agents
    if len(all_agents) > max_agents:
        kept_ids = set()
        kept = []
        for agent in agents:
            if len(kept) >= max_agents:
                break
            kept.append(agent)
            kept_ids.add(agent["id"])

        # Always keep locations referenced by a kept character/carrier so
        # the scenario stays loadable, even past the raw cap.
        needed_location_ids = {
            a["status"].get("location")
            for a in kept
            if a.get("status", {}).get("location")
        }

        remaining_slots = max(0, max_agents - len(kept))
        kept_locations = []
        for loc_agent in location_agents:
            if loc_agent["id"] in needed_location_ids:
                kept_locations.append(loc_agent)
        for loc_agent in location_agents:
            if loc_agent["id"] in needed_location_ids:
                continue
            if len(kept_locations) >= remaining_slots:
                break
            kept_locations.append(loc_agent)
        kept_location_ids = {a["id"] for a in kept_locations}

        remaining_slots = max(0, max_agents - len(kept) - len(kept_locations))
        kept_carriers = []
        for carrier_agent in carrier_agents:
            cloc = carrier_agent.get("status", {}).get("location")
            if cloc is not None and cloc not in kept_location_ids:
                continue
            if len(kept_carriers) >= remaining_slots:
                break
            kept_carriers.append(carrier_agent)

        all_agents = kept + kept_locations + kept_carriers
    else:
        all_agents = agents + location_agents + carrier_agents

    kept_ids = {a["id"] for a in all_agents}
    filtered_kickoff = [k for k in kickoff if any(t in kept_ids for t in k.get("to", []))]

    filtered_edges = [e for e in edges if e[0] in kept_ids and e[1] in kept_ids]

    cfg = {
        "scenario": scenario_name,
        "language": language,
        "agents": all_agents,
        "map": {"default_distance": 20, "edges": filtered_edges},
        "kickoff": filtered_kickoff,
        "_warnings": warnings,
    }
    return cfg


def _write_scenario(cfg: dict, out_yaml: str, carriers: list[dict], corpora_dir: str) -> None:
    kept_ids = {a["id"] for a in cfg["agents"]}
    os.makedirs(corpora_dir, exist_ok=True)
    for carrier in carriers:
        cid = carrier.get("id")
        if cid is None or cid not in kept_ids:
            continue
        corpus_path = os.path.join(corpora_dir, f"{cid}.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            f.write(carrier.get("excerpt", ""))

    to_dump = {k: v for k, v in cfg.items() if not k.startswith("_")}
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(to_dump, f, allow_unicode=True, sort_keys=False)


def _build_llm(config_path: str | None, *, model_override: str | None = None):
    """Build a real LLMClient from config.json, falling back to the
    OPENAI_API_KEY env var for the API key (mirrors society.run).

    `model_override`, when given, replaces config.json's chat_model for
    this run only (the `--model` CLI flag).
    """
    cfg = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    base_url = cfg.get("base_url", "https://api.openai.com/v1")
    chat_model = model_override or cfg.get("chat_model", "gpt-4o-mini")

    return LLMClient(api_key, base_url, chat_model)


def _build_llm_and_embed(config_path: str | None, *, model_override: str | None = None):
    """Build a real LLMClient + EmbeddingClient.embed from config.json
    (mirrors society.run._build_llm_and_embed), honoring `--model`."""
    from society.embeddings import EmbeddingClient

    cfg = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    base_url = cfg.get("base_url", "https://api.openai.com/v1")
    chat_model = model_override or cfg.get("chat_model", "gpt-4o-mini")
    embed_model = cfg.get("embed_model", "text-embedding-3-small")

    llm = LLMClient(api_key, base_url, chat_model)
    embed_client = EmbeddingClient(api_key, base_url, embed_model)
    return llm, embed_client.embed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract an AgentSociety scenario from a novel")
    parser.add_argument("--input", required=True, help="path to input text file")
    parser.add_argument("--output", required=True, help="path to output scenario yaml")
    parser.add_argument("--max-agents", type=int, default=15, help="max number of agents")
    parser.add_argument("--language", default="zh", help="zh or en")
    parser.add_argument("--hints", default="", help="optional extraction hints")
    parser.add_argument(
        "--config", default="config.json", help="path to config.json (api_key, base_url, ...)"
    )
    parser.add_argument(
        "--mode",
        choices=["snapshot", "history"],
        default="snapshot",
        help="snapshot (existing single-scene pipeline, default) or history "
        "(whole-book sedimentation pipeline, Task H3)",
    )
    parser.add_argument(
        "--model", default=None, help="override config.json's chat_model for this run"
    )
    parser.add_argument(
        "--registry-only",
        action="store_true",
        help="history mode: stop after Pass 1, writing <output>.registry.json for review",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="history mode: path to an existing registry json; skips Pass 1",
    )
    return parser


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    if args.mode == "history":
        from society.history_extract import extract_history

        llm, embed_fn = _build_llm_and_embed(args.config, model_override=args.model)

        registry = None
        if args.registry:
            with open(args.registry, "r", encoding="utf-8") as f:
                registry = json.load(f)

        cfg = asyncio.run(
            extract_history(
                text,
                llm,
                args.output,
                embed_fn=embed_fn,
                hints=args.hints,
                language=args.language,
                registry=registry,
                registry_only=args.registry_only,
                max_agents=args.max_agents,
            )
        )
    else:
        llm = _build_llm(args.config, model_override=args.model)

        cfg = asyncio.run(
            extract_scenario(
                text,
                llm,
                args.output,
                max_agents=args.max_agents,
                language=args.language,
                hints=args.hints,
            )
        )

    print(json.dumps({"_warnings": cfg.get("_warnings", [])}, ensure_ascii=False))
    return cfg


if __name__ == "__main__":
    main()
