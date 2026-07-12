"""Whole-book "history sedimentation" extractor (Task H3).

A second scenario-extraction mode, alongside `society.extract`'s existing
"snapshot" pipeline: instead of extracting a single-moment scene, this mode
treats the whole input text as a **timeline that already happened** and
sediments it into shared long-term memory, then assembles a "sequel starting
point" scenario (see docs/specs/2026-07-11-history-sedimentation-design.md).

Two passes:

  Pass 1 (registry, cheap): per-chunk summaries (marker 摘要) -> one merge
  call (marker 注册表) producing a book-wide registry of canonical
  characters/locations/carriers with alias tables. Written to
  `<out_yaml>.registry.json` so it can be reviewed/edited by a human before
  Pass 2 (closed-world attribution depends on it being accurate).

  Pass 2 (sediment, per chunk, marker 沉淀): extracts atomic, time-prefixed
  memories per canonical id (closed-world: only registry ids are legal) plus
  a state_updates list (location/alive) that is folded into a last-write-wins
  state table across chunks in story order. Each memory is inserted via
  `SharedMemory.remember(..., story_order=..., story_time=...)` so the
  existing normalize-gate + consensus-merge machinery (shared owners,
  smaller-story_order-wins) applies unchanged.

Assembly then emits a standard scenario YAML (alive characters with empty
goal stacks, dead characters with `archived: true`, locations, carriers) plus
a kickoff call (marker 起始) and a holographic `SharedMemory.export()` dump
referenced via `ltm_file` so the (expensive) sedimentation never has to be
redone to run/resume the resulting scenario.
"""

import json
import os

import yaml

from society.extract import CHUNK_CHARS, CHUNK_OVERLAP, _chunk_text, _extract_json_block, _id_as_name
from society.ltm import SharedMemory
from society.scenario import load_scenario


# ----------------------------------------------------------------------
# Prompt builders
# ----------------------------------------------------------------------


def _summary_prompt(chunk: str, hints: str) -> str:
    return (
        "请阅读下面的小说片段,生成一份简短【摘要】,并列出片段中出现的人物与地点。"
        '以 JSON 对象返回,形如:{"summary": "一段话摘要", '
        '"characters": ["人物名1", "人物名2"], "locations": ["地点名1", "地点名2"]}。'
        "只输出 JSON,不要输出任何解释文字。\n\n"
        f"提示信息:{hints or ''}\n\n正文:\n{chunk}"
    )


def _registry_prompt(summaries: list[dict], hints: str) -> str:
    return (
        "以下是一部小说各片段的内容小结与人物/地点列表,请合并去重、识别同一人物的不同"
        "别名,构建全书【注册表】。"
        '以 JSON 对象返回,形如:{"characters": [{"id": "短id(拼音或英文)", '
        '"name": "人物姓名", "aliases": ["别名1", "别名2"], "profile": "一句话简介"}], '
        '"locations": [{"id": "短id", "name": "地点名", "profile": "一句话简介"}], '
        '"carriers": [{"id": "短id", "name": "名称", "profile": "一句话简介"}]}。'
        "carriers 数组若书中未出现信息载体(信件/日记/诗稿等)可为空数组 []。"
        "只输出 JSON,不要输出任何解释文字。\n\n"
        f"提示信息:{hints or ''}\n\n"
        f"各片段小结:\n{json.dumps(summaries, ensure_ascii=False)}"
    )


def _alias_table(registry: dict) -> str:
    lines = []
    for c in registry.get("characters", []) or []:
        cid = c.get("id")
        if cid is None:
            continue
        aliases = c.get("aliases") or []
        alias_str = "、".join(aliases) if aliases else "无"
        lines.append(f"- {cid}({c.get('name', cid)}): 别名 {alias_str}")
    for loc in registry.get("locations", []) or []:
        lid = loc.get("id")
        if lid is None:
            continue
        lines.append(f"- {lid}({loc.get('name', lid)})")
    for car in registry.get("carriers", []) or []:
        cid = car.get("id")
        if cid is None:
            continue
        lines.append(f"- {cid}({car.get('name', cid)})")
    return "\n".join(lines)


def _sediment_prompt(chunk: str, chunk_idx: int, registry: dict, hints: str) -> str:
    alias_table = _alias_table(registry)
    return (
        "请阅读下面的小说片段,将其中每个人物(及相关地点/信息载体)的经历"
        "【沉淀】为若干条带时间前缀的原子记忆。"
        "封闭世界原则:只能把记忆归到下面这份规范 id 清单中列出的 id,禁止创造新 id;"
        "遇到别名时请归到其对应的规范 id。别名对照表如下:\n"
        f"{alias_table}\n\n"
        "每条记忆必须带时间前缀,形如“【建安五年】关羽挂印封金离开曹营”;"
        "若无法判断具体时间,可用“【本回】”等占位。\n"
        '以 JSON 对象返回,形如:{"memories": {"<规范id>": ["【时间】原子记忆1", '
        '"【时间】原子记忆2"]}, "state_updates": [{"id": "规范id", '
        '"location": "地点id或null", "alive": true}], '
        '"story_time": "本片段大致所处的时代(自由文本)"}。'
        "只输出 JSON,不要输出任何解释文字。\n\n"
        f"提示信息:{hints or ''}\n\n正文(第{chunk_idx + 1}块):\n{chunk}"
    )


def _kickoff_prompt(state_summary: str, alive_ids: list[str], hints: str) -> str:
    if hints:
        instruction = f"用户提供的后传前提:{hints}。请据此设计【起始】事件。"
    else:
        instruction = "请依据书末状态自拟后传起点(合理推演故事结束后可能发生的事),设计【起始】事件。"
    candidates = "、".join(alive_ids) if alive_ids else "(无在世角色)"
    return (
        f"{instruction}"
        "起始事件应面向 2-4 位在世主要角色,以 JSON 数组返回,每个元素形如:"
        '{"to": ["人物id"], "kind": "system", "content": "事件内容描述"}。'
        f"在世主要角色候选:{candidates}。"
        "只输出 JSON,不要输出任何解释文字。\n\n"
        f"书末状态:\n{state_summary}"
    )


# ----------------------------------------------------------------------
# Pass 1 -- registry
# ----------------------------------------------------------------------


async def _run_registry_pass(llm, chunks: list[str], hints: str, warnings: list[str]) -> dict:
    summaries = []
    for chunk in chunks:
        prompt = _summary_prompt(chunk, hints)
        raw = await llm.chat(prompt, bucket="extract")
        try:
            parsed = _extract_json_block(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"history summary pass failed for a chunk: {exc}; chunk skipped")
            continue
        summaries.append(parsed)

    if not summaries:
        raise ValueError(
            "history registry pass: no chunk produced a valid summary; cannot build registry"
        )

    merge_prompt = _registry_prompt(summaries, hints)
    raw = await llm.chat(merge_prompt, bucket="extract")
    try:
        registry = _extract_json_block(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"history registry merge pass failed to produce valid JSON: {exc}")

    if not isinstance(registry, dict):
        raise ValueError("history registry merge pass did not return a JSON object")

    registry.setdefault("characters", [])
    registry.setdefault("locations", [])
    registry.setdefault("carriers", [])
    return registry


# ----------------------------------------------------------------------
# Pass 2 -- sedimentation
# ----------------------------------------------------------------------


def _canonical_ids(registry: dict) -> set:
    ids = set()
    for key in ("characters", "locations", "carriers"):
        for item in registry.get(key, []) or []:
            if "id" in item:
                ids.add(item["id"])
    return ids


async def _run_sediment_pass(
    llm,
    shared: SharedMemory,
    chunks: list[str],
    registry: dict,
    hints: str,
    warnings: list[str],
) -> dict:
    """Runs Pass 2 across all chunks in story order.

    Returns the final state table: {id: {"location": str|None, "alive": bool}}.
    """
    canonical_ids = _canonical_ids(registry)
    state: dict[str, dict] = {}

    for chunk_idx, chunk in enumerate(chunks):
        prompt = _sediment_prompt(chunk, chunk_idx, registry, hints)
        raw = await llm.chat(prompt, bucket="extract")
        try:
            parsed = _extract_json_block(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            warnings.append(f"history sediment pass failed for chunk {chunk_idx}: {exc}; chunk skipped")
            continue

        if not isinstance(parsed, dict):
            warnings.append(f"history sediment pass: chunk {chunk_idx} did not return a JSON object; skipped")
            continue

        chunk_story_time = parsed.get("story_time")
        memories = parsed.get("memories") or {}

        mem_counter = 0
        for cid, mems in memories.items():
            if cid not in canonical_ids:
                warnings.append(
                    f"history sediment: chunk {chunk_idx} attributed memories to unknown id "
                    f"{cid!r} (not in registry); skipped"
                )
                continue
            for mem_text in mems:
                await shared.remember(
                    cid,
                    mem_text,
                    source="history",
                    story_order=chunk_idx * 1000 + mem_counter,
                    story_time=chunk_story_time,
                )
                mem_counter += 1

        for update in parsed.get("state_updates") or []:
            uid = update.get("id")
            if uid is None:
                continue
            entry = state.setdefault(uid, {"location": None, "alive": True})
            if update.get("location") is not None:
                entry["location"] = update["location"]
            if update.get("alive") is not None:
                entry["alive"] = update["alive"]

    return state


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------


def _assemble_history_scenario(
    *,
    registry: dict,
    state: dict,
    scenario_name: str,
    language: str,
) -> tuple[dict, list[dict]]:
    """Returns (cfg-without-kickoff-or-ltm_file, carriers) so the caller can
    run the kickoff LLM call (which needs the assembled alive-id list) and
    the sediment export before finishing the cfg dict."""
    locations = [dict(loc) for loc in (registry.get("locations") or []) if "id" in loc]
    location_ids = {loc["id"] for loc in locations}
    fallback_location = next(iter(location_ids), None)

    agents = []
    for c in registry.get("characters", []) or []:
        cid = c.get("id")
        if cid is None:
            continue
        st = state.get(cid, {})
        loc = st.get("location") or fallback_location
        alive = st.get("alive", True)

        if loc is not None and loc not in location_ids:
            locations.append({"id": loc, "name": loc, "profile": "(自动补全的地点)"})
            location_ids.add(loc)

        agent = {
            "id": cid,
            "kind": "character",
            "brain": "llm",
            "profile": c.get("profile", ""),
            "status": {"location": loc} if loc is not None else {},
            "goals": [],
        }
        name = c.get("name")
        if name:
            agent["name"] = name
        if not alive:
            agent["archived"] = True
        agents.append(agent)

    location_agents = []
    for loc in locations:
        loc_agent = {
            "id": loc["id"],
            "kind": "environment",
            "brain": "rule",
            "profile": loc.get("profile", ""),
        }
        name = loc.get("name") or _id_as_name(loc["id"]).get("name")
        if name:
            loc_agent["name"] = name
        location_agents.append(loc_agent)

    carriers = [c for c in (registry.get("carriers") or []) if c.get("id") is not None]
    carrier_agents = []
    for car in carriers:
        cid = car["id"]
        carrier_agent = {
            "id": cid,
            "kind": "info_carrier",
            "brain": "retrieval",
            "profile": car.get("profile", ""),
            "status": {},
            "corpus": f"corpora/{cid}.txt",
        }
        name = car.get("name") or _id_as_name(cid).get("name")
        if name:
            carrier_agent["name"] = name
        carrier_agents.append(carrier_agent)

    all_agents = agents + location_agents + carrier_agents

    cfg = {
        "scenario": scenario_name,
        "language": language,
        "agents": all_agents,
        "map": {"default_distance": 20, "edges": []},
    }
    return cfg, carriers


def _write_corpora(carriers: list[dict], corpora_dir: str) -> None:
    if not carriers:
        return
    os.makedirs(corpora_dir, exist_ok=True)
    for car in carriers:
        cid = car["id"]
        with open(os.path.join(corpora_dir, f"{cid}.txt"), "w", encoding="utf-8") as f:
            f.write(car.get("profile", ""))


# ----------------------------------------------------------------------
# Top-level entry point
# ----------------------------------------------------------------------


async def extract_history(
    text: str,
    llm,
    out_yaml: str,
    *,
    embed_fn=None,
    hints: str = "",
    language: str = "zh",
    registry: dict | None = None,
    registry_only: bool = False,
    chunk_chars: int = CHUNK_CHARS,
    max_agents: int = 15,
) -> dict:
    """Runs the two-pass history-sedimentation pipeline and writes a scenario.

    If `registry` is given (e.g. loaded from a prior `--registry-only` run,
    possibly hand-edited), Pass 1 is skipped entirely. Otherwise Pass 1 runs
    and its result is written to `<out_yaml>.registry.json`.

    If `registry_only` is True, returns right after Pass 1 (writing the
    registry file, if Pass 1 ran) without touching the LLM or SharedMemory
    again -- `{"registry": ..., "_warnings": [...]}`.

    Otherwise runs Pass 2 sedimentation into a fresh `SharedMemory` built
    from `embed_fn` (required in that case), assembles + writes the
    scenario YAML (agents, map, kickoff, `ltm_file`), exports the sediment
    to `<out_yaml>.ltm.json`, validates the result with `load_scenario`, and
    returns the assembled cfg dict (always has a `_warnings` list).

    `max_agents` is accepted for CLI-flag symmetry with the snapshot
    pipeline but is not enforced in this v1 (archived agents must never be
    dropped just to make room, and that policy needs its own design pass --
    see design doc §9).
    """
    warnings: list[str] = []
    # `_chunk_text`'s default overlap (500) is tuned for the default 8000-char
    # chunk size; a much smaller chunk_chars (e.g. tests exercising chunking
    # with a short text) would otherwise make step = chunk_chars - overlap
    # go non-positive and produce a pathological number of near-duplicate
    # chunks. Scale the overlap down to stay well under chunk_chars.
    overlap = min(CHUNK_OVERLAP, max(0, chunk_chars // 4))
    chunks = _chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)

    out_dir = os.path.dirname(os.path.abspath(out_yaml)) or "."
    os.makedirs(out_dir, exist_ok=True)

    ran_pass1 = registry is None
    if registry is None:
        registry = await _run_registry_pass(llm, chunks, hints, warnings)
        registry_path = out_yaml + ".registry.json"
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

    if registry_only:
        return {"registry": registry, "_warnings": warnings}

    if embed_fn is None:
        raise ValueError("extract_history: embed_fn is required for Pass 2 (sedimentation)")

    shared = SharedMemory(embed_fn, llm)
    state = await _run_sediment_pass(llm, shared, chunks, registry, hints, warnings)

    scenario_name = os.path.splitext(os.path.basename(out_yaml))[0] or "extracted"
    cfg, carriers = _assemble_history_scenario(
        registry=registry, state=state, scenario_name=scenario_name, language=language
    )

    alive_ids = [a["id"] for a in cfg["agents"] if a["kind"] == "character" and not a.get("archived")]
    state_lines = []
    for a in cfg["agents"]:
        if a["kind"] != "character":
            continue
        loc = (a.get("status") or {}).get("location")
        status_word = "已故" if a.get("archived") else "在世"
        state_lines.append(f"{a.get('name', a['id'])}({a['id']}): {status_word}, 位置 {loc}")
    state_summary = "\n".join(state_lines)

    kickoff_prompt = _kickoff_prompt(state_summary, alive_ids[:4] or alive_ids, hints)
    raw = await llm.chat(kickoff_prompt, bucket="extract")
    try:
        kickoff = _extract_json_block(raw)
        if not isinstance(kickoff, list):
            raise ValueError("kickoff response is not a JSON array")
    except (ValueError, json.JSONDecodeError) as exc:
        warnings.append(f"history kickoff stage failed: {exc}; no kickoff events generated")
        kickoff = []

    known_ids = {a["id"] for a in cfg["agents"]}
    cfg["kickoff"] = [
        k for k in kickoff if isinstance(k, dict) and any(t in known_ids for t in k.get("to", []))
    ]

    corpora_dir = os.path.join(out_dir, "corpora")
    _write_corpora(carriers, corpora_dir)

    exported = shared.export()
    ltm_path = out_yaml + ".ltm.json"
    with open(ltm_path, "w", encoding="utf-8") as f:
        json.dump(exported, f, ensure_ascii=False)
    cfg["ltm_file"] = os.path.basename(ltm_path)

    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    # Validate loadability (also checks the ltm_file reference resolves).
    load_scenario(out_yaml)

    cfg["_warnings"] = warnings
    cfg["registry"] = registry
    return cfg
