"""Whole-book "history sedimentation" extractor (Task H3, hardened in H5).

A second scenario-extraction mode, alongside `society.extract`'s existing
"snapshot" pipeline: instead of extracting a single-moment scene, this mode
treats the whole input text as a **timeline that already happened** and
sediments it into shared long-term memory, then assembles a "sequel starting
point" scenario (see docs/specs/2026-07-11-history-sedimentation-design.md).

Two passes:

  Pass 1 (registry, cheap): per-chunk summaries (marker 摘要) -> one merge
  call (marker 注册表) producing a book-wide registry of canonical
  characters/locations/carriers with alias tables (characters AND locations
  both carry "aliases": [...]). Written to `<out_yaml>.registry.json` so it
  can be reviewed/edited by a human before Pass 2 (closed-world attribution
  depends on it being accurate).

  Pass 2 (sediment): extracts atomic memories owned by one or more
  canonical role ids (characters/locations/carriers, plus a reserved
  `narrator` catch-all) plus a state_updates list (location/alive) that is
  folded into a last-write-wins state table across chunks in story order.
  Runs in one of three modes (`--detail`):

    - "fast" (the original v1 behaviour): one LLM call per chunk (marker
      沉淀) returning memories for every character AND the state_updates,
      in a single JSON object.
    - "exhaustive": per chunk, a roster call (marker 出场) identifies which
      registry characters appear plus state_updates/story_time, then ONE
      沉淀 call PER appearing character demanding an exhaustive atomic-fact
      enumeration, followed by a configurable number of coverage-audit
      rounds (marker 补漏) that catch missed facts. This mode tends to
      over-produce per-character micro-memories and, because its prompts
      are Chinese, can push English-source memories toward Chinese too.
    - "atomic" (the DEFAULT, Task B2): per chunk, a roster call (state
      only) -> one atomize call breaking the chunk into complete, self-
      contained, source-language fragments (each capped at ~50 tokens) ->
      one assign call attributing each fragment to the owner role id(s)
      that experience/know it (falling back to the reserved `narrator`
      role for pure scene-setting with no clear owner). Each fragment is
      deposited exactly once via `SharedMemory.remember_atomic`, however
      many owners it has, instead of once per owning character. The
      read-only roster/atomize/assign LLM calls run concurrently across
      chunks; deposits are sequential in story order (consensus insert is
      stateful).

  Text is chunked by 回目 chapter boundaries when the input looks like a
  classic chaptered novel (see `_chunk_history_text`) in both "exhaustive"
  and "atomic" modes, so chapter titles can serve as a time anchor for
  memories and prompts.

  Every location reference (state_updates[].location, in either mode) is
  resolved through a deterministic id/name/alias -> canonical-id resolver
  built from the registry (§ `_RegistryResolvers`). References that don't
  resolve are collected across the whole pass and, if any remain once every
  chunk has been processed, fed to a single "registry补全" LLM call (marker
  补注册) that proposes new location entries; the registry is amended,
  re-written to disk, the resolver rebuilt, and the collected references
  re-resolved. Anything still unresolved after that has its location update
  dropped (the previous location, if any, is kept) with a warning -- it
  never produces a dangling/placeholder location.

  Each memory is inserted via `SharedMemory.remember(..., story_order=...,
  story_time=...)` so the existing normalize-gate + consensus-merge
  machinery (shared owners, smaller-story_order-wins) applies unchanged.

Assembly then emits a standard scenario YAML (alive characters with empty
goal stacks, dead characters with `archived: true`, locations, carriers)
plus a kickoff call (marker 起始) and a holographic `SharedMemory.export()`
dump referenced via `ltm_file` so the (expensive) sedimentation never has to
be redone to run/resume the resulting scenario. Assembly enforces three hard
location invariants (raising ValueError if they still don't hold after
auto-fixing what can be auto-fixed):

  I1: no two environments share a name or alias -- environments whose
      name/alias sets collide are auto-merged first (kept id = first seen).
  I2: every character status.location is a defined environment id.
  I3: every environment id is ASCII ([a-z0-9_]+) -- non-ascii ids are
      auto-slugged (id changes, Chinese name preserved) first.

A fourth invariant, I4 (memory attribution may only target CHARACTER ids),
is enforced earlier during sedimentation: a memory attributed to a known
but non-character id (a location or carrier) is skipped with a warning,
exactly like an attribution to a wholly unknown id.
"""

import asyncio
import json
import os
import re

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
        "别名,构建全书【注册表】。地点也常有别名(例如“长安”别名“西京”),请一并列出。"
        '以 JSON 对象返回,形如:{"characters": [{"id": "短id(拼音或英文)", '
        '"name": "人物姓名", "aliases": ["别名1", "别名2"], "profile": "一句话简介"}], '
        '"locations": [{"id": "短id", "name": "地点名", "aliases": ["别名1", "别名2"], '
        '"profile": "一句话简介"}], '
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
        aliases = loc.get("aliases") or []
        alias_str = "、".join(aliases) if aliases else "无"
        lines.append(f"- {lid}({loc.get('name', lid)}): 别名 {alias_str}")
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
        "若无法判断具体时间,可用“【本回】”等占位。"
        "state_updates 中的 location 请填规范地点 id(遇到别名也可填别名,系统会自动归并)。"
        '以 JSON 对象返回,形如:{"memories": {"<规范id>": ["【时间】原子记忆1", '
        '"【时间】原子记忆2"]}, "state_updates": [{"id": "规范id", '
        '"location": "地点id或别名或null", "alive": true}], '
        '"story_time": "本片段大致所处的时代(自由文本)"}。'
        "只输出 JSON,不要输出任何解释文字。\n\n"
        f"提示信息:{hints or ''}\n\n正文(第{chunk_idx + 1}块):\n{chunk}"
    )


def _roster_prompt(chunk_text: str, chapter_title: str | None, registry: dict, hints: str) -> str:
    alias_table = _alias_table(registry)
    char_ids = [c["id"] for c in registry.get("characters", []) or [] if "id" in c]
    title_line = f"章节标题:{chapter_title}\n\n" if chapter_title else ""
    return (
        "请阅读下面的小说片段,判断下面这份封闭人物 id 列表中,哪些角色在本片段中【出场】"
        "(出现、被提及、有行动或对话均算出场)。同时给出本片段涉及到的人物地点/存亡状态更新,"
        "以及本片段大致所处的时代。"
        '以 JSON 对象返回,形如:{"characters": ["规范id1", "规范id2"], '
        '"state_updates": [{"id": "规范id", "location": "地点id或别名或null", "alive": true}], '
        '"story_time": "本片段大致所处的时代(自由文本)"}。'
        "只输出 JSON,不要输出任何解释文字。\n\n"
        f"封闭人物 id 列表:{', '.join(char_ids)}\n"
        f"别名对照表:\n{alias_table}\n\n"
        f"{title_line}"
        f"提示信息:{hints or ''}\n\n正文:\n{chunk_text}"
    )


def _char_sediment_prompt(
    chunk_text: str, chapter_title: str | None, char_entry: dict, hints: str
) -> str:
    cid = char_entry.get("id")
    name = char_entry.get("name", cid)
    aliases = char_entry.get("aliases") or []
    alias_str = "、".join(aliases) if aliases else "无"
    time_anchor = f"【{chapter_title}】" if chapter_title else "【本回】"
    return (
        f"请阅读下面的小说片段,只关注角色「{name}」(id={cid},别名:{alias_str})"
        "本段的经历,将其【沉淀】为尽量详尽、覆盖全部细节的原子记忆列表:"
        "该角色本段的每一个行动、每一句关键对话要点、重要见闻、人际关系变化、得失、"
        "情感与立场转变,都应拆成一条独立记忆,一条只讲一件事,不要遗漏、不要合并多件事。"
        f"每条记忆不超过80字,并带时间前缀(如 {time_anchor},或从正文推断的具体年号/时间);"
        "若该角色在本片段中没有实际经历(仅被提及而未真正参与),返回空数组。"
        '以 JSON 数组返回,形如:["【时间】原子记忆1", "【时间】原子记忆2"]。'
        "只输出 JSON,不要输出任何解释文字。\n\n"
        f"提示信息:{hints or ''}\n\n正文:\n{chunk_text}"
    )


def _coverage_prompt(
    chunk_text: str, chapter_title: str | None, extracted: dict, hints: str
) -> str:
    time_anchor = f"【{chapter_title}】" if chapter_title else "【本回】"
    return (
        "请阅读下面的小说片段与已抽取的各角色记忆列表,检查是否有重要遗漏"
        "(重要行动、对话要点、见闻、人际关系变化、得失、情感或立场转变等未被记录),"
        "为每一个有遗漏的角色【补漏】,补充遗漏的原子记忆"
        f"(同样每条不超过80字、带时间前缀,如 {time_anchor})。"
        '以 JSON 对象返回,形如:{"<角色id>": ["遗漏记忆1", "遗漏记忆2"]}。'
        "若确实没有遗漏,返回空对象 {}。"
        "只输出 JSON,不要输出任何解释文字。\n\n"
        f"已抽取记忆:{json.dumps(extracted, ensure_ascii=False)}\n\n"
        f"提示信息:{hints or ''}\n\n正文:\n{chunk_text}"
    )


def _atomize_prompt(chunk_text: str, chapter_title: str | None, hints: str) -> str:
    title_line = f"Chapter/section title: {chapter_title}\n\n" if chapter_title else ""
    hints_line = f"Hints: {hints}\n\n" if hints else ""
    return (
        "[atomize] Break the passage below into a JSON array of atomic memory "
        "fragments. Each fragment must state ONE complete event or fact, must be "
        "self-contained (resolve pronouns to explicit names instead of \"he\"/\"she\"/"
        "\"it\"/\"they\"), and MUST be written in the SAME LANGUAGE as the passage -- "
        "do not translate it into any other language, whatever that language is. "
        "Keep each fragment short: roughly one sentence, at most ~50 tokens. Output "
        "ONLY a JSON array of strings, e.g. [\"fragment 1\", \"fragment 2\"]. Do not "
        "output any explanation or extra text.\n\n"
        f"{title_line}{hints_line}Passage:\n{chunk_text}"
    )


def _assign_prompt(fragments: list[str], registry: dict, hints: str) -> str:
    alias_table = _alias_table(registry)
    numbered = "\n".join(f"{i}: {f}" for i, f in enumerate(fragments))
    return (
        "[assign] Below is a table of role ids (characters, locations, and "
        "information carriers) with their names/aliases, followed by a numbered "
        "list of memory fragments. For EACH fragment, list the id(s) of the "
        "role(s) it belongs to -- the character(s) who experience or know about "
        "it, or the location/carrier it concerns. A fragment MAY have MULTIPLE "
        "owners. If a fragment is pure scene-setting/narration with no clear "
        "character or location owner, assign it to the location it happens in, or "
        "to the reserved id \"narrator\" if nothing else applies. Only use ids "
        "from the role table below (plus \"narrator\") -- never invent new ids.\n"
        "Output ONLY a JSON array where element i is itself a JSON array of "
        "owner-id strings for fragment i (same length and order as the fragment "
        "list below), e.g. [[\"id1\"], [\"id1\", \"id2\"], [\"narrator\"]]. Do not "
        "output any explanation or extra text.\n\n"
        f"Role table:\n{alias_table}\n(reserved catch-all id: narrator)\n\n"
        f"Hints: {hints or ''}\n\nFragments:\n{numbered}"
    )


def _registry_augment_prompt(unresolved: list[str], registry: dict) -> str:
    existing = registry.get("locations") or []
    return (
        "在抽取小说人物经历的过程中,发现下面这些地点引用无法对应到现有地点登记信息"
        "(可能是新出现的地名,也可能是尚未登记的别名),请为它们【补注册】新的地点条目"
        "(如果某个引用其实是已有地点的别名,也可以把它作为 aliases 加入对应的新条目说明,"
        "但仍需给出一个独立的地点条目)。"
        '以 JSON 对象返回,形如:{"locations": [{"id": "短id(拼音或英文,ascii)", '
        '"name": "地点名", "aliases": ["别名1"], "profile": "一句话简介"}]}。'
        "只输出 JSON,不要输出任何解释文字。\n\n"
        f"待补注册的地点引用:{json.dumps(unresolved, ensure_ascii=False)}\n\n"
        f"现有地点登记信息:{json.dumps(existing, ensure_ascii=False)}"
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


async def _run_registry_augment(
    llm, registry: dict, unresolved: list[str], warnings: list[str]
) -> bool:
    """Runs ONE 补注册 call proposing new location entries for `unresolved`
    location references that didn't resolve against the current registry.
    Appends any valid new entries to `registry["locations"]` in place.

    Returns True iff at least one new location entry was added.
    """
    if not unresolved:
        return False

    prompt = _registry_augment_prompt(sorted(set(unresolved)), registry)
    raw = await llm.chat(prompt, bucket="extract")
    try:
        parsed = _extract_json_block(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        warnings.append(
            f"history registry augment (补注册) call failed to produce valid JSON: {exc}; "
            f"unresolved locations {unresolved!r} dropped"
        )
        return False

    if not isinstance(parsed, dict):
        warnings.append(
            "history registry augment (补注册) call did not return a JSON object; "
            f"unresolved locations {unresolved!r} dropped"
        )
        return False

    new_locations = parsed.get("locations") or []
    locations = registry.setdefault("locations", [])
    existing_ids = {loc.get("id") for loc in locations}
    added = False
    for loc in new_locations:
        if not isinstance(loc, dict) or not loc.get("id"):
            continue
        lid = loc["id"]
        if lid in existing_ids:
            continue
        locations.append(loc)
        existing_ids.add(lid)
        added = True

    if not added:
        warnings.append(
            f"history registry augment (补注册) call produced no usable new location "
            f"entries; unresolved locations {unresolved!r} dropped"
        )
    return added


# ----------------------------------------------------------------------
# Deterministic id/name/alias -> canonical-id resolver
# ----------------------------------------------------------------------


def _build_ref_map(entries: list[dict]) -> dict[str, str]:
    """Maps id -> id, name -> id, alias -> id for a registry category
    (characters, locations, or carriers). Ids always win ties; the first
    entry to claim a given name/alias string wins ties among those."""
    mapping: dict[str, str] = {}
    for e in entries or []:
        eid = e.get("id")
        if eid is None:
            continue
        mapping[eid] = eid
    for e in entries or []:
        eid = e.get("id")
        if eid is None:
            continue
        name = e.get("name")
        if name and name not in mapping:
            mapping[name] = eid
        for alias in e.get("aliases") or []:
            if alias and alias not in mapping:
                mapping[alias] = eid
    return mapping


class _RegistryResolvers:
    """Deterministic ref -> canonical-id resolvers built from a registry,
    kept separate per category (characters/locations/carriers) plus a
    combined `classify()` for attribution checks (I4)."""

    def __init__(self, registry: dict):
        self.character = _build_ref_map(registry.get("characters") or [])
        self.location = _build_ref_map(registry.get("locations") or [])
        self.carrier = _build_ref_map(registry.get("carriers") or [])
        self._rebuild_all()

    def _rebuild_all(self) -> None:
        combined: dict[str, tuple] = {}
        for ref, cid in self.location.items():
            combined.setdefault(ref, (cid, "location"))
        for ref, cid in self.carrier.items():
            combined.setdefault(ref, (cid, "carrier"))
        # Characters resolved last-wins-ties-first among the priority order
        # above only matters for genuinely ambiguous refs shared across
        # categories, which shouldn't happen in a well-formed registry;
        # characters take precedence since memory attribution cares most.
        for ref, cid in self.character.items():
            combined[ref] = (cid, "character")
        self.all = combined

    def resolve_character(self, ref: str | None) -> str | None:
        if ref is None:
            return None
        return self.character.get(ref)

    def resolve_location(self, ref: str | None) -> str | None:
        if ref is None:
            return None
        return self.location.get(ref)

    def classify(self, ref: str | None):
        """Returns (canonical_id, kind) or None if `ref` doesn't resolve to
        anything in the registry."""
        if ref is None:
            return None
        return self.all.get(ref)

    def refresh_locations(self, registry: dict) -> None:
        """Rebuild the location map (and combined map) after `registry`'s
        locations list has been mutated (e.g. by `_run_registry_augment`)."""
        self.location = _build_ref_map(registry.get("locations") or [])
        self._rebuild_all()


# ----------------------------------------------------------------------
# Chapter-aware chunking (H5)
# ----------------------------------------------------------------------

_CHAPTER_RE = re.compile(r"^第[一二三四五六七八九十百]+回[　 ]", re.MULTILINE)


def _split_by_chapters(text: str) -> list[str] | None:
    """Splits `text` at every 回目 heading. Returns None if the text
    doesn't look like a chaptered classic novel (no heading found)."""
    matches = list(_CHAPTER_RE.finditer(text))
    if not matches:
        return None
    chapters = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        chapters.append(text[start:end])
    return chapters


def _chapter_title(chapter_text: str) -> str:
    return chapter_text.split("\n", 1)[0].strip()


def _chunk_history_text(
    text: str, *, chunk_chars: int = CHUNK_CHARS, overlap: int = CHUNK_OVERLAP
) -> list[dict]:
    """Chapter-aware chunker for the exhaustive sedimentation pipeline.

    If `text` matches the 回目 chapter-heading pattern, splits by chapter
    (each chapter = one chunk; chapters longer than chunk_chars are further
    sub-split by `_chunk_text`). Otherwise falls back to plain char-count
    chunking (`_chunk_text`), one "chapter" per chunk with no title.

    Returns a list of {"text", "chapter_idx", "seq", "title", "flat_idx"}
    dicts in story order; `flat_idx` is a flat 0-based index over the whole
    list (used as the `*1000` story_order base, exactly like the old
    flat-chunk-index scheme -- a chapter that doesn't need sub-splitting
    gets `flat_idx == chapter_idx`).
    """
    chapters = _split_by_chapters(text)
    raw: list[tuple[str, int, int, str | None]] = []
    if chapters is not None:
        for chapter_idx, chapter_text in enumerate(chapters):
            title = _chapter_title(chapter_text)
            if len(chapter_text) <= chunk_chars:
                raw.append((chapter_text, chapter_idx, 0, title))
            else:
                for seq, sub in enumerate(_chunk_text(chapter_text, chunk_chars=chunk_chars, overlap=overlap)):
                    raw.append((sub, chapter_idx, seq, title))
    else:
        for idx, c in enumerate(_chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)):
            raw.append((c, idx, 0, None))

    return [
        {"text": t, "chapter_idx": ci, "seq": seq, "title": title, "flat_idx": flat_idx}
        for flat_idx, (t, ci, seq, title) in enumerate(raw)
    ]


# ----------------------------------------------------------------------
# Pass 2 -- sedimentation: shared state-resolution phase
# ----------------------------------------------------------------------


async def _resolve_and_finalize_state(
    llm,
    registry: dict,
    resolvers: _RegistryResolvers,
    raw_state_updates: list[tuple[int, dict]],
    warnings: list[str],
) -> dict:
    """Phase 2 of state handling, shared by fast and exhaustive modes.

    `raw_state_updates` is `[(chunk_flat_idx, {"id":..., "location":...,
    "alive":...}), ...]` in story order, with ids/locations still raw
    (un-resolved) strings from the LLM. This:

      1. Collects every location string that doesn't resolve against the
         current registry.
      2. If any remain, runs ONE 补注册 registry-augmentation call,
         appends the new location entries to `registry` (mutated in
         place -- callers re-serialize it to disk if they care), and
         rebuilds the location resolver.
      3. Replays `raw_state_updates` in order building the last-write-wins
         state table, resolving each `id` through the character resolver
         (unknown ids are dropped with a warning) and each `location`
         through the (possibly just-rebuilt) location resolver. A location
         that still doesn't resolve after augmentation is dropped (the
         previous location, if any, is kept) with a warning -- it never
         becomes a raw/placeholder value in the state table.
    """
    unresolved_raw: set[str] = set()
    for _, update in raw_state_updates:
        loc_raw = update.get("location")
        if loc_raw is not None and resolvers.resolve_location(loc_raw) is None:
            unresolved_raw.add(loc_raw)

    if unresolved_raw:
        augmented = await _run_registry_augment(llm, registry, sorted(unresolved_raw), warnings)
        if augmented:
            resolvers.refresh_locations(registry)

    state: dict[str, dict] = {}
    for _, update in raw_state_updates:
        uid_raw = update.get("id")
        if uid_raw is None:
            continue
        uid = resolvers.resolve_character(uid_raw)
        if uid is None:
            warnings.append(
                f"history state update: unknown character id {uid_raw!r} (not in registry); skipped"
            )
            continue

        entry = state.setdefault(uid, {"location": None, "alive": True})

        loc_raw = update.get("location")
        if loc_raw is not None:
            loc = resolvers.resolve_location(loc_raw)
            if loc is not None:
                entry["location"] = loc
            else:
                warnings.append(
                    f"history state update: location {loc_raw!r} for {uid!r} remained "
                    "unresolved after registry augmentation; keeping previous location"
                )

        if update.get("alive") is not None:
            entry["alive"] = update["alive"]

    return state


async def _remember_memory(
    shared: SharedMemory,
    resolvers: _RegistryResolvers,
    cid_raw: str,
    mem_text: str,
    *,
    story_order: int,
    story_time: str | None,
    warnings: list[str],
    context: str,
) -> bool:
    """I4: attribute one memory string to `cid_raw`, resolved through the
    combined registry resolver. Skips (with a warning) refs that don't
    resolve to anything (unknown) or that resolve to a known non-character
    id (location/carrier)."""
    classified = resolvers.classify(cid_raw)
    if classified is None:
        warnings.append(
            f"{context}: attributed memory to unknown id {cid_raw!r} (not in registry); skipped"
        )
        return False
    cid, kind = classified
    if kind != "character":
        warnings.append(
            f"{context}: attributed memory to non-character id {cid_raw!r} "
            f"(resolved to {cid!r}, kind={kind!r}); skipped"
        )
        return False
    await shared.remember(
        cid, mem_text, source="history", story_order=story_order, story_time=story_time
    )
    return True


# ----------------------------------------------------------------------
# Pass 2 -- fast mode (single call per chunk, v1 behaviour)
# ----------------------------------------------------------------------


async def _process_chunk_fast(
    llm,
    shared: SharedMemory,
    chunk_text: str,
    flat_idx: int,
    registry: dict,
    resolvers: _RegistryResolvers,
    hints: str,
    warnings: list[str],
    raw_state_updates: list[tuple[int, dict]],
) -> None:
    prompt = _sediment_prompt(chunk_text, flat_idx, registry, hints)
    raw = await llm.chat(prompt, bucket="extract")
    try:
        parsed = _extract_json_block(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        warnings.append(f"history sediment pass failed for chunk {flat_idx}: {exc}; chunk skipped")
        return

    if not isinstance(parsed, dict):
        warnings.append(f"history sediment pass: chunk {flat_idx} did not return a JSON object; skipped")
        return

    chunk_story_time = parsed.get("story_time")
    memories = parsed.get("memories") or {}

    mem_counter = 0
    for cid_raw, mems in memories.items():
        for mem_text in mems:
            ok = await _remember_memory(
                shared,
                resolvers,
                cid_raw,
                mem_text,
                story_order=flat_idx * 1000 + mem_counter,
                story_time=chunk_story_time,
                warnings=warnings,
                context=f"history sediment: chunk {flat_idx}",
            )
            if ok:
                mem_counter += 1

    for update in parsed.get("state_updates") or []:
        raw_state_updates.append((flat_idx, update))


async def _run_sediment_pass(
    llm,
    shared: SharedMemory,
    chunks: list[str],
    registry: dict,
    hints: str,
    warnings: list[str],
) -> dict:
    """Fast-mode Pass 2: one 沉淀 call per chunk, `chunks` a plain list of
    chunk strings (as produced by `society.extract._chunk_text`).

    Returns the final state table: {id: {"location": str|None, "alive": bool}}.
    """
    resolvers = _RegistryResolvers(registry)
    raw_state_updates: list[tuple[int, dict]] = []

    for flat_idx, chunk_text in enumerate(chunks):
        await _process_chunk_fast(
            llm, shared, chunk_text, flat_idx, registry, resolvers, hints, warnings, raw_state_updates
        )

    return await _resolve_and_finalize_state(llm, registry, resolvers, raw_state_updates, warnings)


# ----------------------------------------------------------------------
# Pass 2 -- exhaustive mode (roster + per-character + coverage audit)
# ----------------------------------------------------------------------


async def _process_chunk_exhaustive(
    llm,
    shared: SharedMemory,
    chunk: dict,
    registry: dict,
    resolvers: _RegistryResolvers,
    hints: str,
    warnings: list[str],
    coverage_rounds: int,
    raw_state_updates: list[tuple[int, dict]],
) -> None:
    chunk_text = chunk["text"]
    chapter_title = chunk["title"]
    flat_idx = chunk["flat_idx"]

    roster_prompt = _roster_prompt(chunk_text, chapter_title, registry, hints)
    raw = await llm.chat(roster_prompt, bucket="extract")
    try:
        roster = _extract_json_block(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        warnings.append(f"history roster (出场) pass failed for chunk {flat_idx}: {exc}; chunk skipped")
        return
    if not isinstance(roster, dict):
        warnings.append(f"history roster pass: chunk {flat_idx} did not return a JSON object; skipped")
        return

    chunk_story_time = roster.get("story_time") or (f"【{chapter_title}】" if chapter_title else None)

    appearing_ids: list[str] = []
    for ref in roster.get("characters") or []:
        cid = resolvers.resolve_character(ref)
        if cid is None:
            warnings.append(
                f"history roster: chunk {flat_idx} listed unknown character {ref!r}; skipped"
            )
            continue
        if cid not in appearing_ids:
            appearing_ids.append(cid)

    for update in roster.get("state_updates") or []:
        raw_state_updates.append((flat_idx, update))

    char_by_id = {c["id"]: c for c in registry.get("characters", []) or [] if "id" in c}

    # Per-character 沉淀 calls are independent (each only reads chunk_text +
    # one character's profile) -- issue them all concurrently. asyncio.gather
    # preserves input order in its results list regardless of completion
    # order, so we assemble per_char_entries/warnings by iterating
    # appearing_ids (NOT completion order) to keep deterministic downstream
    # deposit order and story_order assignment. return_exceptions=True so a
    # genuine post-retry exception from one call can't cancel siblings or
    # abort the chunk -- it degrades to the same skip+warning as an
    # invalid-JSON response.
    async def _fetch(cid: str) -> str:
        char_entry = char_by_id.get(cid, {"id": cid})
        prompt = _char_sediment_prompt(chunk_text, chapter_title, char_entry, hints)
        return await llm.chat(prompt, bucket="extract")

    raw_results = await asyncio.gather(*(_fetch(cid) for cid in appearing_ids), return_exceptions=True)

    per_char_entries: dict[str, list[str]] = {}
    for cid, raw_or_exc in zip(appearing_ids, raw_results):
        # Never swallow cancellation: let it propagate so an external
        # timeout/cancel scope (e.g. scenario-level parallelism) isn't
        # silently degraded into a per-character skip.
        if isinstance(raw_or_exc, asyncio.CancelledError):
            raise raw_or_exc
        if isinstance(raw_or_exc, Exception):
            warnings.append(
                f"history per-character sediment (沉淀) failed for {cid!r} in chunk {flat_idx}: "
                f"{raw_or_exc}; skipped"
            )
            continue
        raw = raw_or_exc
        try:
            mems = _extract_json_block(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            warnings.append(
                f"history per-character sediment (沉淀) failed for {cid!r} in chunk {flat_idx}: "
                f"{exc}; skipped"
            )
            continue
        if not isinstance(mems, list):
            warnings.append(
                f"history per-character sediment: {cid!r} chunk {flat_idx} did not return a "
                "JSON array; skipped"
            )
            continue
        per_char_entries[cid] = [str(m).strip() for m in mems if str(m).strip()]

    for round_no in range(coverage_rounds):
        if not per_char_entries:
            break
        audit_prompt = _coverage_prompt(chunk_text, chapter_title, per_char_entries, hints)
        raw = await llm.chat(audit_prompt, bucket="extract")
        try:
            missed = _extract_json_block(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            warnings.append(
                f"history coverage audit (补漏) round {round_no} failed for chunk {flat_idx}: "
                f"{exc}; skipped"
            )
            continue
        if not isinstance(missed, dict):
            warnings.append(
                f"history coverage audit: chunk {flat_idx} round {round_no} did not return a "
                "JSON object; skipped"
            )
            continue
        for ref, mems in missed.items():
            cid = resolvers.resolve_character(ref)
            if cid is None:
                warnings.append(
                    f"history coverage audit: chunk {flat_idx} attributed a missed fact to "
                    f"unknown id {ref!r}; skipped"
                )
                continue
            bucket = per_char_entries.setdefault(cid, [])
            for m in mems or []:
                m = str(m).strip()
                if m:
                    bucket.append(m)

    mem_counter = 0
    for cid, mems in per_char_entries.items():
        for mem_text in mems:
            await shared.remember(
                cid,
                mem_text,
                source="history",
                story_order=flat_idx * 1000 + mem_counter,
                story_time=chunk_story_time,
            )
            mem_counter += 1


async def _run_sediment_pass_exhaustive(
    llm,
    shared: SharedMemory,
    chunks: list[dict],
    registry: dict,
    hints: str,
    warnings: list[str],
    *,
    coverage_rounds: int = 1,
) -> dict:
    """Exhaustive-mode Pass 2: `chunks` a list of chunk dicts as produced
    by `_chunk_history_text` (chapter-aware). Per chunk: one roster call
    (出场) -> one 沉淀 call per appearing character -> `coverage_rounds`
    coverage-audit calls (补漏).

    Returns the final state table (same shape as `_run_sediment_pass`).
    """
    resolvers = _RegistryResolvers(registry)
    raw_state_updates: list[tuple[int, dict]] = []

    for chunk in chunks:
        await _process_chunk_exhaustive(
            llm, shared, chunk, registry, resolvers, hints, warnings, coverage_rounds, raw_state_updates
        )

    return await _resolve_and_finalize_state(llm, registry, resolvers, raw_state_updates, warnings)


# ----------------------------------------------------------------------
# Pass 2 -- atomic mode (roster + atomize + assign) [Task B2, default]
# ----------------------------------------------------------------------

NARRATOR_ID = "narrator"


def _ensure_narrator_role(registry: dict) -> bool:
    """Ensure `registry` has a reserved catch-all environment role with id
    `NARRATOR_ID`, so atomic-mode scene/narration fragments with no clear
    character/location owner always have a valid owner to fall back to.

    Added to `registry["locations"]` (not a separate category) so it is
    picked up by `_RegistryResolvers` and by `_assemble_history_scenario`
    for free, exactly like any other environment role. No-op (returns
    False) if an entry with that id is already present -- idempotent
    across repeated runs against a reused/hand-edited registry.
    """
    locations = registry.setdefault("locations", [])
    if any(loc.get("id") == NARRATOR_ID for loc in locations):
        return False
    locations.append(
        {
            "id": NARRATOR_ID,
            "name": "Narrator/旁白",
            "aliases": [],
            "profile": (
                "Reserved catch-all owner for scene-setting/narration fragments "
                "with no clear character or location owner."
            ),
        }
    )
    return True


async def _process_chunk_atomic(
    llm,
    chunk: dict,
    registry: dict,
    resolvers: _RegistryResolvers,
    hints: str,
    warnings: list[str],
) -> dict:
    """Read-only LLM phase for one chunk in atomic mode: roster (state only)
    -> atomize -> assign. Deliberately does NOT touch `shared` -- callers run
    this concurrently across chunks (asyncio.gather) and then deposit the
    returned fragments sequentially in story order (consensus insert is
    stateful, so deposit order must be preserved even though this read-only
    phase need not be).

    Returns {"flat_idx": int, "story_time": str | None,
    "state_updates": [raw update dicts, as from `_roster_prompt`],
    "deposits": [(fragment_text, [owner_id, ...]), ...]}.
    """
    chunk_text = chunk["text"]
    chapter_title = chunk["title"]
    flat_idx = chunk["flat_idx"]

    # 1. Roster call: state_updates + story_time only in this mode (its
    # per-character `characters` list isn't used for extraction here --
    # that's the whole point of replacing per-character extraction). A
    # failed/malformed roster response degrades to best-effort state (no
    # state_updates, story_time falls back to the chapter title) rather
    # than aborting the chunk -- atomization/assignment don't depend on it.
    story_time: str | None = f"【{chapter_title}】" if chapter_title else None
    state_updates: list[dict] = []
    roster_prompt = _roster_prompt(chunk_text, chapter_title, registry, hints)
    raw = await llm.chat(roster_prompt, bucket="extract")
    try:
        roster = _extract_json_block(raw)
        if not isinstance(roster, dict):
            raise ValueError("roster (出场) pass did not return a JSON object")
    except (ValueError, json.JSONDecodeError) as exc:
        warnings.append(
            f"history roster (出场) pass failed for chunk {flat_idx}: {exc}; "
            "proceeding with atomize/assign only (state is best-effort)"
        )
    else:
        story_time = roster.get("story_time") or story_time
        state_updates = roster.get("state_updates") or []

    empty_result = {
        "flat_idx": flat_idx,
        "story_time": story_time,
        "state_updates": state_updates,
        "deposits": [],
    }

    # 2. Atomize call.
    atomize_prompt = _atomize_prompt(chunk_text, chapter_title, hints)
    raw = await llm.chat(atomize_prompt, bucket="extract")
    try:
        fragments_raw = _extract_json_block(raw)
        if not isinstance(fragments_raw, list):
            raise ValueError("atomize pass did not return a JSON array")
    except (ValueError, json.JSONDecodeError) as exc:
        warnings.append(
            f"history atomize pass failed for chunk {flat_idx}: {exc}; chunk's memories skipped"
        )
        return empty_result

    fragments = [str(f).strip() for f in fragments_raw if str(f).strip()]
    if not fragments:
        return empty_result

    # 3. Assign call: owner id(s) per fragment.
    assign_prompt = _assign_prompt(fragments, registry, hints)
    raw = await llm.chat(assign_prompt, bucket="extract")
    try:
        owners_raw = _extract_json_block(raw)
        if not isinstance(owners_raw, list):
            raise ValueError("assign pass did not return a JSON array")
    except (ValueError, json.JSONDecodeError) as exc:
        warnings.append(
            f"history assign pass failed for chunk {flat_idx}: {exc}; every fragment in "
            f"the chunk falls back to {NARRATOR_ID!r}"
        )
        owners_raw = [[] for _ in fragments]

    if len(owners_raw) != len(fragments):
        warnings.append(
            f"history assign pass: chunk {flat_idx} returned {len(owners_raw)} owner "
            f"list(s) for {len(fragments)} fragment(s); padding/truncating defensively"
        )
        if len(owners_raw) < len(fragments):
            owners_raw = list(owners_raw) + [[] for _ in range(len(fragments) - len(owners_raw))]
        else:
            owners_raw = owners_raw[: len(fragments)]

    deposits: list[tuple[str, list[str]]] = []
    for fragment, raw_owner_list in zip(fragments, owners_raw):
        owners: list[str] = []
        if isinstance(raw_owner_list, list):
            for ref in raw_owner_list:
                classified = resolvers.classify(ref)
                if classified is None:
                    warnings.append(
                        f"history assign: chunk {flat_idx} assigned a fragment to unknown "
                        f"id {ref!r}; dropped"
                    )
                    continue
                cid, _kind = classified
                if cid not in owners:
                    owners.append(cid)
        if not owners:
            # Pure scene-setting, an empty owner list, or an owner list that
            # was entirely unresolvable -- the reserved catch-all always
            # resolves (it's a registry entry, added by `_ensure_narrator_role`
            # before this is ever called) so a fragment is never dropped for
            # lack of an owner.
            owners = [NARRATOR_ID]
        deposits.append((fragment, owners))

    return {
        "flat_idx": flat_idx,
        "story_time": story_time,
        "state_updates": state_updates,
        "deposits": deposits,
    }


async def _run_sediment_pass_atomic(
    llm,
    shared: SharedMemory,
    chunks: list[dict],
    registry: dict,
    hints: str,
    warnings: list[str],
) -> dict:
    """Atomic-mode Pass 2 (Task B2, default): `chunks` a list of chunk dicts
    as produced by `_chunk_history_text` (chapter-aware). Per chunk: one
    roster call (state only) -> one atomize call -> one assign call -- no
    per-character extraction. The read-only roster/atomize/assign LLM calls
    for ALL chunks run concurrently (`asyncio.gather`, `return_exceptions=True`);
    a `CancelledError` from any chunk is re-raised rather than swallowed (an
    external timeout/cancel scope must still see it), while any other
    exception degrades that one chunk to a skip+warning without aborting the
    rest. Once every chunk's read-only phase has resolved, deposits
    (`shared.remember_atomic`) happen sequentially in chunk/story order --
    consensus insert is stateful, so deposit order must be preserved even
    though gathering the LLM calls need not preserve completion order.

    Returns the final state table (same shape as `_run_sediment_pass`).
    """
    _ensure_narrator_role(registry)
    resolvers = _RegistryResolvers(registry)
    raw_state_updates: list[tuple[int, dict]] = []

    raw_results = await asyncio.gather(
        *(_process_chunk_atomic(llm, chunk, registry, resolvers, hints, warnings) for chunk in chunks),
        return_exceptions=True,
    )

    resolved: list[dict | None] = []
    for chunk, result in zip(chunks, raw_results):
        flat_idx = chunk["flat_idx"]
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, Exception):
            warnings.append(
                f"history atomic pipeline failed for chunk {flat_idx}: {result}; chunk skipped"
            )
            resolved.append(None)
            continue
        resolved.append(result)

    for entry in resolved:
        if entry is None:
            continue
        for update in entry["state_updates"]:
            raw_state_updates.append((entry["flat_idx"], update))

    for entry in resolved:
        if entry is None:
            continue
        flat_idx = entry["flat_idx"]
        story_time = entry["story_time"]
        for i, (fragment, owners) in enumerate(entry["deposits"]):
            await shared.remember_atomic(
                owners,
                fragment,
                source="history",
                story_order=flat_idx * 1000 + i,
                story_time=story_time,
            )

    return await _resolve_and_finalize_state(llm, registry, resolvers, raw_state_updates, warnings)


# ----------------------------------------------------------------------
# Assembly
# ----------------------------------------------------------------------

_ASCII_ID_RE = re.compile(r"^[a-z0-9_]+$")


def _is_ascii_id(s) -> bool:
    return isinstance(s, str) and bool(_ASCII_ID_RE.match(s))


def _merge_duplicate_locations(
    locations: list[dict], warnings: list[str]
) -> tuple[list[dict], dict[str, str]]:
    """I1 auto-fix: merges environments whose name or any alias collides
    with an earlier environment's name/alias set. The first environment to
    claim a given name/alias wins; later duplicates are dropped and their
    aliases folded into the surviving entry.

    Returns (kept_locations, remap) where `remap` maps every dropped id to
    the id of the environment it was merged into.
    """
    kept: list[dict] = []
    key_owner: dict[str, str] = {}
    by_id: dict[str, dict] = {}
    remap: dict[str, str] = {}

    for loc in locations:
        lid = loc.get("id")
        if lid is None:
            continue
        keys = set(loc.get("aliases") or [])
        name = loc.get("name")
        if name:
            keys.add(name)

        owner = None
        for k in keys:
            if k in key_owner:
                owner = key_owner[k]
                break

        if owner is not None:
            remap[lid] = owner
            kept_loc = by_id[owner]
            merged_aliases = set(kept_loc.get("aliases") or [])
            merged_aliases.update(loc.get("aliases") or [])
            if name:
                merged_aliases.add(name)
            merged_aliases.discard(kept_loc.get("name"))
            kept_loc["aliases"] = sorted(merged_aliases)
            for k in keys:
                key_owner[k] = owner
            warnings.append(
                f"history assembly (I1): merged duplicate environment {lid!r} into "
                f"{owner!r} (colliding name/alias)"
            )
            continue

        for k in keys:
            key_owner[k] = lid
        by_id[lid] = loc
        kept.append(loc)

    return kept, remap


def _slugify_ascii_id(name: str, used_ids: set[str], counter: list[int]) -> str:
    ascii_part = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    candidate = ascii_part if _is_ascii_id(ascii_part) else None
    if candidate is None:
        while True:
            candidate = f"loc_{counter[0]}"
            counter[0] += 1
            if candidate not in used_ids:
                return candidate
    base = candidate
    n = 1
    while candidate in used_ids:
        candidate = f"{base}_{n}"
        n += 1
    return candidate


def _ensure_ascii_location_ids(locations: list[dict], warnings: list[str]) -> dict[str, str]:
    """I3 auto-fix: replaces any non-ascii environment id with a synthesized
    ascii id (pinyin-ish slug of the name, or `loc_<n>` if that fails),
    preserving the original (Chinese) display name in "name". Mutates
    `locations` entries in place. Returns a remap of old id -> new id."""
    remap: dict[str, str] = {}
    used_ids = {loc["id"] for loc in locations if _is_ascii_id(loc.get("id"))}
    counter = [1]

    for loc in locations:
        lid = loc.get("id")
        if lid is None or _is_ascii_id(lid):
            continue
        name = loc.get("name") or lid
        new_id = _slugify_ascii_id(name, used_ids, counter)
        remap[lid] = new_id
        warnings.append(
            f"history assembly (I3): non-ascii environment id {lid!r} replaced with ascii id "
            f"{new_id!r} (name preserved: {name!r})"
        )
        loc["id"] = new_id
        if not loc.get("name"):
            loc["name"] = name
        aliases = set(loc.get("aliases") or [])
        aliases.add(lid)
        loc["aliases"] = sorted(aliases)
        used_ids.add(new_id)

    return remap


def _dedupe_role_ids(registry: dict, warnings: list[str]) -> None:
    """Guarantees globally-unique ids across `registry["characters"]`,
    `registry["locations"]`, and `registry["carriers"]` (processed in that
    deterministic order), fixing a Pass-1 LLM failure mode where the
    registry-merge call assigns the same id to two different entities (e.g.
    two distinct locations both getting id "daguanlou"), which would
    otherwise crash `_assemble_history_scenario` with "duplicate agent id".

    For each role, in order:
      - if its id is missing or non-ascii, a fresh ascii id is synthesized
        from its name (via `_slugify_ascii_id`); the original id (if any)
        is kept as an alias.
      - if its id was already claimed by an earlier role with the SAME
        name, it's a genuine duplicate entry describing the same entity --
        the later duplicate is dropped and its aliases are merged into the
        surviving role (no second agent is created).
      - if its id was already claimed by an earlier role with a DIFFERENT
        name, it's a real collision between two distinct entities -- this
        (later) role gets a fresh id synthesized from its own name, and the
        shared id it lost is added to its aliases so name/alias resolution
        still finds it.

    Every id handed out (kept or synthesized) is tracked in a single
    `used_ids` mapping shared across all three categories, so a newly
    synthesized id can never collide with anything, including ids from a
    different category. Mutates `registry` in place (each category's list
    may shrink if duplicates were dropped) and appends a warning for every
    rename/drop.
    """
    used_ids: dict[str, dict] = {}
    counter = [1]

    for category in ("characters", "locations", "carriers"):
        roles = registry.get(category) or []
        kept: list[dict] = []
        for role in roles:
            rid = role.get("id")
            name = role.get("name")

            if rid is None or not _is_ascii_id(rid):
                new_id = _slugify_ascii_id(name or category, used_ids, counter)
                if rid is not None:
                    aliases = set(role.get("aliases") or [])
                    aliases.add(rid)
                    role["aliases"] = sorted(a for a in aliases if a)
                    warnings.append(
                        f"history registry dedupe: {category} id {rid!r} missing/non-ascii; "
                        f"synthesized ascii id {new_id!r}"
                    )
                role["id"] = new_id
                used_ids[new_id] = role
                kept.append(role)
                continue

            prior = used_ids.get(rid)
            if prior is None:
                used_ids[rid] = role
                kept.append(role)
                continue

            if prior.get("name") == name:
                # Same id + same name -> genuinely the same entity: drop
                # this later duplicate, merging its aliases into the entry
                # that was kept.
                merged_aliases = set(prior.get("aliases") or [])
                merged_aliases.update(role.get("aliases") or [])
                prior["aliases"] = sorted(a for a in merged_aliases if a)
                warnings.append(
                    f"history registry dedupe: dropped duplicate {category} entry "
                    f"{rid!r} (same id, same name {name!r})"
                )
                continue

            # Same id, different name -> Pass 1 mis-assigned one id to two
            # distinct entities. Disambiguate this (later) role with a
            # fresh id, keeping the old shared id resolvable as an alias.
            new_id = _slugify_ascii_id(name or rid, used_ids, counter)
            aliases = set(role.get("aliases") or [])
            aliases.add(rid)
            role["aliases"] = sorted(a for a in aliases if a)
            role["id"] = new_id
            warnings.append(
                f"history registry dedupe: {category} id {rid!r} collided between "
                f"{prior.get('name')!r} and {name!r}; renamed the latter to {new_id!r} "
                "(old id kept as alias)"
            )
            used_ids[new_id] = role
            kept.append(role)

        registry[category] = kept


def _assemble_history_scenario(
    *,
    registry: dict,
    state: dict,
    scenario_name: str,
    language: str,
    warnings: list[str],
) -> tuple[dict, list[dict]]:
    """Returns (cfg-without-kickoff-or-ltm_file, carriers) so the caller can
    run the kickoff LLM call (which needs the assembled alive-id list) and
    the sediment export before finishing the cfg dict.

    Enforces I1-I3 (raising ValueError if they still don't hold once the
    available auto-fixes -- I1 duplicate-environment merge, I3 ascii-id
    slugging -- have been applied)."""
    raw_locations = [dict(loc) for loc in (registry.get("locations") or []) if "id" in loc]

    locations, merge_remap = _merge_duplicate_locations(raw_locations, warnings)
    slug_remap = _ensure_ascii_location_ids(locations, warnings)

    def resolve_loc_ref(raw):
        if raw is None:
            return None
        cur = merge_remap.get(raw, raw)
        cur = slug_remap.get(cur, cur)
        return cur

    location_ids = {loc["id"] for loc in locations}

    # I1 sanity check (should never trigger given the auto-merge above).
    seen_keys: dict[str, str] = {}
    for loc in locations:
        keys = set(loc.get("aliases") or [])
        if loc.get("name"):
            keys.add(loc["name"])
        for k in keys:
            owner = seen_keys.get(k)
            if owner is not None and owner != loc["id"]:
                raise ValueError(
                    f"I1 violated: environment name/alias {k!r} shared by {owner!r} and "
                    f"{loc['id']!r}"
                )
            seen_keys[k] = loc["id"]

    # I3 sanity check (should never trigger given the auto-slug above).
    for loc in locations:
        if not _is_ascii_id(loc["id"]):
            raise ValueError(f"I3 violated: environment id {loc['id']!r} is not ascii")

    fallback_location = next(iter(location_ids), None)

    agents = []
    for c in registry.get("characters", []) or []:
        cid = c.get("id")
        if cid is None:
            continue
        st = state.get(cid, {})
        loc = resolve_loc_ref(st.get("location"))
        loc = loc if loc is not None else fallback_location
        alive = st.get("alive", True)

        # I2: hard invariant, no auto-fix -- by construction every location
        # that reaches here already passed through the registry resolver
        # (+ 补注册 augmentation) during sedimentation, so this should only
        # fire for a genuinely malformed state table.
        if loc is not None and loc not in location_ids:
            raise ValueError(
                f"I2 violated: character {cid!r} status.location {loc!r} is not a defined "
                "environment"
            )

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
    detail: str = "atomic",
    coverage_rounds: int = 1,
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

    `detail` selects the Pass 2 strategy: "atomic" (DEFAULT, Task B2) runs
    a roster call (state only) + one atomize call + one assign call per
    chunk -- atomic, source-language, self-contained fragments each
    deposited once via `SharedMemory.remember_atomic` under however many
    owner role ids (character/location/carrier/`narrator`) they're
    assigned to; "exhaustive" runs a roster call + one 沉淀 call per
    appearing character + coverage audits per chunk; "fast" runs the
    original v1 single-call-per-chunk pipeline. "atomic" and "exhaustive"
    both use chapter-aware chunking (see `_chunk_history_text`). Any
    location reference collected during Pass 2 that doesn't resolve
    against the registry (in any mode) triggers a single 补注册
    registry-augmentation call once all chunks are processed; the
    (possibly amended) registry is re-written to `<out_yaml>.registry.json`
    if it changed.

    `max_agents` is accepted for CLI-flag symmetry with the snapshot
    pipeline but is not enforced in this v1 (archived agents must never be
    dropped just to make room, and that policy needs its own design pass --
    see design doc §9).
    """
    if detail not in ("atomic", "exhaustive", "fast"):
        raise ValueError(f"extract_history: unknown detail mode {detail!r}")

    warnings: list[str] = []
    # `_chunk_text`'s default overlap (500) is tuned for the default 8000-char
    # chunk size; a much smaller chunk_chars (e.g. tests exercising chunking
    # with a short text) would otherwise make step = chunk_chars - overlap
    # go non-positive and produce a pathological number of near-duplicate
    # chunks. Scale the overlap down to stay well under chunk_chars.
    overlap = min(CHUNK_OVERLAP, max(0, chunk_chars // 4))

    if detail in ("atomic", "exhaustive"):
        history_chunks = _chunk_history_text(text, chunk_chars=chunk_chars, overlap=overlap)
        flat_texts = [c["text"] for c in history_chunks]
    else:
        flat_texts = _chunk_text(text, chunk_chars=chunk_chars, overlap=overlap)
        history_chunks = flat_texts

    out_dir = os.path.dirname(os.path.abspath(out_yaml)) or "."
    os.makedirs(out_dir, exist_ok=True)
    registry_path = out_yaml + ".registry.json"

    ran_pass1 = registry is None
    if registry is None:
        registry = await _run_registry_pass(llm, flat_texts, hints, warnings)

    # Guarantee globally-unique role ids BEFORE anything else touches the
    # registry (resolvers, Pass-2 sedimentation/deposit-owner assignment,
    # assembly) -- whether `registry` just came out of Pass 1 above or was
    # supplied verbatim via the `registry=` argument (a reused/hand-edited
    # registry can carry the same duplicate-id bug). Without this, two
    # roles sharing an already-ascii id (so `_ensure_ascii_location_ids`
    # wouldn't touch it) survive all the way to `_assemble_history_scenario`
    # and crash it with "duplicate agent id".
    _dedupe_role_ids(registry, warnings)

    if ran_pass1:
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

    if registry_only:
        return {"registry": registry, "_warnings": warnings}

    if embed_fn is None:
        raise ValueError("extract_history: embed_fn is required for Pass 2 (sedimentation)")

    shared = SharedMemory(embed_fn, llm)
    locations_before = len(registry.get("locations") or [])
    if detail == "atomic":
        state = await _run_sediment_pass_atomic(llm, shared, history_chunks, registry, hints, warnings)
    elif detail == "exhaustive":
        state = await _run_sediment_pass_exhaustive(
            llm, shared, history_chunks, registry, hints, warnings, coverage_rounds=coverage_rounds
        )
    else:
        state = await _run_sediment_pass(llm, shared, history_chunks, registry, hints, warnings)

    if len(registry.get("locations") or []) != locations_before:
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

    scenario_name = os.path.splitext(os.path.basename(out_yaml))[0] or "extracted"
    cfg, carriers = _assemble_history_scenario(
        registry=registry, state=state, scenario_name=scenario_name, language=language, warnings=warnings
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
    kickoff_resolvers = _RegistryResolvers(registry)
    resolved_kickoff = []
    for k in kickoff:
        if not isinstance(k, dict):
            continue
        to_resolved = []
        for t in k.get("to", []) or []:
            rid = kickoff_resolvers.resolve_character(t)
            to_resolved.append(rid if rid is not None else t)
        if any(t in known_ids for t in to_resolved):
            k2 = dict(k)
            k2["to"] = to_resolved
            resolved_kickoff.append(k2)
    cfg["kickoff"] = resolved_kickoff

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
