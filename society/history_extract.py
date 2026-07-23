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
  canonical role ids (characters/locations/carriers) plus a state_updates
  list (location/alive) that is folded into a last-write-wins state table
  across chunks in story order. Runs in one of three modes (`--detail`):

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
    - "atomic" (the DEFAULT, Task B2, reworked in Task F2, group-ref/
      per-event/background-drop fixes in Task F2.1): per chunk, a roster
      call (state only) -> one atomize call breaking the chunk into EVENT
      GROUPS of complete, self-contained, source-language fragments (each
      capped at ~64 tokens; no pronouns, no unresolved collective/group
      references like "三人"/"they", no chapter-heading/marker prefixes,
      no authorial framing, no dynastic-history/background exposition) ->
      one assign call PER EVENT (an event larger than ~20 fragments is
      split into <=20-fragment sub-batches that still share the full
      event as context) attributing each fragment to EVERY role id that
      would plausibly know/be aware of it -- participants, witnesses, the
      location(s) it happens in, and any carrier involved -- over-
      inclusion preferred over under-inclusion; assigning per event (not
      one flat batched fragment list for the whole chunk) gives the model
      the whole scene so it can resolve group references left over from
      atomize. There is no reserved `narrator` catch-all: a fragment with
      no resolvable owner falls back first to the location(s) OTHER
      fragments in the SAME EVENT resolved to, then to the chunk's
      roster-derived location id(s); if even that fails to resolve, the
      fragment is dropped with a warning instead of ever being deposited
      ownerless. Each fragment is deposited exactly once via
      `SharedMemory.remember_atomic`, however many owners it has, instead
      of once per owning character, and every
      event's fragments are linked together afterwards via
      `SharedMemory.link_group` so they're mutually affiliated. The
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
  story_time=...)` (fast/exhaustive modes) or `SharedMemory.remember_atomic(
  ..., story_order=..., affiliated=...)` (atomic mode, which no longer
  stores story_time -- story_order alone anchors it in the timeline) so the
  existing normalize-gate + consensus-merge machinery (shared owners,
  smaller-story_order-wins) applies unchanged.

Assembly then emits a standard scenario YAML (alive characters with empty
goal stacks, dead characters with `archived: true`, locations, carriers)
plus a kickoff call (marker 起始) and a holographic `SharedMemory.export()`
dump referenced via `ltm_file` so the (expensive) sedimentation never has to
be redone to run/resume the resulting scenario. Assembly enforces three
location invariants; registry data-quality issues (I1, I3) are auto-fixed
with a warning rather than crashing the run, since a scenario should never
die at the final assembly step over noisy Pass-1 registry data -- only I2,
which should only ever fire for a genuinely malformed state table, remains
a hard raise:

  I1: no two environments share a name or alias -- environments whose
      name/alias sets collide are auto-merged first (kept id = first seen);
      a pre-Pass-2 sanitizer (`_dedupe_env_aliases`) also strips
      cross-environment alias contamination (e.g. a general environment
      whose aliases are actually a more specific environment's name) before
      Pass 2 ever sees the registry, and any still-residual collision at
      assembly time is resolved the same way (drop the alias from every
      non-rightful environment) rather than raised.
  I2: every character status.location is a defined environment id (hard
      invariant, still raises).
  I3: every environment id is ASCII ([a-z0-9_]+) -- non-ascii ids are
      auto-slugged (id changes, Chinese name preserved) first, and any
      still-residual non-ascii id at assembly time is slugged the same way
      rather than raised.

A fourth invariant, I4 (memory attribution may only target CHARACTER ids),
is enforced earlier during sedimentation for "fast"/"exhaustive" mode: a
memory attributed to a known but non-character id (a location or carrier)
is skipped with a warning, exactly like an attribution to a wholly unknown
id. Atomic mode is a deliberate exception: its assign step may legitimately
attribute a fragment to a location or carrier id (a witness-location or an
involved information carrier), so `remember_atomic` owners are resolved
through the combined classifier without the character-only restriction.
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
        "[atomize] Break the passage below into EVENT GROUPS. Group the passage's "
        "atomic facts by the coherent scene or happening they belong to -- ONE "
        "EVENT is one scene/happening, and its FRAGMENTS are the individual atomic "
        "facts of that scene. Output ONLY a JSON array of events, where each event "
        "is itself a JSON array of fragment strings, e.g. "
        "[[\"fragment 1a\", \"fragment 1b\"], [\"fragment 2a\"]]. Do not output "
        "any explanation or extra text.\n\n"
        "Each fragment must state ONE complete fact or event and be FULLY "
        "SELF-CONTAINED: explicitly name the people and the place involved and "
        "state WHO did WHAT, WHERE -- resolve every pronoun (\"he\"/\"she\"/\"it\"/"
        "\"they\", etc.) to an explicit name instead. Do NOT include vague filler "
        "that carries no real information (e.g. \"nothing happened that night\") -- "
        "either make it concrete (state what actually happened) or leave it out "
        "entirely. Fragments MUST be written in the SAME LANGUAGE as the passage -- "
        "do not translate them into any other language, whatever that language is. "
        "Keep each fragment short: roughly one sentence, at most ~64 tokens.\n\n"
        "CRITICAL -- resolve every COLLECTIVE/GROUP reference to the explicit names "
        "of its members: no fragment may contain an unresolved group reference such "
        "as \"三人\"/\"我们三人\"/\"二人\"/\"众人\"/\"他们\"/\"the three (of them)\"/"
        "\"they\"/\"the group\" -- name the individuals instead (e.g. write "
        "\"刘备、关羽、张飞三人结为兄弟\", not \"三人结为兄弟\"; write \"Tom and Jerry "
        "agreed\", not \"they agreed\"). If a scene's participants were named earlier "
        "in the same passage, carry those names into every later fragment about "
        "them -- a fragment must be understandable completely on its own, without "
        "needing an earlier fragment to know who \"they\" refers to.\n\n"
        "Do NOT emit a chapter title or section heading as its own fragment, and do "
        "NOT prefix any fragment with a chapter marker (e.g. \"Chapter 1\"/\"第X回\") "
        "or a bracketed time marker (e.g. \"[Chapter 1]\"/\"【...】\"). Also DROP pure "
        "authorial framing that no character would personally remember or witness "
        "-- opening poems/verses, thematic authorial commentary or asides, and "
        "restatements of the chapter title. DROP, too, any AUTHORIAL/HISTORICAL "
        "BACKGROUND exposition that is not a concrete in-world event tied to "
        "identifiable characters and/or a specific place -- dynastic-history recaps "
        "(e.g. a sentence summarizing centuries of dynastic rise-and-fall), sweeping "
        "historical generalizations (\"the realm, long divided, must unite\"-style "
        "aphorisms), and other scene-setting narrator commentary that spans eras "
        "rather than describing something a character did or witnessed. KEEP only "
        "concrete events/facts that have identifiable participant(s) and/or a "
        "specific place -- a fragment that merely mentions a dynasty or era while "
        "describing a real character's concrete action (e.g. where/when they were "
        "born) is NOT background exposition and must be kept. None of this -- poems, "
        "chapter-title restatements, or background exposition -- may ever become a "
        "fragment.\n\n"
        f"{title_line}{hints_line}Passage:\n{chunk_text}"
    )


def _assign_prompt(
    fragments: list[str], registry: dict, hints: str, *, scene_context: list[str] | None = None
) -> str:
    """`fragments` is the list actually being assigned owners for in THIS
    call (either a whole event, or -- for an unusually large event -- one
    <=`_ASSIGN_BATCH_SIZE`-fragment sub-batch of it). `scene_context`, when
    given, is the FULL event's fragment list (Task F2.1 change #2): passed
    only when `fragments` is a sub-batch smaller than its event, so the
    model still sees the whole scene and can resolve a residual group
    reference ("they"/"the three of them") using sibling fragments it isn't
    itself assigning owners for in this call."""
    alias_table = _alias_table(registry)
    numbered = "\n".join(f"{i}: {f}" for i, f in enumerate(fragments))
    context_block = ""
    if scene_context:
        context_numbered = "\n".join(f"- {f}" for f in scene_context)
        context_block = (
            "The fragments below are only PART of one larger event/scene. For full "
            "scene context (to resolve any group reference like \"they\"/\"the three "
            "of them\" using sibling fragments from the same scene), here is the "
            f"COMPLETE list of that event's fragments:\n{context_numbered}\n\n"
        )
    return (
        "[assign] Below is a table of role ids (characters, locations, and "
        "information carriers) with their names/aliases, followed by a numbered "
        "list of self-contained memory fragments -- all from the SAME event/scene, "
        "so use them together as context for each other. For EACH fragment, list "
        "the id(s) of EVERY role that would plausibly know about, remember, or be "
        "aware of it -- every named participant/actor, everyone present or "
        "directly affected, the location(s) it happens in, and any information "
        "carrier (letter/diary/poem/etc.) involved. A fragment usually has "
        "MULTIPLE owners, often spanning different kinds of role (characters AND "
        "a location AND/or a carrier) -- when genuinely unsure whether an entity "
        "is aware of the fragment, OVER-INCLUDE it rather than leave it out; a "
        "memory should reach everyone who would plausibly know it. Base ownership "
        "only on entities actually named or clearly, directly implicated in the "
        "fragment -- do not attach an unrelated nearby character who isn't "
        "mentioned. Only use ids from the role table below -- never invent a new "
        "id, and there is no reserved/narrator catch-all id to fall back to. If a "
        "fragment genuinely names no character or carrier, you may still list its "
        "location id as an owner; if nothing in the role table applies at all, "
        "return an empty list for it.\n"
        "Output ONLY a JSON array where element i is itself a JSON array of "
        "owner-id strings for fragment i (same length and order as the fragment "
        "list below), e.g. [[\"id1\"], [\"id1\", \"id2\", \"loc1\"], []]. Do not "
        "output any explanation or extra text.\n\n"
        f"Role table:\n{alias_table}\n\n"
        f"{context_block}"
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


def _carrier_extract_prompt(source_text: str, carrier: dict, language: str, hints: str) -> str:
    """Prompt for the Task F4 per-carrier document-content extraction call:
    given the WHOLE source text plus one info_carrier's name/profile, asks
    for the carrier's CONTENT as a JSON array of sentence strings -- VERBATIM
    wherever the source actually quotes the document, otherwise a faithful
    reconstruction from the carrier's role/profile."""
    name = carrier.get("name") or carrier.get("id")
    profile = carrier.get("profile") or ""
    return (
        "[carrier-extract] Below is the full source text of a story, followed by the "
        f"name and profile of one INFORMATION CARRIER (a letter/edict/proclamation/diary/"
        f"poem/etc.) that appears in it: \"{name}\" (profile: {profile or 'n/a'}). Extract "
        "this carrier's own CONTENT -- the text it actually contains, or would plausibly "
        "contain -- as a JSON array of sentence strings, in reading order. Wherever the "
        "source text actually QUOTES this document, reproduce those sentences VERBATIM; "
        "wherever it doesn't quote the document directly, faithfully RECONSTRUCT plausible "
        "content instead, based on the carrier's role/profile and the surrounding context. "
        f"Write in the same language as the source text (language code: {language}). Each "
        "sentence must be fully SELF-CONTAINED (name the people/places involved explicitly "
        "-- no bare pronouns) and roughly one sentence, at most ~64 tokens. If this carrier "
        "has no discoverable or reconstructible content at all, return an empty JSON array.\n"
        "Output ONLY a JSON array of strings, e.g. [\"sentence 1\", \"sentence 2\"]. Do not "
        "output any explanation or extra text.\n\n"
        f"Hints: {hints or ''}\n\n"
        f"Carrier name: {name}\nCarrier profile: {profile}\n\nSource text:\n{source_text}"
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

# Max fragments per assign call (Task F2 change #3, restructured to be
# per-EVENT rather than a flat chunk-wide batch in Task F2.1 fix #2): each
# event's fragments are assigned together in one call so the model has the
# whole scene as context; only an unusually large event (more fragments than
# this) is split into sub-batches, each still sharing the full event as
# context, to keep any single call's role table + fragment list short.
_ASSIGN_BATCH_SIZE = 20

# Matches an old-style "第X回..." chapter-heading restatement (reusing
# `_CHAPTER_RE`'s pattern, but checked as a prefix of the stripped fragment
# rather than anchored to a line start) or a fragment that is ENTIRELY a
# bracketed marker (e.g. a bare "【楔子】" or "【建安五年】" with nothing else) --
# both are heading/marker artifacts the atomize prompt tells the model never
# to emit, but this is a defensive code-side filter rather than trusting
# prompt discipline alone (Task F2 change #2).
_HEADING_PREFIX_RE = re.compile(r"^第[一二三四五六七八九十百]+回[　 ]?")
_PURE_MARKER_RE = re.compile(r"^【[^】]*】$")


def _is_heading_or_marker_fragment(fragment: str) -> bool:
    """True if `fragment` (already stripped) looks like a chapter-heading
    restatement or a bare old-style time-marker rather than a real
    self-contained fact -- see the module-level regexes above."""
    return bool(_HEADING_PREFIX_RE.match(fragment) or _PURE_MARKER_RE.match(fragment))


# Defensive code-side filter (Task F2.1 fix #1) for authorial/historical
# BACKGROUND exposition -- dynastic-history recaps and sweeping historical
# generalizations -- that the atomize prompt tells the model to drop, but
# which (like the heading/marker case above) shouldn't be trusted to prompt
# discipline alone. Deliberately a small, narrow set of telltale
# sweeping-generalization phrases (drawn from the real validation failure:
# "周末七国分争最终秦统一天下"/"汉朝自高祖刘邦斩白蛇起义一统天下"-style openings)
# rather than anything keyed on a bare dynasty/era name, so a fragment that
# merely MENTIONS a dynasty while describing a real character's concrete
# action (e.g. where/when they were born) is never caught by this and stays.
_BACKGROUND_EXPOSITION_RE = re.compile(
    r"分久必合|合久必分|一统天下|统一天下|七国分争|六国分争|斩白蛇起义"
)


def _is_background_exposition_fragment(fragment: str) -> bool:
    """True if `fragment` (already stripped) reads as authorial/historical
    background exposition (a dynastic recap or sweeping historical
    generalization) rather than a concrete in-world event -- see
    `_BACKGROUND_EXPOSITION_RE` above."""
    return bool(_BACKGROUND_EXPOSITION_RE.search(fragment))


def _ensure_narrator_role(registry: dict) -> bool:
    """Ensure `registry` has a reserved catch-all environment role with id
    `NARRATOR_ID`. NOT used by the atomic path any more (Task F2 change #1
    removed the `narrator` fallback in favor of a location fallback) -- kept
    only so any other code that still imports/calls it (or a caller building
    its own registry-augmentation flow) keeps working. No-op (returns False)
    if an entry with that id is already present -- idempotent across
    repeated runs against a reused/hand-edited registry.
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


def _chunk_location_ids(state_updates: list[dict], resolvers: _RegistryResolvers) -> list[str]:
    """Distinct canonical location ids referenced by this chunk's roster
    state_updates, in first-seen order. This is the SECOND-priority
    no-owner fallback for assign (Task F2.1 fix #3, tightened from Task F2
    change #3's only fallback): a fragment with no resolvable character/
    carrier/location owner AND no sibling-fragment location within its own
    event falls back to wherever the chunk's own state updates say the
    action was happening, never to a reserved catch-all id. See
    `_process_chunk_atomic` for the event-local fallback that's tried
    first.
    """
    ids: list[str] = []
    for update in state_updates:
        loc_raw = update.get("location")
        if loc_raw is None:
            continue
        loc = resolvers.resolve_location(loc_raw)
        if loc is not None and loc not in ids:
            ids.append(loc)
    return ids


async def _assign_one_batch(
    llm,
    batch: list[str],
    registry: dict,
    hints: str,
    warnings: list[str],
    *,
    flat_idx: int,
    event_idx: int,
    scene_context: list[str] | None,
) -> list[list]:
    """Issues ONE assign call for `batch` -- a whole event's fragments, or
    (only for an event larger than `_ASSIGN_BATCH_SIZE`) one sub-batch of
    it, in which case `scene_context` carries the FULL event's fragment
    list so the model still sees the whole scene. Returns a list of raw
    (unresolved) owner-ref lists, always the same length as `batch`
    (padded/truncated defensively on a length mismatch -- never drifting
    into a sibling batch/event)."""
    assign_prompt = _assign_prompt(batch, registry, hints, scene_context=scene_context)
    raw = await llm.chat(assign_prompt, bucket="extract")
    try:
        owners_raw = _extract_json_block(raw)
        if not isinstance(owners_raw, list):
            raise ValueError("assign pass did not return a JSON array")
    except (ValueError, json.JSONDecodeError) as exc:
        warnings.append(
            f"history assign pass failed for chunk {flat_idx} event {event_idx}: {exc}; "
            "fragments in this batch have no LLM-assigned owner (event/chunk-location "
            "fallback applies)"
        )
        return [[] for _ in batch]

    if len(owners_raw) != len(batch):
        warnings.append(
            f"history assign pass: chunk {flat_idx} event {event_idx} returned "
            f"{len(owners_raw)} owner list(s) for {len(batch)} fragment(s) in this batch; "
            "padding/truncating defensively"
        )
        if len(owners_raw) < len(batch):
            owners_raw = list(owners_raw) + [[] for _ in range(len(batch) - len(owners_raw))]
        else:
            owners_raw = owners_raw[: len(batch)]

    return owners_raw


async def _assign_event(
    llm,
    event_fragments: list[str],
    registry: dict,
    hints: str,
    warnings: list[str],
    *,
    flat_idx: int,
    event_idx: int,
) -> list[list]:
    """Assigns owners for ONE event's fragments (Task F2.1 fix #2): a
    single assign call sees the event's ENTIRE fragment list together so
    the model has the whole scene as context and can resolve a residual
    collective/group reference ("they"/"the three of them") using sibling
    fragments in the same scene -- unless the event is unusually large
    (> `_ASSIGN_BATCH_SIZE` fragments), in which case it's split into
    <=`_ASSIGN_BATCH_SIZE`-fragment sub-batches issued sequentially (rare;
    each sub-batch still carries the full event as `scene_context`).
    Different EVENTS are gathered concurrently by the caller
    (`_process_chunk_atomic`), not here. Returns owners aligned index-for-
    index to `event_fragments`."""
    if len(event_fragments) <= _ASSIGN_BATCH_SIZE:
        return await _assign_one_batch(
            llm,
            event_fragments,
            registry,
            hints,
            warnings,
            flat_idx=flat_idx,
            event_idx=event_idx,
            scene_context=None,
        )

    owners: list[list] = []
    for start in range(0, len(event_fragments), _ASSIGN_BATCH_SIZE):
        batch = event_fragments[start : start + _ASSIGN_BATCH_SIZE]
        batch_owners = await _assign_one_batch(
            llm,
            batch,
            registry,
            hints,
            warnings,
            flat_idx=flat_idx,
            event_idx=event_idx,
            scene_context=event_fragments,
        )
        owners.extend(batch_owners)
    return owners


async def _process_chunk_atomic(
    llm,
    chunk: dict,
    registry: dict,
    resolvers: _RegistryResolvers,
    hints: str,
    warnings: list[str],
) -> dict:
    """Read-only LLM phase for one chunk in atomic mode: roster (state only)
    -> atomize (event groups) -> assign, one call PER EVENT (thorough
    owners, whole-scene context; Task F2.1 fix #2 -- events within a chunk
    are gathered concurrently, see below). Deliberately does NOT touch
    `shared` -- callers run this concurrently across chunks (asyncio.gather)
    and then deposit the returned fragments sequentially in story order
    (consensus insert is stateful, so deposit order must be preserved even
    though this read-only phase need not be).

    Returns {"flat_idx": int, "state_updates": [raw update dicts, as from
    `_roster_prompt`], "events": [[(fragment_text, [owner_id, ...]), ...],
    ...]} -- one inner list per EVENT, in atomize's original event order, so
    callers can `remember_atomic` each fragment and then `link_group` the
    ids returned for that event. Fragments that end up with no owner at all
    (no LLM-assigned owner AND no event/chunk-location fallback) are dropped
    (with a warning) and never appear in "events". story_time is no longer
    threaded through here (Task F2 change #5) -- it's not stored on atomic
    deposits any more, so there's nothing downstream that needs it.
    """
    chunk_text = chunk["text"]
    chapter_title = chunk["title"]
    flat_idx = chunk["flat_idx"]

    # 1. Roster call: state_updates only in this mode (its per-character
    # `characters` list, and its story_time, aren't used for extraction here
    # -- replacing per-character extraction is the whole point of this
    # mode, and atomic deposits no longer carry story_time at all). A
    # failed/malformed roster response degrades to best-effort state (no
    # state_updates) rather than aborting the chunk -- atomization/
    # assignment don't depend on it.
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
        state_updates = roster.get("state_updates") or []

    empty_result = {"flat_idx": flat_idx, "state_updates": state_updates, "events": []}

    # 2. Atomize call: JSON array of events, each itself a JSON array of
    # fragment strings.
    atomize_prompt = _atomize_prompt(chunk_text, chapter_title, hints)
    raw = await llm.chat(atomize_prompt, bucket="extract")
    try:
        events_raw = _extract_json_block(raw)
        if not isinstance(events_raw, list):
            raise ValueError("atomize pass did not return a JSON array")
    except (ValueError, json.JSONDecodeError) as exc:
        warnings.append(
            f"history atomize pass failed for chunk {flat_idx}: {exc}; chunk's memories skipped"
        )
        return empty_result

    events: list[list[str]] = []
    for event_raw in events_raw:
        if not isinstance(event_raw, list):
            warnings.append(
                f"history atomize: chunk {flat_idx} produced a non-array event entry; skipped"
            )
            continue
        frags: list[str] = []
        for f in event_raw:
            frag = str(f).strip()
            if not frag:
                continue
            if _is_heading_or_marker_fragment(frag):
                warnings.append(
                    f"history atomize: chunk {flat_idx} dropped a heading/marker fragment "
                    f"{frag!r}"
                )
                continue
            if _is_background_exposition_fragment(frag):
                warnings.append(
                    f"history atomize: chunk {flat_idx} dropped a background-exposition "
                    f"fragment {frag!r}"
                )
                continue
            frags.append(frag)
        if frags:
            events.append(frags)

    if not events:
        return empty_result

    # 3. Assign call(s): owner id(s) per fragment, issued PER EVENT (Task
    # F2.1 change #2) -- one assign call sees ONE event's fragments
    # together (or, for an unusually large event, <=`_ASSIGN_BATCH_SIZE`-
    # fragment sub-batches sharing the same full-event scene context) --
    # instead of the old flat batched fragment list across the whole
    # chunk, which lost event boundaries entirely and made it impossible
    # to resolve a group reference like "三人" from its sibling fragments
    # in the same scene. Different EVENTS' assign calls are independent
    # reads -> gather them concurrently; a `CancelledError` from any event
    # is re-raised (an external timeout/cancel scope must still see it)
    # while any other exception degrades just that event to an
    # all-fragments-ownerless result (event/chunk-location fallback then
    # applies) without aborting the rest. `asyncio.gather` preserves input
    # order in its results list regardless of completion order, so
    # zipping against `events` keeps index alignment correct per event.
    async def _assign_for_event(event_idx: int, event_fragments: list[str]) -> list[list]:
        return await _assign_event(
            llm, event_fragments, registry, hints, warnings, flat_idx=flat_idx, event_idx=event_idx
        )

    owners_raw_results = await asyncio.gather(
        *(_assign_for_event(i, event) for i, event in enumerate(events)), return_exceptions=True
    )

    owners_per_event_raw: list[list[list]] = []
    for event_idx, (event, result) in enumerate(zip(events, owners_raw_results)):
        if isinstance(result, asyncio.CancelledError):
            raise result
        if isinstance(result, Exception):
            warnings.append(
                f"history assign pass failed for chunk {flat_idx} event {event_idx}: "
                f"{result}; fragments in this event have no LLM-assigned owner "
                "(event/chunk-location fallback applies)"
            )
            owners_per_event_raw.append([[] for _ in event])
            continue
        owners_per_event_raw.append(result)

    # No reserved catch-all any more (Task F2 change #1). Fallback for a
    # fragment with no resolvable owner is now EVENT-LOCAL (Task F2.1 fix
    # #3): prefer the location(s) that OTHER fragments IN THE SAME EVENT
    # resolved to (that event's own place) over every location referenced
    # anywhere in the chunk; only fall through to this chunk's
    # roster-derived location(s) if the event itself named no location at
    # all; if even that's empty, the fragment is dropped rather than ever
    # deposited ownerless.
    chunk_location_ids = _chunk_location_ids(state_updates, resolvers)

    result_events: list[list[tuple[str, list[str]]]] = []
    for event_idx, (event, owners_raw) in enumerate(zip(events, owners_per_event_raw)):
        resolved_owners: list[list[str]] = []
        for raw_owner_list in owners_raw:
            owners: list[str] = []
            if isinstance(raw_owner_list, list):
                for ref in raw_owner_list:
                    classified = resolvers.classify(ref)
                    if classified is None:
                        warnings.append(
                            f"history assign: chunk {flat_idx} event {event_idx} assigned a "
                            f"fragment to unknown id {ref!r}; dropped"
                        )
                        continue
                    cid, _kind = classified
                    if cid not in owners:
                        owners.append(cid)
            resolved_owners.append(owners)

        # This event's own location(s): every distinct location-kind id any
        # of ITS fragments actually resolved to, in first-seen order -- the
        # preferred (first) source for this event's no-owner fallback.
        event_location_ids: list[str] = []
        for owners in resolved_owners:
            for oid in owners:
                classified = resolvers.classify(oid)
                if classified is not None and classified[1] == "location" and oid not in event_location_ids:
                    event_location_ids.append(oid)

        event_out: list[tuple[str, list[str]]] = []
        for fragment, owners in zip(event, resolved_owners):
            if not owners:
                if event_location_ids:
                    owners = list(event_location_ids)
                    warnings.append(
                        f"history assign: chunk {flat_idx} event {event_idx} fragment "
                        f"{fragment!r} had no owner; falling back to this event's own "
                        f"location(s) {event_location_ids!r}"
                    )
                elif chunk_location_ids:
                    owners = list(chunk_location_ids)
                    warnings.append(
                        f"history assign: chunk {flat_idx} fragment {fragment!r} had no owner; "
                        f"falling back to chunk location(s) {chunk_location_ids!r}"
                    )
                else:
                    warnings.append(
                        f"history assign: chunk {flat_idx} fragment {fragment!r} had no owner and "
                        "no chunk location to fall back to; dropped"
                    )
            if owners:
                event_out.append((fragment, owners))
        if event_out:
            result_events.append(event_out)

    return {"flat_idx": flat_idx, "state_updates": state_updates, "events": result_events}


async def _run_sediment_pass_atomic(
    llm,
    shared: SharedMemory,
    chunks: list[dict],
    registry: dict,
    hints: str,
    warnings: list[str],
) -> dict:
    """Atomic-mode Pass 2 (Task B2, reworked in Task F2, per-event assign in
    Task F2.1): `chunks` a list of chunk dicts as produced by
    `_chunk_history_text` (chapter-aware). Per chunk: one roster call (state
    only) -> one atomize call (event groups) -> one assign call PER EVENT
    (see `_process_chunk_atomic`) -- no per-character extraction, and no
    reserved `narrator` role (removed -- Task F2 change #1). The read-only
    roster/atomize/assign LLM calls for ALL chunks run concurrently
    (`asyncio.gather`, `return_exceptions=True`); a `CancelledError` from
    any chunk is re-raised rather than swallowed (an
    external timeout/cancel scope must still see it), while any other
    exception degrades that one chunk to a skip+warning without aborting the
    rest. Once every chunk's read-only phase has resolved, deposits
    (`shared.remember_atomic`) happen sequentially in chunk/story order --
    consensus insert is stateful, so deposit order must be preserved even
    though gathering the LLM calls need not preserve completion order.

    Per event, every fragment is deposited (via `remember_atomic`, with
    story_order but no story_time -- Task F2 change #5) and then
    `shared.link_group` is called on the ids actually returned (skipping any
    that came back None, i.e. empty after stripping) so the event's
    memories end up mutually affiliated (Task F2 change #4) -- this is
    exactly what F1's `link_group`/`affiliated` machinery in ltm.py exists
    for.

    Returns the final state table (same shape as `_run_sediment_pass`).
    """
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
        frag_counter = 0
        for event in entry["events"]:
            event_ids: list[str] = []
            for fragment, owners in event:
                inserted = await shared.remember_atomic(
                    owners,
                    fragment,
                    source="history",
                    story_order=flat_idx * 1000 + frag_counter,
                )
                frag_counter += 1
                if inserted is not None:
                    event_ids.append(inserted["id"])
            if event_ids:
                shared.link_group(event_ids)

    return await _resolve_and_finalize_state(llm, registry, resolvers, raw_state_updates, warnings)


# ----------------------------------------------------------------------
# Pass 2 -- info-carrier document chaining (Task F4, purely additive to the
# LTM: unifies documents into the entry model alongside the event memories
# deposited above, without touching scenario assembly/corpus wiring).
# ----------------------------------------------------------------------


async def _sediment_carriers(
    llm,
    shared: SharedMemory,
    registry: dict,
    source_text: str,
    language: str,
    hints: str,
    warnings: list[str],
    *,
    story_order_base: int,
) -> None:
    """Turns each `registry["carriers"]` entry into a CHAIN of sentence
    memory-entries, run AFTER the Pass-2 event sediment has deposited into
    `shared` and BEFORE `shared.export()` (see `extract_history`) so the
    chains are included in the holographic ltm export. Purely additive: this
    never touches `registry`, `_assemble_history_scenario`'s info_carrier
    agent build, or `_write_corpora` -- the corpus/RetrievalBrain wiring for
    info_carrier agents stays exactly as it was; only step 4 (the sim-side
    switch to reading these chains) will retire it, and that's a later task.

    For EACH carrier {id, name, profile}:

      1. One extraction LLM call (bucket="extract", `_carrier_extract_prompt`)
         asks for the carrier's CONTENT as a JSON array of sentence strings
         given the whole `source_text` plus the carrier's name/profile --
         verbatim wherever the source quotes the document, otherwise a
         faithful reconstruction from its profile. These per-carrier calls
         are independent reads, so they're gathered concurrently
         (`asyncio.gather(..., return_exceptions=True)`); a `CancelledError`
         from any of them is re-raised (an external timeout/cancel scope must
         still see it) while any other exception degrades just that carrier
         to a skip+warning without aborting the rest -- same pattern as the
         event-sediment gathers above. An empty array, a non-array response,
         or a JSON-parse failure also degrades to a skip+warning (no crash).

      2. Once every carrier's extraction has resolved, deposits happen
         SEQUENTIALLY (consensus insert + chaining are stateful, exactly why
         the event-deposit loop above is sequential too): each sentence is
         deposited via `shared.remember_atomic([carrier_id], sentence,
         source="document", readable=True, story_order=<monotonic>)` in
         array order, skipping any that come back None (empty after
         stripping). `story_order` is a single counter shared across every
         carrier's sentences, starting at `story_order_base` (callers pass
         something that sorts AFTER every event memory's story_order, e.g.
         `(num_chunks + 1) * 1000`) and incrementing by 1 per DEPOSITED
         sentence, in registry-carrier order then within-carrier sentence
         order -- so a document's own sentences always sort contiguously
         and in reading order relative to each other, and every carrier's
         chain sorts after the whole event timeline.

      3. Chain: for each pair of consecutively-deposited ids (ids[i],
         ids[i+1]) within ONE carrier, `shared.add_affiliations(ids[i],
         [ids[i+1]])` -- a DIRECTIONAL next-link (i -> i+1 only; i+1's
         affiliated set is NOT also given ids[i], unlike `link_group`'s
         symmetric pairwise linking used for event fragments). The LAST
         sentence of a carrier's chain has no next. Different carriers'
         chains are entirely independent of each other -- nothing links
         carrier A's entries to carrier B's.
    """
    carriers = [c for c in (registry.get("carriers") or []) if c.get("id")]
    if not carriers:
        return

    async def _extract(carrier: dict) -> str:
        prompt = _carrier_extract_prompt(source_text, carrier, language, hints)
        return await llm.chat(prompt, bucket="extract")

    raw_results = await asyncio.gather(*(_extract(c) for c in carriers), return_exceptions=True)

    story_order = story_order_base
    for carrier, raw_or_exc in zip(carriers, raw_results):
        cid = carrier["id"]

        # Never swallow cancellation: let it propagate so an external
        # timeout/cancel scope isn't silently degraded into a per-carrier
        # skip (same rationale as the event-sediment gathers above).
        if isinstance(raw_or_exc, asyncio.CancelledError):
            raise raw_or_exc
        if isinstance(raw_or_exc, Exception):
            warnings.append(
                f"history carrier sediment: extraction failed for carrier {cid!r}: "
                f"{raw_or_exc}; skipped"
            )
            continue

        try:
            sentences = _extract_json_block(raw_or_exc)
            if not isinstance(sentences, list):
                raise ValueError("carrier extract pass did not return a JSON array")
        except (ValueError, json.JSONDecodeError) as exc:
            warnings.append(
                f"history carrier sediment: carrier {cid!r} extraction failed to produce "
                f"valid JSON: {exc}; skipped"
            )
            continue

        sentences = [str(s).strip() for s in sentences if str(s).strip()]
        if not sentences:
            warnings.append(
                f"history carrier sediment: carrier {cid!r} produced no usable sentences; "
                "skipped"
            )
            continue

        ids: list[str] = []
        for sentence in sentences:
            res = await shared.remember_atomic(
                [cid],
                sentence,
                source="document",
                readable=True,
                story_order=story_order,
            )
            story_order += 1
            if res is not None:
                ids.append(res["id"])

        for i in range(len(ids) - 1):
            shared.add_affiliations(ids[i], [ids[i + 1]])


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


def _resolve_alias_collisions(locations: list[dict], warnings: list[str], *, context: str) -> None:
    """Shared alias-collision-resolution core, used both by the pre-Pass-2
    registry sanitizer (`_dedupe_env_aliases`) and by assembly's I1 safety
    net (`_assemble_history_scenario`).

    Ensures every name/alias key across `locations` is claimed by at most
    one environment: builds a map of alias/name-key -> the list of
    environments that carry it (as an alias), then for every key claimed by
    more than one environment -- or claimed as an alias when it's really
    ANOTHER environment's own `id`/`name` (cross-environment contamination,
    e.g. a general env like 贾府 carrying `荣国府`/`宁国府` as aliases when
    those are the actual NAMES of the specific envs `rongguofu`/`ningguofu`)
    -- keeps the key on the "rightful owner" (the environment whose own
    `name` or `id` equals the key; if none matches, the first environment
    in `locations` order to claim it) and strips the key from every other
    environment's `aliases` list. An environment's own `name` field is
    never touched, only `aliases` lists.

    Mutates `aliases` lists in place; appends one warning per alias
    removed, prefixed with `context` so callers (pre-Pass-2 sanitizer vs.
    assembly-time safety net) stay distinguishable in logs.
    """
    name_owner: dict[str, dict] = {}
    id_owner: dict[str, dict] = {}
    for loc in locations:
        name = loc.get("name")
        if name and name not in name_owner:
            name_owner[name] = loc
        lid = loc.get("id")
        if lid is not None and lid not in id_owner:
            id_owner[lid] = loc

    claimants: dict[str, list[dict]] = {}
    for loc in locations:
        for alias in loc.get("aliases") or []:
            claimants.setdefault(alias, []).append(loc)

    for key, envs in claimants.items():
        rightful = name_owner.get(key) or id_owner.get(key)
        if rightful is None:
            if len(envs) <= 1:
                continue
            rightful = envs[0]  # no name/id match -- first claimant wins

        for loc in envs:
            if loc is rightful:
                continue
            aliases = loc.get("aliases") or []
            if key not in aliases:
                continue
            loc["aliases"] = [a for a in aliases if a != key]
            if name_owner.get(key) is rightful or id_owner.get(key) is rightful:
                reason = f"is {rightful.get('id')!r}'s own name/id"
            else:
                reason = f"also claimed by {rightful.get('id')!r}"
            warnings.append(
                f"{context}: removed alias {key!r} from environment {loc.get('id')!r} "
                f"({reason}); kept on {rightful.get('id')!r}"
            )


def _dedupe_env_aliases(registry: dict, warnings: list[str]) -> None:
    """Pre-Pass-2 registry sanitizer: guarantees every environment
    name/alias key in `registry["locations"]` is claimed by at most one
    environment, fixing a Pass-1 LLM failure mode where a general/parent
    environment's alias list contaminates a more specific child
    environment's identity (e.g. a general env 贾府/jiafu listing
    `荣国府`/`宁国府` as its own aliases when those are the NAMES of the
    specific envs `rongguofu`/`ningguofu`). Left unfixed, one alias key
    would map to multiple environments -- ambiguous for the Pass-2
    name/alias -> canonical-id resolver, and fatal at assembly's I1 check
    ("no two environments share a name or alias").

    Delegates to `_resolve_alias_collisions` (see there for the exact
    keep-the-rightful-owner rule). Mutates `registry["locations"]` in
    place; a clean registry (no colliding aliases) is left untouched and no
    warnings are appended.
    """
    _resolve_alias_collisions(
        registry.get("locations") or [], warnings, context="history registry dedupe"
    )


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

    I1 and I3 are safety-net auto-fixed (never raise -- registry data
    quality must never crash a (possibly hours-long) sediment run at the
    final assembly step): I1 collisions are resolved by
    `_resolve_alias_collisions` (drop the alias from every non-rightful
    environment) and I3 non-ascii ids are re-slugged via
    `_ensure_ascii_location_ids`. Both should be no-ops in practice --
    `_dedupe_env_aliases` already sanitizes aliases before Pass 2 and the
    auto-merge/auto-slug immediately below already handle the common
    cases -- these are defense-in-depth for registry noise (e.g. a
    hand-edited or reused `registry=`) that slips past those. I2 remains a
    hard invariant (see below) since it should only ever fire for a
    genuinely malformed state table, not a registry data-quality issue."""
    raw_locations = [dict(loc) for loc in (registry.get("locations") or []) if "id" in loc]

    locations, merge_remap = _merge_duplicate_locations(raw_locations, warnings)
    slug_remap = _ensure_ascii_location_ids(locations, warnings)

    # I1 safety net: resolve any residual name/alias collision instead of
    # raising (should normally be a no-op -- see docstring above).
    _resolve_alias_collisions(locations, warnings, context="history assembly (I1)")

    # I3 safety net: auto-slug any residual non-ascii id instead of raising
    # (should normally be a no-op -- see docstring above). Reuses the exact
    # same machinery as the auto-slug pass immediately above, so it's
    # idempotent: a second call with nothing left to fix returns {}.
    i3_remap = _ensure_ascii_location_ids(locations, warnings)
    if i3_remap:
        # The rename above may reintroduce the old (now-alias) id as a
        # fresh collision -- resolve it the same way, defensively.
        _resolve_alias_collisions(locations, warnings, context="history assembly (I1)")

    def resolve_loc_ref(raw):
        if raw is None:
            return None
        cur = merge_remap.get(raw, raw)
        cur = slug_remap.get(cur, cur)
        cur = i3_remap.get(cur, cur)
        return cur

    location_ids = {loc["id"] for loc in locations}

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

    `detail` selects the Pass 2 strategy: "atomic" (DEFAULT, Task B2,
    reworked in Task F2, per-event assign in Task F2.1) runs a roster call
    (state only) + one atomize call (event groups) + one assign call PER
    EVENT per chunk -- atomic, source-language, self-contained fragments
    each deposited once via `SharedMemory.remember_atomic` under however
    many owner role ids
    (character/location/carrier -- no reserved `narrator` id any more)
    they're assigned to, with each event's fragments linked together via
    `SharedMemory.link_group`; "exhaustive" runs a roster call + one 沉淀
    call per appearing character + coverage audits per chunk; "fast" runs the
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

    # Same rationale as `_dedupe_role_ids` immediately above (must run BEFORE
    # resolvers/Pass-2/assembly ever see the registry, whether it just came
    # out of Pass 1 or was supplied verbatim via `registry=`): a Pass-1
    # registry can carry a general environment whose aliases contaminate a
    # more specific environment's name (cross-environment alias
    # contamination), which is ambiguous for Pass-2 resolution and fatal at
    # assembly's I1 check. See `_dedupe_env_aliases` docstring.
    _dedupe_env_aliases(registry, warnings)

    if ran_pass1:
        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

    if registry_only:
        return {"registry": registry, "_warnings": warnings}

    if embed_fn is None:
        raise ValueError("extract_history: embed_fn is required for Pass 2 (sedimentation)")

    # Task F2 change #6: sedimentation caps atomic fragments at 64 tokens
    # (was the SharedMemory default of 50) to match the new atomize prompt's
    # "~64 tokens" fragment budget. This only affects this sediment-pass
    # SharedMemory, not ltm.py's global default (untouched).
    shared = SharedMemory(embed_fn, llm, max_tokens=64)
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

    # Task F4: sediment each info_carrier's own content as a chained
    # (i -> i+1) run of readable document entries, ADDITIVE to the LTM --
    # AFTER the Pass-2 event sediment above has deposited into `shared` and
    # BEFORE the holographic export below so the chains are included in it.
    # Does not touch `registry`, the info_carrier agent build above, or
    # `_write_corpora` -- corpus/RetrievalBrain wiring is untouched (a later
    # task switches the sim side to read these chains instead).
    await _sediment_carriers(
        llm,
        shared,
        registry,
        text,
        language,
        hints,
        warnings,
        story_order_base=(len(history_chunks) + 1) * 1000,
    )

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
