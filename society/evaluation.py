"""Offline evaluation harness for a completed simulation of one memory method.

These functions score a finished run (our `society.ltm.SharedMemory` or any of
the `society.baselines` backends) to produce the numbers reported in the
paper: continuation fidelity (did the simulated story reproduce the salient
plot / character trajectories of the held-out ground truth?), memory-grounded
QA accuracy (did the memory mechanism preserve retrievable facts?), narrative
quality (LLM-judged rubric), and footprint/cost (no LLM calls).

Every function here is PURE: it takes an injected `llm` / `judge` /
`embed_fn` and does nothing else side-effecting, so tests exercise it with
fakes and never touch a real API. `llm` and `judge` duck-type
`society.llm.LLMClient` (`async chat(prompt, system=None, bucket=...) -> str`
plus `usage() -> dict`); in production `judge` is a separate, stronger model
(gpt-4.1) used only to grade, never to generate. `embed_fn` duck-types
`async callable(list[str]) -> list[vector]`.

This module does not read or write simulation state, does not touch
`society.ltm`, `society.baselines`, or `society.metrics`, and is not wired
into `run.py` / the kernel -- it is called after the fact, offline, against
whatever a completed run left behind (a memory backend instance and/or its
exported entries, plus rendered transcript/screenplay text).
"""

import json
import math
import re


def _parse_json(text, default=None):
    """Tolerantly parse JSON out of an LLM reply.

    Handles three shapes, in order:
      1. A ```json ... ``` (or bare ``` ... ```) fenced code block -- the
         fenced content is parsed.
      2. A direct `json.loads` of the whole (stripped) string.
      3. Prose with a JSON value embedded somewhere in it -- scans for the
         first `{` or `[` and JSON-decodes from there, ignoring any trailing
         non-JSON text after the value.

    On any failure (empty/None input, no JSON found, malformed JSON) returns
    `default` instead of raising -- callers always get a value of the shape
    they expect, even from a misbehaving model.

    Args:
        text: Raw LLM reply (or None).
        default: Value to return if no JSON could be extracted.

    Returns:
        The parsed JSON value (list/dict/etc.), or `default`.
    """
    if not text:
        return default
    s = text.strip()

    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()

    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        pass

    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is None:
        return default

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(s[start:])
        return obj
    except (json.JSONDecodeError, ValueError):
        return default


def _cosine(a, b) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _clamp01(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))


# ==========================================================================
# Continuation fidelity
# ==========================================================================


async def extract_events(held_out_text: str, llm, *, max_events: int = 60) -> list[dict]:
    """Extract salient plot events (the ground truth) from held-out chapters.

    Makes ONE `llm.chat` call asking for a strict JSON list of the most
    salient plot events in `held_out_text`, each tagged with the characters
    involved and their narrative order. Used as the gold standard that a
    simulation's transcript is later checked against (`event_hit_rate`).

    Args:
        held_out_text: The later chapters/continuation withheld from the
            simulation, treated as ground truth.
        llm: Object with `async chat(prompt, system=None, bucket=...) -> str`.
        max_events: Cap on the number of events returned.

    Returns:
        list[dict]: [{"event": str, "characters": [str], "order": int}, ...],
        capped at `max_events`. Malformed/missing fields are normalized
        (missing "characters" -> [], missing "order" -> its list index). If
        the model's reply doesn't parse as JSON, returns [].
    """
    prompt = (
        "You are extracting salient plot events from a piece of narrative "
        "text, to be used as ground truth for evaluating a story simulation.\n\n"
        f"Text:\n{held_out_text}\n\n"
        f"List up to {max_events} of the most salient plot events, in "
        "narrative order. Return STRICT JSON: a list of objects, each "
        '{"event": "<short description>", "characters": ["<name>", ...], '
        '"order": <int>}. Return ONLY the JSON, no other text.'
    )
    reply = await llm.chat(prompt, system=None, bucket="eval_extract_events")
    parsed = _parse_json(reply, default=[])
    if not isinstance(parsed, list):
        return []

    events = []
    for i, item in enumerate(parsed[:max_events]):
        if not isinstance(item, dict):
            continue
        characters = item.get("characters", [])
        if not isinstance(characters, list):
            characters = []
        events.append(
            {
                "event": str(item.get("event", "")),
                "characters": [str(c) for c in characters],
                "order": item.get("order", i),
            }
        )
    return events


async def event_hit_rate(gold_events: list[dict], sim_transcript: str, judge) -> dict:
    """Score what fraction of gold events actually occurred in the sim.

    For EACH gold event, makes one `judge.chat` call asking whether an
    equivalent event occurred anywhere in `sim_transcript`.

    Args:
        gold_events: Output of `extract_events`.
        sim_transcript: Rendered event log / screenplay text of the run
            being scored.
        judge: Object with `async chat(prompt, system=None, bucket=...) -> str`.

    Returns:
        dict: {"hits": int, "total": int, "rate": float,
               "per_event": [{"event", "occurred", "evidence"}, ...]}
        `rate` is `hits / total` (0.0 if `total == 0`). A judge reply that
        fails to parse counts as occurred=False with empty evidence (fails
        closed, never crashes, never inflates the score).
    """
    per_event = []
    hits = 0
    for gold in gold_events:
        event_text = gold.get("event", "")
        prompt = (
            "Below is a simulated story transcript. Determine whether an "
            "event equivalent to the given gold event occurred anywhere in "
            "the transcript (paraphrases / different wording count as a "
            "match; only judge whether the same underlying event happened).\n\n"
            f"Gold event: {event_text}\n\n"
            f"Transcript:\n{sim_transcript}\n\n"
            'Return STRICT JSON: {"occurred": true/false, "evidence": '
            '"<short quote or note, or empty string>"}. Return ONLY the JSON.'
        )
        reply = await judge.chat(prompt, system=None, bucket="eval_judge")
        parsed = _parse_json(reply, default={"occurred": False, "evidence": ""})
        if not isinstance(parsed, dict):
            parsed = {"occurred": False, "evidence": ""}
        occurred = bool(parsed.get("occurred", False))
        evidence = str(parsed.get("evidence", "") or "")
        if occurred:
            hits += 1
        per_event.append({"event": event_text, "occurred": occurred, "evidence": evidence})

    total = len(gold_events)
    rate = (hits / total) if total else 0.0
    return {"hits": hits, "total": total, "rate": rate, "per_event": per_event}


async def extract_arcs(held_out_text: str, characters: list[str], llm) -> dict:
    """Extract each character's canonical arc from the held-out text.

    Makes ONE `llm.chat` call asking for a strict JSON object mapping each
    requested character name to a one-paragraph description of their arc
    (behaviour/goals/relationships trajectory) in `held_out_text`.

    Args:
        held_out_text: The later chapters/continuation withheld from the
            simulation, treated as ground truth.
        characters: Character names to extract arcs for.
        llm: Object with `async chat(prompt, system=None, bucket=...) -> str`.

    Returns:
        dict: {character: "<one-paragraph canonical arc>"}. Every name in
        `characters` is present as a key (empty string if the model omitted
        it or the reply failed to parse).
    """
    prompt = (
        "You are summarizing character arcs from a piece of narrative text, "
        "to be used as ground truth for evaluating a story simulation.\n\n"
        f"Text:\n{held_out_text}\n\n"
        f"Characters: {', '.join(characters)}\n\n"
        "For EACH character listed above, write one paragraph describing "
        "their canonical arc (behaviour, goals, relationships, how they "
        "change) in this text. Return STRICT JSON: an object mapping each "
        "character name to their one-paragraph arc string. Return ONLY the "
        "JSON, no other text."
    )
    reply = await llm.chat(prompt, system=None, bucket="eval_extract_arcs")
    parsed = _parse_json(reply, default={})
    if not isinstance(parsed, dict):
        parsed = {}
    return {c: str(parsed.get(c, "")) for c in characters}


async def trajectory_consistency(gold_arcs: dict, sim_transcript: str, judge) -> dict:
    """Score how consistent each character's simulated behaviour is with their gold arc.

    For each character in `gold_arcs`, makes one `judge.chat` call asking
    for a 0.0-1.0 consistency score between the canonical arc and the
    character's behaviour as it actually played out in `sim_transcript`.

    Args:
        gold_arcs: Output of `extract_arcs`.
        sim_transcript: Rendered event log / screenplay text of the run
            being scored.
        judge: Object with `async chat(prompt, system=None, bucket=...) -> str`.

    Returns:
        dict: {"per_character": {char: float}, "mean": float}. Scores are
        clamped to [0.0, 1.0]; a judge reply that fails to parse scores 0.0
        (fails closed). `mean` is the average over `per_character` (0.0 if
        empty).
    """
    per_character = {}
    for char, arc in gold_arcs.items():
        prompt = (
            "Below is a character's canonical arc (ground truth) and a "
            "simulated story transcript. Score how consistent the "
            "character's behaviour/arc in the transcript is with the "
            "canonical arc, on a scale from 0.0 (completely inconsistent / "
            "character absent or contradicted) to 1.0 (fully consistent).\n\n"
            f"Character: {char}\n"
            f"Canonical arc: {arc}\n\n"
            f"Transcript:\n{sim_transcript}\n\n"
            'Return STRICT JSON: {"score": <float between 0.0 and 1.0>}. '
            "Return ONLY the JSON."
        )
        reply = await judge.chat(prompt, system=None, bucket="eval_judge")
        parsed = _parse_json(reply, default={"score": 0.0})
        if not isinstance(parsed, dict):
            parsed = {"score": 0.0}
        per_character[char] = _clamp01(parsed.get("score", 0.0))

    mean = (sum(per_character.values()) / len(per_character)) if per_character else 0.0
    return {"per_character": per_character, "mean": mean}


# ==========================================================================
# Memory-grounded QA
# ==========================================================================


async def generate_qa(sedimented_source_text: str, llm, *, n: int = 40) -> list[dict]:
    """Auto-generate factual questions answerable from the sedimented source text.

    Makes ONE `llm.chat` call asking for `n` short factual question/answer
    pairs whose answers are explicitly present in `sedimented_source_text`
    (the material that was sedimented into memory BEFORE simulation started).
    These questions are later asked against a memory backend in `run_qa` to
    measure how well the memory mechanism preserved retrievable facts.

    Args:
        sedimented_source_text: The pre-simulation source text that was
            sedimented into the memory backend under test.
        llm: Object with `async chat(prompt, system=None, bucket=...) -> str`.
        n: Number of QA pairs requested / cap on how many are returned.

    Returns:
        list[dict]: [{"q": str, "a": str}, ...], capped at `n`. "a" is a
        short, factual gold answer. Returns [] if the reply fails to parse.
    """
    prompt = (
        "You are generating a factual quiz over the text below. Write "
        f"{n} short factual question/answer pairs whose answers are "
        "explicitly stated in the text (names, dates, places, relationships, "
        "objects, outcomes). Keep answers short (a few words).\n\n"
        f"Text:\n{sedimented_source_text}\n\n"
        'Return STRICT JSON: a list of objects, each {"q": "<question>", '
        '"a": "<short factual answer>"}. Return ONLY the JSON, no other text.'
    )
    reply = await llm.chat(prompt, system=None, bucket="eval_generate_qa")
    parsed = _parse_json(reply, default=[])
    if not isinstance(parsed, list):
        return []

    qa = []
    for item in parsed[:n]:
        if not isinstance(item, dict):
            continue
        qa.append({"q": str(item.get("q", "")), "a": str(item.get("a", ""))})
    return qa


async def run_qa(backend, qa_items: list[dict], answerer, embed_fn, judge, *, top_k: int = 5) -> dict:
    """Answer `qa_items` closed-book against `backend`'s stored memory, and score.

    Retrieval here is deliberately METHOD-AGNOSTIC: for each question, the
    question is embedded and ranked by cosine similarity against the TEXTS
    of `backend.all_entries()` (embedded via `embed_fn`), and the top_k
    entries are taken. This is NOT the same as calling `backend.recall()`.
    Why: `recall()` differs per backend in ways that are the very thing this
    harness is trying to compare (e.g. `GenerativeAgentsMemory` scopes
    recall to a single agent's private stream and adds recency/importance
    terms; `CollaborativeMemory` gates recall on a per-fragment read ACL;
    `SharedMemory` returns consensus-merged rows). If QA used each backend's
    own `recall()`, differences in per-agent routing/gating -- not
    differences in what the memory MECHANISM actually preserved and kept
    retrievable -- would drive the accuracy numbers, making cross-method
    comparison unfair (e.g. a GenerativeAgents run would look artificially
    bad simply because the asking "agent" never personally observed the
    fact, even though the fact is sitting in the store). Ranking directly
    over `all_entries()` isolates the one thing that legitimately varies
    across methods: HOW MUCH of the source material each method's
    compression/dedup strategy kept, and in what form.

    Once the top_k entries are chosen, `answerer` is asked to answer
    CLOSED-BOOK using ONLY those texts ("if not answerable, say UNKNOWN"),
    and `judge` scores the answer against the gold answer.

    Args:
        backend: Any object duck-typing the SharedMemory interface
            (`all_entries()`, `stats()`, ...).
        qa_items: Output of `generate_qa` (or hand-written): [{"q","a"}, ...].
        answerer: Object with `async chat(prompt, system=None, bucket=...) -> str`
            used to produce closed-book answers.
        embed_fn: `async callable(list[str]) -> list[vector]`.
        judge: Object with `async chat(prompt, system=None, bucket=...) -> str`
            used to grade answers against gold.
        top_k: Number of entries retrieved per question.

    Returns:
        dict: {"accuracy": float, "n": int,
               "per_item": [{"q","gold","answer","correct","retrieved_ids"}, ...]}
        `accuracy` is correct/n (0.0 if n == 0). A judge reply that fails to
        parse counts as incorrect (fails closed).
    """
    entries = backend.all_entries()
    entry_ids = [e["id"] for e in entries]
    entry_texts = [e["text"] for e in entries]
    entry_vecs = await embed_fn(entry_texts) if entry_texts else []

    per_item = []
    correct_count = 0
    for item in qa_items:
        question = item.get("q", "")
        gold = item.get("a", "")

        retrieved_ids = []
        retrieved_texts = []
        if entry_vecs:
            q_vec = (await embed_fn([question]))[0]
            scored = [
                (_cosine(q_vec, vec), eid, text)
                for vec, eid, text in zip(entry_vecs, entry_ids, entry_texts)
            ]
            scored.sort(key=lambda triple: triple[0], reverse=True)
            top = scored[:top_k]
            retrieved_ids = [eid for _, eid, _ in top]
            retrieved_texts = [text for _, _, text in top]

        notes = "\n".join(f"- {t}" for t in retrieved_texts) if retrieved_texts else "(no notes)"
        answer_prompt = (
            "Answer the question using ONLY the notes below (closed-book). "
            "If the notes do not contain the answer, reply exactly UNKNOWN. "
            "Keep the answer short.\n\n"
            f"Notes:\n{notes}\n\n"
            f"Question: {question}"
        )
        answer = (await answerer.chat(answer_prompt, system=None, bucket="eval_answer")).strip()

        judge_prompt = (
            "Compare the candidate answer to the gold answer for the given "
            "question. Judge it correct if it conveys the same fact, even "
            "if worded differently; judge it incorrect if it is UNKNOWN, "
            "wrong, or missing the key fact.\n\n"
            f"Question: {question}\n"
            f"Gold answer: {gold}\n"
            f"Candidate answer: {answer}\n\n"
            'Return STRICT JSON: {"correct": true/false}. Return ONLY the JSON.'
        )
        judge_reply = await judge.chat(judge_prompt, system=None, bucket="eval_judge")
        parsed = _parse_json(judge_reply, default={"correct": False})
        if not isinstance(parsed, dict):
            parsed = {"correct": False}
        is_correct = bool(parsed.get("correct", False))
        if is_correct:
            correct_count += 1

        per_item.append(
            {
                "q": question,
                "gold": gold,
                "answer": answer,
                "correct": is_correct,
                "retrieved_ids": retrieved_ids,
            }
        )

    n = len(qa_items)
    accuracy = (correct_count / n) if n else 0.0
    return {"accuracy": accuracy, "n": n, "per_item": per_item}


# ==========================================================================
# Narrative quality
# ==========================================================================


async def narrative_quality(screenplay_text: str, judge) -> dict:
    """Score a rendered screenplay on a 4-dimension 1-5 rubric.

    Makes ONE `judge.chat` call asking for four integer 1-5 ratings:
    coherence, character_distinctiveness, drama, fidelity.

    Args:
        screenplay_text: Rendered screenplay/transcript text of the run
            being scored.
        judge: Object with `async chat(prompt, system=None, bucket=...) -> str`.

    Returns:
        dict: {"coherence": int, "character_distinctiveness": int,
               "drama": int, "fidelity": int, "overall": float}
        `overall` is the mean of the four dimensions. Each dimension is
        clamped to [1, 5]; a reply that fails to parse degrades to all-1s
        (fails closed / worst score, never crashes).
    """
    prompt = (
        "Rate the following screenplay/story transcript on four dimensions, "
        "each on an integer scale from 1 (very poor) to 5 (excellent):\n"
        "- coherence: does the plot make logical sense and flow?\n"
        "- character_distinctiveness: do characters have distinct, "
        "consistent voices and behaviour?\n"
        "- drama: is there meaningful tension/conflict/stakes?\n"
        "- fidelity: does it stay faithful to the source material's tone "
        "and world?\n\n"
        f"Transcript:\n{screenplay_text}\n\n"
        'Return STRICT JSON: {"coherence": <int 1-5>, '
        '"character_distinctiveness": <int 1-5>, "drama": <int 1-5>, '
        '"fidelity": <int 1-5>}. Return ONLY the JSON.'
    )
    reply = await judge.chat(prompt, system=None, bucket="eval_judge")
    default = {
        "coherence": 1,
        "character_distinctiveness": 1,
        "drama": 1,
        "fidelity": 1,
    }
    parsed = _parse_json(reply, default=default)
    if not isinstance(parsed, dict):
        parsed = default

    def _clamp15(value):
        try:
            v = int(round(float(value)))
        except (TypeError, ValueError):
            v = 1
        return max(1, min(5, v))

    result = {dim: _clamp15(parsed.get(dim, 1)) for dim in default}
    result["overall"] = sum(result.values()) / len(default)
    return result


# ==========================================================================
# Footprint & cost (no LLM calls)
# ==========================================================================


def footprint(backend) -> dict:
    """Compute storage footprint of a memory backend (no LLM calls).

    Args:
        backend: Any object duck-typing the SharedMemory interface
            (`all_entries()`, `stats()`).

    Returns:
        dict: {"entries": int, "text_bytes": int, "shared": int, "ratio": float}
        `text_bytes` is the sum of UTF-8 encoded byte-lengths of every
        entry's text. `shared`/`ratio` are pulled straight from
        `backend.stats()`.
    """
    entries = backend.all_entries()
    text_bytes = sum(len(e["text"].encode("utf-8")) for e in entries)
    stats = backend.stats()
    return {
        "entries": len(entries),
        "text_bytes": text_bytes,
        "shared": stats["shared"],
        "ratio": stats["ratio"],
    }


def cost_summary(usage: dict, wall_clock_s: float) -> dict:
    """Summarize LLM cost for a run (no LLM calls).

    Args:
        usage: Output of `llm.usage()`: {bucket: {"calls","tokens"}, "_total": {...}}.
        wall_clock_s: Measured wall-clock duration of the run, in seconds.

    Returns:
        dict: {"calls": int, "tokens": int, "wall_clock_s": float}, where
        calls/tokens are pulled from `usage["_total"]` (0 if missing).
    """
    total = usage.get("_total", {}) or {}
    return {
        "calls": total.get("calls", 0),
        "tokens": total.get("tokens", 0),
        "wall_clock_s": wall_clock_s,
    }
