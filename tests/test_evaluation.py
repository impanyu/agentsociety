import json

from society.baselines import GenerativeAgentsMemory
from society.evaluation import (
    _parse_json,
    cost_summary,
    event_hit_rate,
    extract_arcs,
    extract_events,
    footprint,
    generate_qa,
    narrative_quality,
    run_qa,
    trajectory_consistency,
)
from tests.helpers import FakeLLM, afake_embed


# ----------------------------------------------------------------------
# _parse_json
# ----------------------------------------------------------------------


def test_parse_json_handles_fenced_block():
    text = '```json\n{"a": 1}\n```'
    assert _parse_json(text, default=None) == {"a": 1}


def test_parse_json_handles_bare_json_with_surrounding_prose():
    text = 'Sure, here you go: {"a": 2} -- hope that helps!'
    assert _parse_json(text, default=None) == {"a": 2}


def test_parse_json_garbage_returns_default_without_raising():
    assert _parse_json("not json at all", default=[]) == []
    assert _parse_json("", default={"x": 1}) == {"x": 1}
    assert _parse_json(None, default="fallback") == "fallback"


# ----------------------------------------------------------------------
# extract_events
# ----------------------------------------------------------------------


async def test_extract_events_parses_canned_json():
    canned = json.dumps(
        [
            {"event": "The king dies", "characters": ["king"], "order": 1},
            {"event": "The war begins", "characters": ["king", "general"], "order": 2},
        ]
    )
    llm = FakeLLM(responses=[canned])
    events = await extract_events("held out chapters text", llm)
    assert events == [
        {"event": "The king dies", "characters": ["king"], "order": 1},
        {"event": "The war begins", "characters": ["king", "general"], "order": 2},
    ]


async def test_extract_events_respects_max_events_cap():
    canned = json.dumps(
        [{"event": f"event {i}", "characters": [], "order": i} for i in range(5)]
    )
    llm = FakeLLM(responses=[canned])
    events = await extract_events("held out text", llm, max_events=2)
    assert len(events) == 2


# ----------------------------------------------------------------------
# event_hit_rate
# ----------------------------------------------------------------------


async def test_event_hit_rate_counts_hits_and_misses():
    gold_events = [
        {"event": "The king dies", "characters": ["king"], "order": 1},
        {"event": "The war begins", "characters": ["king"], "order": 2},
        {"event": "The city floods", "characters": [], "order": 3},
    ]
    judge = FakeLLM(
        responses=[
            json.dumps({"occurred": True, "evidence": "seen in transcript"}),
            json.dumps({"occurred": True, "evidence": "also seen"}),
            json.dumps({"occurred": False, "evidence": "no mention"}),
        ]
    )
    result = await event_hit_rate(gold_events, "some transcript text", judge)
    assert result["hits"] == 2
    assert result["total"] == 3
    assert abs(result["rate"] - (2 / 3)) < 1e-9
    assert len(result["per_event"]) == 3
    assert result["per_event"][0]["occurred"] is True
    assert result["per_event"][2]["occurred"] is False
    assert result["per_event"][0]["event"] == "The king dies"


# ----------------------------------------------------------------------
# extract_arcs + trajectory_consistency
# ----------------------------------------------------------------------


async def test_extract_arcs_parses_canned_dict():
    canned = json.dumps(
        {
            "Alice": "Alice grows from timid scribe to bold leader.",
            "Bob": "Bob betrays his allies for power.",
        }
    )
    llm = FakeLLM(responses=[canned])
    arcs = await extract_arcs("held out text", ["Alice", "Bob"], llm)
    assert arcs == {
        "Alice": "Alice grows from timid scribe to bold leader.",
        "Bob": "Bob betrays his allies for power.",
    }


async def test_trajectory_consistency_scores_each_character():
    gold_arcs = {
        "Alice": "Alice grows from timid scribe to bold leader.",
        "Bob": "Bob betrays his allies for power.",
    }
    judge = FakeLLM(
        responses=[
            json.dumps({"score": 0.9}),
            json.dumps({"score": 0.3}),
        ]
    )
    result = await trajectory_consistency(gold_arcs, "sim transcript text", judge)
    assert result["per_character"] == {"Alice": 0.9, "Bob": 0.3}
    assert abs(result["mean"] - 0.6) < 1e-9


# ----------------------------------------------------------------------
# generate_qa
# ----------------------------------------------------------------------


async def test_generate_qa_parses_and_caps_n():
    canned = json.dumps(
        [{"q": f"question {i}?", "a": f"answer {i}"} for i in range(5)]
    )
    llm = FakeLLM(responses=[canned])
    qa = await generate_qa("sedimented source text", llm, n=3)
    assert len(qa) == 3
    assert qa[0] == {"q": "question 0?", "a": "answer 0"}


# ----------------------------------------------------------------------
# run_qa
# ----------------------------------------------------------------------


async def test_run_qa_retrieves_correct_entry_and_scores_accuracy():
    FACT_A = "The treaty was signed in the year 1200."
    FACT_B = "The old stone castle overlooked the river valley."

    backend = GenerativeAgentsMemory(afake_embed)
    await backend.remember("alice", FACT_A)
    await backend.remember("alice", FACT_B)

    entries = backend.all_entries()
    fact_a_id = next(e["id"] for e in entries if e["text"] == FACT_A)

    # Query text is identical to FACT_A so the fake (hash-seeded) embedding
    # guarantees cosine==1.0 against that entry -- deterministically proving
    # the ranking mechanism picks the right entry, independent of whether
    # the fake embedding carries any real semantics.
    qa_items = [
        {"q": FACT_A, "a": "1200"},
        {"q": "What color was the sky in the story?", "a": "unknown"},
    ]
    answerer = FakeLLM(responses=["1200", "UNKNOWN"])
    judge = FakeLLM(
        responses=[
            json.dumps({"correct": True}),
            json.dumps({"correct": False}),
        ]
    )

    result = await run_qa(backend, qa_items, answerer, afake_embed, judge, top_k=2)

    assert result["n"] == 2
    assert abs(result["accuracy"] - 0.5) < 1e-9
    assert len(result["per_item"]) == 2

    first = result["per_item"][0]
    assert first["q"] == FACT_A
    assert first["gold"] == "1200"
    assert first["answer"] == "1200"
    assert first["correct"] is True
    assert first["retrieved_ids"][0] == fact_a_id

    second = result["per_item"][1]
    assert second["answer"] == "UNKNOWN"
    assert second["correct"] is False


async def test_run_qa_uses_all_entries_not_per_agent_recall():
    # A fact remembered by one agent must still be retrievable when scoring
    # against the whole backend -- run_qa ranks over backend.all_entries(),
    # not backend.recall(agent_id, ...), so it is agnostic to which agent
    # "owns" the memory.
    FACT = "The bridge collapsed during the storm."
    backend = GenerativeAgentsMemory(afake_embed)
    await backend.remember("someone_else", FACT)

    qa_items = [{"q": FACT, "a": "the storm"}]
    answerer = FakeLLM(responses=["the storm"])
    judge = FakeLLM(responses=[json.dumps({"correct": True})])

    result = await run_qa(backend, qa_items, answerer, afake_embed, judge, top_k=1)
    assert result["accuracy"] == 1.0
    assert len(result["per_item"][0]["retrieved_ids"]) == 1


# ----------------------------------------------------------------------
# narrative_quality
# ----------------------------------------------------------------------


async def test_narrative_quality_computes_overall_mean():
    canned = json.dumps(
        {
            "coherence": 4,
            "character_distinctiveness": 3,
            "drama": 5,
            "fidelity": 4,
        }
    )
    judge = FakeLLM(responses=[canned])
    result = await narrative_quality("some screenplay text", judge)
    assert result["coherence"] == 4
    assert result["character_distinctiveness"] == 3
    assert result["drama"] == 5
    assert result["fidelity"] == 4
    assert abs(result["overall"] - 4.0) < 1e-9


# ----------------------------------------------------------------------
# footprint
# ----------------------------------------------------------------------


async def test_footprint_matches_hand_computed_bytes():
    backend = GenerativeAgentsMemory(afake_embed)
    text_1 = "关羽千里走单骑"
    text_2 = "The quick brown fox"
    await backend.remember("guanyu", text_1)
    await backend.remember("guanyu", text_2)

    result = footprint(backend)
    expected_bytes = len(text_1.encode("utf-8")) + len(text_2.encode("utf-8"))
    assert result["entries"] == 2
    assert result["text_bytes"] == expected_bytes
    assert "shared" in result
    assert "ratio" in result
    stats = backend.stats()
    assert result["shared"] == stats["shared"]
    assert result["ratio"] == stats["ratio"]


# ----------------------------------------------------------------------
# cost_summary
# ----------------------------------------------------------------------


def test_cost_summary_pulls_from_total_and_passes_wall_clock():
    usage = {
        "decide": {"calls": 3, "tokens": 300},
        "importance": {"calls": 2, "tokens": 200},
        "_total": {"calls": 5, "tokens": 500},
    }
    result = cost_summary(usage, 12.5)
    assert result == {"calls": 5, "tokens": 500, "wall_clock_s": 12.5}
