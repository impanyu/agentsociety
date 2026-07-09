import argparse
import asyncio
import json
import os

import yaml

from society.embeddings import EmbeddingClient
from society.events import EventLog
from society.llm import LLMClient
from society.scenario import build_society, load_scenario
from society.screenplay import generate_screenplay


def write_transcripts(events: list[dict], agents: dict, out_dir: str) -> None:
    """Write one human-readable markdown transcript per agent.

    For each agent id in `agents`, writes `{out_dir}/transcripts/<id>.md`
    containing, in event order:
      - action events by that agent: `[tick N] name(params) -> ok/err: <data or error>`
      - message events where the agent is a recipient:
        `[tick N] <- <kind> from <sender>: <content>`
    """
    transcripts_dir = os.path.join(out_dir, "transcripts")
    os.makedirs(transcripts_dir, exist_ok=True)

    for aid in agents:
        lines = []
        for event in events:
            kind = event.get("kind")
            tick = event.get("tick")

            if kind == "action" and event.get("agent") == aid:
                action = event["action"]
                result = event["result"]
                name = action["name"]
                params = action["params"]
                if result.get("ok"):
                    lines.append(f"[tick {tick}] {name}({params}) -> ok: {result.get('data')}")
                else:
                    lines.append(f"[tick {tick}] {name}({params}) -> err: {result.get('error')}")

            elif kind == "message" and event.get("recipient") == aid:
                msg = event["message"]
                lines.append(
                    f"[tick {tick}] <- {msg['kind']} from {msg['sender']}: {msg['content']}"
                )

        path = os.path.join(transcripts_dir, f"{aid}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")


def _llm_config_snapshot(llm, embed_fn) -> dict:
    """Effective runtime LLM config for config_snapshot.yaml's "llm_config" key.

    NEVER includes api_key. Duck-types both arguments so this never crashes
    on the fakes injected by tests: a real LLMClient exposes chat_model /
    max_concurrency / max_calls / max_tokens as plain attributes, and the
    embed_model is read off the bound EmbeddingClient instance behind
    `embed_fn` when it is a `.embed` bound method. Any attribute missing
    (e.g. on a FakeLLM or a bare `afake_embed` function) is simply omitted,
    so injected fakes yield whatever subset is available, or {} if nothing
    duck-types.
    """
    cfg: dict = {}
    for key in ("chat_model", "max_concurrency", "max_calls", "max_tokens"):
        if hasattr(llm, key):
            cfg[key] = getattr(llm, key)

    embed_client = getattr(embed_fn, "__self__", None)
    embed_model = getattr(embed_client, "embed_model", None)
    if embed_model is not None:
        cfg["embed_model"] = embed_model

    return cfg


def write_outputs(
    kernel, out_dir: str, scenario_cfg: dict, summary: dict, *, embed_fn=None
) -> None:
    """Write llm_usage.json, config_snapshot.yaml, and per-agent transcripts.

    llm_usage.json: kernel.llm.usage() if kernel.llm duck-types usage(),
    else {}.
    config_snapshot.yaml: the scenario cfg (minus private "_"-prefixed keys
    such as "_dir") plus a "run_summary" key holding `summary`, plus an
    "llm_config" key holding the effective runtime LLM config (never
    api_key) -- see `_llm_config_snapshot`.
    """
    os.makedirs(out_dir, exist_ok=True)

    usage_fn = getattr(kernel.llm, "usage", None)
    usage = usage_fn() if usage_fn is not None else {}
    with open(os.path.join(out_dir, "llm_usage.json"), "w", encoding="utf-8") as f:
        json.dump(usage, f, ensure_ascii=False, indent=2)

    snapshot_cfg = {k: v for k, v in scenario_cfg.items() if not k.startswith("_")}
    snapshot_cfg["run_summary"] = summary
    snapshot_cfg["llm_config"] = _llm_config_snapshot(kernel.llm, embed_fn)
    with open(os.path.join(out_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(snapshot_cfg, f, allow_unicode=True)

    write_transcripts(kernel.event_log.all(), kernel.agents, out_dir)


async def run_scenario(scenario_path, ticks, out_dir, *, llm, embed_fn) -> dict:
    """Load, build, and run a scenario, then write all outputs.

    Orchestrates: load_scenario -> EventLog(events.jsonl) -> build_society
    (wired to write stats snapshots into out_dir) -> kernel.run(max_ticks) ->
    a final metrics snapshot (so stats/ always has at least one file, even
    if the run quiesces before the first periodic interval) -> write_outputs.
    Returns the summary dict from kernel.run().
    """
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_scenario(scenario_path)
    event_log = EventLog(os.path.join(out_dir, "events.jsonl"))

    kernel = await build_society(
        cfg, llm=llm, embed_fn=embed_fn, event_log=event_log, out_dir=out_dir
    )

    summary = await kernel.run(max_ticks=ticks)

    if kernel.metrics is not None:
        kernel.metrics.snapshot(kernel.tick)

    write_outputs(kernel, out_dir, cfg, summary, embed_fn=embed_fn)
    return summary


def _build_llm_and_embed(config_path: str | None):
    """Build a real LLMClient + EmbeddingClient.embed from a config.json,
    falling back to the OPENAI_API_KEY env var for the API key."""
    cfg = {}
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    api_key = cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    base_url = cfg.get("base_url", "https://api.openai.com/v1")
    chat_model = cfg.get("chat_model", "gpt-4o-mini")
    embed_model = cfg.get("embed_model", "text-embedding-3-small")
    max_concurrency = cfg.get("max_concurrency", 16)
    max_calls = cfg.get("max_calls")
    max_tokens = cfg.get("max_tokens")

    llm = LLMClient(
        api_key,
        base_url,
        chat_model,
        max_concurrency=max_concurrency,
        max_calls=max_calls,
        max_tokens=max_tokens,
    )
    embed_client = EmbeddingClient(api_key, base_url, embed_model)
    return llm, embed_client.embed


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run an AgentSociety scenario")
    parser.add_argument("--scenario", required=True, help="path to scenario yaml")
    parser.add_argument("--ticks", type=int, required=True, help="max ticks to run")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument(
        "--screenplay", action="store_true", help="generate a screenplay (Task 12)"
    )
    parser.add_argument(
        "--config", default="config.json", help="path to config.json (api_key, base_url, ...)"
    )
    args = parser.parse_args(argv)

    llm, embed_fn = _build_llm_and_embed(args.config)

    summary = asyncio.run(
        run_scenario(args.scenario, args.ticks, args.out, llm=llm, embed_fn=embed_fn)
    )

    if args.screenplay:
        cfg = load_scenario(args.scenario)
        language = cfg.get("language", "zh")
        events = EventLog.load(os.path.join(args.out, "events.jsonl"))
        asyncio.run(
            generate_screenplay(
                events, llm, out_path=os.path.join(args.out, "screenplay.md"), language=language
            )
        )

    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
