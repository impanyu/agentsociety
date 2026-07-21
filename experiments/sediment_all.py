"""Full-corpus sedimentation runner for the Agentsensus paper experiments.

Runs history-sedimentation for all four scenarios (their SEDIMENT spans only)
concurrently, logging per-scenario token usage, wall-clock, and memory-store
size. Sources live in scenarios/sources/; outputs go to scenarios/<name>.yaml
(+ .registry.json + .ltm.json).

The SEDIMENT spans (locked with the user):
  三国演义         ch 1-40   (reference ch 41-60)
  红楼梦           ch 1-40   (reference ch 41-80)
  War and Peace   Vol 1-2   (reference Vol 3)
  俄乌 timeline    2022/02-2024/04 (reference 2024/05-2026/07)

Slicing helpers below are finalized against the ACTUAL fetched files after the
corpus subagent reports (chapter markers / book markers / date lines may vary).
Run: venv/bin/python -m experiments.sediment_all
"""
import asyncio
import json
import os
import time

from society.run import _build_llm_and_embed
from society.history_extract import extract_history, _split_by_chapters

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(BASE, "scenarios", "sources")
OUT = os.path.join(BASE, "scenarios")
CONFIG = os.path.join(BASE, "config.json")
# per-scenario concurrency; 4 scenarios in parallel -> 4*PER_CONC in flight
PER_CONC = 8


# --- source slicing (finalized post-fetch) ------------------------------------
def _first_n_chapters(text: str, n: int) -> str:
    chs = _split_by_chapters(text)
    if not chs:
        raise ValueError("no 第X回 chapter headings found")
    return "".join(chs[:n])


def three_kingdoms_sediment() -> str:
    a = open(os.path.join(SRC, "three_kingdoms_ch01-10.txt"), encoding="utf-8").read()
    b = open(os.path.join(SRC, "three_kingdoms_ch11-60.txt"), encoding="utf-8").read()
    return _first_n_chapters(a + "\n" + b, 40)


def red_chamber_sediment() -> str:
    t = open(os.path.join(SRC, "dream_red_chamber_ch01-80.txt"), encoding="utf-8").read()
    return _first_n_chapters(t, 40)


def war_peace_sediment() -> str:
    # Fetched file is Volume I (Books ONE/TWO/THREE, all 1805). Sediment span =
    # Books 1-2 ONLY; Book 3 is the held-out reference (kept in the file for
    # event extraction). Slice from "BOOK ONE:" up to (not including) "BOOK THREE:".
    import re
    lines = open(os.path.join(SRC, "war_and_peace_vol1-3.txt"), encoding="utf-8").read().splitlines()
    starts = [i for i, l in enumerate(lines) if re.match(r"^BOOK (ONE|THREE):", l)]
    if len(starts) < 2:
        raise ValueError("war_peace_sediment: could not locate BOOK ONE / BOOK THREE markers")
    return "\n".join(lines[starts[0]:starts[1]])


def russia_ukraine_sediment() -> str:
    # sediment span = dated lines up to 2024-04 inclusive
    raw = open(os.path.join(SRC, "russia_ukraine_timeline.txt"), encoding="utf-8").read()
    raw = raw.replace("&nbsp;", " ").replace("&amp;", "&")
    lines = raw.splitlines()
    keep = [ln for ln in lines if ln[:2] == "20" and ln[:7] <= "2024-04"]
    return "\n".join(keep)


SCENARIOS = [
    {"name": "three_kingdoms", "lang": "zh", "slice": three_kingdoms_sediment},
    {"name": "red_chamber", "lang": "zh", "slice": red_chamber_sediment},
    {"name": "war_and_peace", "lang": "en", "slice": war_peace_sediment},
    {"name": "russia_ukraine", "lang": "en", "slice": russia_ukraine_sediment},
]


async def sediment_one(spec: dict) -> dict:
    name = spec["name"]
    out_yaml = os.path.join(OUT, f"{name}.yaml")
    # fresh client per scenario so usage/cost is isolated
    llm, embed_fn = _build_llm_and_embed(CONFIG)
    llm.max_concurrency = PER_CONC
    llm._semaphore = asyncio.Semaphore(PER_CONC)
    text = spec["slice"]()
    # Reuse the already-reviewed registry (from experiments.extract_roles) so
    # Pass 1 is skipped and we sediment against the exact same role set.
    registry_path = out_yaml + ".registry.json"
    registry = None
    if os.path.exists(registry_path):
        with open(registry_path, "r", encoding="utf-8") as rf:
            registry = json.load(rf)
    t0 = time.time()
    try:
        cfg = await extract_history(
            text, llm, out_yaml, embed_fn=embed_fn, language=spec["lang"],
            detail="atomic", registry=registry,
        )
        dt = time.time() - t0
        ltm = json.load(open(out_yaml + ".ltm.json"))
        usage = llm.usage()
        rec = {
            "name": name, "ok": True, "wall_clock_s": round(dt, 1),
            "memories": len(ltm), "usage": usage.get("_total"),
            "warnings": len(cfg.get("_warnings", [])),
        }
    except Exception as exc:  # keep other scenarios running
        rec = {"name": name, "ok": False, "error": repr(exc), "wall_clock_s": round(time.time() - t0, 1)}
    json.dump(rec, open(os.path.join(OUT, f"{name}.sediment_stats.json"), "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rec, ensure_ascii=False))
    return rec


async def main():
    results = await asyncio.gather(*(sediment_one(s) for s in SCENARIOS))
    print("=== SEDIMENT SUMMARY ===")
    for r in results:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
