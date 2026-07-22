"""Apply per-scenario memory-count thresholds to select the ACTIVE simulated cast.

A character participates in simulation iff it owns MORE than T memories (T chosen
per scenario from its memory-count distribution). Below-threshold characters are
dropped from the agent list but their memories REMAIN in the LTM sidecar (they stay
owners of the shared record — just never scheduled). Environments and info_carriers
are kept as-is. Writes a curated <name>.sim.yaml (original <name>.yaml untouched).

Run: venv/bin/python -m experiments.select_cast
"""
import json
import os
from collections import Counter

import yaml

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SC = os.path.join(BASE, "scenarios")

# per-scenario thresholds (character kept iff memory_count > T), locked with user
THRESHOLDS = {
    "three_kingdoms": 50,
    "red_chamber": 100,
    "war_and_peace": 20,
    "russia_ukraine": 3,
}


def char_mem_counts(ltm, char_ids):
    cnt = Counter()
    for e in ltm:
        for o in e.get("owners", []):
            if o in char_ids:
                cnt[o] += 1
    return cnt


def curate(name, T):
    yml = os.path.join(SC, f"{name}.yaml")
    d = yaml.safe_load(open(yml, encoding="utf-8"))
    ltm = json.load(open(os.path.join(SC, f"{name}.yaml.ltm.json"), encoding="utf-8"))
    agents = d["agents"]
    char_ids = {a["id"] for a in agents if a.get("kind") == "character"}
    cnt = char_mem_counts(ltm, char_ids)

    keep_char = {cid for cid in char_ids if cnt.get(cid, 0) > T}
    dropped = char_ids - keep_char

    kept_agents = [
        a for a in agents
        if a.get("kind") != "character" or a["id"] in keep_char
    ]
    # clean kickoff: strip dropped ids from `to`, drop now-empty messages
    kickoff = d.get("kickoff") or []
    new_kickoff = []
    for msg in kickoff:
        to = [t for t in (msg.get("to") or []) if t not in dropped]
        if to:
            m = dict(msg)
            m["to"] = to
            new_kickoff.append(m)

    d["agents"] = kept_agents
    d["kickoff"] = new_kickoff
    out = os.path.join(SC, f"{name}.sim.yaml")
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(d, f, allow_unicode=True, sort_keys=False)

    kinds = Counter(a.get("kind") for a in kept_agents)
    return {
        "name": name, "T": T,
        "chars_total": len(char_ids), "chars_active": len(keep_char),
        "chars_dropped": len(dropped),
        "environments": kinds.get("environment", 0),
        "info_carriers": kinds.get("info_carrier", 0),
        "kickoff_msgs": len(new_kickoff),
        "out": os.path.relpath(out, BASE),
    }


def main():
    for name, T in THRESHOLDS.items():
        r = curate(name, T)
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
