import os

from society.actions import Action, Message
from society.agent import Agent
from society.brains.rule_brain import RuleBrain
from society.events import EventLog
from society.kernel import Kernel
from society.persistence import load_checkpoint, restore_society, save_checkpoint
from society.scenario import build_society
from society.stm import STM
from tests.helpers import FakeLLM, afake_embed

SKILL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "society", "skills"
)


def char(aid, loc, name=None, fn=None, goals=None, archived=False):
    stm = STM(status={"location": loc}, goals=goals or [])
    return Agent(aid, "character", RuleBrain(fn=fn), stm, name=name, archived=archived)


def env(aid, name=None):
    return Agent(aid, "environment", RuleBrain(), STM(), name=name)


def build(agents, edges=None):
    from society.worldmap import WorldMap

    envs = [a.id for a in agents if a.kind == "environment"]
    return Kernel(
        {a.id: a for a in agents},
        WorldMap(envs, edges=edges, default_distance=3),
        EventLog(None),
    )


# ----------------------------------------------------------------------
# 1. archived agents are never scheduled
# ----------------------------------------------------------------------

async def test_archived_never_scheduled():
    hall = env("hall")
    guan_yu = char("guan_yu", "hall", goals=["revenge"], archived=True)
    guan_yu.stm.inbox.put_nowait(
        Message(id="m1", sender="cao_cao", recipients=["guan_yu"], kind="say",
                content="hi", tick_sent=0)
    )
    k = build([hall, guan_yu])

    assert k.is_eligible(guan_yu) is False

    await k.run(max_ticks=3)

    actions = [
        e for e in k.event_log.all()
        if e["kind"] == "action" and e["agent"] == "guan_yu"
    ]
    assert actions == []


# ----------------------------------------------------------------------
# 2. archived agents are excluded from presence and cannot be observed
# ----------------------------------------------------------------------

async def test_archived_not_in_presence_and_not_observable():
    hall = env("hall")
    guan_yu = char("guan_yu", "hall", archived=True)
    amy = char("amy", "hall")
    k = build([hall, guan_yu, amy])

    assert "guan_yu" not in k.presence.get("hall", set())
    assert k.presence["hall"] == {"amy"}

    r_env = await k.execute(amy, Action("observe", {"target": "hall"}))
    assert r_env.ok is True
    occupant_ids = [o["id"] for o in r_env.data["occupants"]]
    assert "guan_yu" not in occupant_ids

    r_char = await k.execute(amy, Action("observe", {"target": "guan_yu"}))
    assert r_char.ok is False
    assert "archived" in r_char.error or "已故" in r_char.error


# ----------------------------------------------------------------------
# 3. say to an archived target fails, even if colocated on paper
# ----------------------------------------------------------------------

async def test_say_to_archived_fails():
    hall = env("hall")
    # guan_yu's stored status.location matches amy's -- "colocated on paper"
    # -- but archived agents are never in the presence index and must still
    # be rejected explicitly.
    guan_yu = char("guan_yu", "hall", archived=True)
    amy = char("amy", "hall")
    k = build([hall, guan_yu, amy])

    r = await k.execute(amy, Action("say", {"targets": ["guan_yu"], "content": "久违了"}))

    assert r.ok is False
    assert "guan_yu" in r.error


# ----------------------------------------------------------------------
# 4. goal_hint appears only when the goal stack is empty
# ----------------------------------------------------------------------

async def test_goal_hint_only_when_stack_empty():
    captured = {}

    def make_fn(key):
        def fn(view):
            captured[key] = view
            return Action("wait", {})
        return fn

    empty_agent = char("empty_agent", "hall", goals=[], fn=make_fn("empty"))
    goal_agent = char("goal_agent", "hall", goals=["fundamental goal"], fn=make_fn("goal"))
    hall = env("hall")
    k = build([empty_agent, goal_agent, hall])

    # empty_agent has an empty goal stack AND an empty inbox, so it would
    # never be scheduled at all -- give it a kickoff message so it wakes.
    k.send(
        Message(id="kick", sender="scenario", recipients=["empty_agent"],
                kind="system", content="开始", tick_sent=-1)
    )

    await k.run(max_ticks=3)

    assert "empty" in captured and "goal" in captured

    assert "goal_hint" in captured["empty"]
    assert captured["empty"]["goal_hint"] == Kernel._GOAL_HINT_ZH

    assert "goal_hint" not in captured["goal"]


# ----------------------------------------------------------------------
# 5. archived flag survives a checkpoint save/restore roundtrip
# ----------------------------------------------------------------------

CKPT_SCEN = {
    "scenario": "archived_ckpt_test", "language": "zh",
    "defaults": {"stats_interval": 100, "distance": 3},
    "agents": [
        {"id": "hall", "kind": "environment", "brain": "rule"},
        {"id": "amy", "kind": "character", "brain": "rule",
         "status": {"location": "hall"}, "goals": ["chat"]},
        {"id": "guan_yu", "kind": "character", "brain": "rule",
         "status": {"location": "hall"}, "archived": True},
    ],
    "map": {"default_distance": 3},
}


async def test_archived_survives_checkpoint(tmp_path):
    kernel = await build_society(
        CKPT_SCEN, llm=FakeLLM(), embed_fn=afake_embed, event_log=EventLog(None)
    )
    assert kernel.agents["guan_yu"].archived is True
    assert "guan_yu" not in kernel.presence.get("hall", set())

    ckpt_path = str(tmp_path / "ckpt.json")
    save_checkpoint(kernel, ckpt_path)
    ckpt = load_checkpoint(ckpt_path)
    assert ckpt["agents"]["guan_yu"]["archived"] is True

    restored = await restore_society(
        ckpt, llm=FakeLLM(), embed_fn=afake_embed, event_log=EventLog(None)
    )

    assert restored.agents["guan_yu"].archived is True
    assert restored.is_eligible(restored.agents["guan_yu"]) is False
    assert "guan_yu" not in restored.presence.get("hall", set())


# ----------------------------------------------------------------------
# 6. skill docs mention the bootstrap-reflection pipeline
# ----------------------------------------------------------------------

def test_skill_mentions_bootstrap():
    zh_path = os.path.join(SKILL_DIR, "actions_skill_zh.md")
    en_path = os.path.join(SKILL_DIR, "actions_skill_en.md")

    with open(zh_path, "r", encoding="utf-8") as f:
        zh_text = f.read()
    with open(en_path, "r", encoding="utf-8") as f:
        en_text = f.read()

    assert "开局自省" in zh_text
    assert "Bootstrap" in en_text
