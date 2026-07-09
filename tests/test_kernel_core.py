import asyncio
from society.actions import Action, Message
from society.agent import Agent
from society.brains.rule_brain import RuleBrain
from society.events import EventLog
from society.kernel import Kernel
from society.stm import STM
from society.worldmap import WorldMap


def make_char(aid, loc, fn=None, goals=None):
    stm = STM(status={"location": loc}, goals=goals or [])
    return Agent(aid, "character", RuleBrain(fn=fn), stm)


def make_env(aid):
    return Agent(aid, "environment", RuleBrain(), STM())


def society(agents, edges=None):
    envs = [a.id for a in agents if a.kind == "environment"]
    k = Kernel({a.id: a for a in agents}, WorldMap(envs, edges=edges, default_distance=3),
               EventLog(None))
    return k


async def test_sleeping_agents_not_scheduled_and_quiescence():
    calls = []
    a = make_char("a", "hall", fn=lambda v: calls.append(v) or Action("noop"))
    k = society([a, make_env("hall")])
    summary = await k.run(max_ticks=5)
    assert calls == []                        # empty inbox + empty goals → never scheduled
    assert summary["stop_reason"] == "quiescent" and summary["ticks_run"] == 0


async def test_goal_keeps_agent_awake_and_max_ticks():
    a = make_char("a", "hall", fn=lambda v: Action("noop"), goals=["exist"])
    k = society([a, make_env("hall")])
    summary = await k.run(max_ticks=3)
    assert summary["stop_reason"] == "max_ticks"
    acts = [e for e in k.event_log.all() if e["kind"] == "action" and e["agent"] == "a"]
    assert len(acts) == 3                     # exactly one action per tick


async def test_message_visible_next_tick_and_wakes_sleeper():
    seen = []

    def bfn(v):
        seen.append((v["tick"], v["inbox_size"]))
        return Action("pop_message")

    b = make_char("b", "hall", fn=bfn)

    def afn(v):
        return Action("say", {"targets": ["b"], "content": "hi"}) if v["tick"] == 0 else Action("noop")

    a = make_char("a", "hall", fn=afn, goals=["talk"])
    k = society([a, b, make_env("hall")])
    await k.run(max_ticks=4)
    assert seen and seen[0][0] == 1 and seen[0][1] == 1    # b first scheduled at t=1 with 1 msg


async def test_wait_timeout_wakes_and_fast_forward():
    ticks_seen = []

    def fn(v):
        ticks_seen.append(v["tick"])
        return Action("wait", {"timeout_ticks": 5}) if v["tick"] == 0 else Action("noop")

    a = make_char("a", "hall", fn=fn, goals=["g"])
    k = society([a, make_env("hall")])
    await k.run(max_ticks=20)
    # t0 acts then waits; no other work → fast-forward to t5, acts again (noop), then goals nonempty keeps it awake
    assert ticks_seen[0] == 0 and ticks_seen[1] == 5


async def test_pop_and_registers():
    log = []

    def fn(v):
        step = len(log)
        log.append(1)
        return [Action("push_goal", {"text": "small"}),
                Action("update_status", {"key": "mood", "value": "calm"}),
                Action("pop_goal"),
                Action("noop")][min(step, 3)]

    a = make_char("a", "hall", fn=fn, goals=["big"])
    k = society([a, make_env("hall")])
    await k.run(max_ticks=3)
    assert a.stm.goals.items() == ["big"]
    assert a.stm.status.get("mood") == "calm"
