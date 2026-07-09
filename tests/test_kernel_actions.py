import uuid
from society.actions import Action
from society.agent import Agent
from society.brains.rule_brain import RuleBrain
from society.brains.retrieval_brain import RetrievalBrain
from society.events import EventLog
from society.kernel import Kernel
from society.ltm import SharedMemory
from society.stm import STM
from society.worldmap import WorldMap
from tests.helpers import FakeLLM, afake_embed


def build(agents, llm=None, shared=None, edges=None):
    envs = [a.id for a in agents if a.kind == "environment"]
    return Kernel({a.id: a for a in agents},
                  WorldMap(envs, edges=edges, default_distance=4),
                  EventLog(None), shared_memory=shared, llm=llm)


def char(aid, loc):
    return Agent(aid, "character", RuleBrain(), STM(status={"location": loc}))


async def test_say_rejects_cross_location():
    a, b = char("a", "hall"), char("b", "garden")
    k = build([a, b, Agent("hall", "environment", RuleBrain(), STM()),
               Agent("garden", "environment", RuleBrain(), STM())])
    r = await k.execute(a, Action("say", {"targets": ["b"], "content": "hi"}))
    assert r.ok is False and "b" in r.error
    r2 = await k.execute(a, Action("say", {"targets": ["ghost"], "content": "hi"}))
    assert r2.ok is False


async def test_observe_environment_lists_occupants():
    a, b = char("a", "hall"), char("b", "hall")
    k = build([a, b, Agent("hall", "environment", RuleBrain(), STM(status={"desc": "大厅"}))])
    r = await k.execute(a, Action("observe", {"target": "hall"}))
    assert r.ok and [o["id"] for o in r.data["occupants"]] == ["b"]


async def test_observe_character_requires_colocation():
    a, b = char("a", "hall"), char("b", "garden")
    k = build([a, b, Agent("hall", "environment", RuleBrain(), STM()),
               Agent("garden", "environment", RuleBrain(), STM())])
    assert (await k.execute(a, Action("observe", {"target": "b"}))).ok is False


async def test_read_info_carrier():
    a = char("a", "hall")
    book = Agent("book", "info_carrier", RetrievalBrain("宝玉衔玉而生。"), STM(status={"location": "hall"}))
    k = build([a, book, Agent("hall", "environment", RuleBrain(), STM())])
    r = await k.execute(a, Action("read", {"target": "book", "query": "宝玉"}))
    assert r.ok and "宝玉衔玉而生" in r.data


async def test_move_sets_transit_and_arrival():
    a = char("a", "hall")
    k = build([a, Agent("hall", "environment", RuleBrain(), STM()),
               Agent("garden", "environment", RuleBrain(), STM())],
              edges=[("hall", "garden", 2)])
    r = await k.execute(a, Action("move", {"destination": "garden"}))
    assert r.ok and a.transit == {"dest": "garden", "arrive_at": 2}   # tick=0 + distance 2
    assert "a" not in k.presence.get("hall", set())
    await k.run(max_ticks=5)                                          # arrival processed
    assert a.location() == "garden" and a.transit is None
    arrival_events = [e for e in k.event_log.all()
                       if e["kind"] == "system" and e.get("event") == "arrival"]
    assert any(e["agent"] == "a" and e["dest"] == "garden" for e in arrival_events)


async def test_move_rejects_unconnected_or_nonenv():
    a = char("a", "hall")
    k = build([a, Agent("hall", "environment", RuleBrain(), STM()),
               Agent("tower", "environment", RuleBrain(), STM())],
              edges=[("hall", "tower", 3)])
    assert (await k.execute(a, Action("move", {"destination": "a"}))).ok is False
    k2 = build([char("x", "hall"), Agent("hall", "environment", RuleBrain(), STM()),
                Agent("far", "environment", RuleBrain(), STM())], edges=[])
    # fully_connected default True → give explicit non-connected map:
    from society.worldmap import WorldMap as WM
    k2.worldmap = WM(["hall", "far"], edges=[], fully_connected=False)
    assert (await k2.execute(k2.agents["x"], Action("move", {"destination": "far"}))).ok is False


async def test_memory_actions_roundtrip():
    shared = SharedMemory(afake_embed, llm=None, collection_name=f"t_{uuid.uuid4().hex[:8]}")
    a = char("a", "hall")
    k = build([a, Agent("hall", "environment", RuleBrain(), STM())], shared=shared)
    r = await k.execute(a, Action("remember", {"text": "花园着火"}))
    assert r.ok and r.data[0]["text"] == "花园着火"
    r2 = await k.execute(a, Action("recall", {"query": "花园着火"}))
    assert r2.ok and r2.data[0]["text"] == "花园着火"
    r3 = await k.execute(a, Action("forget", {"memory_id": r.data[0]["id"]}))
    assert r3.ok
    assert (await k.execute(a, Action("recall", {"query": "花园"}))).data == []


async def test_think_uses_llm_bucket():
    llm = FakeLLM(responses=["结论:该走了"])
    a = char("a", "hall")
    k = build([a, Agent("hall", "environment", RuleBrain(), STM())], llm=llm)
    r = await k.execute(a, Action("think", {"question": "下一步?"}))
    assert r.ok and "结论" in r.data and llm.calls[0][0] == "think"


async def test_act_on_rule_env_pushes_env_result():
    a = char("a", "hall")
    env = Agent("hall", "environment",
                RuleBrain(act_on_fn=lambda actor, desc, view: f"{actor}打开了{desc},门开了"),
                STM())
    k = build([a, env])
    r = await k.execute(a, Action("act_on", {"target": "hall", "description": "大门"}))
    assert r.ok
    k.deliver_pending()      # public delivery step
    assert a.stm.inbox.qsize() == 1
    msg = a.stm.inbox.get_nowait()
    assert msg.kind == "env_result" and "门开了" in msg.content
