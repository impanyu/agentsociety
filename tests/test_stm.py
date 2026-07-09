import asyncio
from society.stm import FifoCache, GoalStack, StatusRegister, STM

def test_fifo_evicts_oldest_pair():
    f = FifoCache(maxlen=2)
    f.append({"name": "a1"}, {"ok": True})
    f.append({"name": "a2"}, {"ok": True})
    f.append({"name": "a3"}, {"ok": True})
    assert len(f) == 2
    assert [a["name"] for a, _ in f.items()] == ["a2", "a3"]

def test_goal_stack_bottom_is_fundamental():
    g = GoalStack()
    g.push("fundamental"); g.push("immediate")
    assert g.items() == ["fundamental", "immediate"]
    assert g.peek() == "immediate"
    assert g.pop() == "immediate"
    assert g.pop() == "fundamental"
    assert g.pop() is None and g.empty()

def test_goal_replace_top():
    g = GoalStack(); g.push("a"); g.replace("b")
    assert g.items() == ["b"]
    g2 = GoalStack(); g2.replace("x")           # empty → push
    assert g2.items() == ["x"]

def test_status_public_private():
    s = StatusRegister({"mood": "sad", "appearance": "tall", "location": "hall"})
    assert "mood" not in s.public_view()
    assert s.public_view() == {"appearance": "tall", "location": "hall"}
    assert s.get("mood") == "sad"
    s2 = StatusRegister({"mood": "happy"}, private_keys=set())  # override: mood public
    assert s2.public_view() == {"mood": "happy"}

def test_stm_wiring_and_initial_goals():
    stm = STM(fifo_size=3, status={"location": "hall"}, goals=["deep", "top"])
    assert stm.goals.items() == ["deep", "top"]
    assert isinstance(stm.inbox, asyncio.Queue)
    assert stm.status.get("location") == "hall"
