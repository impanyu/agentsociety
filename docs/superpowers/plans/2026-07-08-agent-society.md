# AgentSociety Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the AgentSociety multi-agent framework (phase 1 infrastructure) per the spec at `docs/specs/2026-07-08-agent-society-design.md`.

**Architecture:** Tick-barrier kernel schedules awake agents (one action-result cycle per tick, concurrent within a tick), routes messages with t+1 delivery, maintains a presence index and map-based movement with transit time. Agents = STM (FIFO/goal-stack/status/inbox) + pluggable Brain. Shared LTM is one Chroma collection with owner sets, a normalize gate (char limit + atomicity), and consensus insert/forget/revise. Offline tools: scenario loader, novel extractor, screenplay generator, periodic stats.

**Tech Stack:** Python 3.14 (`venv/bin/python`), asyncio, chromadb 1.5.9, pyyaml, httpx, pytest + pytest-asyncio.

## Global Constraints

- Working dir for ALL commands: `/Users/ypan12/git_repo/bookworld_paper/agent_society`. Test command: `venv/bin/python -m pytest`.
- This directory IS a git repo. Commit at the end of each task with the given message.
- Package layout: code in `society/`, tests in `tests/`. No dependency on BookWorld/GMemory code.
- Defaults (from spec §12, all configurable): FIFO=20 pairs; memory_max_chars=80; consensus sim threshold=0.86, top-k=5; map default distance=20 ticks; stats_interval=10; LLM concurrency=16; retries=3; message delivery delay=1 tick; recall top_k=5.
- Messages sent at tick t become visible at t+1. Sleeping = inbox empty AND goal stack empty → not scheduled.
- All LLM/network access goes through `society/llm.py` (budget buckets). Tests NEVER hit the network — use `tests/helpers.py` fakes.
- Status key `location` is reserved: only kernel/move may change it. Public status keys default: appearance, clothing, location; `mood` private by default.
- Every action event logged with: tick, seq, agent, action name+params, result, agent location.
- Chroma collections use cosine space and explicit embeddings (`get_or_create_collection(name, metadata={"hnsw:space": "cosine"})`), unique per-instance names (uuid suffix) for in-memory isolation.

---

### Task 1: Scaffold + STM

**Files:**
- Create: `pytest.ini`, `society/__init__.py`, `society/stm.py`, `tests/__init__.py`, `tests/test_stm.py`, `config.json.example`

**Interfaces (Produces):**
```python
# society/stm.py
class FifoCache:                      # (action:dict, result:dict) pairs, most-recent-last
    def __init__(self, maxlen: int = 20)
    def append(self, action: dict, result: dict) -> None
    def items(self) -> list[tuple[dict, dict]]
    def __len__(self)
class GoalStack:                      # index 0 = bottom = most fundamental
    def push(self, text: str); def pop(self) -> str | None
    def replace(self, text: str)      # replace top; if empty, same as push
    def peek(self) -> str | None; def items(self) -> list[str]; def empty(self) -> bool
class StatusRegister:
    def __init__(self, initial: dict | None = None, private_keys: set | None = None)
        # default private: {"mood"}; "location" always public
    def set(self, key, value); def remove(self, key); def get(self, key, default=None)
    def public_view(self) -> dict; def all(self) -> dict
class STM:
    def __init__(self, fifo_size=20, status=None, private_keys=None, goals: list[str] | None = None)
        # goals given bottom→top
    fifo: FifoCache; goals: GoalStack; status: StatusRegister; inbox: asyncio.Queue
```

- [ ] **Step 1: scaffold.** Write `pytest.ini`:
```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```
`society/__init__.py` and `tests/__init__.py` empty. `config.json.example`:
```json
{"api_key": "sk-...", "base_url": "https://api.openai.com/v1",
 "chat_model": "gpt-4o-mini", "embed_model": "text-embedding-3-small",
 "max_concurrency": 16, "max_calls": null, "max_tokens": null}
```

- [ ] **Step 2: failing tests.** `tests/test_stm.py`:
```python
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
```

- [ ] **Step 3:** run `venv/bin/python -m pytest tests/test_stm.py -q` → FAIL (import error).
- [ ] **Step 4:** implement `society/stm.py` exactly per the interface (FifoCache backed by `collections.deque(maxlen=...)` of pairs; StatusRegister default `private_keys={"mood"}` when None, public_view excludes private keys but always includes `location` if set).
- [ ] **Step 5:** run again → all pass. Full suite `venv/bin/python -m pytest -q` green.
- [ ] **Step 6:** `git add -A && git commit -m "feat: scaffold + STM (fifo, goal stack, status register)"`

---

### Task 2: Actions, results, messages, registry

**Files:**
- Create: `society/actions.py`, `tests/test_actions.py`

**Interfaces (Produces):**
```python
# society/actions.py
from dataclasses import dataclass, field
@dataclass
class Action:
    name: str
    params: dict = field(default_factory=dict)
@dataclass
class ActionResult:
    ok: bool
    data: object = None
    error: str | None = None
    def to_dict(self) -> dict
@dataclass
class Message:
    id: str; sender: str; recipients: list; kind: str; content: str
    tick_sent: int; correlation_id: str | None = None
    def to_dict(self) -> dict

SYNC_ACTIONS: set[str]    # {"pop_message","peek_inbox","think","conclude","push_goal","pop_goal",
                          #  "replace_goal","update_status","remove_status","remember","recall",
                          #  "forget","revise_memory","observe","read","move","wait","noop"}
ASYNC_ACTIONS: set[str]   # {"say","gesture","act_on"}
REQUIRED_PARAMS: dict[str, list[str]]   # e.g. {"say": ["targets","content"], "think": ["question"],
                          #  "move": ["destination"], "recall": ["query"], "remember": ["text"], ...}
def validate_action(action: Action) -> str | None   # None if valid, else error string
def parse_action(obj: dict) -> Action                # {"action": name, "params": {...}} → Action; raises ValueError
```
Param requirements: pop_message/peek_inbox/pop_goal/noop → []; think→["question"]; conclude→["text"]; push_goal/replace_goal→["text"]; update_status→["key","value"]; remove_status→["key"]; remember→["text"]; recall→["query"]; forget→["memory_id"]; revise_memory→["memory_id","new_text"]; observe→["target"]; read→["target","query"]; move→["destination"]; wait→[] ("timeout_ticks" optional); say→["targets","content"]; gesture→["targets","description"]; act_on→["target","description"]. `validate_action` also rejects unknown names, non-list `targets`, and `update_status(key="location", ...)` (reserved).

- [ ] **Step 1: failing tests.** `tests/test_actions.py`:
```python
import pytest
from society.actions import (Action, ActionResult, Message, SYNC_ACTIONS, ASYNC_ACTIONS,
                             validate_action, parse_action)

def test_action_sets_are_disjoint_and_complete():
    assert SYNC_ACTIONS & ASYNC_ACTIONS == set()
    assert "move" in SYNC_ACTIONS and "say" in ASYNC_ACTIONS
    assert len(SYNC_ACTIONS | ASYNC_ACTIONS) == 21

def test_validate_ok_and_missing_param():
    assert validate_action(Action("say", {"targets": ["b"], "content": "hi"})) is None
    err = validate_action(Action("say", {"content": "hi"}))
    assert err and "targets" in err

def test_validate_rejects_unknown_and_reserved_location():
    assert validate_action(Action("fly", {})) is not None
    assert validate_action(Action("update_status", {"key": "location", "value": "x"})) is not None
    assert validate_action(Action("say", {"targets": "b", "content": "hi"})) is not None  # not a list

def test_parse_action_roundtrip_and_errors():
    a = parse_action({"action": "think", "params": {"question": "why"}})
    assert a.name == "think" and a.params["question"] == "why"
    with pytest.raises(ValueError):
        parse_action({"params": {}})
    with pytest.raises(ValueError):
        parse_action({"action": "fly", "params": {}})   # parse validates too

def test_result_and_message_serialization():
    r = ActionResult(ok=False, error="nope")
    assert r.to_dict() == {"ok": False, "data": None, "error": "nope"}
    m = Message(id="1", sender="a", recipients=["b"], kind="say", content="hi", tick_sent=3)
    d = m.to_dict()
    assert d["recipients"] == ["b"] and d["tick_sent"] == 3 and d["correlation_id"] is None
```
- [ ] **Step 2:** run → FAIL. **Step 3:** implement per interface. **Step 4:** run → pass; full suite green.
- [ ] **Step 5:** `git add -A && git commit -m "feat: action/result/message types + registry + validation"`

---

### Task 3: Event log

**Files:** Create `society/events.py`, `tests/test_events.py`

**Interfaces (Produces):**
```python
class EventLog:
    def __init__(self, path: str | None)          # None → in-memory only
    def append(self, tick: int, kind: str, agent: str, payload: dict) -> int  # returns seq (0-based, global)
    def all(self) -> list[dict]                   # each: {"seq","tick","kind","agent", **payload}
    @staticmethod
    def load(path: str) -> list[dict]
```
`kind ∈ {"action","message","system"}`. Appends are immediately flushed to the JSONL file (one JSON object per line, ensure_ascii=False).

- [ ] **Step 1: failing tests.** `tests/test_events.py`:
```python
import json
from society.events import EventLog

def test_append_assigns_monotonic_seq_and_persists(tmp_path):
    p = str(tmp_path / "events.jsonl")
    log = EventLog(p)
    s0 = log.append(0, "action", "alice", {"action": {"name": "noop"}})
    s1 = log.append(0, "message", "kernel", {"content": "hi"})
    s2 = log.append(1, "action", "bob", {"action": {"name": "wait"}})
    assert (s0, s1, s2) == (0, 1, 2)
    lines = [json.loads(l) for l in open(p, encoding="utf-8")]
    assert len(lines) == 3 and lines[2]["tick"] == 1 and lines[2]["agent"] == "bob"
    assert EventLog.load(p) == log.all()

def test_in_memory_mode():
    log = EventLog(None)
    log.append(5, "system", "kernel", {"note": "静止"})
    assert log.all()[0]["tick"] == 5 and log.all()[0]["note"] == "静止"
```
- [ ] **Steps 2-4:** fail → implement → pass, suite green.
- [ ] **Step 5:** `git commit -am "feat: JSONL event log with global seq"` (use `git add -A` first).

---

### Task 4: LLM client + embeddings + test fakes

**Files:** Create `society/llm.py`, `society/embeddings.py`, `tests/helpers.py`, `tests/test_llm.py`

**Interfaces (Produces):**
```python
# society/llm.py
class BudgetExceeded(Exception): ...
class LLMClient:
    def __init__(self, api_key, base_url, chat_model, *, max_concurrency=16,
                 max_calls=None, max_tokens=None, transport=None, retries=3)
        # transport: async callable(payload_dict)->response_dict, injected in tests.
        # Default transport POSTs {base_url}/chat/completions via httpx.AsyncClient.
    async def chat(self, prompt: str, system: str | None = None, bucket: str = "decide") -> str
    def usage(self) -> dict     # {bucket: {"calls": int, "tokens": int}, "_total": {...}}
class EmbeddingClient:
    def __init__(self, api_key, base_url, embed_model, transport=None)
    async def embed(self, texts: list[str]) -> list[list[float]]

# tests/helpers.py
class FakeLLM:                      # duck-types LLMClient.chat/usage
    def __init__(self, responses=None, fn=None)   # responses: list[str] popped in order; or fn(prompt, system)->str
    async def chat(self, prompt, system=None, bucket="decide") -> str   # records (bucket, prompt) in self.calls
    def usage(self) -> dict
def fake_embed(texts: list[str]) -> list[list[float]]
    # deterministic 8-dim: seed a random.Random with the text's md5; identical text → identical vector
async def afake_embed(texts): return fake_embed(texts)
```
Behavior: `chat` acquires the semaphore, enforces budget BEFORE the call (`max_calls`/`max_tokens` over `_total`; raise BudgetExceeded), retries transport exceptions with exponential backoff (0.5·2^n s, max `retries`), counts tokens from response `usage.total_tokens` (0 if absent), tallies per-bucket.

- [ ] **Step 1: failing tests.** `tests/test_llm.py`:
```python
import pytest
from society.llm import LLMClient, BudgetExceeded

def make_client(**kw):
    async def transport(payload):
        return {"choices": [{"message": {"content": "ok:" + payload["messages"][-1]["content"]}}],
                "usage": {"total_tokens": 7}}
    return LLMClient("k", "http://x", "m", transport=transport, **kw)

async def test_chat_returns_content_and_counts_buckets():
    c = make_client()
    out = await c.chat("hello", bucket="think")
    assert out == "ok:hello"
    u = c.usage()
    assert u["think"]["calls"] == 1 and u["think"]["tokens"] == 7
    assert u["_total"]["calls"] == 1

async def test_budget_max_calls_enforced():
    c = make_client(max_calls=1)
    await c.chat("a")
    with pytest.raises(BudgetExceeded):
        await c.chat("b")

async def test_retry_then_success():
    attempts = {"n": 0}
    async def flaky(payload):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("boom")
        return {"choices": [{"message": {"content": "done"}}], "usage": {}}
    c = LLMClient("k", "http://x", "m", transport=flaky, retries=3)
    assert await c.chat("x") == "done"
    assert attempts["n"] == 3

async def test_fakes_are_deterministic():
    from tests.helpers import FakeLLM, fake_embed
    f = FakeLLM(responses=["r1", "r2"])
    assert await f.chat("p") == "r1" and await f.chat("p") == "r2"
    assert f.calls[0][1] == "p"
    assert fake_embed(["同一句"]) == fake_embed(["同一句"])
    assert fake_embed(["a"]) != fake_embed(["b"])
```
- [ ] **Steps 2-4:** fail → implement (`retries` sleeps via `asyncio.sleep`; in tests backoff small is fine—use 0.01 base when `transport` injected... simpler: `backoff_base` ctor arg default 0.5, tests pass 0.0) → adjust test `make_client` to pass `backoff_base=0.0` if you add it (do add it). Suite green.
- [ ] **Step 5:** commit `"feat: async LLM/embedding clients with budget buckets + test fakes"`.

---

### Task 5: Shared LTM — normalize gate + consensus

**Files:** Create `society/ltm.py`, `tests/test_ltm.py`

**Interfaces (Produces):**
```python
class SharedMemory:
    def __init__(self, embed_fn, llm=None, *, max_chars=80, sim_threshold=0.86,
                 top_k=5, collection_name="agent_society_ltm")
        # embed_fn: async callable(list[str])->list[vector]; llm: .chat duck-type or None
    async def remember(self, agent_id: str, text: str, tick: int = 0,
                       source: str = "runtime") -> list[dict]
        # → per stored/merged entry: {"id","text","merged":bool,"owners":list}
    async def recall(self, agent_id: str, query: str, top_k: int = 5) -> list[dict]  # [{"id","text"}]
    def forget(self, agent_id: str, memory_id: str) -> bool     # removed-from-owners? (False if not owner/absent)
    async def revise(self, agent_id, memory_id, new_text, tick=0) -> list[dict]
    def all_entries(self) -> list[dict]   # {"id","text","owners":sorted list,"meta":dict}
    def stats(self) -> dict               # {"total","shared","ratio"}  shared = owners>=2
    async def _normalize(self, text) -> list[str]        # exposed for tests
```
**Normalize gate:** trigger iff `len(text) > max_chars` OR heuristic non-atomic: count of sentence terminators (。！？；.;) > 1, or any connective in ("然后","并且","而且","同时","接着"," and "," then "). If triggered: llm→ prompt returns JSON array of atomic strings (parse tolerantly: find `[...]`); entries longer than max_chars are hard-truncated; llm None → fallback split on 。；;.! then truncate each. Untriggered → `[text]` unchanged (zero LLM).

**Consensus insert (per normalized entry):** embed → chroma query top_k (no owner filter) → candidates sim ≥ threshold (sim = 1 - cosine distance) → none: add new `{owners:[agent]}` → else ONE llm call listing candidates 0..n, "which is semantically equivalent to the new memory? reply index or -1" → -1/parse-fail/llm-None: add new. Equivalent i: keep SHORTER text (if new shorter → `collection.update` doc+embedding), owners ∪ {agent} (update metadata), no new entry.
**Metadata per entry:** `{"owners": json list str, "owner_<id>": True flags, "created_at": iso, "source", "tick"}`. **forget:** remove agent from owners; empty → `collection.delete`. **revise = forget(old) + remember(new).** **recall:** query with `where={"owner_<agent>": True}`.

- [ ] **Step 1: failing tests.** `tests/test_ltm.py`:
```python
import json, pytest
from society.ltm import SharedMemory
from tests.helpers import FakeLLM, afake_embed

def mem(llm=None, **kw):
    import uuid
    return SharedMemory(afake_embed, llm=llm,
                        collection_name=f"t_{uuid.uuid4().hex[:8]}", **kw)

async def test_short_atomic_text_skips_llm_and_stores():
    llm = FakeLLM()
    m = mem(llm)
    out = await m.remember("alice", "黛玉葬花")
    assert len(out) == 1 and out[0]["merged"] is False
    assert llm.calls == []                      # gate not triggered
    assert m.stats() == {"total": 1, "shared": 0, "ratio": 0.0}

async def test_normalize_splits_long_or_compound(monkeypatch):
    llm = FakeLLM(responses=['["宝玉挨打", "贾政动怒"]'])
    m = mem(llm)
    out = await m.remember("alice", "宝玉挨打了，然后贾政大发雷霆并且惊动了贾母。")
    texts = {e["text"] for e in out}
    assert texts == {"宝玉挨打", "贾政动怒"}
    assert any(c[0] == "normalize" for c in llm.calls) or llm.calls  # one normalize call

async def test_normalize_fallback_without_llm():
    m = mem(None)
    entries = await m._normalize("句子一。句子二。")
    assert entries == ["句子一", "句子二"]

async def test_consensus_merges_equivalent_keeps_shorter_and_unions_owners():
    # identical text → identical fake embedding → sim 1.0 → candidate; llm says index 0
    llm = FakeLLM(responses=["0"])
    m = mem(llm)
    await m.remember("alice", "国王死于春天")
    out = await m.remember("bob", "国王死于春天")     # same text: equal length → keep existing
    assert out[0]["merged"] is True
    entries = m.all_entries()
    assert len(entries) == 1 and set(entries[0]["owners"]) == {"alice", "bob"}
    assert m.stats()["shared"] == 1 and m.stats()["ratio"] == 1.0

async def test_consensus_not_equivalent_adds_new():
    llm = FakeLLM(responses=["-1"])
    m = mem(llm)
    await m.remember("alice", "国王死于春天")
    await m.remember("bob", "国王死于春天")           # candidates found but llm says -1
    assert m.stats()["total"] == 2 and m.stats()["shared"] == 0

async def test_recall_owner_filtered():
    m = mem(None)
    await m.remember("alice", "花园着火")
    await m.remember("bob", "厨房进贼")
    got = await m.recall("alice", "花园着火", top_k=5)
    assert [e["text"] for e in got] == ["花园着火"]

async def test_forget_and_shared_survival():
    llm = FakeLLM(responses=["0"])
    m = mem(llm)
    await m.remember("alice", "国王死于春天")
    await m.remember("bob", "国王死于春天")
    (entry,) = m.all_entries()
    assert m.forget("alice", entry["id"]) is True
    (e2,) = m.all_entries()                       # bob still owns → survives
    assert e2["owners"] == ["bob"]
    assert m.forget("bob", e2["id"]) is True
    assert m.all_entries() == []                  # empty owners → deleted

async def test_revise_is_forget_plus_consensus_insert():
    m = mem(None)
    (e,) = await m.remember("alice", "旧的记忆内容"), 
    e = e[0] if isinstance(e, list) else e
    out = await m.revise("alice", e["id"], "新记忆")
    entries = m.all_entries()
    assert len(entries) == 1 and entries[0]["text"] == "新记忆"
```
Note the awkward tuple line in the last test — write it cleanly instead:
```python
    stored = await m.remember("alice", "旧的记忆内容")
    out = await m.revise("alice", stored[0]["id"], "新记忆")
```
- [ ] **Steps 2-4:** fail → implement → pass, suite green. Use `bucket="normalize"` and `bucket="consensus"` for the two llm.chat call sites (FakeLLM records bucket as calls[i][0]).
- [ ] **Step 5:** commit `"feat: shared LTM with normalize gate + consensus insert/forget/revise"`.

---

### Task 6: Brains + agent skill files

**Files:** Create `society/brains/__init__.py`, `society/brains/base.py`, `society/brains/rule_brain.py`, `society/brains/retrieval_brain.py`, `society/brains/llm_brain.py`, `society/skills/actions_skill_zh.md`, `society/skills/actions_skill_en.md`, `tests/test_brains.py`

**Interfaces (Produces):**
```python
# base.py
class Brain(ABC):
    async def decide(self, view: dict) -> Action
# rule_brain.py
class RuleBrain(Brain):
    def __init__(self, fn=None)        # fn(view)->Action; default: Action("wait")
    # passive env interface:
    def handle_act_on(self, actor_id: str, description: str, view: dict) -> str
        # default: f"{actor_id} 对环境做了: {description}"; overridable via ctor arg act_on_fn
# retrieval_brain.py
class RetrievalBrain(Brain):
    def __init__(self, corpus_text: str = "", chunk_size: int = 300)
    async def decide(self, view) -> Action        # always Action("wait")
    def retrieve(self, query: str, top_k: int = 2) -> str
        # keyword-overlap scoring over chunks (split corpus by blank lines then size); no LLM
# llm_brain.py
SKILL_DIR: str
class LLMBrain(Brain):
    def __init__(self, llm, profile: str, language: str = "zh", extra_skill: str = "")
    async def decide(self, view: dict) -> Action
        # system = profile + skill md + format instructions (respond ONLY {"action":...,"params":{...}})
        # user = json.dumps(view, ensure_ascii=False)
        # parse via actions.parse_action; on ValueError retry ≤2 more times appending the error;
        # final failure → Action("noop", {"note": "decide-parse-failed"})
```
Skill files: write `actions_skill_zh.md` covering EVERY action (signature, sync/async, what the result is) and the five pipelines from spec §5.3 (消息处理/社交/移动/记忆卫生/目标管理). `actions_skill_en.md` = English equivalent. LLMBrain loads by language.

- [ ] **Step 1: failing tests.** `tests/test_brains.py`:
```python
from society.actions import Action
from society.brains.rule_brain import RuleBrain
from society.brains.retrieval_brain import RetrievalBrain
from society.brains.llm_brain import LLMBrain
from tests.helpers import FakeLLM

async def test_rule_brain_default_and_custom():
    assert (await RuleBrain().decide({})).name == "wait"
    rb = RuleBrain(fn=lambda v: Action("noop"))
    assert (await rb.decide({})).name == "noop"
    assert "推门" in RuleBrain().handle_act_on("alice", "推门", {}) or "alice" in RuleBrain().handle_act_on("alice", "推门", {})

async def test_retrieval_brain_retrieve_and_idle():
    rb = RetrievalBrain("石头记开篇。\n\n宝玉衔玉而生。\n\n黛玉进贾府。")
    assert (await rb.decide({})).name == "wait"
    out = rb.retrieve("宝玉 衔玉")
    assert "宝玉衔玉而生" in out

async def test_llm_brain_parses_action_and_injects_skill():
    llm = FakeLLM(responses=['{"action": "say", "params": {"targets": ["b"], "content": "hi"}}'])
    b = LLMBrain(llm, profile="你是黛玉", language="zh")
    a = await b.decide({"tick": 1, "goals": []})
    assert a.name == "say" and a.params["targets"] == ["b"]
    bucket, prompt = llm.calls[0][0], llm.calls[0][1]
    assert bucket == "decide"
    sys = llm.calls[0][2] if len(llm.calls[0]) > 2 else ""
    # FakeLLM must record system too: calls entries are (bucket, prompt, system)
    assert "你是黛玉" in sys and "pop_message" in sys   # skill内容注入

async def test_llm_brain_retries_then_noop():
    llm = FakeLLM(responses=["not json", "still bad", "nope"])
    b = LLMBrain(llm, profile="p")
    a = await b.decide({})
    assert a.name == "noop" and len(llm.calls) == 3
```
(Update `tests/helpers.py` FakeLLM so `self.calls` entries are `(bucket, prompt, system)` — adjust Task 4's test accordingly if needed: it indexes `calls[0][1] == "p"`, which stays valid.)
- [ ] **Steps 2-4:** fail → implement + write both skill md files (zh must mention every action name; test greps "pop_message") → pass, suite green.
- [ ] **Step 5:** commit `"feat: rule/retrieval/llm brains + agent action skills (zh/en)"`.

---

### Task 7: World map + kernel core (tick loop, delivery, sleep/wake, transit, quiescence)

**Files:** Create `society/worldmap.py`, `society/agent.py`, `society/kernel.py`, `tests/test_kernel_core.py`

**Interfaces (Produces):**
```python
# worldmap.py
class WorldMap:
    def __init__(self, env_ids: list[str], edges: list[tuple[str,str,int]] | None = None,
                 default_distance: int = 20, fully_connected: bool = True)
    def connected(self, a, b) -> bool
    def distance(self, a, b) -> int | None      # None if not connected; explicit edges override default

# agent.py
class Agent:
    def __init__(self, agent_id: str, kind: str, brain, stm: STM, *, portable=False, holder=None,
                 profile: str = "")
    id, kind, brain, stm, profile, portable, holder
    # runtime state (kernel-managed): waiting_until: int|None ("INF"=-1 semantics: use None=not waiting,
    #   -1=wait forever, else wake tick), transit: dict|None ({"dest","arrive_at"})
    def location(self) -> str | None
    def build_view(self, tick: int) -> dict     # {"tick","agent_id","kind","goals":items,
        # "status":all(),"fifo":[{"action","result"}...],"inbox_size",n,"inbox_head":{"sender","kind"}|None}

# kernel.py
class Kernel:
    def __init__(self, agents: dict[str, Agent], worldmap: WorldMap, event_log: EventLog,
                 shared_memory=None, llm=None, metrics=None, config: dict | None = None)
    presence: dict[str, set]                 # location -> agent ids (character + info_carrier with fixed loc)
    tick: int
    def send(self, msg: Message)             # queue for t+1 delivery
    async def run(self, max_ticks: int | None = None, max_wall_seconds: float | None = None) -> dict
        # returns {"ticks_run": int, "stop_reason": "max_ticks"|"quiescent"|"budget"|"wall_time"}
    def is_eligible(self, a: Agent) -> bool
    async def execute(self, agent: Agent, action: Action) -> ActionResult   # Task 8 fills handlers;
        # in this task implement: noop, wait, and a stub raising for others is NOT acceptable —
        # implement the full dispatch table here but only the handlers listed for Task 7:
        # noop, wait, pop_message, peek_inbox, conclude, push_goal, pop_goal, replace_goal,
        # update_status, remove_status. Remaining handlers land in Task 8.
```
**Tick loop (reference algorithm):**
```
while True:
    if stop conditions (max_ticks reached / wall time / budget flag) → break
    process arrivals: agents with transit and transit["arrive_at"] <= tick:
        set location(dest) via status register internal setter; update presence;
        push arrival Message(kind="arrival", sender="kernel") directly into agent.inbox (visible NOW);
        notify dest env + origin env (system messages via send() at current tick → visible t+1);
        clear transit; log system event
    awake = [a for a in agents.values() if is_eligible(a)]
    if awake:
        await asyncio.gather(*(self._step(a) for a in awake))
    deliver self._pending (msgs sent this tick) → inbox.put_nowait per recipient; clear waiting for recipients;
        log "message" events; metrics.on_message for say/gesture per recipient
    metrics.maybe_snapshot(tick) if metrics
    if not awake and not pending-just-delivered and no transit and no waiting-with-timeout → quiescent break
    if not awake (but transit/timers pending) → fast-forward: tick jumps to min(next arrive_at, next wake_at)
    else tick += 1
is_eligible(a): a.transit is None and (
    inbox nonempty  OR  (not waiting and goals nonempty) OR (waiting with wake_at not None and wake_at<=tick))
    # waiting forever (-1) + empty inbox → ineligible; waking from timeout clears waiting.
_step(a): view=a.build_view(tick); action=await a.brain.decide(view);
    err=validate_action(action) → result=ActionResult(False,error=err) if err else await execute(a,action);
    a.stm.fifo.append({"name":action.name,"params":action.params}, result.to_dict());
    log "action" event {action, result, location}
```
Handlers for this task: `wait` sets `a.waiting_until = tick + int(params.get("timeout_ticks")) if given else -1`, result ok; inbox arrival clears waiting (set waiting_until=None on delivery); `pop_message` gets from inbox (empty → ok=False error "inbox empty"), returns msg.to_dict(); `peek_inbox` lists `{"sender","kind"}` without consuming (drain to list & restore or maintain deque mirror — implement inbox as `collections.deque` inside STM? NO — STM.inbox is asyncio.Queue per Task 1; use its internal `_queue` read-only for peek: acceptable single-process). Register ops mutate STM; `update_status` on "location" already rejected by validation.

- [ ] **Step 1: failing tests.** `tests/test_kernel_core.py`:
```python
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
        step = len(log); log.append(1)
        return [Action("push_goal", {"text": "small"}),
                Action("update_status", {"key": "mood", "value": "calm"}),
                Action("pop_goal"),
                Action("noop")][min(step, 3)]
    a = make_char("a", "hall", fn=fn, goals=["big"])
    k = society([a, make_env("hall")])
    await k.run(max_ticks=3)
    assert a.stm.goals.items() == ["big"]
    assert a.stm.status.get("mood") == "calm"
```
- [ ] **Steps 2-4:** fail → implement `worldmap.py`, `agent.py`, `kernel.py` per reference → pass, suite green. Async actions in `execute`: for this task implement `say`/`gesture`/`act_on` minimally as: build Message(kind=name; content/description), `self.send(msg)`, result ok "sent" — full validation (co-location etc.) comes in Task 8; keep handlers factored so Task 8 extends them.
- [ ] **Step 5:** commit `"feat: worldmap + kernel tick loop (delivery, sleep/wake, wait, fast-forward, quiescence)"`.

---

### Task 8: Full action executor (validation, observe/read/move, memory actions, think, env act_on)

**Files:** Modify `society/kernel.py`; Create `tests/test_kernel_actions.py`

**Interfaces (Consumes):** SharedMemory (Task 5), RetrievalBrain.retrieve / RuleBrain.handle_act_on (Task 6), WorldMap (Task 7).
**Produces (final handler semantics):**
- `say`/`gesture`: every target must exist and share sender's location → else ok=False error naming offenders, NO message sent. On success one Message per action (recipients=targets), kind = "say"/"gesture".
- `observe(target)`: environment → `{"status": env public status, "occupants": [{"id","kind","status":public} …]}` from presence (exclude observer). character → must share location → public status. info_carrier → must be readable (same loc, or portable and holder==observer) → `{"meta": {"kind","portable"}, "status": public}`.
- `read(target, query)`: target must be info_carrier + readable → `RetrievalBrain.retrieve(query)` string.
- `move(destination)`: destination must be environment + connected from current loc (same loc → error "already there"). distance d = map.distance; leave: presence remove + system msg to origin env; set `agent.transit={"dest","arrive_at": tick+d}`; result ok `{"eta": tick+d}`. Arrival handled by Task 7 loop (already implemented).
- `remember(text)` → `await shared.remember(agent.id, text, tick)` → data = entries list; no shared memory configured → ok=False "no shared memory".
- `recall(query, top_k?)`, `forget(memory_id)`, `revise_memory` → corresponding SharedMemory calls.
- `think(question)`: requires kernel.llm → `llm.chat(prompt with view+question, bucket="think")` → data = conclusion.
- `act_on(target, description)`: target env, must be at it. RuleBrain env → synchronous: `handle_act_on(...)` → kernel wraps as env_result Message sent to actor (t+1). LLM-brain env → send Message(kind="act_on") to env's inbox (its own loop responds) — for v1 both paths deliver an `env_result`/message; test covers RuleBrain path.

- [ ] **Step 1: failing tests.** `tests/test_kernel_actions.py`:
```python
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
    kinds = [m["payload"]["kind"] if "payload" in m else m.get("msg_kind") for m in []]  # (placeholder-free: assert via inbox)
    assert any(True for _ in [1])   # arrival message reached agent:
    assert a.stm.inbox.qsize() >= 1 or len(a.stm.fifo) >= 0

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
    k._deliver_pending()      # expose or call the delivery step; simplest: make delivery a public method deliver_pending()
    assert a.stm.inbox.qsize() == 1
    msg = a.stm.inbox.get_nowait()
    assert msg.kind == "env_result" and "门开了" in msg.content
```
Clean up the two sloppy asserts in `test_move_sets_transit_and_arrival` (drop the placeholder lines; assert arrival via `a.location()=="garden"` plus a "arrival" kind message recorded in event log). Make `deliver_pending()` a public kernel method used by the tick loop.
- [ ] **Steps 2-4:** fail → implement handlers + refactors → pass, suite green.
- [ ] **Step 5:** commit `"feat: full action executor (observe/read/move/memory/think/act_on with validation)"`.

---

### Task 9: Metrics snapshots

**Files:** Create `society/metrics.py`, `tests/test_metrics.py`

**Interfaces (Produces):**
```python
class Metrics:
    def __init__(self, agents: dict, shared_memory, out_dir: str | None, interval: int = 10)
    def on_message(self, sender: str, recipient: str, kind: str)   # counts only say|gesture
    def snapshot(self, tick: int) -> dict     # always builds dict; writes stats/tick_%06d.json if out_dir
    def maybe_snapshot(self, tick: int) -> dict | None   # only when tick>0 and tick % interval == 0
```
Snapshot dict:
```python
{"tick": t,
 "consensus_ratio": {"total": n, "shared": m, "ratio": m/n or 0.0},
 "comm_graph": {"directed": {"a->b": w, ...}, "undirected": {"a|b": w, ...}},  # a|b key sorted
 "consensus_owners": [{"id","text","owners"} for entries with len(owners)>=2]}
```

- [ ] **Step 1: failing tests.** `tests/test_metrics.py`:
```python
import json, os, uuid
from society.metrics import Metrics
from society.ltm import SharedMemory
from tests.helpers import FakeLLM, afake_embed

async def test_snapshot_contents(tmp_path):
    shared = SharedMemory(afake_embed, llm=FakeLLM(responses=["0"]),
                          collection_name=f"t_{uuid.uuid4().hex[:8]}")
    await shared.remember("alice", "国王死于春天")
    await shared.remember("bob", "国王死于春天")     # merges → shared entry
    await shared.remember("bob", "厨房进贼")
    m = Metrics({}, shared, str(tmp_path), interval=10)
    m.on_message("a", "b", "say"); m.on_message("a", "b", "say"); m.on_message("b", "a", "gesture")
    m.on_message("a", "b", "system")               # ignored
    snap = m.snapshot(10)
    assert snap["consensus_ratio"] == {"total": 2, "shared": 1, "ratio": 0.5}
    assert snap["comm_graph"]["directed"] == {"a->b": 2, "b->a": 1}
    assert snap["comm_graph"]["undirected"] == {"a|b": 3}
    owners = snap["consensus_owners"]
    assert len(owners) == 1 and set(owners[0]["owners"]) == {"alice", "bob"}
    assert os.path.exists(tmp_path / "stats" / "tick_000010.json")
    on_disk = json.load(open(tmp_path / "stats" / "tick_000010.json", encoding="utf-8"))
    assert on_disk["tick"] == 10

async def test_maybe_snapshot_interval(tmp_path):
    m = Metrics({}, None, str(tmp_path), interval=10)
    assert m.maybe_snapshot(0) is None
    assert m.maybe_snapshot(7) is None
    assert m.maybe_snapshot(10) is not None
    # shared_memory None → consensus fields zeroed
    snap = m.snapshot(20)
    assert snap["consensus_ratio"]["total"] == 0 and snap["consensus_owners"] == []
```
- [ ] **Steps 2-4:** fail → implement → pass. Wire into Kernel: constructor already accepts `metrics`; tick loop calls `metrics.maybe_snapshot(tick)` and `deliver_pending` calls `metrics.on_message` per recipient — add a kernel test asserting on_message got called? Covered implicitly in Task 11 integration; unit tests above suffice.
- [ ] **Step 5:** commit `"feat: periodic metrics (consensus ratio, comm graph, owner snapshots)"`.

---

### Task 10: Scenario loader + society builder

**Files:** Create `society/scenario.py`, `tests/test_scenario.py`, `scenarios/demo_min.yaml`

**Interfaces (Produces):**
```python
def load_scenario(path: str) -> dict          # parsed+validated: raises ValueError on missing id/kind, dup ids,
                                              # unknown brain, seed location not an environment, bad map edge refs
async def build_society(cfg: dict, *, llm, embed_fn, event_log, out_dir=None,
                        metrics_interval=None) -> Kernel
    # creates agents (brain per kind/brain field; llm brains get profile+language),
    # info_carrier corpus loaded from file path relative to scenario file,
    # WorldMap from cfg["map"] (default_distance, edges; envs = all environment agents),
    # SharedMemory(embed_fn, llm, max_chars=defaults.memory_max_chars),
    # seeds: for each agent's seed_memories → await shared.remember(agent_id, text, tick=0, source="scenario_seed")
    # kickoff: each {"to":[ids],"kind","content"} → kernel.send at tick 0 → visible tick 1... 
    #   BUT tick-0 sleepers must wake: send kickoff BEFORE run so delivery happens in first loop pass;
    #   implement: kernel.send(...) with tick_sent=-1 then kernel delivers pending at loop start of tick 0.
    # Metrics attached (interval from defaults.stats_interval unless metrics_interval given).
```
`scenarios/demo_min.yaml` (also used by tests):
```yaml
scenario: demo_min
language: zh
defaults: {fifo_size: 20, memory_max_chars: 80, distance: 5, stats_interval: 10}
agents:
  - {id: hall, kind: environment, brain: rule, profile: "大厅"}
  - {id: garden, kind: environment, brain: rule, profile: "花园"}
  - id: alice
    kind: character
    brain: llm
    profile: "爱丽丝,好奇。"
    status: {location: hall, mood: "好奇"}
    goals: ["弄清此地何处"]
    seed_memories: ["爱丽丝掉进了兔子洞"]
  - id: guide_book
    kind: info_carrier
    brain: retrieval
    status: {location: hall}
    corpus: corpora/demo_guide.txt
map:
  default_distance: 5
  edges: [[hall, garden, 2]]
kickoff:
  - {to: [alice], kind: system, content: "你醒来发现自己在大厅。"}
```
Create `scenarios/corpora/demo_guide.txt` with two paragraphs of any content.

- [ ] **Step 1: failing tests.** `tests/test_scenario.py`:
```python
import pytest, yaml
from society.events import EventLog
from society.scenario import load_scenario, build_society
from tests.helpers import FakeLLM, afake_embed

async def test_load_and_build_demo():
    cfg = load_scenario("scenarios/demo_min.yaml")
    assert cfg["scenario"] == "demo_min"
    llm = FakeLLM(fn=lambda p, s=None: '{"action": "noop", "params": {}}')
    k = await build_society(cfg, llm=llm, embed_fn=afake_embed, event_log=EventLog(None))
    assert set(k.agents) == {"hall", "garden", "alice", "guide_book"}
    assert k.worldmap.distance("hall", "garden") == 2
    assert "alice" in k.presence["hall"]
    entries = k.shared_memory.all_entries()
    assert any("兔子洞" in e["text"] for e in entries)      # seed loaded via consensus path
    summary = await k.run(max_ticks=3)                      # kickoff wakes alice at tick 0/1
    acted = [e for e in k.event_log.all() if e["kind"] == "action" and e["agent"] == "alice"]
    assert acted, "kickoff must wake alice"

def test_load_validation_errors(tmp_path):
    bad = {"scenario": "x", "agents": [{"id": "a", "kind": "character", "brain": "llm",
                                        "status": {"location": "nowhere"}}]}
    p = tmp_path / "bad.yaml"; p.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises(ValueError):
        load_scenario(str(p))
    dup = {"scenario": "x", "agents": [{"id": "a", "kind": "environment", "brain": "rule"},
                                       {"id": "a", "kind": "environment", "brain": "rule"}]}
    p2 = tmp_path / "dup.yaml"; p2.write_text(yaml.safe_dump(dup), encoding="utf-8")
    with pytest.raises(ValueError):
        load_scenario(str(p2))
```
- [ ] **Steps 2-4:** fail → implement (+ kickoff delivery at loop start of tick 0 in kernel if not already) → pass, suite green.
- [ ] **Step 5:** commit `"feat: scenario loader + society builder (seeds, kickoff, map, metrics wiring)"`.

---

### Task 11: Runner CLI + transcripts + integration smoke

**Files:** Create `society/run.py`, `tests/test_run_integration.py`

**Interfaces (Produces):**
```python
# society/run.py
def write_transcripts(events: list[dict], agents: dict, out_dir: str)   # transcripts/<id>.md
    # per agent: action events "[tick N] <name>(params) -> ok/err: data...` + messages received
def write_outputs(kernel, out_dir, scenario_cfg, summary)   # llm_usage.json (kernel.llm.usage() if any),
    # config_snapshot.yaml (scenario cfg + summary), calls write_transcripts
async def run_scenario(scenario_path, ticks, out_dir, *, llm, embed_fn) -> dict  # orchestrates; returns summary
def main(argv=None)   # argparse: --scenario --ticks --out [--screenplay] [--config config.json]
    # builds real LLMClient/EmbeddingClient from config.json (env OPENAI_API_KEY fallback)
```
- [ ] **Step 1: failing integration test.** `tests/test_run_integration.py` — the spec §14 smoke (3 characters + 1 env + 1 info_carrier, FakeLLM scripted, 30 ticks):
```python
import json, os, yaml
from society.run import run_scenario
from tests.helpers import FakeLLM, afake_embed

SCEN = {
  "scenario": "smoke", "language": "zh",
  "defaults": {"stats_interval": 10, "distance": 3},
  "agents": [
    {"id": "hall", "kind": "environment", "brain": "rule", "profile": "hall"},
    {"id": "study", "kind": "environment", "brain": "rule", "profile": "study"},
    {"id": "book", "kind": "info_carrier", "brain": "retrieval",
     "status": {"location": "hall"}, "corpus": "corpora/smoke.txt"},
    {"id": "amy", "kind": "character", "brain": "llm", "profile": "amy",
     "status": {"location": "hall"}, "goals": ["chat"]},
    {"id": "ben", "kind": "character", "brain": "llm", "profile": "ben",
     "status": {"location": "hall"}},
    {"id": "cid", "kind": "character", "brain": "llm", "profile": "cid",
     "status": {"location": "hall"}},
  ],
  "map": {"default_distance": 3},
  "kickoff": [{"to": ["amy"], "kind": "system", "content": "开始"}],
}

def scripted(prompt, system=None):
    # amy: observe → say to ben,cid → remember → noop...; ben/cid: pop → say back once → noop
    # Deterministic by matching agent profile in system prompt:
    if system and "amy" in system:
        for probe, resp in [
            ("observe", None)]:
            pass
    return '{"action": "noop", "params": {}}'

async def test_smoke_run_produces_all_outputs(tmp_path):
    scen_dir = tmp_path / "scen"; (scen_dir / "corpora").mkdir(parents=True)
    (scen_dir / "corpora" / "smoke.txt").write_text("规则手册第一条。", encoding="utf-8")
    spath = scen_dir / "smoke.yaml"
    spath.write_text(yaml.safe_dump(SCEN, allow_unicode=True), encoding="utf-8")

    seq = {"amy": ['{"action": "observe", "params": {"target": "hall"}}',
                   '{"action": "say", "params": {"targets": ["ben", "cid"], "content": "大家好"}}',
                   '{"action": "remember", "params": {"text": "我在大厅打了招呼"}}',
                   '{"action": "pop_goal", "params": {}}'],
           "ben": ['{"action": "pop_message", "params": {}}',
                   '{"action": "say", "params": {"targets": ["amy"], "content": "你好"}}'],
           "cid": ['{"action": "pop_message", "params": {}}',
                   '{"action": "gesture", "params": {"targets": ["amy"], "description": "挥手"}}']}
    def fn(prompt, system=None):
        for aid in seq:
            if system and aid in system:
                return seq[aid].pop(0) if seq[aid] else '{"action": "noop", "params": {}}' \
                       if False else (seq[aid].pop(0) if seq[aid] else '{"action": "wait", "params": {}}')
        return '{"action": "wait", "params": {}}'
    llm = FakeLLM(fn=fn)

    out = str(tmp_path / "run1")
    summary = await run_scenario(str(spath), ticks=30, out_dir=out, llm=llm, embed_fn=afake_embed)
    assert os.path.exists(f"{out}/events.jsonl")
    assert os.path.exists(f"{out}/transcripts/amy.md")
    assert os.path.exists(f"{out}/config_snapshot.yaml")
    assert os.path.exists(f"{out}/llm_usage.json")
    stats = sorted(os.listdir(f"{out}/stats"))
    assert "tick_000010.json" in stats
    snap = json.load(open(f"{out}/stats/tick_000010.json", encoding="utf-8"))
    und = snap["comm_graph"]["undirected"]
    assert und.get("amy|ben") and und.get("amy|cid")        # both directions counted
    amy_md = open(f"{out}/transcripts/amy.md", encoding="utf-8").read()
    assert "remember" in amy_md and "[tick" in amy_md
```
Simplify the messy `fn` in the test to a clean pop-with-default helper before running (no dead conditional). Ensure agents with empty goals+inbox sleep so the run quiesces before tick 30 (summary stop_reason may be "quiescent" — assert `summary["ticks_run"] >= 10` so a stats snapshot exists; if quiescence lands before tick 10, keep `ben`/`cid` goals empty but give amy 4 actions and let `wait` idle — the metrics `maybe_snapshot` must also fire on the FINAL tick before stopping regardless of interval: implement "final snapshot on stop" in run_scenario writing `stats/tick_<final>.json`; adapt the assertion to `any(s.startswith("tick_") for s in stats)` plus the amy|ben edge in the FINAL snapshot instead of exactly tick_000010).
- [ ] **Steps 2-4:** fail → implement run.py (+ final-snapshot-on-stop) → clean the test per note → pass, suite green.
- [ ] **Step 5:** commit `"feat: runner CLI, transcripts, usage/config outputs + integration smoke"`.

---

### Task 12: Screenplay generator

**Files:** Create `society/screenplay.py`, `tests/test_screenplay.py`

**Interfaces (Produces):**
```python
async def generate_screenplay(events: list[dict], llm, out_path: str | None = None,
                              language: str = "zh", scene_gap: int = 5) -> str
    # 1) keep only kinds: action(say/gesture/think/conclude/move/act_on results) + message(say/gesture)
    #    → beats sorted by (tick, seq)
    # 2) split scenes: new scene when location changes or tick jumps > scene_gap
    # 3) LLM pass per scene (bucket="screenplay"): given beats JSON → select worthy beats + render
    #    scene block: "## 第N幕 · <location> · tick a–b" then dialogue/stage directions;
    #    think/conclude rendered as (内心独白/旁白)
    # 4) concatenate scenes → markdown; write file if out_path
```
- [ ] **Step 1: failing tests.** `tests/test_screenplay.py`:
```python
from society.screenplay import generate_screenplay
from tests.helpers import FakeLLM

EVENTS = [
 {"seq": 0, "tick": 0, "kind": "action", "agent": "amy",
  "action": {"name": "say", "params": {"targets": ["ben"], "content": "走吧"}},
  "result": {"ok": True}, "location": "hall"},
 {"seq": 1, "tick": 1, "kind": "action", "agent": "ben",
  "action": {"name": "think", "params": {"question": "去哪"}},
  "result": {"ok": True, "data": "还是花园好"}, "location": "hall"},
 {"seq": 2, "tick": 9, "kind": "action", "agent": "amy",
  "action": {"name": "gesture", "params": {"targets": ["ben"], "description": "指向花园"}},
  "result": {"ok": True}, "location": "garden"},
]

async def test_scene_split_and_render(tmp_path):
    calls = []
    def fn(prompt, system=None):
        calls.append(prompt)
        return f"（第{len(calls)}场渲染文本）"
    llm = FakeLLM(fn=fn)
    out = str(tmp_path / "sp.md")
    md = await generate_screenplay(EVENTS, llm, out_path=out, scene_gap=5)
    assert len(calls) == 2                      # hall scene + garden scene (location change & gap)
    assert "第1幕" in md and "hall" in md and "garden" in md
    assert "（第1场渲染文本）" in md and open(out, encoding="utf-8").read() == md
    assert "走吧" in calls[0] and "还是花园好" in calls[0]   # beats reach the LLM

async def test_noop_and_failed_actions_excluded():
    evs = EVENTS + [{"seq": 3, "tick": 9, "kind": "action", "agent": "amy",
                     "action": {"name": "noop", "params": {}}, "result": {"ok": True},
                     "location": "garden"},
                    {"seq": 4, "tick": 9, "kind": "action", "agent": "amy",
                     "action": {"name": "say", "params": {"targets": ["x"], "content": "?"}},
                     "result": {"ok": False, "error": "x not here"}, "location": "garden"}]
    llm = FakeLLM(fn=lambda p, s=None: "ok")
    await generate_screenplay(evs, llm)
    joined = "".join(c[1] for c in llm.calls)
    assert "noop" not in joined and "not here" not in joined
```
- [ ] **Steps 2-4:** fail → implement → pass, suite green.
- [ ] **Step 5:** commit `"feat: screenplay generator (scene split + two-stage LLM render)"`. Wire `--screenplay` flag in run.py main() to call it after the run (no new test needed; flag exercised in Task 14 docs check).

---

### Task 13: Novel → scenario extractor

**Files:** Create `society/extract.py`, `tests/test_extract.py`

**Interfaces (Produces):**
```python
async def extract_scenario(text: str, llm, out_yaml: str, *, max_agents: int = 15,
                           language: str = "zh", hints: str = "") -> dict
    # chunks text (chunk_chars=8000, overlap=500); per stage llm.chat(bucket="extract") returning JSON
    # stages: characters → locations+map → info_carriers → seed memories per character → kickoff
    # merge/dedupe across chunks by name; caps agents at max_agents (characters prioritized);
    # writes out_yaml (standard scenario format) + corpora/<carrier_id>.txt next to it; returns cfg dict
    # every stage JSON schema-checked; parse failure retried once then stage skipped with warning list in cfg["_warnings"]
def main(argv=None)   # argparse: --input file --output scenarios/x.yaml [--max-agents] [--language] [--hints]
```
- [ ] **Step 1: failing tests.** `tests/test_extract.py`:
```python
import json, os, yaml
from society.extract import extract_scenario
from society.scenario import load_scenario
from tests.helpers import FakeLLM

STAGES = {
 "characters": json.dumps([{"id": "daiyu", "name": "林黛玉", "profile": "多愁善感",
                            "status": {"location": "xiaoxiang", "mood": "忧郁"},
                            "goals": ["求真情", "试探宝玉"]}], ensure_ascii=False),
 "locations": json.dumps({"locations": [{"id": "xiaoxiang", "profile": "潇湘馆"},
                                        {"id": "hengwu", "profile": "蘅芜苑"}],
                          "edges": [["xiaoxiang", "hengwu", 5]]}, ensure_ascii=False),
 "carriers": json.dumps([{"id": "shitou_ji", "profile": "石头记",
                          "location": "xiaoxiang", "excerpt": "满纸荒唐言"}], ensure_ascii=False),
 "memories": json.dumps({"daiyu": ["黛玉葬花", "宝玉赠帕"]}, ensure_ascii=False),
 "kickoff": json.dumps([{"to": ["daiyu"], "kind": "system", "content": "宝玉遣人送帕"}], ensure_ascii=False),
}
def fake(prompt, system=None):
    for key, marker in [("characters", "角色"), ("locations", "地点"),
                        ("carriers", "信息载体"), ("memories", "记忆"), ("kickoff", "起始")]:
        if marker in prompt:
            return STAGES[key]
    return "[]"

async def test_extract_produces_loadable_scenario(tmp_path):
    out = str(tmp_path / "red.yaml")
    cfg = await extract_scenario("……黛玉葬花……", FakeLLM(fn=fake), out, max_agents=10)
    loaded = load_scenario(out)                      # must pass the real loader's validation
    ids = {a["id"] for a in loaded["agents"]}
    assert {"daiyu", "xiaoxiang", "hengwu", "shitou_ji"} <= ids
    daiyu = next(a for a in loaded["agents"] if a["id"] == "daiyu")
    assert daiyu["goals"] == ["求真情", "试探宝玉"] and daiyu["seed_memories"] == ["黛玉葬花", "宝玉赠帕"]
    assert loaded["map"]["edges"] == [["xiaoxiang", "hengwu", 5]]
    corpus = os.path.join(str(tmp_path), "corpora", "shitou_ji.txt")
    assert os.path.exists(corpus) and "满纸荒唐言" in open(corpus, encoding="utf-8").read()
    assert loaded["kickoff"][0]["to"] == ["daiyu"]

async def test_stage_failure_is_skipped_with_warning(tmp_path):
    def flaky(prompt, system=None):
        if "信息载体" in prompt: return "not json at all"
        return fake(prompt)
    cfg = await extract_scenario("文本", FakeLLM(fn=flaky), str(tmp_path / "x.yaml"))
    assert cfg["_warnings"] and any("carrier" in w or "信息载体" in w for w in cfg["_warnings"])
    load_scenario(str(tmp_path / "x.yaml"))          # still loadable without carriers
```
Prompts MUST contain the stage markers used above (角色/地点/信息载体/记忆/起始) so the fake can route.
- [ ] **Steps 2-4:** fail → implement (stage prompts in `society/prompts/` or inline in extract.py — inline acceptable) → pass, suite green.
- [ ] **Step 5:** commit `"feat: novel→scenario extractor with staged schema-validated pipeline"`.

---

### Task 14: Docs + example scenario + final polish

**Files:** Create `docs/actions.md`, `scenarios/demo_red_chamber.yaml` (+ `scenarios/corpora/` file), `README.md`. Modify: none.

- [ ] **Step 1:** `docs/actions.md` — human-reader reference: all 21 actions grouped sync/async with params/results/validation rules; the 5 pipelines; ticks/delivery/sleep semantics summary. Content must agree with `society/actions.py` REQUIRED_PARAMS (copy from code, not memory).
- [ ] **Step 2:** `scenarios/demo_red_chamber.yaml` — a hand-written 红楼梦 mini scenario: 3 characters (llm), 2 environments (rule), 1 info_carrier (corpus file), map edges, seeds, kickoff — must pass `load_scenario` (add a 3-line test in `tests/test_scenario.py`: `def test_demo_red_chamber_loads(): load_scenario("scenarios/demo_red_chamber.yaml")`).
- [ ] **Step 3:** `README.md` — quickstart: venv, config.json from example, run demo (`venv/bin/python -m society.run --scenario scenarios/demo_red_chamber.yaml --ticks 50 --out runs/demo --screenplay`), extract usage, outputs layout, test command.
- [ ] **Step 4:** full suite `venv/bin/python -m pytest -q` green.
- [ ] **Step 5:** commit `"docs: action reference, README quickstart, red-chamber demo scenario"`.

---

## Final verification

- [ ] `venv/bin/python -m pytest -q` — entire suite green.
- [ ] `venv/bin/python -c "from society import run"` imports clean.
- [ ] `venv/bin/python -m society.run --help` and `venv/bin/python -m society.extract --help` both print usage.
- [ ] Spec-coverage sweep vs `docs/specs/2026-07-08-agent-society-design.md` §3–§12 (tick model, 21 actions, consensus semantics, outputs, defaults table) — each maps to a shipped test.
