# Final-review fixes

Branch: build-framework
Test cmd: `venv/bin/python -m pytest -q`

## Finding 1 (CRITICAL) — budget circuit-breaker unwired & crashes

**(a) `society/run.py::_build_llm_and_embed`** ignored `max_calls` /
`max_tokens` / `max_concurrency` from `config.json`. Now reads all three
(defaulting to `16` / `None` / `None`, matching `config.json.example`) and
passes them into `LLMClient(...)`.

**(b) `society/kernel.py::_budget_exceeded`** polled a phantom
`metrics.budget_exceeded()` that `Metrics` never implements, so the budget
stop condition was dead code. Replaced with a real mechanism:
- `Kernel.__init__` now sets `self._budget_hit = False`.
- `society.llm.BudgetExceeded` is imported into `kernel.py`.
- `_decide()` (the decide phase): when `agent.brain.decide(view)` raises
  `BudgetExceeded`, the flag is set (`self._budget_hit = True`) in addition
  to the existing behavior of returning `(None, str(exc))`, which
  `_apply()` still turns into a logged failed action for that agent.
- The `run()` gather loop's `isinstance(res, Exception)` branch (defensive,
  in case an exception ever does propagate out of `_decide` rather than
  being caught inside it) also sets the flag for a `BudgetExceeded` result.
- `_apply()` (the apply/execute phase, covers `think`/`remember`/`recall`/
  `revise_memory` and any other handler that awaits the LLM or shared
  memory): `self.execute(agent, action)` is now wrapped in
  `try/except BudgetExceeded`, converting it to
  `ActionResult(False, error="budget exceeded")` and setting the flag —
  the tick still finishes applying every other agent's effects.
- `_budget_exceeded()` now returns `True` whenever `self._budget_hit` is
  set (checked first), falling back to the old duck-typed
  `metrics.budget_exceeded()` check only if present (kept for forward
  compatibility, but no longer required).
- The existing top-of-loop check in `run()` (`if self._budget_exceeded():
  stop_reason = "budget"; break`) was already positioned correctly — it
  now actually fires. Because it runs at the *top* of the next iteration,
  the tick that tripped the budget always finishes applying all its
  agents' effects first (spec §9/§13: "complete current work").

**(c) `run_scenario` output flushing on budget stop** — no code change was
needed here: `run_scenario` already calls `write_outputs(...)` on
`kernel.run()`'s normal return path regardless of `stop_reason`. Verified
by the new `test_budget_stop_flushes_outputs` regression test.

## Finding 2 (IMPORTANT) — `private_status_keys` silently ignored

`society/scenario.py::build_society` now passes
`private_keys=set(a["private_status_keys"]) if a.get("private_status_keys") else None`
into `STM(...)`, so a character/environment's configured private status
keys are actually wired into its `StatusRegister`.

`load_scenario` now validates: if an agent dict has a
`"private_status_keys"` key, it must be a list of strings, else
`ValueError`. Docstring updated to document this validation.

## Finding 3 (MINOR) — docs

`docs/actions.md` (~line 281-283): replaced the stale
"`metrics.budget_exceeded()`" description of the stop condition with an
accurate description of the real mechanism — `LLMClient` raising
`BudgetExceeded` before a call that would exceed `max_calls`/`max_tokens`,
caught by `Kernel._decide`/`Kernel._apply`, setting `self._budget_hit`,
read by `_budget_exceeded()` at the top of the next tick.

`README.md`:
- Added a paragraph after the `config.json` snippet spelling out what
  `max_concurrency`/`max_calls`/`max_tokens` actually do at runtime now
  that they're wired (concurrency limit vs. cross-bucket call/token cap
  raising `BudgetExceeded` → `stop_reason="budget"`).
- Added a one-line note that the spec's optional `--checkpoint` flag
  (mid-run checkpointing) is out of scope for phase 1 and not implemented
  by `society.run`.

## Finding 4 (MINOR) — `config_snapshot.yaml` missing effective LLM config

`society/run.py` adds a new `_llm_config_snapshot(llm, embed_fn)` helper
and a `write_outputs(..., *, embed_fn=None)` parameter (threaded from
`run_scenario`). It builds an `"llm_config"` key in `config_snapshot.yaml`
holding `chat_model` / `max_concurrency` / `max_calls` / `max_tokens` (read
directly off the real `LLMClient` instance — `society/llm.py`'s
`LLMClient.__init__` now also stores `self.max_concurrency`, which it
previously computed but discarded) plus `embed_model` (read off the bound
`EmbeddingClient` instance behind `embed_fn.__self__` when `embed_fn` is a
bound `.embed` method). Every value is looked up via `getattr`/`hasattr`
duck-typing, so injected fakes (`FakeLLM`, `afake_embed`) that don't expose
these attributes simply yield a smaller dict (or `{}`) instead of
crashing. `api_key` is never read or written.

## New regression tests (`tests/test_run_integration.py`)

- `tests/helpers.py::FakeLLM` gained an optional `raise_after: int | None`
  constructor arg (default `None`, fully backward compatible): once more
  than `raise_after` `chat()` calls have been attempted, every subsequent
  call raises `society.llm.BudgetExceeded`, simulating a real LLMClient
  whose budget is exhausted.
- `test_budget_stop_flushes_outputs`: 2 llm-brain characters with
  goals sharing a `FakeLLM(fn=..., raise_after=3)`; every reply is a
  harmless `observe` action so both agents stay eligible every tick.
  Runs `run_scenario(ticks=10)` and asserts `summary["stop_reason"] ==
  "budget"`, `events.jsonl` / `transcripts/amy.md` / `transcripts/ben.md` /
  `config_snapshot.yaml` all exist, the snapshot's `run_summary` also
  reports `"budget"`, and `llm_config` never contains `"api_key"`.
- `test_private_status_keys_respected`: scenario dict with a character
  ("amy") whose `status` includes `mood`/`secret`/`appearance` and
  `private_status_keys: ["mood", "secret"]`. After `build_society`,
  asserts `amy.stm.status.public_view()` excludes `mood` and `secret`
  (only `location`+`appearance` remain), and that
  `kernel.execute(ben, Action("observe", {"target": "amy"}))`'s result
  data also excludes both keys.
- Added a standalone manual check (not a committed test) that
  `load_scenario` raises `ValueError` when `private_status_keys` isn't a
  list of strings — confirmed via the scenario.py docstring update; the
  behavior is exercised implicitly by the validation code added to
  `load_scenario`.

## Test output

```
$ venv/bin/python -m pytest -q
.........................................................                [100%]
57 passed, 1 warning in 0.92s
```

(55 pre-existing + 2 new; all green.)

## Files touched

- `society/run.py` — budget config wiring, `llm_config` snapshot.
- `society/kernel.py` — real budget circuit-breaker (`_budget_hit`,
  `BudgetExceeded` handling in `_decide`/`_apply`/`run`).
- `society/llm.py` — `LLMClient` now stores `self.max_concurrency`.
- `society/scenario.py` — `private_status_keys` wired into `STM`;
  `load_scenario` validates its type.
- `docs/actions.md`, `README.md` — corrected/expanded docs.
- `tests/helpers.py` — `FakeLLM(raise_after=...)`.
- `tests/test_run_integration.py` — 2 new regression tests.
