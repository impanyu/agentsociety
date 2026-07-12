# History-Sedimentation Mode Implementation Plan

> Spec: docs/specs/2026-07-11-history-sedimentation-design.md. Execute task-by-task (subagent per task), full suite green after each. Baseline: 67 tests, master @ bbb68a9.

**Goal:** Whole-book init v2 — sediment a book's timeline into shared LTM (time-marked, consensus-merged), simulate the sequel with alive agents (empty goals, self-bootstrapped) and archived dead agents.

## Task H1: LTM story-time metadata + reusable sediment file
- `SharedMemory.remember(..., story_order: int|None=None, story_time: str|None=None)` → metadata fields; consensus merge keeps the SMALLER story_order (earlier fact) and its story_time; export()/restore() carry them (holographic).
- `build_society`: scenario key `ltm_file: <path relative to scenario>` → `await shared.restore(json.load(...))` INSTEAD of seeding seed_memories (seeds still allowed for agents when no ltm_file). Loader validates file exists.
- Tests: story fields persisted + merge-keeps-smaller; ltm_file restore path (zero embed recompute, seeds skipped).

## Task H2: archived agents + goal bootstrap
- Scenario agent key `archived: true` (loader validates bool). Agent.archived attr.
- Kernel: archived → never eligible, excluded from presence, observe(target archived) → ok=False "已故/archived", say/gesture to archived → treated as not present.
- Goal bootstrap: when goal stack empty, kernel-enriched view gains `goal_hint` (zh/en per scenario language — kernel needs language; take from config dict) instructing: recall 自己的过去 → observe 环境 → conclude → push_goal(根本) → push_goal(当前).
- skills zh/en: new "开局自省" pipeline section.
- Tests: archived exclusion trio (schedule/presence/observe), goal_hint appears only when stack empty, skill file mentions 开局自省.

## Task H3: extractor v2 — history mode
- CLI: `--mode {snapshot,history}` (default snapshot = existing path untouched), `--model` override (chat model for this run), `--registry-only`, `--registry <path>`, `--hints`.
- Pass 1 registry: per-chunk summary calls (marker 摘要) → one merge call (marker 注册表) → {characters:[{id,name,aliases,profile}], locations:[...], carriers:[...]} → written to `<output>.registry.json`.
- Pass 2 sediment: per chunk (marker 沉淀) with registry closed-world instruction → {"memories": {id: ["【时间】fact", ...]}, "state_updates": [{id, location|null, alive}], "story_time": str}; memories → shared.remember(id, text, source="history", story_order=chunk_idx*1000+i, story_time=...); unknown ids skipped w/ warning; state last-write-wins along chunk order.
- Assembly: characters alive → agents goals=[], dead → archived:true (location = last known); locations/carriers as usual; `ltm_file` written via shared.export() to `<output>.ltm.json`; kickoff from --hints (LLM call, marker 起始) or generated from final state.
- Tests (FakeLLM routed by markers): registry merge+aliases; closed-world attribution (alias → canonical id, unknown skipped); story_order monotonic; dead → archived; ltm.json written and scenario loads + build_society restores it; --registry reuse skips pass 1.

## Task H4: e2e + docs
- E2E test: 2-chunk micro-novel, full history pipeline (FakeLLM), build_society, run 5 ticks: alive agent bootstraps a goal (scripted brain pushes goal after hint), archived agent silent, multi-owner consensus entry exists from overlapping memories.
- README: history-mode section (commands for ch01-10 run); docs/actions.md: goal-bootstrap pipeline note.

Ledger: .superpowers/sdd/progress.md (append H1-H4 lines).
