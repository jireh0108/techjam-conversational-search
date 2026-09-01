# CLAUDE.md — Operating Manual

TechJam Conversational E-Commerce Search Challenge (72h hackathon). Build a multi-turn shopping
`Agent` that finds a hidden target product (`parent_asin` from a frozen 50k-item Clothing catalog)
as early and highly ranked as possible, within 10 turns. Submission = one `Agent` entry file +
helpers + setup instructions + a short report. Full detail: `docs/plan/RECON.md`,
`docs/plan/FEASIBILITY.md`.

## THE FOUR EVALUATOR FACTS (everything downstream depends on these)

1. **The user simulator reacts ONLY to the structured `ask_attribute` field, never to the
   free-text `message`.** It is deterministic per sample (seeded by `sample_id`+`scenario_type`),
   but not a fixed script — what it says next depends on which `ask_attribute` we ask for.
   Writing clever natural-language questions extracts nothing extra; only picking the right
   `ask_attribute` value does. (`evaluator/local_evaluator.py:166-185`)
2. **Recommendations are scored every turn, not just the last one.** First turn the target
   appears in the top-10 wins the session. (`local_evaluator.py:251-255`)
3. **Scoring:** `TechnicalScore = 0.50*HitRate@10 + 0.30*MRR + 0.20*Efficiency`,
   `Efficiency = clip((11-MTTC)/10, 0, 1)`, miss = turn 11 for MTTC, 0 for MRR.
   (`local_evaluator.py:278-280`)
4. **Failure handling is asymmetric.** `respond()` exceptions / malformed output are caught and
   downgraded to an empty-recommendation miss for that turn only (`local_evaluator.py:239-244`).
   **`reset()` exceptions are NOT caught — they crash the entire evaluation run.**
   `reset()` must never be allowed to raise.

## Agent entry-point interface (verbatim, do not change signatures)

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None: ...
    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": str,
            "ask_attribute": "category"|"material"|"color"|"size"|"style"|"brand"|"budget"|"feature"|"use_case"|"other"|None,
            "recommendations": [{"parent_asin": str, "score"?: number}, ...],  # only first 10 valid unique scored
            "usage": {"prompt_tokens": int, "completion_tokens": int},  # optional
        }
```

State is carried entirely by the Agent instance, keyed by `session_id` — the harness never
replays prior turns back to `respond()`. `starter/agent.py` is a thin shim re-exporting
`src.agent.Agent` (the real implementation) — the evaluator's import path never changes.

## Repo map

```
data/            catalog.jsonl (50k, gitignored), public_set.jsonl (200 sessions)
docs/            spec/contracts/scoring config/baseline reference — read-only, don't edit
docs/plan/       RECON.md, FEASIBILITY.md, strategy.md, architecture.md (gitignored)
evaluator/       local_evaluator.py — DO NOT EDIT
starter/agent.py thin shim: `from src.agent import Agent`. No logic here, ever.
tests/           evaluator, component, integration, and failure-contract coverage
src/contracts.py FROZEN. Every cross-component dataclass. Changes need unanimous agreement.
src/config.py    shared config-loader (not a "component" — every leaf may import it)
src/retrieval/   R1. BM25 + popularity live; category/multi-turn live; dense fusion optional.
src/ranking/     R2. Local cross-encoder primary; optional key-gated listwise; Null fallback.
src/dialog/      R3. Catalog-grounded slots, routing, overrides, and structured questions.
src/memory/      R5. Live intra-session distillation with permanent Null fallback.
src/agent.py     R4. The only module that imports every component.
eval/            R4. run_eval.py, generate_split.py, dev_holdout_split.json (committed), results_log.jsonl
config.yaml      every tunable — grep src/ to verify no magic numbers outside it.
Makefile         eval / eval-fast / eval-holdout / test / smoke targets.
results.json     evaluator output, gitignored, regenerated each `py -m evaluator.local_evaluator` run.
```

One owner per directory; nobody edits another's directory except `agent.py`, which knows the graph.

## Architectural invariants (implemented, not aspirational)

- Components are leaves, never import each other; only `src/agent.py` imports the whole graph
  (verified: `retrieval/`, `ranking/`, `dialog/`, `memory/` each only import `src.contracts`/`src.config`).
- Every component is a pure function over `src/contracts.py` dataclasses.
- Every component has a permanent Null implementation, not scaffolding: `NullDialog` → raw
  utterance as `canonical_query`, `ask_attribute=None`. `NullMemory` → empty boosts/summary.
  `NullReranker` → input order unchanged. Retrieval's own fallback is the BM25 baseline itself; if
  BM25 fails too, `agent.py._fallback_candidates` drops to a precomputed rating-sorted pad pool.
- Behavior-affecting tunables live in `config.yaml`, including the contract turn guard.
- Failure contract, enforced in `agent.py`: every component call goes through
  `_call_with_fallback()` (one isolated worker per component, with `config.yaml` timeouts),
  falling back to the **explicit** `null_*.py` function — never re-invoking the primary — on any
  exception or timeout. A stuck ranker cannot queue dialog or retrieval behind it. `reset()` has
  its own nested try/except and can never propagate.
  `respond()`'s outer try/except catches even a bug in `agent.py`'s own glue code.
  `_ensure_top_k()` guarantees exactly `top_k` valid, unique IDs. Proven in
  `tests/test_failure_contract.py` (14 tests).

**Gotcha:** `agent.py` queries the shared BM25 SQLite connection from the `ThreadPoolExecutor`
worker thread, not the thread that built it. `sqlite3.connect` defaults to
`check_same_thread=True`, which silently raised on every retrieval call during Phase 2 build — the
try/except swallowed it and every response quietly fell back to the pad pool, still "scoring"
instead of crashing loudly. Fixed with `check_same_thread=False`
(`src/retrieval/bm25.py:build_index`). Any non-thread-safe resource needs the same care.

## Real, tested commands

Use `python3` on Linux/WSL/macOS or `py` on Windows. The Make targets are convenience aliases;
run the corresponding module command directly where `make` is unavailable.

- `make eval` / `python3 -m eval.run_eval --mode full` — all 150 dev sessions.
- `make eval-fast` / `python3 -m eval.run_eval --mode fast` — first 50 of the 150 dev sessions.
- `make eval-holdout` / `python3 -m eval.run_eval --mode holdout` — 50 holdout sessions. **Use
  rarely** — the only defense against overfitting to the public set.
- `make test` / `python3 -m unittest discover -s tests -v` — full suite.
- `make smoke` — failure-contract tests only.
- Original unmodified baseline: `py -m evaluator.local_evaluator`.
- Catalog download/verify/unpack: see `docs/plan/RECON.md` V4 (already done here).

## Frozen `src/contracts.py`

`src/contracts.py` is the authoritative contract. Do not duplicate its dataclasses in docs: the
live `RetrievalRequest` includes session, turn, feedback, override, and profile fields, while
`RetrievalResult` and `SessionState` carry pool-size and relaxation diagnostics.

## Per-directory ownership

| Dir | Owner | Null does today | Replacing it means |
|---|---|---|---|
| `src/retrieval/` | R1 | Rating-sorted fallback below BM25 | Maintain lexical/dense/filter/multi-turn retrieval behind `search()` |
| `src/ranking/` | R2 | Identity pass-through | Maintain local CE and optional listwise ranking |
| `src/dialog/` | R3 | Echoes utterance, `ask_attribute=None` | Maintain catalog-grounded dialog and question policy |
| `src/memory/` | R5 | Empty boosts/summary | Maintain intra-session distillation only |
| `src/agent.py`+`eval/` | R4 | orchestrator/harness itself | Keep every fallback pointed at the explicit `null_*` function, never the primary |

## Current scores (all actually run, 2026-09-01)

- Original baseline, 200 public sessions: `hit=0.125, mrr=0.068034, mttc=9.81, score=0.10671`
  (matches `docs/baseline_results.json`).
- Integrated fast-50: `hit=0.96, mrr=0.567786, mttc=3.62, score=0.797936`, 58.054s.
- Integrated dev-150: `hit=0.94, mrr=0.546347, mttc=4.02, score=0.773504`.
- Holdout-50 checkpoint: `hit=0.90, mrr=0.588960, mttc=4.64, score=0.753888`; the modest score
  gap versus dev does not indicate severe public-set overfitting.
- Determinism: dev-150 was run twice; every metric and scenario field was identical. Only
  `wall_clock_seconds` differed (191.486s versus 204.422s).
- Tests: `python3 -m unittest discover -s tests -v` → 103/103 pass, including complete failure
  injection, R4 request forwarding, timeout isolation, and the configurable turn guard.

## Environment constraints

- No LLM API keys or org endpoint anywhere on this machine. LLM components need a runtime key +
  non-LLM fallback, since final scoring **may** run offline (`docs/submission_rules.md:59`).
- `requirements.txt` pins the core and optional local-ML runtime. The cross-encoder remains
  fail-soft and offline (`local_files_only: true`) during scored runs.
- Dense fusion requires a separately packaged embedding cache and remains disabled by default.
- Unresolved: private-800 outcomes and whether final judging disables network entirely. Neither
  affects the deterministic non-LLM fallback path.

## Data facts

- Catalog (50,000 rows): `parent_asin, title, features, description, price, categories, details,
  average_rating, rating_number, store`. **No top-level `brand`** — `details.Brand` covers only
  4.7%; `store` (99.4% present) is the closest proxy, → `ProductMeta.store`. **`price` null in
  78.9%** — `ProductMeta.price` is `float | None`, never hard-filter without a fallback.
  `categories` (breadcrumb list) is the only field always present and non-empty.
- Sessions (200: 80/80/30/10 buying/browsing/override/boundary): conversational content is **not
  stored** — derived at eval time from the target product's fields, seeded by
  `sample_id`+`scenario_type`. Only real slot vocabulary shipped: `user_profile.preference_tags`
  (`fit`, `material`, `comfort`, `style`, `durability`, `performance`, `warmth`, `weather`).
  `purchase_frequency` is constant across all 200 — not a useful signal.
- Dev/holdout split (`eval/dev_holdout_split.json`, committed, seed=42): 150/50, stratified so
  both halves keep the 40/40/15/5 scenario mix.

## Decisions

Full justification in `docs/plan/FEASIBILITY.md`.

| Component | In/Out | One-line reason |
|---|---|---|
| Dual-track Buy/Browse router | **In** | `scenario_type` never sent to the agent — inferred either way. |
| BM25 + dense + RRF fusion | **In** | BM25 exactly matches baseline through the full pipeline; embedding libs downloadable. |
| Structured (category) route | **In, category-only** | `categories` always populated; `brand`/`price` become soft signals only. |
| Cross-encoder rerank | **In, primary ranker** | Wheels available; still needs a real install + smoke test. |
| LLM listwise rerank | **In, optional, off by default** | Zero API keys on this machine; gate behind config + key check. |
| Dialogue state + override detection | **In** | Override turns are a fixed literal message — regex detection suffices. |
| Clarification policy | **In, `ask_attribute`-only** | Simulator never reads `message` — only the enum choice scores. |
| Cross-session long-term memory | **Out** | No user ID anywhere in the contract to key a store by. |
| Intra-session distillation | **In** | Fully supported by state the agent already owns privately. |

## Do not do this

- Delete a `null_*.py` implementation after a real component replaces it — they're permanent.
- Let a component import another component — only `src/agent.py` imports across the graph.
- Tune anything on `eval/dev_holdout_split.json`'s `holdout` list — dev-only for iteration.
- Hard-code a behavioral tunable in `src/` instead of `config.yaml`.
- Edit `evaluator/local_evaluator.py` or `tests/test_evaluator.py` — reuse via `eval/` instead.
- Open a shared resource (DB, file handle, model) without checking thread-safety against
  `agent.py`'s `ThreadPoolExecutor` worker thread — see the SQLite gotcha above.
