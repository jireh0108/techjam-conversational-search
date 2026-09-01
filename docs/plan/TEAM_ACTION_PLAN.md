# TechJam Build Plan

A working, scoring, integrated skeleton already exists on `main`. This document is for anyone
picking up a piece of it: what's already true, how to add to it without breaking it, and who
builds what next. If you only read one section, read "The loop" — it's the same four steps for
every person on every task.

## Where things stand

- The full pipeline (dialog → memory → retrieval → ranking → agent) runs end to end and scores
  **exactly the same** as the original weak BM25 baseline: `hit_rate_at_10=0.133333,
  mrr=0.073378, technical_score=0.114147` on the 150-session dev split. Every "smart" component
  right now is a stub that does the dumbest safe thing — the score is 100% BM25, 0% intelligence.
  That's expected. It proves the wiring works before anyone builds anything clever on top of it.
- Four facts about how this gets scored change what's worth building. Full detail in
  `CLAUDE.md` and `RECON.md`; the short version:
  1. The simulated customer only ever reacts to the structured `ask_attribute` field you return —
     never to your `message` text. Writing a beautifully-worded question buys nothing; picking the
     *right attribute* buys real information.
  2. Recommendations are scored on every turn, not just the last one. Always send your current
     best 10 guesses, even on a turn where you're also asking a question.
  3. The catalog has no `brand` field and `price` is missing on 79% of products. `store` is the
     closest thing to a brand signal. Don't hard-filter on either — soft-score them instead.
  4. There's no user ID anywhere in the contract, so there's no way to remember a shopper between
     sessions. Memory work is scoped to *within* one conversation only.

## The loop

Every task below — no matter who's doing it — is the same four steps:

1. **Write the logic as a plain function.** Same inputs, same output shape as the stub (the
   `null_*.py` file in your folder) it's replacing. Don't touch `agent.py`. Don't touch anyone
   else's folder.
2. **Unit test it alone.** Feed it fake data, check the output looks right. No catalog, no
   evaluator — just your function. This should take seconds to run.
3. **Plug it in.** Point the `primary` import in your folder's `__init__.py` at your new function
   instead of the null one.
4. **Run the two checks, always in this order:**
   - `make test` — the full suite must stay green. A failure here means you broke a component or
     the failure contract
     (something you wrote can crash or hang the agent), which is a different and more urgent
     problem than a bad score.
   - `make eval-fast` — compare the score to what it was before your change.

## Testing, cheapest to most expensive

| Layer | Command | What it proves | Cost |
|---|---|---|---|
| Unit test | (your own test file) | Your function does the right thing on data you made up | milliseconds |
| Failure contract | `make test` | The whole agent survives your component crashing or hanging | ~3 seconds |
| Fast eval | `make eval-fast` | Your change actually helps (or hurts) real sessions | ~15 seconds |
| Full eval | `make eval` | Same, on all 150 dev sessions — run before merging | ~30 seconds |
| Holdout | `make eval-holdout` | Are we overfitting to the public set? | run rarely, by agreement |

**If a score looks suspiciously unchanged or suspiciously bad**, don't trust it and move on —
check first that your component isn't silently failing and falling back to its Null path. That's
a real bug that happened during the skeleton build: a broken component still produced a plausible,
non-zero, wrong number instead of an obvious crash.

**Never run `make eval-holdout` while developing.** It's the only honest signal the team has for
whether the private 800-session set will disagree with the public 200 — spend it carefully, at
agreed checkpoints, not as a personal sanity check.

## Five-person plan

Ownership matches the folders the skeleton already set up. Each list is roughly sequential — do
item 1 before item 2.

### R1 — Retrieval · `src/retrieval/`

**Status: complete (all 7 build phases). Committed pipeline TechScore 0.1278 → 0.1693 (+32%),
Recall@100 0.573 → 0.773, dev-150. Full phase-by-phase log, ablation table and negatives in
`docs/r1_log.md`. 76/76 `make test`; `make eval` = 0.169271.**

- [x] Embed all 50k products (`title + categories[2:5] + store + top features + key details`).
      bge-small-en-v1.5, 384-dim, SHA-keyed `.npy` cache (`.cache/`, gitignored). BLAIR tested and
      rejected (−12pt R@100, 5× slower).
- [x] Dense-search function (`dense_search` / `dense_search_batch`), same `list[Candidate]` shape,
      plain numpy cosine, no FAISS. **Dense alone loses to BM25 (R@100 0.43 vs 0.64)** — near-
      duplicate catalog buries the gold; it only earns its place in fusion.
- [x] Fusion (`fusion.py`). **RRF was a wash (TechScore +0.0003)** — flat `1/(k+rank)` demotes
      BM25's rank-1/2 golds. **z-score fusion (magnitude-aware), bm25:3/dense:1, is the one that
      works: +0.026 alone.** `enabled: false` — costs the `torch` dependency + a 77 MB cache that
      must ship or rebuild (~30 min).
- [x] Category **hard** filter + relaxation ladder (`postprocess.py`). R4 now supplies R3's
      most-specific catalog category on the live path. Price/store remain soft boosts only.
- [x] Relaxation: pool below `min_pool_size` → drop lowest-priority filter, retry, report in
      `RetrievalResult.dropped_constraints`.
- [x] **Popularity prior (`postprocess.py`) — the big lever, `enabled: true` w=0.2: TechScore
      0.1278 → 0.1693.** 148/150 dev targets are above catalog-median rating count (median at the
      99.5th percentile); the evaluator picks popular products as targets and the spec says the
      private set is built the same way.
- [x] Multi-turn (`multiturn.py`) — query accumulation is live through R4's
      `RetrievalRequest.{session_id,turn,...}` wiring and resets on explicit intent overrides.
      Profile-blend measured and rejected (net-negative on the full pipeline). Rocchio shipped but
      unmeasurable (no accept/reject signal in this eval).
- [x] `eval/recall_probe.py` — standalone Recall@{10,50,100,500} harness (+`--dense`,
      `--multiturn`), re-run after every change. Oracle line stays 1.000 (index is healthy).

**Remaining R1 headroom (deferred, not blocking integration — detail in `docs/r1_log.md`):**

1. **Popularity weight is conservative (0.2).** The dev-150 curve keeps climbing to TechScore
   0.28 at w=0.5; fusion + w=0.3 hits 0.2977. Left low on purpose — measured with NullReranker,
   untested on holdout, and a high weight makes the agent ignore the conversation. **R4 should
   sweep `retrieval.popularity.weight` on the holdout with the real reranker in place.**
2. **Fusion is off.** One config line (`retrieval.fusion.enabled: true`) once the team accepts
   the `torch` dependency and ships/builds the embedding cache. +0.026 alone, +0.17 stacked with
   a higher popularity weight.
3. **Query→document vocabulary mismatch** — Phase 1 named this the core problem (oracle R@100 =
   1.0, real turn-1 R@100 = 0.57). Stopwords helped; no synonym expansion or learned query
   rewriting was tried. Likely the largest untapped lever.
4. **Popularity transform variants** — only `log1p(rating_number)` min-max blend was tried; not
   `average_rating` weighting, percentile-rank, or `rating × log(count)`.
5. **Fusion weights swept coarsely** (bm25 1/2/3, k 20/40/60/100); no fractional or per-scenario.
6. **Dense doc-template A/B used an 8k-distractor subsample** (Phase 3); worth re-confirming on
   the full 50k if fusion is turned on.
7. **Boundary scenario** (n=8, R@100 0.625) is essentially unaddressed.

Items 1-2 need the real R2/R3 and a team decision; 3-7 are offline-testable R1 work if there is
time after integration.

### R2 — Ranking · `src/ranking/` — complete

- [x] Install `ms-marco-MiniLM-L-6-v2` and verify the local inference path.
- [x] Wire it in as the real reranker behind `NullReranker`'s exact signature. Leave
      `NullReranker` itself alone — it's the permanent fallback, not scaffolding.
- [x] Cover rendering, ordering, invalid output, model failure, and configured depth in tests.
- [x] Add an optional LLM listwise reranker behind a config flag and key-presence check; it is
      intentionally off by default.

### R3 — Dialog · `src/dialog/` — complete

- [x] Derive dialog behavior from the evaluator and catalog rather than nonexistent transcripts.
- [x] Build a deterministic buy-vs-browse classifier from the latest message.
- [x] Build slot extraction from words that actually appear in the catalog (regex/keyword list),
      not a general-purpose parser.
- [x] Build the config-ordered `ask_attribute` picker specified in
      `r3_dialog_implementation.md`, including no-preference and configurable turn guards.
- [x] Build override detection with a marker-word list (`actually`, `instead`, `never mind`,
      `change of plans`) — the override message follows a fixed template, so this is easy to catch.
- [x] Always attach the current best 10 recommendations alongside any question — never send a
      question with an empty list.

### R4 — Agent & Eval · `src/agent.py`, `eval/`

- [x] Integrate all primary components while retaining explicit Null fallbacks.
- [x] Forward the complete conversational retrieval contract and copy relaxation diagnostics.
- [x] Isolate component timeout workers so one hung dependency cannot stall the other stages.
- [x] After every swap: run tests and fast evaluation, logging the result.
- [ ] Build a simple ablation table (BM25 / +dense / +cross-encoder / +dialog) so the team can see
      which piece is actually earning its score, not just guess.
- [x] Own `config.yaml` — behavior-affecting thresholds, including the turn guard, live there.
- [x] Gatekeep the holdout split. Run `make eval-holdout` at agreed checkpoints only, and announce
      the result to the team rather than letting people check it ad hoc.

### R5 — Memory & Docs · `src/memory/`, `docs/`, demo

- [x] Build intra-session distillation only — compress this session's rejected items and confirmed
      constraints. There's no user ID anywhere in the contract, so cross-session memory has
      nothing to attach to; don't build it.
- [x] Feed the distilled state into retrieval's `soft_prefs` as a boost.
- [x] Write the architecture diagram, README reproduction steps, and limitations section — none of
      this blocks anyone else, so it can happen in parallel from day one.
- [x] Restore the catalog and run the full checks; 103/103 tests pass.
- [x] Integrated dev-150 score is `0.773504`; holdout-50 is `0.753888`, with deterministic
      metric fields across two full runs.
- [ ] Record the demo once the team beats baseline end to end; capture the moment where the agent
      asks a question, narrows down, and hits.

## Ground rules for everyone

- A component never imports another component. Only `src/agent.py` is allowed to know about all
  of them.
- Every tunable number lives in `config.yaml`. If it affects behavior, it doesn't belong in code.
- Don't delete a `null_*.py` file after replacing it — it's the permanent fallback, not a draft.
- Don't edit `evaluator/local_evaluator.py` or `tests/test_evaluator.py`.
