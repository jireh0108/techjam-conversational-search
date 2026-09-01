# Architecture: The Walking Skeleton and the Joint Block

## The architecture to build first

### Runtime data flow

```text
User message + anonymized profile
              |
              v
     agent.py / SessionState
        |              |
        v              v
      dialog       memory.distill
        |              |
        +-------> canonical query
                       + MemoryProfile.boosts
                                  |
                                  v
                         retrieval.soft_prefs
                                  |
                                  v
                           ranking -> Top 10
```

Memory is scoped to the `SessionState` created by `reset()`. It is not persisted between sessions: `session_id` identifies a conversation for the running agent, not a reusable shopper identity.

One rule drives everything: **`agent.py` orchestrates, components are leaves.** No component ever imports another component. All state flows one direction through a single turn function. This is what makes five people on one pipeline safe.

Four properties make this architecture safe for five people:

1. **Components are leaves.** `retrieval/` never imports `dialog/`. `ranking/` never imports `memory/`. Only `agent.py` knows the whole graph. This is why five people can work simultaneously — the dependency graph is a star, not a mesh.
2. **Every component is a pure function over dataclasses.** `f(state, input) -> output`. No hidden mutation, no shared globals. This makes every component independently stubbable and testable, which is what the joint tasks below exploit.
3. **Every component has a Null implementation** that returns valid, empty-but-well-formed output. `NullReranker` returns input order. `NullDialog` returns the raw utterance as `canonical_query`. `NullMemory` returns an empty profile string. These get written first and never deleted — they're your permanent fallback when someone's component breaks at 3am.
4. **All tunables live in one `config.yaml`.** Route weights, RRF `k`, rerank depth, clarification thresholds, decay rate. Nothing hard-coded anywhere. Phase 3's sweep is impossible otherwise, and you don't want to be grepping for magic numbers at T-30h.

---

## The joint block: T-72 to T-64

Nobody claims a pillar until all eight tasks are green. All five people, one room, one screen where it matters. This looks like a lot of shared time — it is roughly 11% of your budget, and it buys you the other 89% free of coordination cost.

### 1. Kick off the embedding job — first 20 minutes

Before anything else. Pick `all-MiniLM-L6-v2`, write the naive document template (title + brand + category_path), start encoding all 50k, cache to `.npy`. It runs in the background for the rest of this block. You will refine the template later; you cannot refine an embedding you haven't computed.

### 2. Evaluator forensics — read it together

Open the scoring script and the session-replay loop on one screen and read it out loud as a group. You are answering:

- Are user turns a **fixed script**, or generated in response to your agent? (The question from before — this changes the whole system.)
- Is the recommendation list scored **every turn** or **only the final turn**?
- Exactly how does **MTTC combine with MRR** in `TechnicalScore`? Write the formula on the whiteboard.
- What happens on a **malformed response** — zero for the turn, or crash the session?

Every architectural decision for the next 70 hours descends from these four answers. One person reading this alone and summarizing to the others loses too much.

### 3. Session read-along — 40 sessions, 8 each

Everyone reads real dev sessions and pastes observations into one shared doc. You are hunting for: what slots actually appear, how overrides are actually phrased, how the gold product relates to the opening utterance, and how long a typical session runs. This is where your slot vocabulary and your override regex list come from — the data, not your imagination. Budget 45 minutes.

### 4. Freeze `contracts.py`

30-minute meeting. Agree exact key names — `color` not `colour`, `price_max` not `max_price`, `category_path` as a list not a slash-joined string. Write the file, commit it, and declare it frozen. Changes from here require all five to agree, and there should be at most one or two for the rest of the hackathon.

### 5. Build the walking skeleton — the centerpiece

This is the task that matters most. Build the entire pipeline end-to-end using **only Null implementations plus the starter BM25**. Every module returns valid types. Nothing is intelligent. The agent responds with 10 ASINs from BM25 and it scores something non-zero.

Do this as a group, on one machine, with two people driving and three watching. It takes about 90 minutes.

**The payoff:** from the moment it's green, integration is already done. Every subsequent task is "replace one Null with a real implementation behind an unchanged signature." There is no integration day. There is no big-bang merge at T-20h where nothing works and nobody knows whose fault it is. That failure mode kills more hackathon teams than any modeling mistake.

### 6. Eval harness and scoreboard

Fast loop (50 sessions, target under 90s) and full loop. Dev-150 / holdout-50 split committed to the repo so nobody can accidentally tune on the holdout. A one-line script that prints Hit@10, MRR, MTTC, and TechnicalScore, and appends to a shared results log. Baseline number recorded and posted.

### 7. Repo hygiene

Directory-per-owner structure. `CODEOWNERS` if you want teeth. `config.yaml` with every current constant. Fixed seeds everywhere, verified by running the eval twice and diffing. `make eval` and `make eval-fast` targets — these are also what your README's reproduction section will point at, so build them now rather than reconstructing them at T-12h.

### 8. Failure contract — 10 minutes, but non-negotiable

Agree out loud, as a rule everyone commits to: **no component ever raises, and every response contains exactly 10 valid ASINs.** Timeouts on every external call. Try/except at every component boundary falling back to the Null path. A degraded answer scores; an exception scores zero and can poison the rest of the session.

---

## The gate

Do not split until every line is true:

- [x] Baseline reproduced, number posted
- [x] The four evaluator questions answered in writing
- [ ] 40 sessions read, slot vocabulary drafted from real data
- [x] `contracts.py` committed and frozen
- [x] Walking skeleton runs end-to-end and scores non-zero
- [x] Fast eval under 90 seconds, holdout split committed
- [x] Two identical runs produce identical scores
- [x] Embedding job finished or nearly finished
- [x] Every component has a working Null fallback

Then and only then: R1 takes `retrieval/`, R2 takes `ranking/`, R3 takes `dialog/`, R4 takes `agent.py` and `eval/`, R5 takes `memory/` and `docs/`. Each of them replaces one stub. Nobody touches anyone else's directory for the next 50 hours.

---

## One thing to resist

The temptation to have each person start "just prototyping my part" during this block. It feels like parallelism and it isn't — it produces five components built against five different mental models of the contract, and you'll spend Phase 1 reconciling them instead of improving scores. **Eight hours of genuine convergence beats eight hours of divergence you have to undo.**
