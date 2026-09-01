# `src/contracts.py` change — conversational retrieval + relaxation diagnostics

**Status:** landed on `main` before role work started, by team agreement (normally
`contracts.py` needs unanimous sign-off — this is that sign-off, done up front so nobody
branches off a contract that is about to move).

**Integration status:** complete. R4 (`src/agent.py`) now supplies the conversational request
fields, activates the catalog category hard filter, and copies retrieval diagnostics back into
`SessionState`. R1 remains the owner of `src/retrieval/`.

**Proven non-breaking:** `make test` 17/17 green, `make eval-fast` byte-identical to the
pre-change skeleton (`hit=0.08, mrr=0.035, mttc=10.26, score=0.0653` on fast-50). Every new
field has a default that reproduces the old single-shot behaviour, so the skeleton runs
unchanged for older callers that omit them.

---

## Why

The R1 build plan's Phase 6–7 (category hard-filter + relaxation ladder, then Rocchio
relevance feedback / query accumulation / profile blending) cannot function while
retrieval is a stateless `search(index, request, config)` call that receives only the
current query. It needs (a) session identity so it can keep per-session vector state,
(b) the accumulated negative/positive feedback, (c) the user profile, (d) an override
signal, and it needs to hand back (e) the surviving candidate-pool size and (f) which
constraints the relaxation ladder had to drop — R3's over-generality "ask a narrowing
question" trigger consumes exactly (e)+(f).

Rather than renegotiate the contract mid-hackathon, all of that is added now.

---

## What changed

### 1. `RetrievalRequest` — six new optional fields

| field | default | populated by | read by |
|---|---|---|---|
| `session_id: str` | `""` | agent.py (from `SessionState.session_id`) | retrieval — keys its own per-session state |
| `turn: int` | `0` | agent.py (from `SessionState.turn`) | retrieval — recency weighting |
| `negatives: list` | `[]` | agent.py (from `SessionState.negatives`) | retrieval — Rocchio γ term |
| `accepted: list` | `[]` | agent.py (usually empty in this eval) | retrieval — Rocchio β term |
| `intent_changed: bool` | `False` | agent.py (from `DialogResult.intent_override`) | retrieval — hard-resets the accumulated query vector |
| `profile: dict` | `{}` | agent.py (from `SessionState.profile`) | retrieval — profile blending from turn 1 |

All defaulted, all after the five existing non-default fields, so
`RetrievalRequest(canonical_query=…, intent=…, hard_filters=…, soft_prefs=…, top_k=…)`
still constructs.

### 2. `RetrievalResult(list)` — new return type for `search()`

A `list[Candidate]` **subclass** (not a frozen dataclass) that also carries:

- `pool_size: int` — candidates surviving the hard category filter before top-k truncation
- `dropped_constraints: list` — relaxation-ladder constraints dropped this call, in drop order

It is a `list` subclass on purpose: `search()` used to return a bare `list[Candidate]`, and
keeping that literally true means `ranking`, `agent._ensure_top_k`, and the fallback path
all work with zero changes. Consumers that want the diagnostics read the two attributes;
consumers that don't are untouched. `bm25.search()` already returns one (with
`pool_size = len(candidates)`, `dropped_constraints = []` — it runs no ladder).

### 3. `SessionState` — two new fields

`retrieval_pool_size: int = 0`, `dropped_constraints: list = field(default_factory=list)`.
agent.py copies these off the `RetrievalResult` each turn so **R3's dialog component reads
them straight off the `SessionState` it already receives** — no cross-component import.

### 4. `DialogResult` — two new fields

`intent: str = ""` (`"buy"`/`"browse"`/`""`), `intent_override: bool = False`. R3 fills them;
`NullDialog` leaves the defaults (behaviour identical to today). agent.py uses `intent` to
set `SessionState.intent` and forwards `intent_override` to
`RetrievalRequest.intent_changed`.

---

## The agent.py glue integrated by R4

In `_respond_unsafe`, where the `RetrievalRequest` is built and retrieval is called:

```python
request = RetrievalRequest(
    canonical_query=state.canonical_query,
    intent=state.intent,
    hard_filters=category_hard_filter,
    soft_prefs=memory_profile.boosts,
    top_k=effective_top_k,
    session_id=key,
    turn=turn,
    negatives=list(state.negatives),
    accepted=[],
    intent_changed=bool(getattr(dialog_result, "intent_override", False)),
    profile=state.profile,
)
result = self._call_with_fallback(
    "retrieval",
    retrieval_primary, self._fallback_candidates,
    self.config["timeouts"]["retrieval_seconds"],
    self.index, request, self.config,
)
candidates = list(result)
state.retrieval_pool_size = getattr(result, "pool_size", len(candidates))
state.dropped_constraints = list(getattr(result, "dropped_constraints", []))
```

`_fallback_candidates` may keep returning a plain `list` — the `getattr` guards handle it.
This is the active production path. Plain-list fallbacks remain supported by the guarded
diagnostic reads.

---

## Ownership boundary

R1 does not edit `src/agent.py`. Query accumulation is live; profile blending and Rocchio remain
independently config-gated because the former measured negative and the evaluator supplies no
accepted/rejected-product feedback for the latter. Their offline ceilings remain measurable via
`eval/recall_probe.py`.
