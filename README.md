# CodeShift — Conversational E-Commerce Search Agent

CodeShift is our submission for the TechJam Conversational E-Commerce Search Challenge: a deterministic shopping agent that asks useful follow-up questions and recommends a customer's hidden target product within at most 10 turns.

## Project overview

The challenge requires an agent to convert a short, evolving shopping conversation into ranked catalog recommendations. CodeShift uses a modular pipeline:

`dialog → memory → retrieval → ranking → recommendations`

- **Dialog** extracts catalog-grounded categories and constraints, identifies intent changes, and selects the next allowed clarification attribute.
- **Memory** preserves useful in-session preferences and negative signals without creating a cross-session user profile.
- **Retrieval** uses SQLite FTS5 BM25, category-aware candidate handling, popularity-aware ordering, and multi-turn query accumulation.
- **Ranking** preserves a deterministic baseline while supporting optional local cross-encoder and API-based listwise reranking paths.

Every stage has a fail-safe fallback. The agent continues returning valid, unique catalog IDs if an optional component is unavailable, slow, or malformed.

## Setup and installation

The deterministic core requires Python, PyYAML, NumPy, and the provided catalog. Python 3.10 or 3.11 is recommended, especially when enabling the optional local ML dependencies.

```bash
git clone https://github.com/jireh0108/techjam-conversational-search.git
cd techjam-conversational-search
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install PyYAML==6.0.3 numpy==2.4.3
```

For optional dense retrieval or local cross-encoder ranking, install the additional packages listed in `requirements.txt` using a Python version supported by the pinned PyTorch wheel. The core agent remains functional when those optional packages or model artifacts are absent.

## Dataset

The project uses the `Clothing_Shoes_and_Jewelry` subset of Amazon Reviews 2023, published by the McAuley Lab at UCSD. The competition package provides a frozen 50,000-product catalog and a 200-session labeled public development set; the private evaluation set is not included.

Download `catalog.jsonl.gz` from this repository's GitHub Release, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the download with the published `SHA256SUMS` file. See `DATA_ATTRIBUTION.md` for data-use details.

## Reproduce the recorded results

From the repository root, activate the environment created above and ensure `data/catalog.jsonl` is present.

```bash
python3 -m unittest discover -s tests -v
python3 -m eval.run_eval --mode fast
python3 -m eval.run_eval --mode full
```

On Linux, macOS, or WSL systems with `make`, the equivalent commands are `make test`, `make eval-fast`, and `make eval`. On Windows, run the Python commands directly.

The fast evaluation uses the configured development sample; the full evaluation uses the 150-session development split. Results are written to `eval/results_log.jsonl`.

The recorded integrated development result is **0.773504 TechnicalScore** on dev-150, with **0.94 Hit Rate@10**, **0.546347 MRR**, and **4.02 MTTC**. The recorded holdout-50 result is **0.753888 TechnicalScore**. Scores should be compared with the weak starter baseline in `docs/baseline_results.json`; exact reproduction assumes the same catalog, configuration, and optional-model availability.

## Architecture and session memory

The agent is an orchestration layer around independent, replaceable components:

```text
customer turn
     |
     v
  dialog -----> SessionState <----- reset(user_profile)
     |                |
     |                v
     +---------> memory distillation
                       |
                       v
                MemoryProfile.boosts
                       |
                       v
              retrieval soft preferences
                       |
                       v
                 ranking -> Top 10
```

Memory is deliberately intra-session only. It compresses confirmed constraints, rejected signals, and useful anonymized preference tags into retrieval soft boosts. There is no cross-session store because the contract provides no stable user ID.

## Limitations and future improvements

- Memory can only use information exposed by the current session and the dialog component; it cannot recover preferences that were never stated or extracted.
- Rejection language is ambiguous, so negative signals are soft penalties rather than hard filters.
- The catalog has sparse prices and no reliable top-level brand field; store and nested brand values are best-effort signals.
- Retrieval can only help if the target product enters its candidate pool. Ranking and memory cannot recover a product that retrieval never returns.
- Evaluation requires an exact `parent_asin` match and scores recommendations on every turn.
- The local evaluator's generated conversations are deterministic and may not represent all private evaluation behavior.

Given more time, we would validate the local cross-encoder on a fully supported ML runtime, improve structured filters beyond categories, and make clarification selection adaptive to candidate-pool uncertainty. We would also run further ablations on the held-out data before enabling more expensive dense or API-based ranking paths.

## Team contributions

CodeShift is a five-person team of sophomore and junior computer science students. To work effectively across busy schedules and an entirely virtual workflow, we split ownership by component while sharing integration, testing, and review responsibility:

- **R1 — Retrieval ([`@jireh0108`](https://github.com/jireh0108)):** BM25 search, candidate-pool processing, category constraints, multi-turn retrieval, and retrieval evaluation.
- **R2 — Ranking ([`@bevanpoh`](https://github.com/bevanpoh)):** local cross-encoder support, optional listwise reranking, and safe ranking fallbacks.
- **R3 — Dialog ([`@chungdarren123`](https://github.com/chungdarren123)):** catalog-grounded slot extraction, intent and override handling, canonical query construction, and clarification selection.
- **R4 — Agent and evaluation integration ([`@toxicpeanuts`](https://github.com/toxicpeanuts)):** component orchestration, failure/timeout handling, evaluation, and regression coverage.
- **R5 — Memory and documentation ([`@axelheng`](https://github.com/axelheng)):** intra-session preference distillation, retrieval soft preferences, architecture, and submission documentation.

All team members contributed to virtual coordination, Git-based review, conflict resolution, final testing, and evaluation.
