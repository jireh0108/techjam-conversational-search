"""The only module allowed to import every component. Orchestrates dialog -> memory ->
retrieval -> ranking into a single turn, enforcing the failure contract at every boundary:
no component call may raise or hang the agent, and every response carries exactly `top_k`
valid, unique catalog IDs.

Phase 2 (the walking skeleton) wires only Null implementations plus the existing BM25 baseline
retrieval -- zero intelligence, full integration. See docs/plan/architecture.md and
docs/plan/FEASIBILITY.md for why each component is scoped this way.

Each component is imported twice on purpose:
  - the "primary" import comes from the package's public __init__ (the swappable slot future
    phases will point at real implementations);
  - the "_null" import comes explicitly from the null_*.py module and is never swapped out.
The fallback path always calls the explicit null function directly -- never the primary again --
so a primary that hangs or raises can never take the fallback down with it.
"""

from __future__ import annotations

import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, TypeVar

from src.config import load_config
from src.contracts import AgentResponse, Candidate, DialogResult, MemoryProfile, RetrievalRequest, SessionState
from src.dialog import build_category_lexicon, update as dialog_primary
from src.dialog.null_dialog import update as dialog_null
from src.memory import distill as memory_primary
from src.memory.null_memory import distill as memory_null
from src.ranking import rerank as ranking_primary
from src.ranking.null_reranker import rerank as ranking_null
from src.retrieval import BM25Index, build_index, search as retrieval_primary

T = TypeVar("T")


class Agent:
    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.config = load_config()
        random.seed(self.config["seed"])  # no component uses randomness yet; fixed for when one does
        self.index: BM25Index = build_index(catalog_path, self.config)
        # This immutable catalog-only structure is intentionally built once: dialog must not
        # invent category aliases from customer text or re-scan the catalog each turn.
        self.category_lexicon = build_category_lexicon(self.index.products, self.config)
        self.top_k_default: int = self.config["contract"]["top_k"]
        self._sessions: dict[str, SessionState] = {}
        # One shared worker thread for all timeout-guarded component calls -- created once to
        # avoid per-turn thread-pool spin-up cost across a multi-hundred-turn eval run.
        self._executor = ThreadPoolExecutor(max_workers=1)

    # ------------------------------------------------------------------ #
    # Required interface (docs/agent_api_contract.json)
    # ------------------------------------------------------------------ #

    def reset(self, session_id: str, user_profile: dict) -> None:
        """Must never raise: the harness does not catch reset() exceptions and an uncaught one
        aborts the entire evaluation run, not just this session (docs/plan/RECON.md E4)."""
        try:
            key = str(session_id)
            profile = user_profile if isinstance(user_profile, dict) else {}
            self._sessions[key] = SessionState(
                session_id=key,
                turn=0,
                intent="unknown",
                slots={},
                slot_turn_added={},
                profile=profile,
            )
        except Exception:
            try:
                key = str(session_id)
                self._sessions[key] = SessionState(
                    session_id=key, turn=0, intent="unknown", slots={}, slot_turn_added={},
                )
            except Exception:
                pass  # absolute last resort -- respond() defensively re-creates state if missing

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        try:
            response = self._respond_unsafe(session_id, user_message, turn, top_k)
        except Exception:
            response = self._fallback_response(top_k)
        return {
            "message": response.message,
            "ask_attribute": response.ask_attribute,
            "recommendations": [{"parent_asin": asin} for asin in response.recommendations],
            "usage": response.usage,
        }

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #

    def _respond_unsafe(self, session_id: str, user_message: str, turn: int, top_k: int) -> AgentResponse:
        key = str(session_id)
        state = self._sessions.get(key)
        if state is None:  # defensive: respond() called without a prior reset()
            state = SessionState(session_id=key, turn=0, intent="unknown", slots={}, slot_turn_added={})
            self._sessions[key] = state
        state.turn = turn
        effective_top_k = top_k if isinstance(top_k, int) and top_k > 0 else self.top_k_default

        dialog_result = self._call_with_fallback(
            dialog_primary, dialog_null, self.config["timeouts"]["dialog_seconds"],
            state, user_message, self.category_lexicon,
        )
        state.canonical_query = dialog_result.canonical_query
        state.slots = dialog_result.slots
        if dialog_result.ask_attribute and dialog_result.ask_attribute not in state.asked_attributes:
            state.asked_attributes.append(dialog_result.ask_attribute)
        state.intent = dialog_result.intent if dialog_result.intent in {"buy", "browse", "unknown"} else "unknown"
        # Record the turn before distillation so current-turn confirmations or rejections
        # are available immediately. The same entry is not appended again after retrieval.
        state.history.append({"turn": turn, "user_message": user_message, "ask_attribute": dialog_result.ask_attribute})

        memory_profile = self._call_with_fallback(
            memory_primary, memory_null, self.config["timeouts"]["memory_seconds"],
            state, state.profile,
        )

        request = RetrievalRequest(
            canonical_query=state.canonical_query,
            intent=state.intent,
            hard_filters={},
            soft_prefs=memory_profile.boosts,
            top_k=effective_top_k,
        )
        candidates = self._call_with_fallback(
            retrieval_primary, self._fallback_candidates, self.config["timeouts"]["retrieval_seconds"],
            self.index, request, self.config,
        )

        ranked = self._call_with_fallback(
            ranking_primary, ranking_null, self.config["timeouts"]["ranking_seconds"],
            state, candidates,
        )

        recommendations = self._ensure_top_k(ranked, effective_top_k)

        return AgentResponse(
            recommendations=recommendations,
            message=dialog_result.message,
            ask_attribute=dialog_result.ask_attribute,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )

    def _call_with_fallback(self, primary: Callable[..., T], fallback: Callable[..., T], timeout: float, *args) -> T:
        """Runs `primary` under a timeout; on any exception or timeout, calls `fallback` directly
        (synchronously, outside the executor). `fallback` must always be one of the explicit
        null_*.py functions, which are guaranteed fast and side-effect-free -- never `primary`
        again -- so a hung or broken primary can never take the fallback down with it."""
        try:
            future = self._executor.submit(primary, *args)
            return future.result(timeout=timeout)
        except Exception:
            return fallback(*args)

    def _fallback_candidates(self, index: BM25Index, request: RetrievalRequest, config: dict) -> list[Candidate]:
        """Retrieval has no simpler Null path below BM25 (an empty list can never score a hit),
        so its own fallback is the precomputed, rating-sorted pad pool built once at index time."""
        return [
            Candidate(parent_asin=asin, score=0.0, route="fallback", meta=index.products[asin])
            for asin in index.fallback_pool
            if asin in index.products
        ]

    def _ensure_top_k(self, candidates: list[Candidate], top_k: int) -> list[str]:
        seen: set[str] = set()
        result: list[str] = []
        for candidate in candidates:
            asin = candidate.parent_asin
            if asin in self.index.products and asin not in seen:
                seen.add(asin)
                result.append(asin)
            if len(result) >= top_k:
                return result[:top_k]
        for asin in self.index.fallback_pool:
            if asin not in seen:
                seen.add(asin)
                result.append(asin)
            if len(result) >= top_k:
                break
        return result[:top_k]

    def _fallback_response(self, top_k: int) -> AgentResponse:
        effective_top_k = top_k if isinstance(top_k, int) and top_k > 0 else self.top_k_default
        recommendations = self.index.fallback_pool[:effective_top_k]
        return AgentResponse(
            recommendations=recommendations,
            message="",
            ask_attribute=None,
            usage={"prompt_tokens": 0, "completion_tokens": 0},
        )
