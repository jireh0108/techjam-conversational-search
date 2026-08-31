"""Ranking selector: optional listwise LLM over the local cross-encoder."""

from __future__ import annotations

from src.contracts import Candidate, SessionState

from .cross_encoder import rerank as cross_encoder_rerank
from .listwise import rerank_or_none as listwise_rerank_or_none


def rerank(state: SessionState, candidates: list[Candidate]) -> list[Candidate]:
    """Use listwise LLM only when explicitly enabled and available; otherwise use local CE."""
    listwise_result = listwise_rerank_or_none(state, candidates)
    if listwise_result is not None:
        return listwise_result
    return cross_encoder_rerank(state, candidates)
