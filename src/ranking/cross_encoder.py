"""Configurable local cross-encoder reranking.

The public ``rerank`` function deliberately keeps NullReranker's exact signature.  The model
is loaded lazily on the first reranking request.  Any model/configuration failure returns the
input order.

The implementation only reorders the first ``ranking.rerank_depth`` candidates.  The remaining
retrieval results are preserved as-is so ranking cannot invent ASINs or shrink the fallback pool.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from src.config import load_config
from src.contracts import Candidate, ProductMeta, SessionState


def render_candidate(meta: ProductMeta, fields: Sequence[str], max_chars: int) -> str:
    """Render configured ProductMeta fields into the compact text scored by the model."""
    sections: list[str] = []
    for field_name in fields:
        value = getattr(meta, str(field_name), None)
        if value in (None, "", []):
            continue
        if isinstance(value, (list, tuple)):
            value = "; ".join(str(item) for item in value if str(item).strip())
        sections.append(f"{field_name}: {value}")
    return " | ".join(sections)[:max_chars]


class CrossEncoderReranker:
    """Small adapter around any sentence-transformers-compatible cross encoder.

    ``model`` and ``model_factory`` are injectable so unit tests and future compatible models do
    not need to download anything.  Production construction uses the model reference in config.
    """

    def __init__(
        self,
        config: dict | None = None,
        *,
        model: Any | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._model = model
        self._model_factory = model_factory
        self._load_attempted = model is not None

    def _config(self) -> dict:
        return self.config if self.config is not None else load_config()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._load_attempted:
            raise RuntimeError("cross-encoder model could not be loaded")
        self._load_attempted = True
        ranking_cfg = self._config().get("ranking", {})
        model_cfg = ranking_cfg.get("cross_encoder", {})
        factory = self._model_factory
        if factory is None:
            # Optional dependency: importing it here keeps NullReranker usable without ML wheels.
            threads = model_cfg.get("torch_num_threads")
            if threads:
                import torch

                torch.set_num_threads(int(threads))
            from sentence_transformers import CrossEncoder

            factory = CrossEncoder
        model_kwargs = {
            "max_length": int(model_cfg.get("max_length", 512)),
            "local_files_only": bool(model_cfg.get("local_files_only", False)),
        }
        device = model_cfg.get("device")
        if device:
            model_kwargs["device"] = str(device)
        self._model = factory(str(model_cfg["model"]), **model_kwargs)
        return self._model

    def _enabled(self) -> bool:
        return bool(
            self._config().get("ranking", {}).get("cross_encoder", {}).get("enabled", False)
        )

    def rerank(self, state: SessionState, candidates: list[Candidate]) -> list[Candidate]:
        original = list(candidates)
        try:
            if not original or not self._enabled():
                return original
            query = str(getattr(state, "canonical_query", "") or "").strip()
            if not query:
                return original

            ranking_cfg = self._config().get("ranking", {})
            model_cfg = ranking_cfg.get("cross_encoder", {})
            depth = max(0, min(len(original), int(ranking_cfg.get("rerank_depth", len(original)))))
            if depth == 0:
                return original
            fields = model_cfg.get("candidate_fields", ["title", "categories", "features", "store", "price"])
            max_chars = int(model_cfg.get("max_candidate_chars", 1200))
            pairs = [
                (query, render_candidate(candidate.meta, fields, max_chars))
                for candidate in original[:depth]
            ]
            model = self._load_model()
            raw_scores = model.predict(
                pairs,
                batch_size=int(model_cfg.get("batch_size", 16)),
                show_progress_bar=bool(model_cfg.get("show_progress_bar", False)),
            )
            scores = [float(score) for score in raw_scores]
            if len(scores) != depth or not all(math.isfinite(score) for score in scores):
                raise ValueError("cross-encoder returned invalid score output")
            order = sorted(range(depth), key=lambda index: (-scores[index], index))
            return [original[index] for index in order] + original[depth:]
        except Exception:
            return original


_DEFAULT_RERANKER = CrossEncoderReranker()


def rerank(state: SessionState, candidates: list[Candidate]) -> list[Candidate]:
    """Primary ranking entry point; fail-soft behavior mirrors NullReranker's contract."""
    return _DEFAULT_RERANKER.rerank(state, candidates)
