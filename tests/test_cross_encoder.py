"""Unit tests for the configurable cross-encoder adapter using a fake scorer."""

from __future__ import annotations

import unittest

from src.contracts import Candidate, ProductMeta, SessionState
from src.ranking.cross_encoder import CrossEncoderReranker, render_candidate


def _meta(title: str) -> ProductMeta:
    return ProductMeta(
        title=title,
        price=29.99,
        categories=["Clothing", "Running Shoes"],
        features=["rubber sole", "lace-up"],
        description=[],
        store="Example Store",
        details_brand=None,
        average_rating=4.5,
        rating_number=100,
    )


def _state(query: str = "black running shoes") -> SessionState:
    return SessionState(session_id="s1", turn=1, intent="buy", slots={}, slot_turn_added={}, canonical_query=query)


def _candidates(count: int = 4) -> list[Candidate]:
    return [Candidate(parent_asin=f"A{i}", score=float(count - i), route="bm25", meta=_meta(f"Product {i}")) for i in range(count)]


class FakeModel:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.pairs = None
        self.kwargs = None

    def predict(self, pairs, **kwargs):
        self.pairs = pairs
        self.kwargs = kwargs
        return self.scores


def _config(depth: int = 2, enabled: bool = True) -> dict:
    return {
        "ranking": {
            "rerank_depth": depth,
            "cross_encoder": {
                "enabled": enabled,
                "model": "fake-model",
                "batch_size": 8,
                "max_length": 128,
                "device": "cpu",
                "show_progress_bar": False,
                "candidate_fields": ["title", "categories", "features", "store", "price"],
                "max_candidate_chars": 500,
            },
        }
    }


class CrossEncoderRerankerTest(unittest.TestCase):
    def test_reorders_only_configured_head_and_preserves_tail(self) -> None:
        model = FakeModel([0.1, 0.9])
        reranker = CrossEncoderReranker(_config(depth=2), model=model)
        candidates = _candidates()

        result = reranker.rerank(_state(), candidates)

        self.assertEqual([candidate.parent_asin for candidate in result], ["A1", "A0", "A2", "A3"])
        self.assertIs(result[0], candidates[1])
        self.assertEqual(result[2:], candidates[2:])

    def test_uses_state_query_and_configured_product_fields(self) -> None:
        model = FakeModel([1.0, 0.0])
        reranker = CrossEncoderReranker(_config(), model=model)

        reranker.rerank(_state("waterproof hiking shoes"), _candidates(2))

        self.assertEqual(model.pairs[0][0], "waterproof hiking shoes")
        self.assertIn("title: Product 0", model.pairs[0][1])
        self.assertIn("categories: Clothing; Running Shoes", model.pairs[0][1])
        self.assertEqual(model.kwargs["batch_size"], 8)
        self.assertFalse(model.kwargs["show_progress_bar"])

    def test_model_factory_receives_configured_device(self) -> None:
        calls = {}

        def factory(model_name, **kwargs):
            calls["model_name"] = model_name
            calls["kwargs"] = kwargs
            return FakeModel([1.0, 0.0])

        reranker = CrossEncoderReranker(_config(), model_factory=factory)

        result = reranker.rerank(_state(), _candidates(2))

        self.assertEqual(len(result), 2)
        self.assertEqual(calls["model_name"], "fake-model")
        self.assertEqual(calls["kwargs"]["device"], "cpu")
        self.assertEqual(calls["kwargs"]["max_length"], 128)

    def test_disabled_cross_encoder_is_identity(self) -> None:
        model = FakeModel([0.0])
        reranker = CrossEncoderReranker(_config(depth=1, enabled=False), model=model)
        candidates = _candidates(2)

        result = reranker.rerank(_state(), candidates)

        self.assertEqual(result, candidates)
        self.assertIsNone(model.pairs)

    def test_model_failure_returns_original_order(self) -> None:
        class BrokenModel:
            def predict(self, pairs, **kwargs):
                raise RuntimeError("test failure")

        candidates = _candidates()
        result = CrossEncoderReranker(_config(), model=BrokenModel()).rerank(_state(), candidates)
        self.assertEqual(result, candidates)

    def test_invalid_scores_return_original_order(self) -> None:
        candidates = _candidates()
        result = CrossEncoderReranker(_config(), model=FakeModel([float("nan"), 1.0])).rerank(_state(), candidates)
        self.assertEqual(result, candidates)

    def test_empty_query_and_candidates_are_safe(self) -> None:
        model = FakeModel([])
        reranker = CrossEncoderReranker(_config(), model=model)
        self.assertEqual(reranker.rerank(_state(""), _candidates()), _candidates())
        self.assertEqual(reranker.rerank(_state(), []), [])

    def test_render_candidate_skips_missing_fields_and_truncates(self) -> None:
        text = render_candidate(_meta("A"), ["title", "price", "not_a_field"], 12)
        self.assertEqual(len(text), 12)
        self.assertTrue(text.startswith("title:"))


if __name__ == "__main__":
    unittest.main()
