from __future__ import annotations

import unittest

from src.contracts import Candidate, DialogResult, MemoryProfile, ProductMeta, SessionState
from src.dialog.null_dialog import update as dialog_update
from src.memory.null_memory import distill as memory_distill
from src.ranking.null_reranker import rerank as ranking_rerank

SAMPLE_META = ProductMeta(
    title="Test Product", price=19.99, categories=["Clothing", "Shoes"],
    features=["cotton"], description=["a shoe"], store="TestStore",
    details_brand=None, average_rating=4.5, rating_number=100,
)


def _state() -> SessionState:
    return SessionState(session_id="s1", turn=1, intent="unknown", slots={}, slot_turn_added={})


class NullPathTest(unittest.TestCase):
    def test_null_dialog_returns_dialog_result_echoing_utterance(self) -> None:
        result = dialog_update(_state(), "I want cotton running shoes", None)
        self.assertIsInstance(result, DialogResult)
        self.assertEqual(result.canonical_query, "I want cotton running shoes")
        self.assertIsNone(result.ask_attribute)
        self.assertIsInstance(result.message, str)
        self.assertTrue(result.message)

    def test_null_memory_returns_empty_profile(self) -> None:
        result = memory_distill(_state(), {"summary": "x"})
        self.assertIsInstance(result, MemoryProfile)
        self.assertEqual(result.boosts, {})
        self.assertEqual(result.summary, "")

    def test_null_reranker_returns_input_order_unchanged(self) -> None:
        candidates = [
            Candidate(parent_asin="B1", score=3.0, route="bm25", meta=SAMPLE_META),
            Candidate(parent_asin="B2", score=1.0, route="bm25", meta=SAMPLE_META),
            Candidate(parent_asin="B3", score=2.0, route="bm25", meta=SAMPLE_META),
        ]
        result = ranking_rerank(_state(), candidates)
        self.assertEqual([c.parent_asin for c in result], ["B1", "B2", "B3"])
        self.assertIsNot(result, candidates)  # returns a new list, not the same object


if __name__ == "__main__":
    unittest.main()
