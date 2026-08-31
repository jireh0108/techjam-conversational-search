"""Tests for the optional listwise reranker without an API key or network call."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.contracts import Candidate, ProductMeta, SessionState
from src.ranking.listwise import ListwiseReranker


def _meta(title: str) -> ProductMeta:
    return ProductMeta(title, 10.0, ["Shoes"], ["comfortable"], [], "Store", None, 4.0, 10)


def _state() -> SessionState:
    return SessionState("s1", 1, "buy", {}, [], canonical_query="comfortable shoes")


def _candidates() -> list[Candidate]:
    return [Candidate(asin, 1.0, "bm25", _meta(title)) for asin, title in (("A", "First"), ("B", "Second"), ("C", "Third"))]


def _config(enabled: bool = True) -> dict:
    return {
        "ranking": {
            "cross_encoder": {
                "candidate_fields": ["title", "categories", "features", "store", "price"],
                "max_candidate_chars": 500,
            },
            "listwise": {
                "enabled": enabled,
                "api_key_env": "TEST_LLM_KEY",
                "model": "test-model",
                "max_candidates": 2,
                "prompt_instruction": "Return ranked IDs only.",
                "temperature": 0.0,
                "max_output_tokens": 64,
            },
        }
    }


class FakeResponses:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self.text)


class ListwiseRerankerTest(unittest.TestCase):
    def test_requires_config_flag_and_key(self) -> None:
        client = FakeResponses('["B", "A"]')
        reranker = ListwiseReranker(_config(), client=SimpleNamespace(responses=client), environment={})
        self.assertFalse(reranker.available())
        self.assertIsNone(reranker.rerank_or_none(_state(), _candidates()))
        self.assertEqual(client.calls, [])

    def test_parses_valid_ids_and_preserves_unrequested_tail(self) -> None:
        client = FakeResponses('["B", "A"]')
        reranker = ListwiseReranker(
            _config(), client=SimpleNamespace(responses=client), environment={"TEST_LLM_KEY": "present"}
        )
        result = reranker.rerank_or_none(_state(), _candidates())
        self.assertEqual([candidate.parent_asin for candidate in result], ["B", "A", "C"])
        self.assertIn("comfortable shoes", client.calls[0]["input"])
        self.assertEqual(client.calls[0]["model"], "test-model")
        self.assertEqual(client.calls[0]["temperature"], 0.0)
        self.assertEqual(client.calls[0]["max_output_tokens"], 64)
        self.assertTrue(client.calls[0]["input"].startswith("Return ranked IDs only."))

    def test_filters_hallucinated_ids_and_deduplicates(self) -> None:
        client = FakeResponses('["UNKNOWN", "B", "B"]')
        reranker = ListwiseReranker(
            _config(), client=SimpleNamespace(responses=client), environment={"TEST_LLM_KEY": "present"}
        )
        result = reranker.rerank_or_none(_state(), _candidates())
        self.assertEqual([candidate.parent_asin for candidate in result], ["B", "A", "C"])

    def test_malformed_response_falls_back(self) -> None:
        client = FakeResponses("not json")
        reranker = ListwiseReranker(
            _config(), client=SimpleNamespace(responses=client), environment={"TEST_LLM_KEY": "present"}
        )
        self.assertIsNone(reranker.rerank_or_none(_state(), _candidates()))


if __name__ == "__main__":
    unittest.main()
