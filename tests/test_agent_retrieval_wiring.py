"""End-to-end contract tests for R4's dialog-to-retrieval integration glue."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent import Agent
from src.contracts import Candidate, RetrievalResult


def _catalog(path: Path) -> None:
    rows = [
        {
            "parent_asin": "RUNNING",
            "title": "Leather running shoe",
            "categories": ["Clothing", "Shoes", "Running Shoes"],
            "features": ["leather"],
            "description": [],
            "details": {},
            "store": "Example",
            "average_rating": 4.5,
            "rating_number": 100,
        },
        {
            "parent_asin": "EARRINGS",
            "title": "Gold hoop earrings",
            "categories": ["Clothing", "Jewelry", "Earrings"],
            "features": ["gold"],
            "description": [],
            "details": {},
            "store": "Example",
            "average_rating": 4.5,
            "rating_number": 100,
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class AgentRetrievalWiringTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        path = Path(self.directory.name) / "catalog.jsonl"
        _catalog(path)
        self.agent = Agent(path)
        self.agent.reset("session-1", {"preference_tags": ["comfort"]})
        self.requests = []

    def tearDown(self) -> None:
        self.agent.close()
        self.directory.cleanup()

    def _capture_retrieval(self, index, request, config):
        self.requests.append(request)
        candidate = Candidate("RUNNING", 1.0, "test", index.products["RUNNING"])
        return RetrievalResult([candidate], pool_size=17, dropped_constraints=["category"])

    def test_forwards_session_state_and_category_filter_to_retrieval(self) -> None:
        with patch("src.agent.retrieval_primary", side_effect=self._capture_retrieval), \
             patch("src.agent.ranking_primary", side_effect=lambda state, candidates: candidates):
            response = self.agent.respond("session-1", "I want leather running shoes", 1, 10)

        self.assertEqual(response["recommendations"][0]["parent_asin"], "RUNNING")
        request = self.requests[-1]
        self.assertEqual(request.session_id, "session-1")
        self.assertEqual(request.turn, 1)
        self.assertEqual(request.negatives, [])
        self.assertEqual(request.accepted, [])
        self.assertEqual(request.profile, {"preference_tags": ["comfort"]})
        self.assertFalse(request.intent_changed)
        self.assertEqual(request.hard_filters, {"category": "Running Shoes"})
        state = self.agent._sessions["session-1"]
        self.assertEqual(state.retrieval_pool_size, 17)
        self.assertEqual(state.dropped_constraints, ["category"])

    def test_forwards_dialog_override_to_retrieval(self) -> None:
        with patch("src.agent.retrieval_primary", side_effect=self._capture_retrieval), \
             patch("src.agent.ranking_primary", side_effect=lambda state, candidates: candidates):
            self.agent.respond("session-1", "I want running shoes", 1, 10)
            self.agent.respond("session-1", "Actually, I want earrings instead.", 2, 10)

        request = self.requests[-1]
        self.assertEqual(request.turn, 2)
        self.assertTrue(request.intent_changed)
        self.assertEqual(request.hard_filters, {"category": "Earrings"})

    def test_timed_out_component_does_not_block_other_components(self) -> None:
        release = threading.Event()

        def blocked():
            release.wait()
            return "late"

        try:
            ranked = self.agent._call_with_fallback(
                "ranking", blocked, lambda: "ranking fallback", 0.01
            )
            dialog = self.agent._call_with_fallback(
                "dialog", lambda: "dialog primary", lambda: "dialog fallback", 0.1
            )
        finally:
            release.set()

        self.assertEqual(ranked, "ranking fallback")
        self.assertEqual(dialog, "dialog primary")


if __name__ == "__main__":
    unittest.main()
