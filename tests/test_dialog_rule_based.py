from __future__ import annotations

import unittest
from unittest.mock import patch

from src.config import load_config
from src.contracts import ProductMeta, SessionState
from src.dialog import build_category_lexicon
from src.dialog.rule_based import update


class RuleBasedDialogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        products = {
            "earring": ProductMeta("Ignored title", None, ["Clothing, Shoes & Jewelry", "Women", "Earrings", "Hoop"], ["not an alias"], [], None, None, 0.0, 0),
            "shoe": ProductMeta("Shoe title", None, ["Clothing, Shoes & Jewelry", "Men", "Shoes"], [], [], None, None, 0.0, 0),
        }
        cls.lexicon = build_category_lexicon(products, load_config())

    def test_extracts_constraint_and_asks_next_attribute(self) -> None:
        state = SessionState("s", 1, "unknown", {}, {})
        result = update(state, "For that, what matters is: black; leather.", self.lexicon)
        self.assertEqual(result.slots["color"], ["black"])
        self.assertEqual(result.slots["material"], ["leather"])
        self.assertEqual(result.ask_attribute, "category")
        self.assertEqual(result.intent, "unknown")

    def test_accumulates_slots_and_detects_override(self) -> None:
        state = SessionState("s", 3, "buy", {"category": ["Shoes"], "color": ["black"]}, {}, asked_attributes=["category", "material"])
        result = update(state, "Actually, ignore my earlier preference. What I need is: red color.", self.lexicon)
        self.assertTrue(result.intent_override)
        self.assertEqual(result.slots["color"], ["red"])
        self.assertEqual(result.slots["category"], ["Shoes"])

    def test_catalog_only_lexicon_matches_suffixes_deterministically(self) -> None:
        matches = self.lexicon.match("I need EARRINGS hoop")
        self.assertEqual([match.canonical for match in matches], ["Earrings Hoop", "Earrings", "Hoop"])
        self.assertEqual(self.lexicon.match("ignored title"), [])
        self.assertEqual(self.lexicon.match("women"), [])

    def test_opening_turn_keeps_category_and_residual_feature(self) -> None:
        result = update(SessionState("s", 1, "unknown", {}, {}), "I want earrings hoop with buckle closure.", self.lexicon)
        self.assertEqual(result.slots["category"][:2], ["Earrings Hoop", "Earrings"])
        self.assertEqual(result.slots["feature"], ["buckle closure"])
        self.assertIn("Earrings Hoop", result.canonical_query)
        self.assertNotIn("i want", result.canonical_query.casefold())

    def test_intent_precedence_and_constraint_follow_up(self) -> None:
        state = SessionState("s", 2, "unknown", {}, {})
        self.assertEqual(update(state, "I am browsing shoes", self.lexicon).intent, "browse")
        self.assertEqual(update(state, "I want to buy shoes but I am exploring", self.lexicon).intent, "browse")
        self.assertEqual(update(state, "I want shoes", self.lexicon).intent, "buy")
        self.assertEqual(update(state, "For that, what matters is cotton", self.lexicon).intent, "unknown")

    def test_merging_override_and_question_selection(self) -> None:
        state = SessionState(
            "s", 4, "buy",
            {"category": ["Shoes"], "material": ["leather"], "color": ["black"], "budget": ["under 20"], "use_case": ["running"], "feature": ["buckle closure"]},
            {}, asked_attributes=["category"],
        )
        result = update(state, "Actually I want earrings hoop in red.", self.lexicon)
        self.assertTrue(result.intent_override)
        self.assertEqual(result.slots["category"][0], "Earrings Hoop")
        self.assertEqual(result.slots["color"], ["red"])
        self.assertNotIn("material", result.slots)
        self.assertNotIn("feature", result.slots)
        self.assertEqual(result.slots["budget"], ["under 20"])
        self.assertEqual(result.slots["use_case"], ["running"])
        self.assertEqual(result.ask_attribute, "material")

    def test_no_preference_skips_prior_question_and_turn_ten_completes(self) -> None:
        state = SessionState("s", 3, "buy", {"category": ["Shoes"]}, {}, asked_attributes=["material"])
        result = update(state, "No preference", self.lexicon)
        self.assertEqual(result.slots, {"category": ["Shoes"]})
        self.assertEqual(result.ask_attribute, "color")
        complete = update(SessionState("s", 10, "unknown", {}, {}), "anything", self.lexicon)
        self.assertIsNone(complete.ask_attribute)

    def test_turn_guard_uses_contract_configuration(self) -> None:
        config = load_config()
        configured = {**config, "contract": {**config["contract"], "max_turns": 3}}
        state = SessionState("s", 3, "unknown", {}, {})
        with patch("src.dialog.rule_based.load_config", return_value=configured):
            result = update(state, "I want earrings", self.lexicon)
        self.assertIsNone(result.ask_attribute)


if __name__ == "__main__":
    unittest.main()
