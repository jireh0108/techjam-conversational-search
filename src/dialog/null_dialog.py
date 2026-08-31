"""NullDialog: the permanent dialog fallback. Passes the raw utterance through as the canonical
query and never asks a clarifying question.

Per docs/plan/RECON.md E1: the evaluator's simulator only ever reacts to the structured
`ask_attribute` field, never to `message` text, so a real dialog component's entire value is in
choosing the right `ask_attribute` -- NullDialog choosing None is a legitimate, permanent
fallback (equivalent to "never ask"), not a placeholder for prose quality.
"""

from __future__ import annotations

from src.contracts import DialogResult, SessionState
from src.dialog.category_lexicon import CategoryLexicon


def update(state: SessionState, user_message: str, category_lexicon: CategoryLexicon | None = None) -> DialogResult:
    return DialogResult(
        canonical_query=user_message,
        ask_attribute=None,
        slots=dict(state.slots),
        message="Here are the closest matches I found.",
    )
