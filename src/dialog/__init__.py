"""Dialog leaf component. Never imports retrieval/, ranking/, or memory/."""

from .category_lexicon import CategoryLexicon, build_category_lexicon
from .rule_based import update

__all__ = ["CategoryLexicon", "build_category_lexicon", "update"]
