"""Catalog-only category matching for the deterministic dialog policy."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Mapping

from src.contracts import ProductMeta


_WORDS = re.compile(r"[\W_]+", re.UNICODE)
_DEFAULT_IGNORED_ROOT_PHRASES = ("clothing shoes jewelry", "clothing", "men", "women")


def normalize_phrase(value: object) -> str:
    """Case-fold text and leave a single space between alphanumeric words."""
    if not isinstance(value, str):
        return ""
    return _WORDS.sub(" ", value.casefold()).strip()


@dataclass(frozen=True)
class CategoryMatch:
    canonical: str
    normalized: str


@dataclass(frozen=True)
class CategoryLexicon:
    """Read-only category phrases derived solely from ``ProductMeta.categories``."""

    phrases: dict[str, str]
    ignored_root_phrases: frozenset[str]

    @classmethod
    def from_products(cls, products: Mapping[str, ProductMeta], config: dict) -> "CategoryLexicon":
        dialog_cfg = config.get("dialog", {}) if isinstance(config, dict) else {}
        category_cfg = dialog_cfg.get("category", {}) if isinstance(dialog_cfg, dict) else {}
        configured_ignored = category_cfg.get("ignored_root_phrases") if isinstance(category_cfg, dict) else None
        ignored = {
            normalize_phrase(value)
            for value in (configured_ignored if isinstance(configured_ignored, (list, tuple)) else _DEFAULT_IGNORED_ROOT_PHRASES)
            if normalize_phrase(value)
        }
        # A dict makes duplicate catalog paths deterministic regardless of product map order.
        phrases: dict[str, str] = {}
        product_values = products.values() if isinstance(products, Mapping) else ()
        for meta in product_values:
            crumbs = [str(item).strip() for item in (getattr(meta, "categories", None) or []) if str(item).strip()]
            # Individual non-root components and each trailing breadcrumb suffix are
            # useful names for the same category ("Earrings" and "Earrings Hoop").
            for start in range(len(crumbs)):
                for phrase in (crumbs[start], " ".join(crumbs[start:])):
                    normalized = normalize_phrase(phrase)
                    if not normalized or (phrase == crumbs[start] and normalized in ignored):
                        continue
                    current = phrases.get(normalized)
                    if current is None or phrase < current:
                        phrases[normalized] = phrase
        return cls(phrases=phrases, ignored_root_phrases=frozenset(ignored))

    def match(self, text: object) -> list[CategoryMatch]:
        """Return every matching catalog phrase, longest first, at word boundaries."""
        normalized_text = normalize_phrase(text)
        if not normalized_text:
            return []
        found = [
            CategoryMatch(canonical=canonical, normalized=phrase)
            for phrase, canonical in self.phrases.items()
            if re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", normalized_text)
        ]
        return sorted(found, key=lambda item: (-len(item.normalized), item.normalized, item.canonical))

    # Friendly aliases for callers/tests which use either noun or verb phrasing.
    extract = match
    matches = match

    @staticmethod
    def compatible(left: object, right: object) -> bool:
        left_normalized = normalize_phrase(left)
        right_normalized = normalize_phrase(right)
        return bool(
            left_normalized
            and right_normalized
            and (
                left_normalized == right_normalized
                or left_normalized in right_normalized
                or right_normalized in left_normalized
            )
        )

    is_compatible = compatible


def build_category_lexicon(products: Mapping[str, ProductMeta], config: dict) -> CategoryLexicon:
    return CategoryLexicon.from_products(products, config)
