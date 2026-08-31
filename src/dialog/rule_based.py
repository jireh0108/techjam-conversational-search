"""Config-driven, deterministic dialog policy with catalog-grounded categories."""

from __future__ import annotations

import re
from collections.abc import Iterable

from src.config import load_config
from src.contracts import ASK_ATTRIBUTES, DialogResult, SessionState
from src.dialog.category_lexicon import CategoryLexicon, normalize_phrase


_DEFAULT_DIALOG = {
    "attribute_order": list(ASK_ATTRIBUTES),
    "override_markers": ["actually", "instead", "never mind", "ignore my earlier", "change of plans", "different"],
    "browse_markers": ["exploring", "browse", "browsing", "ideas", "inspiration", "not sure"],
    "buy_markers": ["buy", "purchase", "need", "want"],
    "no_preference_patterns": ["do not have a preference", "don't have a preference", "no preference"],
    "category": {"ignored_root_phrases": ["clothing shoes jewelry", "clothing", "men", "women"]},
    "vocabularies": {
        "material": ["cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk", "rayon", "fabric"],
        "color": ["black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey", "purple", "yellow", "orange"],
        "style": ["fit", "sleeve", "neck", "casual", "formal"],
        "use_case": ["hiking", "running", "gym", "winter", "outdoor", "work"],
    },
    "patterns": {
        "size": r"\b(?:size\s*)?(?:xxs|xs|s|m|l|xl|xxl|\d{1,2}(?:\.\d+)?|wide|narrow)\b",
        "budget": r"(?:\$\s*\d+(?:\.\d{1,2})?|(?:under|below|less than|around)\s*\$?\d+(?:\.\d{1,2})?)",
        "constraint_clause": r"(?:what matters is|key requirement is|what i need is|i need is|prefer(?:s)?)\s*:?[ \t]*(.+?)(?:[.!?]|$)",
        "brand": r"\b(?:brand\s*(?:is\s*)?|by\s+)([a-z0-9][a-z0-9&' -]{0,40})",
    },
    "messages": {attribute: f"Do you have a {attribute} preference?" for attribute in ASK_ATTRIBUTES} | {"complete": "Here are the closest matches I found."},
}

_APPEND_FIELDS = frozenset({"material", "style", "feature", "use_case", "other"})
_REPLACE_FIELDS = frozenset({"category", "color", "size", "brand", "budget"})
_PRODUCT_FIELDS = frozenset({"category", "material", "color", "size", "style", "brand", "feature", "other"})
_SCAFFOLDING = re.compile(
    r"\b(?:i|im|am|looking|for|a|an|the|please|can|you|show|me|have|do|you|"
    r"want|need|buy|purchase|prefer|preference|what|matters|is|key|requirement|"
    r"that|my|earlier|actually|instead|never|mind|ignore|change|of|plans|different|"
    r"exploring|explore|browse|browsing|ideas|inspiration|still|shopping|sure|m|s|"
    r"color|material|brand|size|style|use|it|to|with|in|on|and|or|but)\b",
    re.I,
)
_REFUSAL = re.compile(r"\b(?:no|not|don'?t|don t|do not|without|avoid|anything|none)\b", re.I)


def _strings(value: object, fallback: Iterable[str]) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return list(fallback)
    output = [normalize_phrase(item) for item in value]
    return [item for item in output if item] or list(fallback)


def _config() -> dict:
    """Validate soft configuration locally; malformed optional values never break dialog."""
    try:
        raw = load_config().get("dialog", {})
    except Exception:
        raw = {}
    raw = raw if isinstance(raw, dict) else {}
    cfg: dict = {}
    requested_order = raw.get("attribute_order")
    order = [str(item) for item in requested_order] if isinstance(requested_order, list) else []
    cfg["attribute_order"] = [item for item in order if item in ASK_ATTRIBUTES] or list(_DEFAULT_DIALOG["attribute_order"])
    for key in ("override_markers", "browse_markers", "buy_markers", "no_preference_patterns"):
        cfg[key] = _strings(raw.get(key), _DEFAULT_DIALOG[key])
    raw_vocabularies = raw.get("vocabularies") if isinstance(raw.get("vocabularies"), dict) else {}
    cfg["vocabularies"] = {
        field: _strings(raw_vocabularies.get(field), values)
        for field, values in _DEFAULT_DIALOG["vocabularies"].items()
    }
    raw_patterns = raw.get("patterns") if isinstance(raw.get("patterns"), dict) else {}
    cfg["patterns"] = {}
    for field, fallback in _DEFAULT_DIALOG["patterns"].items():
        candidate = raw_patterns.get(field, fallback)
        try:
            re.compile(candidate if isinstance(candidate, str) else fallback, re.I)
            cfg["patterns"][field] = candidate if isinstance(candidate, str) else fallback
        except re.error:
            cfg["patterns"][field] = fallback
    raw_messages = raw.get("messages") if isinstance(raw.get("messages"), dict) else {}
    cfg["messages"] = {
        key: value if isinstance(value := raw_messages.get(key), str) and value.strip() else fallback
        for key, fallback in _DEFAULT_DIALOG["messages"].items()
    }
    return cfg


def _contains(text: str, phrase: str) -> bool:
    return bool(phrase and re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", text))


def _remove_spans(text: str, phrases: Iterable[str]) -> str:
    result = text
    for phrase in sorted({normalize_phrase(item) for item in phrases if normalize_phrase(item)}, key=len, reverse=True):
        result = re.sub(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", " ", result)
    return result


def _clean_feature(text: str, consumed: Iterable[str]) -> str:
    remaining = _remove_spans(normalize_phrase(text), consumed)
    remaining = _SCAFFOLDING.sub(" ", remaining)
    remaining = _REFUSAL.sub(" ", remaining)
    return normalize_phrase(remaining)


def _extract(message: str, lexicon: CategoryLexicon, cfg: dict) -> dict[str, list[str]]:
    matches = lexicon.match(message)
    values: dict[str, list[str]] = {}
    if matches:
        values["category"] = [item.canonical for item in matches]
    consumed = [item.normalized for item in matches]
    normalized = normalize_phrase(message)

    for field, vocabulary in cfg["vocabularies"].items():
        found = [word for word in vocabulary if _contains(normalized, word)]
        if found:
            values[field] = found
            consumed.extend(found)
    budget_matches = list(re.finditer(cfg["patterns"]["budget"], message, re.I))
    budget_values = [normalize_phrase(match.group(0)) for match in budget_matches]
    if budget_values:
        values["budget"] = budget_values
        consumed.extend(budget_values)
    # Numeric prices match the deliberately broad size pattern too.  Treat a number inside a
    # configured budget expression as budget only, while retaining an explicit ``size 10``.
    size_values = [
        normalize_phrase(match.group(0))
        for match in re.finditer(cfg["patterns"]["size"], message, re.I)
        if not any(match.start() < budget.end() and budget.start() < match.end() for budget in budget_matches)
        and not (match.start() > 0 and message[match.start() - 1] in "'’")
    ]
    if size_values:
        values["size"] = size_values
        consumed.extend(size_values)
    for match in re.finditer(cfg["patterns"]["brand"], message, re.I):
        brand = normalize_phrase(match.group(1) if match.lastindex else match.group(0))
        # Trim conjunctions/scaffolding greedily captured by a permissive configured pattern.
        brand = re.split(r"\b(?:and|or|with|in|for|under|below)\b", brand, maxsplit=1)[0].strip()
        if brand:
            values.setdefault("brand", []).append(brand)
            consumed.append(brand)

    # Clauses carry evaluator requirements.  Once known values are removed, the residue is
    # still meaningful product text (e.g. "buckle closure") and belongs in feature.
    clause_residue: list[str] = []
    for match in re.finditer(cfg["patterns"]["constraint_clause"], message, re.I):
        residue = _clean_feature(match.group(1), consumed)
        if residue:
            clause_residue.append(residue)
    residual = _clean_feature(message, consumed)
    for value in clause_residue + ([residual] if residual else []):
        values.setdefault("feature", []).append(value)
    return {field: _dedupe(field_values) for field, field_values in values.items() if _dedupe(field_values)}


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = normalize_phrase(value)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(value.strip() if isinstance(value, str) else cleaned)
    return result


def _slot_values(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return _dedupe(str(item) for item in value)
    return _dedupe([str(value)]) if value not in (None, "") else []


def _merge(state_slots: object, extracted: dict[str, list[str]], override: bool, lexicon: CategoryLexicon) -> tuple[dict[str, list[str]], bool]:
    slots = {
        str(field): _slot_values(value)
        for field, value in (state_slots or {}).items()
        if str(field) in ASK_ATTRIBUTES
    } if isinstance(state_slots, dict) else {}
    new_categories = extracted.get("category", [])
    incompatible = bool(new_categories and any(
        not lexicon.compatible(previous, latest)
        for previous in slots.get("category", [])
        for latest in new_categories
    ))
    if override and incompatible:
        for field in _PRODUCT_FIELDS:
            slots.pop(field, None)
    elif incompatible:
        slots.pop("category", None)

    for field, latest in extracted.items():
        latest = _dedupe(latest)
        if field == "category":
            if incompatible:
                slots[field] = latest
            else:
                slots[field] = _dedupe(slots.get(field, []) + latest)
        elif field in _APPEND_FIELDS:
            slots[field] = _dedupe(slots.get(field, []) + latest)
        elif field in _REPLACE_FIELDS:
            slots[field] = latest
    return {field: values for field, values in slots.items() if values}, bool(override)


def _intent(message: str, cfg: dict) -> str:
    text = normalize_phrase(message)
    if any(_contains(text, marker) for marker in cfg["browse_markers"]):
        return "browse"
    for marker in cfg["buy_markers"]:
        if marker in {"need", "want"}:
            if re.search(r"\b(?:i|we)\s+" + re.escape(marker) + r"\b", text):
                return "buy"
        elif _contains(text, marker):
            return "buy"
    return "unknown"


def _canonical_query(slots: dict[str, list[str]], fallback: str, order: list[str]) -> str:
    parts: list[str] = []
    fields = ["category"] + [field for field in order if field != "category"]
    for field in fields:
        parts.extend(slots.get(field, []))
    return " ".join(parts) if parts else fallback


def update(state: SessionState, user_message: str, category_lexicon: CategoryLexicon | None = None) -> DialogResult:
    """Process one authoritative customer message without mutating ``state``."""
    cfg = _config()
    message = user_message if isinstance(user_message, str) else ""
    normalized_message = normalize_phrase(message)
    cleaned_message = _clean_feature(message, [])
    lexicon = category_lexicon or CategoryLexicon({}, frozenset())
    no_preference = any(_contains(normalized_message, phrase) for phrase in cfg["no_preference_patterns"])
    override = any(_contains(normalized_message, marker) for marker in cfg["override_markers"])
    extracted = {} if no_preference else _extract(message, lexicon, cfg)
    slots, intent_override = _merge(state.slots, extracted, override, lexicon)
    asked = {item for item in (state.asked_attributes or []) if item in ASK_ATTRIBUTES}
    ask_attribute = next((field for field in cfg["attribute_order"] if field not in slots and field not in asked), None)
    if not isinstance(state.turn, int) or state.turn >= 10:
        ask_attribute = None
    return DialogResult(
        canonical_query=_canonical_query(slots, cleaned_message, cfg["attribute_order"]),
        ask_attribute=ask_attribute if ask_attribute in ASK_ATTRIBUTES else None,
        slots=slots,
        message=cfg["messages"][ask_attribute] if ask_attribute else cfg["messages"]["complete"],
        intent=_intent(message, cfg),
        intent_override=intent_override,
    )
