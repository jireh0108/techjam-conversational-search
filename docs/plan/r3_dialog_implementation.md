# R3 Dialog Implementation Specification

## Purpose

Implement the deterministic dialog component in `src/dialog/`. Its job is to interpret the latest customer message, maintain catalog-grounded conversational constraints, choose the next structured `ask_attribute`, and return a retrieval-ready `canonical_query`.

The local simulator reacts only to `ask_attribute`; response prose remains user-facing but is not a scoring signal. The implementation must use no LLM, network call, or optional model dependency.

## Scope and interfaces

R3 owns `src/dialog/`. The integration change is limited to constructing a read-only catalog-derived lexicon once in `Agent.__init__` from the already-loaded `self.index.products`, then passing it to both primary and Null dialog calls.

Use this file split:

- `category_lexicon.py`: `CategoryLexicon`, catalog normalization, phrase extraction, and category compatibility.
- `rule_based.py`: config validation, message parsing, slot merge/reset rules, intent classification, query construction, and question selection.
- `null_dialog.py`: unchanged behavior except for accepting the optional lexicon argument.
- `__init__.py`: exports the active `update` function plus the lexicon builder.

```python
class CategoryLexicon:
    @classmethod
    def from_products(cls, products: Mapping[str, ProductMeta], config: dict) -> "CategoryLexicon":
        ...

def update(
    state: SessionState,
    user_message: str,
    category_lexicon: CategoryLexicon,
) -> DialogResult:
    ...
```

`NullDialog.update` accepts the same third argument, which may be ignored. The package exports `CategoryLexicon`, `build_category_lexicon`, and `update`.

`SessionState.slots` is now a `dict[str, list[str]]`. Each list is ordered, normalized, and de-duplicated. All existing consumers must accept list values; `memory.distiller` already does. `slot_turn_added` is not used by this R3 version and must not be mutated as an undocumented side effect.

`DialogResult` remains the output contract. Its `intent` is always one of `"buy"`, `"browse"`, or `"unknown"`; the existing agent glue will assign this latest value to session state.

## Catalog category lexicon

Build the lexicon exclusively from `ProductMeta.categories`; product titles, features, descriptions, stores, and brands must not create category aliases.

- Normalize source and message phrases by case-folding, converting punctuation and whitespace to single spaces, and retaining alphanumeric word boundaries.
- Index every non-root breadcrumb component and every contiguous breadcrumb suffix. This lets the dialog match both catalog phrases such as `"Earrings"` and evaluator-style combined phrases such as `"Earrings Hoop"`.
- Ignore configurable root/scaffolding phrases such as `"clothing shoes jewelry"`, `"men"`, and `"women"` when they are the only match. Retain them when part of a more-specific suffix.
- Match normalized phrases at word boundaries. Return every relevant matching phrase, ordered longest phrase first and then lexicographically, rather than choosing only one match.
- Preserve each match's canonical catalog phrase and normalized phrase. A category comparison is compatible when phrases are equal or either normalized phrase contains the other; otherwise it is a category conflict.

The dialog adds all matches to `slots["category"]` and to the canonical query. A later incompatible category replaces prior category values; compatible matches append.

## Turn processing

Apply these steps on every call, using the latest user message as the authoritative input.

1. Normalize malformed input to an empty string and detect a configured no-preference reply before extracting constraints. A no-preference reply adds no slot values or query terms.
2. Extract every catalog category phrase from the full message.
3. Extract deterministic attribute values using configured regexes and vocabularies: `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, and `use_case`. Scan the full message for configured vocabulary matches so a phrase such as `black leather` yields both color and material. Also parse the evaluator's `key requirement is`, `what matters is`, and `what I need is` clauses; split these clauses on semicolons and classify residual non-empty fragments as features.
4. Preserve meaningful unmatched product text as `feature` after removing matched category text, configured conversational scaffolding, and refusal wording. This is required for opening turns such as a category followed by `"Buckle closure"`.
5. Reclassify intent from this message only:
   - `browse` when a configured exploration marker is present, including when the message also contains a generic `looking for` phrase;
   - `buy` only when an explicit purchase marker is present;
   - `unknown` otherwise. A constraint-only follow-up such as `"For that, what matters is cotton"` is `unknown`.
6. Apply category-override handling, merge extracted values, build the canonical query, and pick the next attribute.

### Slot merging

Slot values are normalized, stable-order lists. Repeated values are ignored.

- Append-compatible fields are `material`, `style`, `feature`, `use_case`, and `other`.
- Mutually exclusive fields are `category`, `color`, `size`, `brand`, and `budget`.
- A newly extracted mutually exclusive value replaces the old list for that field. Budget keeps the latest stated value; R3 does not derive ranges or arithmetic constraints.
- A no-preference message leaves all active values untouched. The agent has already recorded the previously asked attribute, so it will not be asked again.

### Explicit override handling

An override requires a configured marker such as `actually`, `instead`, `never mind`, `ignore my earlier`, or `change of plans`.

- If the message contains no newly detected category, treat the override as a same-category refinement: retain existing category and other values, then apply normal field merge rules.
- If it contains only compatible category phrases, also apply normal field merge rules.
- If it contains an incompatible category, clear `category`, `material`, `color`, `size`, `style`, `brand`, `feature`, and `other`; preserve `budget` and `use_case`; then merge the latest extracted values. Set `intent_override=True`.
- `intent_override` is also true for a same-category explicit override so retrieval can observe the event, even though R3 retains compatible constraints.

## Canonical query and clarification

Build `canonical_query` from active values, not the full dialogue transcript. Serialize fields in the configured attribute order, with `category` first; serialize each field's values in their stable list order. If extraction produces no active values at all, use the cleaned latest message as the fallback query.

The dialog asks until it exhausts eligible attributes, always returning recommendations through the existing agent pipeline. The default human-oriented order is:

```yaml
category, material, color, size, style, brand, budget, feature, use_case, other
```

Choose the first attribute that is neither already present in `state.slots` nor already present in `state.asked_attributes`. `other` remains last; R3 must not exploit its evaluator wildcard behavior as an early shortcut. At `contract.max_turns` (turn 10), return `ask_attribute: null`. When no eligible attribute remains, return `ask_attribute: null`.

Use configurable attribute-specific templates for `message`, for example `"Do you have a material preference?"`. Use a configurable completion message when no question is asked.

## Configuration

Add a top-level `dialog` section to `config.yaml`; no behavior-affecting vocabulary or threshold belongs only in Python constants.

```yaml
dialog:
  attribute_order: [category, material, color, size, style, brand, budget, feature, use_case, other]
  override_markers: [actually, instead, "never mind", "ignore my earlier", "change of plans", different]
  browse_markers: [exploring, browse, browsing, ideas, inspiration, "not sure"]
  buy_markers: [buy, purchase, need, want]
  no_preference_patterns: ["do not have a preference", "don't have a preference", "no preference"]
  category:
    ignored_root_phrases: [clothing_shoes_jewelry, clothing, men, women]
  vocabularies:
    material: [cotton, polyester, nylon, leather, wool, spandex, silk, rayon, fabric]
    color: [black, white, blue, red, pink, green, brown, gray, grey, purple, yellow, orange]
    style: [fit, sleeve, neck, casual, formal]
    use_case: [hiking, running, gym, winter, outdoor, work]
  patterns:
    size: '\\b(?:size\\s*)?(?:xxs|xs|s|m|l|xl|xxl|\\d{1,2}(?:\\.\\d+)?|wide|narrow)\\b'
    budget: '(?:\\$\\s*\\d+(?:\\.\\d{1,2})?|(?:under|below|less than|around)\\s*\\$?\\d+(?:\\.\\d{1,2})?)'
    constraint_clause: '(?:what matters is|key requirement is|what i need is|i need is|prefer(?:s)?)\\s*(.+?)(?:[.!?]|$)'
  messages:
    category: "What product category are you considering?"
    material: "Do you have a material preference?"
    color: "Do you have a color preference?"
    size: "Do you have a size or fit preference?"
    style: "Do you have a style preference?"
    brand: "Do you have a brand preference?"
    budget: "Do you have a budget preference?"
    feature: "Is there a feature that matters most?"
    use_case: "What will you use it for?"
    other: "What other requirement matters most?"
    complete: "Here are the closest matches I found."
```

Implement the shown values with valid regex syntax and normalized phrase strings. Validate at load time inside `src/dialog/`: invalid or missing optional values fall back to safe defaults; unknown attributes are ignored; `ask_attribute` is always an allowed contract value or `None`.

## Required tests and acceptance criteria

Extend dialog unit tests without requiring the real catalog. Use a small fake `ProductMeta` map to build a `CategoryLexicon`.

- The lexicon indexes catalog categories only, recognizes suffix phrases, and returns all relevant matches in deterministic order.
- Opening messages extract catalog category phrases plus residual feature text.
- Constraint-only follow-ups produce `intent="unknown"`; exploration and explicit-buy messages produce `browse` and `buy` respectively.
- Compatible values append; color, size, brand, budget, and incompatible categories replace.
- Same-category explicit overrides retain compatible slots; a conflicting category override clears only product-specific fields and preserves budget/use case.
- No-preference replies add no slot/query value and cause selection to advance beyond the already asked attribute.
- Selection follows configured human order, skips populated and previously asked fields, keeps `other` last, and returns `None` on turn 10 or exhaustion.
- Canonical queries contain active category and slot values, exclude refusal/scaffolding text, and fall back to cleaned raw input only when no constraints exist.
- `NullDialog` and the agent failure path remain valid with the added lexicon argument.

Acceptance requires all existing tests plus the new dialog tests to pass. With `data/catalog.jsonl` available, `eval.run_eval --mode fast` must complete deterministically and must not regress the failure contract: `reset()` and `respond()` still never leak dialog failures.
