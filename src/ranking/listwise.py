"""Optional listwise LLM reranker.

This path is intentionally dormant by default.  It requires both an explicit config flag and a
non-empty environment variable.  A failed or malformed LLM response returns ``None`` so the
primary ranking adapter can fall back to the local cross-encoder.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Mapping
from typing import Any

from src.config import load_config
from src.contracts import Candidate, SessionState

from .cross_encoder import render_candidate


class ListwiseReranker:
    def __init__(
        self,
        config: dict | None = None,
        *,
        client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self._client = client
        self._client_factory = client_factory
        self._environment = environment if environment is not None else os.environ

    def _config(self) -> dict:
        return self.config if self.config is not None else load_config()

    def _llm_config(self) -> dict:
        return self._config().get("ranking", {}).get("listwise", {})

    def _api_key(self) -> str:
        key_name = str(self._llm_config().get("api_key_env", "OPENAI_API_KEY"))
        return str(self._environment.get(key_name, "") or "").strip()

    def available(self) -> bool:
        cfg = self._llm_config()
        return bool(cfg.get("enabled", False) and self._api_key())

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        factory = self._client_factory
        if factory is None:
            from openai import OpenAI

            factory = OpenAI
        self._client = factory(api_key=self._api_key())
        return self._client

    def _prompt(self, state: SessionState, candidates: list[Candidate]) -> str:
        model_cfg = self._config().get("ranking", {}).get("cross_encoder", {})
        fields = model_cfg.get("candidate_fields", ["title", "categories", "features", "store", "price"])
        max_chars = int(model_cfg.get("max_candidate_chars", 1200))
        instruction = str(
            self._llm_config().get(
                "prompt_instruction",
                "Return only a JSON array of candidate parent_asin strings, ordered best first.",
            )
        )
        lines = [
            instruction,
            f"Shopping query: {state.canonical_query}",
            "Candidates:",
        ]
        for candidate in candidates:
            text = render_candidate(candidate.meta, fields, max_chars)
            lines.append(f"{candidate.parent_asin}: {text}")
        return "\n".join(lines)

    @staticmethod
    def _response_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if isinstance(output_text, str):
            return output_text
        if isinstance(response, str):
            return response
        return ""

    def _parse_order(self, text: str, candidates: list[Candidate]) -> list[Candidate] | None:
        match = re.search(r"\[[\s\S]*\]", text)
        if not match:
            return None
        values = json.loads(match.group(0))
        if not isinstance(values, list):
            return None
        by_asin = {candidate.parent_asin: candidate for candidate in candidates}
        ordered: list[Candidate] = []
        seen: set[str] = set()
        for value in values:
            asin = str(value).strip()
            if asin in by_asin and asin not in seen:
                ordered.append(by_asin[asin])
                seen.add(asin)
        if not ordered:
            return None
        ordered.extend(candidate for candidate in candidates if candidate.parent_asin not in seen)
        return ordered

    def rerank_or_none(self, state: SessionState, candidates: list[Candidate]) -> list[Candidate] | None:
        original = list(candidates)
        try:
            if not original or not self.available():
                return None
            cfg = self._llm_config()
            limit = max(1, min(len(original), int(cfg.get("max_candidates", 20))))
            head = original[:limit]
            client = self._get_client()
            request = {
                "model": str(cfg["model"]),
                "input": self._prompt(state, head),
            }
            if cfg.get("temperature") is not None:
                request["temperature"] = float(cfg["temperature"])
            if cfg.get("max_output_tokens") is not None:
                request["max_output_tokens"] = int(cfg["max_output_tokens"])
            response = client.responses.create(**request)
            ranked = self._parse_order(self._response_text(response), head)
            return None if ranked is None else ranked + original[limit:]
        except Exception:
            return None


_DEFAULT_LISTWISE = ListwiseReranker()


def rerank_or_none(state: SessionState, candidates: list[Candidate]) -> list[Candidate] | None:
    return _DEFAULT_LISTWISE.rerank_or_none(state, candidates)
