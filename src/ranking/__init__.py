"""Ranking leaf component. Never imports retrieval/, dialog/, or memory/."""

from .primary import rerank

__all__ = ["rerank"]
