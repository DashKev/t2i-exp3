"""Shared local/remote LLM utilities for prompt-rewrite baselines."""

from .openai_compatible import ChatResponse, OpenAICompatibleClient

__all__ = ["ChatResponse", "OpenAICompatibleClient"]
