"""Evaluator provider registry.

Register new backends (Codex, Gemini, ...) by adding a factory here. Host-side
scoring and report assembly are independent of provider selection.
"""

from __future__ import annotations

from typing import Callable

from open_enricher.evaluation.providers.base import EvaluatorProvider
from open_enricher.evaluation.providers.codex import CodexProvider
from open_enricher.evaluation.providers.cursor_grok import CursorGrokProvider

ProviderFactory = Callable[..., EvaluatorProvider]

PROVIDERS: dict[str, ProviderFactory] = {
    "codex": CodexProvider,
    "cursor": CursorGrokProvider,
}

DEFAULT_PROVIDER = "codex"


def get_provider(
    name: str = DEFAULT_PROVIDER,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> EvaluatorProvider:
    """
    Construct an evaluator provider by registry key.

    Args:
        name: Registered provider key (``cursor`` or ``codex``).
        api_key: Optional API key forwarded to the provider.
        model: Optional model id override.

    Returns:
        A ready-to-use ``EvaluatorProvider``.

    Raises:
        KeyError: When ``name`` is not registered.
    """
    try:
        factory = PROVIDERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(PROVIDERS))
        raise KeyError(f"Unknown evaluator provider {name!r}. Known: {known}") from exc
    return factory(api_key=api_key, model=model)


__all__ = [
    "DEFAULT_PROVIDER",
    "PROVIDERS",
    "CursorGrokProvider",
    "CodexProvider",
    "EvaluatorProvider",
    "get_provider",
]
