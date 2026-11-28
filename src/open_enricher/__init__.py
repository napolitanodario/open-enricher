"""Open-Enricher: local video understanding pipeline for recommender metadata.

Stages run in process on a GPU host:
audio (WhisperX) -> dialogue (Qwen3-8B plot, then per-interval descriptions) ->
vlm (Qwen3-VL) -> llm (VideoProfile).

Scoring is separate: ``python -m open_enricher.evaluation`` judges scene understanding and RecSys profiles
with an external provider (Codex with a ChatGPT login by default).

Experiment variants under ``open_enricher.experiments.variants`` are executed
in sequence by ``python -m open_enricher``.
"""

__version__ = "0.2.0"
