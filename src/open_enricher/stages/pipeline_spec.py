"""Experiment pipeline knobs for the single local metadata run.

A JSON file under the package (or the path in OPEN_ENRICHER_PIPELINE_SPEC)
sets the experiment id and the fixed interval used as a scene. The local
experiment loop points this env var at the packaged variant.

Optional ``features``, ``budgets``, ``frame_allocation`` and ``checkpoints``
blocks select the scene, graph and visual-memory experiments. An absent
block leaves variants 01-08 on the current four-stage path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_PATH = Path(__file__).resolve().parent / "pipeline_spec.json"

_ALLOCATION_MODES = {"adaptive", "duration"}
DEFAULT_NARRATION_CONTEXT_SCENES = 2


@dataclass(frozen=True)
class FeatureFlags:
    """Optional experiment features. Every flag defaults off."""

    bassl: bool = False
    adaptive_frames: bool = False
    character_graph: bool = False
    visual_memory: bool = False

    @property
    def any_enabled(self) -> bool:
        """True when this variant leaves the legacy four-stage path."""
        return (
            self.bassl
            or self.adaptive_frames
            or self.character_graph
            or self.visual_memory
        )


@dataclass(frozen=True)
class Budgets:
    """Per-feature limits. Absent JSON keys keep these defaults."""

    max_qwen_frames: int = 40
    history_frame_fraction: float = 0.25
    face_sample_fps: float = 5.0
    memory_candidate_fps: float = 1.0
    short_term_capacity: int = 16
    short_term_target: int = 8
    long_term_capacity: int = 64
    coverage_reserve_frames: int = 2
    pacing_cap: float = 0.35
    match_threshold: float = 0.45
    match_margin: float = 0.08
    max_scene_seconds: int = 300


@dataclass(frozen=True)
class Checkpoints:
    """Local weight overrides. ``None`` uses ``dataset/models_cache/external``."""

    bassl: str | None = None
    insightface: str | None = None
    talknet: str | None = None
    eva_clip: str | None = None
    qformer: str | None = None


@dataclass(frozen=True)
class PipelineSpec:
    """
    Validated knobs for the scene-context pipeline.

    Fields mirror ``experiments/variants/01-scene-context.json``. Optional
    feature blocks are present on variants 09-12 and stay off otherwise.
    """

    experiment_id: str
    title: str
    interval_seconds: int
    vlm_memory_scenes: int | None
    visual_only: bool
    narrate_from_visual: bool
    features: FeatureFlags = FeatureFlags()
    frame_allocation: str | None = None
    budgets: Budgets = Budgets()
    checkpoints: Checkpoints = Checkpoints()
    narration_context_scenes: int = DEFAULT_NARRATION_CONTEXT_SCENES
    select_dialogue_exchanges: bool = False
    # The dialogue stage can still select exchanges. This option changes only
    # the narrator's inputs, leaving upstream summary generation comparable.
    narrate_full_visual_without_exchanges: bool = False

    def allocation_mode(self) -> str | None:
        """
        Return how current frames are counted for this variant.

        ``None`` keeps the legacy clip sampler. ``duration`` and ``adaptive``
        build an explicit frame plan. Visual memory without an explicit mode
        still plans frames so the history reserve can be charged.
        """
        if self.frame_allocation in _ALLOCATION_MODES:
            return self.frame_allocation
        if self.features.adaptive_frames:
            return "adaptive"
        if self.features.bassl or self.features.visual_memory:
            return "duration"
        return None

    def history_fraction(self) -> float:
        """History share of the Qwen frame budget. Zero when memory is off."""
        if not self.features.visual_memory:
            return 0.0
        return self.budgets.history_frame_fraction


def _require_positive_int(name: str, value: Any) -> int:
    """
    Require a positive integer field.

    Args:
        name: Field name for error messages.
        value: Raw JSON value.

    Returns:
        The validated positive int.

    Raises:
        ValueError: When the value is not a positive int.
    """
    if type(value) is not int or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive int")
    return value


def _optional_positive_int(name: str, value: Any) -> int | None:
    """
    Accept a positive int or JSON null.

    Args:
        name: Field name for error messages.
        value: Raw JSON value. ``None`` means the limit is unset.

    Returns:
        The positive int, or ``None`` when the field is null or absent.

    Raises:
        ValueError: When the value is present and not a positive int.
    """
    if value is None:
        return None
    return _require_positive_int(name, value)


def _require_bool(name: str, value: Any, *, default: bool = False) -> bool:
    """
    Require a JSON boolean, treating null and absence as ``default``.

    Args:
        name: Field name for error messages.
        value: Raw JSON value.
        default: Value used when the field is null or missing.

    Returns:
        The boolean value.

    Raises:
        ValueError: When the value is present and not a JSON boolean.
    """
    if value is None:
        return default
    if type(value) is not bool:
        raise ValueError(f"{name} must be a bool")
    return value


def _require_unit_float(name: str, value: Any, default: float) -> float:
    """Accept a finite float in ``[0, 1]``, or ``default`` when the key is absent."""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if number < 0 or number > 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return number


def _optional_str(value: Any) -> str | None:
    """Accept a path string or JSON null."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("checkpoint overrides must be non-empty strings or null")
    return value.strip()


def _load_features(value: Any) -> FeatureFlags:
    """Parse the optional ``features`` object."""
    if value is None:
        return FeatureFlags()
    if not isinstance(value, dict):
        raise ValueError("features must be an object")
    return FeatureFlags(
        bassl=_require_bool("features.bassl", value.get("bassl"), default=False),
        adaptive_frames=_require_bool(
            "features.adaptive_frames", value.get("adaptive_frames"), default=False
        ),
        character_graph=_require_bool(
            "features.character_graph", value.get("character_graph"), default=False
        ),
        visual_memory=_require_bool(
            "features.visual_memory", value.get("visual_memory"), default=False
        ),
    )


def _load_budgets(value: Any) -> Budgets:
    """Parse the optional ``budgets`` object, keeping defaults for missing keys."""
    if value is None:
        return Budgets()
    if not isinstance(value, dict):
        raise ValueError("budgets must be an object")
    defaults = Budgets()
    max_frames = value.get("max_qwen_frames", defaults.max_qwen_frames)
    return Budgets(
        max_qwen_frames=_require_positive_int("budgets.max_qwen_frames", max_frames),
        history_frame_fraction=_require_unit_float(
            "budgets.history_frame_fraction",
            value.get("history_frame_fraction"),
            defaults.history_frame_fraction,
        ),
        face_sample_fps=float(
            value.get("face_sample_fps", defaults.face_sample_fps)
        ),
        memory_candidate_fps=float(
            value.get("memory_candidate_fps", defaults.memory_candidate_fps)
        ),
        short_term_capacity=_require_positive_int(
            "budgets.short_term_capacity",
            value.get("short_term_capacity", defaults.short_term_capacity),
        ),
        short_term_target=_require_positive_int(
            "budgets.short_term_target",
            value.get("short_term_target", defaults.short_term_target),
        ),
        long_term_capacity=_require_positive_int(
            "budgets.long_term_capacity",
            value.get("long_term_capacity", defaults.long_term_capacity),
        ),
        coverage_reserve_frames=_require_positive_int(
            "budgets.coverage_reserve_frames",
            value.get("coverage_reserve_frames", defaults.coverage_reserve_frames),
        ),
        pacing_cap=_require_unit_float(
            "budgets.pacing_cap", value.get("pacing_cap"), defaults.pacing_cap
        ),
        match_threshold=_require_unit_float(
            "budgets.match_threshold",
            value.get("match_threshold"),
            defaults.match_threshold,
        ),
        match_margin=_require_unit_float(
            "budgets.match_margin", value.get("match_margin"), defaults.match_margin
        ),
        max_scene_seconds=_require_positive_int(
            "budgets.max_scene_seconds",
            value.get("max_scene_seconds", defaults.max_scene_seconds),
        ),
    )


def _load_checkpoints(value: Any) -> Checkpoints:
    """Parse optional local checkpoint overrides."""
    if value is None:
        return Checkpoints()
    if not isinstance(value, dict):
        raise ValueError("checkpoints must be an object")
    return Checkpoints(
        bassl=_optional_str(value.get("bassl")),
        insightface=_optional_str(value.get("insightface")),
        talknet=_optional_str(value.get("talknet")),
        eva_clip=_optional_str(value.get("eva_clip")),
        qformer=_optional_str(value.get("qformer")),
    )


def _load_allocation(value: Any) -> str | None:
    """Parse ``frame_allocation``. Absence keeps the legacy sampler."""
    if value is None:
        return None
    if value not in _ALLOCATION_MODES:
        raise ValueError("frame_allocation must be 'adaptive', 'duration', or null")
    return str(value)


def load_pipeline_spec(path: Path | None = None) -> PipelineSpec:
    """
    Load and validate the active experiment pipeline specification.

    Preference order for the file path:
    1. ``OPEN_ENRICHER_PIPELINE_SPEC`` when set
    2. Explicit ``path`` argument
    3. Packaged ``pipeline_spec.json`` next to this module

    Args:
        path: Optional explicit path used when the env override is empty.

    Returns:
        Validated ``PipelineSpec`` dataclass.

    Raises:
        FileNotFoundError: When the resolved path does not exist.
        ValueError: When the JSON shape or field types are invalid.
    """
    override = os.environ.get("OPEN_ENRICHER_PIPELINE_SPEC", "").strip()
    spec_path = Path(override) if override else (path or _DEFAULT_PATH)
    if not spec_path.is_file():
        raise FileNotFoundError(f"Pipeline spec not found: {spec_path}")
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Pipeline spec must be a JSON object")

    face_fps = data.get("budgets", {}) if isinstance(data.get("budgets"), dict) else {}
    for key in ("face_sample_fps", "memory_candidate_fps"):
        if key in face_fps and (
            isinstance(face_fps[key], bool)
            or not isinstance(face_fps[key], (int, float))
            or float(face_fps[key]) <= 0
        ):
            raise ValueError(f"budgets.{key} must be a positive number")

    narrate = _require_bool("narrate_from_visual", data.get("narrate_from_visual"))
    exchanges = _require_bool("select_dialogue_exchanges", data.get("select_dialogue_exchanges"))
    if exchanges and not narrate:
        raise ValueError("select_dialogue_exchanges requires narrate_from_visual")
    full_visual = _require_bool("narrate_full_visual_without_exchanges", data.get("narrate_full_visual_without_exchanges"))
    if full_visual and not narrate:
        raise ValueError("narrate_full_visual_without_exchanges requires narrate_from_visual")
    context = data.get("narration_context_scenes", DEFAULT_NARRATION_CONTEXT_SCENES)
    if type(context) is not int or context < 0:
        raise ValueError("narration_context_scenes must be a non-negative integer")
    return PipelineSpec(
        experiment_id=str(data.get("experiment_id") or "default").strip(),
        title=str(data.get("title") or "default").strip(),
        interval_seconds=_require_positive_int(
            "interval_seconds", int(data.get("interval_seconds", 300))
        ),
        vlm_memory_scenes=_optional_positive_int(
            "vlm_memory_scenes", data.get("vlm_memory_scenes")
        ),
        visual_only=_require_bool("visual_only", data.get("visual_only"), default=False),
        narrate_from_visual=narrate,
        narration_context_scenes=context,
        select_dialogue_exchanges=exchanges,
        narrate_full_visual_without_exchanges=full_visual,
        features=_load_features(data.get("features")),
        frame_allocation=_load_allocation(data.get("frame_allocation")),
        budgets=_load_budgets(data.get("budgets")),
        checkpoints=_load_checkpoints(data.get("checkpoints")),
    )
