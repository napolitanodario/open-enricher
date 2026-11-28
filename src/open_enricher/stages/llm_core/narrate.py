"""Rewrite one scene from its dialogue summary plus visual extracts."""

from __future__ import annotations

import json
from typing import Any

from open_enricher.language import language_prompt
from open_enricher.stages.pipeline_spec import DEFAULT_NARRATION_CONTEXT_SCENES as NARRATION_CONTEXT_SCENES
from open_enricher.stages.prompt_load import fill_prompt, load_prompt
from open_enricher.stages.vlm_core.enrichment import format_dialogue_lines


def previous_scene_summaries(
    scenes: list[dict[str, Any]],
    index: int,
    *,
    limit: int = NARRATION_CONTEXT_SCENES,
) -> list[dict[str, Any]]:
    """
    Return the dialogue summaries of the scenes immediately before ``index``.

    Args:
        scenes: Scene records in timeline order. Each may carry
            ``dialogue_description``.
        index: Index of the scene being narrated.
        limit: How many earlier scenes to keep. The oldest of those is first.

    Returns:
        Up to ``limit`` earlier scene dicts. Empty when ``index`` is 0.
    """
    if index <= 0 or limit <= 0:
        return []
    start = max(0, index - limit)
    return list(scenes[start:index])


def _exchange_text(scene: dict[str, Any], limit: int) -> str:
    """Select by importance, then present the retained exchanges in spoken order."""
    exchanges = (scene.get("dialogue_exchanges") or [])[:limit]
    return "\n\n".join(
        "\n".join(format_dialogue_lines(exchange))
        for exchange in sorted(exchanges, key=lambda lines: lines[0]["line_id"])
    )


def format_prior_summaries(scenes: list[dict[str, Any]], *, with_exchanges: bool = False) -> str:
    """
    Format earlier dialogue summaries for the narration prompt.

    Args:
        scenes: Records from ``previous_scene_summaries``.

    Returns:
        A plain-text block, or a line saying this is the opening scene.
    """
    if not scenes:
        return "None. This is the opening scene."
    blocks: list[str] = []
    for scene in scenes:
        start = str(scene.get("start_time") or "")
        end = str(scene.get("end_time") or "")
        summary = str(scene.get("dialogue_description") or "").strip() or "(no summary)"
        blocks.append(f"{start}-{end}: {summary}")
        if with_exchanges and (exchange := _exchange_text(scene, 1)):
            blocks[-1] += f"\nEarlier exchange:\n{exchange}"
    return "\n\n".join(blocks)


def narration_system_prompt(*, with_exchanges: bool = False, language: str = "en") -> str:
    """Return the system message for one scene narration."""
    template = "scene_exchange_narration_system_prompt.md" if with_exchanges else "scene_narration_system_prompt.md"
    return language_prompt(load_prompt(template), language)


def narration_user_prompt(
    scene: dict[str, Any], prior: list[dict[str, Any]], *, with_exchanges: bool = False,
    full_visual: bool = False, language: str = "en",
) -> str:
    """
    Return the user message for one scene narration.

    Args:
        scene: Current scene, including ``dialogue_description``,
            ``subjects_and_objects``, and ``actions``.
        prior: Earlier scenes whose summaries are continuity only.
    """
    use_exchanges = with_exchanges and not full_visual
    fields = {
        "START": str(scene.get("start_time") or ""),
        "END": str(scene.get("end_time") or ""),
        "PRIOR": format_prior_summaries(prior, with_exchanges=use_exchanges),
        "SCENE_SUMMARY": str(scene.get("dialogue_description") or "").strip() or "(no dialogue summary)",
    }
    if full_visual:
        # Before narration, description is the VLM's own sentence. The archived
        # timeline also preserves it explicitly as visual_description.
        observations = {
            "visual_description": scene.get("visual_description") or scene.get("description", ""),
            **{key: scene.get(key) for key in (
                "subjects_and_objects", "actions", "camera_and_shot", "lighting_and_color", "on_screen_text"
            )},
        }
        fields["VISUAL"] = json.dumps(observations, ensure_ascii=False)
        template = "scene_full_visual_narration_user_prompt.md"
    else:
        objects = scene.get("subjects_and_objects") or []
        actions = scene.get("actions") or []
        fields["OBJECTS"] = "\n".join(f"- {item}" for item in objects) or "- none listed"
        fields["ACTIONS"] = "\n".join(f"- {item}" for item in actions) or "- none listed"
        template = "scene_narration_user_prompt.md"
        if use_exchanges:
            fields["EXCHANGES"] = _exchange_text(scene, 2) or "(none selected)"
            template = "scene_exchange_narration_user_prompt.md"
    return fill_prompt(language_prompt(load_prompt(template), language), fields)
