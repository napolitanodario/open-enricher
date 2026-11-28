"""One VLM pass per dialogue interval, with earlier metadata passed back in."""

from __future__ import annotations

import json
import re
import tempfile
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any

from open_enricher.logging_setup import get_logger
from open_enricher.language import language_prompt
from open_enricher.stages.prompt_load import fill_prompt, load_prompt
from open_enricher.stages.vlm_core.config import MetadataConfig, ModelConfig
from open_enricher.stages.vlm_core.frames import sample_scene
from open_enricher.stages.vlm_core.model import (
    extract_metadata,
    extract_metadata_from_images,
    load_model,
    unload_model,
)
from open_enricher.stages.vlm_core.video import clip_video_range
from open_enricher.timings import TimingRecorder

logger = get_logger(__name__)


def _timed(
    timing: TimingRecorder | None,
    name: str,
    **extra: Any,
) -> Any:
    """
    Return a timing span, or a no-op context when no recorder was passed.

    Args:
        timing: Active recorder, or ``None`` when the caller does not time.
        name: Span name stored on the recorder.
        **extra: Fields stored next to the duration and the VRAM peak.

    Returns:
        A context manager.
    """
    if timing is None:
        return nullcontext()
    return timing.span(name, **extra)


SEGMENT_MAX_FRAMES = 40
SEGMENT_MAX_NEW_TOKENS = 450
# Prior scene JSON is re-sent on every call. Past this size, keep a short
# description per interval so the current frames still fit.
MEMORY_CHAR_BUDGET = 8000


def _parse_object(raw_text: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"```(?:json)?", "", raw_text).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    candidate = re.sub(r",\s*([}\]])", r"\1", cleaned[start : end + 1])
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _as_list(value: Any, *, limit: int = 6) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()][:limit]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def select_prior_scenes(prior: list[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    """
    Keep the scenes that will be re-sent as VLM memory.

    Args:
        prior: Metadata already written, oldest first.
        limit: How many of the most recent scenes to keep. ``None`` keeps
            every earlier scene.

    Returns:
        A new list, still oldest first.
    """
    if limit is None or len(prior) <= limit:
        return list(prior)
    return list(prior[-limit:])


def records_for_vlm_memory(
    prior: list[dict[str, Any]],
    *,
    visual_only: bool,
) -> list[dict[str, Any]]:
    """
    Copy scene records for the VLM memory block.

    Args:
        prior: Metadata already written.
        visual_only: When True, drop ``dialogue_description`` so speech from
            earlier windows does not re-enter the visual prompt.

    Returns:
        Copies of the records, oldest first.
    """
    rows: list[dict[str, Any]] = []
    for item in prior:
        row = dict(item)
        if visual_only:
            row.pop("dialogue_description", None)
        rows.append(row)
    return rows


def _memory_block(
    prior: list[dict[str, Any]],
    *,
    limit: int | None,
    visual_only: bool = False,
) -> tuple[str, bool]:
    """Return the prompt block of earlier metadata and whether it was compacted."""
    kept = records_for_vlm_memory(
        select_prior_scenes(prior, limit),
        visual_only=visual_only,
    )
    if not kept:
        return "", False
    if limit is None:
        scope = "Every earlier interval is included."
    else:
        scope = (
            f"Only the {limit} most recent intervals are included. "
            "Earlier intervals are omitted."
        )
    body = json.dumps(kept, ensure_ascii=False, separators=(",", ":"))
    compacted = False
    if len(body) > MEMORY_CHAR_BUDGET:
        compacted = True
        compact = [
            {
                "start_time": segment.get("start_time"),
                "end_time": segment.get("end_time"),
                "description": str(segment.get("description") or "")[:180],
            }
            for segment in kept
        ]
        body = json.dumps(compact, ensure_ascii=False, separators=(",", ":"))
        logger.warning(
            "ALERT: prior VLM metadata does not fit the memory budget; "
            "passing compact descriptions (chars=%s budget=%s scenes=%s kept=%s)",
            len(body),
            MEMORY_CHAR_BUDGET,
            len(prior),
            len(kept),
        )
    block = fill_prompt(
        load_prompt("vlm_scene_memory.md"),
        {"SCOPE": scope, "PRIOR_METADATA": body},
    )
    return "\n" + block + "\n", compacted


def _scene_prompt(
    scene: dict[str, Any],
    memory: str,
    *,
    visual_only: bool,
    extract_visual: bool = False,
    language: str = "en",
) -> str:
    start = str(scene.get("start_time") or "")
    end = str(scene.get("end_time") or "")
    if extract_visual:
        prompt = fill_prompt(
            language_prompt(load_prompt("vlm_scene_extract_prompt.md"), language),
            {"START": start, "END": end},
        )
    elif visual_only:
        prompt = fill_prompt(
            language_prompt(load_prompt("vlm_scene_visual_prompt.md"), language),
            {"START": start, "END": end, "MEMORY": memory},
        )
    else:
        prompt = fill_prompt(
            language_prompt(load_prompt("vlm_scene_prompt.md"), language),
            {
                "START": start,
                "END": end,
                "SCENE_DESCRIPTION": str(scene.get("description") or "").strip(),
                "MEMORY": memory,
            },
        )
    extra = str(scene.get("prompt_extra") or "").strip()
    if extra:
        prompt = f"{prompt.rstrip()}\n\n{extra}\n"
    return prompt


def _read_visual_fields(model: Any, processor: Any, clip: Path, config: MetadataConfig, language: str) -> str:
    """Collect evidence from the same clip when both free-form JSON responses are unusable."""
    questions = {
        "description": "Describe what is visible in one short sentence.",
        "subjects_and_objects": "List up to three visible people, objects or settings, one per line.",
        "actions": "List up to three visible actions or movements, one per line.",
        "camera_and_shot": "Describe the framing and camera movement in one short phrase.",
        "lighting_and_color": "Describe the lighting and colors in one short phrase.",
        "on_screen_text": "Copy at most one short, clearly readable example of on-screen text in its original language.",
    }
    fields = {}
    for field, question in questions.items():
        empty_instruction = ("Describe a blank screen if that is all the frames show." if field == "description"
                             else "Write NONE if nothing is visible for this field.")
        prompt = language_prompt(
            f"{question} Use only the current frames. Return plain text without labels, JSON or a code block. "
            + empty_instruction + " Write descriptions in {{LANGUAGE}}.",
            language,
        )
        answer = extract_metadata(model, processor, clip, replace(config, prompt=prompt, max_new_tokens=100)).strip()
        if not answer or answer.startswith(("{", "[", "```")):
            raise ValueError(f"Invalid visual field response for {field}")
        if answer.rstrip(".").upper() == "NONE":
            answer = ""
        if field == "description" and not answer:
            raise ValueError("Missing visual description after field recovery")
        if field in ("subjects_and_objects", "actions", "on_screen_text"):
            lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", s).strip() for s in answer.splitlines() if s.strip()]
            fields[field] = lines[:1 if field == "on_screen_text" else 3]
        else:
            fields[field] = answer
    return json.dumps(fields, ensure_ascii=False)


def _clip_generation(
    video_path: Path,
    start: float,
    end: float,
    model: Any,
    processor: Any,
    window_cfg: MetadataConfig,
    *,
    require_visual_json: bool = False,
    language: str = "en",
    frame_spans: list[dict[str, Any]] | None = None,
    video_filename: str = "",
    experiment_id: str = "",
    scene_index: int = 0,
    scene: dict[str, Any] | None = None,
) -> str:
    """Clip one window and run the legacy video sampler on it."""
    scene = scene or {}
    with tempfile.TemporaryDirectory(prefix="open-enricher-segment-") as tmp:
        clip = clip_video_range(
            video_path,
            start,
            max(1.0, end - start),
            Path(tmp) / f"window_{scene_index}.mp4",
        )
        if frame_spans is not None:
            frame_started = time.perf_counter()
            try:
                frame_span = sample_scene(
                    clip,
                    video_processor=processor.video_processor,
                    video_filename=video_filename,
                    experiment_id=experiment_id,
                    scene_index=scene_index,
                    start_seconds=start,
                    sample_fps=window_cfg.fps,
                    min_frames=window_cfg.min_frames,
                    max_frames=window_cfg.max_frames,
                )
            except Exception as exc:  # noqa: BLE001 - frames are not inference
                logger.warning("VLM frame cache skipped for window %s: %s", scene_index, exc)
                frame_span = {
                    "name": "vlm_frames",
                    "window": scene_index,
                    "error": str(exc),
                    "frames": [],
                }
            frame_span["seconds"] = round(time.perf_counter() - frame_started, 3)
            frame_span["start"] = scene.get("start_time")
            frame_span["end"] = scene.get("end_time")
            frame_spans.append(frame_span)
        raw = extract_metadata(model, processor, clip, window_cfg)
        if require_visual_json and not (_parse_object(raw) or {}).get("description"):
            logger.info("Requesting a complete visual JSON object for window %s", scene_index)
            retry_cfg = replace(
                window_cfg,
                prompt=window_cfg.prompt + (
                    "\nUse one short sentence for description and at most three short entries "
                    "per list. For credits, describe the credits once and quote at most one "
                    "readable example. Finish the JSON object including its closing brace."
                ),
                max_new_tokens=max(window_cfg.max_new_tokens, 600),
            )
            first_response = raw
            raw = extract_metadata(model, processor, clip, retry_cfg)
            if not (_parse_object(raw) or {}).get("description"):
                logger.warning("Reading visual fields individually for window %s after two invalid JSON responses", scene_index)
                if frame_spans and frame_spans[-1].get("window") == scene_index:
                    frame_spans[-1]["visual_recovery"] = {
                        "method": "individual_fields", "invalid_responses": [first_response, raw],
                    }
                raw = _read_visual_fields(model, processor, clip, window_cfg, language)
        return raw


def run_segmented(
    video_path: Path,
    video_filename: str,
    model_cfg: ModelConfig,
    meta_cfg: MetadataConfig,
    scenes: list[dict[str, Any]],
    *,
    timing: TimingRecorder | None = None,
    experiment_id: str = "",
    vlm_memory_scenes: int | None = None,
    visual_only: bool = False,
    extract_visual: bool = False,
    continue_on_error: bool = False,
    language: str = "en",
) -> tuple[str, dict[str, Any], bool, int, list[dict[str, Any]]]:
    """
    Run the VLM once per dialogue interval and accumulate the metadata.

    Each call receives the dialogue description of that interval and the
    metadata written for earlier intervals. Qwen does not keep that record
    internally between ``generate`` calls.

    Args:
        video_path: Source video.
        video_filename: File name, used only in the saved payload.
        model_cfg: VLM load configuration.
        meta_cfg: Sampling and generation configuration.
        scenes: Dialogue-stage scene objects with start/end seconds and a
            prose ``description``.
        timing: Optional recorder for load and per-window generate spans.
        experiment_id: Active experiment id copied into the payload.
        vlm_memory_scenes: How many previous scene records to pass. ``None``
            passes every earlier scene.
        visual_only: When True, the prompt has frames and prior visual
            metadata only. The dialogue description is stored on the segment
            for the final LLM and is not shown to the VLM.
        extract_visual: When True, each call sees only the frames of the
            current window. No dialogue summary and no earlier scenes are
            passed. The prompt asks for visible objects and actions.

    Returns:
        Raw joined text, parsed timeline document, whether every window
        parsed as JSON, the window count, and one frame-sample span per
        window. Frame files are an explainability copy; inference ignores them.
    """
    if not scenes:
        raise ValueError(f"No dialogue scenes to show the VLM for {video_filename}")

    logger.info(
        "VLM over %s scenes (experiment=%s)",
        len(scenes),
        experiment_id or "default",
    )
    logger.info("Loading VLM: %s (4-bit=%s)...", model_cfg.model_id, model_cfg.load_in_4bit)
    with _timed(timing, "vlm_load", model_id=model_cfg.model_id):
        model, processor = load_model(model_cfg)

    segments: list[dict[str, Any]] = []
    raw_parts: list[str] = []
    frame_spans: list[dict[str, Any]] = []
    parsed_count = 0
    memory_compacted = False
    try:
        for index, scene in enumerate(scenes, start=1):
            start = float(scene["start_seconds"])
            end = float(scene["end_seconds"])
            logger.info(
                "VLM scene %s/%s %s-%s",
                index,
                len(scenes),
                scene.get("start_time"),
                scene.get("end_time"),
            )
            if extract_visual:
                memory, compacted = "", False
            else:
                memory, compacted = _memory_block(
                    segments,
                    limit=vlm_memory_scenes,
                    visual_only=visual_only,
                )
            memory_compacted = memory_compacted or compacted
            prompt = _scene_prompt(
                scene,
                memory,
                visual_only=visual_only,
                extract_visual=extract_visual,
                language=language,
            )
            window_cfg = replace(
                meta_cfg,
                prompt=prompt,
                min_frames=2,
                max_frames=int(scene.get("max_frames") or SEGMENT_MAX_FRAMES),
                max_new_tokens=SEGMENT_MAX_NEW_TOKENS,
                video_seconds=None,
            )
            if scene.get("no_frame"):
                raw_parts.append(
                    f"--- {scene.get('start_time')}-{scene.get('end_time')} ---\n"
                    "No frame was sampled for this scene."
                )
                segments.append(
                    {
                        "start_time": scene.get("start_time"),
                        "end_time": scene.get("end_time"),
                        "dialogue_description": str(scene.get("description") or "").strip(),
                        "description": ("Nessun fotogramma campionato per questa scena." if language == "it"
                                        else "No frame was sampled for this scene."),
                        "subjects_and_objects": [],
                        "actions": [],
                        "camera_and_shot": "",
                        "lighting_and_color": "",
                        "on_screen_text": [],
                    }
                )
                frame_spans.append(
                    {
                        "name": "vlm_frames",
                        "window": index,
                        "seconds": 0,
                        "frames": [],
                        "no_frame": True,
                        "start": scene.get("start_time"),
                        "end": scene.get("end_time"),
                    }
                )
                continue
            image_paths = [
                path for path in (scene.get("image_paths") or []) if Path(path).is_file()
            ]
            with _timed(
                timing,
                "vlm_generate",
                window=index,
                start=scene.get("start_time"),
                end=scene.get("end_time"),
                model_id=model_cfg.model_id,
            ):
                try:
                    if image_paths:
                        frame_started = time.perf_counter()
                        frame_span = {
                            "name": "vlm_frames",
                            "window": index,
                            "sampled_frames": len(image_paths),
                            "frames": [
                                {"file": path, "source": "explicit"} for path in image_paths
                            ],
                            "seconds": round(time.perf_counter() - frame_started, 3),
                        }
                        frame_span["start"] = scene.get("start_time")
                        frame_span["end"] = scene.get("end_time")
                        frame_spans.append(frame_span)
                        try:
                            raw_text = extract_metadata_from_images(
                                model, processor, image_paths, window_cfg
                            )
                        except Exception as exc:
                            if not continue_on_error:
                                raise
                            frame_span["fallback"] = "clip"
                            frame_span["error"] = str(exc)
                            logger.warning(
                                "Explicit frames failed for window %s, using the clip: %s",
                                index,
                                exc,
                            )
                            raw_text = _clip_generation(
                                video_path, start, end, model, processor, window_cfg
                            )
                    else:
                        raw_text = _clip_generation(
                            video_path,
                            start,
                            end,
                            model,
                            processor,
                            window_cfg,
                            require_visual_json=extract_visual and not continue_on_error,
                            language=language,
                            frame_spans=frame_spans,
                            video_filename=video_filename,
                            experiment_id=experiment_id,
                            scene_index=index,
                            scene=scene,
                        )
                except Exception:
                    if not continue_on_error:
                        raise
                    logger.warning("VLM scene %s failed; keeping the dialogue record", index)
                    raw_text = ""

            raw_parts.append(f"--- {scene.get('start_time')}-{scene.get('end_time')} ---\n{raw_text}")
            parsed = _parse_object(raw_text)
            no_description = "Nessuna descrizione." if language == "it" else "No description."
            if parsed is None:
                description = raw_text.strip()[:500] or no_description
                subjects: list[str] = []
                actions: list[str] = []
                camera = ""
                lighting = ""
                texts: list[str] = []
            else:
                parsed_count += 1
                description = str(parsed.get("description") or "").strip() or no_description
                list_limit = 12 if extract_visual else 6
                subjects = _as_list(parsed.get("subjects_and_objects"), limit=list_limit)
                actions = _as_list(parsed.get("actions"), limit=list_limit)
                camera = str(parsed.get("camera_and_shot") or "").strip()
                lighting = str(parsed.get("lighting_and_color") or "").strip()
                texts = _as_list(parsed.get("on_screen_text"))
            segments.append(
                {
                    "start_time": scene.get("start_time"),
                    "end_time": scene.get("end_time"),
                    "dialogue_description": str(scene.get("description") or "").strip(),
                    "description": description,
                    "subjects_and_objects": subjects,
                    "actions": actions,
                    "camera_and_shot": camera,
                    "lighting_and_color": lighting,
                    "on_screen_text": texts,
                }
            )
    finally:
        logger.info("Unloading VLM and freeing VRAM...")
        unload_model(model, processor)

    payload = {
        "experiment_id": experiment_id,
        "scene_timeline": segments,
        "segments_parsed": parsed_count,
        "segments_total": len(scenes),
        "vlm_memory_scenes": 0 if extract_visual else vlm_memory_scenes,
        "visual_only": visual_only or extract_visual,
        "extract_visual": extract_visual,
        "memory_compacted": memory_compacted,
    }
    raw_document = "\n\n".join(raw_parts)
    json_valid = parsed_count == len(scenes) and len(scenes) > 0
    return raw_document, payload, json_valid, len(scenes), frame_spans
