"""Two sequential Qwen3-8B passes over the Whisper transcript.

First pass: one high-level plot summary of every dialogue line, then the
model is unloaded. Second pass: the same weights are loaded again and, for
each interval, summarize only the lines assigned to that interval. Narration
variants can select ASR exchanges during the same summary generation.
"""

from __future__ import annotations

import gc
import json
from functools import partial
from typing import Any

from open_enricher import storage
from open_enricher.fingerprint import current_stamp
from open_enricher.language import language_prompt, metadata_language
from open_enricher.logging_setup import get_logger
from open_enricher.manifest import matching_artifact
from open_enricher.narrative import cue_from_lines, parse_label
from open_enricher.stages.dialogue_exchanges import MissingIntervalSummary, resolve_exchanges
from open_enricher.stages.pipeline_spec import load_pipeline_spec
from open_enricher.stages.prompt_load import fill_prompt, load_prompt
from open_enricher.stages.vlm_core.enrichment import (
    SUMMARY_MODEL_ID,
    _strip_think,
    fit_dialogue_lines,
    format_dialogue_lines,
)
from open_enricher.stages.vlm_core.video import probe_duration_seconds
from open_enricher.timeline import Interval, assign_utterances, fixed_windows
from open_enricher.timings import TimingRecorder

logger = get_logger(__name__)

PLOT_MAX_NEW_TOKENS = 520
SCENE_MAX_NEW_TOKENS = 380
_EMPTY_SCENE = "No transcribed dialogue in this interval."


def interval_windows(duration: float, interval_seconds: int) -> list[tuple[float, float]]:
    """
    Split a duration into contiguous intervals of ``interval_seconds``.

    Args:
        duration: Video length in seconds.
        interval_seconds: Window length. The last window is shorter when the
            file does not divide evenly.

    Returns:
        ``(start, end)`` pairs in seconds. Empty when ``duration`` is under
        one second.
    """
    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration - 1:
        end = min(duration, start + interval_seconds)
        windows.append((start, end))
        start = end
    return windows


def lines_in_window(lines: list[dict[str, Any]], start: float, end: float) -> list[dict[str, Any]]:
    """
    Return Whisper rows whose start timestamp falls in ``[start, end)``.

    Args:
        lines: Audio-stage dialogue objects.
        start: Window start in seconds.
        end: Window end in seconds.
    """
    picked: list[dict[str, Any]] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        try:
            stamp = float(line.get("start_timestamp", -1))
        except (TypeError, ValueError):
            continue
        if start <= stamp < end:
            picked.append(line)
    return picked


def _clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _sampling_note(sampled: bool) -> str:
    if sampled:
        return "Some middle lines were sampled out because the full transcript did not fit the model input."
    return "Every dialogue line is included, in order."


def _plot_messages(transcript: str, *, sampled: bool, language: str = "en") -> list[dict[str, str]]:
    return [
        {"role": "system", "content": language_prompt(load_prompt("plot_system_prompt.md"), language)},
        {
            "role": "user",
            "content": fill_prompt(
                language_prompt(load_prompt("plot_user_prompt.md"), language),
                {"SAMPLING_NOTE": _sampling_note(sampled), "TRANSCRIPT": transcript},
            ),
        },
    ]


def _scene_messages(
    transcript: str,
    *,
    start: float,
    end: float,
    sampled: bool,
    select_exchanges: bool = False,
    language: str = "en",
) -> list[dict[str, str]]:
    system = (
        "scene_exchange_selection_system_prompt.md" if select_exchanges
        else "scene_dialogue_system_prompt.md"
    )
    return [
        {"role": "system", "content": language_prompt(load_prompt(system), language)},
        {
            "role": "user",
            "content": fill_prompt(
                language_prompt(load_prompt("scene_dialogue_user_prompt.md"), language),
                {
                    "START": _clock(start),
                    "END": _clock(end),
                    "SAMPLING_NOTE": _sampling_note(sampled),
                    "TRANSCRIPT": transcript,
                },
            ),
        },
    ]


def _prompt_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    encoded = tokenizer([prompt_text], return_tensors="pt")
    return int(encoded.input_ids.shape[-1])


def _load_text_model(timing: TimingRecorder, span_name: str) -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"
    quantization_config = None
    if device == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    logger.info("Loading %s (%s)...", SUMMARY_MODEL_ID, span_name)
    with timing.span(span_name, model_id=SUMMARY_MODEL_ID):
        tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            SUMMARY_MODEL_ID,
            quantization_config=quantization_config,
            device_map="auto" if device == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    return model, tokenizer, device


def _unload_text_model(model: Any, tokenizer: Any) -> None:
    import torch

    logger.info("Unloading %s and freeing VRAM...", SUMMARY_MODEL_ID)
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate(
    model: Any,
    tokenizer: Any,
    device: str,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int,
    min_chars: int = 40,
) -> str:
    import torch

    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    model_inputs = tokenizer([prompt_text], return_tensors="pt").to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.05,
        )
    new_tokens = generated_ids[0][len(model_inputs.input_ids[0]) :]
    text = _strip_think(tokenizer.decode(new_tokens, skip_special_tokens=True))
    if len(text) < min_chars:
        raise RuntimeError(f"Text model output is too short to be useful: {text!r}")
    return text


def _input_budget(model: Any, reserved_tokens: int) -> int:
    max_position = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    budget = max_position - reserved_tokens
    if budget < 1:
        raise RuntimeError(
            f"{SUMMARY_MODEL_ID} max_position_embeddings={max_position} "
            f"does not leave room for {reserved_tokens} generated tokens"
        )
    return budget


class DialogueStage:
    """Plot summary, then per-interval dialogue descriptions, on Qwen3-8B."""

    name = "dialogue"

    def run(self, video_filename: str) -> dict[str, Any]:
        """
        Write the plot summary and one dialogue description per interval.

        Args:
            video_filename: Exact file name under ``video_inputs``.

        Returns:
            Status dict with the output file name and scene count.

        Raises:
            FileNotFoundError: When the video or the audio JSON is missing.
        """
        video_path = storage.video_inputs() / video_filename
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found in dataset: {video_path}")
        audio_path = storage.get_output_dir("audio") / f"{video_filename}_audio.json"
        spec = load_pipeline_spec()
        timing = TimingRecorder(self.name)
        if not audio_path.is_file():
            if spec.features.any_enabled:
                return self._write_without_audio(video_filename, video_path, spec, timing)
            raise FileNotFoundError(f"Audio transcript not found: {audio_path}")

        raw_lines = json.loads(audio_path.read_text(encoding="utf-8"))
        language = metadata_language(video_filename)
        if not isinstance(raw_lines, list):
            raise ValueError("Audio metadata must be a list of dialogue lines")
        if spec.select_dialogue_exchanges:
            raw_lines = [
                row for row in raw_lines
                if isinstance(row, dict) and str(row.get("text") or "").strip()
            ]
            dialogue_lines = format_dialogue_lines(raw_lines) if raw_lines else []
        else:
            dialogue_lines = format_dialogue_lines(raw_lines)
        speakers = {
            str(line.get("speaker") or "UNKNOWN")
            for line in raw_lines
            if isinstance(line, dict)
        }
        if speakers <= {"UNKNOWN"}:
            logger.warning(
                "ALERT: speaker diarization produced no distinct speakers; "
                "every line is UNKNOWN"
            )
        logger.info(
            "Dialogue stage for %s: %s lines, speakers=%s",
            video_filename,
            len(dialogue_lines),
            sorted(speakers),
        )

        duration = probe_duration_seconds(video_path)
        buckets = None
        scene_ids = None
        if spec.features.any_enabled:
            intervals = _intervals_for(video_filename, duration, spec.interval_seconds)
            buckets = assign_utterances(raw_lines, intervals)
            scene_ids = [item.id for item in intervals]
            windows = [(item.start, item.end) for item in intervals]
        else:
            windows = interval_windows(duration, spec.interval_seconds)
        try:
            if dialogue_lines:
                plot_summary, plot_sampled = self._plot_pass(dialogue_lines, timing, language=language)
                scenes = self._scene_pass(
                    raw_lines,
                    windows,
                    timing,
                    buckets=buckets,
                    scene_ids=scene_ids,
                    score_narrative=spec.features.adaptive_frames,
                    select_exchanges=spec.select_dialogue_exchanges,
                    language=language,
                )
            else:
                plot_summary, plot_sampled = "", False
                scenes = _empty_scenes(windows, scene_ids, buckets, raw_lines, language=language)
                for scene in scenes:
                    scene["dialogue_exchanges"] = []
            status = "success"
            error = None
        except Exception as exc:
            if not spec.features.any_enabled:
                raise
            logger.warning("Dialogue model failed for %s: %s", video_filename, exc)
            plot_summary, plot_sampled = "", False
            scenes = _empty_scenes(windows, scene_ids, buckets, raw_lines, language=language)
            status = "degraded"
            error = str(exc)

        payload = {
            "experiment_id": spec.experiment_id,
            "interval_seconds": spec.interval_seconds,
            "plot_summary": plot_summary,
            "plot_sampled": plot_sampled,
            "metadata_language": language,
            "scenes": scenes,
        }
        if spec.features.any_enabled:
            payload.update(current_stamp())
            payload["status"] = status
        output_dir = storage.get_output_dir(self.name)
        output_path = output_dir / f"{video_filename}_dialogue.json"
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        timings_path = output_dir / f"{video_filename}_dialogue_timings.json"
        timing.save(timings_path)
        logger.info("Saved dialogue stage to %s (%s scenes)", output_path.name, len(scenes))
        result = {
            "stage": self.name,
            "status": status,
            "scenes": len(scenes),
            "plot_sampled": plot_sampled,
            "timings": timing.document(),
            "output_file": output_path.name,
            "timings_file": timings_path.name,
        }
        if error:
            result["error"] = error
        return result

    def _plot_pass(
        self,
        dialogue_lines: list[str],
        timing: TimingRecorder,
        *, language: str = "en",
    ) -> tuple[str, bool]:
        model, tokenizer, device = _load_text_model(timing, "plot_load")
        try:
            budget = _input_budget(model, PLOT_MAX_NEW_TOKENS)

            def prompt_tokens(body: str, sampled: bool) -> int:
                return _prompt_token_count(tokenizer, _plot_messages(body, sampled=sampled, language=language))

            transcript, sampled = fit_dialogue_lines(
                dialogue_lines,
                input_budget=budget,
                prompt_tokens=prompt_tokens,
            )
            with timing.span("plot_generate", model_id=SUMMARY_MODEL_ID):
                summary = _generate(
                    model,
                    tokenizer,
                    device,
                    _plot_messages(transcript, sampled=sampled, language=language),
                    max_new_tokens=PLOT_MAX_NEW_TOKENS,
                )
            return summary, sampled
        finally:
            _unload_text_model(model, tokenizer)

    def _scene_pass(
        self,
        raw_lines: list[Any],
        windows: list[tuple[float, float]],
        timing: TimingRecorder,
        *,
        buckets: dict[str, list[dict[str, Any]]] | None = None,
        scene_ids: list[str] | None = None,
        score_narrative: bool = False,
        select_exchanges: bool = False,
        language: str = "en",
    ) -> list[dict[str, Any]]:
        model, tokenizer, device = _load_text_model(timing, "scene_load")
        scenes: list[dict[str, Any]] = []
        try:
            budget = _input_budget(model, SCENE_MAX_NEW_TOKENS)
            for index, (start, end) in enumerate(windows, start=1):
                if buckets is None or scene_ids is None:
                    window_rows = lines_in_window(raw_lines, start, end)
                    scene_id = None
                else:
                    scene_id = scene_ids[index - 1]
                    window_rows = buckets.get(scene_id, [])
                logger.info(
                    "Scene dialogue %s/%s %s-%s (%s lines)",
                    index,
                    len(windows),
                    _clock(start),
                    _clock(end),
                    len(window_rows),
                )
                exchanges = []
                if not window_rows:
                    description = _empty_description(language)
                    sampled = False
                else:
                    formatted = format_dialogue_lines(window_rows)
                    if select_exchanges:
                        formatted = [
                            f"{line_id}. {' '.join(line.splitlines())}"
                            for line_id, line in enumerate(formatted, start=1)
                        ]

                    messages_for = partial(
                        _scene_messages, start=start, end=end,
                        select_exchanges=select_exchanges, language=language,
                    )

                    transcript, sampled = fit_dialogue_lines(
                        formatted,
                        input_budget=budget,
                        prompt_tokens=lambda body, sampled: _prompt_token_count(
                            tokenizer, messages_for(body, sampled=sampled)
                        ),
                    )
                    with timing.span("scene_generate", model_id=SUMMARY_MODEL_ID, window=index):
                        description = _generate(
                            model,
                            tokenizer,
                            device,
                            messages_for(transcript, sampled=sampled),
                            max_new_tokens=SCENE_MAX_NEW_TOKENS,
                            min_chars=1 if select_exchanges else 40,
                        )
                    if select_exchanges:
                        reply = description
                        try:
                            description, exchanges = resolve_exchanges(reply, window_rows, formatted, transcript)
                        except MissingIntervalSummary:
                            logger.info("Requesting missing dialogue summary for window %s", index)
                            messages = _scene_messages(
                                transcript, start=start, end=end, sampled=sampled, language=language,
                            )
                            with timing.span("scene_summary_retry", model_id=SUMMARY_MODEL_ID, window=index):
                                prose = _generate(
                                    model, tokenizer, device, messages,
                                    max_new_tokens=SCENE_MAX_NEW_TOKENS, min_chars=1,
                                )
                            description, exchanges = resolve_exchanges(
                                f"{prose}\n{reply}", window_rows, formatted, transcript,
                            )
                scenes.append(
                    {
                        "start_time": _clock(start),
                        "end_time": _clock(end),
                        "start_seconds": round(start, 3),
                        "end_seconds": round(end, 3),
                        "dialogue_lines": len(window_rows),
                        "dialogue_sampled": sampled,
                        "description": description,
                        **({"dialogue_exchanges": exchanges} if select_exchanges else {}),
                        **({"scene_id": scene_id} if scene_id else {}),
                        **(
                            _narrative_fields(model, tokenizer, device, window_rows)
                            if score_narrative
                            else {}
                        ),
                    }
                )
        finally:
            _unload_text_model(model, tokenizer)
        return scenes

    def _write_without_audio(
        self,
        video_filename: str,
        video_path: Path,
        spec: Any,
        timing: TimingRecorder,
    ) -> dict[str, Any]:
        """Keep the timeline when the transcript is missing."""
        duration = probe_duration_seconds(video_path)
        intervals = _intervals_for(video_filename, duration, spec.interval_seconds)
        windows = [(item.start, item.end) for item in intervals]
        language = metadata_language(video_filename)
        scenes = _empty_scenes(windows, [item.id for item in intervals], None, [], language=language)
        payload = {
            "experiment_id": spec.experiment_id,
            "interval_seconds": spec.interval_seconds,
            "plot_summary": "",
            "metadata_language": language,
            "plot_sampled": False,
            "scenes": scenes,
            "status": "degraded",
        }
        payload.update(current_stamp())
        output_dir = storage.get_output_dir(self.name)
        output_path = output_dir / f"{video_filename}_dialogue.json"
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        timings_path = output_dir / f"{video_filename}_dialogue_timings.json"
        timing.save(timings_path)
        return {
            "stage": self.name,
            "status": "degraded",
            "scenes": len(scenes),
            "error": "audio transcript missing",
            "output_file": output_path.name,
            "timings_file": timings_path.name,
        }


def _intervals_for(video_filename: str, duration: float, interval_seconds: int) -> list[Interval]:
    document = matching_artifact(
        storage.get_output_dir("timeline") / f"{video_filename}_timeline.json"
    )
    bassl = matching_artifact(storage.get_output_dir("bassl") / f"{video_filename}_bassl.json")
    source = bassl or document
    intervals: list[Interval] = []
    for row in (source or {}).get("intervals") or []:
        if not isinstance(row, dict):
            continue
        try:
            intervals.append(
                Interval(
                    id=str(row["id"]),
                    start=float(row["start"]),
                    end=float(row["end"]),
                    kind=str(row.get("kind") or "window"),
                    source=str(row.get("source") or "fixed"),
                    repaired=bool(row.get("repaired")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return intervals or fixed_windows(duration, interval_seconds)


def _empty_description(language: str) -> str:
    return "Nessun dialogo trascritto in questo intervallo." if language == "it" else _EMPTY_SCENE


def _empty_scenes(
    windows: list[tuple[float, float]],
    scene_ids: list[str] | None,
    buckets: dict[str, list[dict[str, Any]]] | None,
    raw_lines: list[Any],
    *, language: str = "en",
) -> list[dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    for index, (start, end) in enumerate(windows, start=1):
        if buckets is not None and scene_ids is not None:
            rows = buckets.get(scene_ids[index - 1], [])
            scene_id = scene_ids[index - 1]
        else:
            rows = lines_in_window(raw_lines, start, end)
            scene_id = None
        row = {
            "start_time": _clock(start),
            "end_time": _clock(end),
            "start_seconds": round(start, 3),
            "end_seconds": round(end, 3),
            "dialogue_lines": len(rows),
            "dialogue_sampled": False,
            "description": _empty_description(language),
        }
        if scene_id:
            row["scene_id"] = scene_id
        scenes.append(row)
    return scenes


def _narrative_fields(
    model: Any,
    tokenizer: Any,
    device: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask the loaded text model, then fall back to cue words."""
    score, line_ids = cue_from_lines(rows)
    method = "heuristic"
    if rows:
        transcript = "\n".join(
            f"{line.get('id', index)}: {line.get('text', '')}" for index, line in enumerate(rows)
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify the lines. Reply with JSON only: "
                    '{"label": "decision|revelation|reversal|none", "line_ids": []}.'
                ),
            },
            {"role": "user", "content": transcript},
        ]
        try:
            import torch

            prompt_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            model_inputs = tokenizer([prompt_text], return_tensors="pt").to(device)
            with torch.no_grad():
                generated = model.generate(
                    **model_inputs,
                    max_new_tokens=80,
                    do_sample=False,
                )
            new_tokens = generated[0][len(model_inputs.input_ids[0]) :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            parsed = parse_label(text)
            if parsed is not None:
                score, line_ids = parsed
                method = "qwen"
        except Exception as exc:  # noqa: BLE001 - cues remain available
            logger.warning("Narrative label fell back to cues: %s", exc)
    return {
        "narrative_score": score,
        "narrative_line_ids": line_ids,
        "narrative_method": method,
    }
