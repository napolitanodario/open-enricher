"""Propagate the spoken language without changing the existing ASR list format."""

from __future__ import annotations

import json
import os

from open_enricher import storage
from open_enricher.stages.prompt_load import load_prompt


def requested_audio_language() -> str | None:
    """Optional Whisper language for a run with a known audio language."""
    return os.environ.get("OPEN_ENRICHER_AUDIO_LANGUAGE", "").strip().lower() or None


def metadata_language(video: str) -> str:
    """Italian speech gets Italian metadata. Older artifacts default to English."""
    path = storage.get_output_dir("audio") / f"{video}_audio_language.json"
    language = requested_audio_language()
    if path.is_file():
        language = json.loads(path.read_text(encoding="utf-8"))["language"]
    return "it" if language == "it" else "en"


def language_prompt(template: str, language: str) -> str:
    """End the instructions in the target language before inserting the evidence verbatim."""
    prompt = template.replace("{{LANGUAGE}}", "Italian" if language == "it" else "English")
    rule = load_prompt("output_language_it.md" if language == "it" else "output_language_en.md")
    return f"{prompt.rstrip()}\n\n{rule}"
