"""VLM memory keeps only the last n scene records when the spec says so."""

from __future__ import annotations

from open_enricher.stages.vlm_core.segments import (
    _scene_prompt,
    records_for_vlm_memory,
    select_prior_scenes,
)


def _scenes(count: int) -> list[dict[str, str]]:
    return [{"start_time": f"{index:02d}"} for index in range(count)]


def test_null_limit_keeps_every_prior_scene() -> None:
    prior = _scenes(6)
    assert select_prior_scenes(prior, None) == prior


def test_limit_two_keeps_the_latest_two() -> None:
    prior = _scenes(6)
    kept = select_prior_scenes(prior, 2)
    assert [item["start_time"] for item in kept] == ["04", "05"]


def test_limit_four_keeps_the_latest_four() -> None:
    prior = _scenes(6)
    kept = select_prior_scenes(prior, 4)
    assert [item["start_time"] for item in kept] == ["02", "03", "04", "05"]


def test_short_history_is_not_padded() -> None:
    prior = _scenes(2)
    assert select_prior_scenes(prior, 4) == prior


def test_visual_only_memory_drops_dialogue() -> None:
    prior = [{"description": "a room", "dialogue_description": "they argue about the patent"}]
    kept = records_for_vlm_memory(prior, visual_only=True)
    assert kept == [{"description": "a room"}]
    assert "patent" not in _scene_prompt(
        {"start_time": "00:00:00", "end_time": "00:05:00", "description": "they argue about the patent"},
        memory="",
        visual_only=True,
    )


def test_extract_prompt_omits_summary_and_memory() -> None:
    """The extract pass sees neither the dialogue summary nor prior scenes."""
    text = _scene_prompt(
        {
            "start_time": "00:00:00",
            "end_time": "00:05:00",
            "description": "they argue about the patent",
        },
        memory="earlier scene about the judge",
        visual_only=False,
        extract_visual=True,
    )
    assert "patent" not in text
    assert "judge" not in text
    assert "actions" in text


def test_truncated_visual_json_is_regenerated_on_the_same_clip(tmp_path, monkeypatch):
    import pytest
    from dataclasses import replace
    from open_enricher.stages.vlm_core import segments
    from open_enricher.stages.vlm_core.config import build_configs

    generated = []
    def generate(model, processor, clip, cfg):
        generated.append((clip, cfg.prompt))
        return next(replies)

    monkeypatch.setattr(segments, "clip_video_range", lambda source, start, duration, dest: dest)
    monkeypatch.setattr(segments, "extract_metadata", generate)
    cfg = replace(build_configs()[1], prompt="Observe the current frames.")
    incomplete = '{"description": "Closing credits", "on_screen_text": ["VISUAL'
    complete = '{"description": "Closing credits", "on_screen_text": ["VISUAL EFFECTS"]}'
    replies = iter([incomplete, complete])
    args = (tmp_path / "film.mkv", 0.0, 240.0, None, None, cfg)
    assert segments._clip_generation(*args, require_visual_json=True) == complete
    assert generated[0][0] == generated[1][0]
    assert generated[0][1] != generated[1][1]
    replies = iter([incomplete, incomplete, "White lettering scrolls over black.", "white letters\nblack screen",
                    "scrolling upward", "stationary shot", "black and white", "NONE"])
    import json
    record = json.loads(segments._clip_generation(*args, require_visual_json=True))
    assert record["description"] == "White lettering scrolls over black."
    assert record["subjects_and_objects"] == ["white letters", "black screen"]
    assert record["on_screen_text"] == []
    assert len({clip for clip, _ in generated[-8:]}) == 1
    replies = iter([incomplete, incomplete, "NONE"])
    with pytest.raises(ValueError, match="Missing visual description"):
        segments._clip_generation(*args, require_visual_json=True)
