"""A short-file archive must not count as the full film."""

from __future__ import annotations

import json
from pathlib import Path

from open_enricher.experiments.archive import (
    append_progress,
    archive_matches_sources,
    retain_previous_run,
)
from open_enricher.fingerprint import source_fingerprint
from open_enricher.runner import restore_cached_stage


def _video(directory: Path, name: str, payload: bytes) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(payload)
    return path


def _manifest(runs: Path, video: str, fingerprint: str) -> None:
    folder = runs / "12-combined" / video
    folder.mkdir(parents=True)
    (folder / f"{video}_manifest.json").write_text(
        json.dumps({"source_fingerprint": fingerprint, "profile_kind": "degraded"}),
        encoding="utf-8",
    )
    (folder / f"{video}_llm.json").write_text('{"kept": true}\n', encoding="utf-8")


def test_short_archive_does_not_cover_the_full_films(
    tmp_path: Path, monkeypatch
) -> None:
    """One 30-second HIMYM row does not settle the three-film request."""
    runs = tmp_path / "runs"
    monkeypatch.setattr("open_enricher.experiments.archive.RUNS_ROOT", runs)
    monkeypatch.setattr(
        "open_enricher.experiments.archive.PROGRESS_PATH",
        runs / "progress.jsonl",
    )
    videos = tmp_path / "video_inputs"
    short = _video(videos, "How_I_Met_Your_Mother.S01E02.mkv", b"short-clip")
    _manifest(runs, short.name, source_fingerprint(short))
    append_progress(
        {
            "experiment_id": "12-combined",
            "step": "archived",
            "status": "ok",
            "results": [{"video": short.name, "status": "degraded"}],
        }
    )
    requested = [
        short.name,
        "Silicon_Valley.S06E01.mkv",
        "The_Kingdom.S01E02.mkv",
    ]
    monkeypatch.setenv("OPEN_ENRICHER_VIDEO_INPUTS", str(videos))
    assert archive_matches_sources("12-combined", requested, "archived", settings="now") is False


def test_same_name_with_a_different_file_is_not_done(
    tmp_path: Path, monkeypatch
) -> None:
    """A manifest from the clip does not match the full file of the same name."""
    runs = tmp_path / "runs"
    monkeypatch.setattr("open_enricher.experiments.archive.RUNS_ROOT", runs)
    monkeypatch.setattr(
        "open_enricher.experiments.archive.PROGRESS_PATH",
        runs / "progress.jsonl",
    )
    videos = tmp_path / "video_inputs"
    short = _video(videos, "How_I_Met_Your_Mother.S01E02.mkv", b"short-clip")
    fingerprint = source_fingerprint(short)
    _manifest(runs, short.name, fingerprint)
    append_progress(
        {
            "experiment_id": "12-combined",
            "step": "archived",
            "status": "ok",
            "results": [{"video": short.name, "status": "degraded"}],
        }
    )
    _video(videos, short.name, b"full-film-bytes")
    monkeypatch.setenv("OPEN_ENRICHER_VIDEO_INPUTS", str(videos))
    assert archive_matches_sources("12-combined", [short.name], "archived", settings="now") is False

    kept = retain_previous_run("12-combined", short.name)
    assert kept is not None
    assert kept.is_dir()
    assert json.loads((kept / f"{short.name}_llm.json").read_text(encoding="utf-8")) == {
        "kept": True
    }
    assert not (runs / "12-combined" / short.name).exists()


def test_audio_cache_from_another_file_is_not_restored(
    tmp_path: Path, monkeypatch
) -> None:
    """A cache without this file's fingerprint stays on disk and is not copied."""
    videos = tmp_path / "video_inputs"
    video = _video(videos, "How_I_Met_Your_Mother.S01E02.mkv", b"full-film")
    cache = tmp_path / "cache" / video.name / "audio"
    cache.mkdir(parents=True)
    (cache / f"{video.name}_audio.json").write_text("[]\n", encoding="utf-8")
    monkeypatch.setenv("OPEN_ENRICHER_DATASET_ROOT", str(tmp_path / "dataset"))
    monkeypatch.setenv("OPEN_ENRICHER_VIDEO_INPUTS", str(videos))
    (tmp_path / "dataset" / "outputs" / "audio_results").mkdir(parents=True)

    assert restore_cached_stage("audio", video.name, tmp_path / "cache") is False
    assert not (
        tmp_path / "dataset" / "outputs" / "audio_results" / f"{video.name}_audio.json"
    ).exists()
    assert (cache / f"{video.name}_audio.json").is_file()


def test_pipeline_cache_validates_source_before_reusing_live_audio(tmp_path, monkeypatch):
    from open_enricher import storage
    from open_enricher.runner import primary_artifact_path, run_pipeline

    monkeypatch.setenv("OPEN_ENRICHER_DATASET_ROOT", str(tmp_path / "data"))
    monkeypatch.delenv("OPEN_ENRICHER_PIPELINE_SPEC", raising=False)
    video = _video(storage.video_inputs(), "episode.mkv", b"short")
    cache = tmp_path / "cache"
    calls = []

    class Audio:
        model = "first-checkpoint"

        def cache_settings(self):
            return {"model": self.model}

        def run(self, name):
            payload = video.read_text()
            calls.append(payload)
            storage.atomic_write_text(primary_artifact_path("audio", name), json.dumps([payload]))
            return {"stage": "audio", "status": "success"}

    def run():
        return run_pipeline(video.name, stages=("audio",), cache_root=cache, stage_registry={"audio": Audio()})

    assert run()["stages"][0]["status"] == "success"
    video.write_bytes(b"longer episode")
    assert run()["stages"][0]["status"] == "success"
    assert json.loads(primary_artifact_path("audio", video.name).read_text()) == ["longer episode"]
    assert run()["stages"][0]["status"] == "skipped_cache"
    assert calls == ["short", "longer episode"]
    Audio.model = "second-checkpoint"
    assert run()["stages"][0]["status"] == "success"
    assert calls[-1] == "longer episode"


def test_resume_requires_same_attempt_settings_and_intact_files(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    video = _video(tmp_path / "inputs", "episode.mkv", b"episode")
    monkeypatch.setenv("OPEN_ENRICHER_VIDEO_INPUTS", str(video.parent))
    monkeypatch.setattr("open_enricher.experiments.archive.RUNS_ROOT", runs)
    monkeypatch.setattr("open_enricher.experiments.archive.PROGRESS_PATH", runs / "progress.jsonl")
    dest = runs / "variant" / video.name
    dest.mkdir(parents=True)
    (dest / "output.json").write_text("{}")
    (dest / "result.json").write_text(json.dumps({
        "archive_id": "a", "settings_hash": "old", "status": "success",
        "source_fingerprint": source_fingerprint(video), "archived_files": ["output.json"],
        "artifact_sha256": {"output.json": source_fingerprint(dest / "output.json")},
    }))
    append_progress({
        "experiment_id": "variant", "step": "archived", "status": "ok",
        "settings_hash": "old", "results": [{"video": video.name, "archive_id": "a"}],
    })
    assert archive_matches_sources("variant", [video.name], "archived", settings="old")
    assert not archive_matches_sources("variant", [video.name], "archived", settings="changed")
    result = json.loads((dest / "result.json").read_text())
    (dest / "result.json").write_text(json.dumps(dict(result, status="degraded")))
    assert not archive_matches_sources("variant", [video.name], "archived", settings="old")
    (dest / "result.json").write_text(json.dumps(result))
    (dest / "output.json").write_text('{"stale": true}')
    assert not archive_matches_sources("variant", [video.name], "archived", settings="old")


def test_prompt_edit_invalidates_same_variant(tmp_path, monkeypatch):
    from open_enricher import fingerprint
    from open_enricher.stages.pipeline_spec import PipelineSpec

    package = tmp_path / "open_enricher"
    prompts = package / "stages" / "prompts"
    prompts.mkdir(parents=True)
    prompt = prompts / "scene.md"
    prompt.write_text("Describe the current frame")
    monkeypatch.setattr(fingerprint, "__file__", str(package / "fingerprint.py"))
    monkeypatch.setattr(fingerprint, "audio_model_identity", lambda: {})
    spec = PipelineSpec("same-id", "scene", 300, None, False, False)
    original = fingerprint.settings_hash(spec)
    prompt.write_text("Describe the current action")
    assert fingerprint.settings_hash(spec) != original


def test_evaluation_resume_requires_both_scores_and_current_synopsis(tmp_path, monkeypatch):
    from open_enricher.evaluation import synopsis
    from open_enricher.evaluation.understanding import REPORT_NAME

    runs = tmp_path / "runs"
    video = _video(tmp_path / "inputs", "episode.mkv", b"episode")
    monkeypatch.setenv("OPEN_ENRICHER_VIDEO_INPUTS", str(video.parent))
    monkeypatch.setattr("open_enricher.experiments.archive.RUNS_ROOT", runs)
    monkeypatch.setattr("open_enricher.experiments.archive.PROGRESS_PATH", runs / "progress.jsonl")
    monkeypatch.setattr(synopsis, "DEFAULT_SYNOPSIS_DIR", tmp_path)
    folder = runs / "variant" / video.name
    folder.mkdir(parents=True)
    timeline = folder / f"{video.name}_vlm.json"
    timeline.write_text("{}")
    ref = synopsis.synopsis_path(video.name)
    ref.write_text("A reference.")
    recsys = folder / "score.json"
    recsys.write_text("{}")
    understanding = folder / REPORT_NAME
    (folder / "result.json").write_text(json.dumps({
        "archive_id": "a", "settings_hash": "s", "status": "success",
        "source_fingerprint": source_fingerprint(video), "archived_files": [timeline.name],
        "artifact_sha256": {timeline.name: source_fingerprint(timeline)},
    }))
    receipt = {"video": video.name, "archive_id": "a", "score_sha256": source_fingerprint(recsys)}
    event = {"experiment_id": "variant", "step": "evaluation", "status": "ok", "settings_hash": "s",
             "results": [receipt]}

    def done():
        return archive_matches_sources("variant", [video.name], "evaluation", settings="s")

    append_progress(event)
    assert not done()
    understanding.write_text(json.dumps({"synopsis_sha256": source_fingerprint(ref)}))
    receipt["understanding_score_sha256"] = source_fingerprint(understanding)
    event["provider"] = "cursor"
    append_progress(event)
    assert done()
    assert not archive_matches_sources("variant", [video.name], "evaluation", settings="s", provider="codex")
    event["provider"] = "codex"
    append_progress(event)
    assert archive_matches_sources("variant", [video.name], "evaluation", settings="s", provider="codex")
    ref.write_text("Another cut.")
    assert not done()
    ref.write_text("A reference.")
    understanding.write_text("{}")
    assert not done()
