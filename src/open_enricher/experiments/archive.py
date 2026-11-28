"""Local run archive under runs/local and progress resume helpers.

The experiment loop writes per-variant folders here, appends progress
events for resume, and merges per-stage timing documents after a run.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from open_enricher.experiments.config import PROGRESS_PATH, RUNS_ROOT


def experiment_dir(experiment_id: str) -> Path:
    """
    Return (and create) the archive folder for one experiment id.

    Args:
        experiment_id: Stable id from the variant JSON ``experiment_id`` field.

    Returns:
        Path ``runs/local/<experiment_id>/``.
    """
    path = RUNS_ROOT / experiment_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def video_archive_dir(experiment_id: str, video_filename: str) -> Path:
    """
    Return (and create) the archive folder for one video inside an experiment.

    Args:
        experiment_id: Stable experiment id.
        video_filename: Exact video file name (used as a subfolder name).

    Returns:
        Path ``runs/local/<experiment_id>/<video_filename>/``.
    """
    path = experiment_dir(experiment_id) / video_filename
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, document: Any) -> Path:
    """
    Serialize a document as indented UTF-8 JSON.

    Args:
        path: Destination file.
        document: JSON-serializable value.

    Returns:
        The same ``path`` for chaining.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def append_progress(event: dict[str, Any]) -> None:
    """
    Append one progress event to ``runs/local/progress.jsonl``.

    Args:
        event: Dict describing experiment_id, step, status, and optional
            details. A UTC ``ts`` field is added when missing.
    """
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = dict(event)
    row.setdefault("ts", datetime.now(timezone.utc).isoformat())
    with PROGRESS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_progress() -> list[dict[str, Any]]:
    """
    Load all progress events from the JSONL resume log.

    Returns:
        List of parsed event dicts. Malformed lines are skipped.
    """
    if not PROGRESS_PATH.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in PROGRESS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def archive_matches_sources(
    experiment_id: str, videos: list[str], step: str, *, settings: str, provider: str | None = None
) -> bool:
    """Require the latest step, archived attempt, source bytes and settings to agree."""
    from open_enricher import storage
    from open_enricher.evaluation.synopsis import load_synopsis
    from open_enricher.evaluation.understanding import REPORT_NAME
    from open_enricher.fingerprint import source_fingerprint
    from open_enricher.manifest import load_json

    row = next(
        (event for event in reversed(load_progress())
         if event.get("experiment_id") == experiment_id and event.get("step") == step),
        {},
    )
    if not videos or row.get("status") != "ok" or row.get("settings_hash") != settings:
        return False
    if provider is not None and row.get("provider") != provider:
        return False
    receipts = {item.get("video"): item for item in row.get("results", []) if isinstance(item, dict)}
    for video in videos:
        folder = RUNS_ROOT / experiment_id / video
        result = load_json(folder / "result.json") or {}
        receipt = receipts.get(video, {})
        source = storage.video_inputs() / video
        if result.get("status") != "success" or result.get("settings_hash") != settings:
            return False
        if not receipt.get("archive_id") or receipt["archive_id"] != result.get("archive_id"):
            return False
        if not source.is_file() or result.get("source_fingerprint") != source_fingerprint(source):
            return False
        if step == "evaluation":
            score = folder / "score.json"
            if not score.is_file() or receipt.get("score_sha256") != source_fingerprint(score):
                return False
            understanding = folder / REPORT_NAME
            if (
                not understanding.is_file()
                or receipt.get("understanding_score_sha256") != source_fingerprint(understanding)
            ):
                return False
            try:
                synopsis = load_synopsis(video)
            except (OSError, ValueError):
                return False
            report = load_json(understanding) or {}
            if report.get("synopsis_sha256") != hashlib.sha256(synopsis.encode("utf-8")).hexdigest():
                return False
        checksums = result.get("artifact_sha256")
        if not isinstance(checksums, dict) or not checksums or set(checksums) != set(result.get("archived_files", [])):
            return False
        for name, expected in checksums.items():
            artifact = folder / name
            if not artifact.is_file() or source_fingerprint(artifact) != expected:
                return False
    return True


def retain_previous_run(experiment_id: str, video: str) -> Path | None:
    """Keep the entire previous attempt, including legacy results without a manifest."""
    folder = RUNS_ROOT / experiment_id / video
    if not folder.is_dir():
        return None
    kept = folder.parent / "_previous" / f"{video}.{uuid4().hex}"
    kept.parent.mkdir(exist_ok=True)
    folder.rename(kept)
    return kept


def list_variants(variants_dir: Path) -> list[Path]:
    """
    List variant JSON files in sorted order.

    Args:
        variants_dir: Directory containing ``*.json`` variant specs.

    Returns:
        Sorted list of variant paths.
    """
    return sorted(variants_dir.glob("*.json"))


def merge_timings(paths: Iterable[Path]) -> dict[str, Any]:
    """
    Merge per-stage timing JSON files into one document.

    Args:
        paths: Candidate timing file paths. Missing or invalid files are skipped.

    Returns:
        Dict with a ``stages`` list of timing documents.
    """
    documents = []
    for path in paths:
        if path.is_file():
            try:
                documents.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
    return {"stages": documents}
