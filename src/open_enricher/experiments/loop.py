"""Sequential local experiment loop.

For each pipeline variant JSON, point the process at that spec, run the
in-process enricher on the configured videos, archive artifacts under
``runs/local/<experiment_id>/``, score the profiles and scene timelines with the default
evaluator, and continue to the next variant. No deploy, HTTP client, Colab,
Kaggle, or Google Drive is involved.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any

from open_enricher.evaluation.providers import DEFAULT_PROVIDER

from open_enricher.experiments.archive import (
    append_progress,
    archive_matches_sources,
    experiment_dir,
    list_variants,
    merge_timings,
    retain_previous_run,
    video_archive_dir,
    write_json,
)
from open_enricher.timings import (
    inference_device_document,
    start_device_vram_series,
    write_performance_csv,
    write_vram_series_csv,
)
from open_enricher.experiments.config import (
    CACHE_ROOT,
    PIPELINE_STAGES,
    RUNS_ROOT,
    VARIANTS_DIR,
)
from open_enricher.experiments.stages import STAGE_CHOICES
from open_enricher import storage
from open_enricher.logging_setup import configure_logging, get_logger

logger = get_logger(__name__)


def _runner():
    """
    Lazy-import runner helpers that pull in heavy stage dependencies.

    Returns:
        The ``open_enricher.runner`` module.
    """
    from open_enricher import runner as runner_module

    return runner_module


def _variant_id(path: Path) -> str:
    """
    Read the experiment id from a variant file, falling back to the stem.

    Args:
        path: Path to a variant JSON document.

    Returns:
        Non-empty experiment id string.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return str(data.get("experiment_id") or path.stem)


def _load_variant(path: Path) -> dict[str, Any]:
    """
    Load a variant JSON object and drop private underscore keys.

    Args:
        path: Path to a variant JSON document.

    Returns:
        Dict suitable for ``load_pipeline_spec`` and for archiving.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Variant must be a JSON object: {path}")
    return {key: value for key, value in data.items() if not str(key).startswith("_")}


def _write_stub_experiment_md(
    exp_dir: Path,
    experiment_id: str,
    variant: dict[str, Any],
    video_results: list[dict[str, Any]],
    evaluation_results: list[dict[str, Any]] | None = None,
    evaluation_error: str | None = None,
) -> Path:
    """
    Write a short EXPERIMENT.md stub summarizing the local run.

    Args:
        exp_dir: Archive directory for this experiment.
        experiment_id: Stable experiment id.
        variant: Cleaned variant dict.
        video_results: Per-video pipeline summaries.
        evaluation_results: Per-video evaluator status dicts.
        evaluation_error: Error from the evaluator when it could not start.

    Returns:
        Path to the written markdown file.
    """
    lines = [
        f"# Experiment {experiment_id}",
        "",
        "## Spec",
        "",
        f"- Title: `{variant.get('title')}`",
        f"- Interval seconds: `{variant.get('interval_seconds')}`",
        f"- VLM memory scenes: `{variant.get('vlm_memory_scenes')}`",
        f"- Visual only: `{variant.get('visual_only')}`",
        f"- Narrate from visual: `{variant.get('narrate_from_visual')}`",
        "",
        "## Execution",
        "",
    ]
    for result in video_results:
        status = result.get("status")
        err = result.get("error")
        lines.append(
            f"- `{result.get('video')}`: **{status}**"
            + (f" - {err}" if err else "")
        )
    lines.extend(["", "## Quality", ""])
    if evaluation_error:
        lines.append(f"Evaluation failed to start: {evaluation_error}")
    elif evaluation_results:
        for row in evaluation_results:
            lines.append(f"- `{row['video']}`")
            for label, assessment in (("RecSys", row), ("Video understanding", row["video_understanding"])):
                status = assessment.get("recsys_status", assessment["status"])
                overall = assessment.get("overall")
                score = f" overall `{overall}`" if overall is not None else ""
                error = assessment.get("error")
                lines.append(f"  - {label}: **{status}**{score}" + (f" - {error}" if error else ""))
    else:
        lines.append("No evaluation results.")
    lines.append("")
    path = exp_dir / "EXPERIMENT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _collect_scores(exp_dir: Path, videos: list[str]) -> dict[str, Any]:
    """
    Load archived evaluator scores for each video under an experiment folder.

    Args:
        exp_dir: Archive directory for this experiment.
        videos: Video filenames that were processed.

    Returns:
        Mapping of video filename to score document when present.
    """
    scores: dict[str, Any] = {}
    for video in videos:
        path = exp_dir / video / "score.json"
        if path.is_file():
            scores[video] = json.loads(path.read_text(encoding="utf-8"))
    return scores


def _activate_variant(variant_path: Path) -> None:
    """
    Point the process at one variant JSON for the shared dataset pipeline.

    Args:
        variant_path: Absolute path to the variant JSON file.

    Side effects:
        Sets ``OPEN_ENRICHER_PIPELINE_SPEC``. The dataset layout is shared
        across experiments; only the archived copy under ``runs/local/`` is
        per experiment.
    """
    os.environ["OPEN_ENRICHER_PIPELINE_SPEC"] = str(variant_path.resolve())


def _evaluate_experiment(
    experiment_id: str,
    videos: list[str],
) -> list[dict[str, Any]]:
    """
    Score archived profiles and scene timelines with the default provider.

    Fidelity is judged against ``synopsis/<video>_summary.txt``. A missing
    synopsis fails that video; there is no fallback to model memory.

    Args:
        experiment_id: Stable experiment id under ``runs/local/``.
        videos: Video filenames whose archived outputs should be scored.

    Returns:
        Per-video statuses and separate RecSys and video understanding reports.
    """
    from open_enricher.evaluation.run import evaluate_videos

    logger.info(
        "Evaluating experiment %s with provider=%s synopsis=yes",
        experiment_id,
        DEFAULT_PROVIDER,
    )
    return evaluate_videos(
        videos,
        provider_name=DEFAULT_PROVIDER,
        experiment_id=experiment_id,
        with_synopsis=True,
    )


def run_one_experiment(
    variant_path: Path,
    *,
    videos: list[str],
    stages: tuple[str, ...] = PIPELINE_STAGES,
    use_cache: bool = True,
    skip_done: bool = True,
    defer_evaluation: bool = False,
) -> dict[str, Any]:
    """
    Run the full local pipeline for one variant, then score it with the default provider.

    Args:
        variant_path: Path to the variant JSON file.
        videos: Video filenames under the dataset ``video_inputs``.
        stages: Ordered stage names to execute.
        use_cache: Reuse shared audio cache when available.
        skip_done: Skip pipeline and evaluation steps already marked done.

    Returns:
        Summary dict with experiment_id, per-video results, evaluation
        results, and archive path.
    """
    experiment_id = _variant_id(variant_path)
    variant = _load_variant(variant_path)
    logger.info("======== Experiment %s ========", experiment_id)

    from open_enricher.fingerprint import settings_hash, source_fingerprint
    from open_enricher.stages.pipeline_spec import load_pipeline_spec

    _activate_variant(variant_path)
    settings = settings_hash(load_pipeline_spec(), stages=stages)
    pipeline_done = skip_done and archive_matches_sources(
        experiment_id, videos, "archived", settings=settings
    )
    evaluation_done = pipeline_done and archive_matches_sources(
        experiment_id, videos, "evaluation", settings=settings, provider=DEFAULT_PROVIDER
    )
    if pipeline_done and evaluation_done:
        logger.info(f"Skipping {experiment_id}: already archived and evaluated")
        return {
            "experiment_id": experiment_id,
            "status": "skipped_done",
            "results": [],
            "evaluation": [],
            "runs_dir": str(experiment_dir(experiment_id)),
        }

    video_results: list[dict[str, Any]] = []
    if pipeline_done:
        logger.info(f"Skipping pipeline for {experiment_id}: already archived")
        exp_dir = experiment_dir(experiment_id)
        video_results = [
            json.loads((exp_dir / video / "result.json").read_text(encoding="utf-8"))
            for video in videos
        ]
    else:
        runner = _runner()
        registry = runner.build_stages()
        try:
            write_json(
                experiment_dir(experiment_id) / "performance" / "device.json",
                inference_device_document(),
            )
        except Exception as exc:  # noqa: BLE001 - metrics must not stop the run
            logger.warning("Inference device snapshot skipped: %s", exc)

        for video in videos:
            logger.info(f"Pipeline for video {video}")
            vram_series = None
            try:
                vram_series = start_device_vram_series()
            except Exception as exc:  # noqa: BLE001 - metrics must not stop the run
                logger.warning("Device VRAM series skipped: %s", exc)
            try:
                summary = runner.run_pipeline(
                    video,
                    stages=stages,
                    stage_registry=registry,
                    cache_root=CACHE_ROOT if use_cache else None,
                    use_cache=use_cache,
                )
            except Exception as exc:  # noqa: BLE001 - keep the loop moving
                summary = {
                    "video": video,
                    "status": "error",
                    "error": str(exc),
                    "stages": [],
                }
            finally:
                vram_rows: list[dict[str, Any]] = []
                if vram_series is not None:
                    try:
                        vram_rows = vram_series.stop()
                    except Exception as exc:  # noqa: BLE001 - metrics must not stop the run
                        logger.warning("Device VRAM series stop skipped: %s", exc)
                write_vram_series_csv(
                    experiment_dir(experiment_id) / "performance" / f"{video}_vram.csv",
                    vram_rows,
                )

            source = storage.video_inputs() / video
            retained = retain_previous_run(experiment_id, video)
            if retained is not None:
                logger.info("Kept the previous archive at %s", retained)
            dest = video_archive_dir(experiment_id, video)
            archive_stages = summary.get("stages_ran") or ()
            archived = runner.archive_artifacts(video, dest, stages=archive_stages)
            summary["archive_id"] = uuid.uuid4().hex
            summary["source_fingerprint"] = source_fingerprint(source) if source.is_file() else ""
            summary["settings_hash"] = settings

            timing_paths = [dest / name for name in archived if name.endswith("_timings.json")]
            timing_doc = merge_timings(timing_paths)
            write_json(dest / "timings.json", timing_doc)
            archived.append("timings.json")
            summary["archived_files"] = archived
            summary["artifact_sha256"] = {name: source_fingerprint(dest / name) for name in archived}
            write_performance_csv(
                experiment_dir(experiment_id) / "performance" / f"{video}.csv",
                timing_doc,
                experiment=experiment_id,
                video=video,
            )
            write_json(dest / "result.json", summary)
            video_results.append(summary)

            append_progress(
                {
                    "experiment_id": experiment_id,
                    "video": video,
                    "step": "pipeline",
                    "status": "ok" if summary.get("status") == "success" else "error",
                    "detail": summary.get("status"),
                }
            )

    exp_dir = experiment_dir(experiment_id)
    if not pipeline_done:
        write_json(exp_dir / "video_results.json", video_results)
        write_json(exp_dir / "variant.json", variant)
        pipeline_ok = all(
            row.get("status") in {"success", "degraded"} for row in video_results
        )
        append_progress(
            {
                "experiment_id": experiment_id,
                "step": "archived",
                "settings_hash": settings,
                "status": "ok" if pipeline_ok else "error",
                "results": [
                    {"video": row.get("video"), "status": row.get("status"), "archive_id": row["archive_id"]}
                    for row in video_results
                ],
            }
        )

    if defer_evaluation:
        _write_stub_experiment_md(exp_dir, experiment_id, variant, video_results)
        return {
            "experiment_id": experiment_id,
            "status": "metadata_complete" if all(
                row.get("status") in {"success", "degraded"} for row in video_results
            ) else "error",
            "results": video_results,
            "evaluation": [],
            "runs_dir": str(exp_dir),
        }

    evaluation_results: list[dict[str, Any]] = []
    evaluation_error: str | None = None
    if evaluation_done:
        logger.info(f"Skipping evaluation for {experiment_id}: already evaluated")
    else:
        try:
            evaluation_results = _evaluate_experiment(experiment_id, videos)
        except Exception as exc:  # noqa: BLE001 - keep the loop moving
            evaluation_error = str(exc)
            logger.exception("Evaluation failed for %s", experiment_id)
        write_json(exp_dir / "scores.json", _collect_scores(exp_dir, videos))
        eval_ok = (
            evaluation_error is None
            and bool(evaluation_results)
            and all(row.get("status") in {"success", "degraded"} for row in evaluation_results)
        )
        archive_ids = {row["video"]: row.get("archive_id") for row in video_results}

        def report_hash(path: Path) -> str | None:
            return source_fingerprint(path) if path.is_file() else None

        append_progress(
            {
                "experiment_id": experiment_id,
                "step": "evaluation",
                "settings_hash": settings,
                "status": "ok" if eval_ok else "error",
                "provider": DEFAULT_PROVIDER,
                "uses_synopsis": True,
                "detail": evaluation_error,
                "results": [
                    {
                        "video": row.get("video"),
                        "status": row.get("status"),
                        "overall": row.get("overall"),
                        "archive_id": archive_ids.get(row.get("video")),
                        "score_sha256": report_hash(exp_dir / row["video"] / "score.json"),
                        "understanding_score_sha256": report_hash(
                            Path(row["video_understanding"]["output_file"])
                        ),
                        "understanding_overall": row["video_understanding"].get("overall"),
                    }
                    for row in evaluation_results
                ],
            }
        )

    _write_stub_experiment_md(
        exp_dir,
        experiment_id,
        variant,
        video_results,
        evaluation_results,
        evaluation_error,
    )

    pipeline_ok = (
        all(row.get("status") in {"success", "degraded"} for row in video_results)
        if video_results
        else pipeline_done
    )
    eval_ok = evaluation_done or (
        evaluation_error is None
        and bool(evaluation_results)
        and all(row.get("status") in {"success", "degraded"} for row in evaluation_results)
    )
    overall = "ok" if pipeline_ok and eval_ok else "error"

    return {
        "experiment_id": experiment_id,
        "status": overall,
        "results": video_results,
        "evaluation": evaluation_results,
        "runs_dir": str(exp_dir),
    }


def run_loop(
    *,
    only: list[str] | None = None,
    from_id: str | None = None,
    videos: list[str] | None = None,
    stages: tuple[str, ...] = PIPELINE_STAGES,
    use_cache: bool = True,
    skip_done: bool = True,
    defer_evaluation: bool = False,
) -> list[dict[str, Any]]:
    """
    Run selected variants in sequence on the local GPU host.

    Args:
        only: Optional list of experiment ids or variant stems to run.
        from_id: When set without ``only``, start at this id and continue.
        videos: Optional explicit video filenames. Defaults to every file in
            ``video_inputs``.
        stages: Ordered stage names for each run.
        use_cache: Reuse shared audio cache when available.
        skip_done: Skip experiments already marked archived in progress.jsonl.

    Returns:
        List of per-experiment summary dicts. Also writes
        ``runs/local/loop_summary.json``.
    """
    runner = _runner()
    runner.bootstrap_runtime(tee_log=False)
    selected_videos = runner.list_videos(videos)

    variants = list_variants(VARIANTS_DIR)
    if only:
        wanted = set(only)
        variants = [
            path
            for path in variants
            if path.stem in wanted or _variant_id(path) in wanted
        ]
    elif from_id:
        start = None
        for index, path in enumerate(variants):
            if path.stem == from_id or _variant_id(path) == from_id:
                start = index
                break
        if start is None:
            raise ValueError(f"Unknown --from experiment id: {from_id}")
        variants = variants[start:]

    if not variants:
        raise ValueError("No variants selected to run")

    logger.info(
        "Videos: %s | variants: %s",
        ", ".join(selected_videos),
        len(variants),
    )

    summaries: list[dict[str, Any]] = []
    for path in variants:
        summaries.append(
            run_one_experiment(
                path,
                videos=selected_videos,
                stages=stages,
                use_cache=use_cache,
                skip_done=skip_done,
                defer_evaluation=defer_evaluation,
            )
        )

    write_json(RUNS_ROOT / "loop_summary.json", summaries)
    return summaries


def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for the local experiment loop.

    Returns:
        Configured ``ArgumentParser`` instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run Open-Enricher experiment variants locally in sequence. "
            "Each variant sets OPEN_ENRICHER_PIPELINE_SPEC and runs the "
            "in-process pipeline on the videos under the local dataset."
        )
    )
    parser.add_argument(
        "--only",
        nargs="+",
        help="Run only these experiment ids or variant stems (e.g. 01-scene-context).",
    )
    parser.add_argument(
        "--from",
        dest="from_id",
        help="Start at this experiment id and continue through later variants.",
    )
    parser.add_argument(
        "--video",
        nargs="+",
        dest="videos",
        help="Video filenames under video_inputs. Default: every file there.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGE_CHOICES),
        help="Subset of stages to run. Default: audio dialogue vlm llm.",
    )
    parser.add_argument(
        "--defer-evaluation",
        action="store_true",
        help="Archive metadata now and leave both external evaluations for a separate job.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not restore or update the shared audio cache.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run experiments even when progress.jsonl already marks them done.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List known variants and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """
    CLI entrypoint for ``python -m open_enricher`` and the experiments module.

    Args:
        argv: Optional argument list for tests. Defaults to ``sys.argv[1:]``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list:
        # Plain stdout for scripting (not CSV log lines).
        for path in list_variants(VARIANTS_DIR):
            print(_variant_id(path), path.name)
        return

    configure_logging()
    stages = tuple(args.stages) if args.stages else PIPELINE_STAGES
    run_loop(
        only=args.only,
        from_id=args.from_id,
        videos=args.videos,
        stages=stages,
        use_cache=not args.no_cache,
        skip_done=not args.force,
        defer_evaluation=args.defer_evaluation,
    )


if __name__ == "__main__":
    main()
