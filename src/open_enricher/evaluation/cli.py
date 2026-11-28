"""CLI for the external metadata evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from open_enricher.evaluation.providers import DEFAULT_PROVIDER, PROVIDERS
from open_enricher.evaluation.run import evaluate_videos
from open_enricher.evaluation.synopsis import DEFAULT_SYNOPSIS_DIR
from open_enricher.logging_setup import configure_logging, get_logger
from open_enricher import storage

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """
    Build the CLI parser for ``python -m open_enricher.evaluation``.

    Returns:
        Configured ``ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Score RecSys profiles and scene-level video understanding with an external evaluator "
            "(Codex with a ChatGPT login by default)."
        )
    )
    parser.add_argument(
        "--video",
        nargs="+",
        dest="videos",
        help="Video filenames to score. Default: every file in video_inputs.",
    )
    parser.add_argument(
        "--experiment",
        help=(
            "Experiment id under runs/local/. Reads archived profiles and timelines and "
            "writes the selected reports there."
        ),
    )
    parser.add_argument(
        "--task", choices=("both", "recsys", "understanding"), default="both",
        help="Tasks to score (default: both). Both tasks require the reference synopsis.",
    )
    parser.add_argument(
        "--with-synopsis", action="store_true", help=argparse.SUPPRESS,  # Legacy CLI spelling.
    )
    parser.add_argument(
        "--synopsis-dir",
        type=Path,
        default=None,
        help=f"Synopsis directory (default: {DEFAULT_SYNOPSIS_DIR}).",
    )
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=sorted(PROVIDERS),
        help=f"Evaluator provider (default: {DEFAULT_PROVIDER}).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Provider model id override (default: provider-specific / env).",
    )
    return parser


def _resolve_videos(names: list[str] | None) -> list[str]:
    """Resolve video filenames from CLI args or the dataset."""
    inputs = storage.video_inputs()
    if names:
        return list(names)
    if not inputs.is_dir():
        raise FileNotFoundError(f"Video input directory does not exist: {inputs}")
    found = sorted(
        path.name
        for path in inputs.iterdir()
        if path.is_file() and not path.name.startswith(".")
    )
    if not found:
        raise FileNotFoundError(f"No videos found under {inputs}")
    return found


def main(argv: list[str] | None = None) -> None:
    """
    Entrypoint for ``python -m open_enricher.evaluation``.

    Args:
        argv: Optional argument list for tests.
    """
    env_file = Path(__file__).resolve().parents[3] / ".env"
    if env_file.is_file():
        load_dotenv(env_file)

    configure_logging()
    parser = build_parser()
    args = parser.parse_args(argv)
    videos = _resolve_videos(args.videos)

    results = evaluate_videos(
        videos,
        provider_name=args.provider,
        synopsis_dir=args.synopsis_dir,
        experiment_id=args.experiment,
        model=args.model,
        task=args.task,
    )

    print(json.dumps(results, indent=2, ensure_ascii=False))
    if any(row.get("status") != "success" for row in results):
        raise SystemExit(1)
