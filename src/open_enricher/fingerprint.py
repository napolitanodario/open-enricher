"""Source content and inference settings recorded for each run."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from open_enricher.adapters.upstream import LAVIS_REV, TALKNET_REV
from open_enricher.language import requested_audio_language
from open_enricher.stages.pipeline_spec import PipelineSpec

ATTEMPT_ID = "OPEN_ENRICHER_ATTEMPT_ID"
SOURCE_FINGERPRINT = "OPEN_ENRICHER_SOURCE_FINGERPRINT"
SETTINGS_HASH = "OPEN_ENRICHER_SETTINGS_HASH"


def source_fingerprint(path: Path) -> str:
    """SHA-256 of the complete source. Reuse the digest while the inode is unchanged."""
    path = path.resolve()
    stat = path.stat()
    return _file_digest(path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=1024)
def _file_digest(path: Path, *identity: int) -> str:
    from open_enricher.manifest import file_sha256

    return file_sha256(path)


@lru_cache(maxsize=1)
def _speech_package_versions() -> dict[str, str | None]:
    versions = {}
    for package in ("whisperx", "faster-whisper", "pyannote.audio", "torch", "torchaudio"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    return versions


def _hub_revisions(tags: tuple[str, ...]) -> dict[str, str]:
    from open_enricher import storage

    hf_home = Path(os.environ.get("HF_HOME") or storage.models_cache() / "huggingface")
    hub = Path(os.environ.get("HF_HUB_CACHE") or hf_home / "hub")
    return {
        str(ref.relative_to(hub)): ref.read_text().strip()
        for ref in hub.glob("models--*/refs/*")
        if ref.is_file() and any(tag in ref.parent.parent.name.lower() for tag in tags)
    }


def audio_model_identity() -> dict[str, Any]:
    """Record the installed speech stack and downloaded ASR/alignment revisions."""
    from open_enricher import storage

    torch_home = Path(os.environ.get("TORCH_HOME") or storage.models_cache() / "torch")
    revisions = _hub_revisions(("whisper", "wav2vec", "wavlm", "pyannote"))
    aligners = {
        path.name: source_fingerprint(path)
        for path in (torch_home / "hub" / "checkpoints").glob("*")
        if path.is_file()
    }
    return {"versions": _speech_package_versions(), "hub_revisions": revisions, "torch_checkpoints": aligners}


def settings_hash(spec: PipelineSpec, *, stages: tuple[str, ...] | None = None) -> str:
    """Hash the knobs that change experimental artifacts."""
    payload = {
        "experiment_id": spec.experiment_id,
        "stages": stages,
        "audio_models": audio_model_identity(),
        "audio_language": requested_audio_language(),
        "language_models": _hub_revisions(("qwen--qwen3",)),
        "interval_seconds": spec.interval_seconds,
        "vlm_memory_scenes": spec.vlm_memory_scenes,
        "visual_only": spec.visual_only,
        "narrate_from_visual": spec.narrate_from_visual,
        "narration_context_scenes": spec.narration_context_scenes,
        "select_dialogue_exchanges": spec.select_dialogue_exchanges,
        "narrate_full_visual_without_exchanges": spec.narrate_full_visual_without_exchanges,
        "frame_allocation": spec.frame_allocation,
        "features": asdict(spec.features),
        "budgets": asdict(spec.budgets),
        "checkpoints": asdict(spec.checkpoints),
    }
    from open_enricher.adapters.bassl import checkpoint_dir
    from open_enricher.adapters.face_speech import insightface_root, talknet_weights
    from open_enricher.adapters.visual_encode import eva_path, qformer_path

    weights: list[Path] = []
    if spec.features.bassl:
        directory = checkpoint_dir(spec)
        weights.extend((directory / "configuration.json", directory / "pytorch_model.pt"))
    if spec.features.character_graph:
        weights.append(talknet_weights(spec))
        weights.extend((insightface_root(spec) / "models" / "buffalo_l").glob("*.onnx"))
        payload["talknet_revision"] = TALKNET_REV
    if spec.features.visual_memory:
        weights.extend((eva_path(spec), qformer_path(spec)))
        payload["lavis_revision"] = LAVIS_REV
    payload["model_weights"] = {
        str(path): source_fingerprint(path) if path.is_file() else None for path in weights
    }
    # Include the inference implementation and the prompts actually distributed
    # with it. This works in editable installs and unpacked wheels, without Git.
    package = Path(__file__).resolve().parent
    dependencies = [
        path for path in package.rglob("*.py") if "evaluation" not in path.relative_to(package).parts
    ]
    dependencies.extend((package / "stages" / "prompts").glob("*.md"))
    dependencies.extend((package / "stages" / "vlm_core").glob("*.json"))
    payload["implementation"] = {
        str(path.relative_to(package)): source_fingerprint(path) for path in sorted(dependencies)
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def current_stamp() -> dict[str, str]:
    """Return the attempt identifiers for this process, possibly empty."""
    return {
        "attempt_id": os.environ.get(ATTEMPT_ID, "").strip(),
        "source_fingerprint": os.environ.get(SOURCE_FINGERPRINT, "").strip(),
        "settings_hash": os.environ.get(SETTINGS_HASH, "").strip(),
    }
