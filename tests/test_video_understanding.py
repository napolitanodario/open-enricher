"""Exercise both evaluator entry points without an external API or a GPU."""

import hashlib
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from open_enricher import storage
from open_enricher.evaluation import cli, run
from open_enricher.evaluation.understanding import DIMENSIONS, REPORT_NAME
from open_enricher.experiments import archive

VIDEO = "Demo.mkv"
SYNOPSIS = "Rita opens the gate. Her brother returns."
SCENES = [
    {"start_time": "00:00:00", "end_time": "00:03:00", "description": "Rita unlocks it.",
     "visual_description": "A rusty gate.", "actions": ["She opens the gate."]},
    {"start_time": "00:03:00", "end_time": "00:06:00"},
    {"start_time": "00:06:00", "end_time": "00:08:31", "dialogue_description": "I am home."},
]


def grade():
    return {
        "summary": "The reunion is clear.",
        "dimensions": {name: {"score": i, "note": name} for i, name in enumerate(DIMENSIONS, 2)},
        "findings": [
            {"kind": "supported", "claim": "She opens it.", "synopsis_quote": "Rita opens the gate.",
             "evidence": [{"scene_id": 1, "quote": "opens the gate"}]},
            {"kind": "partial", "claim": "The relation is unclear.", "synopsis_quote": "Her brother returns.",
             "evidence": [{"scene_id": 3, "quote": "I am home."}]},
        ],
    }


@pytest.fixture
def setup_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "RUNS_ROOT", tmp_path / "runs")
    monkeypatch.setenv("OPEN_ENRICHER_DATASET_ROOT", str(tmp_path / "dataset"))
    folder = archive.video_archive_dir("08", VIDEO)
    (folder / f"{VIDEO}_vlm.json").write_text(json.dumps({"scene_timeline": SCENES}))
    (folder / f"{VIDEO}_llm.json").write_text('{"profile": "PROFILE_ONLY"}')
    (tmp_path / f"{VIDEO}_summary.txt").write_text(SYNOPSIS)
    provider = Mock(name="provider")
    provider.name = "cursor"
    provider.model = "custom"
    factory = Mock(return_value=provider)
    monkeypatch.setattr(run, "get_provider", factory)
    return folder, provider, factory


def test_both_tasks_share_provider_and_keep_independent_evidence(setup_archive, tmp_path):
    folder, provider, factory = setup_archive
    provider.complete.side_effect = [json.dumps({"dimensions": {}, "summary": "RecSys"}), json.dumps(grade())]
    row, = run.evaluate_videos([VIDEO], provider_name="cursor", api_key="test-key", model="custom", synopsis_dir=tmp_path,
                               experiment_id="08")
    factory.assert_called_once_with("cursor", api_key="test-key", model="custom")
    recsys_call, understanding_call = provider.complete.call_args_list
    assert "PROFILE_ONLY" in recsys_call.args[1]
    assert SYNOPSIS in recsys_call.args[1]
    assert SCENES[0]["visual_description"] not in recsys_call.args[1]
    user = json.loads(understanding_call.args[1])
    assert set(user) == {"synopsis", "scene_timeline"}
    assert user["synopsis"] == SYNOPSIS
    assert user["scene_timeline"] == [dict(s, scene_id=i) for i, s in enumerate(SCENES, 1)]
    assert row["status"] == "success" and row["video_understanding"]["overall"] == 3.5
    score = json.loads((folder / REPORT_NAME).read_text())
    assert score["method"]["model"] == "custom" and score["scene_count"] == 3
    assert score["findings"][1]["evidence"] == [
        {"scene_id": 3, "start_time": "00:06:00", "end_time": "00:08:31", "quote": "I am home."}
    ]
    assert json.loads((folder.parent / "scores.json").read_text())[VIDEO] == json.loads((folder / "score.json").read_text())


@pytest.mark.parametrize("issue", ["missing reference", "empty reference", "missing timeline",
                                   "empty timeline", "mismatched manifest", "invalid time window"])
def test_unavailable_inputs_do_not_call_provider(setup_archive, tmp_path, issue):
    folder, provider, _ = setup_archive
    timeline = folder / f"{VIDEO}_vlm.json"
    synopsis = tmp_path / f"{VIDEO}_summary.txt"
    if issue == "missing reference":
        synopsis.unlink()
    elif issue == "empty reference":
        synopsis.write_text("  ")
    elif issue == "missing timeline":
        live = storage.get_output_dir("vlm")
        live.mkdir(parents=True, exist_ok=True)
        timeline.rename(live / timeline.name)
    elif issue == "empty timeline":
        timeline.write_text('{"scene_timeline": []}')
    elif issue == "mismatched manifest":
        (folder / f"{VIDEO}_manifest.json").write_text(json.dumps({
            "artifacts": [{"name": timeline.name, "sha256": "another-attempt"}]
        }))
    else:
        timeline.write_text('{"scene_timeline":[{"description":"Missing interval."}]}')
    run.evaluate_videos([VIDEO], task="understanding", synopsis_dir=tmp_path, experiment_id="08")
    provider.complete.assert_not_called()
    report = json.loads((folder / REPORT_NAME).read_text())
    assert report["status"] == "failed" and not report["scored"]
    assert report["error"] and "overall" not in report


@pytest.mark.parametrize("failure", ["provider", "json", "dimension", "scene_id", "scene quote", "synopsis quote", "no reference claims"])
def test_invalid_grade_cannot_hide_behind_valid_recsys(setup_archive, tmp_path, failure):
    folder, provider, _ = setup_archive
    payload = grade()
    finding = payload["findings"][0]
    if failure == "dimension":
        payload["dimensions"]["event_coverage"]["score"] = True
    elif failure == "no reference claims":
        for entry in payload["findings"]:
            entry.update(kind="unverifiable", synopsis_quote="")
    elif failure == "scene_id":
        finding["evidence"][0]["scene_id"] = 0
    elif failure == "scene quote":
        finding["evidence"][0]["quote"] = "I am home."
    elif failure == "synopsis quote":
        finding["synopsis_quote"] = "The gate stayed closed."
    response = RuntimeError("provider failed") if failure == "provider" else json.dumps(payload)
    provider.complete.side_effect = ["{}", "invalid json" if failure == "json" else response]
    row, = run.evaluate_videos([VIDEO], synopsis_dir=tmp_path, experiment_id="08")
    assert row["status"] == "failed" and row["recsys_status"] == "success"
    assert row["video_understanding"]["status"] == "failed"
    assert "overall" not in row["video_understanding"]
    report = json.loads((folder / REPORT_NAME).read_text())
    if failure != "provider":
        assert (folder / report["raw_output_file"]).is_file()


def test_degraded_profile_still_allows_scene_assessment(setup_archive, tmp_path):
    folder, provider, _ = setup_archive
    timeline = folder / f"{VIDEO}_vlm.json"
    (folder / f"{VIDEO}_manifest.json").write_text(json.dumps({
        "profile_kind": "degraded",
        "artifacts": [{"name": timeline.name, "sha256": hashlib.sha256(timeline.read_bytes()).hexdigest()},
                      {"name": f"{VIDEO}_llm.json"}],
    }))
    provider.complete.return_value = json.dumps(grade())
    row, = run.evaluate_videos([VIDEO], experiment_id="08", synopsis_dir=tmp_path)
    assert row["recsys_status"] == "degraded"
    assert row["video_understanding"]["status"] == "success"
    provider.complete.assert_called_once()


@pytest.mark.parametrize("legacy_flag", [[], ["--with-synopsis"]])
def test_cli_can_backfill_only_understanding_or_recsys(setup_archive, tmp_path, legacy_flag):
    folder, provider, _ = setup_archive
    profile = folder / f"{VIDEO}_llm.json"
    profile.unlink()
    for path in [folder / "score.json", folder.parent / "scores.json"]:
        path.write_text("prior result")
    payload = grade()
    payload["findings"] = [
        {"kind": "missing", "claim": "No reunion.", "synopsis_quote": "Her brother returns.", "evidence": []},
        {"kind": "unverifiable", "claim": "The gate is rusty.", "synopsis_quote": "",
         "evidence": [{"scene_id": 1, "quote": "A rusty gate."}]},
    ]
    provider.complete.return_value = json.dumps(payload)
    args = ["--experiment", "08", "--video", VIDEO, "--synopsis-dir", str(tmp_path)]
    cli.main(args + ["--task", "understanding"])
    assert (folder / "score.json").read_text() == (folder.parent / "scores.json").read_text() == "prior result"
    original = (folder / REPORT_NAME).read_bytes()
    assert json.loads(original)["findings"][0]["evidence"] == []
    profile.write_text("{}")
    provider.complete.return_value = "{}"
    cli.main(args + ["--task", "recsys"] + legacy_flag)
    assert SYNOPSIS in provider.complete.call_args.args[1]
    assert provider.complete.call_count == 2
    assert (folder / REPORT_NAME).read_bytes() == original


def test_understanding_uses_live_outputs_without_profile(setup_archive, tmp_path):
    folder, provider, _ = setup_archive
    timeline = storage.get_output_dir("vlm") / f"{VIDEO}_vlm.json"
    timeline.parent.mkdir(parents=True, exist_ok=True)
    (folder / timeline.name).rename(timeline)
    provider.complete.return_value = json.dumps(grade())
    row, = run.evaluate_videos([VIDEO], task="understanding", synopsis_dir=tmp_path)
    assert row["status"] == "success"
    assert Path(row["output_file"]) == storage.outputs_dir() / "evaluation" / f"{VIDEO}_video_understanding.json"
    assert not (folder / REPORT_NAME).exists()


@pytest.mark.parametrize("reference", [None, "  "])
def test_recsys_requires_reference_even_with_legacy_flag_false(setup_archive, tmp_path, reference):
    _, provider, _ = setup_archive
    path = tmp_path / f"{VIDEO}_summary.txt"
    if reference is None:
        path.unlink()
    else:
        path.write_text(reference)
    result, = run.evaluate_videos([VIDEO], task="recsys", with_synopsis=False,
                                  experiment_id="08", synopsis_dir=tmp_path)
    assert result["status"] == "failed" and "synopsis" in result["error"]
    provider.complete.assert_not_called()
