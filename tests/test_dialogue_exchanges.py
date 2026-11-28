"""Exercise ASR selection, evidence handoff, language and the visual-only prompt."""

import json
from types import SimpleNamespace

import pytest

from open_enricher import storage
from open_enricher.experiments.config import VARIANTS_DIR
from open_enricher.language import metadata_language
from open_enricher.runner import collect_artifacts
from open_enricher.stages import dialogue, vlm
from open_enricher.stages.dialogue_exchanges import resolve_exchanges
from open_enricher.stages.llm import LLMStage
from open_enricher.stages.llm_core.prompt import (
    profile_repair_user_prompt,
    profile_system_prompt,
    profile_user_prompt,
)
from open_enricher.stages.pipeline_spec import load_pipeline_spec
from open_enricher.stages.vlm_core import segments
from open_enricher.timings import TimingRecorder


def test_resolver_only_accepts_visible_adjacent_rows_and_preserves_rank():
    rows = [
        {"start_timestamp": i + 0.12, "end_timestamp": i + 0.56, "text": str(i)}
        for i in range(6)
    ]
    numbered = [f"{i}. [{row['text']}]" for i, row in enumerate(rows, 1)]
    shown = "\n".join(numbered[:3] + numbered[4:])
    reply = "A proposal.\nEXCHANGES: [[4],[1,3],[true],[5,6],[6],[1,2],[3]]"
    summary, exchanges = resolve_exchanges(reply, rows, numbered, shown)
    assert resolve_exchanges("EXCHANGES: [[1]]\nA proposal.", rows, numbered, shown)[0] == "A proposal."
    assert summary == "A proposal."
    assert [[row["line_id"] for row in ex] for ex in exchanges] == [[5, 6], [1, 2]]
    assert exchanges[0][0] == dict(rows[4], line_id=5, speaker="UNKNOWN")
    for footer in ("", "\nEXCHANGES: broken", "\nEXCHANGES: []"):
        assert resolve_exchanges(summary + footer, rows, numbered, shown) == (summary, [])


def test_italian_exchange_variant_from_asr_to_narration(tmp_path, monkeypatch):
    monkeypatch.setenv("OPEN_ENRICHER_DATASET_ROOT", str(tmp_path))
    monkeypatch.delenv("OPEN_ENRICHER_AUDIO_LANGUAGE", raising=False)
    monkeypatch.delenv("OPEN_ENRICHER_PIPELINE_SPEC", raising=False)
    baseline = load_pipeline_spec(VARIANTS_DIR / "08-narrate-visual.json")
    assert (baseline.interval_seconds, baseline.narration_context_scenes, baseline.select_dialogue_exchanges) == (300, 2, False)
    monkeypatch.setenv("OPEN_ENRICHER_PIPELINE_SPEC", str(VARIANTS_DIR / "13-narrate-exchanges.json"))
    spec = load_pipeline_spec()
    video = "Doppiaggio.[IT].avi"
    storage.setup_dataset()
    (storage.video_inputs() / video).touch()
    audio_dir = storage.get_output_dir("audio")
    assert metadata_language(video) == "en"
    (audio_dir / f"{video}_audio_language.json").write_text('{"language":"it"}')
    rows = [
        {"start_timestamp": 240 * index + offset, "speaker": "SPEAKER_04", "text": f"Battuta {index}.{offset}"}
        for index in (0, 1, 2, 3, 4) for offset in (1, 2)
    ]
    (audio_dir / f"{video}_audio.json").write_text(json.dumps(rows))
    assert len(collect_artifacts(video, ["audio"])) == 2
    monkeypatch.setattr(dialogue, "probe_duration_seconds", lambda _: 1240)
    dummy = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=8192))
    monkeypatch.setattr(dialogue, "_load_text_model", lambda *args: (dummy, None, "cpu"))
    monkeypatch.setattr(dialogue, "_unload_text_model", lambda *args: None)
    monkeypatch.setattr(dialogue, "_prompt_token_count", lambda *args: 100)
    selectors = []

    def generate(model, tokenizer, device, messages, **kwargs):
        assert all("interamente in italiano" in message["content"] for message in messages)
        if "EXCHANGES:" in messages[0]["content"]:
            selectors.append(messages[1]["content"])
            if len(selectors) == 2:
                return "EXCHANGES: [[2],[1]]"
            return f"Riassunto {len(selectors)}.\nEXCHANGES: [[2],[1]]"
        if "Summarize only the dialogue" in messages[1]["content"]:
            assert selectors[-1] == messages[1]["content"]
            return "Riassunto 2."
        return "Trama complessiva."

    monkeypatch.setattr(dialogue, "_generate", generate)
    dialogue.DialogueStage().run(video)
    assert len(selectors) == 5  # One exchange selection per speaking window.
    assert "2. [00:00:02] SPEAKER_04: Battuta 0.2" in selectors[0]

    def generate_visual(video_path, start, end, model, processor, cfg, **kwargs):
        assert cfg.max_frames == 40
        assert "interamente in italiano" in cfg.prompt
        assert "Battuta" not in cfg.prompt and "Riassunto" not in cfg.prompt
        return '{"description":"Una scrivania.","actions":["Si alza."]}'

    monkeypatch.setattr(segments, "load_model", lambda cfg: (None, None))
    monkeypatch.setattr(segments, "unload_model", lambda *args: None)
    monkeypatch.setattr(segments, "_clip_generation", generate_visual)
    timeline = vlm.VLMStage().run(video)["result"]["scene_timeline"]
    assert timeline[0]["end_time"] == "00:04:00"
    assert timeline[-1]["dialogue_exchanges"] == []
    assert timeline[-1]["dialogue_description"].startswith("Nessun dialogo")
    narrator = LLMStage()
    prompts = []

    def narrate(model, tokenizer, messages, **kwargs):
        assert all("interamente in italiano" in message["content"] for message in messages)
        prompts.append(messages[1]["content"])
        return "Descrizione della scena."

    monkeypatch.setattr(narrator, "_generate_text", narrate)
    scenes = narrator._narrate_scenes(None, None, timeline, TimingRecorder("llm"), spec, "it")
    narrator._write_narrated_timeline(video, scenes, spec.narration_context_scenes)
    assert "Battuta 0." not in prompts[4] and "Riassunto 1." not in prompts[4]
    for index in (1, 2, 3):
        assert f"Battuta {index}.2" in prompts[4] and f"Battuta {index}.1" not in prompts[4]
    assert "Battuta 4.1" in prompts[4] and "Battuta 4.2" in prompts[4]
    payload = json.loads((storage.get_output_dir("vlm") / f"{video}_vlm.json").read_text())
    assert payload["narration_context_scenes"] == 3
    assert payload["scene_timeline"][0]["visual_description"] == "Una scrivania."


@pytest.mark.parametrize("language,rule", [("it", "interamente in italiano"), ("en", "entirely in English")])
def test_language_after_the_evidence_in_summaries_and_profile(language, rule):
    evidence = "SPEAKER_01: Keep the literal {{LANGUAGE}} as spoken."
    for messages in (dialogue._plot_messages(evidence, sampled=True, language=language),
                     dialogue._scene_messages(evidence, start=0, end=12, sampled=False, language=language)):
        assert rule in messages[0]["content"]
        user = messages[1]["content"]
        assert user.index(evidence) < user.index(rule)
    example = {"recsys_metadata": {"primary_genre": "Comedy", "pacing": "Moderate"}}
    profile = profile_user_prompt(script_text=evidence, example_structure=example, language=language)
    assert profile.index(evidence) < profile.index('"Moderate"') < profile.index(rule)
    assert rule in profile_system_prompt(language)
    assert rule in profile_repair_user_prompt(error="test", language=language)
