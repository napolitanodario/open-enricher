import json
from pathlib import Path
import subprocess

import pytest

from open_enricher.evaluation.providers import get_provider
from open_enricher.evaluation.cli import build_parser


def test_default_provider_uses_chatgpt_without_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("CURSOR_API_KEY", raising=False)
    assert get_provider().name == build_parser().parse_args([]).provider == "codex"


def test_chatgpt_rubric_is_separate_and_keys_are_excluded(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "dummy")
    monkeypatch.setenv("OPEN_ENRICHER_CODEX_MODEL", "test-model")

    def run(command, **kwargs):
        assert kwargs["input"] == "user evidence"
        assert "OPENAI_API_KEY" not in kwargs["env"]
        assert 'forced_login_method="chatgpt"' in command
        assert command[command.index("--model") + 1] == "test-model"
        assert (Path(kwargs["cwd"]) / "rubric.md").read_text().startswith("rubric\n")
        Path(command[command.index("--output-last-message") + 1]).write_text('{"score": 4}')
        event = {"type": "item.completed", "item": {"type": "agent_message"}}
        return subprocess.CompletedProcess(command, 0, json.dumps(event), "")

    monkeypatch.setattr(subprocess, "run", run)
    assert json.loads(get_provider("codex").complete("rubric", "user evidence")) == {"score": 4}


def test_tool_use_invalidates_the_grade(monkeypatch):
    event = {"type": "item.started", "item": {"type": "web_search"}}
    monkeypatch.setattr(subprocess, "run", lambda command, **_: subprocess.CompletedProcess(
        command, 0, json.dumps(event), ""))
    with pytest.raises(RuntimeError, match="web_search"):
        get_provider("codex").complete("rubric", "evidence")
    with pytest.raises(ValueError, match="ChatGPT login"):
        get_provider("codex", api_key="dummy")
