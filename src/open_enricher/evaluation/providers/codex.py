"""One-shot OpenAI evaluations using the Codex CLI's ChatGPT login."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile


class CodexProvider:
    name = "codex"

    def __init__(self, *, api_key: str | None = None, model: str | None = None) -> None:
        if api_key:
            raise ValueError("The codex provider uses a ChatGPT login. Run codex login.")
        self.model = model or os.getenv("OPEN_ENRICHER_CODEX_MODEL") or "gpt-6.2-sol"

    def complete(self, system: str, user: str) -> str:
        # Keep the grading rubric separate from the untrusted synopsis and metadata.
        # Each exec starts a fresh conversation, outside the repository.
        with tempfile.TemporaryDirectory(prefix="open_enricher_codex_") as tmp:
            rubric = Path(tmp) / "rubric.md"
            rubric.write_text(system + "\nUse only the material in the request. Do not call tools.\n", encoding="utf-8")
            last = Path(tmp) / "answer.txt"
            command = [
                "codex", "exec", "--ephemeral", "--ignore-user-config", "--skip-git-repo-check",
                "--sandbox", "read-only", "--model", self.model, "--json",
                "-c", 'forced_login_method="chatgpt"', "-c", 'web_search="disabled"',
                "-c", 'approval_policy="never"', "-c", 'model_reasoning_effort="low"',
                "-c", 'features={shell_tool=false,unified_exec=false,plugins=false,apps=false,computer_use=false,image_generation=false}',
                "-c", f"model_instructions_file={json.dumps(str(rubric))}",
                "--output-last-message", str(last), "-",
            ]
            env = {key: value for key, value in os.environ.items() if key in {
                "HOME", "PATH", "CODEX_HOME", "LANG", "LC_ALL", "TMPDIR", "XDG_RUNTIME_DIR",
                "SSL_CERT_FILE", "SSL_CERT_DIR", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY",
            }}
            try:
                result = subprocess.run(command, input=user, text=True, encoding="utf-8",
                                        capture_output=True, cwd=tmp, env=env, timeout=900)
            except FileNotFoundError as exc:
                raise RuntimeError("Install the Codex CLI and run codex login to use the codex provider.") from exc
            for line in result.stdout.splitlines():
                event = json.loads(line)
                if event.get("type") in {"item.started", "item.completed", "item.updated"}:
                    item = event.get("item", {}).get("type")
                    if item not in {"agent_message", "reasoning"}:
                        raise RuntimeError(f"Codex used an unexpected item during grading: {item}")
                if event.get("type") == "turn.failed":
                    raise RuntimeError(f"Codex evaluation failed: {event['error'].get('message', 'unknown error')}")
            if result.returncode:
                detail = result.stderr.strip().splitlines()[-1:] or ["no diagnostic"]
                raise RuntimeError(f"Codex exited with status {result.returncode}: {detail[0]}")
            answer = last.read_text(encoding="utf-8").strip() if last.is_file() else ""
            if not answer:
                raise RuntimeError("Codex returned no final answer.")
            return answer
