# External evaluation

The experiment loop evaluates two outputs for each video. Both tasks use the same selected provider and model. Each task starts a fresh conversation and uses the supplied text and a nonempty reference synopsis. The experiment loop and standalone evaluator default to Codex with an OpenAI model and a ChatGPT login.

| Task | Input sent to the judge | Archived report |
| --- | --- | --- |
| RecSys profile quality | Title, final `*_llm.json` and the reference synopsis | `score.json` |
| Video understanding | Reference synopsis and all ordered windows from `*_vlm.json` | `video_understanding_score.json` |

## Video understanding

Supply an independent reference at `synopsis/<exact video filename>_summary.txt` or use `--synopsis-dir`. Use the synopsis for the correct cut or episode. Keep references for titles awaiting their video under `synopsis/pending/`. Once the video arrives, match its exact filename, including its container extension. A synopsis that only describes the premise can only establish fidelity and coverage for that part of the story.

Both judges write their notes, summaries and findings in Italian when the generated
descriptions are consistently Italian. They use English for English, mixed or
indeterminate prose. Evidence quotations retain the exact wording and language
of the source synopsis or scene. Controlled labels and character names do not
determine the language of the review.

The [video understanding system prompt](../src/open_enricher/evaluation/prompts/video_understanding.md) holds the grading instructions. The user prompt is a JSON object containing the synopsis and the ordered scenes for that video. The Cursor provider joins both parts into one message for `Agent.prompt`.

The judge receives narration, local dialogue descriptions, visual descriptions, actions, subjects, on-screen text, camera and lighting when available. It sees the complete text timeline in one request, including sparse windows. It receives no title, final RecSys profile, raw video, or previous scores. The assessment measures the story recoverable from the generated text. Checking what actually happened on screen requires separate audiovisual review.

| Dimension | What earns credit |
| --- | --- |
| Event coverage | The timeline captures the principal events, turning points and outcome named in the reference. |
| Factual fidelity | The timeline assigns actions to the correct characters and describes their outcomes accurately. |
| Chronology and causality | The viewer can follow changes, cause and effect, and transitions between storylines. |
| Scene grounding | Concrete local actions or dialogue locate pivotal events in particular windows. |

Each dimension has a score from 1 to 5 and an explanation. The host computes the unweighted mean (`overall`). Rubric `video_understanding_v2` asks for one to three examples that explain the scores for the whole timeline. The report records findings as `supported`, `partial`, `contradicted`, `missing` or `unverifiable`. For example, when the timeline mentions a subplot absent from the synopsis, the judge can mark it unverifiable. Checkable findings quote the synopsis. For claims present in the timeline the host verifies that each cited excerpt occurs in its scene and attaches the original start and end timestamps. An omitted event has no invented timestamp.

The report includes the number of windows, the timeline SHA-256, the SHA-256 of the reference text after loading, the model and the rubric version. A missing reference, empty timeline, provider error (including a context limit), or invalid response produces a failed report without a numeric grade. Malformed model responses are saved alongside the report as `*.raw.txt`. The full timeline can still be evaluated if the final profile is absent or degraded.

## RecSys profile quality

The existing judge uses five dimensions on a 1–5 scale. Their weights are completeness 0.15, specificity 0.25, fidelity 0.35, consistency 0.10 and recommendation usefulness 0.15. The host reports the weighted mean and a normalized 0–1 score. The profile is usable when the mean is at least 3.0 and fidelity is at least 2. The judge assesses factual fidelity against the supplied synopsis. If the synopsis is missing or empty, the evaluation reports an input error before contacting the provider.

## Running the evaluator

For the default Codex provider, install the Codex CLI and sign in with your ChatGPT account. On a headless machine, run `codex login --device-auth`. Check the active login with `codex login status`. Evaluation uses the subscription under its Codex usage limits. The default model is `gpt-6.2-sol`. Use `--model` or `OPEN_ENRICHER_CODEX_MODEL` to choose another model available to that login. Each call runs in an ephemeral directory, sends the rubric as model instructions, and passes the evidence on standard input. The host rejects turns that invoke a tool. [Codex authentication](https://developers.openai.com/codex/auth/) · [Non-interactive Codex](https://developers.openai.com/codex/noninteractive/)

To select Grok, pass `--provider cursor`, install the evaluation extra with `pip install -e '.[evaluation]'` and set `CURSOR_API_KEY`. The default model is `grok-4.7`. Use `--model` or `OPEN_ENRICHER_EVAL_MODEL` to override it.

```bash
# Score both outputs with the default Codex provider and the reference synopsis.
python -m open_enricher.evaluation --video suits_s01e02.mkv

# Select Grok through Cursor explicitly.
python -m open_enricher.evaluation --provider cursor --video suits_s01e02.mkv

# Score only scene understanding using an existing experiment archive.
archive=local-archive/2026-11-24/worktrees/open-enricher
OPEN_ENRICHER_RUNS_ROOT="$archive/.old_runs/local2" python -m open_enricher.evaluation \
  --experiment 08-narrate-visual --video Suits.S01E02.mkv --task understanding \
  --synopsis-dir "$archive/synopsis"

# Run only the original RecSys assessment.
python -m open_enricher.evaluation --video suits_s01e02.mkv --task recsys
```

With `--experiment`, reports are written under `runs/local/<experiment>/<video>/`. Set `OPEN_ENRICHER_RUNS_ROOT` to the parent of an experiment directory to evaluate an older archive in place. Selecting only understanding leaves the RecSys report and its experiment aggregate `scores.json` untouched. For live outputs, the files are `dataset/outputs/evaluation/<video>_evaluation.json` and `<video>_video_understanding.json`. The experiment summary shows both scores independently. [Prompt templates](../src/open_enricher/evaluation/prompts/)
