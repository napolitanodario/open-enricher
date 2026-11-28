# Open-Enricher

Open-Enricher processes a film or TV episode into timestamped dialogue, visual
scene descriptions and a recommender profile. The combined pipeline adds
BaSSL boundaries, face and active-speaker evidence, and visual-memory retrieval.
The `*_vlm.json` file contains the scene timeline. The `*_llm.json` file
contains the final profile.

## Install

Create a Python environment on the worker. Install compatible PyTorch and
torchvision builds for its CUDA version and make FFmpeg and ffprobe available.
Then install the pipeline and optional encoder runtimes.

```bash
python -m venv .venv
source .venv/bin/activate
# Install the worker's PyTorch and torchvision builds in this environment.
pip install -r requirements.txt
pip install -e .
python -m open_enricher.adapters.upstream
```

Keep input videos in `dataset/video_inputs/` and reusable weights under
`dataset/models_cache/`. The encoder setup guide lists the external weights
and inference implementations. Set `HF_TOKEN` for gated models. The default
evaluator uses the Codex CLI with a ChatGPT login on the machine running the
evaluation. Use `codex login status` to check the login. The optional Cursor
provider uses `CURSOR_API_KEY`. [Encoder setup](docs/encoders.md) · [Evaluator](docs/evaluator.md)

## Run

```bash
python -m open_enricher --list
python -m open_enricher --only 12-combined --video suits_s01e02.mkv
sbatch run_job.sh --only 12-combined --video suits_s01e02.mkv
```

Submit from the checkout root. A variant selects its models and budgets.
Use `09-bassl-uniform` for scene boundaries, `10-character-graph` for face
tracks and TalkNet, or `11-visual-memory` for EVA/BLIP-2 retrieval.
`01-scene-context` remains available as a historical control. By default the
Python CLI iterates the packaged variants and the Slurm launcher selects
`12-combined`. [Local experiment guide](docs/local_loop.md)

## Outputs

| File suffix in the episode directory | Contents |
| --- | --- |
| `_audio.json` | Time-aligned speech with available speaker labels |
| `_timeline.json`, `_bassl.json` | Source timeline and scene boundaries when enabled |
| `_dialogue.json` | Transcript-based episode summary and interval descriptions |
| `_graph.json`, `_memory.json` | Character and speaker evidence, visual memory and retrievals when enabled |
| `_frame_plan.json`, `_vlm_frames.json` | Planned and recorded frame inputs |
| `_vlm.json` | Timestamped scene descriptions and structured observations |
| `_llm.json` | Recommender profile |
| `_manifest.json` | Stage outcomes and evidence provenance for the combined pipeline |

Working stage files default to `dataset/outputs/`. The loop publishes results
under `runs/local/<experiment_id>/<video>/`. Use separate
`OPEN_ENRICHER_OUTPUT_ROOT` and `OPEN_ENRICHER_RUNS_ROOT` paths to keep two
attempts apart while reusing the same videos and weights. Inspect the manifest
for degraded or skipped stages before interpreting an experiment.

## Historical results on the cluster

The pre-cleanup runs, scripts, experiment notes and outputs are indexed in
`local-archive/2026-11-24/README.md` in the primary checkout. Files retain the
original worktree and run names. This local archive is excluded from Git and
from active inference caches. [Copy results from the cluster](docs/sync_outputs.md)
