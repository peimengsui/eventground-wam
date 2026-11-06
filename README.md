# Predicted futures for robot policies

This repository contains the EventGround-WAM research implementation behind
**Can Predicted Futures Improve Robot Policies?** It compares an action-only
vision-language-action policy with pixel/latent auxiliary training and
pixel/latent coupling, plus a separate event-timing extension.

The main study evaluated five policies on 1,000 perturbed cases and 2,000 clean
episodes. Pixel coupling had the highest observed perturbed success (76.8%,
versus 74.1% for the baseline); pixel auxiliary training reached 76.1%.
Adaptive event timing reduced success in its separate evaluation.
See the [results](docs/evaluation/week12-results-v1/RESULTS.md) for paired
intervals, measurement details, and qualifications.

## Start here

- [Implementation and entry points](docs/CODE_GUIDE.md): models, training,
  inference, evaluation, and figure generation.
- [Blog draft](docs/blog/eventground-wam/site/_drafts/building-a-robot-policy-that-looks-ahead.md):
  readable explanation with figures and recorded examples; not a published post.
- [Reproduce the analysis](docs/evaluation/week12-repro-v1/GUIDE.md): verified
  local inputs and commands for rebuilding the reported statistics and figures.
- [Training protocol](docs/PHASE2_TRAINING_PROTOCOL.md) and
  [experiment history](HISTORY.md): design decisions and retained research records.

## Repository layout

| Directory | Contents |
| --- | --- |
| `src/eventground_wam/` | Models, data contracts, training components, inference and evaluation |
| `scripts/` | Training/evaluation entry points, analysis, packaging and verification |
| `configs/` | Frozen recipes, model references, evaluation protocols and service definitions |
| `environments/` | Separate locked environments for robot training, simulation and data |
| `tests/` | CPU tests and retained-fixture checks |
| `docs/evaluation/week12-repro-v1/` | Portable analysis bundle and dependency/input manifests |
| `docs/blog/eventground-wam/` | Article source, figures, videos and renderers |

## Run the tests

Use Python 3.10 for the implementation. The root environment is separate from
both the GPU experiment runtimes and the historical Python 3.11 analysis runtime.

```bash
uv sync --locked --group dev
uv pip install --python .venv/bin/python -r requirements-test.txt
PYTHONPATH=.:src MPLBACKEND=Agg .venv/bin/python -m pytest
```

The extra requirements supply plotting dependencies used by analysis tests.
The tests exercise small CPU models and retained fixtures; they do not launch
training jobs, download model weights, or run new robot evaluations.

## Rebuild the reported analysis

The committed archive preserves the exact analysis inputs and source snapshot
used for the report. From the repository root, using Python 3.11:

```bash
python3.11 scripts/verify_week12_reproducibility.py \
  --package docs/evaluation/week12-repro-v1 \
  --extract-to /tmp/eventground-analysis-source
python3.11 /tmp/eventground-analysis-source/scripts/reproduce_week12_analysis.py \
  --output /tmp/eventground-analysis-results
```

Both destinations must be new. Install the pinned
[analysis dependencies](docs/evaluation/week12-repro-v1/analysis-requirements.txt)
first; the [guide](docs/evaluation/week12-repro-v1/GUIDE.md) gives the environment
setup and explains exact numerical/image comparisons. The archive is about 71 MB
and contains retained analysis inputs, not model weights or a training dataset.

## Training, inference and external assets

The source release includes the implementation, configs, tests, environment
locks and selected evidence/fixtures. Model checkpoints, backbone weights,
demonstrations, benchmark assets and full raw recordings remain external;
their identities and original locations are recorded in the
[external-asset manifest](docs/evaluation/week12-repro-v1/external-assets.json)
and [checkpoint manifest](docs/evaluation/week12-repro-v1/checkpoints.json).
They are not public download endpoints supplied by this repository.

Historical launch scripts include host, source-hash and output-directory guards.
Use the [code guide](docs/CODE_GUIDE.md) and the matching run/environment records;
a fresh clone alone does not reproduce GPU training or robot rollouts.
Historical source archives identify the original runs even where the current
implementation has evolved. Chronology documents may reference additional
local/remote research artifacts that are outside this curated source release.
