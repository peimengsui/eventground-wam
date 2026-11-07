# Predicted futures for robot policies

This code-only repository contains the EventGround-WAM implementation behind
**Can Predicted Futures Improve Robot Policies?** It compares an action-only
vision-language-action policy with pixel/latent auxiliary training and
pixel/latent coupling, plus a separate event-timing extension.

## Start here

- [Implementation guide](docs/CODE_GUIDE.md): models, training, inference,
  evaluation and figure-generation entry points.
- [Training protocol](docs/PHASE2_TRAINING_PROTOCOL.md): initialization,
  trainable/frozen components and matched action exposure.
- [Research artifacts](docs/RESEARCH_ARTIFACTS.md): optional evidence checks,
  reproduction inputs and external assets.

## What is versioned

| Directory | Contents |
| --- | --- |
| `src/eventground_wam/` | Models, data contracts, training components, inference and evaluation |
| `scripts/` | Training/evaluation entry points, analysis, packaging and verification |
| `configs/` | Training settings, dataset descriptors and evaluation protocols |
| `environments/` | Locked runtimes and pinned analysis dependencies |
| `tests/` | Self-contained CPU tests and explicitly marked research-artifact audits |
| `docs/` | Implementation/protocol notes and blog rendering source |

JSON configuration files are source inputs and remain versioned. Generated
JSON/JSONL reports, CSV tables, dataset/split manifests, model-normalization
exports, figures, videos, source snapshots and reproduction archives are
ignored and are not part of the current checkout. Existing local artifacts
are preserved. Internal plans, weekly logs and result narratives are also kept
locally and ignored. Setup, implementation and evaluation guides remain versioned;
see [the artifact guide](docs/RESEARCH_ARTIFACTS.md#internal-research-notes).

## Run the code tests

Use Python 3.10 for the implementation:

```bash
uv sync --locked --group dev
uv pip install --python .venv/bin/python -r requirements-test.txt
PYTHONPATH=.:src MPLBACKEND=Agg .venv/bin/python -m pytest
```

The extra requirements supply plotting dependencies used by analysis tests.
The default suite uses small CPU models and synthetic fixtures. Tests that
audit retained research outputs are marked `research_artifacts` and skipped
by default. They remain available through an explicit opt-in after restoring
those files; see [the artifact guide](docs/RESEARCH_ARTIFACTS.md).

## Training and reproduction

Model checkpoints, backbone weights, demonstrations, benchmark assets, generated
evaluation splits and original research outputs must be supplied separately.
No dataset or checkpoint download service is provided by this repository.
Historical launch scripts retain host, source-hash and output-directory guards;
a fresh source checkout alone does not reproduce GPU training or robot rollouts.

Statistics and figure-generation code is included. Rebuilding the reported
analysis also requires its separately retained evidence package and the pinned
Python 3.11 dependencies in [environments/analysis/requirements.txt](environments/analysis/requirements.txt).

The current tree is intentionally lean. Earlier Git commits still contain the
previous artifact snapshot; ordinary ignore rules do not remove Git history.
For a small initial checkout, use `git clone --depth 1`.
