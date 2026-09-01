# EventGround-WAM

EventGround-WAM is a side-project study of whether temporally grounded future
representations improve vision-language-action/world-action-model robustness on
LIBERO-Plus.

The repository is intentionally small at this stage. The immediate goal is to
establish a reproducible environment and evaluation protocol before adding model
code. See the [original 12-week execution plan](docs/EventGround-WAM_12-Week_Execution_Plan.md)
and its shorter [repository roadmap](docs/ROADMAP.md).

## Repository layout

```text
src/eventground_wam/  Python package
tests/                Fast checks
scripts/              Local and remote environment helpers
docs/                 Project roadmap and decisions
```

Large datasets, checkpoints, caches, rollout videos, and experiment outputs are
ignored by Git. They should live on the compute machine or in dedicated artifact
storage rather than in the source repository.

## Environment setup

The project uses `uv` and pins Python 3.10 for compatibility across the modern
model stack and the older LIBERO-Plus simulator dependencies.

```bash
uv sync
uv run pytest
```

On a CUDA compute host, validate PyTorch and the GPU with:

```bash
uv run python scripts/check_compute.py --require-cuda
```

## Compute host

The intended GPU host is available through the SSH alias `feifei`. Run the
read-only environment check from the repository root:

```bash
./scripts/remote_doctor.sh
```

The planned working directory on that machine is `~/code/eventground-wam`. No
remote directory is created by the check.

## Current scope

The first milestone is deliberately limited to [LIBERO-Plus setup](docs/LIBERO_PLUS_SETUP.md), episode
recording, reproducible evaluation splits, and an official-checkpoint sanity
check. World-model and EventGround components come only after the baseline and
evaluation pipeline are reliable.
