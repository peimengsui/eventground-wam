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

## Local setup

Python 3.10 or newer is expected.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
pytest
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

The first milestone is deliberately limited to LIBERO-Plus setup, episode
recording, reproducible evaluation splits, and an official-checkpoint sanity
check. World-model and EventGround components come only after the baseline and
evaluation pipeline are reliable.
