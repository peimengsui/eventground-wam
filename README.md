# EventGround-WAM

EventGround-WAM is a side-project study of whether temporally grounded future
representations improve vision-language-action/world-action-model robustness on
LIBERO-Plus.

Weeks 1–3 are closed, and Week 4 has started: train and evaluate the shared B0
policy, then freeze its checkpoint and the recipe for later experiments.
See the [Week 4 status](docs/WEEK4_STATUS.md),
the [original 12-week execution plan](docs/EventGround-WAM_12-Week_Execution_Plan.md)
and its shorter [repository roadmap](docs/ROADMAP.md).

## Repository layout

```text
src/eventground_wam/  Python package
tests/                Fast checks
scripts/              Local and remote environment helpers
environments/         Locked task-specific uv runtimes
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

The completed foundation includes [LIBERO-Plus setup](docs/LIBERO_PLUS_SETUP.md),
the [locked evaluation protocol](docs/EVALUATION_PROTOCOL.md),
[episode recording](docs/EPISODE_RECORDING.md),
[evaluation splits](docs/EVALUATION_SPLITS.md),
the [frozen dataset](docs/WEEK2_STATUS.md), and the
[implemented and overfit B0 policy](docs/WEEK3_STATUS.md).
Week 4 adds full B0 training, Standard LIBERO clean evaluation, LIBERO-Plus
`dev-210` evaluation, and failure analysis; see the [work ledger](docs/WEEK4_STATUS.md).
The [five-epoch recipe and host-managed recovery](docs/B0_FULL_TRAINING.md) are
implemented. Five-epoch training completed and the selected checkpoint was
verified on 2026-09-16. Clean-40 evaluation completed with **26/40 successes (65%)**
and no episode exceptions/retries. The installed public reference scored
**39/40 (97.5%)** on the identical panel; see the
[verified comparison and matching audit](docs/WEEK4_PUBLIC_REFERENCE.md).
The [larger-comparison acceptance rule](docs/WEEK4_CLEAN_ACCEPTANCE.md) and
2,000-pair manifest are frozen. The [completed larger comparison](docs/WEEK4_CLEAN_COMPARISON_RESULTS.md)
scored **68.25% B0 versus 96.75% OpenVLA** and failed the frozen clean-performance
gate. [Dev-210 and failure analysis](docs/WEEK4_DEV210_FAILURE_ANALYSIS.md)
completed on 2026-09-19: **106/210 (50.48%)**, zero exceptions/retries, and all
210 recordings validated. Failure examples, plots and ranked improvement
hypotheses are available. The clean-performance deficit and Week 4 closure
remain open.
World-model and EventGround components come only after the baseline and evaluation
pipeline are reliable.
