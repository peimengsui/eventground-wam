# EventGround-WAM

EventGround-WAM is a side-project study of whether temporally grounded future
representations improve vision-language-action/world-action-model robustness on
LIBERO-Plus.

**Current milestone, October 25, 2026:** [Week 9 is complete](docs/WEEK9_FINAL_RESULTS.md),
and the approved [Week 10 exploratory weak-label study is closed](docs/WEEK10_CLOSEOUT.md).
SFT, bounded RLVR pilots, the matched SFT continuation, fixed report evaluation
and [train/dev diagnostics](docs/WEEK10_DIAGNOSTIC.md) are packaged. The SFT-only
continuation is the provisional event-predictor handoff; additional RLVR benefit
and robot improvement are not established. Independent label QA remains unmet.
[Week 11 Step 1 is frozen](docs/WEEK11_EXPERIMENT.md): checkpoint pins, three
comparison arms, 467 targeted tasks and 200 clean episodes per arm, budgets and
decision criteria. Horizon mapping, implementation and rollouts are pending;
follow the [Week 11 ledger](docs/WEEK11_STATUS.md).
The chronology below retains earlier milestone descriptions.

Weeks 1–4 are closed under their documented acceptance decisions. The best Week 4
B0 is frozen as the shared internal baseline; Week 5
[future-image pair preparation](docs/WEEK5_FUTURE_PAIRS.md) and
[pixel baselines/success criteria](docs/WEEK5_PIXEL_BASELINES.md) and
[pixel predictor implementation](docs/WEEK5_PIXEL_PREDICTOR.md), and
[training recipe/resource/recovery verification](docs/WEEK5_PIXEL_TRAINING.md)
are complete. [Fresh production CPT](docs/WEEK5_PIXEL_PRODUCTION.md) finished and
passed final verification. [Full Step 6 validation](docs/WEEK5_PIXEL_VALIDATION.md)
passed every frozen gate; `C_pixel-v1` is accepted and Week 5 is complete.
[Week 6 is fully closed](docs/WEEK6_CLOSEOUT.md) with completed JEPA training,
validation, audits, comparisons and a separate training-only follow-up. The
original gate remains **FAIL**: front-camera horizon-2 error is 7.22% worse than
copy against a 2% cap. An explicit October 2 project-progression exception
approves that original endpoint as `C_jepa-study-v1` for M2/M4 research use.
Further repair is deferred; the residual pilot is excluded from the main matrix.
[Week 7](docs/WEEK7_STATUS.md) has started: its [M1/M2 experiment](docs/WEEK7_EXPERIMENT.md)
is frozen after rechecking the Week 6 handoff. [Steps 2–4 are complete](docs/WEEK7_AUXILIARY_VERIFICATION.md):
the shared interface, numerical recipes, correctness/resource pilots and exact
process recovery are verified. M1 production training started on Feifei on
October 3 at 10:53 PDT; M2 remains pending. See the [launch record](docs/evaluation/week7-m1-launch-v2/launch.json).
See the [Week 4 closeout](docs/WEEK4_CLOSEOUT.md),
[Week 5 steps](docs/WEEK5_STATUS.md),
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
Week 4 completed training, clean-2000, LIBERO-Plus `dev-210`, failure analysis
and a training-stage audit. The selected [fresh two-epoch all-L1 B0](docs/WEEK4_FRESH_ALL_L1_V2.md)
uses Qwen3.5-2B, the OFT action path, rank-32 LoRA, trajectory-v2 groups and
predict-eight/execute-eight. It scored **93.35% clean / 77.14% dev**, versus the
installed OpenVLA-OFT+ reference's **96.75% / 80.95%**.

[Week 4 closed on September 24](docs/WEEK4_CLOSEOUT.md) with explicit user
acceptance of this baseline for the internal study. The clean statistical gate
remains inconclusive; strict v2 readiness remains unassessed because of one
audited rendering mismatch. Historical failures and original rules are preserved.
Additional B0 optimization is deferred. [Week 5](docs/WEEK5_STATUS.md) has frozen
9,331,359 train/validation future-image pairs under trajectory-v2, with no test
payload reads. Copy/background controls are verified on all validation pairs,
and numerical success criteria are frozen. The 4.63M-parameter action/horizon
predictor passed controlled and real-training small-batch checks with frozen base
Qwen features. The production recipe is frozen at one full pass through all
8,221,365 training pairs in 221,633 updates. The resource/recovery pilot passed,
including bitwise SIGKILL/restart recovery; the conservative training projection
was about 23 hours on Feifei. Production completed on September 26, 2026, at
10:54 PDT in 14.49 active hours, with zero restarts. Final recovery/export weights,
unchanged base weights and the entire one-pass exposure chain are verified.
The fixed final candidate passed full validation on 1,109,994 pairs: 33.27%
overall L1 and 45.04% medium/long-horizon motion L1 improvements over copy,
positive SSIM improvement and all camera/horizon guardrails. Independent
verification and all 480 fixed examples are complete; `C_pixel-v1` is frozen.
Visible long-horizon ghosting remains, and policy improvement is untested.
[Week 6 JEPA](docs/WEEK6_STATUS.md) completed the same one-pass exposure in
20.70 active hours. Full validation improves overall cosine error by 28.61% and
reaches 64.09% Recall@1 with passing collapse checks, but fails the front h=2
guardrail. All 480 fixed examples and appearance diagnostics are reviewed;
the original failed gate is preserved. The subsequent progression decision
approves the unchanged endpoint for the controlled study with that limitation.
See the [results package and downstream handoff](docs/WEEK6_CLOSEOUT.md).
The [accepted Phase 2 protocol](docs/PHASE2_TRAINING_PROTOCOL.md) gives
each B0/M1–M4 policy two total action-training epochs with identical effective
batches and sampling. New policy arms start from pinned base Qwen and fresh
shared components; completed B0 is the control. CPT/event training is separately
budgeted. Feifei is the default compute host, subject to per-model resource pilots.
Production CPT uses fresh predictor weights and the frozen base model; Step 3–4
diagnostic weights are excluded. New action-policy training has not launched.
