# EventGround-WAM

EventGround-WAM is a side-project study of whether temporally grounded future
representations improve vision-language-action/world-action-model robustness on
LIBERO-Plus.

**October 31, 2026: [Week 11 is closed](docs/WEEK11_CLOSEOUT.md), including its
authorized investigations and follow-ups. Week 12 is ready but not started.**
No Week 11 candidate establishes a reliable improvement over original M4.
Retain M4 as this phase's reference; continuation and adapter endpoints remain
research artifacts. The remaining adapter episodes and optional four-case
inspection are closed/deferred, not pending requirements. The
[closure package](docs/evaluation/week11-final-closeout-v1/README.md) records
the evidence inventory, preserved checkpoints and inactive experiment jobs.
The history below retains the individual results and earlier decisions.

**Current milestone, October 31, 2026:** [Week 9 is complete](docs/WEEK9_FINAL_RESULTS.md),
and the approved [Week 10 exploratory weak-label study is closed](docs/WEEK10_CLOSEOUT.md).
SFT, bounded RLVR pilots, the matched SFT continuation, fixed report evaluation
and [train/dev diagnostics](docs/WEEK10_DIAGNOSTIC.md) are packaged. The SFT-only
continuation is the provisional event-predictor handoff; additional RLVR benefit
and robot improvement are not established. Independent label QA remains unmet.
[Week 11 Step 1 is frozen](docs/WEEK11_EXPERIMENT.md): checkpoint pins, three
comparison arms, 467 targeted tasks and 200 clean episodes per arm, budgets and
decision criteria. [Step 2 mapping is frozen](docs/WEEK11_HORIZON_MAPPING.md):
nearest supported horizon, ties shorter, valid `none` → 8, invalid outputs fail,
and no confidence gate. [Steps 3–4 are complete](docs/evaluation/week11-integration-v1/RESULTS.md):
isolated M4-EG integration, exact GPU parity/reload checks, and all 24 pilot episodes.
Successes were 8/8 fixed, 8/8 event-running fixed8 and 6/8 adaptive; this small pilot
does not establish benefit. All input audits and resource caps passed.
[Week 11 formal analysis is complete](docs/evaluation/week11-closeout-v1/RESULTS.md).
All 2,001 episodes and independent audits passed. Adaptive minus fixed is
−6.85 pp on targeted (95% CI −10.28 to −3.43) and −5.00 pp on clean
(−10.00 to 0.00), failing the original performance gates. Both fixed controls
match all 667 binary outcomes and all 16,500 identical-input action queries;
small rendering differences qualify full-trajectory equality. Latency and
resources pass. M4-fixed remains the baseline.
The [recording-only adaptive investigation](docs/evaluation/week11-adaptive-diagnosis-v1/RESULTS.md)
traces action divergence to horizon changes and finds limited closing-forecast
consistency. Forecast error and coupling sensitivity remain separate unresolved
contributors; no model calls or new rollouts were added.
The subsequent [matched-observation replay](docs/evaluation/week11-horizon-replay-v1/RESULTS.md)
is complete: 130 inputs across five horizons reproduce every recorded action
exactly at its original horizon. Non-8 horizons change arm commands and can shift
gripper timing near closing. Automatic physical candidates identify 18 initial
departures with no new event within 32 frames under either recorded continuation,
despite positive forecasts. Most interaction windows remain ambiguous; the
diagnostic supports forecast disagreement and control sensitivity, without
causally assigning the performance loss. It adds 780 action-model calls and no rollouts.
The subsequent [event-window review](docs/evaluation/week11-event-review-v1/RESULTS.md)
checks all 18 early negative pairs and six pre-close gripper changes. Early
forecasts are premature even before applying strict grasp rules; fixture
contacts expose a separate label-scope limitation. The six closing shifts are
50 ms each, split equally between earlier and later, and occur in wins and losses.
This review uses existing evidence only and does not certify independent label QA.
The requested [closing-gated comparison](docs/evaluation/week11-gate-comparison-v1/RESULTS.md)
is complete: 81 fresh rollouts on the same 27 diagnostic cases give 15 fixed8,
16 original-adaptive and 16 gated successes. Delaying adaptive routing until
after executed closing recovers four adaptive failures and loses four adaptive
successes, with no net gain. All 214 pre-activation queries match fixed8 exactly.
One historical adaptive outcome changes after a one-level wrist-image difference;
these selected cases support no baseline promotion. The formal result is unchanged.
The [inspection of three remaining gated failures](docs/evaluation/week11-gated-failure-inspection-v1/RESULTS.md)
finds failed acquisition/recovery in targeted 1 and 356, and successful lifting
followed by off-center placement in clean 6. Clean 6 never reaches the goal's
30 mm bowl–plate center tolerance, despite matching fixed8 gripper commands
through the common prefix. Closing alone does not protect acquisition or later
placement. This inspection adds no model calls or rollouts.
The [first-chunk intervention](docs/evaluation/week11-first-chunk-v1/RESULTS.md)
completes 12 fresh episodes: fixed8 succeeds on all three cases, gated on none,
and each narrow intervention on two. A single h4 chunk induces targeted 356's
failure; targeted 1 recovers when either the first change or later adaptation
is removed; clean 6 fails only in the two adaptive continuations. All six controls
reproduce exactly and intervention inputs match. These case-specific causal
results support no general routing remedy or baseline change.
The [matched multi-horizon continuation pilot](docs/WEEK11_MULTIHORIZON_PILOT.md)
is complete: two 2,000-update arms share the accepted M4 endpoint and
58,172 training presentations, differing only in candidate-horizon exposure.
Input checks verify all 1,197 required training files and matching action/core
weights. A six-condition, 27-case screen is frozen for the resulting endpoints.
[Training, evaluation and independent audits passed](docs/evaluation/week11-multihorizon-results-v1/RESULTS.md).
The 162-episode screen gives original fixed/adaptive 15/15 successes,
H8 continuation 19/17, and mixed continuation 17/19, each out of 27.
Mixed adaptive gains two over H8 adaptive but mixed fixed8 loses two against
H8 fixed8, failing the frozen preservation criterion. Mixed adaptive also
ties H8 fixed8 at 19/27. The selected single-seed pilot supports no baseline
promotion or larger evaluation under its rule.
The [saved regression/gain inspection](docs/evaluation/week11-continuation-inspection-v1/RESULTS.md)
covers all nine changed cases and 36 recordings. Mixed training changes H8
actions on identical inputs and trades acquisition/placement gains against
fixture, grip-retention, and release/closure failures.
The [matched-observation continuation replay](docs/evaluation/week11-continuation-replay-v1/RESULTS.md)
now completes 5,928 model calls with exact recorded-action parity. Mixed roughly
halves median horizon-32 motion sensitivity, while changing H8 actions. On clean
155's failed state, both checkpoints at all horizons keep holding; on the
successful state, all command release.
The [clean 155 window test](docs/evaluation/week11-placement-window-v1/RESULTS.md)
completes 12 episodes: fixed8, placement-only (138–217), and late-only (218–297)
each succeed 0/3 times; full adaptive succeeds 3/3. Both narrow windows fail in
two comparisons each with exact consumed-input and physical prefixes. A third
fixed8 repeat diverges before intervention and is excluded from paired causal
interpretation. Neither tested narrow window is sufficient.
The [complementary phase and channel tests](docs/evaluation/week11-phase-split-v1/RESULTS.md)
add 24 episodes: early-only adaptive succeeds 3/3, sustained adaptation starting
at 138 fails 3/3, and early motion-only succeeds 3/3 with fixed8 gripper control.
All 192 early same-input candidate gripper sequences agree; earlier motion changes
the state that later permits release. All decisive early comparisons match;
one late-start comparison is excluded for pre-intervention drift. This supports
a [concrete pilot proposal](docs/evaluation/week11-phase-split-v1/NEXT_STEPS.md):
freeze the stronger H8 continuation, learn only non-8 motion corrections, and
require exact H8 preservation plus gains over a matched adapter control.
The [motion-adapter training pilot is complete](docs/evaluation/week11-motion-adapter-v1/RESULTS.md).
Both arms completed 2,000 updates on the same 58,172 training presentations.
All 456 H8 replay chunks remain exact after training and independent endpoint
loading; frozen weights, matched exposure and resource checks pass. Qualification
p95 is about 108 ms, narrowly under twice the fresh fixed control. The full
pipeline took 42.23 minutes. The subsequent nine-case screen is described below;
robot improvement and generalization remain unproven. All prior evidence is
preserved and M4-fixed remains the baseline.
The [motion-adapter evaluation](docs/evaluation/week11-motion-screen-v1/RESULTS.md)
started and stopped at its frozen latency gate after 27/81 episodes. Rollout
p95 was 2.026×/2.023× the fresh original-fixed reference, above the 2× ceiling.
The first repeat scored H8/main/control 6/6/7 out of nine; all 27 pair-prefix
audits pass, but the three-repeat retention/rescue gate is incomplete. No
extension was launched. The [event-free follow-up](docs/evaluation/week11-h8-validation-v1/RESULTS.md)
then completed latency qualification, the three-case adapter inspection and
128 fresh vanilla/H8 rollouts. Removing the unused event branch preserves exact
H8 actions and brings p95 to 53.33 ms versus vanilla's 54.40 ms on matched inputs.
H8 nevertheless loses two net targeted successes and ties clean, failing the
prospective directional criterion. The targeted panel has 24 underlying tasks
across 32 variant cases; task-grouped sensitivity does not change the conclusion.
No new baseline or larger run is approved by these results.
Follow the [Week 11 ledger](docs/WEEK11_STATUS.md).
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
