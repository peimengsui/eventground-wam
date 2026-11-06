# Week 8 coupled policies

**October 9, 2026, 6:05 PM PDT: M3 epoch-2 dev-210 evaluation and result packaging are complete.**
M3 scored **162/210 (77.14%)**, versus M1's **164/210 (78.10%)**. The paired
M3−M1 difference is **−0.95 pp**, with a **[−6.67, +4.76] pp** bootstrap 95%
interval. Exact input matching is 207/210; the three settling-RGB mismatches keep
the whole comparison descriptive. Median/p95 query latency is **46.34/47.79 ms**
for M3 versus **43.33/44.70 ms** for M1. All artifacts passed validation; all M3
failures were policy-step limits. The independent Feifei package contains 1,415
files (18.6 GB). [M3 results and comparison](WEEK8_M3_RESULTS.md).

The user authorized the fresh [Week 9 B0/M1/M3 cohort](WEEK9_STATUS.md), with
M2/M4 added after their fixed endpoints complete. Its runner passed GPU qualification for all three arms and B0
formal evaluation is now running; training settings and scientific protocol remain unchanged.

**October 9, 2026, 5:20 PM PDT check: M3 finished successfully at 1:28 PM PDT.**
It completed all 117,132 updates, 3,412,044 action presentations and 16,259,770
future pairs, with zero service restarts. The completed epoch-2 export passed the
[CPU-only endpoint preflight](evaluation/week8-training-progress/20261010T0020Z/M3/endpoint-preflight.json),
including policy and predictor file checksums. Feifei's GPU is idle; no evaluation
was launched. M4 remains healthy at 51,458 updates (43.93%); M2 is at 59,132
updates (50.48%) and has entered epoch 2. [Latest curves and estimates](evaluation/week8-training-progress/20261010T0020Z/charts/progress-summary.json).


**October 9, 2026: shared M3/M4 evaluation preparation is complete and the
[Week 9 protocol](WEEK9_EVALUATION_PROTOCOL.md) is frozen.** The isolated
[runner](WEEK8_COUPLED_EVALUATION.md) supports strict epoch-2 admission, coupled
checkpoint loading, durable rollout attempts and exact input audits. CPU
preflights on Feifei verify both arms' provenance and the qualified M1 environment;
M4 uses a metadata-only staged bundle. Both correctly wait for epoch 2. No
evaluation was launched and M3 training remains active with zero restarts.
[Preparation evidence](evaluation/week8-coupled-evaluator-v1/preparation.json).


**October 6, 2026: M3 formal training is running on Feifei after explicit evening
launch approval. M4 continues on the Lambda A100 under its separately approved
480-hour time budget.**

| Step | Status |
| --- | --- |
| Freeze common M3/M4 architecture, loss, controls and evaluation | Complete: [design](WEEK8_COUPLED_DESIGN.md), [sealed contract](../configs/training/week8_coupled_design_v1.json). |
| Shared candidate-conditioned future interface and full recovery engine | Implemented; 74 local tests pass and all 215 accepted Week 7 source files remain unchanged. [Implementation](WEEK8_COUPLING_INTERFACE.md). |
| Actual-model numerical correctness, target isolation and export/reload | Both arms pass on both hosts; full joint gradients match the complete-objective reference exactly. |
| Same-device interruption and damaged-checkpoint recovery | Both arms pass on both hosts; replay exactly matches uninterrupted state. |
| M3 on Feifei: 128 exact mixed batches and edge probes | **PASS**, including memory and time. [Qualification](evaluation/week8-m3-feifei-qualification-v1/attempt-02/run/qualification.json). |
| M4 on Lambda A100: 128 exact mixed batches and edge probes | Completed; memory/teacher/finite-gradient checks pass. The conservative frozen projection is **402.33 h**, exceeding 168 h. [Resource report](evaluation/week8-m4-lambda-qualification-v1/attempt-07/run/M4-resources.json). |
| Formal training | **M3 complete** at 1:28 PM PDT October 9; **M4 running** on the Lambda A100. Original initialization and matched exposure rules are unchanged. |
| Fixed-endpoint evaluation and packaging | **M3 complete and packaged**, with M3-versus-M1 analysis; M4 awaits its fixed epoch-2 endpoint. |

## Expected training durations

These are pilot estimates for the complete 117132-update, two-epoch runs. The
practical estimates include the measured checkpoint-save cost at the existing
300-second/2000-update cadence; startup, downtime and evaluation are additional.

| Arm / host | Mean step | Expected duration | Conservative duration including periodic saves |
| --- | ---: | ---: | ---: |
| M3 / Feifei RTX 5090 | 2.042 s | 66.8 h / 2.8 days | 93.3 h / 3.9 days |
| M4 / Lambda A100 SXM4 40 GB | 5.212 s | 179.7 h / 7.5 days | 426.0 h / 17.8 days |

[Completed pilot audit](evaluation/week8-completed-pilot-summary-v1.json) records
all source/runtime/report identities and the exact formulas. The unchanged frozen
admission projections are 92.76 h (M3) and 402.33 h (M4). The user explicitly approved the
[480-hour M4 time amendment](../configs/training/week8_m4_lambda_budget_v1.json)
and launch with “Yes, launch M4 now.” It changes neither exposure nor numerical
hyperparameters, memory limits or storage limits. The original failed time gate
remains preserved; M2's prior exception was not inherited. The practical expected
M4 duration remains about 7.5 days; 480 hours is the approved limit, not the ETA.

## Fairness and memory resolution

The [fairness audit](evaluation/week8-coupled-fairness-v1.json) passes all 31
comparisons for M3 against M1 and all 31 for M4 against M2. The exact original
mixed batches (1..32 anchors), both sample-plan hashes, one optimizer/scheduler
step per logical batch, 3412044 action presentations and 16259770 valid future
pairs are unchanged. Learning rates, optimizer, clipping, loss weights, seeds,
precision and two-epoch exposure remain frozen. The additional identical
1187072-parameter coupling blocks are the intended architectural difference.

The explicit execution revision `coupled-whole-logical-future-batch-v1` computes
the same complete logical auxiliary objective directly, avoiding microbatch
rounding differences amplified by BF16 backbone backward. Candidate prediction
retains microbatch 10. The backbone's inherited precision context is restored
before its single backward pass. No tolerances were relaxed. All prior failed
A100 attempts and diagnostic scripts/logs remain under
[evidence](evaluation/week8-m4-lambda-qualification-v1).

M3's first Feifei timing run completed, but cached GPU reservations reached
29.79 GiB despite a live tensor peak of 15.43 GiB. The successful repeat uses
`PYTORCH_ALLOC_CONF=expandable_segments:True`, pinned in its runtime identity.
It has a 15.41 GiB live peak and 15.77 GiB reserved peak, with no reduction in
effective batch size or change in training arithmetic. The [allocator parity check](evaluation/week8-m3-feifei-qualification-v1/attempt-02/allocator-trace-parity.json) also confirms identical non-timing metrics at all 128 updates, plus bitwise-identical model/optimizer/scheduler/RNG/exposure states at the saved two-update checkpoint (source/runtime identity excluded). M4's resource run has
an 11.49 GiB live peak and 13.36 GiB reserved peak. Every final device trace,
including desktop usage on Feifei, stays below the 90% ceiling.

M1's [training, evaluation and packaging](WEEK7_M1_RESULTS.md) are complete;
Week 7 remains open for M2 on its separate A10. M1's 164/210 against B0's 162/210
remains descriptive under the frozen input-audit rule. No design choice was
retuned from those outcomes. The original JEPA quality gate remains FAIL under
the existing research-use exception. All pilot weights are excluded from future
production initialization.

The A100 service `eventground-m4-lambda-v1.service` is enabled and running,
using [the approved service template](../configs/systemd/eventground-m4-lambda-budget-v1.service).
Its active-time ledger enforces 480 hours; systemd has an additional 481-hour
wall-time guard. Checkpoints/logs are on persistent storage at
`/lambda/nfs/eventground-wam-m4/week8-m4-lambda-v1/run`.
[Launch verification](evaluation/week8-m4-lambda-launch-v1/launch-verification.json)
records fresh original model/CPT initialization, exact pilot parity for logged
updates 1 and 5, the saved update-1 checkpoint, no service restarts, and 19 passing
launch/recovery tests. Every pilot-qualified source file was retained byte for
byte; only separate budget/launch files were added. All batch, loss and optimizer
settings remain unchanged.

The user separately authorized M3 with “Ok, launch now the M3 training on feifei.”
The enabled user service `eventground-m3-coupled-v1.service` started at 6:36:53 PM
PDT October 6, using [the M3 service template](../configs/systemd/eventground-m3-coupled-v1.service).
All 234 source files from the successful Feifei attempt-02 pilot were preserved;
the runtime and admission checks pass unchanged. Its allocator remains
`PYTORCH_ALLOC_CONF=expandable_segments:True`. Fresh original Qwen/C_pixel
initialization, an initially empty optimizer, exact early-update pilot parity,
and the saved recovery checkpoint checksum are recorded in
[M3 launch verification](evaluation/week8-m3-feifei-launch-v1/launch-verification.json).
Training targets 117132 updates over two epochs with the original 192-hour
active-time cap, a 193-hour systemd guard, and unchanged effective batches and
numerical hyperparameters. Output is stored at
`/home/peimengsui/data/eventground-wam/libero-plus/training/week8-m3-coupled-v1`.
Its pilot estimate is about 67 hours, with a conservative estimate of 93 hours;
the expected finish is October 9 afternoon PDT, subject to observed throughput.

The original status-file bytes
referenced by the October 5 freeze are [preserved here](evaluation/week7-m1-final-v1/prior-status/WEEK8_STATUS.md).
The scientific design document and its sealed contract remain unchanged.
