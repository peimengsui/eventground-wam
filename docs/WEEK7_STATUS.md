# Week 7 auxiliary VLA policies

**Updated October 6, 2026. Steps 1–4 complete; M1 training, evaluation and packaging complete; M2 training continues on Lambda.**
The [experiment contract](WEEK7_EXPERIMENT.md) includes both M1 Pixel and M2 JEPA.
Week 6 is fully closed and both module dependencies are satisfied by the
[recorded handoff](WEEK6_CLOSEOUT.md). The original JEPA gate remains FAIL;
`C_jepa-study-v1` is authorized for research use by the October 2 exception.

## Execution steps

| Step | Work | Status |
| --- | --- | --- |
| 1 | Verify Week 6 closure; freeze arms, inputs, matched controls, development evaluation and completion criteria. | **Complete.** [Entry audit](evaluation/week7-freeze-v1/week6-entry-verification.json), [sealed contract](../configs/training/week7_auxiliary_experiment_v1.json) and [completion ledger](evaluation/week7-step1-completion-v1.json). |
| 2 | Design the common OFT-compatible auxiliary feature route and gradient/information boundaries. | **Complete.** [Shared interface](WEEK7_AUXILIARY_INTERFACE.md); original action path preserved. |
| 3 | Implement future targets, masks, objectives and the auxiliary trainer. | **Complete.** Exact action batches, matched future masks, whole-batch losses, joint training and recovery. |
| 4 | Freeze numerical recipes after matched training-only diagnostics; pass correctness, resource and recovery pilots. | **Complete.** [Results](WEEK7_AUXILIARY_VERIFICATION.md), [M1 recipe](../configs/training/week7_m1_auxiliary_v1.json), [M2 recipe](../configs/training/week7_m2_auxiliary_v1.json), [completion ledger](evaluation/week7-steps2-4-completion-v1.json). |
| 5 | Train fresh M1 and M2 policies to the fixed two-epoch endpoint. | **M1 complete:** October 6, 05:21 PDT; 117,132 updates in 66.47 active hours. M2 remains in progress under its 480-hour allowance; 20,121 updates (17.2%) in the [October 6 check](evaluation/week7-m2-progress/20261006T1509Z/summary.json). |
| 6 | Evaluate on paired dev-210, analyze results and freeze the evidence/checkpoints. | **M1 complete:** 164/210 vs B0 162/210, descriptive only because 207/210 full prefixes match. [Results and package](WEEK7_M1_RESULTS.md), [completion](evaluation/week7-m1-final-v1/completion.json). M2 evaluation and whole-week closure remain pending. |

## Fixed controls

- B0 is the completed epoch-2 control. New arms begin from pinned base Qwen and
  fresh shared components, plus their respective original CPT module.
- Each policy receives 117,132 action updates and 3,412,044 presentations with
  the exact existing logical batches and seeds. No extra B0 continuation.
- Both cameras and horizons 2/4/8/16/32 share masks and target availability.
  Action samples remain unchanged when future targets are missing.
- Future prediction affects shared representations through training losses;
  deployed M1/M2 use the original action architecture with auxiliary modules removed.
- Week 7 measures development evidence. Positive, null and negative policy effects
  are valid outcomes; test and larger confirmation panels remain locked.

The shared implementation passes 108 local tests and real RTX 5090 numerical,
resource, target-isolation and exact process-recovery checks. Both arms match
B0's recorded fresh initialization. All failed implementation attempts are
preserved in the [verification report](WEEK7_AUXILIARY_VERIFICATION.md).

Pre-launch conservative fixed-endpoint estimates on **Feifei** were **92.7 hours for M1** and
**166.2 hours for M2**. JEPA is close to the 168-hour feasibility ceiling; both recipes retain
192-active-hour caps in the original recipes. The separate Lambda M2 time
amendment below supersedes only its wall-time limits. All pilot weights are excluded from production. No
validation scoring or rollout had started at that qualification stage. M1 has
since completed training and dev-210 evaluation; M2 remains active.
The A10 supplementary pilot failed the original timing limit; the separate
budget amendment and completed recovery checks below authorize its launch.

## M1 production launch

The October 3 user request authorized starting M1. Service
`eventground-m1-auxiliary-v2.service` runs on Feifei from the unchanged frozen
M1 recipe, fresh shared initialization and original `C_pixel-v1`. All 206
accepted source files are unchanged; a separately hashed production wrapper
adds the full two-epoch loop, endpoint exports, status and resource enforcement.
The run uses the recorded Python/CUDA/package/driver versions and original
base/module hashes. Pilot weights and trained B0 weights are excluded.

The six production-wrapper tests cover exact recovery around epoch export and
transition, budget stopping, exposure counters and GPU process classification;
the 19 auxiliary tests also passed during launch preparation. The first service
attempt stopped before model loading because the GPU check counted small Chrome
and GNOME desktop allocations as a training job. Its zero-update evidence is
[preserved](evaluation/week7-m1-launch-v1/disposition.json). The corrected check
allows only those known desktop applications below 256 MiB each and still
rejects any other foreign compute process. No scientific recipe changed.

Production output is at
`/home/peimengsui/data/eventground-wam/libero-plus/training/week7-m1-auxiliary-v2`
on Feifei. Recovery saves after update 1, every 300 seconds or 2,000 updates,
and around epoch boundaries, retaining three complete states. The service
survives logout and resumes after a host restart; integrity failures stop the
run. The 192-active-hour cap, 100-GiB output cap and 50-GiB free-space reserve
remain enforced. Epoch-2 is the fixed endpoint; action-only checkpoints and
adapted auxiliary predictors are exported separately. Evaluation remains Step 6.

The [prior status snapshots](evaluation/week7-m1-launch-v2/prior-status/manifest.json)
preserve the exact document bytes referenced by the steps 2–4 completion ledger.

## M2 Lambda A10 qualification

The October 4 user request authorizes M2 on a separate Lambda instance while M1
continues on Feifei. A100 capacity was unavailable, so the user provisioned one
A10 with 24 GB nominal VRAM. This is an explicit hardware/environment change;
the original Feifei measurements do not establish A10 feasibility or cross-GPU
bitwise equivalence. The frozen M2 numerical recipe remains unchanged.

The [pilot package](evaluation/week7-m2-lambda-pilot-v1/package.json) preserves all
206 accepted source files and adds a separate A10 qualification wrapper. It uses
the frozen loss weight 1.5140534185764003 without recalibration, verifies the
original shared initialization, repeats correctness and teacher-isolation checks,
measures the first 128 complete logical updates and edge batches, and requires
exact same-device process recovery including a damaged-checkpoint fallback.
Pilot checkpoints are excluded from production. The 168-hour projection ceiling,
three-hour pilot limit, 28-GiB host limit and 90% GPU-memory ceiling remain enforced.

The [runtime profile](evaluation/week7-m2-lambda-pilot-v1/runtime.json) pins Python
3.10.21, the original package versions and CUDA 13.0 on the A10/580.105.08 driver.
Inputs and the isolated runtime were copied directly from Feifei; the temporary
write-only upload key was removed afterward. No personal private key was copied.
The [storage check](evaluation/week7-m2-lambda-pilot-v1/storage-preflight.json)
verified atomic writes, fsync and checksum roundtrip on the persistent filesystem.

Service `eventground-m2-lambda-pilot-v1.service` ran on `147.224.57.188` with its
source and evidence under `/lambda/nfs/eventground-wam/week7-m2-lambda-pilot-v1`.
The dataset, model cache and runtime use the instance's local SSD. Runtime path
aliases preserve the original absolute CPT-module and interpreter references.
The [launch record](evaluation/week7-m2-lambda-pilot-v1/launch.json) records the
package, runtime and service identities.

The [completed A10 pilot](evaluation/week7-m2-lambda-pilot-v1/disposition.json)
passed the original correctness tolerances, initialization/teacher checks and
memory/edge limits. Its first 128 updates averaged **7.0118 seconds** (p95
**11.3866 seconds**), giving **228.14 hours** at the mean and **444.86 hours**
under the frozen conservative projection, including the measured save allowance.
The projection exceeds the **168-hour** gate, and even the mean-based training
estimate exceeds the **192-active-hour** cap. Peak whole-device memory was
**11,387 MiB**; GPU memory capacity was not the limiting factor.

The pilot stopped at 11:22 PDT with its evidence and pilot checkpoint retained.
Process-recovery phases were initially skipped after the failed resource gate;
no full production run or rollout started at that point. The instance remains
provisioned and billed until terminated through Lambda.
H100 PCIe capacity in the same region was also unavailable according to the user.
At the comparison snapshot, M1 had about 41.38 mean-based hours remaining;
adding Feifei's M2 pilot mean estimate of 79.09 hours gives about **120.47 hours**
from that point, versus 228.14 mean-based hours on A10. These are planning
extrapolations, not full-run guarantees. That comparison optimizes M2 completion
time; placing M2 on Lambda instead frees Feifei for M1 evaluation and later M3.

## M2 Lambda time amendment and production launch

The subsequent October 4 instruction, “Can you start M2 training on the lambda
host”, authorizes the previously discussed 480-hour A10 compute allowance.
The [separate amendment](../configs/training/week7_m2_lambda_budget_v1.json)
sets a 480-hour projection limit and 480-active-hour production cap. The
original recipe, original 168-hour timing failure, numerical losses, two-epoch
exposure, initialization, optimizer, memory and storage limits remain unchanged.
The new admission decision uses the recorded 128-update pilot measurements;
it is a budget revision after seeing those timings, not a newly passed original
feasibility gate. No additional timing pilot or loss calibration is performed.

The unchanged A10 pilot implementation completed the missing process
recovery phases under `eventground-m2-lambda-recovery-v1.service`. The first
supplemental wrapper attempt found a missing manifest path and stopped before
model loading; the corrected path uses the existing sealed manifest inside the
source package. Its failure log remains preserved. The
[recovery supplement](evaluation/week7-m2-lambda-pilot-v1/run/recovery-supplement.json)
passed in 286.8 seconds: resumed model, optimizer, scheduler, RNG, cursors and
exposure chains match uninterrupted execution bit for bit, and damaged-latest
fallback works. Independently audited device traces remained below the GPU
ceiling. The [amended qualification](evaluation/week7-m2-lambda-pilot-v1/run/qualification.json)
retains `original_resource_gate_passed: false`.

Production started **October 4, 13:44:04 PDT / 20:44:04 UTC** on
`ubuntu@147.224.57.188`, service `eventground-m2-lambda-v1.service`, initial PID
19134. The [package](evaluation/week7-m2-lambda-launch-v1/package.json) preserves
all 217 qualified files, including the 206 accepted core files, and adds the
production wrapper, amendment and qualification evidence. All 38 relevant local
tests and lint checks passed. Linux admission checks verified the actual data
aliases, frozen numerical recipe, exact runtime, source hashes, amended timing
allowance, original correctness and same-device recovery results before loading
models. The macOS-only package check could not match Linux `/home` aliases;
the full packaged admission check was repeated successfully on Lambda.

The [launch audit](evaluation/week7-m2-lambda-launch-v1/launch-audit.json) verifies
fresh shared initialization and original CPT weights, an empty initial optimizer,
finite production losses and a checksum-verified, loadable update-1 checkpoint
with populated optimizer state. The [launch record](evaluation/week7-m2-lambda-launch-v1/launch.json)
captures progress through update 13, with no service restarts. Pilot checkpoints
are excluded. M1 continued independently on Feifei with no restarts.

Source, logs, recovery checkpoints and epoch exports live under
`/lambda/nfs/eventground-wam/week7-m2-lambda-v1`, with checkpoints in `run/recovery`.
The enabled service survives logout and resumes after host restart when the
required filesystem and local inputs remain available. It checkpoints after
update 1, every 300 seconds or 2,000 updates, and at epoch boundaries, retaining
three complete states. The existing 28-GiB host, 90% GPU, 100-GiB output and
50-GiB free-space limits remain enforced. A crash-persistent active-time ledger
enforces 480 hours across restarts; the service also has a 481-hour per-process
backstop. The endpoint remains exactly 117,132 updates / two epochs, with
evaluation pending Step 6.

The 480-hour allowance limits active training; it does not cap the complete
Lambda bill. Setup, idle time, storage and taxes remain additional. Stopping the
training service does not terminate the provisioned instance.

## M1 evaluation preparation and M3/M4 design freeze

On October 5, the fixed-endpoint M1 evaluator was prepared and staged separately
on Feifei. It preserves all 215 M1 production source files, accepts only the
cross-linked completed epoch-2 export, and includes dev-210 rollout receipts,
full canonical input auditing and the inherited statistical summaries.
The CPU-only host preflight verified source, runtime, LIBERO, panel, normalizers
and availability of all 210 B0 artifacts; it correctly reports waiting for
epoch 2. No evaluation service was installed or started. The new checks and
related production/inference tests passed (29 tests); actual endpoint reload,
GPU inference, rollouts and final packaging are still pending.

The [M3/M4 scientific design](WEEK8_COUPLED_DESIGN.md) is sealed before M1/M2
rollout outcomes. Both use the same candidate-conditioned horizon-8 future-token
attention interface, fresh shared initialization and original CPT modules, with
unchanged two-epoch controls and inherited loss weights. Implementation and
separate Feifei/A100 qualification remain [next steps](WEEK8_STATUS.md).
The original JEPA gate remains FAIL. Neither new production training nor a new
cloud instance has been started by this preparation.

## M1 completion, October 6

M1 completed its fixed epoch-2 endpoint and the entire dev-210 panel. The
[results](WEEK7_M1_RESULTS.md) record 164/210 successes, B0's 162/210, the
descriptive +0.95-point gap and −4.29 to +6.67-point task-bootstrap interval.
The complete canonical-prefix audit is exact on 207/210 tasks; three small
settling-image mismatches require the entire contrast to remain descriptive.
All first-query inputs and simulator states match; all tasks remain included.

The final package on Feifei includes independent checkpoint/recovery copies,
all M1/B0 episodes, source/runtime identities, costs, failed startup evidence,
representative rollout review and a reverified inventory. The endpoint and
predictor are also backed up off host. Feifei's GPU is free for the next
authorized implementation/qualification work. M2 is still running normally on
Lambda, with finite logged losses, no restarts and a verified recovery state
in the October 6 check. This closes M1 only, not Week 7 as a whole.

Pre-closeout document snapshots, including the status document referenced by
the Week 8 design freeze, are [preserved](evaluation/week7-m1-final-v1/prior-status/manifest.json).
