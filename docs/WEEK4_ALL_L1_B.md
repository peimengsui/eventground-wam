# Week 4: all-L1 mixed-task continuation B

User authorized **B first** on September 21, 2026 PDT. The separately sealed
[all-L1 v2 specification](../configs/training/week4_b0_oft_all_l1_v2.json) preserves
the original continuation-v1 evidence. Run A and a longer extension are deferred.

## Completed — September 22, 2026 PDT

**Training finished at 06:27:06 PDT; development-200 finished at 06:49:40 PDT.**
The complete formal pipeline took about **6 hours 3 minutes** from its 00:46:28
launch. It completed **15,000 additional updates / 435,945 examples**, reaching
**87,000 cumulative updates**, with zero service restarts, zero rollout retries,
and no invalid rollouts. The checkpoint reload repeated the exact endpoint
validation metrics; an independent SHA-256 check matched the exported checkpoint
to both training and rollout reports. GPU training/evaluation processes have exited.

### Identical development-200 comparison

Every suite contains 50 rollouts; each task contributes the same five fixed states.

| Policy | Spatial | Object | Goal | Long | Overall |
| --- | ---: | ---: | ---: | ---: | ---: |
| **All-L1 mixed B** | **44/50** | **48/50** | **24/50** | **31/50** | **147/200 (73.5%)** |
| OFT parent, update 72,000 | 40/50 | 49/50 | 11/50 | 33/50 | 133/200 (66.5%) |
| Previous full B0 | 41/50 | 45/50 | 29/50 | 15/50 | 130/200 (65.0%) |
| Cached OpenVLA | 48/50 | 50/50 | 47/50 | 47/50 | 192/200 (96.0%) |

B improves **7.0 percentage points overall** over its parent and **8.5 points**
over the previous B0. Goal gains **13 successes / 26 points** over the parent,
with net improvement on five Goal tasks. Object loses one success and Long loses
two. All predeclared engineering checks pass; Long meets its permitted lower bound.
The remaining same-panel OpenVLA gap is **22.5 points**.

### Fixed-panel validation and gradients

| Metric | Unchanged parent | B endpoint |
| --- | ---: | ---: |
| All-action raw L1 | 1.15197 | 0.13989 |
| Motion L1 | 0.14073 | 0.13967 |
| Gripper raw L1 | 7.21938 | 0.14118 |
| Decoded action MAE | 0.13084 | 0.12982 |
| Binary gripper accuracy | 96.43% | 96.46% |
| Median backbone gripper/motion gradient norm ratio | 7.06× | 0.75× |

The large total-L1 decline mostly reflects bringing former gripper logits close
to signed regression targets. Motion error and binary accuracy change only
slightly. The gradient ratio now reflects more comparable weighted contributions
on the fixed 640-example diagnostic. These descriptive validation examples share
trajectories with training; rollout gains are the primary result.

B **passes the development screen**, which marks it eligible for consideration
for extension. This is not Week 4 acceptance or an automatic extension decision.
These are the existing 200 development states, not the full 2,000 clean trials or
`dev-210`. Both the loss and batching changed and training continued, so the result
does not isolate their individual effects. No A, extension or fresh training has
been launched by this completion check. Historical Week 4 acceptance remains FAIL.

Retained outputs, including pilot/checkpoints/rollout recordings: **12.25 GiB**.
[Completion verification](evaluation/week4-all-l1-b-completion-v2.json),
[training report](evaluation/week4-all-l1-b-v2/training-report.json),
[rollout report](evaluation/week4-all-l1-b-v2/rollouts/evaluation-report.json).

## Fixed experiment

- Same Qwen3.5-2B/OFT architecture and rank-32 LoRA as the 72,000-update parent.
- Restore every trained tensor, Adam moment/step and RNG; no gripper head reset.
  Restart child counters at zero and use constant **1e-5** learning rate.
- Loss is mean absolute error over raw **8 × 7** action values:
  `(6 × motion L1 + gripper L1) / 7`. Signed gripper targets are −1/open and
  +1/close; threshold at zero only for execution and binary accuracy. No BCE.
  The inherited `gripper_bce_weight` field is unused by this versioned objective.
- Exactly **15,000 additional updates / 435,945 examples** from the unchanged
  v1 training assignment. Endpoint: **87,000 cumulative updates / 2,532,526 examples**.
- Original predeclared 128-update mixing windows, identical example multiset and
  batch sizes to the planned episode-blocked arm. 14,885 of 15,000 batches contain
  multiple tasks; mean 13.22 tasks per batch. Native image/text preprocessing and
  original normalizers remain unchanged.
- The new trajectory-grouped v2 split is reserved for fresh training.

## Measurements

Unchanged-parent and fixed-endpoint 2,560-example validation reports motion L1,
gripper L1, all-action L1, decoded error and gripper accuracy. The 640-example
weighted gradient panel uses **6/7 motion and 1/7 gripper L1**. Both preserve model,
optimizer and RNG state. Validation remains descriptive seen-trajectory evidence.

The exported endpoint must reload exactly and repeat the validation metrics.
The same host service then runs **development-200**: five fixed initial states
for each of 40 clean tasks. It verifies original cached rollout artifacts and
compares against the parent, previous B0 and OpenVLA, including identical first
policy inputs. This is development evidence, not the full Week 4 acceptance test.

The engineering screen requires at least **16/50 Goal**, gains on at least three
Goal tasks, **133/200 overall**, **47/50 Object**, **31/50 Long**, **37/50 Spatial**.
Report B eligibility only; no automatic A/B winner, extension or Week 4 PASS.

## Execution and recovery

Feifei checkout: `/home/peimengsui/code/eventground-wam-all-l1-b-v2`.
Outputs: `/home/peimengsui/data/eventground-wam/libero-plus/derived/b0/week4-oft-all-l1-v2/B`.
Runtime: existing pinned `eventground-wam-b0-oft-v1/environments/b0/.venv`.

Before formal training, run 128 pilot updates, interrupt and observe a real host
process resume, compare the complete state of two copies of the next update,
check weighted gradients, reload an exported pilot checkpoint, and run two LIBERO
smoke episodes. Pilot updates never carry into the formal run.

Persistent user systemd services with lingering run independently of SSH. Atomic
recovery contains weights, Adam, scheduler, all RNG, exact mixed-batch cursor and
sample-chain identity. Save after update 1, then every 300 seconds or 2,000 updates;
retain three generations. Restart failures after 60 seconds, at most four starts
per hour. Integrity failures exit 78 and stop. One file lock prevents concurrent
pilot/training/evaluation writers. Resume reconstructs prefetched inputs from the
saved deterministic cursor. The service is enabled for host reboot recovery.

Shared B budget: **10 active hours for pilot/training/measurements**, **12 hours
including rollouts**, **50 GiB output cap**, at least **20 GiB free**. A ten-second
persistent heartbeat carries active time over crashes and conservatively charges
an unclosed interval. Host downtime does not consume active time. A measured pilot
projection with 25% margin must fit before the fixed endpoint is launched.

## Historical launch verification — September 22, 2026 PDT

The formal service started at **00:46:28 PDT**, from immutable source
`f0623200d7f35d9339b32f5afbe990aa70cd60e8`. Local verification: **298 tests passed**
and Ruff passed. Original continuation-v1 source pins still validate.

The [128-update pilot](evaluation/week4-all-l1-b-pilot-v2.json) passed at **1.349 s/update**
(**21.77 examples/s**), with peak allocated GPU memory **9.05 GiB**. An intentional
SIGTERM saved update **51**; systemd resumed it automatically in a new process
([resume evidence](evaluation/week4-all-l1-b-pilot-process-resume-v2.json)). Raw
prediction reload and the next update's complete weights/Adam/RNG/cursor state
were bitwise identical. The two probe updates were rolled back. The exported
pilot checkpoint completed [two LIBERO smoke episodes](evaluation/week4-all-l1-b-pilot-rollouts-v2.json),
with exact cached-control input matching and no retries. Their 2/2 success is a
runtime smoke result, not evidence of improvement.

At launch, the point estimate was **5.62 hours** for 15,000 training updates; **7.03 hours** with the
pilot's 25% training margin. Allow **6–8 hours** from launch for training,
measurements and the queued development-200 run. Runtime is an estimate.

The formal run independently restores the original parent, runs the unchanged-
parent measurements, then trains to the fixed endpoint. It does not inherit
pilot updates. The pilot service is disabled; the formal service is enabled for
reboot recovery. The [launch receipt](evaluation/week4-all-l1-b-launch-v2.json) records the evidence.
At **05:19 PDT**, the latest bounded host read confirmed **12,044/15,000 updates
(80.3%)**, **349,969 additional examples**, and **21.59 examples/s**. Roughly
**1.5–2 hours** were then estimated to remain including measurements and development rollouts. The
service was independently confirmed with zero restarts at update 8,645; a
checksummed recovery record at update 8,452 was present. Later checkpoints are
managed by the same 300-second/2,000-update schedule. Brief SSH timeouts did not
stop training. The optional independent formal checkpoint-payload audit could
not finish during those timeouts; the actual pilot payload/replay verification
is complete.

Parent measurements are saved: all-action L1 **1.15197**, motion L1 **0.14073**,
gripper L1 **7.21938**, accuracy **96.43%** on the fixed 2,560 examples. The high
initial raw gripper error reflects conversion of a trained logit to a signed
regression output. The 640-example weighted gradient panel has a median backbone
gripper/motion norm ratio **7.06×**; all-L1 does not automatically balance backbone
gradients at initialization. These measurements do not change the fixed run or
its rollout-based decision. Single training batches are not comparable with the
fixed validation panel; endpoint metrics and rollout outcomes were pending at that launch snapshot.
