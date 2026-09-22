# Week 4: all-L1 mixed-task continuation B

User authorized **B first** on September 21, 2026 PDT. The separately sealed
[all-L1 v2 specification](../configs/training/week4_b0_oft_all_l1_v2.json) preserves
the original continuation-v1 evidence. Run A and a longer extension are deferred.

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

**Launch status:** implementation and local checks in progress; host pilot and
formal launch receipts will be recorded after verification.
