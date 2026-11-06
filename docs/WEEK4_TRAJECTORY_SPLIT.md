# Trajectory-grouped split for future fresh B0 training

**ID:** `libero-plus-trajectory-v2`.  
**Defined and verified:** September 21, 2026 PDT.  
**Status:** assignments frozen; no training or policy evaluation launched.

This split corrects the demonstrated train/validation trajectory overlap in
`week2-dataset-v1`. Historical splits, checkpoints, scores and acceptance decisions
remain unchanged. It is a new split definition, not a retrospective repair of
the existing B0 models or a replacement for the full Week 2 event-data freeze.

## Frozen assignments

| Split | Trajectory groups | Episodes / visual variants | Frames | Valid eight-action chunks |
| --- | ---: | ---: | ---: | ---: |
| Train | **1,341** | **11,435** | 1,786,067 | 1,706,022 |
| Validation | **180** | **1,528** | 240,946 | 230,250 |
| Test | **160** | **1,384** | 211,023 | 201,335 |
| Total | **1,681** | **14,347** | **2,238,036** | **2,137,607** |

Every split includes all **40 canonical tasks**. Each task contributes 3–5
validation groups and 3–5 test groups; many episodes within a group are visual
variants of one numeric demonstration, not independent demonstrations.

The target is **80/10/10 by trajectory group within each task**. Integer rounding
produces 79.77%/10.71%/9.52% overall. Neither episode count nor frame count is
forced to those ratios, and the split was not rerolled to adjust its statistics.

## Rule frozen before generation

1. Read the complete state/action sequence of every episode in the same pinned
   Sylvest snapshot. Use full float32 arrays, including shape and every frame.
2. Connect episodes with identical full state/action sequences. Conservatively
   also connect identical action sequences even if states differ.
3. Check every pair of distinct, equal-length numeric sequences for near
   identity: every state/action component within **absolute 1e-5**, zero relative
   tolerance, and identical gripper sequence. Join matching pairs transitively.
4. Keep each connected component together. Abort if a component spans canonical
   tasks. **Camera images and perturbation labels do not participate in grouping**,
   so appearance changes cannot split a numeric trajectory across partitions.
5. Within each task, sort groups by SHA-256 using the frozen namespace
   `libero-plus-trajectory-v2`. Assign whole groups to train/validation/test with
   integer largest-remainder quotas using weights 8:1:1. Tie order is train,
   validation, test. No model outcome, old assignment or performance measurement
   influences the new assignment.

The all-source scan found **1,681 exact state/action sequences**, also 1,681 exact
action sequences. The **9,640 equal-length distinct-sequence pairs** checked
produced no additional near-duplicate edges at this tolerance. Thus the final
components equal the exact numeric groups in this source snapshot.

**Verified zero across-split overlap** for group identity, full state/action
identity, action-only identity, and detected near-duplicate edges. All episodes
are assigned exactly once. Reversing the generator's input order reproduced
identical group bytes; a separate offline validator recomputed group identities,
ranking, quotas, membership and counts from the retained fingerprint evidence.

### What this does and does not establish

The split isolates the detected numeric trajectory families. It is still the
same 40-task distribution, not an unseen-task benchmark. Original demonstration
IDs and simulator reset identities remain unavailable. The near check covers
aligned, equal-length sequences; it does not detect every time shift, trim,
resampling or semantically equivalent demonstration. Therefore zero overlap under
these checks is not a guarantee of independent simulator scenes or all possible
forms of near duplication.

The old test episodes were read **only for structural identity and assignment**.
No policy predictions, losses, task success or failure outcomes were computed on
them. The new test partition is now locked for a future predeclared final model
evaluation; it is not an extra development set.

## Required use in future fresh training

- Initialize from the pinned **base Qwen3.5-2B**, with fresh task-trained modules,
  LoRA, optimizer and scheduler. Do not initialize from either B0 checkpoint or
  use its distilled targets/teacher if claiming independence from v2 holdouts.
- Fit **action and state normalization on v2 training episodes only**. Historical
  v1 normalizers must not carry into a v2 training recipe.
- Use version-aware loading and bind this rule, manifest, assignments and new
  normalization digests into the future recipe/checkpoint. The existing v1 B0
  loader deliberately remains pinned to v1; hooking up v2 and a recovery pilot
  are future launch prerequisites. The compact groups file expands deterministically
  into all 14,347 episode assignments through `expand_groups`.
- Keep all windows, action chunks and augmentation variants of an episode in its
  assigned partition. Keep augmentation stochasticity out of group identity.
- Report supervised metrics separately: motion L1, gripper BCE, decoded motion
  error and gripper accuracy. Use equal task means, then equal group means within
  task, equal variant means within group and equal valid-chunk means per episode.
  Also retain ordinary micro averages and all denominators. This avoids counting
  repeated variants as independent validation evidence.
- Predeclare checkpoint schedules and selection in the next experiment. Rollout
  success remains the development decision measure. The split itself changes
  neither the 200-rollout development screen nor the final 2,000-rollout clean
  acceptance rule.

**Existing B0 continuations cannot use v2 as an unseen holdout.** Of the new
validation episodes, **1,523/1,528** have trajectories in old training, covering
**179/180** validation groups. All **1,384/1,384** new test episodes, covering all
160 test groups, match old training. Resplitting does not undo that exposure.
Continuation screens should use their existing descriptive validation and the
frozen development rollouts, not consume v2 test results.

## Files and verification

- [Prospective rule](../configs/datasets/libero_plus_trajectory_v2.json), digest
  `9afb0fc20cbc7a0c8f1f33fff8f888bc184b7636a433d3e21ab8e7c5d25c3f84`.
- [Split manifest](../configs/datasets/libero-plus-trajectory-v2/manifest.json), digest
  `8607f32f0ed0ef6cd27ea99449a8de0237e23a4eee973ee970e06bbf3fca115a`.
- [Complete compact group membership](../configs/datasets/libero-plus-trajectory-v2/groups.jsonl).
- [Independent verification](evaluation/week4-trajectory-split-v2-verification.json).
- [Generator](../scripts/freeze_trajectory_split.py),
  [grouping implementation](../src/eventground_wam/trajectory_split.py), and
  [offline validator](../scripts/validate_trajectory_split.py).

Detailed fingerprints and expanded assignments are retained in
`outputs/week4-trajectory-split-v2` locally and at
`/home/peimengsui/data/eventground-wam/analysis/week4-trajectory-split-v2/result`
on Feifei. Their hashes are in the manifest. The source snapshot is isolated;
all previously audited train/validation parquet hashes matched the earlier audit.

The CPU-only host-owned systemd job completed in **12.97 seconds** of script
runtime with exit zero and zero restarts. No SSH connection owned its lifetime.
**22 targeted tests passed**, covering duplicate/near-chain isolation,
order-independence, task coverage, malformed inputs, diagnostic calculations and
the unchanged continuation freeze. Lint and all 20 prior freeze pins passed.

```bash
.venv/bin/python scripts/validate_trajectory_split.py
.venv/bin/python scripts/validate_continuation_freeze.py
```

The first command requires the retained fingerprint/assignment evidence; pass
`--evidence PATH` if copied elsewhere. Next experiments are proposed separately
in [the priority list](WEEK4_NEXT_EXPERIMENTS.md); none is launched by this split task.
