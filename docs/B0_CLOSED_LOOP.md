# B0 closed-loop evaluator integration

The B0 rollout entry point connects the reviewed ten-demonstration checkpoint
to the existing LIBERO evaluator and HDF5 recorder. It reuses the
reference-policy implementation for environment reset, the locked settling
sequence, 8-step open-loop action execution, per-query latency, retry handling,
resume validation, camera capture, and episode recording.

## Locked inference boundary

Before model tensors are loaded, the adapter verifies one continuous identity
chain:

1. the committed review has the accepted-with-documented-residuals disposition;
2. the review identifies the supplied sealed training report, cohort manifest,
   and checkpoint by digest and file SHA-256;
3. those artifacts pass their own digest or SHA-256 sidecars;
4. the report identifies the exact train-only normalization artifact and records
   a passing checkpoint-reload comparison; and
5. the objective is `continuous-l1-plus-gripper-bce-v1` with BCE weight `1.0`.

The runtime recreates pinned Qwen3.5-2B, installs the default rank-16 LoRA, and
strictly restores only the allowlisted trainable state. Every query contains the
rotated 256 by 256 front and wrist RGB frames, the task instruction, and the 8D
proprioceptive state under the same prompt contract used for training.

The model emits an `8 x 7` normalized action chunk. Dimensions 0–5 are inverted
with the train-split q01/q99 affine statistics and then clipped to LIBERO's
`[-1,1]` execution range. Dimension 6 is already classified by the model and
must be exactly `-1 = open` or `+1 = close`; normalization and execution are not
allowed to alter its polarity. The report counts every clipped continuous value
and records the maximum clipping delta.

The entry point accepts exactly one suite/task pair from the sealed full-10K
manifest. At runtime, it verifies the LIBERO task name and requires the normalized
base instruction of an accepted cohort episode to occur as a token-delimited part
of that name. This handles explicit LIBERO-Plus perturbation suffixes such as
`table_1` without treating an unrelated task as a cohort task. The exact matched
cohort episode, runtime instruction, and policy instruction are recorded in the
rollout report. B0 is conditioned on the exact sealed cohort instruction used in
training; the runtime suffix identifies the perturbation but is not silently
introduced into the overfit prompt. This keeps the Week 3 rollout a training-task
overfit gate rather than a generalization claim.

## Validation without a rollout

Task selection can be checked without CUDA, LIBERO imports, or checkpoint files:

```bash
PYTHONPATH=src:scripts python scripts/evaluate_b0_policy.py \
  --suite libero_spatial \
  --task-id 0 \
  --checkpoint-output-root /unused \
  --normalization /unused \
  --libero-root /unused \
  --output-dir /unused \
  --dry-run
```

On the compute host, `--load-only` additionally verifies the clean project and
LIBERO revisions, the full checkpoint bundle, pinned Qwen reconstruction,
strict trainable-state restoration, CUDA availability, and model memory without
creating an environment or executing an action.

## Closed-loop command

After choosing a manifest task whose runtime instruction appears in the
accepted cohort, run:

```bash
./scripts/run_b0_rollout.sh python scripts/evaluate_b0_policy.py \
  --suite <suite> \
  --task-id <runtime-task-index> \
  --checkpoint-output-root \
    ~/data/eventground-wam/libero-plus/derived/b0/week3-b0-overfit-10demo-bce-v2 \
  --normalization \
    ~/data/eventground-wam/libero-plus/derived/b0/week3-b0-data-v1/action-statistics.json \
  --review-evidence docs/evaluation/week3-b0-overfit-10demo-bce-v1.json \
  --cache-dir ~/data/eventground-wam/huggingface-cache \
  --local-files-only \
  --libero-root ~/code/vendor/LIBERO-plus \
  --output-dir \
    ~/data/eventground-wam/libero-plus/derived/b0/week3-b0-rollout-v1
```

The output directory receives a validated HDF5 episode and a digest-sealed
`rollout-report.json` with a SHA-256 sidecar. The report contains task selection,
source and checkpoint identities, objective/decoding contracts, latency, action
clipping, result, peak CUDA memory, runtime versions, and the B0 environment-lock
hash. Existing output is never overwritten; use `--resume` only to accept an
already-complete artifact whose full provenance matches.

## Unperturbed training-task gate

The full-10K entry point above evaluates a perturbed LIBERO-Plus task. The Week 3
acceptance criterion instead permits an unperturbed base task represented in the
sealed training cohort. Those episodes use the separate locked
`week3-b0-training-rollout-v1` diagnostic protocol so the recorder does not
mislabel a base task as a perturbation. The protocol preserves the shared camera,
proprioception, action, settling, step-budget, seed, retry, and HDF5 contracts.

`evaluate_b0_training_task.py` selects an accepted cohort episode by its sealed
episode index, resolves its instruction uniquely through the canonical ten-task
`tasks_info.txt` inventories, and records the one-based classification ID and
zero-based base-suite runtime index. Run a predeclared cohort task with:

```bash
./scripts/run_b0_rollout.sh python scripts/evaluate_b0_training_task.py \
  --cohort-episode-index <sealed-episode-index> \
  --checkpoint-output-root \
    ~/data/eventground-wam/libero-plus/derived/b0/week3-b0-overfit-10demo-bce-v2 \
  --normalization \
    ~/data/eventground-wam/libero-plus/derived/b0/week3-b0-data-v1/action-statistics.json \
  --review-evidence docs/evaluation/week3-b0-overfit-10demo-bce-v1.json \
  --cache-dir ~/data/eventground-wam/huggingface-cache \
  --local-files-only \
  --libero-root ~/code/vendor/LIBERO-plus \
  --output-dir \
    ~/data/eventground-wam/libero-plus/derived/b0/week3-b0-training-rollout-v1
```

Its `training-rollout-report.json` deliberately scopes the result to one
unperturbed training task. It is not evidence of cohort success rate or
generalization.
