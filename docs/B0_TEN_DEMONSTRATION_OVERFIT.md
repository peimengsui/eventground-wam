# B0 fixed ten-demonstration overfit gate

This gate tests whether the complete B0 policy can memorize every valid action
chunk from ten whole training demonstrations while preserving the locked
normalized-action and gripper semantics.

## Cohort contract

The immutable selection algorithm is
`first-ten-canonical-training-demonstrations-v1`. It selects the first ten
episodes assigned to `train` in canonical dataset order: episode IDs
`0, 1, 3, 4, 5, 8, 9, 10, 11, 12`. They span ten task IDs and contain 1,454
complete `[8,7]` action chunks. No incomplete tail is padded.

Before optimization, the runner seals each episode, its first and last sample,
the ordered identity digest for all 1,454 samples, source and normalization
identity, optimizer settings, tensor shapes, and complete tensor SHA-256 values.
The Qwen processor is then run once for 364 canonical minibatches and its BF16
outputs are cached in host memory. Training shuffles only this fixed minibatch
order with `torch.randperm(seed + zero_based_epoch)`; full-cohort diagnostics
always use canonical order.

## Optimization and acceptance

The predeclared optimizer is AdamW with batch size four, seed 17, no weight
decay, no scheduler, and no augmentation. The decoder/projector learning rate
is `3e-4`; Qwen LoRA uses `1e-4`; betas are `(0.9,0.95)`, epsilon is `1e-8`,
and gradient norm is clipped to `1.0`. Training runs for at most 50 epochs and
evaluates the complete cohort after every epoch, stopping at the first pass.

All 1,454 samples are element-weighted when aggregating L1 and direct
normalized-action MAE. The two independently computed metrics must agree
within `1e-6`. The original locked gate passes only when all three conditions
hold simultaneously:

- final diagnostic L1 is at most 20% of initial diagnostic L1;
- decoded normalized-action MAE is at most `0.10`; and
- gripper sign accuracy is exactly 100% for targets with absolute gripper
  command at least `0.5`.

The trainable allowlist and gradient checks are unchanged from B0. The final
compact checkpoint is deliberately zeroed, must alter full-cohort predictions,
and is then strictly reloaded. Reloaded predictions may differ by at most
`1e-3` and must preserve the acceptance result.

## Real-data command

Run from a clean committed revision:

```bash
./scripts/run_b0.sh python scripts/run_b0_cohort_overfit.py \
  --repository-root ~/data/eventground-wam/libero-plus/datasets/libero-plus-lerobot/current \
  --frozen-root ~/data/eventground-wam/libero-plus/derived/frozen/week2-dataset-v1 \
  --normalization ~/data/eventground-wam/libero-plus/derived/b0/week3-b0-data-v1/action-statistics.json \
  --cache-dir ~/data/eventground-wam/huggingface-cache \
  --local-files-only \
  --device cuda:0 \
  --output-root ~/data/eventground-wam/libero-plus/derived/b0/week3-b0-overfit-10demo-v1
```

The external output directory is exclusive and contains the cohort manifest,
per-minibatch loss/full-cohort diagnostic curve, compact trainable checkpoint,
sealed report, and SHA-256 sidecars. A completed threshold failure is preserved
and returns exit status 2; it is not overwritten or hidden by changing a locked
threshold.
