# B0 one-fixed-batch overfit gate

This gate verifies that the complete real-data B0 path can memorize one
predeclared batch. It is a diagnostic of data/model/gradient/checkpoint wiring,
not a generalization result.

## Fixed batch and optimization contract

The selection algorithm is `first-complete-canonical-b0-batch-v1`: take the
first complete batch from the canonical, unshuffled training stream at epoch
zero. Version 1 uses seed 17 and batch size four. Before model optimization,
the runner seals each sample identity, instruction, source and normalization
identity, optimizer settings, tensor shape, and tensor SHA-256.

The predeclared optimizer is AdamW with no weight decay or scheduler. The
decoder and conditioning/projector parameters use learning rate `1e-3`; Qwen
LoRA uses `1e-4`; betas are `(0.9,0.95)`, epsilon is `1e-8`, and gradient norm
is clipped to `1.0`. The maximum is 2,000 updates with deterministic diagnostic
evaluation every 20 updates. The same in-memory dual-camera batch is used for
every update; there is no augmentation or resampling.

Only the existing exact B0 allowlist is optimized: the 16 upper-layer Qwen LoRA
tensors, Qwen-memory projection, proprioception projector, and action-query
decoder. Gradient presence, finiteness, frozen-base isolation, and nonzero
signal in every required group are checked during the run.

## Locked acceptance

Evaluation uses `model.eval()` and `torch.no_grad()` on the fixed batch. It
independently calculates the model's normalized-action L1 and direct mean
absolute error over `[B,8,7]`; the two values must agree within `1e-6`.

The first evaluation satisfying both original design thresholds passes:

- diagnostic L1 divided by initial diagnostic L1 is at most `0.05`; and
- decoded normalized-action MAE is at most `0.05`.

The thresholds are constants and are not CLI options. A trained compact
checkpoint is deliberately zeroed in memory, must change the predictions, and
is then strictly reloaded. Reloaded predictions may differ by at most `1e-3`
and must retain the same acceptance outcome.

## Real-data command

Run from a clean committed revision in the isolated B0 runtime:

```bash
./scripts/run_b0.sh python scripts/run_b0_overfit.py \
  --repository-root ~/data/eventground-wam/libero-plus/datasets/libero-plus-lerobot/current \
  --frozen-root ~/data/eventground-wam/libero-plus/derived/frozen/week2-dataset-v1 \
  --normalization ~/data/eventground-wam/libero-plus/derived/b0/week3-b0-data-v1/action-statistics.json \
  --cache-dir ~/data/eventground-wam/huggingface-cache \
  --local-files-only \
  --device cuda:0 \
  --output-root ~/data/eventground-wam/libero-plus/derived/b0/week3-b0-overfit-1batch-v1
```

The output directory is created exclusively outside Git. It contains the fixed
batch manifest, per-update loss curve, compact trainable checkpoint, sealed
overfit report, and SHA-256 sidecars. A failed threshold result is also recorded
and returns exit status 2; changing the locked thresholds is not a valid retry.

## Completed result

The version-1 gate passed on the RTX 5090 from clean revision
`795f152b71e62bbbf61f1cf5f330e5be8847f0d1`. Evaluation-mode normalized L1/MAE
fell from `1.2741760015` to `0.0484590121` at step 880, a final/initial ratio of
`0.0380316471`. Gripper sign accuracy was 100%. Every trainable gradient was
present and finite, every required group had nonzero signal, and no frozen-base
gradient appeared. The trained checkpoint reloaded with zero action difference
after deliberate in-memory perturbation. Compact evidence is sealed at
[`evaluation/week3-b0-overfit-1batch-v1.json`](evaluation/week3-b0-overfit-1batch-v1.json).
