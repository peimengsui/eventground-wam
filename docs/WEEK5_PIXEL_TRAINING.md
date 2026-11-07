# Week 5 Step 4: frozen pixel CPT recipe and recovery pilot

**Status: complete.** The numerical recipe and common order are frozen, and the
Feifei resource, process-recovery and accumulation checks pass. Step 5 production
training (local archive: `WEEK5_PIXEL_PRODUCTION.md`) completed and passed final verification on
September 26, 2026, at the fixed 221,633-update endpoint.

The [production recipe](../configs/training/week5_pixel_cpt_v1.json) builds on the
[verified Step 3 architecture](WEEK5_PIXEL_PREDICTOR.md) and preserves the
[Step 2 acceptance criteria](WEEK5_PIXEL_BASELINES.md). Pilot weights are discarded;
production starts with fresh predictor weights and the pinned base Qwen model.

## Frozen exposure and logical batches

The [common CPT order](../configs/datasets/trajectory-v2-cpt-order-v1/manifest.json)
contains **one complete pass through all 8,221,365 training pairs**: 1,763,197
unique current anchors in 11,435 episodes, over **221,633 optimizer updates**.
Each pair supplies front and wrist targets. No test or validation payload enters
training or the pilot. Pixel and JEPA must use the same order, horizons, logical
batches and total pair exposure. This budget is separate from the unchanged
two action-training epochs for B0/M1–M4.

Episodes are ordered by a seeded SHA-256 rank, then grouped into pools of four.
Within each episode, a seeded PCG64 permutation visits every anchor with a valid
h=2 target exactly once. The pool contributes two anchors per episode in round
robin order; eight anchors form a logical batch. Attach all valid horizons in
ascending order, retaining every near-end target and every short final batch.
No sample is repeated, padded or dropped to fill a batch.

Logical batches contain **1–40 future pairs**, averaging **37.0945**. The four
episode pool bounds the decoded-video cache and mixes observations without
decoding a new random episode for every pair. There is no augmentation or extra
task/motion reweighting. The loss gives every pair and both cameras equal weight
within its logical batch; the production exposure follows the natural counts
of the frozen index.

| Horizon (frames) | Training pairs |
| --- | ---: |
| 2 | 1,763,197 |
| 4 | 1,740,327 |
| 8 | 1,694,587 |
| 16 | 1,603,107 |
| 32 | 1,420,147 |

The on-host plan uses four memory-mapped NumPy arrays, about 16.4 MiB in total.
Its hashes and full coverage are independently verified; metadata-only
regeneration must reproduce the exact arrays and manifest.

## Model, loss and optimization

| Choice | Frozen value |
| --- | --- |
| Predictor | Step 3 architecture v2; 4,631,427 trainable parameters |
| Base features | Pinned base Qwen3.5-2B, BF16, evaluation mode, zero trainable parameters |
| Predictor precision | FP32; no autocast or gradient scaler |
| Training objective | L1 + 0.1 × (1 − SSIM), equal pairs/cameras |
| SSIM | Exact Step 2 float64 Gaussian moments, valid borders |
| Optimizer | AdamW; peak LR 0.0001; betas 0.9/0.999; epsilon 1e-8; weight decay 0.0001 |
| Schedule | 1,000-update linear warmup, then cosine to LR 0.00001 at the final update |
| Gradient clipping | Global norm 1, once per complete logical batch |
| Feature microbatch | Four distinct current observations |
| Predictor microbatch | At most ten pairs, with gradient accumulation |
| Initialization | Fresh predictor seed 20260925; no B0-trained, Step 3 or pilot checkpoint |
| Checkpoint selection | Fixed final update 221,633; no validation-based early stopping or best-of-run selection |

L1 retains the implementation-tested reconstruction objective. The 0.1 SSIM
weight adds local structure supervision and was selected before production,
without held-out predictor tuning. Both measurements and final acceptance
remain those of Step 2; passing the pilot says nothing about that gate.

For a logical batch of N pairs, each microbatch mean loss is multiplied by its
size divided by N. Gradients accumulate before one clip, optimizer update and
scheduler update. This also applies to short batches. The pilot compares losses
and every parameter gradient with an unsplit full batch, including a real
one-pair tail batch. It requires loss error ≤1e-6, relative gradient L2 error
≤1e-5 and maximum absolute gradient error ≤1e-5.

Current observations are encoded online, once per anchor within the batch;
their features are shared across its valid horizons. Only batch-local feature
tensors are retained. There is **no persistent feature cache** to budget or
invalidate. Both base extraction and predictor backpropagation belong to the
measured training path. Strict deterministic algorithms are enabled, TF32 is
disabled, and the CUDA workspace, runtime versions and source hashes are part
of the recovery identity.

## Durable recovery and stopping rules

The new runner saves predictor weights, AdamW moments and step counters,
scheduler state, Python/NumPy/CPU/CUDA RNG state, the next logical-batch cursor,
acknowledged pair count, cumulative loss and a chained sample digest. It verifies
recipe, common order, code, runtime, source data and normalization identities.
Pilot and production use different identities; pilot checkpoints cannot resume
a production run.

Checkpoints are committed only at complete optimizer boundaries. The writer
fsyncs an atomic tensor file before publishing its sealed commit record and
latest pointer. It retains three committed generations and prunes only after a
new commit. Incomplete files are ignored; damaged generations fall back to the
newest intact committed checkpoint. Identity drift or the absence of any intact
committed generation stops the run. One exclusive process lock owns each run.

Production saves every 500 updates or 15 minutes, whichever arrives first.
A crash replays work after the last committed boundary, including an entire
partially accumulated batch. SIGTERM requests a save at the next complete
boundary. Nonfinite loss/gradients, integrity failures, resource failures and
the cumulative **72-hour active-runtime cap** stop training without accepting
an incomplete endpoint. The active-time ledger survives process restarts.

Production must pass the exact recipe/code/runtime pilot gate before it starts.
It also requires at least 5 GiB free disk at startup and before each checkpoint.
The final exported weights are a `C_pixel` **candidate** until Step 6 evaluates
the fixed endpoint against the held-out gate. Failure does not trigger an
automatic longer run or checkpoint search.

## Pilot scope and artifacts

The resource pilot selects 32 four-update windows spread across the complete
order, including the longest episode and longest instruction pools. It covers
128 logical updates with source hashing, parquet/video decoding, online frozen
Qwen extraction, pixel loss/backpropagation, accumulation and durable saves.

Three separate processes run the recovery test: an uninterrupted reference,
a process killed by SIGKILL after the first microbatch of update index 19,
and a new process resuming the last committed update 16. Full endpoint state
must match bitwise, including optimizer, scheduler, all RNG state, exposure,
cursor and sample chain. A separate probe damages copied checkpoint generations
to verify fallback without modifying the original recovery artifacts.

Resource acceptance requires full-device VRAM below 85% of capacity, host RSS
below 24 GiB, projected new persistent data below 2 GiB, and at least 5 GiB free
disk. The runtime projection uses observed 95th-percentile compute/warm-load
costs, full-epoch decoding and checkpoint overhead, then adds a 1.5× margin.
It must stay below 60 hours before the 72-hour runtime cap is accepted.
Full validation and JEPA target encoding are outside this estimate.

```text
Common order:
/home/peimengsui/data/eventground-wam/libero-plus/derived/future-pairs/trajectory-v2-cpt-order-v1

Retained source / log root:
/home/peimengsui/data/eventground-wam/analysis/week5-pixel-cpt-pilot-v1

Pilot outputs:
/home/peimengsui/data/eventground-wam/analysis/week5-pixel-cpt-pilot-v1/pilot
```

The pilot runs under a host-owned service with a six-CPU quota and a 24 GiB
memory limit, independent of SSH. The same locked B0 runtime is used throughout.
No optimized kernels or dependencies were installed for this step.

## Verified results

The [completion ledger](evaluation/week5-step4-completion-v1.json) pins the
recipe, source snapshot and evidence. The [pilot report](evaluation/week5-pixel-cpt-pilot-v1/pilot-report.json)
passed all checks, and the host-owned service exited successfully with no
service restarts. Local lint and all **403 tests** pass, including 14 new
training/order/recovery tests.

| Measurement | Result |
| --- | ---: |
| Full-path pilot | 128 logical updates; 4,694 distinct train pairs; 127 decoded episodes |
| Conservative full-pass runtime projection | **22.90 hours**, including a 1.5× margin |
| Pilot throughput, including cold episode decodes | 120.4 pairs/second |
| Full-device GPU peak observed across all checks | **8.50 GiB** |
| Training-process PyTorch allocated / reserved peak | 4.84 / 5.12 GiB |
| Training-process host RSS peak | **5.20 GiB** |
| One complete recovery checkpoint | 53.12 MiB |
| Projected new production artifacts, including write headroom and log allowance | **310.58 MiB** |
| Persistent feature-cache storage | **0** |

The full-device peak includes the larger unsplit-gradient comparison as well
as ordinary accumulated training. Training timing includes source decoding,
current-observation feature extraction, target construction, the combined loss,
backward and optimizer updates. The mean feature and update stages were about
0.116 and 0.086 seconds per logical batch. The runtime projection is a pilot
estimate, not a measured full training run; final validation needs additional time.

The killed process had completed 19 updates and backpropagated 10 of 39 pairs
in its next logical batch. The new process restored update 16, replayed the
uncommitted work, and reached update 128 with **bitwise-identical weights,
optimizer, scheduler, RNG, cursor, pair count, sample chain and cumulative loss**.
All 112 updates after the restore point also match the uninterrupted run's
loss, learning rate, gradient norm and sample identity exactly. Corrupting the
two latest generations in a copied recovery directory recovered update 120.

The 40-pair full/accumulated comparison had loss error **7.45×10⁻⁹**, relative
gradient L2 error **4.59×10⁻⁶**, and maximum gradient error **2.53×10⁻⁷**. The
one-pair tail batch matched exactly. Base Qwen weights remained unchanged.
An [independent local audit](evaluation/week5-pixel-cpt-pilot-v1/local-verification.json)
checked the seals/source pins, replayed-update records, GPU telemetry maximum,
gradient thresholds and runtime calculation.

Step 4 consumed 259 completed diagnostic optimizer updates across the reference,
killed and resumed processes, including replayed work: **9,486 pair presentations**.
The killed partial batch added 10 backward-only presentations, and gradient
equivalence added 82 backward-only presentations with no optimizer update.
These checks read no validation or test payloads and do not contribute weights
to the production initializer. Step 3's separate diagnostic exposure remains
recorded in its own completion ledger.

**Step 5 is complete.** The production record (local archive: `WEEK5_PIXEL_PRODUCTION.md`)
documents the separate source copy, fresh output directory, verified initialization
and persistent service. The runner's `--phase train` passed its exact recipe,
source, order and runtime gate using the retained passed pilot report.
Step 6 (local archive: `WEEK5_PIXEL_VALIDATION.md`) subsequently evaluated the fixed final
candidate against the unchanged held-out success gate. All numerical criteria,
independent verification and the fixed 480-example review passed/completed;
the byte-identical `C_pixel-v1` export is accepted.
