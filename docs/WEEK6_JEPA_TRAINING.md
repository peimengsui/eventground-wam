# Week 6 Step 4 — JEPA training recipe and recovery pilot

**Status: complete.** The frozen recipe, Feifei resource/recovery pilot and
independent audit pass, together with all 471 local tests and new-code lint.
Fresh [production training](WEEK6_JEPA_PRODUCTION.md) completed September 27 at
20:57:28 PDT using the exact passed source/recipe/runtime; the final audit passes.
The
[completion ledger](evaluation/week6-step4-completion-v1.json) pins the recipe,
source snapshot, recovery evidence and verified resource projection.

The [sealed recipe](../configs/training/week6_jepa_cpt_v1.json) reuses the exact
[Pixel common order and optimizer schedule](WEEK5_PIXEL_TRAINING.md), with the
[verified JEPA model and target space](WEEK6_JEPA_PREDICTOR.md). No Step 3 or pilot
weights can initialize production. The [held-out acceptance criteria](WEEK6_JEPA_BASELINES.md)
remain unchanged.

## Frozen numerical recipe

| Item | Value |
| --- | --- |
| Pair order | Same one-pass common order as Pixel; 8,221,365 pairs, 221,633 logical updates |
| Logical batch | Up to eight current anchors from a four-episode pool; all valid horizons 2/4/8/16/32; 1–40 pairs |
| Model | Exact Pixel core (4,238,592 parameters) plus JEPA head (526,336); fresh seeds 20260925/20260926 |
| Frozen base/teacher | Pinned Qwen3.5-2B, BF16, unadapted, evaluation mode |
| Predictor and loss | FP32; cosine + 10 × grouped variance penalty |
| Variance rule | Average repeated horizons of each anchor first; compare population std across anchors at fixed camera/spatial/channel coordinates; half teacher std floor, epsilon 1e-4, sqrt(2048) scaling |
| Short batches | Retain every pair; cosine mean over actual pairs; one distinct anchor gives zero variance penalty |
| AdamW | Peak LR 1e-4; betas 0.9/0.999; epsilon 1e-8; weight decay 1e-4; foreach/fused disabled |
| Schedule | 1,000-update warmup, then cosine decay to 1e-5 at fixed final update 221,633 |
| Gradient clip | Global norm 1, once per logical update |
| Current feature microbatch | Four distinct anchors, as in Pixel |
| Predictor microbatch | Ten pairs with two-pass logical-batch gradients |
| Selection | Fixed final endpoint; no validation tuning, early stopping or best-checkpoint search |

Weight 10 carries forward the training-only Step 3 evidence: it learned the
original controlled action-order task, and the real diagnostic beat copy-latent.
This choice is frozen before production; no held-out predictor results informed it.
The inherited common-order manifest's historical RGB-weighting label describes
its Pixel origin. JEPA uses equal pairs/cameras/spatial tokens for cosine loss
and equal distinct anchors for variance, while preserving every sample and batch.

## Correct accumulation

The variance objective couples predictions from different anchors. Independent
microbatch variance penalties would change the objective. The
[JEPA training engine](../src/eventground_wam/jepa_training.py) therefore:

1. Runs each predictor microbatch without retaining its graph and concatenates
   predicted features into one FP32 logical-batch bank.
2. Computes the complete cosine and grouped variance objective once, obtaining
   its derivative with respect to every predicted feature.
3. Recomputes each predictor microbatch and backpropagates the corresponding
   feature derivative. Recomputed predictions must match the first pass exactly.
4. Clips once, takes one AdamW step and advances the scheduler/cursor once.

No optimizer update or stochastic layer intervenes between the two passes.
This keeps at most one predictor microbatch graph alive. Tests compare every
parameter gradient against an ordinary unsplit forward/backward with uneven
anchor groups, groups crossing microbatch boundaries, short remainders and
singleton anchors. The GPU pilot repeats the check on a real 40-pair batch and
the actual one-pair tail, requiring loss error ≤1e-6, relative gradient L2 error
≤1e-5 and maximum gradient error ≤1e-5.

## Teacher execution and cache budget

Current Qwen features are encoded online once per anchor within a logical batch,
with the same batching and prompt as Pixel. Future targets use the accepted
canonical single-camera vision calls and detached FP32 normalization. The frozen
vision module can be shared with the frozen current-feature encoder here; it is
never included in the optimizer.

The target cache retains exact FP32 features in host RAM, keyed by episode and
target frame, for one four-episode pool. It clears on pool change and process
restart. There is no persistent cache or target quantization. The enclosing run
pins source, teacher, processor and normalization identities. A restart must
recompute identical values, preserving the optimization trajectory.

The complete order needs 1,763,197 distinct target frames. At 1 MiB per paired
target, a persistent FP32 cache would need about 1.68 TiB. Training metadata bounds
the largest pool at 1,193 targets / 1,250,951,168 bytes (1.17 GiB), below the
declared 4 GiB cache cap. The pilot fills that entire pool, checks cache hits,
fresh teacher replay and empty-cache recomputation exactly, and measures
40-pair cache gathering and transfer. Runtime acceptance must include these costs.

## Recovery and pilot

The [runner](../scripts/train_jepa_cpt.py) reuses the tested Pixel optimizer,
scheduler and atomic recovery storage, with a JEPA-specific sealed run identity.
The serializer retains its historical Pixel format label; its identity separates
Pixel, JEPA pilot and JEPA production, and incompatible weights fail strict loading.

Checkpoints contain model, optimizer moments/steps, scheduler, all Python/NumPy/
CPU/CUDA RNG state, completed-update cursor, acknowledged pair count, chained
sample digest and cumulative loss. Saves occur only at complete optimizer
boundaries; three committed generations are retained. Partial/uncommitted files
cannot authorize progress, and damaged committed generations fall back to the
newest intact one. The RAM target cache is recomputed and is not checkpoint state.

The [pilot supervisor](../scripts/check_jepa_training.py) runs the same 32
four-update windows as Pixel, including the longest episode and instruction
pools. A separate process is killed after the first backward microbatch of update
index 19. A new process restores committed update 16 and must reach the same
128-update endpoint with bitwise-identical model, optimizer, scheduler, RNG,
cursor, exposure and sample chain. A copied-checkpoint corruption probe checks
fallback. Independent common-order regeneration checks full source coverage.

The persistent pilot service is capped at one hour, six CPUs and 24 GiB host
memory. Acceptance requires full-device GPU use below 85%, host RSS below
24 GiB, projected new persistent production artifacts below 2 GiB and at least
5 GiB free disk. Runtime projection charges p95 current-feature/update/loading
costs for all updates, canonical teacher cost for all distinct targets, measured
cache transfer, episode decoding and checkpoint overhead, then adds a 1.5× margin.
It must be under 60 hours to support the 72-hour cumulative active-runtime cap.
The estimate excludes later full validation.

Production is gated on this exact recipe, source, order and runtime passing the
pilot. It saves every 500 updates or 15 minutes, stops safely on nonfinite values,
integrity/resource failures or runtime exhaustion, and never accepts an incomplete
endpoint. Production launch is Step 5; this step runs only bounded train-only checks.

## Verified pilot results

The [pilot report](evaluation/week6-jepa-cpt-pilot-v1/pilot-report.json) passes all
prospective numerical and resource criteria without a recipe revision. The
host-owned service completed in 4 minutes 58.3 seconds with no service restart.
Its child processes intentionally include the recovery-test SIGKILL.

| Measurement | Result |
| --- | ---: |
| Reference full-path training | 128 logical updates; 4,694 unique training pairs; 127 decoded episodes |
| Full-pass runtime projection, including 1.5× margin | **34.44 hours** |
| Pilot throughput including cold decoding | 41.77 pairs/second |
| Full-device GPU peak across all probes | **6.47 GiB** |
| Training-process Torch allocation / reserve peak | 4.63 / 4.84 GiB |
| Maximum host RSS across training, resume and stress probes | **5.32 GiB** |
| Largest target-cache pool, fully materialized | 1,193 paired targets; **1.17 GiB** |
| Recovery checkpoint size | 54.63 MiB |
| Projected new production artifacts, including write headroom | 317.13 MiB |
| Persistent feature/target cache storage | **0** |

The teacher cache avoids repeated canonical encoding across horizons and anchors
within each pool. The pilot itself had 4,258 misses and 436 hits; full-epoch
amortization is based on exact order coverage, not extrapolated from this short
window hit rate. Cache hits, fresh teacher calls and recomputation after clearing
the cache all match exactly. Filling the largest pool took 23.37 seconds,
including source loading; p95 40-pair cache gathering/transfer was 0.01054 seconds.

The projection separately charges p95 current-feature extraction plus two-pass
updates (0.16565 seconds/update), warm loading, teacher work (0.01865 seconds/new
paired target), cache transfer, cold episode decoding and checkpoint overhead.
It is a pilot estimate for an uninterrupted complete pass; restarts can add
replayed updates and cold-cache work. The cumulative active-runtime cap remains
72 hours. Full validation needs additional time.

The 40-pair GPU comparison produced identical scalar loss, relative gradient L2
error **5.94e-7** and maximum gradient error **2.37e-8**, comfortably below the
frozen tolerances. The real one-pair tail matched exactly, with zero singleton
variance penalty. The full common-order regeneration matched every array and
verified every anchor, horizon count and logical-batch boundary.

The killed process had completed 19 updates and backpropagated 10 of 39 pairs
in its next update. A new process restored committed update 16 and replayed to
update 128. The actual endpoint checkpoint files are byte-identical; their model,
optimizer, scheduler, all RNG state, cursor, pair count, sample chain and cumulative
loss are bitwise equal. All 112 updates after restoration match the uninterrupted
trace exactly. Corrupting the two newest generations in a copied recovery
directory correctly recovered update 120. Frozen base and teacher weights remain
unchanged.

The [independent local audit](evaluation/week6-jepa-cpt-pilot-v1/local-verification.json)
loaded both actual recovery payloads, compared their values, recomputed state
digests and checked every replayed optimization record. It independently checked
source/recipe/order pins, numerical thresholds, telemetry maximum, cache bounds,
and the runtime and disk projections. It did not rerun the CUDA gradient or cache
probes. Unit tests cover grouped-gradient equivalence, partial-update rejection,
RNG recovery, cache identity/eviction and train-only access; all 471 tests pass.

Step 4 exposure totals **259 completed optimizer updates / 9,486 pair
presentations**, including replayed work. The killed partial update adds 10
backward-only presentations, and gradient equivalence adds 82 backward-only
presentations without optimizer updates. Cache stress and replay encode additional
training targets without optimizing. No validation/test payloads were read.
These pilot weights are discarded as production initializers, and Step 3's
separate diagnostic exposure remains in its own ledger.

**Step 5 is complete.** The [production record](WEEK6_JEPA_PRODUCTION.md) documents
the verified fixed endpoint: 221,633 updates / 8,221,365 presentations, unchanged
frozen weights and exact final export equality. The export remains a candidate
until Step 6 applies the unchanged held-out acceptance gate.
