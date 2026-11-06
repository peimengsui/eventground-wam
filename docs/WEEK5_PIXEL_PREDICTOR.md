# Week 5 step 3: action- and horizon-conditioned pixel predictor

**Status: complete.** Architecture: `pixel-action-horizon-transformer-v2`.
Local tests and the bounded Feifei implementation checks pass.
This step builds a reusable future module. Production CPT, optimizer recovery,
full validation acceptance and action-policy training remain later steps.

The [implementation recipe](../configs/training/week5_pixel_predictor_v3.json)
pins the architecture, base model, preprocessing, train-only normalizers and
bounded diagnostic checks. It preserves the [Step 2 measurement contract](WEEK5_PIXEL_BASELINES.md).

## Inputs, representations and outputs

The model receives current front/wrist images, the instruction through base-Qwen
features, current proprioception, the ordered action sequence `[t,t+h)`, and one
of the five frozen horizons 2/4/8/16/32. Its only image targets are the two frames
at `t+h`, resized with the exact Step 2 2×2 box average to 128×128.

1. **Frozen features:** the pinned base Qwen3.5-2B encodes current 256×256
   front/wrist images and the instruction. Use all non-padding final hidden
   states, dimension 2048, bounded at 512 tokens. The feature prompt asks for a
   current-scene representation for action consequences. No future image,
   action sequence, horizon, B0-trained checkpoint, LoRA or action slots enter
   this feature extractor. The current instruction remains part of the inputs.
2. **Current spatial queries:** a trainable 16×16-stride convolution maps each
   128×128 current image into an 8×8 patch grid. Add learned spatial and camera
   embeddings, producing 128 queries of width 256 across both cameras.
3. **Action/time conditioning:** embed each normalized seven-dimensional action
   separately and add its ordered position embedding. Supply up to 32 actions
   with an exact prefix mask derived from `h`; padded suffix values cannot
   influence the model. A second MLP maps the zero-padded, ordered 32×7 action
   slots through 224→256→256 and adds an action-sequence vector to every query.
   This directly represents order even when sequences have identical action
   multisets. A two-layer MLP embeds `[h/32,h/20]` and adds the horizon
   signal to every query. Current normalized eight-dimensional state is another
   memory token. Qwen features, state and actions have separate memory-type
   embeddings.
4. **Future core:** three pre-normalized transformer decoder blocks, eight
   attention heads, width 256 and feedforward width 1024. Queries self-attend
   across both cameras and cross-attend to the conditioned memory. Dropout is
   zero. The final layer normalization yields **128×256 future tokens**.
5. **Pixel decoder:** reshape the first 64 tokens into the front grid and the
   next 64 into the wrist grid, both in row-major order. Share four bilinear
   upsampling/3×3-convolution/GroupNorm/GELU stages between cameras, with channel
   widths 128/64/32/16. A final three-channel tanh residual is added to the
   current image and clipped to `[0,1]`.

The residual output layer starts at zero, so the initial model is **exactly the
current-frame copy baseline**. The first update activates that output layer;
subsequent updates reach the future core. This gives the predictor a direct way
to preserve unchanged visual details while learning future changes.

| Parameter component | Trainable parameters |
| --- | ---: |
| Future core, projections, image patch queries and embeddings | 4,238,592 |
| Pixel decoder | 392,835 |
| Total future module | **4,631,427** |
| Base Qwen | **0** |

Report the image decoder separately when designing the matched JEPA predictor.
The Step 3 checkpoint interface records architecture, seed, metadata and an exact
tensor-state hash. It is a weight snapshot with an exact reload check; a full
optimizer/RNG/sample-cursor recovery checkpoint still belongs to Step 4.

## Target isolation and source alignment

`FuturePairDataset` validates the approved manifest and selected binary/source
file hashes, then checks each sampled pair's digest, source, ordinal, split and
bounds. It verifies all three selected episode payloads before decoding, checks
telemetry frame/time/task alignment, and retains at most one decoded episode by
default. This bounds the loader cache; throughput and any larger production
cache require profiling in Step 4.

`FutureObservation` contains current data and intervening actions only.
`FutureTrainingSample` stores targets separately, and collation produces a
`PixelTrainingBatch` with distinct `condition` and `targets`. The predictor
accepts only `PixelCondition`; its forward signature has no future-target input.
Sample IDs, target indices and task-group metadata are not passed to the model.

Current state uses the exact frozen train-only q01/q99 transform from the B0
recipe. Actions use the same train-only affine action statistics and preserve
the gripper channel. No statistics are fitted on validation. Test split requests
are rejected before any filesystem access. The frame at `t+h` supplies the
image target, while its action and state cannot enter conditioning.

The frozen feature extractor runs in evaluation mode under `no_grad`, and
rejects adapted backbone parameter names. The predictor itself does not detach
its input features: a later auxiliary VLA trainer can propagate future-loss
gradients into trainable shared representations. That later trainer must define
the common OFT-compatible feature route for Pixel and JEPA explicitly.

## Bounded implementation checks

The diagnostic recipe declares its sample selection and fixed update counts
before its final run, using L1 only, AdamW with learning rate 0.0001, no weight decay,
and global gradient clipping at 1. The loss helper also supports differentiable
L1 plus SSIM; **production loss weights are not frozen by this diagnostic**.

- **Controlled conditioning:** four examples share identical current images,
  Qwen-feature placeholders and state. For each horizon, the action multisets
  are identical but their order differs. Targets depend on action order and
  horizon. After 1,600 updates, require at least 80% L1 reduction, a reversal
  error increase above 0.02, and a visible effect from removing the explicit
  horizon input. There are no sample identifiers or varying observations that
  could let the model distinguish these cases without conditioning.
- **Real-data overfit:** select four training tasks and one episode per task by
  a fixed SHA-256 ranking. Use a common middle anchor with valid h=32 and all
  five horizons, giving 20 pairs. Verify and load the actual base model, encode
  four current observations, and train a fresh predictor for 300 updates.
  Require at least 20% L1 reduction, nonzero gradients in every conditioning
  branch, action/horizon sensitivity and exact weight reload.
- **Frozen-base check:** hash all base-core state tensors before and after
  feature extraction, require no trainable base parameters or gradients, and
  require repeated feature extraction to be identical. The model files and
  processor are checked against the pinned model-source manifest.
- **Ablations:** report zeroed actions, reversed valid actions, removed explicit
  horizon signal and zeroed Qwen features. Action masks retain the real horizon
  during these diagnostic ablations, so removing the explicit horizon branch
  does not remove the implicit duration information in action count.
- **Unit and contract checks:** verify camera order, target and action boundaries,
  padded-tail exclusion, target/future-state mutation isolation, feature gradients,
  loss gradients, invalid inputs, adaptation rejection and checkpoint corruption.

Real overfit and ablation results are implementation evidence on the training
examples. They do not establish causal robot dynamics, instruction-following
generalization, the Step 2 held-out gate, or improved VLA task success.

The first two diagnostic recipes failed the controlled check and are retained.
Architecture v1 supplied action order only through additive action-position
embeddings; its 400-step check reduced L1 by only 4.1%. Architecture v2 adds
the explicit fixed-slot sequence projection; at the same settings it improved
L1 by only 8.1%. Synthetic-only debugging compared learning rates 0.0001 and
0.0003 for up to 1,600 steps. The slower setting learned the conditional targets,
so recipe v3 uses it with 1,600 synthetic updates. All outcome thresholds remain
unchanged. This is implementation/optimizer debugging, not held-out model
selection; no real data were accessed in either rejected diagnostic run.

## VLA handoff and remaining work

`PixelPrediction` exposes both the reconstructed images and the 128 future
tokens. The RGB decoder supplies reconstruction supervision; the token interface
is available for the later auxiliary and coupled experiments. This step makes
no change to the existing OFT action policy, whose future-token coupling remains
disabled. M1/M3 will share the same eventual accepted `C_pixel` checkpoint.

The diagnostic weights are explicitly named `diagnostic-pixel.pt` and marked
`step3-diagnostic-only-not-C_pixel`. They must not be mistaken for the Week 5
production deliverable or a checkpoint selected on the held-out success gate.

Step 4 must profile the complete training path, choose matched Pixel/JEPA
sampling and exposure budgets, freeze loss weights and stopping rules, implement
full recovery, and verify interrupted-process resume. Later coupling must also
define how candidate actions are obtained without demonstration actions at
deployment. The two-epoch action-training comparison remains unchanged.

## Artifacts and execution

```text
Feifei source/log root:
/home/peimengsui/data/eventground-wam/analysis/week5-pixel-predictor-v3

Diagnostic outputs:
/home/peimengsui/data/eventground-wam/analysis/week5-pixel-predictor-v3/diagnostic
```

Run `scripts/check_pixel_predictor.py` from the retained source tree with
`PYTHONPATH=src`, the pinned B0 runtime, the Step 1 `primary` pair bundle, and a
new `--output` path. The host-owned diagnostic service runs one GPU job, has a
six-CPU quota and 24 GiB host-memory cap, and cannot start production CPT.
It retains per-phase results, source/sample identities, the feature receipt and
small feature cache, diagnostic weights, train-example sheets and a final report.

The [completion ledger](evaluation/week5-step3-completion-v1.json) pins the
source, recipes, retained failures and [accepted diagnostic report](evaluation/week5-pixel-predictor-v3/report.json).
Local lint and all **389 tests** pass, including 31 new predictor/data tests.

| Check | Copy / initial | Learned predictor | Outcome |
| --- | ---: | ---: | --- |
| Controlled L1 | 0.087500 | 0.001334 | **98.48% reduction** |
| Controlled reversed-action L1 | — | 0.133444 | Order reversal strongly worsens reconstruction |
| Controlled zero-horizon prediction change | — | 0.012762 | Explicit horizon branch affects outputs |
| Real 20-pair train L1 | 0.041386 | 0.029697 | **28.24% reduction** |
| Real train SSIM | 0.757876 | 0.775640 | Increased |
| Real train motion L1 | 0.190538 | 0.126048 | Reduced |

On the real training examples, zeroing actions raises L1 to 0.034208, reversing
them to 0.032592, zeroing the explicit horizon signal to 0.037772, and zeroing
Qwen features to 0.059372. These are in-batch diagnostic ablations, not held-out
performance estimates. Every conditioning branch receives gradients.

The base-core tensor hash is unchanged; repeated base features and restored
predictor outputs each have maximum difference **0**. Real observations use
178–184 base tokens. Only four training episodes were decoded; no validation
or test payloads were read. The service exited successfully with no restarts.
An [independent CPU replay](evaluation/week5-pixel-predictor-v3/independent-verification.json)
reconstructed all 20 samples and reloaded the checkpoint; its L1, SSIM and motion
L1 values match the GPU report within **4.52×10⁻⁸**. Feature-cache values are
identical across horizons for each common current observation.
Its cached-feature predictor stage peaked at about **1.07 GiB allocated /
1.39 GiB reserved GPU memory**; this excludes the earlier base extraction peak
and is not a production throughput/resource estimate.

All four two-camera example sheets were reviewed at h=8 and h=32. Views and
targets align; static backgrounds are preserved, while moving arms and wrist
views still show blur, ghosting and missed large motion. This is sufficient
for the bounded learning check and **not** evidence that the predictor is ready
for use or has passed the Step 2 generalization gate.

The retained implementation-debug history includes 5,600 synthetic updates
(22,400 example presentations) across the failed, exploratory and final checks.
The only real-data diagnostic training was 300 updates on 20 pairs, or 6,000
training-pair presentations. These exposures are recorded separately from the
future production CPT budget; the diagnostic checkpoint is not its initializer.
