# Week 6 Step 1 — frozen visual target definition

**Status: complete.** `week6-jepa-targets-v2` is accepted after the training-only
Feifei feasibility check and independent numerical verification. All 600 pairs
passed the fixed engineering criteria. No JEPA predictor or action policy was
trained. The earlier batched v1 failure is retained below.

## Target choice

Use the existing pinned **Qwen/Qwen3.5-2B vision tower**, at revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. Select the output of the pretrained
spatial merger after the final vision block, before the language model. In the
locked Transformers 5.17.0 runtime this is `model.visual(...).pooler_output`.
It supplies spatial tokens, despite the output field's name.

This choice reuses the verified model source and runtime and avoids adding a
second pretrained encoder to the main comparison. DINO is deferred unless this
target fails the predeclared feasibility check. These features are an engineering
choice to be tested, not evidence that Qwen is better than DINO for dynamics.

The [sealed contract](../configs/training/week6_jepa_targets_v2.json) pins the
encoder, processing, normalization, inherited inputs and diagnostic criteria.
The [extractor](../src/eventground_wam/jepa_features.py) accepts images only.

## Exact representation

| Item | Definition |
| --- | --- |
| Source | RGB frame `t+h` from the same frozen episode and camera; h=2/4/8/16/32 |
| Input | Native decoded uint8 RGB, two 256 x 256 images, front then wrist |
| Processing | Pinned torchvision image processor; no resize, crop or augmentation; divide by 255, mean/std 0.5 |
| Patches | 16 x 16 patches; duplicate the same still image over the temporal patch size of two |
| Image grid | `[1,16,16]` before the 2 x 2 spatial merger |
| Encoder | All 24 vision blocks plus the pretrained merger, BF16/SDPA, frozen and in evaluation mode |
| Execution shape | Exactly one camera image per vision forward; outer loader batching cannot alter the target |
| Output per camera | 8 x 8 row-major tokens, 2,048 channels each |
| Output per pair | `[128,2048]`, front tokens 0–63 then wrist tokens 64–127 |
| Normalization | FP32 per-token channel LayerNorm without affine parameters (epsilon 1e-5), then unit L2 (epsilon 1e-6) |
| Gradient | Detached target; no target optimization, EMA, learned target projection or fitted normalization statistics |

The same image is repeated inside each temporal patch; the encoder does not see
the previous frame, a future clip or any other temporal observation. It receives
no instruction, state, action, horizon or sample identity. The current-frame
copy-latent control will use this identical transform on frame `t`.

Teacher execution is independent of the language model and policy adapters. No
B0-trained, Pixel-trained or diagnostic weights initialize the teacher. Later
policy training must preserve this separate frozen teacher when shared policy
features become trainable.

The internal future core remains planned at **128 x 256**. Step 3 will add a
separately reported trainable 256-to-2048 prediction projection; the teacher's
target space remains fixed. Loss weights and variance regularization are not
frozen by this step. Unit-vector coordinate scale must be considered when
defining variance thresholds in Steps 2–4.

## Comparison with Pixel

Current-observation conditioning remains the Week 5 `FutureInputBuilder` and
frozen base-Qwen final hidden states, including instruction, plus current state,
ordered intervening actions and horizon. Only the future supervision changes.
The exact future-pair manifest, common order, five horizons, camera identities
and one-pass exposure are inherited without resampling.

Pixel reconstructs 128 x 128 box-reduced RGB. This teacher encodes native
256 x 256 RGB, matching the pinned vision processor and giving an 8 x 8 merged
grid. Target resolution and pretrained feature geometry are declared method
differences; this is not an isolated comparison of loss functions. Future
Pixel/JEPA diagnostics must state their comparison space. Raw RGB loss and
latent cosine loss cannot be ranked against each other.

## Training-only feasibility check

The [diagnostic command](../scripts/check_jepa_targets.py) performs no optimization.
Before any learned-predictor evaluation it selects one training episode per
task by a fixed hash, then three distinct anchors at quarter/half/three-quarter
positions of the h=32-valid range. All five horizons yield **40 tasks, 40 episodes,
120 anchors and 600 pairs**. Selection and pair identities are retained.
No validation or test payloads are read.

Required checks, fixed before the run:

- Verify model/processor files against the existing immutable model manifest;
  verify selected source payloads and pairs against the frozen index.
- Prove camera and row-major patch alignment with a synthetic color grid.
- Require finite unit targets (norm error at most 1e-6), exact repeated encoding,
  and at most 0.002 maximum absolute normalized-feature difference under batch
  composition and camera permutation changes.
- Require unchanged teacher tensor hashes, zero trainable parameters and zero
  teacher gradients.
- In each camera, require pooled across-observation effective rank at least 5
  and fixed-position across-observation standard-deviation RMS at least 1e-4.
  Report within-image spatial diversity separately so it cannot conceal a
  constant representation across observations.
- Require mean h=32 copy-to-future cosine distance at least 1e-4 in both cameras;
  report the remaining horizons without assuming monotonic change.

These are teacher feasibility checks, not the Week 6 predictor success gate.
The 120 anchors are correlated within episodes; their statistics are descriptive,
not independent-trial confidence intervals. Step 2 still defines held-out
controls, retrieval candidates and prospective predictor/collapse thresholds.

The run retains a normalized feature bank, first-observation replay inputs,
per-pair metrics, rank spectra, model/source/runtime identities and a sealed
report. Production feature caching remains a Step 4 decision. Any future cache
must bind source frame/camera, teacher weights, processing, normalization and
runtime identities; the diagnostic bank is not a training cache.

## Verified results and retained failure

The [v2 report](evaluation/week6-jepa-targets-v2/report.json) covers 40 training
tasks/episodes, 120 anchors and 600 pairs, with 720 dual-camera observations
encoded. All ten checks passed; the service exited zero. The accepted target
encoder has 331,416,576 frozen parameters and zero trainable parameters.

| Check | Front | Wrist |
| --- | ---: | ---: |
| Pooled across-observation effective rank (minimum 5) | 18.6450 | 33.4293 |
| Fixed-position across-observation std RMS (minimum 0.0001) | 0.014619 | 0.014708 |
| Mean h=32 copy-to-future cosine distance (minimum 0.0001) | 0.244228 | 0.395185 |

Repeated encoding, outer-batch changes and camera permutation all have **zero
maximum feature difference**. Maximum unit-norm error is 1.79e-7. Vision tensor
hashes before and after the run are identical. The run took 24.09 seconds,
including setup, verification and artifact writing; peak PyTorch GPU allocated
memory was 0.69 GiB. This is a target-only diagnostic, not a production CPT
resource estimate or full-device memory measurement.

The [independent verifier](evaluation/week6-jepa-targets-v2/independent-verification.json)
recomputed per-pair and camera/horizon cosine measurements, non-affine
normalization, unit norms and across-observation diversity from the saved
feature bank using NumPy. Effective rank was recomputed from a Gram-matrix
eigendecomposition rather than the diagnostic's Torch SVD. This is independent
arithmetic verification, not a second full image-encoding replay.

The initial [v1 report](evaluation/week6-jepa-targets-v1/report.json) **failed**
batch composition: maximum normalized-feature difference 0.162676 against the
unchanged 0.002 tolerance. A [follow-up investigation](evaluation/week6-jepa-targets-v1/batching-investigation.json)
reproduced its saved raw output exactly and found identical preprocessed pixels
but different vision outputs under batching. This localizes the issue to vision
execution; it does not establish a specific kernel cause or cross-image attention.
V2 makes each camera image a separate canonical vision call. No threshold,
sample selection, target layer or normalization was relaxed. The existing Pixel
conditioning path and accepted Week 5 artifacts remain unchanged.

Two earlier preflight attempts stopped before model/data access because the
diagnostic initially conflated package and CUDA build version strings, then
attempted to seal a `TorchVersion` subclass. Their source snapshots and logs are
retained; the corrected plain-string receipt has a regression test. User approval
resolved the initial automatic-review block on the source transfer.

All **426 local tests** and Ruff on the four added Python files plus their tests
pass. Twelve focused tests cover layout, freezing, invalid inputs, adaptation
rejection, stop-gradient, normalization, cross-example collapse, version receipts
and enforced single-image execution.

The accepted 57-file source archive and large diagnostic tensors remain on
Feifei under `/home/peimengsui/data/eventground-wam/analysis/week6-jepa-targets-v1/`:
`source-v2/`, `diagnostic-v2/`, and `independent-verification-v2.json`. The parent
directory also preserves v1 and both preflight attempts. Compact source, launch,
selection, report and audit records are retained in
[`evaluation/week6-jepa-targets-v2/`](evaluation/week6-jepa-targets-v2/).
The [completion ledger](evaluation/week6-step1-completion-v1.json) pins the
accepted contract and evidence. Step 2 can now establish held-out baselines and
numerical predictor acceptance criteria.

Upstream reference: [Transformers Qwen3.5 vision API](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5).
The installed 5.17.0 source and checkpoint hashes, rather than a changing online
API page, govern the executed target contract.
