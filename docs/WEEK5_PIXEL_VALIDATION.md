# Week 5 Step 6: final predictor validation

**Status: complete, independently verified, and accepted.** `C_pixel-v1` passes
every frozen numerical criterion on all 1,109,994 validation pairs. All 480 fixed
examples were visually reviewed. The accepted checkpoint is byte-identical to
the prescribed final update, 221,633; there was no checkpoint search or extra
training. See the [acceptance manifest](evaluation/week5-pixel-validation-v1/acceptance.json)
and [Step 6 completion ledger](evaluation/week5-step6-completion-v1.json).

## Results

Primary estimates use the frozen hierarchical task/group weighting. Intervals
below are paired 95% group-bootstrap intervals for the improvement.

| Criterion | Copy → predictor | Improvement [paired 95%] | Required | Result |
| --- | --- | --- | --- | --- |
| Overall L1 ↓ | 0.058562 → 0.039078 | 33.27% [32.81%, 33.74%] | ≥5%; lower bound >0 | PASS |
| Motion L1, h=8/16/32 ↓ | 0.218488 → 0.120079 | 45.04% [44.59%, 45.51%] | ≥10%; lower bound >0 | PASS |
| Overall SSIM ↑ | 0.720547 → 0.757195 | +0.036648 [+0.035917, +0.037369] | Point ≥0; lower bound ≥−0.005 | PASS |
| Overall L1 vs constant ↓ | 0.150935 → 0.039078 | 74.11% [73.52%, 74.69%] | Point and lower bound >0 | PASS |
| Camera/horizon guardrail | All 10 cells improve | L1 reductions 18.29%–41.87% | No cell worse than 1.02× copy | PASS |
| Motion support | 180 groups / 40 tasks | All validation groups supported | ≥120 groups / all 40 tasks | PASS |

The predictor's overall L1 interval is [0.038270, 0.039892], SSIM interval is
[0.753775, 0.760566], and h=8/16/32 motion L1 interval is [0.118956, 0.121176].
Ordinary pair-weighted diagnostics are L1 0.038280 and SSIM 0.761511; they do
not determine acceptance. The [sealed report](evaluation/week5-pixel-validation-v1/report.json)
contains unrounded estimates and all intervals.

| Camera | Horizon | Copy L1 | Predictor L1 | L1 improvement | Predictor SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Front | 2 | 0.026496 | 0.021651 | 18.29% | 0.906703 |
| Front | 4 | 0.031849 | 0.024641 | 22.63% | 0.885965 |
| Front | 8 | 0.037749 | 0.028153 | 25.42% | 0.863216 |
| Front | 16 | 0.043392 | 0.032002 | 26.25% | 0.839279 |
| Front | 32 | 0.048368 | 0.035658 | 26.28% | 0.815277 |
| Wrist | 2 | 0.039400 | 0.030341 | 22.99% | 0.763326 |
| Wrist | 4 | 0.056989 | 0.038460 | 32.51% | 0.705207 |
| Wrist | 8 | 0.078827 | 0.048644 | 38.29% | 0.646061 |
| Wrist | 16 | 0.100416 | 0.060240 | 40.01% | 0.594668 |
| Wrist | 32 | 0.122135 | 0.070992 | 41.87% | 0.552245 |

[Validation comparison chart](../outputs/plots/week5-pixel-validation-v1/validation-summary.png)
· [final training loss curve](../outputs/plots/week5-pixel-cpt-v1-loss-final/loss-curve.png)
· [all 40 fixed prediction sheets and review notes](../outputs/week5-pixel-validation-v1/gallery.md).

## Qualitative findings and limits

The [sealed visual review](evaluation/week5-pixel-validation-v1/visual-review.json)
records every fixed example, image hash and task-level observation. Scene
appearance, backgrounds and many already-visible carried objects are preserved;
small-motion predictions are often close to their targets. Large object/camera
displacements are commonly underestimated, particularly at h=32. Duplicated or
translucent edges, bright rims and blurred newly visible regions remain. One
fixed stove example also misses an observable state transition.

Some input and target front-camera images already contain blur, haze or other
corruption. These are distinguished from predictor-added artifacts. The
truth-defined change mask includes appearance and camera changes; it is not a
physical object segmentation. On the mask complement, overall static-region L1
increases from copy's 0.007604 to 0.011835, despite the large net reconstruction
gain. This diagnostic regression and the visual limitations remain part of the
accepted record. The model is accepted against the predeclared reconstruction
gate, not certified as a reliable simulator or an improved action policy.

## Fixed candidate and measurement

The candidate is the exact export from the [completed production run](WEEK5_PIXEL_PRODUCTION.md):
`C_pixel-candidate.pt`, SHA-256
`2ddd8bcb825f6a2905c4036a0d0e4b60e973f8f0fbe32066989eaabd4f076eaf`.
The accepted copy preserves all 18,557,171 bytes, including the original
candidate-tagged header; its separate acceptance manifest records the passed
gate without modifying model weights or metadata.
The [final checkpoint verification](evaluation/week5-pixel-cpt-final-v1/final-checkpoint-verification.json)
checked complete recovery state, optimizer/scheduler counters, candidate equality
and the full 8,221,365-pair exposure chain. Base Qwen remained frozen.

The [Step 2 measurement protocol](WEEK5_PIXEL_BASELINES.md) applies unchanged:

- All 1,109,994 validation pairs from 1,528 episodes, 180 numeric groups and 40
  tasks; both front and wrist cameras, horizons 2/4/8/16/32 at 20 Hz.
- RGB at 128×128 using the frozen 2×2 box reduction. L1 and SSIM use the exact
  baseline definitions; the change mask is defined by current and target images.
- Equal-anchor means within episode, equal appearance variants within numeric
  group, equal groups within task, then equal tasks, cameras and horizons.
- Paired uncertainty from 10,000 bootstrap resamples of numeric groups within
  task, seed 20260925. Correlated frames are not treated as independent trials.
- The same 480 preselected examples: one episode per task, front/wrist views,
  horizons 8 and 32, and three fixed anchor fractions.

The learned predictor receives current images, instruction, state, intervening
demonstration actions and horizon. Future images are used only as scoring targets.
Demonstration future actions are available in this CPT evaluation; deployment
action generation and VLA coupling remain separate later experiments.

## Prospective acceptance gate

All of the following are required:

| Check | Frozen requirement |
| --- | --- |
| Overall L1 vs copy | At least 5% relative improvement; paired 95% lower bound above zero |
| Motion L1 vs copy at h=8/16/32 | At least 10% relative improvement; paired 95% lower bound above zero |
| Overall SSIM vs copy | Point delta at least zero; paired 95% lower bound at least −0.005 |
| Overall L1 vs train-only constant | Positive improvement and paired 95% lower bound above zero |
| Camera/horizon guardrail | Predictor L1 at most 1.02× copy in every one of the 10 cells |
| Motion support | At least 120 numeric groups and all 40 tasks |
| Qualitative review | Inspect every one of the fixed 480 examples and report limitations |

The final candidate is fixed before validation. An unmet gate is reported as a
failure; the protocol does not select another checkpoint, extend the epoch or
change thresholds after viewing results.

## Execution and independent verification

The scoring service uses an immutable 65-file source snapshot. All 56 files from
the production training snapshot are unchanged. New evaluation code adds
complete per-episode coverage checks, atomic receipts and prediction panels.
The [launch receipt](evaluation/week5-pixel-validation-v1/launch.json) and
[source inventory](evaluation/week5-pixel-validation-v1/source-files.json) pin
the exact files and service. Source archive SHA-256:
`adcca9f3596feaf1c413c646765ce1ddecb4acbb11da7fc1e21914336cf6a430`.

Scoring runs with frozen weights and no optimizer, eight anchors per batch,
feature microbatches of four and predictor microbatches of ten. Each committed
episode contains every valid anchor at every horizon. Resuming verifies the
same identity and completed receipts before skipping work.

The independent auditor is staged outside the immutable scoring source. It
recomputes nested means and bootstrap intervals using explicit sampled arrays,
checks all episode receipts and saved prediction hashes, compares all 480 fixed
images with independent NumPy error/mask calculations and scikit-image SSIM,
and replays 96 predictions from eight predeclared episodes in a fresh process.
The replay uses the original batch boundaries. The accepted replay tolerance is
maximum absolute pixel error ≤1e−6.

The [independent audit](evaluation/week5-pixel-validation-v1/independent-verification.json)
passed all 15,280 episode/camera/horizon rows, every completed episode receipt,
all 10 cell summaries and confidence intervals, and the full paired bootstrap.
Across 480 saved images, maximum NumPy metric discrepancy was 4.11×10⁻⁸ and
maximum scikit-image SSIM discrepancy was 9.44×10⁻¹⁵. All 96 fresh-process
prediction replays were **exactly equal** (maximum pixel error 0).

The [service completion](evaluation/week5-pixel-validation-v1/host-completion.json)
records successful exit at **12:37:16 PDT on September 26, 2026**, after
**84.25 minutes**, with zero restarts. Peak CUDA allocation/reservation was
**4.54/5.02 GiB**, and peak process RSS was **5.45 GiB**. Predictor and base weights
remained unchanged; validation performed zero optimizer updates and read zero
test payloads. The scoring report retains its original pending qualitative flag;
the separate completed visual review and acceptance manifest close that stage.

Local verification passed 85 evaluation-related tests plus the independent
auditor test. These cover exact reproduction of baseline metrics, batch/order
invariance, oracle predictions, duplicate/missing/out-of-split rejection,
bootstrap agreement and detection of altered intervals/cell summaries. See the
[local checks](evaluation/week5-pixel-validation-v1/local-checks.json).

## Artifact locations

On Feifei:

```text
Accepted checkpoint and acceptance manifest (read-only):
/home/peimengsui/data/eventground-wam/libero-plus/checkpoints/week5/C_pixel-v1.pt
/home/peimengsui/data/eventground-wam/libero-plus/checkpoints/week5/C_pixel-v1.acceptance.json

Candidate:
/home/peimengsui/data/eventground-wam/libero-plus/training/week5-pixel-cpt-v1/C_pixel-candidate.pt

Immutable scoring source and control receipts:
/home/peimengsui/data/eventground-wam/analysis/week5-pixel-validation-v1

Evaluation report, episode metrics, receipts and all fixed prediction panels:
/home/peimengsui/data/eventground-wam/libero-plus/evaluation/week5-pixel-validation-v1
```

The frozen training recipe and the full validation artifacts are retained even
if the predictor misses its acceptance targets. Reconstruction measurements do
not establish downstream action-policy improvement.
