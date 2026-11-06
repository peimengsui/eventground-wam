# Week 5 step 2: pixel baselines and success criteria

**Protocol:** `week5-pixel-measurement-v2`, frozen September 25, 2026 before
predictor implementation or training. Full scoring and numerical replay have
completed on Feifei. No CPT or policy training is part of this step.

The [sealed measurement protocol](../configs/evaluation/week5_pixel_baselines_v2.json)
and [executable implementation](../src/eventground_wam/pixel_baselines.py) define
the comparison. This step uses the [step 1 future-pair freeze](WEEK5_FUTURE_PAIRS.md)
without changing its assignments, horizons, camera correspondence or boundaries.

## Controls and coverage

- **Current-frame copy:** predict the image at `t` for target `t+h`, separately
  for each camera. It has no fitted parameters and ignores instruction/actions.
- **Train-only constant background:** one fixed RGB image for each of the 40
  tasks and two cameras. Average all frames within each training episode, then
  average appearance variants within each numeric trajectory group, then average
  groups within each task. Fit on 11,435 training episodes / 1,341 groups only.
  This is a training mean image, including averaged robot and object pixels;
  it is not a segmented empty-scene background.

Score every valid anchor from all 1,528 validation episodes / 180 numeric groups /
40 tasks: **1,109,994 dual-camera pairs**, or 2,219,988 individual camera targets.
Horizons are 2, 4, 8, 16 and 32 frames (0.1–1.6 seconds at 20 Hz). No validation
images contribute to the templates; no test payloads are opened. Both controls
must use exactly the same pairs and preprocessing as the learned predictor.

Decode the pinned 256×256 RGB24 images and average each exact 2×2 pixel block
to produce **128×128**, with float32 values in `[0,1]`. There is no augmentation.
This resolution is now fixed for the initial pixel comparison; training loss
weights, architecture and compute budget remain decisions for steps 3–4.

## Measurements and uncertainty

**L1** is mean absolute error across all RGB values. Lower is better.
**SSIM** uses an 11×11 Gaussian window, sigma 1.5, population covariance,
`K1=.01`, `K2=.03`, unit data range, valid image borders and equal RGB channel
weights. Moments, covariance and SSIM reductions use float64; normalized
images and L1 remain float32. Higher is better. The runtime checks the optimized PyTorch
implementation against scikit-image before proceeding. These choices follow
the [SSIM authors' reference](https://www.cns.nyu.edu/~lcv/ssim/) and
[scikit-image's documented Wang-compatible settings](https://scikit-image.org/docs/0.20.x/api/skimage.metrics.html#skimage.metrics.structural_similarity).

**Motion-sensitive L1** uses the ground-truth image-change mask
`mean_RGB(abs(target-current)) > 10/255`. An example is eligible only if at
least 1% of pixels change; otherwise its mask is all zero. Recover exact integer
2×2 RGB box sums as `round(image*1020)`: the strict threshold is exactly
`sum_RGB(abs(target_sum-current_sum)) > 120`. Eligibility uses
`changed_pixel_count * 100 >= H*W`. This prevents floating-point rounding from
changing membership for pixels exactly on the threshold. The same mask is used
for copy, background and every prediction, so models cannot choose easy pixels.
Aggregate masked error mass divided by aggregate mask area. Report the mask
coverage and the analogous error on its complement as well. This mask measures
visible image change, including wrist-camera movement, lighting and rendering
changes; it does not identify physical object motion or manipulation success.

For the primary metrics, average anchors within each episode/camera/horizon,
then variants within a numeric group, then groups within each task, then give
the 40 tasks, two cameras and five horizons equal weight. Masked metrics use
the ratio of error/area masses after this hierarchy, not an unweighted average
of tiny-region ratios. Report all ten camera/horizon cells and ordinary
pair-weighted metrics alongside the primary estimates.

Use **10,000 paired bootstrap resamples of numeric groups within each fixed
task**, seed `20260925`, with percentile 95% intervals. Reuse each draw for both
methods, all cameras and all horizons. Appearance variants and video frames
are correlated observations, not additional independent trials. These intervals
are conditional on the frozen 40 tasks and split; they do not claim generalization
to a new task population. Validation remains development evidence, not a
pristine final test after repeated model development.

## Prospective success criteria

All conditions must pass for the final pixel checkpoint. Relative gain means
`(baseline_error - predictor_error) / baseline_error`.

| Requirement | Frozen threshold |
| --- | --- |
| Overall L1 versus copy | At least **5%** lower; paired 95% lower bound on gain **> 0** |
| Motion L1 versus copy, horizons 8/16/32 | At least **10%** lower; paired 95% lower bound on gain **> 0** |
| Overall SSIM versus copy | Point delta **≥ 0**; paired 95% lower bound **≥ −0.005** |
| Overall L1 versus train-only background | Point gain and paired 95% lower bound both **> 0** |
| Each of ten camera/horizon cells | Predictor L1 **≤ 1.02 × copy L1** |
| Motion support for horizons 8/16/32 | At least **120 numeric groups** and **all 40 tasks** |

These are prospective practical-effect thresholds for this study, not universal
cutoffs claimed by a paper. Degenerate baseline denominators or inadequate motion
support produce an unassessed result, never a pass. Aggregate gains cannot hide
a camera/horizon regression beyond the 2% guardrail.

Apply this gate to the **predeclared final CPT checkpoint** under a schedule and
fixed compute budget set in step 4, not the best checkpoint selected after
inspecting these metrics. Step 6 must report failures as well as successes.
`assess_candidate` checks complete episode/camera/horizon identities, source
digests, pair counts and identical ground-truth mask areas before comparing a
predictor's `prediction` fields against the saved baseline rows.

## Fixed qualitative review

Freeze 480 examples independently of measured errors: one group and one
appearance variant per validation task, both cameras, horizons 8 and 32, and
anchors at 25%, 50% and 75% of the valid anchor range. Select groups/variants by
the SHA-256 rank specified in the protocol. The exact episode, source digest,
anchor and target identities are retained in the
[frozen panel](../configs/evaluation/week5_pixel_qualitative_v2.json).

The baseline run renders eight deterministically selected task sheets, each
showing the middle anchor for both horizons and cameras: current/copy, fitted
constant, target and image-change mask. Step 6 must add predictions to the same
fixed examples and review all 40 tasks; additional failures may be shown with
their selection rule explicitly described.

## Execution and retained evidence

The runner revalidates the frozen pair bundle, rehashes each selected camera
video against its source record, and fully decodes training and validation
videos while fitting/scoring. The test exclusion check runs before file access.
Scoring reuses per-frame SSIM moments across overlapping horizons and is tested
against direct pair-by-pair computation. It fits no learned model or optimizer.

Feifei artifact root:

```text
/home/peimengsui/data/eventground-wam/libero-plus/derived/pixel-baselines/week5-pixel-baselines-v2
```

The source archive and `job.log` are retained under
`/home/peimengsui/data/eventground-wam/analysis/week5-pixel-baselines-v2`.
The host-owned service is `eventground-pixel-baselines-v2.service`, with a
six-CPU quota, 16 GiB memory limit and the existing pinned B0 runtime. CUDA
accelerates image metrics only; no policy/model checkpoint is loaded.

Artifacts include the protocol/source identity, 80 templates and their train-only
fit manifest, sealed per-episode measurements, a complete `episode-metrics.jsonl`,
group-weighted reports/intervals, fixed qualitative identities, eight preview
sheets, and a completion receipt. Cached templates and episode results can be
resumed only with matching code/config/source/template hashes. Completed output
is immutable. A run is complete only when `completion.json` exists and is verified.

Run from the retained source tree with its `src` on `PYTHONPATH`, using the
existing `/home/peimengsui/code/eventground-wam-b0-oft-v1/environments/b0/.venv/bin/python`:

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$BASELINE_PYTHON" \
  scripts/run_pixel_baselines.py --pairs "$FROZEN_PAIRS" --output "$NEW_OUTPUT"
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$BASELINE_PYTHON" \
  scripts/validate_pixel_baselines.py --pairs "$FROZEN_PAIRS" \
  --output "$NEW_OUTPUT" --receipt "$NEW_RECEIPT"
```

Here `FROZEN_PAIRS` is the step 1 `primary` bundle; `NEW_OUTPUT` and `NEW_RECEIPT`
must be new paths. The second command independently audits identities, hashes,
all 15,280 measurement rows, train/template counts and primary point estimates.
It also replays the eight preview episodes on CPU and checks 64 real-image
SSIM scores against scikit-image. It does not alter the completed baseline run.

## Corrections before acceptance

The initial v1 scoring run saved all measurements but its final receipt rejected
the `TorchVersion` string subclass. Receipt recovery preserved that source and
report; the runner now converts runtime versions to plain strings. Independent
CPU replay then identified up to 0.000528 difference in mask area at the exact
threshold. [V1 is explicitly superseded](evaluation/week5-pixel-baselines-v1-superseded.json),
not accepted as the baseline. V2 uses exact integer threshold/area decisions
and float64 SSIM. The numerical success thresholds and qualitative example
identities did not change, and no predictor has trained.

V2 reuses the exact 80 train-only templates from v1 by their pinned manifest and
array hashes; their fitting rule, input images and 128×128 preprocessing are
unaffected by metric arithmetic. It rescores every validation pair. The original
source, service failure and artifacts remain available for audit. The standalone
validator also received a receipt-only fix to convert NumPy scalars to plain
Python floats; the final audit receipt pins that validator separately from the
original deployment archive.

## Results

Verification passed: **358 local tests** (30 measurement tests), lint, every
artifact/row identity check, and local recomputation of all reported bootstrap
intervals. Eight complete episodes replayed on CPU with maximum metric
difference **7.89×10⁻⁹**; 64 real-image SSIM checks matched scikit-image within
**7.83×10⁻¹⁵**. The v2 service exited successfully with zero restarts. Its validation
run took 116.2 seconds, reusing the already fitted training templates.

Full validation scoring completed on Feifei. Primary estimates below include
95% task-stratified numeric-group bootstrap intervals.

| Control | Overall L1 ↓ | Overall SSIM ↑ | Motion L1, h=8/16/32 ↓ |
| --- | ---: | ---: | ---: |
| Current-frame copy | 0.058562 [0.057449, 0.059677] | 0.720547 [0.717048, 0.723992] | 0.218488 [0.216661, 0.220329] |
| Train-only constant | 0.150935 [0.149016, 0.152840] | 0.582252 [0.578271, 0.586108] | 0.175089 [0.173750, 0.176431] |

| Camera | Horizon | Copy L1 | Copy SSIM | Constant L1 | Constant SSIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| front | 2 | 0.026496 | 0.890894 | 0.140956 | 0.591848 |
| front | 4 | 0.031849 | 0.862083 | 0.141004 | 0.591452 |
| front | 8 | 0.037749 | 0.836037 | 0.141096 | 0.590692 |
| front | 16 | 0.043392 | 0.811963 | 0.141269 | 0.589383 |
| front | 32 | 0.048368 | 0.788492 | 0.141340 | 0.588380 |
| wrist | 2 | 0.039400 | 0.720005 | 0.161367 | 0.573108 |
| wrist | 4 | 0.056989 | 0.653043 | 0.161197 | 0.573488 |
| wrist | 8 | 0.078827 | 0.595421 | 0.160861 | 0.574379 |
| wrist | 16 | 0.100416 | 0.548493 | 0.160284 | 0.575899 |
| wrist | 32 | 0.122135 | 0.499041 | 0.159981 | 0.573890 |

The 5%/10% point-gain requirements correspond to overall L1 about **0.055634**
and medium/long motion L1 about **0.196639**, respectively. These rounded
values are explanatory; the exact relative-gain formulas, paired intervals and
all other guardrails determine acceptance.

All **180 numeric groups / 40 tasks** provide eligible motion pixels.
Their mean area is **30.25%** over horizons 8/16/32.
The constant template has lower error on changing regions despite much worse
whole-image L1 and SSIM. This is why motion error alone cannot establish future
prediction ability; the frozen gate requires the joint whole-image, motion and
SSIM conditions.

Contact-sheet review covered all eight fixed task sheets, with a detailed check
of episode 3527. They show matching camera/target columns, averaged templates,
and masks affected by wrist-camera movement and visual corruption as well as
robot/object changes. Step 6 still needs learned predictions and the full
40-task qualitative review.

[Full metrics and intervals](evaluation/week5-pixel-baselines-report-v2.json),
[runtime receipt](evaluation/week5-pixel-baselines-run-completion-v2.json),
[independent audit](evaluation/week5-pixel-baselines-verification-v2.json), and
[local interval recomputation](evaluation/week5-pixel-baselines-local-verification-v2.json)
retain the numerical evidence. The [step completion ledger](evaluation/week5-step2-completion-v1.json)
pins all accepted artifacts and the superseded history.

**No predictor gate has been assessed and no training has launched.** Next is
step 3: implement the action/horizon-conditioned predictor against this frozen
measurement contract; step 4 will freeze its learning recipe and compute budget.
