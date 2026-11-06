# Week 6 Step 2 — latent controls, retrieval and acceptance criteria

**Status:** complete, September 26, 2026 (PDT). Both controls were evaluated on
all 1,109,994 validation pairs; independent verification passed. The protocol
and implementation were frozen before validation scoring or predictor training.
Predictor training has not started; these measurements do not accept a predictor.
Protocol: `week6-jepa-measurement-v1` in the
[sealed configuration](../configs/evaluation/week6_jepa_baselines_v1.json).

## Measured results

The [complete report](evaluation/week6-jepa-baselines-v1/run/report.json) covers
1,528 episodes, 180 numeric groups, all 40 tasks, both cameras and all five
horizons. All 3,240 retrieval queries were eligible, including all 180 groups and
40 tasks in each of the six retrieval camera/horizon cells. Motion coverage also
includes all 180 groups and 40 tasks. Brackets below give paired-design,
group-bootstrap 95% intervals for each estimate.

| Measurement | Copy-latent | Sampled train-only mean |
| --- | ---: | ---: |
| Overall cosine error ↓ | 0.247452 [0.246507, 0.248384] | 0.223306 [0.222252, 0.224340] |
| Motion cosine error, h=8/16/32 ↓ | 0.361010 [0.359695, 0.362331] | 0.244909 [0.242601, 0.247196] |
| Retrieval Recall@1 ↑ | 20.87% [19.62%, 22.12%] | 14.54% [13.49%, 15.61%] |
| Retrieval Recall@3 ↑ | 62.99% [61.70%, 64.30%] | 42.23% [40.81%, 43.69%] |
| Retrieval MRR ↑ | 0.461633 [0.452568, 0.470653] | 0.366314 [0.358360, 0.374598] |

The mean control has lower aggregate cosine error, particularly for the wrist
camera at longer horizons. Copy retains better overall temporal retrieval.
The mean's within-task group cosine is exactly 1.0 for both cameras, and its
effective ranks (8.78 front, 10.19 wrist) are below half the corresponding
teacher ranks (20.29, 21.32). These results make the mean comparison, retrieval
and collapse requirements consequential: cosine improvement alone is insufficient.

| Horizon | Front copy error | Front mean error | Wrist copy error | Wrist mean error |
| --- | ---: | ---: | ---: | ---: |
| 2 | 0.141754 | 0.212674 | 0.222143 | 0.234540 |
| 4 | 0.157523 | 0.212707 | 0.264281 | 0.234304 |
| 8 | 0.179343 | 0.212784 | 0.315458 | 0.233847 |
| 16 | 0.205908 | 0.213054 | 0.359718 | 0.233022 |
| 32 | 0.234799 | 0.213783 | 0.393594 | 0.232350 |

Copy Recall@1 falls from 34.93%/45.07% at h=8 to 8.12%/6.11% at h=32
(front/wrist). The full camera/horizon retrieval scores and uncertainty are in
the report. Retrieval comparisons remain conditional on the frozen eight-frame
pools. Pool construction recorded 2,370 near-time proposal rejections, no
additional RGB/feature-duplicate rejections, and no excluded queries.

Applied to these controls, the unchanged prospective point thresholds require
overall error at most **0.235079** against copy, motion error at h=8/16/32 at
most **0.324909**, and Recall@1 at least **25.8727%**. The mean requirement is
stricter for overall error: it must be **below 0.223306** with a positive paired
confidence lower bound. These rounded values are explanatory; executable checks
use the full-precision controls and the frozen relative thresholds. Every other
guardrail below remains required.

## Controls and full validation

Use the accepted [single-image Qwen target v2](WEEK6_JEPA_TARGETS.md) unchanged.
Each image produces 64 spatial unit vectors with 2,048 channels; front and wrist
are scored separately. The teacher is frozen, with no EMA or fitted target
normalization. No optimizer or policy rollout is part of this step.

- **Copy-latent:** the current image's target features predict the future features.
- **Sampled train-only mean:** one constant feature field per task/camera/spatial
  position. Cover all 1,341 training numeric groups, choosing one appearance
  variant per group by a fixed hash. Encode 16 evenly spaced frames per chosen
  episode, average frames within group and groups within task in float64, then
  normalize each mean vector to unit L2. This uses 21,456 training observations;
  it is a sampled training mean, not the full training empirical mean.

Neither control uses future images as inputs at prediction time. Validation
features do not enter the fitted mean. Test payload access is forbidden.

Score **all 1,109,994 dual-camera validation pairs**, covering 1,528 episodes,
180 numeric groups, 40 tasks and horizons 2/4/8/16/32. Encode each of the 240,946
validation frames once and reuse its features across overlapping pairs. Every
vision forward still receives exactly one camera image.

## Prediction error and uncertainty

The primary error is `1 - dot(predicted_unit_token, target_unit_token)`, reduced
in float64 and clipped to [0,2] for rounding at unit-vector endpoints. Average
over the 64 matching spatial positions. Prediction outputs must be normalized
in the same space; zero or nonfinite output vectors are invalid.

The motion-sensitive diagnostic uses Week 5's exact truth-defined 128 x 128
image-change mask. A token's weight is the changed-pixel fraction in its aligned
16 x 16 mask tile. Aggregate weighted cosine-error mass divided by mask area,
and report the analogous static-region error. This is an image-change proxy;
vision features have broader receptive fields and are not object segmentation.

Average anchors within episode/camera/horizon, appearance variants within group,
groups within task, then give tasks/cameras/horizons equal weight. Report all ten
camera/horizon cells and ordinary pair-weighted diagnostics. Motion ratios use
hierarchically aggregated masses, not averages of small-mask ratios.

Use 10,000 paired numeric-group bootstrap draws within each fixed task, seed
20260925, with percentile 95% intervals. Share draws between compared methods.
Frames and appearance variants are not independent trials. Intervals are
conditional on these tasks and the frozen development split.

## Fixed temporal retrieval panel

Select one validation episode per numeric group using a fixed hash. Each episode
provides queries at 25%, 50% and 75% of the valid anchor range for h=8/16/32,
separately for both cameras: **3,240 proposed queries** over all 180 groups.
Short h=2/4 pairs remain in full cosine evaluation but are not retrieval queries.

Each query has exactly eight candidate future frames from its own episode and
camera. Start with the true target and propose 33 evenly spaced episode frames
in a deterministic hashed order. Require at least four frames of separation
between every retained pair, distinct RGB hashes, and mean token cosine distance
strictly greater than 0.01. The true target is always retained. Randomize final
slots deterministically; the correct candidate is not always slot zero.

If fewer than eight distinct candidates remain, mark the query ineligible and
record the reason. Never shrink the candidate set, relax deduplication or choose
an easier replacement query. Save every exact pool and correct slot before any
predictor evaluation. Selection depends on the fixed teacher and source content,
never on predictor scores. This within-episode panel controls task, appearance
and scene identity; it is a temporal discrimination diagnostic, not global
semantic retrieval.

Rank candidates by the mean same-position token similarity to the predicted
feature field. Report Recall@1, Recall@3 and mean reciprocal rank (MRR). Scores
within 1e-8 of the true-target score receive expected credit under uniform tie
breaking. An all-tied prediction therefore gets exactly **12.5% Recall@1**, 37.5%
Recall@3 and `H_8/8` MRR, independent of target slot. A tie is not counted as a
full top-1 success.

Average eligible queries within numeric group, supported groups within task,
then tasks equally; bootstrap supported groups within each task. Report eligible
and excluded counts, group/task coverage and camera/horizon breakdowns. Retrieval
conclusions apply to this fixed, distinct-candidate subset, not to excluded
stationary or ambiguous frames.

## Collapse diagnostics

Use all nine fixed query targets per representative group, including queries
excluded from retrieval. Average their feature fields within each group before
cross-group diagnostics. Compute, separately by camera:

- Task/group-weighted RMS standard deviation at corresponding spatial positions.
- Entropy effective rank from covariance eigenvalues of the spatially pooled
  group means, using the same task/group weights.
- Mean cosine similarity between pooled group means within each task, then
  average tasks. This helps expose a predictor that only reproduces a task's
  stationary appearance.

Compare predictions with their corresponding frozen teacher targets. Spatial
texture alone cannot satisfy a cross-example diversity requirement. The same
diagnostics are also reported for copy and the mean control.

## Prospective final-checkpoint gate

All requirements below must pass. They are practical thresholds selected for
this study before predictor training, not universal literature cutoffs.

| Requirement | Frozen threshold |
| --- | --- |
| Overall cosine error vs copy | At least 5% relative reduction; paired 95% lower bound >0 |
| Motion cosine error at h=8/16/32 vs copy | At least 10% relative reduction; paired 95% lower bound >0 |
| Overall cosine error vs sampled mean | Positive point reduction and paired 95% lower bound >0 |
| Each camera/horizon cosine error | At most 1.02 times copy |
| Motion support | At least 120 numeric groups and all 40 tasks |
| Retrieval support | At least 120 numeric groups and all 40 tasks |
| Retrieval Recall@1 | At least 22.5%, at least 5 percentage points above copy, and paired 95% lower bound on that gain >0 |
| Cross-group feature std, each camera | At least 50% of corresponding teacher std |
| Effective rank, each camera | At least 5 and at least 50% of teacher rank |
| Within-task group cosine, each camera | At most teacher +0.05 and at most 0.999; no zero pooled vectors |
| Qualitative review | Inspect all 480 fixed Week 5 example identities using feature/prediction diagnostics |

The [executable gate](../src/eventground_wam/jepa_baselines.py) rejects changed
pair identities, source digests, truth masks or retrieval pools. Degenerate
baseline denominators and inadequate support produce an unassessed result,
never a pass. Step 4 will freeze the numerical training schedule and final
endpoint; Step 6 evaluates that endpoint without a validation-based checkpoint
search or automatic extension after failure.

The mean fit, primary scalar error, motion error and retrieval protocol do not
change the accepted Pixel model or its measurements. Pixel and JEPA raw losses
remain different quantities; policy benefit is a later experiment.

## Execution and verification

The [training-only pilot](evaluation/week6-jepa-baselines-v1/pilot/pilot.json)
processed complete 75- and 505-frame episodes. Peak Torch GPU allocation was
0.68 GiB, the teacher was unchanged, and the conservative full-run projection
was 3 hours 21 minutes against the frozen six-hour limit. The
[launch record](evaluation/week6-jepa-baselines-v1/launch.json) pins the source
archive, protocol, pilot and persistent service identities. Local verification
passed all 438 tests and lint for the new implementation and independent auditor.

The complete evaluation finished at 18:36 PDT on September 26 in 6,282 seconds
(1 hour 45 minutes). Peak Torch allocation was 0.68 GiB; systemd recorded 7.3 GiB
peak host memory. Retained run artifacts occupy approximately 4.1 GiB on Feifei.
The teacher remained identical to Step 1, with zero optimizer updates and zero
test-payload reads. The [run completion receipt](evaluation/week6-jepa-baselines-v1/run/completion.json)
pins the report, full episode metrics, templates and retrieval pools.

The [independent audit](evaluation/week6-jepa-baselines-v1/independent-verification.json)
passed complete coverage, train-only mean reconstruction, all scalar estimates
and bootstrap intervals, all retrieval rankings, and collapse diagnostics.
Maximum disagreement was 4.45e-16 for scalar/interval calculations, 2.50e-16
for two complete saved-feature episode replays, 1.78e-14 for collapse statistics,
and 5.56e-15 for retrieval scores after freshly encoding two source episodes.
Mean templates agreed within 2.98e-8 after float32 storage. The audit does not
claim a second encoding pass over every validation image.

Compact evidence is saved under `docs/evaluation/week6-jepa-baselines-v1/`;
feature arrays, mean templates and replay banks remain at
`/home/peimengsui/data/eventground-wam/analysis/week6-jepa-baselines-v1/run`.
The [Step 2 completion ledger](evaluation/week6-step2-completion-v1.json) records
the frozen protocol, evidence and preserved document history.

The [runner](../scripts/run_jepa_baselines.py) verifies the frozen pair bundle,
source/model/runtime hashes and unchanged Step 1 teacher. A training-only pilot
profiles complete short/long episodes before launch. Full execution uses a
persistent Feifei service, one episode of features in memory, per-group training
mean receipts and per-episode validation receipts. Resume requires identical
code/config/source/template identities. Completed output is immutable.

Retain two complete validation feature/mask replay episodes, the fixed group
feature means, retrieval pools/scores and complete episode metrics. Avoid a
persistent cache of every validation feature. An independent audit must verify
coverage, training isolation, arithmetic, bootstrap intervals and retrieval
ranking before Step 2 closes. A baseline report must explicitly say
`NOT_ASSESSED_NO_PREDICTOR`; it cannot certify a future trained model.
