# Week 10 focused train/dev diagnostic

**Completed October 24, 2026.** This audit uses the existing 396-prompt balanced
development panel (99 per class, 103 physical sources, 40 tasks), five fixed
endpoints and recorded training rewards. It makes no model calls or optimizer
updates. Report predictions were not used to select examples, adjust labels or
choose settings. Numbers below are agreement with automatic labels.

The [reproducible diagnostic](evaluation/week10-closeout-v1/diagnostic.json)
reconstructs every stored per-example score and all five metric summaries,
reaudits the 1,609 training and 396 dev targets against the sealed physical
predicate candidates, and recomputes the rewards of all 6,000 stored sampled
training predictions (three arms × 500 groups × four candidates).

## Event-type errors

For the selected SFT continuation, rows are automatic target types and columns
are predictions:

| Target → prediction | None | Gripper close | Contact | Grasp |
|---|---:|---:|---:|---:|
| None | 97 | 0 | 0 | 2 |
| Gripper close | 13 | 52 | 20 | 14 |
| Contact | 9 | 31 | 50 | 9 |
| Grasp | 7 | 2 | 7 | 83 |

The largest confusion pair is **contact versus gripper_close: 51 mistakes**,
compared with **16** contact/grasp confusions. Focusing only on grasp would miss
the larger boundary problem. Of 85 wrong positive predictions, 30 predict a
type that does have a new onset elsewhere in the same future window; 13 predict
a type already ongoing at the anchor. These groups may overlap. Getting a
related event right is insufficient for the task's **earliest new onset** rule.

There are 114 event-type errors overall, of which 40 have predicted-class
probability at least 0.90. This is descriptive evidence that confidence alone
does not certify correctness; no confidence threshold was tuned.

The continuation retains **185/297 correctly typed positive prompts (62.29%)**,
the same count as starting SFT. Five previously correct types become wrong and
five previously wrong types become correct. The IoU gain should therefore not
be presented as a broad event-recognition improvement.

## False positives and already ongoing events

Both SFT endpoints have the same two none false positives: both predict grasp
while an automatically labeled grasp is already ongoing. The target is none
because there is no **new** event onset in the next 32 frames. Primary,
calibrated and hybrid RLVR each add two further none errors; three of their four
false positives have a grasp already ongoing at the anchor.

This is consistent with difficulty distinguishing “the robot is grasping” from
“a new grasp will begin.” It is a hypothesis about the errors, not proof of the
model's internal mechanism or proof that the labels are correct.

Examples in the [recorded gallery](evaluation/week10-closeout-v1/examples.html):

- **Cases 01–02:** both SFT false positives; predicted grasp at offset 32 while
  contact/grasp are already active, with no new onset label in the window.
- **Cases 03–04:** additional RLVR none false positives; SFT continuation returns
  none. One has no ongoing event and one has ongoing contact/grasp.
- **Cases 05–08:** contact/grasp confusions, including predictions with correct
  onset timing but incorrect event type.
- **Cases 14–15:** the larger contact/gripper-close confusion pair.

## Timing: small typical errors, a consequential tail

Timing is conditional on a correctly predicted positive type, so it must be read
alongside the 62.29% positive type coverage.

| SFT-continuation onset error | Result |
|---|---:|
| Correct positive types | 185 |
| Median absolute error | 1 frame |
| Mean absolute error | 3.95 frames |
| 90th percentile absolute error | 12.6 frames |
| Within two frames | 131/185 (70.81%) |
| More than eight frames | 25/185 (13.51%) |
| Mean signed error (negative = early) | −1.48 frames |

Of the 25 large errors, 15 have an event of the predicted type already ongoing.
The corresponding counts are 12/21 for starting SFT and 18/29 for the hybrid.
More ongoing-event awareness is a plausible future direction, but this audit
does not establish that adding temporal context would fix the errors.

The 17–32-frame target-lead group contains 15 of the 25 large errors. Cases 09
and 10 predict a new grasp at offset 4 or 3 when the next labeled onset is at
32 or 31. Both anchors already contain a grasp. Case 11 predicts gripper_close
at offset 4 instead of 32. These are 28-frame onset errors despite correct types.

On the **same 180 prompts correctly typed by both SFT endpoints**, onset MAE
changes only **3.906 → 3.817 frames**. On each endpoint's own matched set it
changes **3.865 → 3.951 frames**. These different denominators explain why the
aggregate onset statistic does not move with interval IoU. Case 12 illustrates
an interval improvement; case 13 supplies a correct, boundary-censored example.

## Would the predicted timing differ from M4's horizon 8?

The continuation predicts positive events for 270/396 prompts and none for 126.
Of the positive predictions, **255/270 (94.44%)** have a raw start other than 8.
Thus **255/396 (64.39%)** of this balanced panel would differ from fixed horizon
8 under each of three illustrative mappings, if none falls back to 8.

| Raw predicted start | 3 | 4 | 8 | 16 | 29 | 30 | 31 | 32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Positive predictions | 13 | 113 | 15 | 23 | 2 | 50 | 1 | 53 |

**66/270** positive starts fall outside the predictor's supported set
`{2,4,8,16,32}`. Nearest (ties lower), clamped floor and clamped ceiling mappings
all change 255 panel horizons, but disagree about which horizon to use. For
example, offset 30 maps to 16 under floor and 32 under nearest/ceiling. These are
descriptive sensitivity calculations, **not an adopted mapping or fallback**.
Balanced dev frequencies do not estimate deployment event prevalence, and no
claim is made that changing these horizons improves actions.

## RLVR signal audit

All logged rewards recompute from the unchanged weak targets. In the primary
RLVR pilot, equal-reward groups are:

| Target | Tied groups / 125 | Interpretation |
|---|---:|---|
| Contact | 13 | All four candidates have the wrong event type |
| Grasp | 4 | All four have the wrong event type |
| Gripper close | 12 | Ten all-wrong-type; two all-correct-type ties |
| None | 117 | All four candidates correctly predict none |

Calibration has the same counts; the hybrid has 118 tied none groups. These
ties have zero policy advantage, while the KL term remains. Correct-none ties
are not errors. Positive all-wrong ties cannot provide a relative preference
for the absent correct event type. This supports a signal-quality hypothesis,
without demonstrating that it caused dev drift or that more samples would fix it.

## Completed examples and practical implication

The gallery contains **15 deliberately selected examples from 14 existing dev
recordings**, with both cameras and all 33 offsets (990 lossless PNG frames).
The current-only input frame is explicitly separated from future frames used
for retrospective inspection. Automatic target and all five fixed predictions
are overlaid as timeline bands; no interval labeling is required from the user.
The examples remain an assisted visualization, not independent QA certification.

The useful handoff is a fixed event predictor with known weaknesses: distinguishing
related event types, separating ongoing states from future new onsets, and a tail
of large timing errors. SFT is the provisional candidate under the previous dev
decision; these diagnostics neither reopen training nor implement Week 11.
