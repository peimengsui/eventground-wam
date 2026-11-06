# Week 10 reviewer quickstart

Your job is to create an independent reference from the saved demonstrations,
including events an automatic extractor could miss. This guide contains no
candidate predictions, candidate counts or episode-specific event boundaries.
It summarizes the already frozen
[physical event rules](evaluation/week10-instrumented-pilot-v1/labeler-contract.json)
(`a9702609f64b4a424186b57b15034f7dd70097d4f1389714a7a058ce6f210065`).
It does not change those rules or authorize training.

## Before starting

The curator provides the running loopback viewer address, its sealed session
limit, and a copy of the prepared reference form. The initial form is
[review-template.json](evaluation/week10-independent-review-v1/materials/review-template.json);
its empty fields are unfinished work, not negative labels. The recordings live
on Feifei: the local metadata backup alone cannot display their frames. If an
SSH tunnel is needed, use the exact command supplied with that viewer session.

The curator must authorize your reviewer identity separately. Do not view
extractor candidates, model predictions, candidate-based clips or assessment
scores before the reference is sealed. If you have already seen them, tell the
curator; do not attest that the review was blind. Do not use generated reference
annotations or an extractor-author sign-off as independent evidence.

Record your actual start/end times with timezone and active annotation time.
Save a draft before breaks or session expiry; reopening a draft does not make
unreviewed frames complete. A four-hour viewer allocation, if separately sealed,
is a resource ceiling, not a requirement to finish all episodes within four hours.
The current initial population is all 200 recordings, five per task across 40
tasks. Keep the supplied order and include unsuccessful demonstrations.

## Review each whole episode

1. Watch both front and wrist views from the first through the last frame.
   Identify movable objects using the geometry map's canonical object IDs.
   Include objects unrelated to the instruction. Tables, fixtures, robot/self
   contacts and fixed articulated fixture parts are outside the object scope.
2. Scrub potential transitions frame by frame. Check saved raw contacts,
   commands, aperture and object/gripper poses alongside the images. Images
   help assess the scene but cannot establish millimetre thresholds alone.
   Reconcile disagreements; record uncertainty rather than inventing precision.
3. Record every supported contact, grasp and close interval. Different event
   types can overlap: annotate both contact and grasp when both occur. Do not
   apply an earliest-event choice or tie rule while making the reference.
4. Review apparently quiet periods again for missed events. Record uncertainties
   and explicit negative ranges for each type. Only after this full pass confirm
   both-view, physical-evidence, missed-event and negative-case review.

Frame indices are integers, starting at zero, at 20 Hz. A frame is the saved
observation **before** that frame's action. Intervals are half-open `[start,end)`:
for example `[10,15)` contains frames 10 through 14. `end` is the first excluded
frame, not the last included frame. Use a unique reference ID for each event.

## Event rubric

| Event | What establishes it | Boundaries and exclusions |
| --- | --- | --- |
| `contact` | A collision-enabled gripper/object geometry pair has distance ≤0 m and active constraint address ≥0 for at least two consecutive frames. Check exact ownership in the geometry map. Forces are not required. | Start at the first qualifying frame; end at the first frame without raw contact for that object. Do not bridge gaps. Positive-distance proximity is not contact. A one-frame contact run is ambiguous. |
| `grasp` | A five-frame window has close command `+1` throughout, at least one finger contacting the same object throughout, and both finger sides contacting it in at least three frames. Relative object position stays within 5 mm and orientation within 10° of the window start, while the gripper moves at least 10 mm **or** rotates at least 10°. | Start is the first frame of the earliest confirmed window, not the fifth confirmation frame. After confirmation, stationary holding continues while a finger still contacts the object and relative pose stays within the exit limits of 10 mm and 20° from that accepted start. End at the first loss of all finger contact or first exceeded exit limit; do not bridge gaps. Close or contact alone is insufficient. |
| `gripper_close` | A raw command transition from `-1` to `+1`, at least five consecutive `+1` frames, and an aperture decrease of at least 1 mm between onset `t` and frame `t+5`. Aperture is the sum of the absolute two finger-position values. | The interval is `[t,t+6)`, containing the response frame. It is not the entire closed hold. Use `object_id: null`. An initial `+1`, a short/weak transition or an unavailable response frame is ambiguous. |

For grasp, compare object pose **in the gripper frame**, not world-space
distance or visual proximity. Translation/rotation measurements are maximum
departure from the confirmation window's first frame; rotations use the
geodesic angle. A stationary object touched by a closing gripper is not yet a
confirmed grasp. A later stationary hold can remain a grasp after motion has
established retention. Failed or incomplete confirmation requires an explicit
uncertainty record, including the reason; it is not automatically a negative.

Use the exact movable-object ID for contact/grasp. Never invent an ID to force
agreement. If geometry ownership or evidence is missing/corrupt, report the
problem to the curator and leave the episode incomplete.

## Uncertainty, censoring and negatives

For every ambiguity, record event type, object, `[start,end)` and a specific
reason, such as occlusion with conflicting evidence or incomplete confirmation.
Do not copy automatic ambiguity windows. A reference positive and a reference
ambiguity cannot overlap for the same type/object; if uncertainty reaches a
proposed positive, resolve the judgement independently or leave it for curator
review. Do not shorten an uncertainty solely to improve an assessment score.

Set `left_censored: true` whenever a reference begins at frame 0, and
`right_censored: true` whenever it ends at the episode's exclusive length `N`.
Do not create a newly observed onset for something already active at the first
frame. An incomplete confirmation near the end stays ambiguous; it does not
establish a confirmed positive or a negative. Review the whole tail even though
some tail anchors cannot later become training prompts.

For each event type, negative ranges must cover exactly the frames outside all
of that type's reference positives and ambiguities, across all objects. An
uncertain frame is never negative. Keep clear negative intervals explicit even
in an episode with no positives. Reviewed-frame ranges must cover `[0,N)` once,
without gaps or overlaps. Playback or an opened page alone is not confirmation
of review.

## Finish and hand off

Export and retain the original completed reference. Identify yourself as the
human author, record completion and signature times, and sign the form's exact
independence statement only if it is true. The curator checks completeness and
seals the reference before revealing automatic candidates or running matching.
A checksum preserves integrity; it does not authenticate who performed review.

Assessment uses one-to-one, maximum-cardinality matching of same-type,
same-object, positively overlapping intervals with matching censor flags.
Contact and grasp each require at least 100 reviewed positives, precision and
recall ≥0.95, and both start and end mean absolute error ≤2 frames on fully
observed matches. Censored events still count for detection; they are excluded
from boundary MAE. Close is reviewed and reported too; this guide adds no new
close-specific threshold. Counts, uncertainty exclusions and confidence
intervals are reported even when the point-estimate gates pass.

An assessment can fail. Preserve that result and the original reference; do not
edit references after seeing scores to make it pass. Passing QA still requires
separate prompt-coverage, dataset and training admission.
