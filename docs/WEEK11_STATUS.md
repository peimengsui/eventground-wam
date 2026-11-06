# Week 11 status

**October 31, 2026: Week 11 is closed; Week 12 is ready but not started.**
The user approved closing the experimental phase. The
[final closeout](WEEK11_CLOSEOUT.md) covers the formal experiment and all
authorized investigations, training pilots and evaluations. No candidate earns
promotion over original M4. All prior performance and latency failures remain
recorded; the adapter screen remains 27/81, with its remaining 54 episodes and
extension closed without execution. The proposed four-case H8 inspection is
deferred future work and does not block Week 12.

The [closure bundle](evaluation/week11-final-closeout-v1/README.md) inventories
23 prior Week 11 packages and preserves all 6,206 indexed files at task-start
contents, including the one documented pre-existing prose discrepancy.
Nine retained model files freshly match their hashes; all 10,048 formal-inventory
files are present at recorded sizes. No EventGround experiment process or active
project follow-up remains; Feifei has approximately 123.2 GiB free. No new
training, inference, rollout, deletion or baseline promotion occurs here.

No required Week 11 task remains. Historical launch states and suggested next
steps below are superseded by this closure, without changing their evidence.

**October 31, 2026: event-free H8 validation and saved adapter inspection complete.**
The [audited report](evaluation/week11-h8-validation-v1/RESULTS.md) completes all
three authorized follow-ups. Event-free H8 exactly reproduces all 309 saved
chunks; matched-input p95 is 53.33 ms versus vanilla M4's 54.40 ms. Actual rollout
p95 is 56.68 versus 57.18 ms. Removing the unused event branch meets the cost
target without changing H8 actions.

All 128 fresh rollouts complete: vanilla/H8 score 24/22 out of 32 targeted and
29/29 out of 32 clean. H8 has zero targeted gains and two losses, plus one clean
gain and one loss. All 64 paired prefixes and artifacts pass independent audit,
with no retries or excluded outcomes. The prespecified directional criterion
fails. The selected cases exclude the earlier 27 diagnostics, but come from
the existing formal pool and one training seed. The frozen selector retains
layout suffixes, so targeted covers 24 base tasks; a supplementary task-grouped
analysis leaves the conclusion unchanged. No broader inferiority or equivalence
claim follows from this small sample.

The [saved adapter review](evaluation/week11-h8-validation-v1/INSPECTION.md)
locates better bowl retention in clean107, lost late pot-grasp recovery in
clean151, and main-adapter engagement of the wrong drawer in targeted231.
These selective effects do not establish a general adaptation benefit.

The validation takes 17.58 minutes including qualification and in-process
analysis; all resource limits pass. The process has exited, about 123.17 GiB
remains free on Feifei, and all 128 raw recordings are retained. Seventeen
focused tests and Ruff pass. All 5,031 previously indexed files remain unchanged
from task start, including the recorded pre-existing prose discrepancy.
Retain vanilla M4 and pause larger continuation/adapter runs. If investigation
continues, inspect the four fresh discordant pairs before proposing a
preservation objective or another training run. That follow-up is not launched.
Earlier entries below describe historical states.

**October 31, 2026: motion-adapter evaluation launched, then stopped at latency.**
The [audited partial report](evaluation/week11-motion-screen-v1/RESULTS.md)
retains 27/81 episodes, one complete nine-case repeat per condition. Fresh
saved-input preflight passed, including both adapters' 456-input H8 checks.
Rollout p95 reached 112.836/112.650 ms (2.026×/2.023× original-fixed preflight),
exceeding the frozen 111.366 ms ceiling. The process exited after 7.10 minutes;
the remaining 54 episodes and conditional extension remain unrun.

H8 fixed8, main adapter and matched control scored 6/9, 6/9 and 7/9. Both adapters
gained clean107 and lost clean151 in this repeat. All artifacts and 27 pair
prefixes pass independent audit, with no retries or excluded outcomes. The
three-repeat retention/rescue criterion is not evaluable. Seventeen focused
tests pass; source and checkpoint files are unchanged. Feifei is idle with
about 127.37 GiB free. Next profile inference latency on retained inputs before
any new rollout contract; do not resume by relaxing this gate. M4-fixed remains
the baseline. Earlier entries below describe historical states.

**October 31, 2026: both frozen-H8 motion-adapter training arms are complete.**
The [final report](evaluation/week11-motion-adapter-v1/RESULTS.md) records 2,000
updates and 58,172 train-only action presentations per arm, with identical
initialization, fresh optimizers, matched sample/horizon chains and distinct
trained adapters. All 456 H8 replay chunks remain exact after training and
fresh-process endpoint loading; the frozen reference and event branch remain
unchanged. The first 64 production updates per arm reproduce qualification.

Eleven focused tests, resource checks and the independent exact metadata audit
pass. All 4,680 indexed files in fourteen earlier packages remain unchanged.
The full pipeline takes 42.23 minutes, including 29.62 minutes of production
training. Qualification p95 is about 108 ms, narrowly below twice the fresh
fixed control. Both jobs have exited; Feifei is idle with about 128 GiB free.
No rollout evaluation was launched. Next run the frozen nine-case, 81-episode
paired retention/gain screen. Robot improvement remains unproven, the earlier
mixed pilot's failed gate is unchanged, and M4-fixed remains the baseline.
Earlier dated entries retain historical launch and investigation states.

**October 31, 2026: the frozen-H8 motion-adapter training pilot has started.**
The [launch record](evaluation/week11-motion-adapter-v1/LAUNCH.md) documents the
qualified 169,030-parameter adapter and matched H8-information control. Both
arms preserve exact H8 actions on all 456 replay entries before and after trial
updates, and pass fresh-process next-update recovery. Frozen model hashes,
gradient isolation, train-only collation and resources pass. Eleven focused
tests pass. P95 inference is 108.15/108.09 ms versus 54.92 ms for fresh original
fixed control, passing the two-times gate narrowly.

The horizon-aware arm is running for 2,000 updates; its endpoint verification,
the matched control's 2,000 updates, and independent verification follow
automatically. Each arm uses the same 58,172 training presentations. Qualification
projects about 35 minutes including final checks, or 45 minutes conservatively.
This pipeline launches no rollout evaluation. All fourteen earlier evidence
packages remain preserved; the old mixed pilot still fails and M4-fixed remains
the baseline. Results are pending.

**October 31, 2026: the investigation supports a concrete motion-adapter pilot.**
The [phase and channel intervention](evaluation/week11-phase-split-v1/RESULTS.md)
completes 24 new episodes with frozen mixed-continuation weights. Early-only
adaptive control through frame 137 succeeds 3/3; sustained adaptation starting
at 138 fails 3/3, versus fixed8 0/3 and full adaptive 3/3. Eight of nine phase
comparisons match consumed inputs and physical prefixes; late-sustained repeat
3 drifts before intervention and is excluded from causal attribution.

In the follow-up, early motion-only and early-only each succeed 3/3, while fixed8
and gripper-only each fail 3/3. All twelve action-differing pair comparisons
match exactly before intervention. Fixed8 and adaptive candidates produce the
same gripper sequence on all 192 early query inputs. Motion-only follows the
successful early-only actions/states and opens at frame 228 under the fixed8
gripper rule. The earlier motion changes the state in which that rule acts;
arbitrary gripper interventions and other cases remain unresolved.

All 1,197 rollout queries, 600 hybrid chunks, resource checks and source/model
hashes pass independent verification. All twelve figures were inspected and
all 4,256 indexed files in thirteen prior packages remain unchanged. The two
runs take 392.8 seconds including qualification, with no training updates.
The [concrete next-step proposal](evaluation/week11-phase-split-v1/NEXT_STEPS.md)
freezes the stronger H8 continuation and learns only non-8 motion corrections,
with a matched H8-information control, exact H8 preservation, bounded training,
and nine-case then 27-case screens. This repair is unimplemented and unvalidated;
the mixed pilot still fails its original gate and M4-fixed remains the baseline.
Earlier dated entries retain their historical findings and proposed next steps.

**October 30, 2026: clean 155 placement-window causal test complete.**
The [12-episode test](evaluation/week11-placement-window-v1/RESULTS.md) gives
0/3 successes each for fixed8, placement-only (frames 138–217), and late-only
(218–297), versus 3/3 for full adaptive. The mixed continuation checkpoint and
event predictor remain frozen. Neither narrow window releases the bowl; full
adaptive opens at frame 220 and completes the task. At frame 218, bowl height
above its starting position is 84.1 mm under placement-only versus 37.5 mm under
full adaptive. The placement window loses containment at frame 212.

All 687 rollout queries and 12 artifacts pass implementation/integrity checks.
Pairing is qualified: only one of six comparisons matches every saved camera
frame. Four match every consumed policy input, prior action, forecast and
simulator state; the other differences are unused wrist-camera frames. The
third fixed8 repeat diverges at query 74 after five wrist channel values change
by one level. Its two paired intervention comparisons are excluded from causal
interpretation. No replacement episodes were run. All six control outcomes
reproduce, but full trajectory equality does not follow from outcome equality.

Execution takes 205.3 seconds plus a preserved 1.8-second preflight failure;
seven tests, numerical qualification, resource checks and local evidence
validation pass. All 3,995 indexed files in twelve previous packages remain
unchanged. Next compare adaptive through frame 137 then fixed8 with fixed8
through 137 then sustained adaptive, retaining fresh controls and input audits.
Neither early-prefix sufficiency nor a general training remedy is established.
The progression screen still fails; original M4-fixed remains the baseline.
Earlier dated entries below retain their historical findings and proposed next steps.

**October 30, 2026: matched-observation continuation replay complete.**
The [replay](evaluation/week11-continuation-replay-v1/RESULTS.md) evaluates both
checkpoints at horizons 2/4/8/16/32 on 456 recording/query entries from all
nine gain/regression cases. All 456 originating-checkpoint chunks reproduce
exactly before alternatives; all 912 checkpoint/input horizon-8 repeats pass.
The 5,928 action calls take 330.6 seconds, with zero training updates, event
model calls, simulator steps, or new rollouts. Model states, endpoints, sources,
and 72 raw input files remain unchanged. All 2,129 indexed files in eleven
prior evidence packages are preserved.

Across 420 distinct interaction-window inputs, mixed training reduces median
horizon-32 translation sensitivity from 23.2% to 11.5% and rotation sensitivity
from 34.6% to 17.8%. Horizon-8 motion still changes on all 420 inputs, while
gripper commands differ on only 20 chunks, one action each. At clean 155 frame
218, both checkpoints and all horizons command hold on the failed mixed/fixed8
observation, but all command release after two actions on the successful
mixed/adaptive observation. Clean 30/107 pre-slip inputs remain all-close under
all ten combinations. These findings separate horizon response from the state
reached by the trajectory without proving a counterfactual success rescue.

Next prioritize a small causal intervention on clean 155's earlier
insertion/placement motion, with matched control repeats and a frozen window,
before another broad training run. An H8-anchored mix remains a hypothesis.
No intervention was launched; the progression screen still fails and original
M4-fixed remains the baseline. Earlier dated entries retain historical findings.

**October 30, 2026: saved continuation regressions and gains inspected.**
The [inspection](evaluation/week11-continuation-inspection-v1/RESULTS.md) covers
all nine cases with changed H8-versus-mixed outcomes under either route, across
36 saved recordings. Mixed fixed8 loses stove activation in targeted 365,
pot acquisition/recovery in clean 151, grip retention in clean 30, and
release/drawer closure in clean 155. Mixed adaptive gains acquisition in
targeted 129/362 and placement alignment in clean 6. Fixed8 gains also include
drawer engagement in targeted 231 and bowl retention in clean 107; adaptive
still drops the bowl in the latter. No universal gripper or routing rule emerges.

Both fixed8 checkpoints first produce different actions at frame 10 on all nine
cases despite identical inputs, forecasts, and horizon 8. All 54 first-change
condition-pair comparisons have exact inputs and equal forecasts. This supports
a preservation tradeoff in the action policy, while later causal contributions
remain unresolved. All 18 camera sheets were inspected; dense metrics and
27 figure hashes pass checks. All 1,880 indexed files in ten previous packages
remain unchanged. The inspection adds no model calls or simulator integration;
it is outcome-aware assisted review, not independent human QA.

Next perform bounded matched-observation action replay around the identified
acquisition, slip, placement, and release windows. An H8-anchored training mix
remains a hypothesis to define afterward. No follow-up inference/training was
launched. The progression screen still fails and original M4-fixed remains
the baseline. Earlier dated entries below retain their historical recommendations.

**October 30, 2026: continuation results audited; the progression screen fails.**
Both 2,000-update runs and all 162 episodes finished on October 28 at 21:45 PDT,
after 3 hours 10 minutes of production. The [results](evaluation/week11-multihorizon-results-v1/RESULTS.md)
give fixed8/adaptive successes of 15/15 for original M4, 19/17 for H8
continuation, and 17/19 for mixed continuation, each on the same 27 cases.
Mixed adaptive gains two over H8 adaptive, but mixed fixed8 loses two against
H8 fixed8; the allowed loss was at most one. All other progression checks pass.
Mixed adaptive ties H8 fixed8 at 19 successes, so the pilot does not establish
an advantage over simply continuing H8 training and keeping fixed8 control.

Independent checks verify both exports, all 162 recordings, 405 exact initial-input
case pairs, zero retries/query exceptions and matching training exposure. All
1,098 indexed files in nine prior packages remain unchanged. Original adaptive
changes one outcome from the earlier gate run after a one-level wrist-image
difference; fresh controls and the selected-panel qualification remain important.
The recommendation is to inspect the saved fixed8 regressions alongside the
adaptive gains before defining any H8-anchored follow-up. No new training or
evaluation was launched by the analysis. Original M4-fixed remains the baseline.

**October 28, 2026, 18:36 PDT: matched continuation training is running on Feifei.**
The [launch record](evaluation/week11-multihorizon-run-v1/LAUNCH.md) documents both
qualified training arms, exact h8/reference-gradient checks, exact interrupted
restart and damaged-checkpoint recovery, and 128 timing updates per arm. All 19
focused local tests pass. The H8 arm has begun; the mixed-horizon arm and
162-episode, six-condition screen follow automatically after endpoint checks.
Both continuations start from the original M4 endpoint, excluding pilot weights.
The first 128 H8 production updates match qualification exactly, and its trained
recovery checkpoint at update 122 passes an independent hash check.
Measured training projects 2.71 hours total, or 5.71 hours conservatively; screen
preflight projects 42.23 minutes and 11.10 GiB. All frozen caps remain unchanged.
All 969 indexed files across eight preceding packages remain unchanged. Results
are pending, the event predictor stays frozen, and M4-fixed remains the baseline.

**October 28, 2026: matched multi-horizon pilot preparation is complete.**
The [new design](WEEK11_MULTIHORIZON_PILOT.md) freezes H8 versus mixed-horizon
continuations from identical accepted M4 weights: 2,000 updates, 58,172 action
presentations and 277,172 auxiliary pairs per arm, across 399 training episodes
and all 40 tasks. All 1,197 required payloads and checkpoint hashes pass; all
84 shared predictor/core tensors match exactly. Seven local planning checks
pass. The six-condition screen comprises 162 future episodes on the existing
27 diagnostic cases, including wins and losses.

The [preparation report](evaluation/week11-multihorizon-plan-v1/PREPARATION.md)
separates completed metadata/schedule checks from pending implementation and
GPU qualification. Historical M4 A100 timing projects 6.18 hours for both arms
at the mean, or 14.61 hours conservatively with saves; neither is a measured
Feifei ETA. Feifei has about 147 GiB free. The next implementation must load the
trained full predictor, explicitly set all four learning rates, and carry a
new continuation/recovery identity. No training, model forward or rollout has
been launched in this preparation; M4-fixed remains the baseline.

**October 25, 2026: Steps 1–6 are complete. Adaptive v1 fails the original performance gates.**
The [formal closeout](evaluation/week11-closeout-v1/RESULTS.md) contains the
independent audit, frozen paired statistics, fixed-control comparison, figures
and reproducibility package. M4-fixed remains the current baseline. The formal
closeout launched no follow-up; subsequent user-authorized diagnostics are recorded below.

| Step | State |
| --- | --- |
| 1. Experiment design and evidence pins | Complete; immutable parent preserved |
| 2. Horizon mapping and fallback contract | Complete; original frozen rule retained |
| 3. Integration implementation | Complete; separate event/action branches and all three arms |
| 4. Correctness and bounded pilot | Complete; 24 episodes, exact GPU parity/reload and 72 original tests |
| 5. Targeted paired evaluation | Complete; all 2,001 episodes, zero retries or query exceptions |
| 6. Formal analysis and reproducible handoff | Complete; independent artifacts, paired intervals, control audit and negative progression decision |

## Formal results

| Variant | Targeted | Clean |
| --- | ---: | ---: |
| M4-fixed | 311/467 (66.60%) | 187/200 (93.50%) |
| M4-EG-fixed8 | 311/467 (66.60%) | 187/200 (93.50%) |
| M4-EG-adaptive | 279/467 (59.74%) | 177/200 (88.50%) |

Adaptive minus fixed is **−6.85 pp targeted, paired 95% CI [−10.28, −3.43]**,
and **−5.00 pp clean, task-cluster 95% CI [−10.00, 0.00]**. The clean interval
reaches zero. Targeted has 19 adaptive-only wins versus 51 fixed-only wins;
clean has 7 versus 17. Comparison with event/fixed8 gives the same paired results.
All 10,000-draw bootstrap rules and descriptive subsets follow the frozen plan.
These intervals describe the evaluated panels, not training-seed variability
or unseen-task generalization.

Both point-estimate and interval performance criteria fail. Adaptive p95 is
**108.62 ms targeted (1.885× fixed)** and **107.84 ms clean (1.877×)**, so latency
passes. All artifact, pairwise initial-input and resource requirements pass.
There is no positive progression recommendation under v1.

## Fixed-control audit

All **667 fixed/fixed8 success/failure outcomes match**. All **16,500 queries
with identical camera inputs, float32 proprioception and instruction produce
bitwise-identical decoded actions**. The full recording audit found:

- 597 completely identical non-timing trajectories.
- 46 with camera differences only outside policy-query observations.
- 24 with policy-query camera differences and subsequent action differences.
- Eight different episode lengths, preserving the same successful outcome.

At each first differing query in those 24 episodes, simulator state,
proprioception and every preceding action match. Camera differences are at most
one intensity level, affecting at most 38 channel values. The pattern is
consistent with rendering variation preceding action divergence; no rendering
mechanism or adaptive failure cause is established. Full-trajectory equality
must be qualified, while no same-input action discrepancy or outcome difference
was observed from running the fixed8 event branch.

## Execution, preservation and handoff

The evaluation ran on Feifei from **11:52 to 16:25 PDT**, about **4 hours
34 minutes**. The service has exited successfully. Raw files remain under
`/home/peimengsui/data/eventground-wam/libero-plus/results/week11-formal-v1`.
The closeout independently validates all **10,048 raw output files** and all
2,001 HDF5 recordings. All **8,047 metadata files** are mirrored locally and
match the remote SHA-256 inventory. Raw HDF5 remains intact on Feifei for any
separately scoped failure diagnosis; it has not been copied locally or deleted.

The accepted analysis source is `package-v3` in the closeout package. Its 27
focused tests pass both locally and in Feifei's pinned runtime. The original
79-test formal launch and all earlier freezes are preserved. A stopped
postprocessing attempt and a local test-import correction are retained;
all remote analysis-attempt time remains charged. Qualification, evaluation
and analysis total about **4.725 active hours**, plus 5.46 seconds of presentation
rendering. All time/memory/disk limits remain satisfied.

The [experiment](WEEK11_EXPERIMENT.md), [horizon rule](WEEK11_HORIZON_MAPPING.md),
[integration](WEEK11_INTEGRATION.md), [pilot report](evaluation/week11-integration-v1/RESULTS.md),
[launch record](evaluation/week11-formal-v1/RESULTS.md) and
[analysis method](WEEK11_CLOSEOUT_METHOD.md) remain linked for reproduction.
Original M4 and Week 10 exports are unchanged. The closeout adds no event-label
accuracy claim: independent label QA and on-policy event accuracy remain
unverified, as do any causal explanation for the adaptive regression.

## October 28 recording investigation

The separately requested [adaptive failure investigation](evaluation/week11-adaptive-diagnosis-v1/RESULTS.md)
is complete for all 667 event-fixed8/adaptive pairs. All 666 comparable first
non-8 queries change actions on identical inputs; all 1,095 shared-input
horizon-8 queries retain exact actions. All 68 adaptive losses first depart at
horizon 32, with unchanged gripper commands in that first chunk. Most successful
episodes also first depart at 32, so this does not isolate a harmful horizon.

Fixed8 forecasts show a recorded closing response within 32 frames for only
48.39% of eligible targeted and 65.49% of clean `gripper_close` predictions.
This limited command/aperture consistency check is not full event accuracy.
Failures do not have higher average horizon-switch rates. Thirteen recorded
examples cover acquisition, placement, and later task-completion symptoms.

The working explanation combines imperfect forecasts with an action coupling
trained at horizon 8; their individual causal contributions remain unresolved.
All consumed recordings and logs match the original SHA-256 inventory, 25 local
tests pass, and the original closeout remains intact. This pass used CPU analysis
only, with no model inference, simulator steps, or new rollouts. Matched-input
cross-horizon replay with event annotation is the proposed next diagnostic;
the negative v1 result and M4-fixed baseline remain unchanged.

## October 28 matched-observation replay

The requested [next diagnostic](evaluation/week11-horizon-replay-v1/RESULTS.md)
is complete. A frozen, post-hoc panel of 27 outcome-stratified pairs supplies
130 observations at initial departure and around first closing. All five
horizons were replayed at each input, plus a horizon-8 repeat: 780 model calls.
All 130 recorded chunks and 27 shared-prefix adaptive chunks reproduce exactly;
all repeat tensors, proposals, backbone queries and model-state checks pass.

All non-8 horizons change arm commands. Forecast-selected horizons change gripper
commands in 14/52 pre-close observations; on the adaptive arm's own pre-close
inputs, the actually used horizon differs from forced 8 in 6/26 cases. None of
the 27 first departures changes the gripper command, locating this additional
sensitivity around acquisition rather than initial reaching.

Prediction-independent physical reconstruction covers all 54 selected recordings
without integration or rendering. Eighteen first-departure inputs have no new
candidate event within 32 frames under either recorded continuation, despite
positive forecasts routing to 32 in 17 cases and 16 in one. Only 48/130 total
windows are unambiguous under the automatic rules; these are unreviewed candidate
labels, not certified event accuracy. Forecast and coupling contributions to
success-rate regression remain causally unresolved.

GPU replay took 66.7 seconds and successful annotation 116.8 seconds. Three stopped
annotation attempts totaling 4.0 seconds are retained. Seven focused tests pass;
108 original input files, 130 tensor archives and 216 physical evidence files are
rehashed successfully. Model weights, formal source and prior result packages
remain intact. No training or new rollout occurred. Targeted label review is the
next step before selecting a controlled intervention; M4-fixed remains the baseline.

## October 28 event-window review

The requested [continued investigation](evaluation/week11-event-review-v1/RESULTS.md)
reviews all 18 shared negative early windows and all six adaptive pre-close
gripper changes, using 39 existing recordings and prior replay tensors. Dense
evidence confirms open commands and no movable-object contact throughout all
36 early windows. Two pairs contact the cabinet, which the frozen labels exclude;
their forecasts are closing events, so that contact does not validate the prediction.

Seventeen early forecasts place onset at 30 or 32, while later qualified events
begin at 36–247 frames in the 34 continuations with such an event. Median `none`
probability is only 2.49%, suggesting positive-event decisions on distant approach
observations rather than just a small timing error. These are selected diagnostic
cases, not an accuracy estimate.

Each of the six h4/h8 gripper differences changes one action, shifting closing by
50 ms: three earlier and three later. The raw scores cross zero by substantial
amounts. The windows include failed acquisition, successful bowl lifting in an
eventual adaptive loss, and stove-control manipulation in an adaptive win. A
single early-closing explanation is unsupported. Five full near-close windows
remain formally ambiguous; assisted inspection does not constitute independent QA.

All 24 frame sheets were inspected, three focused checks pass, and original
packages remain intact. Extraction and frame rendering took 10.12 seconds with
zero new model calls, rollouts, simulator steps or training updates. A proposed
next causal test would retain h8 through the first executed closing transition
and then enable original adaptive routing, compared with both original controls;
that intervention has not been launched. M4-fixed remains the baseline.

## October 28 closing-gated comparison

The user authorized the proposed [three-way comparison](evaluation/week11-gate-comparison-v1/RESULTS.md).
All 81 fresh episodes on the prior 27 diagnostic cases completed with no retries
or policy exceptions. Fixed8 succeeds on 15, original adaptive on 16, and adaptive
enabled after the first executed closing transition on 16. The gate recovers
four adaptive failures and loses four adaptive successes: no net gain. It
recovers 4/7 historical adaptive losses but loses 4/7 historical adaptive wins.
The object subset loses one success relative to adaptive; the fixture subset gains one.

All initial inputs match across arms. All 214 inactive-gate queries match fixed8
inputs and actions exactly, and independent audits reconstruct all 2,508 query
gate states from executed commands. Exact qualification passes on 27 recorded
inputs with 108 complete policy queries. Both model states remain unchanged.
The gate activates in 26 cases; one successful drawer case never closes.

Fixed8 reproduces all historical outcomes; adaptive reproduces 26/27. Clean 156
changes to success. Its first 12 queries match; the next differs by one intensity
level in one wrist-camera channel value while simulator state, prior actions,
forecast and horizon remain equal. Its resulting action chunk differs. This
qualifies reproducibility and the one-success margin over fixed8; no case is excluded.

The main invocation takes 13.4 minutes and produces 3.26 GiB. A 1.94-second
preflight failure and 2.85-second corrected admission are preserved and charged.
All resource bounds pass. Twenty-three execution/integration tests and three
analysis tests pass locally and remotely. All 431 raw files are rehashed and
350 metadata files mirrored. The four earlier result packages remain unchanged.

No uniform pre-closing gate is recommended. The next recording diagnostic is
post-activation divergence in targeted 1, targeted 356 and clean 6: these retain
the fixed8 approach yet still fail while fixed8 succeeds. No additional rollout
or training follows this comparison; M4-fixed remains the baseline.

## October 28 inspection of the remaining gated failures

The requested [inspection](evaluation/week11-gated-failure-inspection-v1/RESULTS.md)
uses six existing fixed8/gated recordings for targeted 1, targeted 356 and clean 6.
All three pairs have identical inputs, simulator states and prior actions at the
first post-close changed query. Each switches from h8 to h4; arm commands change
immediately, with the eight-action execution cadence unchanged.

Targeted 1 closes empty and fails to recover: bowl height rises only 1.5 mm,
versus 107 mm after fixed8 retries. The first changed chunk includes one extra
open command, but arm commands also differ. Targeted 356 loses soup contact
after frame 88, never reacquires it, and never contacts sauce. Fixed8 recovers
and transports both. Its first changed chunk has identical gripper commands.

Clean 6 successfully acquires and lifts the bowl in both arms. Gripper commands
match throughout the common recorded prefix, but gated transport/descent places
the bowl off-center: 52.0 mm from the plate center at frame 104 versus fixed8's
22.6 mm. The goal requires less than 30 mm, plus contact and vertical ordering.
Gated never meets the XY requirement in any saved frame (minimum 49.9 mm),
later opens at frame 111, and does not correct the placement.

The inspection supports two acquisition/recovery failures and one placement
failure; the first closing command is not sufficient protection. It does not
attribute the entire outcome to one changed chunk or establish a replacement
gate. A first-post-close-chunk h8 override is a possible bounded next causal
test; none was launched. M4-fixed remains the baseline.

Saved-state reconstruction took 16.50 seconds; corrected figure/summary
generation took 4.97 seconds, with zero model calls, integration steps, rendering
calls or new episodes. Eleven camera sheets and three motion plots were inspected.
The initial height summary included settling; preserved v2 results exclude it
and pass an independent six-recording numerical check. Goal code/BDDL, hashes,
source versions and assisted judgments are retained. Original XML is unavailable
and automatic physical candidates remain unreviewed; this is not independent
label QA. The five prior evidence packages remain unchanged.

## October 28 first-post-close chunk intervention

The user authorized the explained [four-arm experiment](evaluation/week11-first-chunk-v1/RESULTS.md):
fixed8, adaptive after closing, first post-close chunk kept at h8 followed by
adaptive, and one first post-close h4 chunk followed by fixed8. All 12 episodes
on targeted 1, targeted 356 and clean 6 completed without retries or policy errors.
The shared eight-action cadence, mapping, checkpoints, seeds and initial states
remain fixed. All first post-close forecasts reproduce h4.

| Case | Fixed8 | Gated adaptive | Skip first change | First change only |
| --- | --- | --- | --- | --- |
| Targeted 1 | Success | Failure | Success | Success |
| Targeted 356 | Success | Failure | Success | Failure |
| Clean 6 | Success | Failure | Failure | Success |

Targeted 356 supplies a case-specific causal result: changing just one chunk
induces failure under an otherwise fixed8 continuation, while preserving that
chunk rescues the adaptive continuation. The first chunk changes arm commands
without changing gripper commands. First-only never secures either target and
later grasps the distractor ketchup; skip-first transports both targets.
Targeted 1 can recover from the first changed chunk, with success under either
narrow intervention. Clean 6 succeeds after one change (24.3 mm final saved
bowl–plate offset), but skip-first still places off-center (minimum 44.7 mm,
outside the required 30 mm). All four clean 6 gripper sequences match through
their 105 common frames. Later arm motion remains implicated there.

All six controls reproduce their previous recorded trajectories exactly. Every
case matches all preceding frames/actions and intervention inputs across arms;
the two first-chunk groups also match through the next observation. Forty-two
GPU qualification queries pass; 422 rollout queries pass independent routing
audits. Model tensors remain unchanged. Thirty-three execution/integration and
two analysis checks pass locally and remotely, and physical metrics pass an
independent 12-recording crosscheck. Nine camera sheets and three motion plots
were reviewed. Physical candidates remain automatic and unreviewed.

Execution took 165.8 seconds plus 1.9 seconds preflight, generating about 615 MiB.
Peak reserved GPU memory was 15.09 GiB and host process-tree RSS 7.44 GiB; all
resource caps passed. Saved-state reconstruction and plotting added 33.29 and
8.62 seconds, with no additional model calls or simulator integration. GPU is
idle afterward. The six prior evidence packages are unchanged.

The results distinguish single-chunk sensitivity, recoverable disturbance and
later placement sensitivity across these cases. Neither narrow rule is a general
remedy. M4-fixed and the formal Week 11 negative conclusion remain unchanged.
No expanded evaluation or training was launched.
