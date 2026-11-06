# Week 10 status

**October 24: Week 10 closeout and the focused train/dev diagnostic are complete.**
The approved exploratory weak-label study is closed with the SFT-only
continuation as its provisional event-predictor handoff. The selected endpoint,
parent SFT and original M4 policy/predictor bytes and completion links were
freshly verified on Feifei. The revised-task RLVR no-go and earliest small-panel
pass are both preserved. Original independent-QA requirements remain unmet;
this does not certify automatic labels or establish algorithm superiority.

The diagnostic reaudited all 1,609 train and 396 dev targets, reconstructed all
five dev metric summaries and recomputed 6,000 logged training rewards. The
largest confusion pair is contact/gripper_close (51 errors), versus 16
contact/grasp errors. SFT continuation has 185/297 correctly typed positives;
conditional onset error has a one-frame median but 25 errors above eight
frames. Both none false positives and 15 of those 25 large timing errors occur
while an event of the predicted type is already ongoing. This suggests a
state-versus-new-onset ambiguity, not a proven causal mechanism.

Raw predicted starts differ from eight on 255/396 balanced dev prompts, with
66 positive starts outside the future predictor's supported horizon set. No
mapping, fallback, threshold or Week 11 design was selected. Fifteen selected
examples include recorded front/wrist frames and target/prediction timelines;
they require no additional labeling to close this approved scope. No new
training, model inference, label changes or robot rollouts occurred.

See [closeout and checkpoint handoff](WEEK10_CLOSEOUT.md),
[diagnostic findings](WEEK10_DIAGNOSTIC.md), and
[recorded examples](evaluation/week10-closeout-v1/examples.html).
The next integration work belongs to Week 11 and has not started.
Earlier entries below retain their historical state.

**October 24, 05:32 PDT: the separately approved SFT-only report evaluation is complete.**
The exact dev-selected continuation endpoint was evaluated once on the unchanged
384-prompt report panel (96/class, 80 canonical sources, 40 tasks). It made no
optimizer updates and did not change model parameters. Runtime was 37.65 seconds;
median/p95 forward latency was 45.9/47.2 ms. The comparison reuses the previously
verified starting-SFT report predictions on identical prompts.

Macro positive IoU increased **35.20% → 36.35% (+1.15 percentage points)**, while
none false positives remained **3/96** and positive match coverage remained
**63.19%**. Conditional onset MAE changed **5.396 → 5.335 frames**. The paired
80-source bootstrap interval is **−1.17 to +3.42 points**, including zero: a
positive point estimate, not established reliable improvement. Most of the gain
came from gripper-close intervals; contact IoU/recall declined slightly. Report
labels/counts and the starting baseline had been inspected previously, so this
remains an exploratory weak-label report rather than untouched confirmation.

The [report comparison](evaluation/week10-sft-continuation-report-v1/analysis/RESULTS.md)
and [RLVR/SFT retrospective](evaluation/week10-sft-continuation-report-v1/analysis/RLVR_SFT_REVIEW.md)
are complete, with [comparison figure](evaluation/week10-sft-continuation-report-v1/analysis/dev-and-report.png).
The retrospective corrects the earlier overbroad statement: the original
48-prompt RLVR pilot **passed** its small dev gate; later revised-task RLVR pilots
on 396 prompts failed. These task versions and absolute scores are not directly
comparable. All full/main RLVR runs remain unlaunched; no new training, model
selection or report-tuned changes followed this evaluation.

Nine metric/bootstrap/calibration checks passed. The new 598-file package is
verified locally and on Feifei, and all 4,815 files in four preceding evidence
packages remain unchanged. Package digest:
`283500d955194c74ef63a0dc6d7541a35e623ccd717a088a4c1eb91d40083c09`.
Earlier entries below are historical.

**October 23, 22:46 PDT: the preservation pilot and matched SFT-only control are complete.**
Both arms finished 32 updates / 500 identical prompts from the same completed
future-onset SFT, using effective batch 16/micro 1, seed 17, FP32 and the same
head/LoRA learning rates. The training-only, no-update diagnostic froze the
supervised preservation weight at **0.27453765**. Hybrid training took 2m29s;
SFT-only control took 1m50s; the full qualification/training/evaluation queue
finished in 8m12s with no retries.

On the unchanged 396-prompt dev panel, starting SFT scored **41.76%** macro
positive IoU with **2/99** none false positives. The hybrid scored **42.52%**
with **4/99** and still failed the IoU-gain and none-FP gates. The matched SFT-only
control scored **44.02%** with **2/99** and passed every frozen gate. Its paired
source-bootstrap IoU gain over SFT was **+2.25 points, 95% interval +0.34 to +4.52**.
The hybrid-minus-control interval was **−3.32 to +0.15 points**, so the observed
ordering alone does not establish a reliable difference between the two arms.

The revised RLVR performance blocker remains; the promising next candidate is
the SFT-only continuation endpoint, pending a separately authorized fixed report
evaluation. **No full training or report evaluation was launched.** This remains
an adaptive exploratory comparison on reused dev data and automatic weak labels.
See [results and curves](evaluation/week10-preservation-pilot-v1/analysis/RESULTS.md).
All endpoint/completion links and M4 checksums passed verification; all 4,187
files in the three earlier packages remain unchanged. The new 628-file metadata
package is verified on both Feifei and this workspace (digest
`9480fe0b2602ab17d8d4d920f66f696a890b4bcd93d6682fd59387c4864de3e2`). Earlier entries below are historical.

**October 23: the user-approved supervised-preservation pilot is running on Feifei.**
The separate [frozen child bundle](evaluation/week10-preservation-pilot-v1/README.md)
uses the completed revised SFT endpoint for one 500-prompt hybrid RLVR+SFT pilot
and a matched SFT-only control. Both use 32 updates, effective batch 16/micro 1,
seed 17, FP32 and identical RLVR learning rates. A fixed no-update diagnostic
on 32 training prompts chooses the preservation coefficient before any new dev
predictions. The unchanged 396-prompt development gates and source bootstrap
will compare both arms with SFT and with each other. No full training, report
evaluation, new recordings or automatic retries are authorized by this child.
Nine new objective/admission tests and 52 existing focused checks passed.
Remote preflight verified 586 deployment files, 637 immutable parent evidence files,
and the exact completed SFT checksum. Service:
`eventground-week10-preservation-pilot-v1.service` (invocation
`45a078fa0ebe4ca486412c08df04750d`). Earlier entries below are historical.

**October 23, 09:17 PDT: the approved next-new-onset experiment completed with an RLVR no-go.**
The revised targets and both held-out coverage gates passed. Fresh M4 SFT finished
202 updates / 3,218 presentations in 9m55s; both GPU qualifications passed bitwise
recovery. On the fixed 396-prompt dev panel, macro positive IoU was 41.76% for SFT,
42.31% for primary RLVR and 42.14% for the predefined calibration. Neither gain
met +1 percentage point, and both pilots increased none false positives from
2/99 to 4/99, exceeding the +2-point ceiling. Therefore **formal RLVR and matched
SFT continuation were not launched**. No thresholds or labels were changed.

SFT's separate 384-prompt report completed: **35.20% macro positive IoU**, 78/96
grasp identifications and 3/96 none false positives. These are label-informed,
automatic weak-label agreement results. See the
[final report and curves](evaluation/week10-future-onset-v1/analysis/RESULTS.md).
The service exited successfully, final endpoint/identity checks passed, and the
3,550 files in the previous pilot/probe packages remain unchanged. The coverage
blocker is resolved; the planned full RLVR comparison is closed at its negative
development gate. A different training plan would require a new explicit protocol.


**October 23: the user approved the separate next-new-onset revision and retraining.**
The [frozen execution bundle](evaluation/week10-future-onset-v1/README.md) predicts
new event onsets even while another event is ongoing, while retaining all physical
labels and uncertainty exclusions. Exact reconstruction gives 1,609 train prompts
(227 grasp), 396 balanced dev prompts, and 384 balanced prompts in the separate
label-informed report panel. The existing experiments and original report remain
preserved. 69 focused tests passed. Fresh-M4 SFT, GPU recovery qualifications,
gated RLVR and the matched SFT-continuation control are authorized under the
frozen child protocol; deployment and preflight are underway. Prior entries below
are historical, including their inactive-proposal statements.


**October 22, 17:45 PDT: the 80-source coverage probe completed, but the current
task still lacks adequate grasp coverage.** All 80 new demonstrations passed
physical verification. Their 105 raw grasp events yield only five eligible
grasp prompts from three episodes/tasks, below the frozen coverage floor.
No automatic collection extension, formal training or model inference followed.
See the [completed evidence and next decision](evaluation/week10-report-coverage-extension-v1/README.md).

A counts-only task-revision check found a concrete alternative: permit forecasts
of a new onset while another event is ongoing, retaining every ambiguity and
all physical labels/caps. The same new recordings would then provide 96 capped
grasp prompts from 25 events / 23 episodes / 17 tasks across all lead bins,
supporting a 384-prompt balanced supplementary panel. The training-only count
would be 1,609 total prompts, including 227 grasp. This is **an inactive proposal**,
not new labels, admission, training or an untouched confirmatory evaluation.
It changes the task semantics and requires a separately approved revision and
fresh SFT/RLVR/control; the previous study and original report stay preserved.

**October 22: a separate bounded report-coverage probe is running on Feifei.**
The user requested continuation to resolve the blocker. The source audit found
1,345 unused canonical source IDs after excluding 655 accessed/assigned sources.
The [frozen probe](evaluation/week10-report-coverage-extension-v1/README.md)
selects exactly 80 new demonstrations, two per task, before payload access. It
uses the unchanged physical labels and conservative eligibility, with dense
eligible anchors only for this separately identified supplemental panel.
The original report, SFT and RLVR pilot remain preserved. Full training and model
predictions are not part of this probe. It must meet its predeclared class and
independent-event/source/task coverage floor before any evaluation admission;
no success or resolution is yet claimed. Service:
`eventground-week10-report-coverage-repair-v1.service`. The first launcher staged
all 80 sources but stopped at a missing ImageMagick environment path before any
source attempt. Its zero-step failure is preserved. The continuation restores the
previously qualified environment, reuses the staged inputs and charges the
earlier runtime without expanding the fixed source or aggregate budgets.

**October 22: requested steps 1–4 completed and packaged, with a final-report
coverage blocker.** The [bounded RLVR report](evaluation/week10-weak-rlvr-v1/README.md)
contains the frozen weak-evidence protocol, both held-out panel manifests, real-GPU
qualification, fixed-endpoint pilot and prospective dev decision.

The primary pilot passed: **404 prompts, 26 updates, 126.86 seconds**. On 48 balanced
dev prompts from 23 sources / 12 tasks, macro positive IoU changed from 45.18% to
46.38%, none false positives stayed 0/12, and conditional onset MAE changed from
2.80 to 2.21 frames. The paired 95% interval for IoU gain is −0.20 to +2.81 percentage
points; this is a modest exploratory engineering pass, not established improvement.
Contact recall decreased by one example. No calibration is permitted after this pass.

Both panels are fully captured and verified: 103 dev sources / 17,123 frames and
77 report sources / 13,070 frames. All physical checks passed; inherited max-one-
channel-level camera qualifications remain explicit. The report manifest has
**423 prompts: 58 contact, 56 close, 309 none and zero grasp**. Only one grasp
anchor was eligible and it was off the unchanged stride-eight grid. Thus the
planned three-class final-report macro cannot be estimated. No source/anchor was
added, no class was dropped from the metric definition, and no protocol was retuned.

**Full RLVR, SFT-continuation and report model predictions remain unstarted.**
The next work should resolve report coverage/scope in a separate explicit plan
before proceeding to full training. The original report stays immutable. The
prospective full-run schedule is frozen (fresh SFT for each arm, 1,332 prompts,
84 updates, one pass), but is not launch authorization. M4, SFT, original source
pins and all completed experiments remain unchanged. Independent label QA is
still not claimed. Earlier in-progress entries below are historical.


**October 21/22: the user approved the next four steps through the bounded RLVR
pilot decision.** The [prospective weak RLVR/evaluation protocol](WEEK10_WEAK_RLVR_PILOT.md)
is sealed. The nested training pilot has 404 unique prompts (125 contact, 29 grasp,
125 close, 125 none), four candidates/prompt and 26 updates. Full RLVR and the
SFT-continuation control are outside this authorization.

The real Feifei RLVR qualification passed: 42.06 seconds, 18.15 GiB peak reserved,
two logical updates plus one bitwise recovery replay. Stored SFT/M4, frozen weights
and the independent frozen reference stayed unchanged. All 103 event-dev sources
staged successfully, and their single-attempt capture/verification/extraction job
is active. Event-report remains payload-locked until the pilot decision and
resulting training scope are sealed. Original independent-QA gates and all prior
experiments remain unchanged; this is explicitly weak-label evaluation.

Live bundle: `docs/evaluation/week10-weak-rlvr-v1` locally and
`/home/peimengsui/data/eventground-wam/libero-plus/results/week10-weak-rlvr-v1`
on Feifei. Services: `eventground-week10-event-dev-capture-v1` (active),
`eventground-week10-weak-rlvr-qual-v1` (passed). The primary launch is prepared,
but waits for the capture/verification GPU job to finish.


**Formal exploratory SFT completed successfully at October 21, 22:29:48 PDT
/ October 22, 05:29:48 UTC.** The fixed epoch-two endpoint contains exactly
168 updates and 2,664 prompt presentations, following both complete passes of
the sealed schedule. Runtime was **501.03 seconds (8 minutes 21 seconds)**;
peak reserved GPU memory was 10.68 GiB. Mean training loss was 0.904919 in epoch
one and 0.590250 in epoch two. These are training losses, not held-out metrics.

The final inference export and all eight recovery checkpoint checksums were
verified, together with every logged prompt batch and completed metadata file.
The inference payload was loaded on CPU and its exact tensor names and tensor
digest verified. Frozen parameters, stored M4 parent and source files remained
unchanged; there were no automatic retries. See the
[completion record](evaluation/week10-weak-sft-v1/run/completion.json) and
[verification](evaluation/week10-weak-sft-v1/completion-verification.json).
RLVR, its weak-evidence reward contract, held-out event panels and final Week 10
comparisons remain outstanding. The exploratory and sparse-grasp limitations
are unchanged. The earlier running-state estimates below are historical.

**October 21, 22:21 PDT / October 22, 05:21 UTC: formal exploratory SFT has
started on Feifei under the user's explicit launch request.** The
[launch setup](evaluation/week10-weak-sft-v1/README.md),
[sealed launch contract](../configs/training/week10_weak_sft_launch_v1.json),
weak-label admission and exact two-pass schedule are complete. 111 local tests
passed; all 343 deployment files and the complete schedule verified on Feifei.

Service: `eventground-week10-weak-sft-v1.service`. The runner verified current
inputs, freshly initialized the M4 child and EventGround head, and verified an
empty optimizer before its first update. No qualification weights are reused.
The fixed endpoint is epoch two / update 168 / 2,664 prompt presentations. Each
epoch contains 83 full batches of 16 and one retained four-prompt batch, with
unchanged FP32 precision and optimizer settings. The first updates are finite
and peak reserved memory is 10.68 GiB. Early timings suggest 9–12 minutes total,
subject to subsequent checkpoint and runtime checks; the hard limit is one hour.
There are no automatic retries. Held-out data and RLVR remain outside this run.

The earlier entries below preserve their historical state. Their statements
that formal SFT has not started are superseded by this launch; the weak-label
and sparse-grasp limitations remain unchanged.

At **22:24 PDT / 05:24 UTC**, the formal run had completed **50/168 updates**
and 800 prompt presentations. The step-25 recovery payload was rehashed, loaded
on CPU and checked against its identity, finite tensors, optimizer steps and
exact epoch/prompt cursor. Recent updates averaged 2.88 seconds; the estimated
remaining duration was 6–9 minutes, with expected completion around 22:30–22:33
PDT. This is an estimate, not a completion claim. The
[saved progress check](evaluation/week10-weak-sft-v1/progress/20261022T052406Z/check.json)
contains the measured evidence.

**October 21/22, latest: the user approved the exploratory weak-supervision
path. The [v4 revision](WEEK10_EXPLORATORY_V4.md) and
[dataset/pilot bundle](evaluation/week10-exploratory-weak-v4/README.md) are frozen.
The conservative pool contains 1,332 prompts from 198 of the 200 source
recordings, across all 40 tasks: 327 contact, 29 grasp, 476 close and 500 none.
All uncertainty exclusions and reviewed/unreviewed provenance are retained.
115 local tests passed, original source pins remain unchanged, and all 336
deployment files plus the dataset rebuild verified on Feifei.**

**The single authorized qualification pilot completed successfully on Feifei**
([results and curve](evaluation/week10-exploratory-weak-v4/analysis/RESULTS.md)).
On 16 fixed examples, 64 logical updates plus one recovery replay reduced loss
from 2.168946 to 0.022247, with 16/16 exact event/start/end matches at the fixed
endpoint. Bitwise checkpoint recovery, finite updates, frozen parameter integrity
and unchanged stored M4 parent all passed. Wall duration was 200.17 seconds,
peak reserved memory 10.68 GiB, and all predeclared resource limits passed.
Service `eventground-week10-weak-v4-qualification.service` exited successfully.

This establishes exploratory fitting feasibility on weak targets, not independent
label QA or generalization. Only 29 grasp prompts are available, all at lead
times of eight frames or less. No formal SFT/RLVR or held-out payload evaluation
has started. The next step is a separate executable weak-label SFT admission
and launch contract for two passes over the 1,332 prompts, freshly initialized
from M4; the qualification checkpoint cannot be promoted. The historical plans
below describe the original independently validated-label route and are preserved;
their blanket statement of zero Week 10 optimizer updates no longer describes
this new, separately authorized pilot.

**October 21, later: the user's saved assisted-review revision 177 was verified
and [preserved](evaluation/week10-assisted-review-v1/review-snapshots/evidence-000177.json).
All event proposals in episodes 001–020 were accepted: 40 contact, 41 grasp and
32 close (113 total), covering four of 40 tasks. There are no saved whole-episode
missed-event checks; all 1,816 ambiguities remain pending. This is an assisted
spot-check, not a pass of the independent QA gate. The user asked whether to
stop further review and use automatic labels; an exploratory weak-supervision
amendment is the proposed route. No new training/data admission has been made.**

**October 21, 2026, latest: the user requested prefilled intervals with correctness
review. The separate [assisted-review page](http://127.0.0.1:8766/) is live;
[guide and provenance](evaluation/week10-assisted-review-v1/README.md). All 894
event proposals and 1,816 uncertainty flags from the frozen extractor are preserved
across the initial 200 recordings. The UI supports per-proposal decisions, plain
language correction requests, explicit episode-level approval and automatic saving
of decisions. No human identity or decision was entered by the agent. This session
expires around 1:00 a.m. PDT October 22; saved decisions persist.**

**This is candidate-assisted feedback, not independent blind QA.** The original
blind preparation, sessions and draft store remain preserved. The current blind
QA assessor does not accept this separate artifact schema. Training admission and
the v4 prompt/coverage proposal remain separate decisions. No new physical rules,
labels, training configuration or previous experiment were changed. The earlier
blind page at port 8765 remains a separate bounded session; do not use it to claim
blindness after viewing these candidate proposals.

**The coverage analysis and [reviewer-safe v4 proposal](evaluation/week10-coverage-resolution-v1/REVIEWER_PROPOSAL.md)
are complete, but actual coverage is not. V3 still has only 29/500 grasp prompts.
A conditional future-onset/adjudication analysis reaches 500 per class with
the existing caps; it cannot replace independent review. V4 approval is pending,
and v3, physical rules and all previous artifacts remain unchanged.**

The EventGround implementation and the actual Feifei FP32 gradient/memory
diagnostic are complete: effective batch 16, 10.68 GiB peak reserved memory and
zero optimizer updates. The earlier 249-test implementation suite passed.
Actual-label/stage qualification remains a later gate. All previous captures,
failures, rendering qualifications, study/source pins and M4 weights are preserved.

The user's three approved preparation steps are complete: **1 (version data
plan), 2 (bounded three-demo recording pilot), and 3 (freeze source identities,
split protection and label rules)**, within the catalog's stated coverage and
the recording's explicit rendering qualification. See the
[v2 design](WEEK10_EXPERIMENT_V2.md),
[completed corrective-pilot report](evaluation/week10-instrumented-pilot-v2/README.md),
and [preserved first attempt](evaluation/week10-instrumented-pilot-v1/README.md).
Production source coverage, manifests and independent label QA remain separate
gates; the limited catalog is not a complete mapping of the historical dataset.

Step 1, freeze the experiment, was completed October 16.
The [experiment design](WEEK10_EXPERIMENT.md) and
[sealed contract](../configs/training/week10_eventground_experiment_v1.json)
bind the completed M4 endpoint and Week 9 closure. A fresh read-only Feifei
check verified both M4 weight payloads and completion links.

The user selected **contact and grasp labels before training**. The new ontology
and its evidence/QA requirements are frozen; unreviewed candidates have now been
extracted from the three pilot recordings, but no labels are admitted.
Gripper-close-only training is not an authorized
substitute for this experiment.

| Step | Status | Deliverable |
| --- | --- | --- |
| 1. Freeze experiment | Complete; v3 source split activated | V1/v2 preserved; approved canonical source partitions active. M4 parent, training controls and prior experiments unchanged. |
| 2. Establish contact/grasp evidence and dataset | Initial 200 captured; candidate extraction complete; human QA and coverage pending | All 33,581 frames verified with retained camera qualification. Blind annotation UI deployed; prospective v4 is unapproved. Labels remain unadmitted. |
| 3. Implement EventGround and reward verifier | V3 implementation complete | Structured head, causal data boundary, SFT/GRPO/control trainers and FP32 child recipe implemented. An approved v4 would require separate builder/binder changes. |
| 4. Verify correctness and resources | Software and real-model gradient/memory checks pass; actual-label qualification pending | Overfit, optimizer/recovery and stage resource admission still require admitted labels. |
| 5. Train SFT | Pending | Fixed two-pass endpoint on 2,000 unique event prompts. |
| 6. Bounded RLVR and continuation control | Pending | 500-prompt gate; at most one specified calibration; gated 2,000-prompt run and matched-prompt SFT control. |
| 7. Evaluate and package | Pending | Held-out event metrics, paired group bootstrap, costs, latency, checksums and Week 11 handoff. |

No Week 10 optimizer update, training service or new cloud instance has been
launched. Subsequent bounded diagnostic replays are recorded below; they admit
no event-label payload.
The head dimensions and optimizer recipe remain explicitly tracked child-contract
work; they must be sealed before optimization pilots. Physical predicates are now
sealed for new instrumented data, but their label quality is unvalidated.

Technical flow: current dual-camera observation + instruction + proprioception
-> M4 pre-coupling features -> EventGround head -> event / interval / horizon.
Offline future evidence supervises SFT and scores four sampled RLVR candidates;
it is never a prediction input. Week 11 connects the horizon to JEPA-WAM control.

## Event-supervision preparation, October 17

The [source audit](evaluation/week10-event-supervision-v1/source-audit.json)
records the evidence and checksums. Feifei's pinned LeRobot metadata contains
images, actions and 8-D robot state, but no full simulator/object/contact state
or original demonstration IDs. No original HDF5 recordings were found in the
audited dataset and vendor directories. Evaluation rollouts are not a substitute
for demonstration evidence.

A bounded public metadata probe found `episode_metadata.file_path` in the
[LIBERO-plus spatial RLDS schema](https://huggingface.co/datasets/Sylvest/libero_plus_data_4suite/tree/1a0f5c97a96e4187fbe52331b9a484f21d244bb2),
but no full simulator states. The original
[LIBERO HDF5 source](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/f13aa24a3da8c43c7225569f28c562979fa0e35a)
does retain full states and a model XML attribute in the inspected demonstration.
This is a candidate source, **not a verified match** to our exact LIBERO-plus
recordings. Only container/schema metadata was read; no episode arrays or RLDS
examples were opened. Findings from one inspected file/suite are not a claim
that every public archive has the same contents.

The [preparation contract](evaluation/week10-event-supervision-v1/preparation/contract.json)
seals metadata-only work and a prospective
[review queue](evaluation/week10-event-supervision-v1/preparation/review-queue.json):

- 200 initial episodes, five per task, spanning all 40 tasks and within-task
  length terciles; 100 additional episodes reserved only for coverage.
- One hash-selected visual variant per training trajectory group; no duplicate
  groups, no event/outcome-based selection and no held-out payload access.
- The first 40 entries are prospective source-qualification candidates, one
  per task. No candidate has yet been aligned, rendered for review or annotated.

The [alignment screen](../src/eventground_wam/event_supervision.py) compares full
action/proprioception traces and rejects omissions, repeated frames, nonfinite
values and approximate replay. Passing numerical agreement cannot admit labels;
physical scene and frame identity require separate evidence. At preparation time the screen had only synthetic tests; the completed pilot
below now applies it to original recordings without admitting physical labels.

Validation: **21 targeted tests passed**, lint passed, the frozen study verifier
still validates all 25 pinned sources, and the saved review queue recomputes
exactly. No contact/grasp labels have been generated or human-reviewed.

Next required input is the original recording and episode/frame conversion map
for `Sylvest/libero_plus_lerobot` revision
`22c57433fef692b5b9ecc0795344daac7fa867a5`, including scene/model/assets and
observation/action phase. The public RLDS path field may help recover that
provenance, but its connection to the pinned LeRobot IDs is unresolved. After
that qualification, seal the physical labeler and reconstruction/annotation
budgets, extract evidence and conduct independent full-episode QA. No training
readiness or physical-label quality claim is made by this preparation.

## Public source-recovery lead, October 17

The follow-up [metadata audit](evaluation/week10-event-supervision-v1/public-recovery-v1/audit.json)
checks the public `katsukiono/libero-plus-episode-categories` table at revision
`64d4c3244865528f8842e4a2ccc3e7a12fc51530` against our exact pinned Sylvest
episode metadata. All **14,347 episode IDs, task strings and lengths match**,
with no duplicates or missing IDs. This establishes metadata consistency only:
every row shares its task/length pair with another episode, and the table has
no original demo ID, frame correspondence or simulator-state payload.

The first 40 previously selected training candidates now have proposed original
LIBERO HDF5 paths. All 40 paths exist in the pinned public original-LIBERO file
inventory. File-name availability does not prove that their recorded scenes or
trajectories match LIBERO-plus; no candidate has passed numerical or physical
alignment yet. The community categories remain unverified metadata.

The upstream recovery script was inspected without execution. It enumerates
all 1,024 RLDS shards, downloads/decompresses their contents, assumes iteration
order for episode IDs, and validates only task/length against mutable
`lerobot/libero_plus`. Running it as-is would not meet this preparation's
metadata-only and pinned-source constraints. The already published small table
is sufficient as a candidate index for the next investigation.

The recommended bounded training-only source-qualification pilot has now run.
Its result and limitations follow; the public metadata table remains a useful
candidate index, not admitted contact/grasp evidence.


## Bounded source-matching pilot, October 17 — complete

The [completed report and evidence index](evaluation/week10-event-supervision-v1/source-matching-pilot-v1/README.md)
and [sealed completion](evaluation/week10-event-supervision-v1/source-matching-pilot-v1/completion.json)
account for all **40 prospective candidates**:

- **26** targets completed numerical comparisons; none passed the full action/proprioception gate.
- **9** were excluded before payload access because task/length overlaps a protected split.
- **2** had no equal-length original demo.
- **2** exceeded the three-demo ambiguity limit.
- **1** remains an unresolved access/resource timeout, not a numerical failure.

Across **47 independently recomputed complete comparisons**, **22 targets**
had an original action sequence passing the frozen action tolerance and exact
gripper-command check. **Zero** passed the full state gate. The first three
queue-ranked action matches also failed a recorded full-state finger-position
diagnostic: maximum discrepancies were **4.736–4.771 mm**, versus the frozen
**0.01 mm** positional tolerance, and remained beyond the first frame.
The ±1-frame diagnostic did not fix those discrepancies. These observations do
not establish whether regeneration or observation processing caused the mismatch.

The initial 40 outcomes remain immutable. A separately sealed operational
recovery allowed one additional attempt for **21 metadata-only timeouts**,
using stored episode-length attributes and checking selected action shapes.
It retained the candidates, protected-split exclusions and numerical criteria.
The original one-hour deadline, cumulative download limits and output ceiling
were preserved: **44.2 minutes**, **9.20 MiB** total HTTP reads.
No held-out target payloads, simulation, GPU training or label admission occurred.

All saved bundle/evidence identities were checked; arrays and diagnostic errors
were recomputed independently. **26 targeted tests passed**, changed Python
files pass lint, and the frozen study still verifies all **25 pinned sources**.
The report, numerical evidence, contracts, failed attempts and checksum manifest
are preserved locally and on Feifei.

**Next boundary:** qualify the exact LIBERO-plus recording/regeneration and
conversion path, including scene, initial state and observation phase. If this
cannot recover the pinned physical trace, any newly instrumented demonstration
set needs an explicit data/protocol revision and independent QA. This completed
pilot does not authorize label transfer from action matches or Week 10 training.


## Generator-provenance inspection begun, October 17

The [read-only follow-up audit](evaluation/week10-event-supervision-v1/generation-provenance-v1/audit.json)
pins eight reference code/metadata files (47,810 bytes). The pinned LeRobot
README does not specify a generation/conversion manifest. The inspected
LIBERO-plus creator records observations after actions, while the pinned
OpenVLA/OpenVLA-OFT regeneration references settle for ten dummy steps and
record fresh observations before retained original actions; they can also
filter no-op steps. Their first full-state row needs independent phase checking
because it is copied from the pre-settling source.

These are testable explanations for matching actions with different states,
**not proof of the exact Sylvest generation path**. No new episode payloads,
simulation or label extraction occurred. Next: establish the actual intermediate
HDF5/conversion identity, then seal and run a small phase/initialization diagnostic
with explicit runtime and scene settings. The original pilot and its thresholds
remain unchanged.

## Regeneration and observation-phase diagnostics completed, October 17

The [source-resolution report](WEEK10_SOURCE_RESOLUTION.md) records six completed
numeric replays on three previously selected training examples, followed by one
camera/timestamp check. Ten dummy settling steps reproduce all 88 frames of
episode 7856 within the unchanged action/state tolerances. Episodes 1897 and
1426 match initially but diverge from frames 99 and 47 respectively. Zero settling
fails all three. These are hypothesis tests, not retroactive changes to the
original pilot or proof of the release's historical generator.

For episode 7856, both cameras and robot observations sample together **2 ms
before** the end-of-control-step full state. Recording at the sensor timestamp
removes the finger-state discrepancy exactly and leaves the replayed trajectory
unchanged. Three predeclared camera-frame pairs per view were inspected; the
front target is blurred and image/conversion provenance remains unresolved.
Neither numerical reproduction nor these sparse images admits physical labels.

All traces, image metrics and timestamps were independently recomputed and
checksummed. Both zero-step operational failures and their bounded recoveries
are preserved. Total work: 1,158 control steps, about 28.76 cumulative runner
seconds, one training episode's two existing videos, no new episode downloads,
no held-out reads, no training. The camera check used existing Feifei rendering.
The 26 targeted tests and unchanged 25-source freeze verification pass.

The historical [instrumented-data revision proposal](WEEK10_INSTRUMENTED_DATA_PROPOSAL.md)
was subsequently approved and activated through v2. Its original text is preserved
as the decision record. Existing labels, earlier experiments and training remain
unchanged.

## Approved v2 preparation and bounded recording attempt

The [v2 contract](../configs/training/week10_eventground_experiment_v2.json)
preserves all 25 v1 source pins and adds six data/rule references. The source
catalog accounts for all 47 previously accessed original-demo comparisons:
22 exact complete float32 action bindings and 25 nonmatching sources. Known
bindings protect whole trajectory groups; unresolved or conflicting aliases are
quarantined before production. No held-out payload was read.

The initial launch failed before simulator creation because a baseline path
omitted its `week10-` prefix. A separate path-only recovery preserved that log and
the same input hashes. It executed 110 control steps: ten settling steps and
100 source actions. At frame 100, forward kinematics normalized a quaternion by
2.220446049250313e-16 and tripped the exact pre/post-physics assertion. The pilot
stopped; the second and third cases were not attempted. The 100-frame partial
HDF5 remains incomplete and excluded. No threshold or failed outcome was changed.

Read-only reconstruction of all 126 saved baseline states reproduced seven
quaternion-only normalization changes, with unchanged time and velocity and zero
additional control steps. The proposed checker requires exact agreement with
`mj_normalizeQuat`, permits at most 1e-12 quaternion roundoff and rejects changes
to every other physics coordinate. It passed all 126 saved-state cases; this is
not a passed recording pilot. The proposed recorder also stores normalized capture
qpos separately from the original integration state.

All failure artifacts and hashes are retained locally and on Feifei. **32 targeted
tests pass**, lint passes, and all 25 v1 / 31 v2 source pins verify. There have been
no new episode downloads, held-out reads, event-label extraction, optimizer updates
or production collection. A complete corrective replay would add 545 steps to
the 110 already used, exceeding the original 600-step ceiling. Its explicit
proposal requests one corrective attempt per case and a 700-step cumulative cap;
no such retry had been launched at that point. The user subsequently approved
the concrete corrective proposal; the completed result follows.

## Approved corrective pilot completed

The [corrective contract](evaluation/week10-instrumented-pilot-v2/recording-contract.json)
records the user's approval. All three cases completed: 126, 301 and 88 frames.
The run retained the second case's false final environment outcome and did not
filter cases by success. It used 545 steps in 15.24 seconds, bringing cumulative
recording steps to 655, below the approved 700 ceiling.

The [independent audit](evaluation/week10-instrumented-pilot-v2/verification-audit.json)
reloaded compiled models and all 515 integration states. Source/action indices,
the saved baseline, proprioception, object/gripper poses and 22,771 raw contact
records agree exactly. No additional control steps were executed. Frame-zero
sampling uses the same fresh boundary as every later frame.

Strict independent camera byte equality did not hold for every fixed sample;
the failed verification attempt remains archived. A separate quantitative audit
found nine of 18 images identical, with the other nine differing by one 8-bit
channel level at 3–147 of 65,536 pixels. No saved pixels or thresholds were changed.
This rendering qualification is explicit, and it does not admit labels. The
[camera preview](evaluation/week10-instrumented-pilot-v2/camera-preview.png)
supports a visual sanity check only; full-episode independent QA remains pending.

The complete evidence is retained locally and on Feifei, including the earlier
failed attempts. Tests (32), lint, and the unchanged v1/v2 source pins pass.
Production source coverage and manifests, capture budgeting, offline extraction
and independent label QA remain the next gates. No event labels, training,
production collection or held-out payload access occurred during this pilot.

## Source-map expansion and event labeler — October 17

The user's requested pre-training steps 1 and 2 have produced:

1. **Source mapping and production planning:** 300 unique exact complete-action
   bindings across all 40 tasks, with 200 initial recordings (five per task) and
   100 reserves in the prospectively selected order. All 399 completed candidate
   outcomes were independently recomputed from saved arrays. Total HTTP reads
   were 6,866,246 bytes; active mapping execution was approximately 20.24 minutes.
   Every timeout, interruption and startup/summary error is preserved. A final
   separately versioned five-minute completion batch finished two timed-out tasks
   using their single continuation; it did not claim the expired original
   calendar deadline. See the [source-plan evidence](evaluation/week10-production-plan-v2/README.md).
2. **Frozen labeler implementation:** contact, grasp, gripper-close, ambiguity,
   censoring and negative-window rules are implemented. The implementation was
   sealed before pilot extraction. The three existing recordings yielded 8
   contact, 7 grasp and 4 close intervals, all unreviewed. **28 new tests and 83
   total targeted tests pass**, as do lint and independent direct predicate
   checks on all 515 frames. V1's 25 and v2's 31 pinned source hashes remain
   unchanged. See the [labeler guide](WEEK10_EVENT_LABELER.md).

**Step 1 is not fully closed:** the recording queue/budget is a proposal, not a
frozen admitted production manifest. Exact full-trace matching cannot establish
absence of cropped/filtered/resampled historical aliases. The active v2 source
closure requirement therefore still blocks production capture. No protected
payloads were opened to try to resolve this limitation.

A [concrete v3 proposal](WEEK10_SOURCE_SPLIT_V3_PROPOSAL.md) defines a new canonical
source-ID split: the 300 training/review sources, 103 new event-dev sources and
77 new event-report sources, all with exact IDs. New event panels exclude every
source already inspected in this preparation/catalog. This is a proposed change
to evaluation membership and the scope of the source-separation claim; it is
**not approved or active**. It cannot claim historical source closure or that the
inherited M4 action learner never saw transformed aliases. The current v1/v2
contracts and prior experiments remain untouched.

After resolving that protocol decision, seal exact capture inputs and admission,
collect the bounded production data, perform independent full-episode label QA,
and then freeze capped/class-balanced prompt manifests. EventGround model/head,
verifier, training recipe and optimization pilots remain later pre-training work.
No production recording, label admission or optimizer update was launched here.


## Approved v3 activation and sealed recording preparation

The user approved the exact [v3 source proposal](WEEK10_SOURCE_SPLIT_V3_PROPOSAL.md).
The [approval](evaluation/week10-source-split-v3/approval.json),
[active memberships](evaluation/week10-source-split-v3/partitions.json) and
[v3 study contract](../configs/training/week10_eventground_experiment_v3.json)
record 300 train, 103 event-dev and 77 event-report sources, with zero source-ID
overlap and all 40 tasks in each partition. This supersedes the earlier
"not approved or active" proposal status above. Historical v1/v2 contracts and
prior experiments remain preserved. New event-source separation is the claim;
complete historical alias closure and novelty to inherited M4 are not claimed.

The [recording package](evaluation/week10-production-capture-v1/README.md) now
seals every source input, runtime dependency, canonical scene identity, recorder
and independent verifier. All 300 source action hashes, initial-state shapes and
ordered joint layouts passed the independent audit. Source staging read
24,896,480 HTTP bytes in 462.28 seconds. The corrected schema preflight passed
all 40 tasks with zero integration steps or rendered images; earlier startup and
overstrict-guard failures remain archived. No held-out payloads were opened.

The initial launch is exactly 200 sources, 33,581 frames and 35,581 control steps
including settling. Capture plus verification must fit the two-hour ceiling and
100-GiB output ceiling. The remaining 100 sources are gated coverage reserves;
any later use shares the same cumulative budget. The launcher requires a separate
exact launch authorization, which has not been created. Full-state/contact
reconstruction is independent; fixed camera samples report strict byte equality
and preserve any disagreements for qualification review. Consistency is not event
label quality, and there is no preexisting full production trace baseline.

Validation: **112 targeted tests passed**, lint passed and all **41 v3 pins**
verified, preserving v1/v2 sources and M4/scientific training controls. The next
step is approval and launch of the bounded initial capture with its verification,
then frozen-label extraction, independent full-episode QA and prompt admission.
No production recording, label admission or Week 10 optimizer update occurred
in this preparation.


## Approved initial 200-source launch

The user replied **"approved"** to the explained initial recording and verification
operation. The [original launch approval](evaluation/week10-production-capture-v1/launch-authorization.json)
records the exact ordered 200 sources and original frozen contract. The initial
service stopped in approval-metadata serialization before simulator import: zero
sources attempted, zero steps, and 0.621699 seconds of execution. All failure
evidence remains in the original bundle.

The [separate startup repair](evaluation/week10-production-capture-startup-recovery-v1/README.md)
changes one metadata-copy line and retains the exact source order, inputs,
simulator settings and independent verifier. It passed the regression test plus
nine launch-gate tests. Continuing the already-approved operation consumes only
the remaining 7,199 seconds and 100 GiB minus a conservative 1-MiB startup charge;
no trajectory has been retried and no retry allowance was added.

Active Feifei service: `eventground-week10-initial200-startup-recovery-v1.service`.
Evidence root:
`/home/peimengsui/data/eventground-wam/libero-plus/results/week10-production-capture-startup-recovery-v1`.
The first two recordings completed 508 frames with exact capture-state
preservation and zero reported consistency errors. The service runs independent
verification after all 200 recordings, then exits; it does not extract/admit
labels, use reserves or start training. Final capture/verification status must be
read from the completion records, not inferred from this initial health check.


## Capture stopped and scene mismatch diagnosed

The initial health check above was superseded by a scene-integrity stop after
15 completed sources, 4,059 recorded frames and 4,209 control steps. Source 16
failed before settling or replay at the canonical-model XML checksum assertion.
The original 200-source list remains unchanged; 184 sources and all 100 reserves
were not attempted. `eventground-week10-initial200-startup-recovery-v1.service`
is failed/inactive; no resumed capture or independent verification is running.

The [bounded no-replay diagnosis](evaluation/week10-scene-mismatch-diagnostic-recovery-v1/README.md)
confirmed the reset-state model exactly matches the frozen schema, but loading
the source initial state changes the stove burner site's visibility alpha from
-1 to 2 through the vendor's post-processing. Only that XML attribute changed.
This diagnostic used no physics integration, action replay or image rendering.
Its initial library-loader failure and corrected result are both preserved.

The capture model copy must explicitly track state-dependent visual fields, and
the independent verifier must reconstruct/check them. The 15 existing recordings
remain unadmitted pending that visual-state audit; captured physics consistency
checks alone do not establish complete scene consistency. The current checksum
and one-attempt rule have not been relaxed, and source 16 has not been retried.
See the diagnosis for the concrete required repair and requalification steps.

## Qualified visual-state repair and authorized continuation

The user requested **“Can you fix it and continue”**. The
[separate continuation bundle](evaluation/week10-production-capture-visual-v2/README.md)
preserves every previous artifact and failed attempt. It retains the exact
canonical reset XML check, permits only qualified burner-site alpha changes,
and verifies those changes against the vendor's state-dependent predicate.
Per-frame visual state is now synchronized into the isolated image model and
saved for independent reconstruction. Physical states, source membership,
observation phase, M4 parent and all scientific training controls are unchanged.

No-physics qualification checked all 40 scenes, 200 initial states and 4,059
previously recorded frames. It found stale burner visibility in 890 frames
across the first five recordings. The first 15 recordings were re-rendered from
saved states into new artifacts; every original non-image dataset and final
environment outcome is preserved, and no source action was replayed. Source
16's specifically authorized zero-step retry has now completed all 368 frames.

The remaining 31,372 control steps share the original 35,581-step initial cap
with the prior 4,209. Conservative prior charges leave 7,000 seconds and 96 GiB
for the continuation, within the original two-hour/100-GiB ceilings. The 100
reserves remain gated. Tests: **135 targeted tests passed**; all 41 v3 study
pins remain intact. Launch preflight verified 2,180 input/runtime/evidence hashes
and 22 bundle hashes.

Current service: `eventground-week10-initial200-visual-v2.service`. It records
the unchanged initial 200, runs full-state/contact/visual reconstruction and
1,200 fixed camera comparisons, then exits. Pixel disagreements are preserved
for explicit qualification review. This supersedes the stopped-service status
above; completion must be read from the new sealed capture and verification
reports. No label extraction, held-out access or training is performed here.

## Initial 200 capture and independent verification complete — October 18

The [completed repair and capture report](evaluation/week10-production-capture-visual-v2/README.md)
supersedes the running status above. All **200 recordings**, five per task,
completed **33,581 frames**. The first 15 were repaired from preserved states
without action replay; source 16 and the remaining 184 completed in their
original order. All outcomes remain present, including three recordings whose
final environment-done flag is false. No reserve was substituted.

Independent reload reconstructed every frame's proprioception, object/gripper
poses, contacts and visual state. Maximum reported physical/contact errors are
zero. All 1,400 evidence-file checksums and 200 result links were reverified.
Capture completion: `e49472eac309fc5aee91fa04a0d4b7129306123a81b489cf37dc0c0fa47e4281`.
Verification: `f7198c6f5a33e5ff94331a2d4f7a6a2bb279d5915f73d57c2cc54823ef921065`.

Of 1,200 fixed camera samples, **845 are bitwise equal**; the remaining 355 differ
by at most one uint8 channel level at 1–167 of 65,536 pixels. The
[qualification review](evaluation/week10-production-capture-visual-v2/camera-qualification-review.json)
independently recomputed every saved disagreement pair and inspected the largest
example. This is a documented rendering limitation, not a relaxed tolerance,
proof of its numerical cause or event-label quality approval.

The run used exactly **35,581 cumulative control steps**. Capture and verification
plus the conservative prior charge used **24.17 active minutes**. New evidence is
**46.99 GiB**, within the remaining 96-GiB allowance. Full recordings and all
previews remain on Feifei; reports, provenance and selected previews are backed
up locally. The original stopped recordings and all failed diagnostics remain
preserved. All 41 v3 pins and M4/scientific training controls are unchanged.

**Next gate:** frozen-label extraction and independent full-episode contact,
grasp and gripper-close QA, followed by capped/class-balanced prompt admission.
Capture consistency does not admit event labels or training. The 100 reserves
remain gated; no held-out payload access or optimizer update occurred.

## Parallel pretraining preparation — October 18

The user authorized completing the remaining preparation and using sub-agents.
Three agents handled the model/objectives, candidate selection/formal runner,
and independent review/evaluation preparation; root integrated the runtime and
performed the bounded Feifei diagnostics. This did not authorize an unreviewed
scientific change or a formal training launch.

### Completed extraction and coverage diagnosis

[Production extraction](evaluation/week10-production-labels-v1/run/completion.json)
completed all 200 recordings, 33,581 frames and 40 tasks in 30.86 CPU wall seconds.
All 402 output hashes and internal seals were independently verified after
transfer. No new physics, recording, held-out payload read or optimizer update
was performed. Completion digest:
`2510a9ca25930501dc870574bde2f340bdc1ae554d1cdfc2d95d0c38dea4a88e`.

| Class | Candidate events | Eligible anchors | Capped main prompts / 500 |
| --- | ---: | ---: | ---: |
| Contact | 374 | 1,090 | 327 |
| Grasp | 278 | 32 | 29 |
| Gripper close | 242 | 1,419 | 476 |
| None | — | 6,496 | 500 |

These are unreviewed candidate counts. The diagnostic manifest contains 1,332
prospective main prompts and 404 prospective pilot prompts; neither is admitted.
The [diagnosis](WEEK10_COVERAGE_DIAGNOSIS.md) identifies a structural conflict:
241 of 278 grasps start inside an already ongoing contact, so earliest-onset
selection plus exclusion of all ongoing events makes them unavailable as next
grasp targets. Ambiguities and the four-anchor cap further restrict the remaining
events. This is not fixed by sampler ordering or independently passing label QA.

The counts-only counterfactual confirms that merely allowing ongoing anchors
would raise the grasp ceiling to 233, still below 500 with all ambiguity checks
retained. It generated no alternate training manifest. A reserve continuation or
a separately reviewed task/eligibility revision is needed; no reserve source was
captured, no ambiguity rule was weakened, and no new protocol was activated.

### Independent review and held-out preparation

[Blind materials](evaluation/week10-independent-review-v1/materials/completion.json)
cover all 200 full episodes with original source/media checksums, a read-only
two-camera/physical-evidence viewer and an explicitly unfinished annotation form.
They are verified locally and on Feifei. No reference events, negatives, reviewer
identity or successful QA receipt were invented. A reviewer still must be assigned
and complete the independent review described in [the guide](WEEK10_LABEL_REVIEW.md).
The aggregate gates remain at least 100 contact/grasp positives each, 95% precision
and recall, and at most two-frame start/end MAE on fully observed matches.

[Evaluation preparation](WEEK10_DEV_REPORT_PREPARATION.md) freezes metadata for
103 event-dev and 77 event-report sources, each spanning all 40 tasks. No held-out
payload was read. Report stays locked until all training decisions are sealed;
dev also waits for train-label QA and a valid training manifest. The global
four/event and sixteen/source caps remain explicit, including the report's
effective ceiling of 1,232 prompts rather than a promised 3,000.

### Implementation and numerical verification

The [implementation](WEEK10_MODEL_IMPLEMENTATION.md) provides the current-only
M4 adapter, 1,475,140-parameter constrained event head, SFT/GRPO objectives,
independent reward-window binding, guarded stage runner and exact checkpoint
recovery. Only the new head and 372 exact inherited language-LoRA tensors are
trainable; 196 inherited visual-LoRA tensors and all remaining parent parameters
are frozen. Root fixed source-file cache mutation detection; independent review
also led to initial-SFT input guards, output reservation before checkpoint writes,
and concrete annotation-to-reward binding. The complete focused suite passes
**249 tests**, with no failures/skips, and the new runtime/test files pass Ruff.

The original [BF16 numerical v1](../configs/training/week10_eventground_numerical_v1.json)
is preserved with its failed real-model check: full-batch versus accumulated
gradient relative L2 was 0.14419, beyond the predeclared 0.01 limit. A bounded
[diagnostic](evaluation/week10-batch-numerics-diagnostic-v2/run/report.json) showed
exact repeated executions but batch-shape sensitivity even with shared padding.
FP32 reduced that error to approximately 1.35e-5. This does not identify a specific
kernel cause. One diagnostic report-serialization failure is preserved and
conservatively charged 20 seconds; the logging-only repair did not change checks.

The [FP32 numerical v2](../configs/training/week10_eventground_numerical_v2.json)
preserves logical batch 16, microbatch 1, sample schedule, learning rates,
parameter allowlist, objectives and tolerances. Its recipe digest is
`1dfd49aee3e852ced7ee72c4a6c9a5fa23b0f387b8bb10386bbb8d01c57228ac`.
The fresh [actual-path check](evaluation/week10-implementation-qualification-v2/run/report.json)
passed with gradient relative L2 **1.36044e-5**, loss difference **2.38419e-7**,
and **10.68 GiB** peak reserved memory including resident Adam moments.
It measured 2.663 seconds of forward/backward work for a logical batch of 16.
The resulting two-pass SFT compute-only projection with a 2x margin is about
22.2 minutes; it excludes measured optimizer/checkpoint/data costs and is not a
formal-run ETA. RLVR's separate frozen-reference memory/time remains unmeasured.

All GPU diagnostics together are charged **62.73 seconds**, including the failed
attempts, against the three-hour qualification budget. All used synthetic
categorical diagnostic outputs on four pinned current observations; **zero
optimizer updates**, unchanged child weights after each preserved successful
check, and no admitted-label training. Actual-label small-set overfit,
actual-host optimizer/recovery validation and stage resource qualification still
require independently admitted data. The runner rejects these gradient-only
reports as formal training admission.

### Current decision boundary

The remaining blockers are independent reference review and a scientifically
acceptable way to obtain the required unique class/lead-time coverage. The 100
fixed reserves remain available only through their bounded capture child; their
sufficiency is not guaranteed. Any change to next-event eligibility or ambiguity
handling requires a separately reviewed protocol retaining all v3 findings.
Once labels and manifests are admitted, finish actual-label/stage qualification
and request approval for a concrete formal launch. No formal training, held-out
evaluation, source replacement, duplicate prompt filling or protocol revision
has been launched by this preparation work.
