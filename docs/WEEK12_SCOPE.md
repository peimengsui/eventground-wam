# Week 12 scope and research claims

**November 1, 2026: Step 1 is frozen.** Week 12 will consolidate the completed
EventGround-WAM study into a reproducible technical report and presentation
package. The immediate deliverables are this scope, the
[report outline](WEEK12_REPORT_OUTLINE.md), and the
[status ledger](WEEK12_STATUS.md). Steps 2–6 remain unstarted.

This is a reporting decision made after results were available. It is not a
preregistration or a new scientific acceptance gate. The
[Week 11 closeout](WEEK11_CLOSEOUT.md) and its
[handoff](evaluation/week11-final-closeout-v1/handoff.json) govern the experimental
boundary. The [Step 1 package](evaluation/week12-scope-v1/README.md) pins the
source documents and frozen versions of these deliverables.

## Purpose and audience

The report should let a robotics or machine-learning researcher assess the
controls, findings and uncertainty, and let a technical hiring reviewer
understand the implementation and experimental judgment. Its central question
is whether future representations and event-based temporal routing improve
vision-language-action policy robustness under the tested LIBERO-Plus shifts.

The main findings are that the five-arm study does not establish an overall
coupling benefit, the weak-label event study does not establish additional RLVR
benefit, and the tested adaptive integration regresses on the formal panels.
Subsequent interventions reveal useful and harmful motion changes in individual
cases, but no Week 11 candidate earns promotion over original M4. These findings
apply to the tested implementations and budgets.

## Included work and deliverables

| Step | Scope | Completion artifact |
| --- | --- | --- |
| 1 | Freeze reporting scope, map questions to evidence, and allocate the report | This scope, report outline, status ledger and versioned source manifest |
| 2 | Consolidate existing outcomes, paired uncertainty, latency and cost | Source-linked tables and numerical cross-checks, with panels and analysis units retained |
| 3 | Prepare figures and balanced examples from retained evidence | Final figure set and success/failure examples with case IDs and selection rules |
| 4 | Package reproducibility | Code/config references, checkpoint and output manifests, environment and reproduction instructions |
| 5 | Write the technical report | A 6–8-page report, with detailed methods and evidence in a linked supplement |
| 6 | Prepare presentation materials and audit claims | A 3–5-minute demo, technical blog, project summary, résumé bullets and final consistency check |

Weeks 9–11 supply the main findings. Weeks 1–8 supply the benchmark, data,
architecture, training controls, predictor validation and acceptance history.
Existing predictions and recordings supply the benchmark artifacts and demo.
Preparation of these deliverables does not itself publish or upload them.

## Research questions and supported conclusions

### RQ1 Pixel versus latent future prediction

**Question:** Which future target better supports temporal prediction and
downstream robustness in this controlled implementation?

The [pixel predictor passed its reconstruction gates](WEEK5_PIXEL_VALIDATION.md).
The [JEPA predictor retained a failed front-camera horizon-2 guardrail](WEEK6_CLOSEOUT.md)
and entered M2/M4 under the documented research-use exception. Their native
losses measure different quantities and cannot rank prediction quality directly.
The [Week 9](WEEK9_FINAL_RESULTS.md) latent-versus-pixel effect intervals include
zero on both panels; the affected comparisons remain descriptive because of
recorded RGB input mismatches. The study establishes no general superiority
of either target. Report M2−M1, M4−M3 and the factorial target effect.

### RQ2 Auxiliary training versus inference coupling

**Question:** Does using predicted future representations in the action path
improve robustness beyond auxiliary training?

[Week 9](WEEK9_FINAL_RESULTS.md) evaluates B0 and the four factorial arms on
1,000 perturbed cases and 2,000 clean episodes per arm. The overall coupling
effect is +1.35 percentage points on perturbed cases and −0.15 points on clean;
both paired intervals include zero. Report M3−M1, M4−M2, the coupling effect
and interaction alongside the B0 contrasts. No overall coupling benefit or
overall winning model is established. Coupling also adds deployed computation
and parameters, so these contrasts do not isolate future semantics from cost.

### RQ3 Fixed versus event-adaptive horizons

**Question:** Does event-based horizon routing improve the fixed M4 policy on
the targeted shifts while preserving clean performance?

The [frozen-protocol Week 11 evaluation](evaluation/week11-closeout-v1/RESULTS.md)
contains three arms and 2,001 episodes. Adaptive minus original fixed M4 is
−6.85 percentage points on targeted cases, with a paired 95% interval of
[−10.28, −3.43], and −5.00 points on clean, with [−10.00, 0.00]. Both performance
gates fail. The event-running fixed8 control matches all 667 binary outcomes
and all 16,500 identical-input action queries; full-trajectory equality remains
qualified by rendering differences.

The [later investigations and pilots](WEEK11_CLOSEOUT.md) remain separate
exploratory evidence. They show case-specific gains and regressions, failed
preservation/progression criteria, a latency-stopped adapter screen, and no
replicated H8 continuation advantage on the latest fresh panel. Retain original
M4 fixed8 without the event branch as the Week 11 reference. This does not
declare M4 the best Week 9 arm or establish that every adaptive design fails.

### RQ4 Event grounding and SFT versus RLVR

**Question:** Can event supervision and reward-based post-training improve
temporal forecasting and downstream robot behavior?

The [Week 10 study](WEEK10_CLOSEOUT.md) supports carrying the development-selected
SFT-only continuation as a provisional event predictor. Its report-panel macro
IoU increases by 1.15 percentage points, with a paired 95% source-bootstrap
interval of [−1.17, +3.42]. Additional RLVR benefit is unestablished. Automatic
labels, prior report exposure and unresolved independent label QA make this an
exploratory weak-label result. Demonstration forecasting scores do not establish
on-policy accuracy or robot improvement; the downstream result is reported
separately under RQ3. The original plan's proposed RLVR robot-policy arm was
not evaluated.

## Evidence and statistical boundaries

“Formal” identifies the completed evaluation under its frozen protocol. It
does not mean an untouched holdout or an independently confirmed generalization
result. Week 9's planned comparisons and Week 11's formal integration retain
their original panel-specific qualifications. Week 10 and the later Week 11
diagnostics, training pilots and follow-ups retain their exploratory labels.

Step 2 should reproduce or reconcile the existing analyses while preserving
their contrasts, denominators, task/source clustering, seeds and exclusions.
Do not pool repeated panels across weeks, turn selected case studies into a
benchmark success rate, or reinterpret evaluation bootstrap intervals as
training-seed uncertainty. Any supplementary analysis must be labeled as such
and must retain the original result alongside it.

The report and its figures must carry the qualifications that affect their
interpretation: one training seed, development/panel reuse, historical
source-alias exposure, renderer/input differences, the JEPA gate failure and
research-use exception, weak event labels, and unresolved on-policy accuracy.
The latest H8 targeted panel contains 32 variants of 24 underlying tasks;
retain its [task-grouped sensitivity analysis](evaluation/week11-h8-validation-v1/RESULTS.md).
Keep timing protocols and hardware explicit, and avoid double-counting shared
CPT cost. Prediction loss, horizon frequency and task success measure different
things. Their association is not established by juxtaposing aggregate scores.
The roadmap's prediction-quality/success correlation remains unresolved unless
a valid aligned measurement already exists; no new measurement is required
for this reporting phase.

## Experimental boundary and deferred work

Full-10K and test panels remain locked. New seeds, training, checkpoint
selection, model inference, simulator rollouts, revised routing/gates, and
label-collection studies are outside this scope. The remaining 54 adapter
episodes and conditional extension stay closed without execution; the optional
four-case H8 inspection stays deferred. Independent event-label QA and on-policy
forecast validation remain future research. A concrete need for a new
experiment should be handled through a separate scoped decision.

This boundary supersedes the original roadmap's full-10K and multi-seed Week 12
launch suggestions. The old plan remains available as history. Existing failed
gates, partial-run counts, checkpoints and evidence packages retain their
recorded dispositions. The known pre-existing prose/hash discrepancy in the
adapter training report remains documented in the Week 11 closure package.

## Step 1 acceptance and change control

Step 1 is complete when all four original questions have a source-linked
conclusion and evidence classification, the current experimental boundary is
explicit, the outline allocates 6–8 pages, and the scope/outline/source
manifest agree. Check links and source hashes; no model test or rollout is
needed for this documentation step.

Later editorial changes may improve wording and layout within this scope.
Changes to questions, evidence inclusion, statistical interpretation or the
experimental boundary require a versioned addendum that records the reason
and results already seen. Preserve this v1 snapshot and the original sources.
