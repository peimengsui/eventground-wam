# Week 7 auxiliary policy experiment

**Frozen October 3, 2026.** This experiment tests whether future-prediction
supervision improves the shared action policy when future prediction is used
only during training. The study includes **M1 Pixel and M2 JEPA**, each compared
with the completed B0 control. The
[sealed experiment contract](../configs/training/week7_auxiliary_experiment_v1.json)
fixes the scope, source artifacts, action exposure, evaluation and claim limits.
It is a study contract; production recipes and the auxiliary trainer are later
deliverables. No policy training or rollout has started for this freeze.

## Verified Week 6 handoff

The [fresh entry verification](evaluation/week7-freeze-v1/week6-entry-verification.json)
confirms that Week 6 and its separately scoped follow-up are complete. It checks
the progression records, the local original JEPA export, preserved records and
all payload hashes in the three result archives. The
[October 2 progression decision](evaluation/week6-progression-v1/decision.json)
authorizes the original JEPA endpoint for downstream research use with a known
limitation. This supersedes the earlier M1-first plan with M2 blocked.

The original JEPA prediction gate remains **FAIL**: front-camera h=2 error is
7.22% worse than copy against a 2% cap. `C_jepa-study-v1` is an alias for the
original bytes, not a repaired or gate-passing export. Residual-pilot weights,
copy/shrinkage gates and further JEPA CPT are excluded. See the
[Week 6 closeout](WEEK6_CLOSEOUT.md) for the interpretation limits.

## Question and fixed comparisons

| Arm | Starting future module | Training objective | Deployed action path |
| --- | --- | --- | --- |
| B0 | None | Completed action-only training | Existing Qwen and OFT policy |
| M1 | Accepted `C_pixel-v1` | Action L1 plus weighted Pixel future loss | Same shared action architecture; future module removed |
| M2 | Research-use `C_jepa-study-v1` | Action L1 plus weighted JEPA future loss | Same shared action architecture; future module removed |

The primary comparisons are M1 minus B0 and M2 minus B0 on policy success.
M2 minus M1 is a secondary comparison of these fixed implementations. Positive,
null and negative effects are all valid outcomes. Week 7 does not assume that
better future-prediction scores imply better control.

M3/M4 belong to Week 8. They must independently initialize from the same base
and respective original CPT module as M1/M2; they do not continue from the
action-trained M1/M2 endpoints. This keeps the later auxiliary-versus-coupled
comparison interpretable. Event-adaptive horizons and EventGround remain later work.

## Inputs and matched action exposure

The [module registry](../configs/training/phase2_future_modules_v1.json) pins
the exact Pixel and JEPA checkpoints. The
[Phase 2 controls](PHASE2_TRAINING_PROTOCOL.md) remain authoritative:

- Each new policy starts from pinned base Qwen3.5-2B and the same fresh OFT head,
  state projector and rank-32 LoRA initialization as B0. Trained B0 weights and
  diagnostic-policy weights are excluded from initialization.
- Each arm receives exactly two action epochs: **3,412,044 presentations and
  117,132 optimizer updates**. Seed 17, the exact ordered batches, optimizer
  boundaries and retained short batches are shared with B0. Completed B0 is
  reused; no extra B0 action training is required.
- The inherited current observations, instruction, front/wrist preprocessing,
  train-only state/action normalizers, all-action L1, shared optimizer schedule,
  predict-eight/execute-eight behavior and 20 Hz control remain fixed.
- Future supervision attaches to those action anchors at **2, 4, 8, 16 and 32
  frames**, using both cameras and the same same-episode validity masks for M1/M2.
  Every available horizon is included. Missing targets never remove or replace
  an action sample. Main future losses average over valid anchor/horizon pairs
  and equally over cameras; normalization covers the complete logical batch.
- Both methods fine-tune their loaded future predictor during action training.
  Pixel retains the existing L1 plus 0.1 SSIM objective; JEPA retains cosine plus
  10 times the existing grouped, teacher-relative variance penalty. The JEPA
  teacher stays frozen and detached. Separate outer weights scale each complete
  future loss; their numerical values require the matched training-only pilot.

The 128 by 256 future-token interface and common predictor core are inherited.
Pixel's image decoder and JEPA's feature projection remain declared differences.
The two original CPT runs each used 8,221,365 pairs and 221,633 updates. This
matches data exposure, not total compute, wall time or prediction quality.

## Auxiliary integration requirements

Step 2 must define a versioned OFT-compatible feature route shared by Pixel and
JEPA. The auxiliary loss must reach trainable shared policy representations;
training a disconnected predictor on permanently detached features would not
test the study question. Both branches must preserve the B0 action computation
when the auxiliary path is disabled, including equality of initial shared tensors.

Future targets are supervision only. Intervening demonstration actions may
condition the auxiliary predictor during training through an isolated path;
they must never enter the action decoder or the shared observation computation.
The decoder cannot attend to predicted future tokens in M1/M2. Deployed policies
accept only current observations, instruction and state and must execute with
the future module and teacher removed. Mutation and gradient tests must verify
these boundaries before pilots. Week 8 must separately define a symmetric
candidate-action route for inference coupling.

## Development evaluation and interpretation

Use only the existing **210-task `dev-210` panel** for Week 7 closed-loop
comparison, with the pinned LIBERO-Plus protocol, task IDs, episode seeds,
initial-state selection and timeout/failure rules. The fixed epoch-2 endpoint
is the evaluated policy; no checkpoint or recipe selection uses rollout results.
Full trajectory-v2 validation can measure action and auxiliary losses at the
inherited epoch endpoints, without optimizer updates, early stopping or selection.
Demonstration test payloads and the larger confirmation panels remain locked.

Report success counts/rates, descriptive Wilson 95% intervals, paired success
differences in percentage points, and the paired win/loss table for all three
contrasts. Use 10,000 paired task bootstrap replicates, NumPy PCG64 seed 7, and
2.5/97.5 percentile endpoints with linear interpolation. Sample the same task
indices for all arms in every replicate and retain all 210 tasks. These intervals
are exploratory task-resampling summaries: related task variants and a single
training seed limit independence and generalization. Report suite, perturbation
category and difficulty breakdowns, including Long tasks. Small subgroups are
descriptive; no significance-based subgroup selection is permitted.

The archived B0 result is **162/210**, with an existing rendering-validity
qualification. Pair identity and input validity must be audited before drawing
paired conclusions. Preserve the existing exact comparison of canonical settling
prefix/first-query images and state, alongside task/scene/initial-state/seed and
runtime identities. An input mismatch makes the full-panel contrast descriptive;
do not drop the offending row or retroactively pass Week 4. A revised tolerance
would require a separately versioned, independently justified rule frozen before
new comparison rollouts. Re-evaluate the same frozen B0 if runtime or input
compatibility prevents reuse; this is evaluation, not additional action training.

Record per-arm inference latency, parameters, training memory, wall time, CPT
cost and pilot/tuning exposure. Auxiliary-loss versus policy-success analysis is
exploratory and within method, using matched task aggregates where available.
Raw Pixel L1 and JEPA cosine values are not interchangeable quality measures.
`dev-210`, prior validation, and clean-2000 have already informed development;
Week 7 results are not untouched confirmation or evidence of universal JEPA or
Pixel superiority. The original JEPA weakness remains a possible disadvantage,
particularly for the later coupled arm.

## Completion criteria and remaining recipe work

Week 7 completes when both policies have verified fixed endpoints, matched
exposure ledgers, all 210 evaluation attempts per arm with pairing/validity
dispositions, comparative results and a reproducible checkpoint/evidence package.
A positive success-rate gain is not a completion gate. Failed correctness,
data-integrity or recovery checks must be resolved or reported as incomplete;
they cannot be relabeled as a negative scientific result.

This freeze completes Step 1 only. Before production, Steps 2–4 must produce:

1. The exact common feature route, tensor/attention boundaries and gradient tests.
2. Implemented future-target attachment, masks, grouped loss normalization and
   full-logical-batch accumulation checks.
3. A prospective, equal-budget, training-only pilot plan for loss scaling and
   module optimizer settings; exact diagnostic examples, trial counts and
   selection rule must be sealed before pilot execution. Report method-specific
   weights and gradient contributions; numerical equality of weights alone does
   not establish fairness. Discard diagnostic-trained weights before production.
4. Sealed M1/M2 numerical recipes, tolerances, trainable-parameter inventories,
   resource estimates/limits and interruption/recovery evidence. Use Feifei with
   one substantial GPU job at a time.

Changes to frozen study choices require a new experiment version and explicit
supersession. Completing the listed implementation and recipe details produces
child artifacts referencing this contract; it does not silently alter the scope.

Run the read-only integrity check from the repository root:

```bash
.venv/bin/python scripts/verify_week7_experiment.py
```

The verifier checks the frozen source hashes, shared controls, original objective
definitions, module mappings, local JEPA export and all B0 development task
identities. It does not replace the later model correctness or GPU pilots.

## Related reading

[Fast-WAM](https://arxiv.org/abs/2603.16666) motivates separating training
supervision from test-time future generation. [PHR-VLA](https://arxiv.org/abs/2608.27609)
is a close example of a training-only auxiliary latent head.
[FutureDuet](https://arxiv.org/abs/2609.34362) motivates camera-specific diagnostics.
Their additional architectures, camera-specific objectives and pretraining data
are not part of the frozen experiment.
