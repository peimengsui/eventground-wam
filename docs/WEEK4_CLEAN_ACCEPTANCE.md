# Frozen Week 4 clean acceptance rule

**Rule:** `week4-clean-acceptance-v1`  
**Frozen:** 2026-09-18 PDT, after the disclosed clean-40 diagnostic and before any
larger-panel rollout. **Measurement status: FAIL** (completed September 18,
verified September 19; see final results (local archive: `WEEK4_CLEAN_COMPARISON_RESULTS.md`)).

The user requested completion of the acceptance-rule freeze. This document and
the [sealed machine-readable rule](../configs/evaluation/week4_clean_acceptance_v1.json)
fix the comparator, checkpoint, panel, statistic and decision boundaries. The
freeze defines the rule; it does not pass the gate or launch the larger run.

## Decision

Let **Δ = B0 success rate − reference success rate**, in percentage points,
using all 2,000 paired trials. Compute the paired 95% interval specified below.

| Outcome | Exact condition | Consequence |
| --- | --- | --- |
| **PASS** | Interval's lower endpoint is **≥ −5.0 points**, and all validity checks pass | Clean operational performance gate satisfied |
| **FAIL** | Interval's upper endpoint is **< −5.0 points**, and the run is valid | Clean gate remains open; diagnose policy quality |
| **INCONCLUSIVE** | Valid complete run with any other interval, including one that straddles −5.0 | Clean gate remains open; no automatic expansion or pass |
| **NOT_ASSESSED** | The full panel has not completed | No performance decision |
| **NOT_ASSESSABLE** | Missing/mismatched evidence or unresolved infrastructure contamination | Repair/audit the evaluation before a scientific decision |

Use unrounded endpoints. A point estimate within five points is insufficient if
the interval extends below −5.0. Examples: [−4, −1] passes; [−9, −6] fails;
[−7, −3] is inconclusive. Neither test significance against zero nor overlapping
separate-policy confidence intervals substitutes for this paired margin test.

The margin remains **five percentage points**. It is measured against the
reference's result on the new panel, not its previous 97.5% score and not a
published headline. There is no additional absolute-success or per-suite
threshold in this version. Passing means relative clean-panel readiness; the
remaining Week 4 requirements still apply.

## Comparator decision and roadmap revision

The original roadmap called for an undefined **configuration-matched public
baseline**. The completed audit established that the installed public model
matches our evaluation conditions but differs in architecture and training.
Under the request to finish this freeze, this rule explicitly replaces that
undefined comparator **for the Week 4 operational clean gate** with the pinned
installed reference. It retains the five-point margin and adds the uncertainty
requirement above. This is a comparator-scope revision, not a performance pass.

- **B0:** `eventground-wam/week4-b0-5epoch-v2`, selected epoch 5,
  checkpoint SHA-256
  `cd1c72fdd96dde44f76bb118842f4ee61862cfb0678a44729b2398793bb83d74`.
  Selection used the frozen validation objective. No checkpoint selection using
  this comparison is permitted.
- **Reference:** `Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata`, revision
  `a85655ec941bae6644c9fbdf62db02b9726d7cf5`, native evaluator revision
  `e4287e94541f459edc4feabc4e181f537cd569a8`.
- Both runtime locks, B0 normalization, reference snapshot verification and
  native preprocessing profiles are pinned in the rule.

The matching audit (local archive: `WEEK4_PUBLIC_REFERENCE.md`) remains part of interpretation.
No claim of equal model capacity, training data, pretraining, compute or causal
architectural comparison is permitted. Public-checkpoint training overlap is
not established. B0 remains the matched control and shared initialization for
the later controlled project experiments.

## Frozen panel

- Four canonical suites, ten tasks each, **50 distinct initial states per task**:
  **2,000 matched pairs / 4,000 total episodes**.
- Use every canonical state index **0–49**, in the recorded order, without
  selecting states based on successes or failures. A read-only Feifei audit
  verified exactly 50 distinct states for every task.
- The [sealed panel](../evaluation/splits/week4-clean-comparison-v1/paired-2000.json)
  records every pair ID, task, instruction, BDDL hash, state-file hash, exact
  initial-state array hash and episode seed. Both policies use the same row.
- Environment seed is 0. Episode seeds derive from SHA-256 of
  `7|week4-clean-comparison-v1|suite|runtime_task_index|trial_index`, using the
  first eight bytes as a big-endian integer modulo 2,147,483,647, replacing zero
  with one. All 2,000 generated seeds are distinct.
- Run B0, then the reference, sequentially on Feifei. Within each policy use
  the declared suite/task/trial order. Resume preserves the original identities.
- Use fresh rollouts for both policies under this comparison identity. Do not
  splice earlier clean-40 artifacts into the new result.

The completed pilot is known: B0 26/40, reference 39/40. Its 40 state-zero initial
conditions are included in this 2,000-state inventory. The rule is therefore
prospective for the larger evaluation **after a disclosed pilot**, not a blinded
or wholly unseen test. The checkpoint remains fixed throughout; the larger panel
is not a training or checkpoint-selection set.

## Rollout and artifact validity

Inherit the frozen clean-v1 simulation, observations, success signal, timeouts,
action execution and recording rules. The new comparison changes the trial
inventory and trial-aware identities; it does not alter the historical v1 files.
Both policies retain their correct native image, prompt and action processing.

Before a performance decision, require:

1. All 2,000 pair identities exactly once for both policies; no dropped tasks or
   reduced denominator, and no duplicated successes.
2. Verified checkpoint/runtime identities, task and state hashes, instructions,
   trial seeds, outcomes and complete per-attempt evidence.
3. Matched initial raw camera, proprioception and simulator inputs through
   settling and the first policy input, as in the clean-40 audit. Input parity
   must be checked before each pair's results support acceptance.
4. Timeouts, policy exceptions and invalid actions count as failures. Do not
   retry completed policy failures. A clearly recorded infrastructure failure
   has at most one retry with the same task, state and seed.
5. An exhausted infrastructure failure stays in the denominator as failure,
   with its original evidence, but the scientific gate is NOT_ASSESSABLE until
   infrastructure validity is resolved. This prevents corrupted reference
   results from making B0 appear acceptable.

A report-writing repair may recover unchanged recorded outcomes; it must not
rerun completed policy failures or conceal the original error.

## Frozen uncertainty calculation

Use a **paired, suite-stratified task-cluster percentile bootstrap**:

1. Compute each task's integer count of B0 successes minus reference successes
   across its 50 paired initial states.
2. Initialize NumPy `Generator(PCG64(20260917))`. In suite order Spatial, Object,
   Goal, LIBERO-10, draw a `(20000, 10)` int64 array of task indices uniformly
   from `[0, 10)` with replacement.
3. Each selected task carries all 50 outcome pairs together. Keep the two
   policies paired; do not treat 2,000 episodes as independent task clusters.
4. Sum selected task differences within each suite, then across the four suites.
   Take the 2.5th and 97.5th percentiles with NumPy's `method="linear"`, then
   divide by 20 to express the 2,000-episode difference in percentage points.

There are **20,000 replicates**, fixed seed, no adaptive resampling, and no
rounding before classification. The analysis runtime uses NumPy **1.26.4**.
The [analysis implementation](../src/eventground_wam/clean_acceptance.py) is
hashed in the rule and tested on pass/fail/inconclusive boundaries, incomplete
inputs, deterministic resampling and state-identity integrity.

This is an approximate interval for task-cluster variation within these suites.
It does not capture training-seed variability or guarantee generalization to
unseen tasks. The exact fixed-panel gap is reported alongside it. Per-suite
intervals are descriptive, without a simultaneous-coverage claim or extra
acceptance thresholds. Separate-policy rates, paired counts and all failure
categories must also be reported.

The official OpenVLA evaluator uses 50 trials per task; its paper additionally
averages over three seeds. Our single frozen trial-seed panel is not a full
paper reproduction. [Official protocol](https://github.com/moojink/openvla-oft/blob/main/LIBERO.md).
Pairing preserves the correspondence of the two policies' observations during
resampling. [Paired-bootstrap semantics](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html).
The task-cluster method and acceptance margin above are this project's explicit
choices, not an upstream OpenVLA acceptance rule.

## Change control and handoff

No early stopping, favorable subset, extra trials to obtain a pass, threshold
relaxation or checkpoint replacement is permitted within this version. If the
result fails or is inconclusive, retain it and diagnose the policy. A new policy
or scientific protocol needs a separately recorded version before its outcomes;
reusing inspected states must be disclosed as development evaluation.

Run the offline freeze validator from the repository root:

```bash
PYTHONPATH=src .venv/bin/python scripts/validate_clean_acceptance.py
```

It verifies the rule, all 2,000 manifest pairs, source inventory, original clean
settings, both runtime locks, selected checkpoints and analysis implementation.
It launches no episodes and cannot mark the empirical gate passed.

The [sealed freeze receipt](evaluation/week4-clean-acceptance-freeze-v1.json)
records successful verification, **240 passing tests**, lint and documentation
checks. A [CPU-only runtime check](evaluation/week4-clean-acceptance-runtime-check-v1.json)
confirmed identical deterministic synthetic results locally and in Feifei's
pinned NumPy 1.26.4 environment. Those synthetic results are implementation tests,
not policy performance measurements. The rule digest is
`0e3404fbe44d16ad5d6742948d8aeaa8a388cfd04ac6527d21454f1dd0e82f03`.

**Execution update (2026-09-18):** the multi-trial runner (local archive: `WEEK4_CLEAN_COMPARISON_RUN.md`)
and trial-aware resume handling are implemented and validated. The formal paired
panel completed under an enabled Feifei user service after a separate two-pair
input-parity and interruption-recovery pilot. The [launch audit](evaluation/week4-clean-comparison-launch-v1.json)
pins the source revision and confirms fresh formal artifacts. The frozen rule
and analysis implementation are unchanged. The completed measurement returned
**FAIL**: B0 68.25%, reference 96.75%, paired gap −28.5 pp and interval
[−35.95, −21.25]. The [final audit](evaluation/week4-clean-comparison-verification-v1.json)
reproduced the exact frozen statistics and verified the completed evidence.
Even a clean PASS leaves `dev-210`, seven-category failure analysis and the final
baseline/evidence freeze required before Week 4 closes.
