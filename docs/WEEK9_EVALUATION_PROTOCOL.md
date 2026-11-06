# Week 9 evaluation and analysis protocol v1

Frozen on 2026-10-09 at the user's request, before M2/M3/M4 rollout outcomes have been inspected. M1 and B0 dev-210 results were already known. This is a prospective analysis freeze, not authorization to launch evaluations or a claim that the remaining endpoints and runtime checks have passed. The sealed machine contract is `configs/evaluation/week9_factorial_v1.json`; verify it with `scripts/verify_week9_protocol.py`.

## Arms and endpoints

Evaluate B0, M1 (pixel auxiliary), M2 (latent auxiliary), M3 (pixel coupled), and M4 (latent coupled). Use the existing B0 fixed endpoint and the fixed epoch-2 endpoint for each M arm: 117,132 updates, 3,412,044 action presentations and 16,259,770 future pairs. No best-checkpoint selection, intermediate-epoch substitution, seed search, test-driven tuning or pooling training seeds. Pin completed endpoint checksums in a separate cohort manifest before any Week 9 rollout; never edit this protocol to insert observed outcomes. M2/M3/M4 checksum values are intentionally pending completion, with their training identities already fixed.

The controlled 2×2 is M1–M4, with factors prediction target (pixel/latent) and coupling (auxiliary/coupled). B0 is an external action-only reference, not a factorial cell. M3−M1 and M4−M2 estimate the complete coupling treatment, including 1,187,072 added parameters, computation and candidate-conditioned refinement. They do not isolate future semantics from parameter count. Training exposure and logical batch/optimizer rules are matched; training hardware and wall time are not identical. Preserve the original JEPA quality-gate FAIL and the approved research-use exception.

## Fixed common panels

| Panel | Membership | Repetitions | Role |
|---|---|---|---|
| LIBERO-Plus stratified-1K | Existing 1,000 manifest tasks, in manifest order | One rollout per variant task, existing episode seed, initial state 0 | Primary perturbed panel |
| Clean paired-2000 | Existing 40 canonical tasks × 50 frozen initial states | One rollout per manifest pair, existing episode seed and state hash | Clean performance and denominator of robustness ratio |
| dev-210 | Existing development panel | Existing protocol | Development diagnosis only; no extra primary endpoint |

Use exactly the versioned paths and file/record digests in the machine contract. The 1K panel **contains all 210 development tasks** and 790 additional tasks. It is not an unseen 1K holdout. Report the full 1K primary result and separately disclose the 210/790 breakdown; the 790 slice is a prespecified secondary slice, not a replacement primary panel. All 40 clean task identities and all 50 state indices stay fixed. Do not draw new states or regenerate seeds.

All five arms receive a fresh Week 9 cohort on both panels, including B0 and M1. Existing clean or dev results remain historical context and cannot silently fill cells. This avoids mixing recordings from different evaluation cohorts; it does not erase prior rendering qualifications. The full-10K panel and any test data remain locked. No additions, drops, reruns selected by success, or performance-dependent sample expansion.

## Execution and recording

All arms execute on Feifei's RTX 5090 using the exact runtime versions, base model revision, action/state normalizers, benchmark revision and recorder sources pinned in the contract. GPU training and other substantial GPU jobs must be absent. Stagger evaluations as endpoints become available, with the same runtime and protocol. Before the cohort starts, archive the driver, GPU UUID, CUDA/cuDNN, package inventory, environment and numerical backend flags. On every arm admission compare these against the cohort lock. A runtime change requires a separately documented cohort decision before seeing affected results.

Inherit only the simulation, scene/state, observation, seeding, outcome and recording contracts from the existing LIBERO-Plus and clean comparison protocols. Their historical model profiles/Week 4 acceptance thresholds are not Week 9 model recipes or decision rules. All five models use the pinned fresh all-action-L1/binary-gripper-v2 action recipe and its normalizers. A later Week 9 launcher must bind this override explicitly; it must not call the old clean evaluator as though its BCE/acceptance profile described the new cohort.

Use 20 Hz control, predict 8/execute 8 open-loop actions, 10 fixed settling steps, canonical 180-degree rotation, two RGB uint8 256×256 views and an eight-dimensional raw state. Suite policy-step limits are spatial 220, object 280, goal 300 and libero_10 520. M3/M4 derive their current 128×128 FP32 images with the exact training box reducer, retain horizon 8 and decoder-free candidate tokens, and receive no future observations, demonstrated actions, teacher embeddings or privileged simulator state as policy input. Physical continuous actions are decoded/clipped identically; gripper outputs are ±1.

Record every control step and every settling frame in lossless HDF5: both images, proprioception, MuJoCo state, predicted action chunks, executed actions, success/termination, latency and full checkpoint/source/task metadata. Preserve all attempts and partial files, hashes, exception records, immutable receipts and attempt journals. Videos are previews, not source data. Resume only the same identity and panel. One infrastructure retry is allowed; a crash consumes its claimed attempt, including a crash before the recorder opens. Policy exceptions, nonfinite actions and timeouts are scored failures. Exhausted infrastructure failures stay in the denominator as failures and make the artifact audit incomplete. No favorable retry selection.

## Input audit and admissibility

Compare each model with B0 and every pair used in the reported contrasts. Require exact scene/initial-state/instruction/seed identities, every canonical settling observation/state/action, and first-query observation/state. Do not demand identical observations after policies act differently. Hash checks must include intermediate settling frames, not just first-query frames. Check receipt identity/order, HDF5 validity and artifact hashes before reading outcomes for analysis.

Any input mismatch or missing first policy query makes the whole affected contrast descriptive. Missing, corrupt or misidentified artifacts additionally mark its audit incomplete; keep the task and the recorded failure. Do not exclude mismatched tasks, loosen tolerances, choose favorable renderer repetitions, or repair the frozen historical comparison. Report mismatch counts, ordinals, fields and raw artifacts. Require a CPU provenance check, isolated GPU inference/recording smoke check, and renderer repeatability audit before formal cohort launch. Those runtime qualifications are pending, not waived by this freeze.

M1's existing dev comparison has three unresolved Light Conditions mismatches (ordinals 100, 203, 204), despite matching first-query inputs and successful outcomes on those tasks. Its whole dev contrast remains descriptive. Preserve the earlier B0 rendering qualification and report new cohort audits independently; never retrospectively mark the old study passed.

## Estimands and bootstrap

Primary per-arm endpoint: micro success rate on all 1,000 perturbed tasks. Clean endpoint: successes / 2,000, equivalently the mean of 40 canonical-task success rates with 50 states each. Report raw counts and denominators. Wilson intervals may accompany descriptive binomial counts, but clean uncertainty for comparisons comes from canonical-task clustering, not 2,000 independent draws.

Compute 10,000 percentile bootstrap replicates, PCG64, linear 2.5%/97.5% quantiles. Use seed 7 to resample the 1,000 perturbed variant tasks with replacement. Use seed 8 to resample the 40 clean canonical tasks with replacement, carrying all 50 states of each sampled task together. The two panel streams are independent; each stream shares exactly the same resampled indices across all five arms and all contrasts. Variant tasks may share underlying canonical tasks: the perturbed interval describes this fixed variant-task panel and is not an uncertainty estimate over independently trained seeds or new canonical-task populations.

Report differences in percentage points: every M arm−B0, M2−M1, M3−M1, M4−M2 and M4−M3. The factorial estimands are:

- Latent−pixel main effect: `((M2−M1) + (M4−M3)) / 2`.
- Coupling main effect: `((M3−M1) + (M4−M2)) / 2`.
- Interaction: `(M4−M2) − (M3−M1)`.

Apply these formulas to each common bootstrap replicate. Do not assemble a factorial estimate from an incomplete arm matrix. Staggered completed-arm tables may be labeled interim; final factorial output waits for all five fully audited panel matrices. Report paired win/loss/both-success/both-failure counts for pairwise contrasts. Intervals are prespecified descriptive uncertainty intervals, without multiplicity correction or confirmatory significance claims. No p-value shopping or declaring a winner from the largest observed contrast.

Robustness ratio is `SR(1K perturbed) / SR(clean paired-2000)` per arm. These are different task distributions, so the aggregate ratio is descriptive and not a matched perturbation causal effect. Combine draw k from the independently seeded panel streams and retain common arm draws. A zero observed denominator yields an undefined point estimate. If any bootstrap denominator is zero, report the count and withhold that ratio's interval; never silently discard or replace those replicates. Ratio differences and ratio-factorial effects are outside v1.

Prespecified secondary tables: suite; all seven perturbation categories; difficulty 1–5 and the original Long/unlabeled category without imputation; suite×category counts; and dev-overlap versus additional-790. For one-way subgroup contrasts resample only that subgroup's variant tasks, seed 7, 10,000 common-index draws. Report n and raw counts, including small/empty cells. Suite×category tables are descriptive counts/rates without extra hypothesis tests. No post hoc regrouping or broad subgroup superiority claims.

## Latency and reporting

Measure wall time around the whole deployed `predict_chunk`, including image/token preprocessing, host-to-device transfer, backbone, future core and refinement for M3/M4, action decoding and CPU return. Synchronize the GPU at timing boundaries in the Week 9 harness for every arm. Measure one complete eight-action query; do not present query milliseconds as per-action latency. Model loading, environment stepping, renderer time and disk writes are separate wall-time quantities. Online inference excludes training-only teacher/decoder work but includes every component that deployed inference actually uses.

Keep all recorded query latencies, including first queries; no warm-up deletion, tail trimming, success-only filtering or timing on Lambda for comparison. Report count, pooled median/p95, first-query median/p95, subsequent-query median/p95 and equal-episode-weight median/p95 of episode-median latency. Separately report episode wall time and evaluation wall time. Audit latency arrays against HDF5, distinguish absent failed queries, and report GPU memory peaks and parameter counts. This measures evaluator-side sequential inference, not a real-robot closed-loop deadline guarantee.

The final package includes source/runtime/cohort/endpoint and panel identities; all raw rollouts, partial attempts and receipts; input-audit detail; per-arm counts and failures; all fixed contrasts and intervals; clean/perturbed rates and ratios; subgroup tables; query latency; model/optimizer/CPT/pilot/training costs; exposure and hardware differences; and the JEPA, renderer, single-seed and development-overlap limitations. Keep training losses separate from rollout performance. No inference about representation quality from unlike pixel/latent loss magnitudes.

A protocol amendment must be a new version stating its rationale, timing and which outcomes were already visible. Never overwrite v1 or use an amendment to choose tasks, checkpoints or statistics based on favorable results. Freezing this protocol does not implement or launch the later five-arm Week 9 orchestrator.
