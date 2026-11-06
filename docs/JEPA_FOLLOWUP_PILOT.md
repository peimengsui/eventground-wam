# Training-only JEPA diagnostic and residual pilot

**Current disposition, October 2, 2026:** the user requested moving on without
further repair. This completed pilot is archived; the correction-strength sweep,
learned gate, larger confirmation run and further JEPA retraining are **deferred**.
The [project-progression exception](evaluation/week6-progression-v1/decision.json)
approves the original fully trained absolute JEPA endpoint for M2/M4 research
use, with its original gate still FAIL. Neither pilot checkpoint enters the
main matrix. The measured pilot result and screen below remain unchanged; their
earlier recommendation for another experiment is no longer the active plan.

**Completed 2026-10-02.** The training-only diagnostic and matched residual comparison both finished, and the independent audit passed. The frozen outcome is **GO_FOR_FURTHER_TRAIN_ONLY_VERIFICATION**: every predeclared pilot screen passed. This is a separate follow-up to the closed Week 6 experiment. The Week 6 candidate remains rejected, and no accepted checkpoint or production run was created.

The residual approach merits another bounded confirmation experiment. It improved average errors and passed action/token-use checks, but its small-change front/h2 slice still regresses against copy and it has a substantial fitting/generalization gap. The result does not resolve the original acceptance failure conclusively.

## Results on groups unseen by the two fresh optimizers

Errors below are cosine errors, with lower being better. These are **training-split pilot results**, not validation/test results. The same 80 evaluation groups and 1,600 dual-camera pairs were used for every row.

| Measurement | Copy | Fresh absolute control | Copy residual | Residual gain vs copy |
| --- | ---: | ---: | ---: | ---: |
| Overall | 0.253028 | 0.325450 | 0.217195 | 14.16% [13.64%, 14.67%] |
| Horizons 8/16/32 | 0.286747 | 0.325693 | 0.235366 | 17.92% [17.48%, 18.35%] |
| Front, horizon 2 | 0.147771 | 0.330304 | 0.147152 | 0.42% [−1.93%, 2.52%] |

Bracketed values are the predeclared paired group-bootstrap 95% intervals. Overall residual improvement versus the fresh control was **33.26% [32.60%, 33.94%]**. All ten camera/horizon point estimates improved versus both copy and control. Front/h2 is effectively copy parity at this precision; its interval includes a regression. The fresh control is weak at this short budget: its overall error is 28.62% worse than copy and its pooled effective ranks are only 3.50/2.72. This comparison demonstrates an advantage at this initialization and budget, not superiority over a fully trained absolute model.

The residual arm's overall fitting error was **0.144099**, compared with **0.217195** on unseen training groups (absolute gap **0.073096**). More independent fitting groups and replication are more informative next steps than simply repeating this small set longer.

![Matched pilot results](evaluation/jepa-followup-v1/pilot-results.png)

## What the diagnostic established

The old production candidate reproduces the front/h2 weakness on this training panel. It has front/h2 error 0.152908 versus copy 0.147771, a 3.48% regression. This is consistent with the earlier validation finding, but the two numbers come from different populations and cannot be substituted for each other.

| Front/h2 RGB-change slice | Examples / tasks | Production v1 vs copy | Residual vs copy |
| --- | ---: | ---: | ---: |
| Low, at most 0.01 | 88 / 32 | 27.61% worse | 9.05% worse |
| Medium, above 0.01 through 0.05 | 195 / 37 | 8.14% worse | 2.46% worse |
| High, above 0.05 | 37 / 8 | 28.31% better | 17.06% better |

Each slice uses equal task/group weighting within the available slice, so slice means need not combine by sample count into the overall mean. The small-change weakness is reduced but persists. Motion-bin performance was diagnostic, not a condition of the frozen pilot screen; passing the screen does not erase these regressions. Production v1 saw these groups in its original training, while the fresh residual did not, so this table diagnoses patterns rather than ranking equally trained competitors.

**Head-capacity hypothesis:** both absolute heads have a centered span of rank 257. For production v1, the optimistic front/h2 future error bound is **0.037106**, far below copy's 0.147771; none of the 320 front/h2 examples has a bound above copy. Therefore this probe does not establish an unavoidable span bottleneck behind the original failure. For the short-budget fresh control, the bound is **0.210668**, above copy in 282/320 front/h2 examples. Its fitted head is restrictive at this checkpoint, but this is not evidence that the architecture cannot learn a better span with more training. The residual bypass avoids needing to reconstruct every current feature through that small learned head.

**Variance-conflict hypothesis:** on the fixed fitting batch, production v1's weighted-variance gradient norm is 0.331 times the cosine gradient norm, with whole-model alignment **+0.090** and front/h2 alignment **+0.148**. Its head-only alignments are negative (−0.294 and −0.247), so there is local tension but no evidence here of whole-model variance dominance. The fresh control has norm ratio **1.737**, alignment **−0.422**, and front/h2 alignment **−0.401**. The residual has ratio **0.659**, alignment **−0.367**, and front/h2 alignment **−0.034**. These are local snapshots, not causal experiments. No variance-weight change is justified by this pilot alone; both arms kept the original weight of 10.

**Learned-path use:** in the residual arm, zeroing actions increases h8/16/32 error by **0.009776**, and swapping core tokens with another numeric group in the same task increases it by **0.007360**. Both exceed the frozen 0.001 threshold. Action reversal increases error by 0.000971 and zeroing the explicit horizon increases it by 0.008024. Zeroing actions changes both output features (mean absolute change 0.003373) and core tokens (0.466262). This supports predictive use of the learned path; it does not establish that a downstream policy benefits from the tokens.

The residual passes both camera collapse checks. Front/wrist prediction effective ranks are **10.12/11.25**, compared with teacher ranks **13.89/16.56**. Fixed-position standard-deviation ratios are approximately **0.923/0.994**, and within-task cosine similarities remain within the declared bounds. Because the model has a direct current-feature path, output non-collapse alone is not proof of useful core tokens; the token-swap check provides complementary evidence.

## Original pilot decision and deferred next experiment

Steps 1 and 2 are complete. The copy-residual design remains a **research candidate
for possible future verification**, currently deferred. It has not produced a
gate-passing checkpoint. Week 6 remains closed with its original failed gate;
the later progression exception approves only the original production endpoint
for the controlled downstream study.

If this residual direction is reopened later, freeze a larger, more diverse
training-only confirmation panel and repeat the matched comparison across
multiple seeds. Keep fitting/evaluation numeric groups disjoint, explicitly
track the low-change front/h2 slice, and repeat action/token-use checks. Record
the extra current-teacher inference cost. Any v2 production proposal requires
its own registered scope. Avoid tuning on the already consulted pilot-evaluation
groups and disclose the prior Week 6 validation exposure; test payloads remain
locked. This follow-up did not launch another experiment.

## Frozen design

The contract was sealed before reading pilot image payloads or optimizing either arm: [`jepa_followup_v1.json`](../configs/training/jepa_followup_v1.json), digest `f44590076c657e7ead0dc523e7f9769373611f745438f94b72acfe3e2bdb1ee3`. Selection uses only the existing training split and deterministic metadata ranking. Each of 40 tasks contributes two numeric groups for fitting and two different numeric groups for pilot evaluation. One episode per group and four anchors per episode give 1,600 dual-camera pairs in each role, including horizons 2, 4, 8, 16 and 32. No numeric group crosses roles.

The production v1 candidate is an **in-sample diagnostic reference**: it already trained on these groups. Only the two fresh pilot models support a matched comparison on groups unseen by their optimizers. Neither role is the project's validation or test split.

| Arm | Output | Training in this pilot |
| --- | --- | --- |
| Production v1 reference | Original absolute JEPA head | None; diagnostic reference only |
| Absolute control | Original absolute JEPA head, fresh initialization | 6,000 updates |
| Copy residual | Current teacher features plus a learned delta, normalized | 6,000 updates |

Each fresh arm has the same 4,238,592-parameter core and 526,336-parameter output head (4,764,928 total). Both expose 128 × 256 core tokens. Both use the same core initialization, exact example order, 40-pair logical batch, 10-pair microbatch, AdamW settings, learning-rate schedule and cosine-plus-weighted-variance objective. Each presents 240,000 pairs to the optimizer, reusing 1,600 unique fitting pairs. This deliberately small, repeated-data experiment is not a substitute for full training.

The residual arm computes `normalize(current_teacher + head(tokens) / sqrt(2048))`, with a zero-initialized head. Current teacher features are frozen and detached; future targets never enter conditioning. The additional current-image teacher path and different head initialization are explicit treatment differences. Equal trainable parameter count does not mean equal total inference cost or information access. At initialization the residual arm reproduces copy-latent, within numerical tolerance.

## Diagnostics and screen

The fixed analysis covers camera/horizon errors, low/medium/high RGB-change bins, an optimistic lower error bound for the original output-head span, and cosine/weighted-variance gradient norms and alignment on a fixed fitting batch. RGB change includes camera and appearance changes; it is not a pure physical-motion label. The head bound permits a freely scaled bias and is optimistic: it can establish a limitation of a frozen head but does not establish which architectural remedy is best. A negative gradient alignment is evidence of local objective tension, not proof of a training failure's cause.

Aggregation weights tasks equally, then numeric groups equally. Intervals use 10,000 within-task group bootstrap samples with the frozen seed. Collapse is measured on 80 pilot-evaluation group means. Zero-action, action-reversal, zero-explicit-horizon and same-task/different-group token-swap interventions test use of the learned path.

The frozen screen requires all of the following: front/h2 improvement of at least 5% versus fresh control; overall improvement of at least 5% and medium/long improvement of at least 10% versus copy; no camera/horizon cell more than 2% worse than either copy or control; a positive lower bootstrap bound for overall improvement versus control; the declared collapse criteria; and at least 0.001 medium/long cosine-error worsening under both zero-action and token-swap interventions, with measurable action-induced feature/token changes. Passing means only `GO_FOR_FURTHER_TRAIN_ONLY_VERIFICATION`. Any failure yields `NO_GO_OR_INCONCLUSIVE`.

This screen differs from the closed Week 6 acceptance gate and does not replace it. One paired seed, sparse anchors and two groups per role per task limit the strength of its conclusions. It provides no downstream policy-success evidence.

## Execution and verification

The source archive is `outputs/jepa-followup-v1/source.tar.gz`, SHA-256 `1e8f530558e07c6005b4652ad4514532e0289446517ca77562ee11c9723a8bde`. The isolated remote root is `/home/peimengsui/data/eventground-wam/analysis/jepa-followup-v1`. The service started on Feifei at 2026-10-02 19:59:46 PDT under a two-hour runtime cap, a 24 GiB host-memory limit and no automatic restart. The frozen source is separate from the audit script and report packaging.

Implementation: [`jepa_followup.py`](../src/eventground_wam/jepa_followup.py), [`run_jepa_followup.py`](../scripts/run_jepa_followup.py), [`verify_jepa_followup.py`](../scripts/verify_jepa_followup.py). Unit checks exercise copy initialization, teacher stop-gradient, live action/core gradients, suffix masking, group separation, the head-span bound and equivalence of microbatch gradients to the full logical variance loss. The entire local test suite passed before result analysis.

The run completed at approximately 20:15:17 PDT: 927.6 seconds inside the runner, including 391.7 seconds for control optimization and 400.5 seconds for residual optimization. Peak allocated GPU memory was 4,599,520,768 bytes; host peak RSS was 16,194,953,216 bytes. Persistent run artifacts occupy 10,984,280,264 bytes, below the frozen 16 GiB cap. The service exited successfully with zero restarts and is inactive.

All **484 local tests passed**, and lint passed for the five new Python files. The independent audit verified all 31 sealed run files, every frozen source pin, numeric-group separation, exact matching of the sample-order hash chain, all saved evaluation scalar errors and aggregate intervals, group means, and collapse statistics. Fresh checkpoint replay covered 80 pairs per arm; fresh raw-image replay covered 40 pairs; four intervention replays covered 40 pairs each per fresh arm. These replays reproduced predictions, tokens, features and intervention values exactly. Independent head-span and collapse recomputations differed by less than 1.5e−14. Group-mean differences were below 1.1e−7.

The first auditor attempt completed its numerical checks but could not serialize a NumPy scalar into the strict JSON receipt. Its log was preserved. Converting receipt scalars to native JSON types fixed that packaging issue; the complete audit then reran and passed. No model, measurements, contract, threshold or training data was changed.

Evidence: [run report](evaluation/jepa-followup-v1/report.json), [independent verification](evaluation/jepa-followup-v1/verification.json), [diagnostic analysis](evaluation/jepa-followup-v1/analysis.json), [PDF figure](evaluation/jepa-followup-v1/pilot-results.pdf), [completion ledger](evaluation/jepa-followup-v1/followup-completion.json). Full per-example JSON measurements remain in `outputs/jepa-followup-v1/results/`; large feature banks, prediction arrays and the two diagnostic checkpoints remain under the remote `run/` directory. These checkpoints are explicitly marked diagnostic and are not promoted into production.

The production v1 candidate SHA remains `be2f863a2092668c3e5d50e62d0e41415115f7f582f4f23fa04bc74df8b38e2e`. Week 6's decision digest remains `ee4bc5f78cd197ebf7398ae0739ab0b73a7021fccad939fbad356be19c51fc70`, and its completion digest remains `1958714567031b78349b1bea8a182fefc2368fa20442fb9e630dd3481baa499a`. The follow-up read no validation or test image payloads.
