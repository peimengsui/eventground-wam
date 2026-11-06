# Week 6 — JEPA temporal CPT

**Planning started:** September 26, 2026. **Closed:** September 28, 2026.
**Project progression revision: October 2, 2026. Status: fully closed; original
JEPA approved for downstream research use by explicit exception.** All six
steps, the final decision, results packaging and training-only follow-up are
complete. **The original frozen gate remains FAIL.** M2/M4 may use the original
byte-identical endpoint as `C_jepa-study-v1`; further repair is deferred.
See the [current closeout and fairness limits](WEEK6_CLOSEOUT.md),
[progression decision](evaluation/week6-progression-v1/decision.json),
[decision](evaluation/week6-final-v1/decision.json) and
[completion ledger](evaluation/week6-completion-v1.json).
The [target-feature contract and extractor](WEEK6_JEPA_TARGETS.md) passed the
600-pair training-only feasibility check and independent numerical audit.
The accepted v2 path encodes each camera image separately; the failed batched
v1 is retained. [Step 2 baselines and criteria](WEEK6_JEPA_BASELINES.md) passed
the resource pilot, full validation evaluation and independent numerical audit.
The measurement protocol and acceptance thresholds were frozen before scoring.
[Step 3](WEEK6_JEPA_PREDICTOR.md) passes the controlled and real training-only
diagnostics, independent replay and all 458 local tests. Its exact Pixel core
and separate JEPA head reduce real diagnostic cosine error to 0.132110 versus
0.231251 for copying current features. Earlier failed checks are retained.
[Step 4](WEEK6_JEPA_TRAINING.md) passes the full-path resource/recovery pilot and
independent audit, together with all 471 local tests. The frozen production recipe
keeps Pixel's exact order/schedule and uses verified logical-batch JEPA gradients
and a bounded RAM target cache. Projected training time is 34.44 hours including
a 1.5× margin. [Step 5](WEEK6_JEPA_PRODUCTION.md) started fresh production on
September 27 at 00:15:20 PDT and completed at 20:57:28 PDT with zero restarts.
Independent verification passes all 8,221,365 pairs, 221,633 optimizer updates,
the complete sample chain, frozen weights and exported candidate equality.
Held-out acceptance fails because front-camera horizon-2 error is 7.22% worse
than copy, above the 2% cap; the other nine camera/horizon cells pass.

## Goal and entry point

Train an action- and horizon-conditioned predictor of future visual features,
then freeze **C_jepa** for M2 (JEPA auxiliary VLA) and M4 (JEPA-WAM).
The Week 6 gate is better future prediction than copying current features,
with no representation collapse. Policy success is measured in later weeks.

[Week 5 is complete](WEEK5_PIXEL_VALIDATION.md): the accepted `C_pixel-v1`
passed all frozen validation gates. It supplies the comparison module and the
shared data/training contract. This plan implements the intent of the
[original Week 6 roadmap](EventGround-WAM_12-Week_Execution_Plan.md#week-6--jepa-temporal-cpt)
under the subsequently frozen fixed-horizon and Phase 2 contracts.

## High-level execution steps

1. **Complete — freeze the JEPA representation contract.** The accepted
   [Qwen vision target v2](WEEK6_JEPA_TARGETS.md) uses frozen post-merger image-only
   features: 64 tokens per camera, 2,048 channels, per-token non-affine LayerNorm
   and unit L2. Single-image execution removes the batch dependence found in v1.
   Current-observation conditioning remains matched to Pixel. All 600 train-only
   diagnostic pairs, independent arithmetic verification, 426 local tests and
   new-code lint pass. No validation/test payloads or optimizer updates were used.

2. **Complete — establish baselines and freeze success criteria.** Copy-latent
   and the sampled train-only mean were measured on all 1,109,994 validation
   pairs, with fixed retrieval pools, motion diagnostics, all camera/horizon
   breakdowns and group-bootstrap intervals. Overall cosine error is
   0.247452/0.223306 (copy/mean); retrieval Recall@1 is 20.87%/14.54%.
   All 3,240 queries qualify across all 180 groups and 40 tasks. The future
   candidate must beat both controls, reach at least 25.8727% Recall@1, and pass
   the frozen uncertainty, motion, camera/horizon and collapse requirements.
   Independent verification and all 438 local tests passed. The complete run
   took 1 hour 45 minutes with 0.68 GiB peak Torch GPU allocation; no predictor
   training or test-payload reads occurred. See the
   [completion ledger](evaluation/week6-step2-completion-v1.json).

3. **Complete — implement and verify the matched predictor.** Reuse Pixel's exact
   4,238,592-parameter core, initialization and conditioning, retaining the
   128 x 256 future-token interface. The separate 256-to-2048 JEPA head has
   526,336 parameters. Cosine loss and an anchor-grouped variance penalty pass
   controlled action-order/horizon checks and the matched 20-pair real overfit
   check. After 1,600 diagnostic updates, cosine error is 42.87% below copy.
   Target isolation, unchanged frozen encoders, active finite gradients and
   exact GPU reload pass. Independent CPU/NumPy replay and fresh teacher
   encoding pass; all 458 local tests pass. Failed v1/v2 checks and all diagnostic
   exposure are recorded in the [completion ledger](evaluation/week6-step3-completion-v1.json).
   No validation/test payloads were read; the diagnostic checkpoint is not C_jepa.

4. **Complete — freeze the recipe and pass a resource/recovery pilot.** The exact
   common order, logical batches, AdamW schedule and final endpoint are preserved.
   Cosine + 10× grouped variance uses verified two-pass gradients across complete
   logical batches, including singleton tails. Canonical teacher targets are
   encoded online with an exact FP32 pool-local RAM cache (1.17 GiB maximum).
   The 128-update pilot and independent actual-checkpoint/trace audit pass;
   SIGKILL recovery is bitwise exact. Full-device GPU peak is 6.47 GiB, host RSS
   peaks at 5.32 GiB and the conservative full-pass projection is 34.44 hours.
   All 471 local tests pass. See the [completion ledger](evaluation/week6-step4-completion-v1.json).
   Pilot weights are excluded from production; no validation/test payloads were read.

5. **Complete — fresh production JEPA CPT.** The exact passed source/recipe/runtime
   finished on Feifei September 27 at 20:57:28 PDT, after 20.70 active hours with
   zero restarts. Independent CPU verification of the actual final payload passes
   all 8,221,365 unique pairs and their sample chain, 221,633 optimizer updates,
   complete recovery state, unchanged frozen weights, fresh initialization and
   export/reload weight equality. Peak allocated GPU memory was 4.63 GiB and peak
   process host RSS was 6.21 GiB. See the
   [completion ledger](evaluation/week6-step5-completion-v1.json).
   `C_jepa-candidate.pt` remains unaccepted after the failed Step 6 gate.

6. **Complete — validation, review and final disposition; acceptance FAIL.** Full validation finished
   September 28 at 00:35:51 PDT with all 1,109,994 pairs and 3,240 fixed queries.
   Overall error improves 28.61% versus copy, Recall@1 reaches 64.09%, and both
   cameras pass collapse checks. Front h=2 regresses 7.22% (95% interval
   6.22–8.25%) against a 2% cap. Independent metric verification passes, all 480
   fixed examples and 120 supplemental front h=2 examples were visually reviewed,
   and the Pixel/JEPA appearance diagnostic and its independent audit are complete.
   JEPA has lower error/drift in the shared teacher output space, while Pixel's
   native future-token drift is lower in all four perturbation point estimates.
   Appearance strengths were fixed before the diagnostic but after validation;
   they are not prospective acceptance criteria. All 480 local tests pass.
   See the [validation record](WEEK6_JEPA_VALIDATION.md),
   [review](WEEK6_JEPA_CLOSEOUT_REVIEW.md) and
   [completion ledger](evaluation/week6-closeout-review-v1.json).
   The original candidate, reports, failed pilots and thresholds are preserved.
   The [final results package and handoff](WEEK6_CLOSEOUT.md) close the experiment
   with the candidate retained for diagnosis and rejected for promotion.

## Inherited controls

| Item | Week 6 contract |
| --- | --- |
| Data | `trajectory-v2-future-pairs-v1`; 8,221,365 train / 1,109,994 validation pairs, both cameras |
| Horizons | 2, 4, 8, 16, 32 frames at 20 Hz; identical episode boundaries and action intervals |
| Exposure | `trajectory-v2-cpt-order-v1`; one pass, 221,633 updates, exact logical batches and short tails |
| Inputs | Same current observations, instruction, state and intervening demonstration actions as Pixel |
| Initialization | Pinned base Qwen and fresh future-module weights; no B0-trained or Pixel-trained initialization |
| Augmentation | No training augmentation or reweighting, matching Pixel; appearance stress tests are evaluation-only |
| Data isolation | Training-only tuning/pilots and normalization; validation for frozen measurements; test payloads remain locked |
| Compute | Feifei by default, one substantial GPU job at a time; JEPA duration is estimated after its own pilot |

The [Week 5 training record](WEEK5_PIXEL_TRAINING.md) defines the exact common
order. The [Phase 2 protocol](PHASE2_TRAINING_PROTOCOL.md) continues to give
each B0/M1–M4 policy two total action epochs, separately from CPT.

Next-event samples require a separately versioned, validated label set. The
primary Week 6 run therefore keeps the five existing horizons and reports
results for each; separately trained single-horizon and next-event variants are
deferred. This preserves the completed Pixel comparison's source exposure.

## Deliverables and handoff

- A pinned target-feature contract, baseline report and prospective acceptance
  criteria.
- A verified predictor, frozen recipe, resource/recovery evidence and production
  exposure ledger.
- An accepted `C_jepa-v1` checkpoint if the gate passes, with retrieval,
  non-collapse, appearance-consistency and Pixel/JEPA diagnostic reports.

The October 2 [future-module registry](../configs/training/phase2_future_modules_v1.json)
records the explicit Week 7 adjustment: M1/M3 use accepted `C_pixel-v1`, and
M2/M4 use original `C_jepa-study-v1` under the research-use exception. This
satisfies both checkpoint dependencies without changing the failed JEPA gate or
substituting the residual pilot. The OFT-compatible auxiliary feature route
still needs a versioned integration design. Inference
coupling and deployment candidate-action generation remain later work;
demonstration future actions used in CPT must not become rollout inputs.

**Remaining Week 6 work:** None. Additional JEPA repair is deferred. The
training-only follow-up is complete and preserved; this progression revision
launches no further training. **Next milestone: Week 7 M1/M2 integration and pilots.**
