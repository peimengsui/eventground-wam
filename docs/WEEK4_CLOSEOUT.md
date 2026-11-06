# Week 4 closeout and shared B0

**Closed:** September 24, 2026, with a user-authorized milestone acceptance
revision. **Next:** [Week 5 pixel temporal CPT](WEEK5_STATUS.md).

The user chose to stop optimizing B0 for now and continue the controlled study
using the best completed Week 4 setup. The project milestone is closed; the
previous statistical acceptance results are preserved. This decision does not
establish equivalence to OpenVLA or pass a previously inconclusive rule.

## Frozen reference

The shared baseline is **fresh all-L1 trajectory-v2 B0, fixed epoch 2**:

| Measurement | Shared B0 | Installed OpenVLA-OFT+ |
| --- | ---: | ---: |
| Clean-2000 | **1,867/2,000 (93.35%)** | 1,935/2,000 (96.75%) |
| LIBERO-Plus dev-210 | **162/210 (77.14%)** | 170/210 (80.95%) |
| Clean Spatial / Object / Goal combined | 1,469/1,500 (97.93%) | 1,469/1,500 (97.93%) |
| Clean Long | 398/500 (79.60%) | 466/500 (93.20%) |

The dev point estimates are descriptive: one pair fails the old exact-image
validity rule. The [evaluation audit](WEEK4_FRESH_V2_FAILURE_ANALYSIS.md)
reproduced rendering nondeterminism with unchanged physics. Both policies
succeeded on that case; the original recordings and validity decision remain.

The checkpoint is on **Feifei**:

```text
/home/peimengsui/data/eventground-wam/libero-plus/derived/b0/week4-fresh-all-l1-v2/epochs/epoch-02/B0.ckpt
SHA-256: 7a385fea68fe5d14f642b31e18f8e2e166ccd984d9b4f1ff5b68484191a9b720
```

Its bytes were rehashed on the host at closure. This policy checkpoint depends
on the pinned base Qwen weights, processor and runtime; it is not a standalone
copy of those dependencies. The sealed
[completion and handoff record](evaluation/week4-completion-v1.json) pins the
checkpoint, model manifest, runtime lock, recipe, split, normalizers, training
report, evaluation identities and evidence. Subsequent runs must verify these
identities and write child artifacts to separate output directories.

Shared controls inherited by later policy experiments:

- **Qwen3.5-2B**, revision `15852e8c16360a2fea060d615a32b45270f8a8fc`;
  OFT action head, state projector and broad rank-32, alpha-16 LoRA.
- Front and wrist RGB at 256×256, eight-dimensional proprioception, current
  preprocessing, train-only action/state normalization and no image augmentation.
- Mean **all-action L1** over eight actions × seven dimensions, including signed
  gripper regression; predict eight / execute eight at 20 Hz.
- Trajectory-v2: **11,435 train episodes / 1,341 numeric groups**; 1,706,022
  valid chunks per epoch. Validation is 1,528 episodes / 180 groups. Test remains
  locked. Numeric groups, not appearance variants, define split independence.
- Baseline training: **117,132 updates / 3,412,044 presentations**, fixed epoch-2
  selection, mixed-task batches, seed 17. The full optimizer/schedule and sample
  order are in the [frozen recipe](../configs/training/week4_b0_fresh_all_l1_v2.json).

## Acceptance revision and completed work

The earlier requirement to resolve the external-reference performance gate
before world-model work is replaced **for project progression only** by acceptance
of this completed, audited B0 as the internal control. No old rule, threshold,
result, or sealed diagnostic is rewritten.

| Item | Closure disposition |
| --- | --- |
| Shared training recipe and infrastructure | Complete; interruption/resume, deterministic replay, checkpoint reload and host-owned execution verified in the training evidence. |
| Full B0 training and validation | Complete; two epochs and full validation at both endpoints. Final validation all-L1 0.139842, gripper accuracy 94.6698%. |
| Clean comparison | Complete; current B0 gap −3.40 pp, paired 95% interval [−7.00, −0.65] pp. Original clean rule remains **INCONCLUSIVE** for this checkpoint. |
| Dev-210 and failure analysis | Complete measurements and audit; 209/210 exact paired inputs. V2 readiness remains **NOT_ASSESSED_INPUT_MISMATCH**, and unchanged outcomes would still be inconclusive. |
| Historical five-epoch B0 | Original clean **FAIL** remains unchanged: 68.25% clean / 50.48% dev. |
| Checkpoint and study handoff | Complete via the checksum-pinned completion record and Week 5 entry plan. |

Known weaknesses are research targets: drawer closure, microwave placement and
closure, completing the second object, and instruction paraphrases. The three
target Long tasks account for 55 of the 68 net clean failures relative to the
reference. These observations motivate temporal modeling but do not establish
its causal benefit or prove that model size explains the gap.

## Fair comparisons after closure

**Yes: later methods can fairly compare against this B0.** OpenVLA remains useful
external context; matching its model size or pretraining is not a prerequisite
for testing our own methods.

1. Keep this checkpoint as the fixed practical reference. The subsequently
   approved [two-total-action-epoch rule](PHASE2_TRAINING_PROTOCOL.md) initializes
   M1–M4 from the same pinned base Qwen and fresh shared components used by B0.
2. Match action-training examples/order, updates, optimizer handling, seeds,
   observations, action head, normalization and rollout conditions across arms.
   Freeze any necessary interface changes and include them in matched controls.
3. Give every B0/M1–M4 policy **two total action-training epochs**, with identical
   effective batches, sample order and 117,132 optimizer updates. The existing
   B0 is the completed control. This replaces the closure-time continuation
   proposal; no extra B0 continuation is required for this comparison.
4. Pixel and JEPA use identical source trajectories, future pairs and horizons.
   Match core predictor capacity and exposure; report decoder overhead, total
   compute, trainable parameters and inference latency separately. Shared-backbone
   CPT updates require a declared extra-training control or a narrower claim.
5. Report paired success differences and uncertainty, including Long and the
   seven perturbation categories. Clean-2000 and dev-210 are already inspected
   development evidence; reserve the larger scheduled panels for confirmation.
6. Before the next formal paired rollout, freeze a prospective rendering-validity
   policy or establish exact reproducibility. Any new tolerance must be justified
   independently and applied symmetrically; it cannot retroactively pass Week 4.

The proposed [C/S late-stage sampling screen](WEEK4_STAGE_SAMPLING_AUDIT.md),
more frequent replanning, longer B0 training and further baseline architecture
changes are **deferred**. Their existing plans remain intact, with no runs
launched. Reopening them would require a versioned baseline decision and matched
re-evaluation, rather than mixing reference checkpoints within the study.

The sealed completion record preserves the original closure-time forward plan.
Its document bytes are retained under `evaluation/week4-completion-v1-snapshots/`;
the new training-control contract records this later user-authorized revision.
