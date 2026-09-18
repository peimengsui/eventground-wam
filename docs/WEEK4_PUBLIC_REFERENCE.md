# Week 4 public reference comparison

**Completed and verified:** 2026-09-17. Public reference **39/40 (97.5%)**;
B0 **26/40 (65%)**, a **32.5 percentage point** reference advantage on this panel.

| Suite | B0 | Installed OpenVLA reference | B0 minus reference |
| --- | --- | --- | --- |
| Spatial | 9/10 (90%) | 10/10 (100%) | −10 pp |
| Object | 8/10 (80%) | 10/10 (100%) | −20 pp |
| Goal | 5/10 (50%) | 9/10 (90%) | −40 pp |
| LIBERO-10 | 4/10 (40%) | 10/10 (100%) | −60 pp |
| **Overall** | **26/40 (65%)** | **39/40 (97.5%)** | **−32.5 pp** |

Paired outcomes: 26 both successful, 13 reference-only successes, zero B0-only
successes, and one shared failure. Both policies timed out on Goal task index 1,
“open the top drawer and put the bowl inside.” The other 13 B0 timeouts were
successful for the reference. These are outcomes, not a diagnosis of the failure
mechanisms. Goal and LIBERO-10 have the largest observed deficits.

Wilson 95% intervals: B0 **49.5–77.9%**; reference **87.1–99.6%**. These describe
the 40-episode panel, not variability across unseen initial states or training
seeds. This result does not support a claim of near-parity with this reference;
it also does not isolate architecture or training effects.

## Scope and interpretation

This comparison runs the already installed
[`Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata`](https://huggingface.co/Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata/tree/a85655ec941bae6644c9fbdf62db02b9726d7cf5)
on the exact frozen clean-40 panel previously used for full B0. The reference
checkpoint revision is `a85655ec941bae6644c9fbdf62db02b9726d7cf5`; its native
inference source is pinned to `e4287e94541f459edc4feabc4e181f537cd569a8`.
The checkpoint's bundled model card calls it LIBERO-Plus mix-SFT / OpenVLA-OFT+;
this name does not mean the FiLM-enabled OFT+ configuration from other work.
The installed wrapper uses `use_film=False` and the checkpoint's own L1 head.

This is an **evaluation-matched external reference**, not a configuration- or
training-matched controlled baseline. B0 remains the initialization and control
for later project experiments. Neither a paper headline nor a previous perturbed
sanity-35 result substitutes for this measured clean-panel comparison.

This document does **not** revise or pass the roadmap's approximately five
percentage point gate. Establishing a configuration-matched comparator, or
explicitly revising that requirement with the user, remains open. One fixed trial
per task is a diagnostic panel; uncertainty and per-suite results must accompany
the aggregate. No larger evaluation or new training is included in this run.

## What is matched

| Dimension | Shared setting / verification |
| --- | --- |
| Tasks | All 40 canonical tasks: ten each from Spatial, Object, Goal and LIBERO-10. Same ordered, sealed [manifest](../evaluation/splits/libero-clean-v1/clean-40.json); no task filtering. |
| Simulator source | LIBERO-Plus revision `4976dc30028e805ff8094b55501d532c48fec182`, MuJoCo 2.3.7 and robosuite 1.4.0; canonical clean BDDL files, not perturbation-suite task indices. |
| Initial conditions | Initial-state index 0 for every task; same initial-state file hashes, array hashes, environment seed 0 and task-specific episode seeds. |
| Instructions | Same canonical instruction strings. Model-specific prompt wrappers differ. |
| Raw observations | Agent and wrist RGB, 256×256, rotated 180 degrees; identical 8D end-effector/gripper state contract. Each policy then uses its native preprocessing. |
| Control | Panda / OSC_POSE, 20 Hz, ten identical settling actions, 8×7 action chunks and eight-step replanning. |
| Episode limits | 220 / 280 / 300 / 520 policy steps for Spatial / Object / Goal / LIBERO-10, excluding settling. |
| Outcomes | Environment success signal; timeouts, policy exceptions and invalid actions count as failures. One documented infrastructure retry allowed. |
| Recording and metrics | Same rollout/recorder implementation, complete HDF5 arrays, task success rate and Wilson 95% interval; per-suite and paired task outcomes. |
| Hardware | Same Feifei RTX 5090, sequential runs. Separate pinned policy runtimes. |

The shared clean inventory builder was extracted unchanged from B0's evaluator.
The reference entry point regenerates the inventory and refuses to run unless
it equals the frozen manifest. It also checks the original B0 report against its
committed sealed record and validates all 40 B0 artifact hashes, outcomes,
instructions, seeds and initial states before loading OpenVLA.

After the run, paired verification checks every reference artifact and compares
all evaluation metadata with B0. It additionally compares both camera arrays,
proprioception and simulator state throughout settling and at the first policy
input. Subsequent observations naturally differ as the policies act differently.
Any observed input differences are reported rather than assumed away.

## What differs or is not established

| Dimension | B0 | Installed public reference |
| --- | --- | --- |
| Backbone / prior training | Qwen3.5-2B vision-language initialization | OpenVLA 7B action-policy lineage; bundled adapter configuration names the joint four-suite OpenVLA-OFT checkpoint as its base |
| Architecture | Project conditioning projections and action-query Transformer; upper-layer rank-16 LoRA | OpenVLA native token representation, L1 regression head and proprioception projector; bundled adapter configuration has rank 32 and different target modules |
| Training data | Frozen project split: 11,521 training episodes, 1,715,954 valid chunks | LIBERO-Plus mix-SFT per its bundled model card. Exact sample membership, split overlap and matched exposure budget have not been established |
| Training budget / selection | Five epochs, 8,579,770 sample exposures, minimum full-validation objective selected epoch 5 | Installed released snapshot and `150000`-named heads. The filename alone does not establish equivalent training exposure or checkpoint-selection rules |
| Objective | Continuous-action L1 plus gripper BCE | Checkpoint's native L1 regression action head, including gripper output |
| Image preprocessing | Pinned Qwen processor; 256×256 with no resize or crop | Native OpenVLA resize followed by a center crop covering 90% of image area |
| Action / state processing | Train-only project action quantiles; raw 8D proprioception input; thresholded gripper logits | Checkpoint's per-suite normalization statistics and native proprioception handling; official OpenVLA gripper conversion |
| Software | Transformers 5.17.0; Pillow 12.3.0 | OpenVLA Transformers 4.40.1 fork, TensorFlow 2.15.0, Pillow 11.3.0; separate runtime lock |

Input semantics and simulator conditions are matched; model weights, capacity,
training, prompt tokens, processed image tensors and normalization statistics
are not. We preserve the correct native inference path for each policy. This
comparison cannot attribute a performance difference to any single architecture
or training choice, or establish held-out generalization for the public model.
Both host runtimes use Torch 2.13.0 / CUDA 13.0 and NumPy 1.26.4. Runtime package
pins do not imply identical model computation, and no cross-model numerical
equivalence is claimed.

## Execution and evidence

All 40 reference HDF5 files passed hash, schema, numeric and outcome validation.
All 40 episode pairs matched the frozen evaluation metadata. Both raw cameras,
proprioception and simulator state were **exactly equal through settling and at
the first policy input for every pair**. The shared rollout, recorder, protocol,
BDDL resolver and sensor helper files are byte-identical to B0's run source;
the extracted clean-inventory functions have identical syntax trees.

The GPU run started at **20:51:06 PDT** and finished its fortieth episode at
**20:55:34 PDT** on 2026-09-17: approximately **4 min 28 s**. Results include one
policy timeout, zero episode exceptions, zero episode retries, and zero partial
artifacts. The 40 HDF5 files occupy **1,161,212,483 bytes** (about 1.16 GB).

The original process then exited with a reporting error: the strict JSON seal
rejected PyTorch's `TorchVersion` string subclass. One automatic service restart
occurred and was stopped during initialization. We preserved all episode files
and the complete progress report, corrected serialization, and finalized and
verified the saved results on CPU. **No episodes were rerun.** The original
exit, automatic restart and report repair remain explicit in the receipts.

Rollout source: `016e42561c0c1f619e33b6bf284c2e6dc9acb01d`.
The report-only fix is committed at `4a9805d`. Local validation: **225 tests
passed**, including a regression test using the actual `TorchVersion` type;
lint passed. The deployed reference runtime passed its 34 relevant tests before
launch. The original run checkout and both policy environments were preserved.

Evidence:

- [Reference report](evaluation/week4-reference-clean40-v1.json).
- [All 40 paired artifact checks](evaluation/week4-reference-clean40-paired-v1.json).
- [Checkpoint and manifest selection](evaluation/week4-reference-clean40-selection-v1.json).
- [Host launch and source-parity receipt](evaluation/week4-reference-clean40-launch-v1.json).
- [Report recovery receipt](evaluation/week4-reference-clean40-report-recovery-v1.json).
- [Completion receipt](evaluation/week4-reference-clean40-completion-v1.json).
- [Historical service/restart audit](evaluation/week4-reference-clean40-service-audit-v1.json).

Host output directory:
`/home/peimengsui/data/eventground-wam/reference-policy/evaluations/week4-clean40-v1`.

The host-owned user systemd service uses an output lock, restart-on-failure and
episode-level resume. Completed artifacts are checked before reuse. User
lingering keeps execution independent of SSH and allows startup after reboot.
The completed report is preserved on subsequent invocations; only verification
is repeated. The service is now stopped and disabled.

Source entry point: [`evaluate_reference_clean.py`](../scripts/evaluate_reference_clean.py).
The immutable clean-v1 protocol retains its historical B0 policy profile; each
reference selection/report separately records the actual reference policy profile.
