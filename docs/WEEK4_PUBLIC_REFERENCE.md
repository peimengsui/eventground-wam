# Week 4 public reference comparison

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
| Simulator source | LIBERO-Plus revision `4976dc30028e805ff8094b55501d532c48fec182`; canonical clean BDDL files, not perturbation-suite task indices. |
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
| Image preprocessing | Pinned Qwen processor; 256×256 with no resize or crop | Native OpenVLA processor with center crop enabled |
| Action / state processing | Train-only project action quantiles; raw 8D proprioception input; thresholded gripper logits | Checkpoint's per-suite normalization statistics and native proprioception handling; official OpenVLA gripper conversion |
| Software | B0's separately pinned Qwen runtime | Reference's separately pinned OpenVLA Transformers fork and TensorFlow image processing |

Input semantics and simulator conditions are matched; model weights, capacity,
training, prompt tokens, processed image tensors and normalization statistics
are not. We preserve the correct native inference path for each policy. This
comparison cannot attribute a performance difference to any single architecture
or training choice, or establish held-out generalization for the public model.

## Execution and evidence

Status: evaluator implemented; run and paired verification pending.

The host-owned user systemd service uses an output lock, restart-on-failure and
episode-level resume. Completed artifacts are checked before reuse. User
lingering keeps execution independent of SSH and allows startup after reboot.
The completed report is preserved on subsequent invocations; only verification
is repeated. The service will be disabled after successful completion.

Source entry point: [`evaluate_reference_clean.py`](../scripts/evaluate_reference_clean.py).
The immutable clean-v1 protocol retains its historical B0 policy profile; each
reference selection/report separately records the actual reference policy profile.
