# Week 4: dev-210 results and failure analysis

**Completed:** 2026-09-19. **Checkpoint:** unchanged five-epoch B0, selected epoch 5.

B0 completed **106/210 tasks (50.48%)**, with a descriptive Wilson 95% interval
of **43.77–57.17%**. All **104 failures were policy-step timeouts**. There were
**zero policy exceptions, infrastructure retries, service restarts, or partial
artifacts**. All 210 HDF5 recordings passed hash, schema, numeric, outcome and
provenance validation. All 26 exported preview videos decoded to their expected
frame counts. This completes **Week 4 step 5**; the clean-performance acceptance
rule remains **FAIL**, and Week 4 closure remains open.

The host-owned evaluation ran from **08:38:43 to 09:16:29 PDT**: **37 min 46 s**.
Analysis and previews finished at **09:17:06 PDT**. Both completed services were
disabled afterward to prevent rerunning at reboot. Evaluation files occupy
approximately **8.97 GB / 8.35 GiB** on Feifei. SSH was only an administration
channel; a temporary loss of access did not interrupt the host-owned jobs.

## Results

| Perturbation category | Successes | Rate | Wilson 95% interval |
| --- | ---: | ---: | ---: |
| Background textures | 16/30 | 53.3% | 36.1–69.8% |
| Camera viewpoints | 20/30 | 66.7% | 48.8–80.8% |
| Language instructions | 10/30 | 33.3% | 19.2–51.2% |
| Light conditions | 21/30 | 70.0% | 52.1–83.3% |
| Objects layout | 11/30 | 36.7% | 21.9–54.5% |
| Robot initial states | 11/30 | 36.7% | 21.9–54.5% |
| Sensor noise | 17/30 | 56.7% | 39.2–72.6% |
| **Overall** | **106/210** | **50.5%** | **43.8–57.2%** |

[Category plot and difficulty matrix](../outputs/week4-failure-analysis/dev210-results.png)
 · [PDF](../outputs/week4-failure-analysis/dev210-results.pdf)
 · [Offline video review gallery](../outputs/week4-failure-analysis/review.html).
The gallery and figures are generated local artifacts; raw recordings remain on
Feifei, and compact evidence is committed under `docs/evaluation`.

| Suite | Successes | Rate | Wilson 95% interval |
| --- | ---: | ---: | ---: |
| LIBERO-Spatial | 29/52 | 55.8% | 42.3–68.4% |
| LIBERO-Object | 44/52 | 84.6% | 72.5–92.0% |
| LIBERO-Goal | 18/53 | 34.0% | 22.7–47.4% |
| LIBERO-10 | 15/53 | 28.3% | 18.0–41.6% |

Difficulty levels 1–5 scored **24/42, 25/42, 21/42, 24/42, 12/42** respectively.
Level 5 is weakest overall. Robot initial-state levels scored **5/6, 3/6, 2/6,
1/6, 0/6**. Each category/difficulty cell contains only six different tasks, so
this apparent trend is a diagnostic signal with substantial task-mix uncertainty.

### Interpretation limits

- This is one initial state per frozen perturbed task, not the clean 2,000-pair
  panel. The numerical difference between 68.25% clean and 50.48% dev is **not**
  an isolated causal effect of perturbations. Category task compositions differ.
- Intervals describe this small development panel; correlated base tasks and
  development-set selection limit population-level inference. Category ordering
  is descriptive, not a demonstrated statistically significant ranking.
- OpenVLA has **not** been evaluated on this exact `dev-210` run. Its **96.75%**
  result is from the clean paired panel. No matched dev robustness gap is claimed.
- All 180 non-language dev tasks pass the canonical instruction **plus benchmark
  identifiers**, such as `light 22` or `view 0 0 100 0 0 initstate 270`. The other
  30 tasks use rewritten language. Training metadata identifies 40 canonical
  instructions. This is the frozen evaluator's native task language, preserved
  correctly in the report, but it creates a text-format shift alongside the
  non-language perturbations. Its impact has not been measured. A prompt-only
  control is warranted; it cannot explain the clean-task gap.

## What the failure evidence shows

### 1. The clean gap is concentrated in goal selection and longer manipulation

The completed matched clean evaluation scored **B0 68.25% versus OpenVLA
96.75%**, a **28.5 percentage-point** deficit. Goal and LIBERO-10 contribute
**452 of the net 570 fewer B0 successes (79.3%)**. Object pickup is substantially
stronger: B0 scored 91.0% clean and 84.6% on the dev Object subset.

| Clean task | B0 | OpenVLA | Observation in the selected paired recording |
| --- | ---: | ---: | --- |
| Open middle drawer | 0/50 | 49/50 | B0 approaches the bowl/cabinet area without opening the required drawer; reference opens it. |
| Wine bottle onto rack | 0/50 | 45/50 | B0 stays near the bowl and leaves the bottle in place; reference grasps and places the bottle. |
| Both moka pots onto stove | 0/50 | 40/50 | B0 handles one pot but fails the required placements; reference completes both. |
| Bowl into bottom drawer, close | 7/50 | 46/50 | B0 gets the bowl into the drawer region but does not close the drawer; reference finishes. |
| Open top drawer, put bowl inside | 5/50 | 44/50 | B0 spends most of the timeout around the bowl without completing the sequence. |
| Mug into microwave, close | 7/50 | 45/50 | B0 grasps the correct mug and stalls at the opening; reference inserts it and closes the door. |

[Ten largest clean deficits](../outputs/week4-failure-analysis/clean-largest-deficits.png).
The six paired examples were selected deterministically from the six largest
per-task deficits, using the first reference-only success pair. They illustrate
mechanisms; their labels are not estimated failure-mode frequencies.

A supplementary CPU diagnostic restored every tenth saved policy state, plus
the final saved state, for these 12 recordings. Physics states round-tripped
exactly. It confirmed partial completion in the B0 bottom-drawer example:
`bowl-in-drawer = true`, `drawer-closed = false`. In the B0 microwave example,
`mug-in-microwave` remained false in sampled states despite grasp contacts with
the correct mug. The recorded states are **pre-action**; the final success-making
transition is absent, so a false predicate in the last saved reference state does
not contradict its recorded success. Freshly computed end-effector observations
differed from recorded sensor observations by up to 0.945 mm; these deltas are
retained, and no sensor/state equivalence claim is made.

### 2. Similar failures appear across the perturbation categories

All **14 dev failure contact sheets**, two per category, and **12 clean paired
contact sheets** were inspected by the assistant. Every sheet contains eight
sampled frames. Full MP4s are available for review. This is an assistant review,
not a human annotation study or exhaustive frame-by-frame review.

| Category | Illustrative observations | Preview pair |
| --- | --- | --- |
| Background | Bowl pickup from stove does not complete; wine-bottle task stays near the bowl. | [A](../outputs/week4-failure-analysis/previews/dev-category-1-1.mp4), [B](../outputs/week4-failure-analysis/previews/dev-category-1-2.mp4) |
| Camera | Cream-cheese acquisition fails; another episode transports the bowl but misses successful placement/release. | [A](../outputs/week4-failure-analysis/previews/dev-category-2-1.mp4), [B](../outputs/week4-failure-analysis/previews/dev-category-2-2.mp4) |
| Language | Paraphrased microwave task stalls; “basin atop cupboard” results in moving the bowl to the plate area. | [A](../outputs/week4-failure-analysis/previews/dev-category-3-1.mp4), [B](../outputs/week4-failure-analysis/previews/dev-category-3-2.mp4) |
| Lighting | Bowl ends beside the plate; microwave sequence stalls near its opening. | [A](../outputs/week4-failure-analysis/previews/dev-category-4-1.mp4), [B](../outputs/week4-failure-analysis/previews/dev-category-4-2.mp4) |
| Layout | Cluttered two-mug task completes neither full sequence; displaced bowl acquisition fails. | [A](../outputs/week4-failure-analysis/previews/dev-category-5-1.mp4), [B](../outputs/week4-failure-analysis/previews/dev-category-5-2.mp4) |
| Robot state | Repeated mug approaches fail; orange juice remains outside the basket under changed wrist orientation. | [A](../outputs/week4-failure-analysis/previews/dev-category-6-1.mp4), [B](../outputs/week4-failure-analysis/previews/dev-category-6-2.mp4) |
| Sensor noise | Both-pot sequence fails with blurred front input; tomato-sauce can ends on its side outside the basket with fogged front input. Wrist frames remain clearer. | [A](../outputs/week4-failure-analysis/previews/dev-category-7-1.mp4), [B](../outputs/week4-failure-analysis/previews/dev-category-7-2.mp4) |

Dev samples were ranked by a hash of their artifact hash, with the second sample
preferring a different suite. Selection was not based on the most dramatic
failure. Per-case annotations and source hashes are in the
[visual review](evaluation/week4-failure-visual-review-v1.json).

### 3. Stalling is frequent; action saturation is rare

Across all 635 B0 clean failures, **228 (35.9%)** finish with less than 1 cm
end-effector span during the final 40 steps, despite mean translational-command
norm at least 0.1. The same flag occurs in **32/104 (30.8%)** dev failures.
This supports prioritizing contact, placement and recovery diagnostics, but does
not identify a unique physical cause.

Median gripper-command switches are **8 in B0 clean failures versus 2 in
successes**, and **10.5 versus 2** on dev. Failed episodes run longer, and the
reference's failures also have a median of 8 switches. These counts do not prove
that gripper classification causes the B0 gap.

The mean per-episode continuous-action saturation fraction in B0 failures is
only **0.0059% clean / 0.0090% dev**. Pervasive clipping is therefore a weak leading
explanation, although this does not rule out other normalization issues. Dev
median/p95 policy-query latency is **32.15/33.58 ms**. No numerical-instability
or traceback log lines were found in this dev run.

## Hypotheses and how to test/address them

These are ranked working hypotheses, not causal findings. Multiple mechanisms
can coexist. All proposed B0 changes retain the project's Qwen-only initialization.

| Priority | Hypothesis and evidence | Small diagnostic first | Potential remedy if supported |
| --- | --- | --- | --- |
| 1 | **Weak instruction-to-goal grounding.** Zero-success clean goal tasks, wrong-destination language example, and 10/30 language success. | Hold images/state fixed; compare canonical instruction, meaning-preserving paraphrases and a different plausible target instruction. Then run identical-state canonical-versus-perturbed prompt controls. Measure target/destination choices, not only action differences. | Train semantic paraphrases and target contrasts; strengthen language conditioning with gated/cross-attention or FiLM-style conditioning as a controlled architecture ablation. |
| 2 | **Contact precision, phase transitions and recovery are insufficient.** Correct-object grasps can still stall at insertion, release or closing; 79.3% of the net clean deficit is in Goal/Long tasks. | Evaluate grasp, transport, insertion, release and closure separately; compare saved pre-contact/post-contact states and transition-localized action/gripper error. Test executing 4 or 2 of the predicted 8 actions before replanning under a separately named diagnostic protocol. | Phase-balanced sampling, more placement/release/closing demonstrations, and recovery demonstrations or dataset aggregation. Preserve complete temporal sequences and report rollout gains. |
| 3 | **The five-epoch, narrowly adapted model is undertrained or under-adapted for control.** Validation objective still fell from 0.286897 to 0.273990 in epoch 5; vision/base weights are frozen and rank-16 LoRA touches only attention layers 19 and 23. | Compare existing checkpoints on a fixed diagnostic panel; audit per-task/phase validation loss. Predeclare a small continuation budget separately from a broader-adapter experiment. | More optimization if rollout and validation improve; independently test broader LoRA coverage/rank or selective vision adaptation. More epochs alone are not guaranteed to fix systematic goal errors. |
| 4 | **Uneven data coverage and narrow input variation hurt generalization.** Per-task training counts range 118–416; two-pot and microwave tasks have 118/135 episodes. Recipe augmentation is `none`; layouts and robot states both score 11/30. | Audit task/phase and instruction coverage; run prompt-suffix controls before attributing category losses to vision. Analyze per-dimension proprioception errors and same-task robot-state sensitivity. | Task/phase-balanced sampling; semantic language, calibrated visual and simulator-based state/layout variation. Test train-only per-dimension state normalization separately from the current raw-state LayerNorm projector. |
| Context | **OpenVLA has different capacity, robot pretraining and task-specific history.** B0 is a Qwen 2B model with fresh action components; reference is a 7B action-policy lineage with different adapters and LIBERO training. | Preserve the matching audit; obtain exact reference data/exposure details where possible. Use controlled B0 ablations to isolate changes. | Treat OpenVLA as the operational target and a source of failure comparisons; avoid attributing the entire gap to parameter count or pretraining alone. |

Data scarcity alone is insufficient: the middle-drawer task has **294 training
episodes** and still scored 0/50. Long demonstrations also provide many action
chunks despite fewer independent episodes. More chunks are not the same as more
independent initial conditions or recovery examples.

The original OpenVLA work reports robot pretraining on 970,000 episodes, which
supports the plausibility of a stronger robotics prior, but does not measure
its contribution to this particular gap. [OpenVLA project](https://openvla.github.io/).
OFT reports language-conditioning experiments and strong continuous L1/chunked
policies; both policies here already use eight-action chunks, so neither “add
chunking” nor “L1 is inherently inadequate” follows from our results.
[OpenVLA-OFT project](https://openvla-oft.github.io/).
Recovery-data collection is motivated by imitation learning's shift from expert
states to states induced by the learned policy; it remains a proposed experiment
here. [DAgger paper](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf).

### Recommended sequence

1. **Run a small diagnostic pass before another large training job:** canonical
   versus rewritten/suffixed instructions at identical states; per-task/phase
   validation audit; correct-object versus correct-destination checks.
2. **Choose one controlled training change from those results.** A semantic/task/
   phase sampling variant is a reasonable first candidate; compare against a
   continuation-only control at matched additional sample exposure. Broader
   adapters and visual/state augmentation should be separate ablations.
3. **Select with training-validation data and declared development diagnostics.**
   The inspected dev set is now development evidence, not an untouched final
   test. Keep test trajectories out of tuning and preserve all original scores.
4. **Recheck the fixed clean gate after a declared checkpoint selection**, then
   dev robustness. A future same-panel OpenVLA dev run would supply the missing
   matched robustness comparator; its public headline score cannot substitute.

No retraining, model replacement, prompt cleanup, retry of failed policy episodes,
or acceptance-rule change was performed as part of this analysis.

## Evidence and reproducibility

- [Sealed dev evaluation](evaluation/week4-b0-dev210-v1.json),
  [selection](evaluation/week4-b0-dev210-selection-v1.json), and
  [completion audit](evaluation/week4-b0-dev210-completion-v1.json).
- [Compact analysis and instruction audit](evaluation/week4-failure-analysis-summary-v1.json),
  [training coverage](evaluation/week4-training-coverage-v1.json),
  [visual review](evaluation/week4-failure-visual-review-v1.json),
  [selected saved-state diagnostics](evaluation/week4-selected-clean-goals-v1.json),
  [preview manifest](evaluation/week4-failure-previews-v1.json).
- Evaluation source: `7e40768d74fa37f856292d7a1b91e5fd1403122f`.
  Analysis source: `93aa2884230a76fd2e23e0cd25d5c46b79d37dad`.
  The analysis implementation passed **253 tests and Ruff** before deployment.
  Output-only diagnostic/plot scripts are retained with artifacts on Feifei;
  saved-state and final audit reports identify their script hashes.
- Checkpoint SHA-256:
  `cd1c72fdd96dde44f76bb118842f4ee61862cfb0678a44729b2398793bb83d74`.
- Feifei evaluation directory:
  `/home/peimengsui/data/eventground-wam/libero-plus/derived/b0/week4-b0-dev210-v1`.
- Feifei full analyses, plots, preview MP4/PNG files and supporting scripts:
  `/home/peimengsui/data/eventground-wam/analysis/week4-failure-analysis-v1`.
- Clean-model/training differences and unknown reference split overlap remain
  documented in the [matching audit](WEEK4_PUBLIC_REFERENCE.md). The gap is an
  evaluation-matched comparison, not a controlled comparison of model designs.

## Executive summary

**The evaluation infrastructure worked; B0's policy quality is the bottleneck.**
Dev-210 finished at **50.5%**, with no execution failures. B0 is strongest at
object pickup and weakest at following changed instructions, handling changed
object/robot configurations, and completing multi-step manipulation. The clean
**28.5-point** gap is heavily concentrated in goal/long tasks, with concrete
examples of wrong destinations and stalled insertion/release/closure.

**Prioritize goal grounding and contact/phase recovery, then test training
coverage and adaptation capacity.** Start with small prompt and phase diagnostics
before spending another multi-day training budget. The prompt suffix issue may
contribute to dev weakness; the clean gap requires a broader policy improvement.
OpenVLA's different pretraining, model and data history remain substantial
confounds. Step 5 is complete; the failed clean gate still blocks Week 4 closure.
