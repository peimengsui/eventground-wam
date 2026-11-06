# Fresh B0: evaluation audit and failure review

**Completed:** September 24, 2026. **Scope:** the requested audit and failure
review, using the fixed epoch-2 Qwen3.5-2B / all-L1 checkpoint. Week 4 remains open.

## Executive summary

The new baseline is substantially stronger, but its remaining clean deficit is
concentrated in particular manipulation stages. **Drawer closing is the clearest
target:** all 37 failed B0 drawer trials reached a recorded state with the bowl
inside, but none reached a recorded closed-drawer state. The failures usually
persisted for approximately 19 simulated seconds after placement. The two-mug
and microwave examples also show acquisition, insertion and recovery problems.
These findings do not establish that missing memory is the main bottleneck.

There is also direct retrospective evidence of **instruction sensitivity**:
11 of the 13 B0 language failures succeed in an existing canonical-instruction
recording with identical starting images/state and identical parsed scene/goals.
The other two are the already-weak drawer task. This makes language robustness a
separate target, even when a paraphrase manifests as a grasp/control failure.

The evaluation-input discrepancy is reproducible **rendering nondeterminism**:
the same runtime, scene and physics can produce either original first-query
image. The analyzer now reports a permanent integrity stop instead of restarting
the pipeline. The exact-input rule and all original outcomes are preserved;
the investigation does not convert an invalid pair into a valid one or establish
statistical noninferiority.

| Fixed endpoint | B0 | Installed OpenVLA-OFT+ |
| --- | ---: | ---: |
| Clean-2000 | 1,867/2,000 (93.35%) | 1,935/2,000 (96.75%) |
| dev-210 | 162/210 (77.14%) | 170/210 (80.95%) |
| Clean Long | 398/500 (79.6%) | 466/500 (93.2%) |
| dev language | 17/30 (56.7%) | 27/30 (90.0%) |

The comparison is evaluation-matched, with model/training differences retained;
it is not a controlled architecture comparison. The supervised test split was
not used. No new training or policy rollout was launched in this investigation.

## 1. Evaluation audit and restart fix

### Original evidence

- Revalidated all **420 dev HDF5 recordings** with the original artifact hash,
  recorder, outcome, checkpoint, source and task checks.
- **209/210 pairs** match the frozen scene/initial/first-query input requirement.
- Ordinal **203**, Long / Light Conditions, has **four front-camera channel
  values differing by one uint8 level** at policy query step 10. Both policies
  succeeded. Initial images, first-query wrist input, proprioception and physics
  match. The full settling-prefix audit additionally found seven wrist-channel
  differences by one level at step 7, before any policy query; this is separate
  from the four first-query differences.
- All original clean pairs remain valid. Original report digests and outcomes
  are unchanged; no failed trial was replaced or excluded.

### Reproduction

Repeated only the frozen reset and ten settling actions, without loading either
policy: three fresh environments in the B0 runtime and three in the reference
runtime. Across all six repetitions, recorded physics and proprioception match
the original prefix at every one of its 11 states.

In the B0 runtime, one repetition produced **exactly the original B0 image** and
two produced **exactly the original reference image**, at the first policy query.
All three reference-runtime repetitions produced the original B0 image. Thus the
two image variants do not require different models or different Python runtimes.

Thirty direct renders at the same final state produced **3–4 distinct image
hashes per repetition** in both runtimes. Differences from the first render were
1–9 channel values, each by one level, with unchanged physics and camera/light
parameters. These direct renders are supplementary: cached observations can
differ from fresh renders because of sensor timing. The stronger reproduction
is the exact match to both original cached first-query images above.

**Conclusion:** nondeterminism in the rendering path is demonstrated. Its
specific GPU/driver numerical mechanism remains unisolated. This explains the
discrepancy without treating approximate equality as exact equality.

### Maintenance and acceptance

The final analyzer's uncaught `ValueError` returned exit 1, which systemd treated
as retryable. It now catches integrity/missing-file errors, writes a separate
sealed `analysis-status.json`, and returns **78**, the service's existing
`RestartPreventExitStatus`. An analyzer-only host service exercised the actual
ordinal-203 failure and ended with **exit 78, zero restarts**. The completed
production service is stopped and disabled for boot startup. Its original
analyzer is archived; only final-analysis error handling changed in its checkout.

Verification: **14 focused tests passed**, including shell exit propagation and
preservation of input evidence; Ruff passed. Unexpected runtime failures remain
distinct from integrity failures.

- Historical clean five-point gate: **INCONCLUSIVE**, paired task-cluster
  bootstrap gap interval **[−7.00, −0.65] percentage points**.
- Formal v2 readiness: **NOT ASSESSED — INPUT MISMATCH**. No `completion.json`
  was fabricated.
- The descriptive dev interval remains **[−8.10, +0.48] points**. Even accepting
  the tiny rendering discrepancy under a separately documented decision would
  leave the unchanged v2 performance rule **INCONCLUSIVE**.

## 2. Where the Long tasks fail

The first three clean suites have zero net success-count deficit versus the
reference. These three Long tasks contribute **55/68 = 80.9%** of the overall net
clean deficit; the drawer task alone contributes **33/68 = 48.5%**.

| Task | B0 /50 | Reference /50 | Findings from every recorded state in the B0 failures |
| --- | ---: | ---: | --- |
| Bowl into bottom drawer, close | 13 | 46 | **37/37** failures reached bowl-inside; **0/37** reached closed-drawer. **34/37** still had the bowl inside at the final recorded state. |
| Two mugs onto respective plates | 36 | 49 | All **14** failures reached exactly one placement predicate at some point; none reached both. Six reached only white-mug placement; eight only yellow-white-mug placement. At the final recorded state, 13 retained one placement and one had lost it. |
| Mug into microwave, close | 36 | 45 | Of **14** failures, **10** reached mug-inside at some point. Four retained insertion but not closure; six lost insertion; four never reached insertion. No failure had both goal predicates true simultaneously. |

### Drawer: placement succeeds; closing and recovery fail

The bowl-inside predicate first becomes true with **18.6–19.6 simulated seconds
remaining**, median **19.2 seconds**, in the failed trials. This is strong evidence
against merely giving the policy a few more timeout steps as the first remedy.

**25/37 failures** also satisfy the existing descriptive late-stall flag:
less than 1 cm end-effector span in the last 40 steps despite mean translation
command norm at least 0.1. This suggests blocked or ineffective contact control;
the flag alone does not establish a contact-force or action-decoding cause.

Paired examples 1550 and 1551 show B0 placing the bowl and then working around
the drawer front/handle without completing closure. OpenVLA completes closing.
The both-success example 1555 shows that B0 can execute the full sequence.

### Two mugs: the reviewed failures attempt the remaining object

Examples 1856 and 1859 place the white mug and then approach the yellow-white
mug. One repeatedly manipulates around that mug; the other moves toward its
destination with an empty gripper. The reference completes both placements.
This supports acquisition/control and failed-grasp recovery as concrete targets.
It does **not** support simply saying B0 forgot that a second mug existed.

The exhaustive predicate counts include both first-placement identities; the
two illustrated examples are not estimates of visual failure-mode prevalence.

### Microwave: losing placement and proceeding to the door

Examples 1600 and 1607 transport the correct mug toward the microwave but leave
it outside/on its side near the opening. B0 subsequently manipulates the door
without recovering insertion. The reference retains the mug placement and closes.
The both-success example 1601 shows the intended sequence under B0 as well.

These examples motivate checking release/insertion alignment and whether control
reacts appropriately to a lost object, in addition to closure itself.

## 3. Language failures are heterogeneous

All **13 B0 language failures** were reviewed alongside the corresponding
reference recording. The reference succeeds in 12; ordinal 131 fails under both
policies. Two other language cases succeed under B0 but fail under the reference,
giving the net ten-success deficit.

| Behavior observed | Examples | Interpretation |
| --- | --- | --- |
| Wrong-object interaction | 23 repeatedly engages the cookie box when instructed to move the adjacent bowl. | A concrete object-grounding failure; dense grasp checks identify the cookie box. |
| Acquisition failure with a distractor contact | 75 remains among the containers; recorded contacts include tomato sauce, not the salad-dressing target. | Grounding and grasp control are both candidates; contact identity alone does not prove intended target selection. |
| Correct target approached but transfer fails | 22, 25, 26; 132 opens the top drawer but does not insert the bowl. | Language robustness can degrade control without an obvious wrong-object choice. |
| Familiar Long manipulation failure | 182 puts the bowl inside but does not close; 183 holds it near the opening without insertion. | Some language failures resemble existing clean-task control weaknesses. |
| Destination/insertion problem | 184 grasps the book but manipulates it around the caddy's front/central area without rear containment. | Destination interpretation versus precise insertion remains unisolated. |
| Activation/acquisition never completes | 181 never activates the stove; 186 approaches both moka pots without placing either. | These observations do not by themselves distinguish semantic understanding from control error. |

Ordinal 131 is a useful caution: B0 remains around the cabinet, while the
reference transports the bowl toward the stove and also fails. Public-reference
success is not assumed for every language example.

### Retrospective canonical-versus-paraphrase comparison

Mapped all 30 language cases to existing clean trial-0 recordings of their
canonical task, separately for each policy. Across **all 60 comparisons**:

- Initial and first-query front/wrist images, proprioception and physics match
  exactly; initial-state metadata also matches.
- The parsed BDDL scene differs **only in `language_instruction`**, including
  identical goals, objects, regions and initial-condition definitions.
- The checkpoint is identical within policy, as are simulation settings,
  observation settings, action execution length and per-suite step limits.

| Policy | Cached canonical outcomes, weighted by these 30 cases | Language outcomes | Canonical success → language failure | Canonical failure → language success |
| --- | ---: | ---: | ---: | ---: |
| B0 | **27/30** | **17/30** | **11** | **1** |
| OpenVLA | **28/30** | **27/30** | **2** | **1** |

For B0, only language failures **182 and 183** also fail canonically; both reuse
the bowl/drawer state. The microwave case 187 succeeds with its paraphrase despite
failure in the cached canonical recording. Instruction effects can therefore
change grasp/control trajectories in either direction; wrong-object choice is
not the only visible manifestation.

This is stronger evidence of prompt sensitivity than comparing unrelated clean
and perturbed averages. It remains a retrospective development diagnostic:
the 30 cases reuse **25 unique canonical task/state pairs**, episode seeds differ,
and the renderer is not bitwise deterministic. It does not establish a 30-case
independent causal estimate, identify the architectural cause, or justify editing
the benchmark prompts. [Complete cached comparison](evaluation/week4-fresh-v2-cached-prompt-audit.json).

## 4. Implications for the next experiment

1. **Prioritize contact control and recovery around phase transitions.** The
   initially proposed predict-eight/execute-four diagnostic was subsequently
   deferred when the user chose to preserve the OpenVLA-matched eight-step
   schedule. The [completed training-stage audit and continuation
   design](WEEK4_STAGE_SAMPLING_AUDIT.md) now define the next controlled screen.
   These diagnostics do not establish that either change will improve success.
2. **If training is needed, inspect the training data at the specific failing
   stages:** drawer closing after release, stable mug insertion/release, and
   acquisition of the remaining object. Compare any targeted sampling change
   against equal extra updates with the current sampler.
3. **Treat language robustness as a separate controlled improvement.** The cached
   matched-input comparison supports it as a real weakness. A future train-only,
   meaning-preserving paraphrase augmentation ablation is reasonable, with an
   equal-update control and no copying of dev prompts into training. The present
   evidence does not isolate semantic grounding from instruction-conditioned
   control; do not classify every language timeout as a wrong-object error.
4. **Defer a memory module until there is direct evidence of temporal ambiguity
   or loss of completed-subtask information.** The reviewed two-mug cases already
   attempt the next object, and the drawer cases spend ample time at the closing
   stage. Longer blanket training is not yet a demonstrated remedy either.

These are recommendations only; no execute-four experiment, prompt replacement,
additional training or acceptance-rule change was made.

## Evidence and limits

- Offline traces: **1,060 recordings** — all 500 clean Long trials for both
  policies and all 30 language trials for both policies — every tenth policy
  state plus the last recorded state.
- Dense verification: **38,380 recorded policy states across 78 B0 failures** —
  all 65 failures on the three targeted clean tasks and all 13 language failures.
- Visual review: **44 contact sheets**, eight sampled front/wrist frames each;
  all 44 corresponding videos are provided. The assistant inspected the sheets,
  not every video frame. The clean examples are illustrative; all B0 language
  failures are represented. **All 88 PNG/MP4 hashes** were checked after transfer.
- State round-trip error was at most **2.22e−16**. Freshly recomputed end-effector
  sensors differed from cached recorded observations by up to **2.03 mm**; no
  exact sensor equivalence is claimed. Simulator predicates are offline
  diagnostic evidence and are never supplied to the evaluated policy.
- Recordings store pre-action states and omit the final post-action state. A
  successful reference's last stored closing predicate may therefore be false.
  Recorded outcomes remain authoritative. Contact flags are not proof of a stable
  intended grasp, and motion flags are not causal labels.

[Machine-readable summary](evaluation/week4-fresh-v2-closeout-summary.json) ·
[Per-pair visual annotations](evaluation/week4-fresh-v2-visual-review.json) ·
[Cached prompt comparison](evaluation/week4-fresh-v2-cached-prompt-audit.json) ·
[Video/contact-sheet index](../outputs/week4-fresh-v2-closeout-20260924/README.md) ·
[Original final results](../outputs/week4-fresh-v2-evaluation-20260924/README.md)

Full source snapshots, rendering reproductions, input audit and state traces are
under `outputs/week4-fresh-v2-closeout-20260924/` locally and
`/home/peimengsui/data/eventground-wam/analysis/week4-fresh-v2-closeout-20260924/`
on Feifei. The original frozen source commit remains
`d38968007067cecfcd0dbcd2b1a297a18ad7283d`; the deployed maintenance patch has an
independent source hash recorded in the summary.
