# Proposed Week 10 instrumented-data revision

**Prepared for approval; not an activated protocol or permission to train.**

## Decision and consequence

Use newly instrumented simulator recordings for Week 10 event supervision and
event evaluation, instead of treating approximate reconstructions as physical
truth for the existing Sylvest videos. Preserve Week 10 v1 and all negative and
positive diagnostic evidence. A new v2 contract would explicitly identify the
new dataset, observation phase and limits of its distribution.

This would establish contact/grasp truth for **new recordings**. It would not
recover or certify the historical contacts of `Sylvest/libero_plus_lerobot`.
The first pilot uses canonical scenes; it cannot establish coverage of the
original dataset's visual/physical perturbations or a robustness benefit.

The [frozen v1 experiment](WEEK10_EXPERIMENT.md) requires physical evidence for
the same demonstration trajectories and explicitly says that a change to data
exposure or scientific scope needs a new version. Approval is requested for this
data-design change, rather than for another routine diagnostic or training run.

## First authorized unit if approved

Seal and execute a **three-demonstration instrumented recording pilot** on Feifei.
Use only the existing original demonstration payloads already read for train
episodes 1897, 1426 and 7856 (original demos 9, 8 and 17 respectively). No additional
source download, held-out payload, optimizer update or cloud instance is needed.

Use the verified runtime and the fixed ten-step settling recipe. Record new
dual-camera frames and proprioception together with complete simulator state,
simulation time, action index, object identities/poses, gripper geometry and raw
contact pairs/distances at the actual observation sample time. Save scene XML,
asset hashes, controller/runtime configuration, camera/corruption settings and
the complete source-to-new-frame mapping. The original and new episode IDs remain
distinct. Preserve failures and ambiguous evidence; never retain only successes.

Before running, freeze the exact instrumentation and its consistency checks.
In particular, verify when MuJoCo's derived transforms/contact arrays correspond
to the rendered image and qpos sample. Do not force a simulator update inside the
original sensor loop merely to make a comparison pass. If an explicitly fresh
sampling boundary is necessary, version that recording behavior and use it for
all of the new data.

Proposed pilot ceiling: one recording attempt per demonstration; at most 600
control steps total, 15 minutes wall time, 0.25 active GPU hours, 2 GiB new outputs,
at least 50 GiB free storage, and zero training. Seal the operational retry policy
before execution. Stop and retain any integrity failure. This pilot records raw
physical evidence; it does not extract/admit labels or launch a 200-episode run.

## Gates before expanding or training

1. Pass the recording consistency checks, including frame zero, sensor timestamps,
   camera/robot/object/contact alignment, full episode lengths and immutable
   identities. Package all three outcomes. Preserve the original pilot's failures;
   agreement with old Sylvest observations is no longer the new-data truth criterion.
2. Establish a source-demo identity map and reconcile it with trajectory-v2 splits
   **using metadata before opening more payloads**. All visual variants of one
   physical source stay in one partition. Cross-partition or ambiguous source
   identities are quarantined; no protected source is reassigned to training.
   Do not select only numerically successful old targets or silently drop tasks.
3. Seal the production data/partition manifests, sampling distribution, capture
   budget and contact/grasp predicates before label extraction or QA outcomes.
   Define geometry exclusions, contact persistence and grasp-retention/displacement
   thresholds explicitly. The pilot does not supply validated thresholds.
4. Retain the existing independent QA gate: at least 200 training episodes across
   all 40 tasks, at most 300 for coverage; at least 100 reviewed positives per new
   type, precision and recall at least 95%, and mean boundary error at most two
   frames. Failure or inadequate coverage blocks production. Full videos must
   support missed-event and negative-case review; an automated labeler cannot
   certify itself.
5. Retain the 2,000-prompt balanced SFT design, causal current-observation input,
   frozen M4 parent and SFT/RLVR/control comparison. Define new event-dev/report
   recordings under the existing protection rules only after source identity is
   resolved; test payloads and event-report remain locked at their existing gates.
   Qualify the model/trainer and resources separately before any training launch.

Approval of this proposal means versioning the data design and running the small
recording pilot. It does not approve relaxed identity/QA tolerances, a broad source
search, a different event ontology or an unrestricted production/training run.
