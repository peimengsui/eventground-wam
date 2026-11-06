# Week 10 v3 — approved canonical source split

The user approved the exact source-split proposal with digest
`3732521c81d17ee29b7fbc15909c224f8e43c0bc2312fe22384f5a7cf061e963`.
V3 activates that proposal. It preserves v1/v2 as historical records and changes
only source partitioning and the associated scope of generalization claims.

The [active partition contract](evaluation/week10-source-split-v3/partitions.json)
contains 300 train/review sources, 103 event-dev sources and 77 event-report
sources. Each identity is the pinned original repository, revision, HDF5 path
and demonstration ID. New variants inherit their original source's partition.
Unknown sources are denied. All three partitions cover all 40 tasks, with no
source-ID overlap. The exact training queue has five initial recordings per
task (200 total), followed by 100 coverage reserves in the approved order.

New event-dev/report memberships exclude all 475 previously inspected or
training sources listed in the approved proposal. Their payloads remain locked.
They are new memberships, not renamed historical validation groups. Old split
files and all completed Week 7–9 artifacts remain unchanged.

## Scope of the claim

Source disjointness applies to the newly generated canonical event-supervision
partitions. Historical cropped, filtered or resampled aliases remain unresolved.
V3 does not claim historical source closure or that the inherited M4 action
learner never encountered these sources. Comparisons are conditional on the
same fixed M4 parent. Later robotics evaluation after adaptation needs its own
justified design; old panels do not automatically become clean source-held-out
validation for the adapted model.

The M4 epoch-2 endpoint, event ontology, physical thresholds, independent QA
gates, current-observation-only inputs, prompt budgets/caps, numerical recipe
requirements, losses/rewards and promotion rules are unchanged. Bootstrap groups
for the new event dataset are original physical source IDs, so variants cannot
be treated as independent samples.

## Preparation and launch boundaries

Approval activates the source split and authorizes finishing the production
manifest. Training-side source staging reads only the selected 300 sources'
initial full-state row and model XML; previously verified full action arrays
are reused. No original videos, new event-dev/report payloads, simulation or
optimization are needed for that staging step. Each payload and runtime input
is hashed before capture admission.

The train-side capture plan is bounded to 200 initial recordings and at most
100 coverage reserves, 54,033 control steps including ten settling steps per
source, two hours of active recording/verification work and 100 GiB of outputs,
with 150 GiB free before launch. Each source gets one attempt; errors stop the
queue and preserve partial evidence. Reserves may be used only to address
predeclared coverage shortfalls, not to replace failed outcomes or tune rules.
No source is selected by robot success. Dev/report capture needs a separate
bounded child plan.

The synchronized capture remains the qualified fresh pre-action snapshot from
v2: two 256×256 cameras at 20 Hz, robot state, object/gripper poses, exact geometry
ownership, contacts and complete integration state. The pilot's rendering
qualification remains explicit. Production captures must preserve live simulator
state and pass independent physical reconstruction; they have no preexisting
full replay baseline, and must not claim such equality.

V3 approval alone does **not** launch production recording or training. Capture
requires the sealed payload/runtime/recorder manifest and explicit launch
approval; training additionally requires independent full-episode QA, prompt
manifests, implementation/recipe and numerical/resource qualification. Frozen
contract readiness fields describe their creation time; later child admissions
record completed gates without rewriting the study.
