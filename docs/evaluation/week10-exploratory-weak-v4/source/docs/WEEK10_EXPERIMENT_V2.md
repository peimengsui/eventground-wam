# Week 10 v2: instrumented event supervision

The user approved this data revision and its three-demonstration recording pilot
on October 17, 2026. The v1 experiment and all preceding diagnostics remain
immutable historical evidence. This revision changes the event-supervision data
and recording phase; the frozen M4 parent, causal model inputs, SFT/RLVR/control
comparison, prompt counts, rewards, evaluation and quality gates are inherited.

## Recording identity and phase

New recordings use the pinned original LIBERO demonstration actions and initial
full state, canonical task scenes, ten settling steps, Panda OSC_POSE at 20 Hz,
seed zero, and no image corruption. The pilot retains the three already selected
training cases (1897/demo_9, 1426/demo_8, 7856/demo_17), including unsuccessful
replays. These new recordings have distinct IDs. They do not certify physical
truth for the old Sylvest videos or reproduce their perturbation distribution.

At each boundary before a source action, copy the live simulator's complete
integration state into an independent compiled-model snapshot. Forward only the
snapshot, then record both cameras, robot/object poses and raw contact geometry
from it. The images are 256x256 lossless RGB, rotated 180 degrees. Physical state
and derived geometry share this fresh boundary, including frame zero. This is an
explicit change from the old cached sensor phase, which was observed 2 ms before
the end of a control step in the earlier diagnostic.

Forward dynamics computes derived quantities before integration; fresh boundary
geometry therefore requires an explicit update. See the pinned-version
[MuJoCo simulation documentation](https://mujoco.readthedocs.io/en/2.3.7/programming/simulation.html).
The live replay is never forwarded by the recorder. Every live state must equal
the saved ten-settle baseline, capture must preserve its integration state, and
an independent position-stage computation must reproduce snapshot geometry and
contacts within 1e-10. Rendering must preserve snapshot physics and geometry.

Save source/action/frame mapping, full integration state, compiled model, XML,
asset hashes, runtime/controller settings, camera transforms, object/geom maps
and contact pairs/distances. Store task/instruction provenance in the source
catalog. No event label is extracted by the recording pilot.

## Source protection

The source identity is repository + pinned revision + HDF5 path + original demo
ID. All new visual/scene variants of that source stay together. The catalog
reconciles previously accessed training arrays against the trajectory-v2 group
metadata and exact complete float32 action traces. This binds known full-trace
aliases for split protection; it does not establish historical scene identity.
Approximate matches, unknown original identities, trimmed/time-warped aliases
and cross-partition conflicts remain unresolved or quarantined. No held-out
payload is read to resolve a mapping, and no protected source is reassigned.

The historical 103 event-dev, 77 event-report and 160 test group assignments
remain protected. A production manifest needs verified source-level partition
closure before capture. The limited pilot catalog is not a complete mapping of
all 1,681 historical groups. Unknown groups are not automatically training data.

## Label rules and remaining gates

The sealed labeler child contract fixes geometry exclusions, two-frame contact
persistence, and grasp confirmation through bilateral contact, relative-pose
retention and gripper motion. It specifies ambiguity, censoring, event ordering
and negative-window rules. Thresholds are engineering choices, not validated
labels. Failure of independent QA requires a new version; it cannot be repaired
by silently relaxing these rules. The earlier gripper-close QA does not transfer
automatically to new images.

Independent full-episode QA must cover at least 200 training episodes and all
40 tasks, at most 300 for coverage, with at least 100 reviewed positives per new
type, precision/recall at least 95%, and mean start/end error at most two frames.
Missed events and negative cases must be reviewed. The three-demo pilot cannot
satisfy this gate. Production sampling, manifests and capture budget remain
unsealed until source identity coverage is resolved; no 200-episode collection,
label admission or training is authorized by the present pilot.

The pilot permits one attempt per demonstration, at most 600 control steps,
900 seconds wall time (a conservative 0.25 GPU-hour upper bound), 2 GiB outputs,
and requires 50 GiB free storage. Any runtime/integrity failure is preserved and
stops further cases; no automatic trajectory retry. No additional downloads,
held-out payloads, optimizer updates or cloud instances are involved.
