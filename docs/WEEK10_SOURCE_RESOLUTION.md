# Week 10 source-resolution results — October 17, 2026

**Partial resolution: ten settling steps reproduce one complete training trace,
and observation timestamps can be captured correctly. Historical physical-source
qualification remains incomplete; no contact/grasp labels are admitted.**

## What was run

The original 40-candidate pilot and its zero-qualified-source result are unchanged.
A separate [sealed diagnostic](evaluation/week10-event-supervision-v1/regeneration-diagnostic-v1/contract.json)
tested the first three previously selected action-matching training examples.
Each used canonical task geometry, the saved original initial full state, Panda,
OSC_POSE, seed 0, 20 Hz, MuJoCo 2.3.7 and robosuite 1.4.0. The two fixed settings
used zero or ten dummy settling steps, then captured observations before each
retained original action, following the inspected OpenVLA regeneration reference.
These action sequences contain no removable no-ops. All outcomes were retained.

This is a reproduction hypothesis, not proof that the reference generator was
used for the pinned Sylvest release. Exact target perturbations were not known.
No numerical tolerances, original results, split assignments or model weights changed.

| Target episode | Frames | Zero settling: max finger error | Ten settling: max finger error | Ten settling: complete numerical screen |
| --- | ---: | ---: | ---: | --- |
| 1897 | 126 | 9.540 mm | 0.427 mm | Fail; first mismatch at zero-based frame 99 |
| 1426 | 301 | 15.502 mm | 1.098 mm | Fail; first mismatch at zero-based frame 47 |
| 7856 | 88 | 9.533 mm | 0.00000184 mm | Pass on all 88 frames |

The screen uses the existing absolute tolerances: 1e-6 for actions and 1e-5
for each state component, zero relative tolerance and exact gripper commands.
The positional state tolerance is 0.01 mm; orientation components are in radians.
The table's finger error is only one component of the full eight-dimensional test.
Episode 7856's maximum state-component error is 1.19e-7. All three ten-step runs
match frame zero within tolerance, but two diverge later. Their cause has not
been established; do not call them solved by an offset or loosen the tolerance.

All six saved numeric traces, frame indices, hashes and metric calculations were
[independently recomputed](evaluation/week10-event-supervision-v1/regeneration-diagnostic-v1/verification.json).
The runs used 1,060 control steps and 22.22 seconds of cumulative runner time.
An import-time failure used zero steps; its log is preserved alongside the
separately sealed recovery using the already installed ImageMagick library paths.
Both the original 1,100-step and 600-second cumulative ceilings were respected.

## The observation-phase issue is understood for the reproduced example

A separate [one-episode check](evaluation/week10-event-supervision-v1/observation-phase-diagnostic-v1/contract.json)
replayed episode 7856 with both cameras and read-only sensor timestamp hooks.
It preserved the exact numeric trajectory from the camera-disabled run.

- Front camera, wrist camera, end-effector pose and gripper observations sampled
  at the same simulator time on every one of the 88 retained frames.
- Their sample time was **2 ms before** the returned end-of-control-step state.
- Finger positions captured at the sensor call match the observations exactly.
  Reading only the returned full state had produced a maximum 0.198 mm difference.

Thus an instrumented recording must bind physical evidence to sensor timestamps;
simply saving the full state after `env.step` is insufficient for this runtime.
This finding does not certify object poses or contact arrays at that timestamp:
MuJoCo's derived geometry/contact update phase also needs an explicit check.

The same check decoded only this training episode's two videos. It compared
preselected frames 0, 44 and 87 in both cameras under the two predeclared image
orientations (0 and 180 degrees). The 180-degree orientation agrees more closely:
wrist RGB mean absolute error is 2.46–2.93 on a 0–255 scale; front error is
12.52–14.48. The inspected target front image is visibly blurred. Lossy AV1,
unknown sensor corruption and absent conversion provenance prevent interpreting
these diagnostic values as a source-admission pass. Three images are not a
whole-episode physical identity test or independent contact/grasp QA.

![Preselected target/replay camera pairs](evaluation/week10-event-supervision-v1/observation-phase-diagnostic-v1/camera-comparison.png)

The [phase verification](evaluation/week10-event-supervision-v1/observation-phase-diagnostic-v1/verification.json)
recomputes all timestamp, finger-state and image comparisons. This used 98 control
steps and 6.54 cumulative runner seconds. A zero-step sensor-wrapper metadata
failure and its separately sealed correction are preserved; the correction reused
the six decoded images. The original 100-step, 300-second ceilings were respected.
Rendering used the existing Feifei GPU; these wall times are not measured GPU-hours.

## Remaining decision

There is now a working way to reproduce a trajectory and record synchronized
evidence. There is still no qualified physical source covering the required
200-episode QA set, no complete scene/frame mapping for the pinned release,
and no admitted contact/grasp label set. A one-example numerical pass cannot
justify transferring labels to the original dataset or ignoring failed examples.

The practical route that avoids waiting for missing historical files is the
[instrumented-data revision proposal](WEEK10_INSTRUMENTED_DATA_PROPOSAL.md):
capture new RGB, robot, object and contact evidence together and label those new
recordings. This changes the event dataset and requires an explicit scientific
revision; it is not a transparent repair of the old recordings. The proposal is
prepared for review and has not been activated.

Validation: 26 existing source/supervision/freeze tests passed; both diagnostic
scripts pass lint. Direct metric checks reject first/last-frame disagreements,
shortened sequences and changed gripper commands. The unchanged Week 10 freeze
still verifies all 25 pinned sources. No held-out payloads, model training or
contact/grasp label admission occurred.
