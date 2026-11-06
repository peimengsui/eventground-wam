# Week 10 physical event labeler

The frozen contact, grasp and gripper-close rules are implemented in
[`instrumented_labeling.py`](../src/eventground_wam/instrumented_labeling.py).
The implementation is sealed before its first application to the three completed
instrumented recordings. Outputs remain unreviewed; neither these examples nor
the synthetic tests establish population label quality.

## Evidence and behavior

The input is synchronized numeric evidence from the new canonical recordings:
actions, robot state, object and gripper poses, exact geometry ownership and
complete raw contacts. Images and task-success flags do not influence extraction.
Every movable free object is considered, including an object unrelated to task
success. Fixtures, table and self contacts are excluded.

- **Contact:** a collision-enabled gripper/object pair with nonpositive distance
  and an active constraint on at least two consecutive frames. No gap filling.
- **Grasp:** the frozen five-frame close/contact, bilateral-contact, relative-pose
  and gripper-motion conditions. After confirmation, stationary holds continue
  until contact is lost or the frozen exit pose limit is exceeded.
- **Gripper close:** the existing raw command-transition/aperture-response rule:
  five close-command frames and at least 1 mm aperture reduction at frame t+5.
  Optional historical smoothing and old human QA are not imported.
- **Ambiguity:** singleton contacts, failed or incomplete grasp confirmation,
  initial close commands and failed close transitions are retained with reasons.
  Earlier failed confirmation windows remain ambiguous even if a later window
  confirms. Malformed geometry, identities, timestamps or numeric evidence reject
  the recording; they cannot generate negative examples.

Intervals are half-open at 20 Hz. Initial and final intervals carry censoring
flags. The diagnostic prompt preview requires a complete 32-frame future window,
excludes ongoing-event and ambiguous windows, selects the earliest event with the
frozen tie order, and emits `none` only for a fully observed clear window.
It exports an anchor and target, never future physical tensors as model inputs.
The four-anchor/event and sixteen-prompt/source caps apply at later manifest
sampling; previews deliberately enumerate all anchors and are not training data.

## Verification and pilot outputs

The [implementation contract](evaluation/week10-labeler-implementation-v1/implementation-contract.json)
pins the implementation and its interpretation before extraction. **28 new
synthetic tests and 83 total targeted tests pass.** Coverage includes missing or
corrupt evidence, contact gaps, single-finger and sliding cases, stationary holds,
rotation-only grasps, coordinate-frame invariance, censoring and horizon limits.

| Completed recording | Frames | Contact | Grasp | Gripper close |
| --- | ---: | ---: | ---: | ---: |
| instrumented-00-demo_9 | 126 | 2 | 3 | 1 |
| instrumented-02-demo_8 | 301 | 3 | 3 | 2 |
| instrumented-04-demo_17 | 88 | 3 | 1 | 1 |
| Total | 515 | 8 | 7 | 4 |

The failed-task recording is retained. The 515 anchor previews include 63 `none`,
66 gripper-close, 4 grasp and 1 contact preview; 96 incomplete tails, 178 ongoing
anchors and 107 ambiguity-overlapping anchors are excluded. These are unsampled
preview counts, not 134 unique eligible training prompts. The conservative rules
and earliest-onset selection reduce contact/grasp preview availability; coverage
must be measured after production QA, without silently relaxing rules.

A [separate numeric audit](evaluation/week10-labeler-implementation-v1/numeric-verification.json)
reconstructs every contact and close interval, checks all accepted grasp
predicates and end boundaries, and validates negative windows directly against
the saved evidence. This is implementation verification, **not independent human
QA or a precision/recall result**. No new simulation or training was run.

## Reproduce extraction

Use a fresh output path; existing outputs are never overwritten:

```bash
PYTHONPATH=src .venv/bin/python scripts/extract_week10_event_candidates.py \
  --pilot-base docs/evaluation/week10-instrumented-pilot-v2 \
  --implementation-contract docs/evaluation/week10-labeler-implementation-v1/implementation-contract.json \
  --output /tmp/week10-labeler-reproduction
```

The loader checks sealed pilot completion, source identities and input checksums,
frame order, timestamps and sampling period. The underlying rules and prior
recording packages remain unchanged. The sealed pilot's historical status
`label_extraction_performed: false` describes that earlier recording operation;
this separate child package records the subsequent extraction.

## Remaining admission gates

Production source partition closure and the recording manifest precede capture.
Then independent full-episode review must cover at least 200 training episodes,
all 40 tasks, and at least 100 positives for each new event type; the frozen
precision/recall and boundary-error gates still apply. Up to 100 additional
recordings are reserved for coverage only. Gripper-close quality must also be
reviewed on these new recordings. Training manifests, class/source caps and
numerical training recipes remain separate work.
