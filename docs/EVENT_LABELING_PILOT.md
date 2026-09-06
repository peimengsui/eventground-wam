# Event-labeling feasibility pilot

This pilot tests which Week 2 robot events can be proposed honestly from the
pinned LeRobot telemetry before investing in visual or simulator-derived
labeling. It does not admit any event into the reliable ontology.

## Gripper convention

The pinned OpenVLA-OFT evaluator documents LIBERO's executed-action convention
as `-1 = open` and `+1 = close`. This is independently visible in the source
demonstrations because the final two state dimensions are the opposed finger
joint positions. The pilot defines aperture as:

```text
abs(state[-2]) + abs(state[-1])
```

A read-only audit of every demonstration found:

- 16,972 transitions into the `+1` close command;
- 99.23% decreased aperture by more than `0.0001` within five frames;
- 16,012 transitions into the `-1` open command;
- 98.05% increased aperture by more than `0.0001` within five frames.

This establishes command polarity. It does not establish contact, grasp, or
release because those claims require object-relative evidence.

## Pilot cohort and detector

`per-task-length-strata-v1` groups episodes by the 40 task IDs, orders each task
by `(episode length, episode index)`, and selects its median-length episode. The
initial cohort therefore covers every task without favoring unusually short or
long trajectories.

The `gripper-close-telemetry/v1` detector proposes a raw `gripper_close`
candidate only when all of the following hold:

1. the command changes from `-1` to `+1`;
2. the close command persists for at least five frames;
3. finger aperture decreases by at least `0.001` within five frames.

The fixed candidate confidence of `0.95` is a conservative, uncalibrated pilot
score. It is not a measured precision estimate. Candidates remain separate from
final `EventInterval` records until human review measures precision and boundary
error.

## Result

The first run processed 40 episodes and 6,548 frames:

- 45 close-command transitions;
- 45 telemetry-confirmed `gripper_close` candidates;
- zero short-command, incomplete-response, or weak-response rejections;
- 40 of 40 measurable open transitions increased aperture;
- 45 of 45 measurable close transitions decreased aperture.

The run produced 40 trajectory contracts, 45 candidate contracts, and 45 review
items. All contract digests and trajectory foreign keys passed validation.

The final run used clean project revision
`e883d8cd80170d6b0d34c2c41f689eb02cb35b9d` and is marked
`publishable: true`. An independent validation pass accepted all 40 trajectory
contracts and 45 candidate contracts, verified every candidate-to-trajectory
foreign key, and matched all 45 review items to their candidate digests.

This completes the feasibility decision: telemetry can support raw
`gripper_close` candidates, while the eight object-relative event types cannot
be admitted from this source without added evidence. The generated review queue
feeds the larger human-QA work in roadmap step 6; that later review does not
change the scope of this pilot result.

## Deferred event types

The following remain deferred: `reach`, `contact`, `grasp_established`, `lift`,
`transport`, `target_region_entry`, `release`, and `success`. RGB,
simulator-sidecar, or human evidence is required for these object-relative
claims. In particular, an open command alone is not a physical release label.

## Run the pilot

On the compute host, from a clean project checkout:

```bash
./scripts/run_demonstrations.sh python scripts/run_event_labeling_pilot.py
```

The default output directory is:

```text
~/data/eventground-wam/libero-plus/derived/event-labeling/week2-pilot-v1/
```

It contains:

- `selection.json` — the deterministic cohort and per-episode counts;
- `trajectory-contracts.jsonl` — streaming references to source episodes;
- `event-candidates.jsonl` — raw, digest-sealed close candidates;
- `review-items.jsonl` — video paths and boundary context for human review;
- `feasibility-report.json` — detector diagnostics and ontology decision.

Existing files are never overwritten. `--allow-dirty` is available only for an
explicitly non-publishable development run.
