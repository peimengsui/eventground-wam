# Deterministic event extraction

Roadmap step 4 converts raw detector proposals into reproducible interval and
decision artifacts. Version 1 supports telemetry-backed `gripper_close` only;
object-relative events remain deferred unless a later detector supplies RGB,
simulator-sidecar, or human evidence.

## Version 1 filter

`gripper-close-deterministic/v1` applies the following locked rules:

- two-frame command hysteresis removes one-frame `-1/+1` chatter;
- a centered three-frame median smooths the two-finger aperture;
- a close command must persist for at least five frames;
- the smoothed aperture must fall by at least `0.001` within five frames;
- a raw proposal must begin within one frame of the debounced transition and
  have confidence of at least `0.9`.

Drops from `0.0005` up to `0.001`, incomplete response windows, short command
runs, missing proposals, and unmatched proposals are recorded as ambiguous.
Episodes with neither a persistent close run nor a proposal are explicitly
absent. Object-relative proposals are explicitly deferred.

Every accepted telemetry proposal becomes an `EventInterval` with raw-candidate
digest lineage and merged evidence, but remains `ontology_tier: weak` and
`review_status: unreviewed`. Only the human-QA gate can promote an event to a
reliable training target.

## Generate intervals

Run from the project root:

```bash
./scripts/run_demonstrations.sh python scripts/generate_event_intervals.py \
  --trajectory-contracts /path/to/trajectory-contracts.jsonl \
  --event-candidates /path/to/event-candidates.jsonl \
  --output-dir /path/to/week2-event-extraction
```

The command streams telemetry for the selected trajectories and atomically
publishes:

- `event-candidates.jsonl`, preserving every raw proposal;
- `event-intervals.jsonl`, containing filtered weak intervals;
- `event-decisions.jsonl`, containing `admitted_weak`, `absent`, `ambiguous`,
  and `deferred` dispositions; and
- `manifest.json`, locking configuration, counts, SHA-256 values, provenance,
  and publishability.

Existing output is never overwritten. A dirty tree is rejected unless
`--allow-dirty` is supplied for an explicitly non-publishable development run.

## Integrated pilot result

The 40-episode feasibility cohort produced 45 raw candidates, 45 weak/unreviewed
intervals, 45 `admitted_weak` decisions, and two explicit episode-level absence
decisions. There were no ambiguous or deferred proposals in this cohort. The
candidate output SHA-256 matched the pilot input byte-for-byte, and an
independent pass validated the interval contracts, lineage, and manifest digest.
