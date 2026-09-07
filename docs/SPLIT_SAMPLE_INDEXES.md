# Leakage-resistant split and sample indexes

Week 2 uses episode-level grouping because the pinned LIBERO-Plus snapshot does
not publish reset seeds or MuJoCo state. The locked surrogate strategy is
`task-state-rgb-surrogate-v1`. Its SHA-256 identity covers:

- the source task index;
- all eight initial proprioception values, each deterministically converted to
  its source-native IEEE-754 float32 representation and encoded as big-endian
  hexadecimal;
- a SHA-256 fingerprint of the initial front frame after conversion to packed
  decoded RGB24; and
- the corresponding initial wrist-frame fingerprint.

The source episode index is deliberately excluded. Episodes with identical
surrogate evidence therefore share a group. This exact-match strategy avoids
claiming equivalence that the source metadata cannot prove; near-identical
states or images remain separate groups.

## Split assignment

The default train/validation/test ratios are explicitly `0.8/0.1/0.1`. A SHA-256
bucket over the group digest and versioned namespace `week2-splits-v1` assigns
the entire group. Assignment does not depend on input order or global dataset
size. The manifest reports unique group and episode counts per split and records
`cross_split_group_overlap_count: 0` only after checking all three pairwise
intersections.

## Sample policy

For fixed horizons `2`, `4`, `8`, `16`, and `32`, every anchor whose target is
inside the same episode is emitted. For `next-event`, only intervals with
`ontology_tier == reliable` and `review_status` equal to `accepted` or
`corrected` are eligible. Weak and unreviewed labels are never silently promoted
to targets. Each anchor points to every eligible event at the earliest strictly
later event start. Simultaneous event labels are therefore retained rather than
silently selecting one. No sample may cross an episode boundary. The manifest
counts eligible targets, excluded intervals, excluded weak intervals, and
excluded unreviewed intervals; the two exclusion-reason counts may overlap.

Every split assignment is checked against its trajectory contract. Every sample
is checked against both its trajectory and split assignment; next-event samples
also validate their event digest and boundary. Outputs are canonical JSONL plus
a digest-sealed manifest containing record hashes and counts. Publication
refuses existing files and removes partial output if generation fails.

## Generate a pilot subset

The trajectory input defines the selected episode set. This command uses the
existing 40-episode feasibility-pilot trajectories while permitting the event
file to be a superset:

```bash
./scripts/run_demonstrations.sh python scripts/generate_split_sample_indexes.py \
  --snapshot /path/to/libero-plus-lerobot/current \
  --trajectory-index outputs/week2-event-labeling-pilot-clean/trajectory-contracts.jsonl \
  --event-index outputs/week2-event-extraction/event-intervals.jsonl \
  --output-dir outputs/week2-split-sample-pilot \
  --fingerprint-workers 4 \
  --allow-partial-source \
  --allow-dirty
```

Only the first telemetry row and first decoded frame from each camera are read
for grouping. Split assignments and samples then stream directly to their
temporary JSONL files. `--fingerprint-workers` bounds concurrent initial-frame
reads while results remain in canonical episode order. Omit
`--allow-partial-source` for the complete 14,347 episode source; full-media
fingerprinting and artifact freezing are part of the scale/freeze pass.

## Real-data smoke result

A one-episode real-media integration run decoded both initial camera frames,
created one isolated train group, and emitted 643 in-episode fixed-horizon
samples: 139, 137, 133, 125, and 109 samples for horizons 2, 4, 8, 16, and 32,
respectively. Its sole event interval was weak and unreviewed, so it was counted
in both exclusion-reason fields and produced zero `next-event` samples. An
independent pass validated the trajectory, split, sample, manifest, and
foreign-key digests. The complete 14,347-episode generation remains roadmap
step 7 rather than being represented by this bounded I/O smoke test.
