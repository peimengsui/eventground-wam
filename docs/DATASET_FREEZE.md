# Week 2 dataset freeze

The freeze command turns the complete pinned demonstration snapshot into one
digest-sealed dataset version. It runs only from a clean committed project
revision and refuses to overwrite an existing version.

Run it next to the source snapshot rather than through a network filesystem:

```bash
./scripts/run_demonstrations.sh python scripts/run_dataset_freeze.py \
  --dataset-version week2-dataset-v1 \
  --fingerprint-workers 8
```

The output defaults to
`~/data/eventground-wam/libero-plus/derived/frozen/week2-dataset-v1/` and keeps
the large artifacts outside Git:

```text
week2-dataset-v1/
├── primary/
│   ├── candidates/
│   ├── events/
│   └── splits/
├── replica/
│   ├── candidates/
│   ├── events/
│   └── splits/
├── logs/
├── freeze-progress.json
└── dataset-freeze-manifest.json
```

The primary and replica independently process all 14,347 episodes. The final
validator checks:

- all 2,238,036 frames and all 40 tasks are represented;
- every trajectory, candidate, interval, and split contract and digest;
- raw-candidate byte preservation and event-decision lineage;
- every multi-horizon sample contract, episode bound, split assignment, and
  optional reliable-event foreign key;
- pairwise group isolation across train, validation, and test;
- manifest file hashes and record counts; and
- byte identity for every deterministic primary/replica artifact.

Stage wall times and source, media-fingerprint, and sample-output throughputs are
sealed into the final manifest. `--resume` reuses a stage only when the same
dataset version and Git revision have a recorded completed stage and every
expected artifact exists.

Freezing and ontology admission are deliberately separate. A technically
publishable freeze may contain only weak, unreviewed event intervals. Such a
version remains `event_conditioned_training_ready: false`, produces no
`next-event` targets, and cannot pass the reliable-event quality gate until the
human boundary-review report is complete.
