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

## Frozen `week2-dataset-v1` result

The complete primary and replica builds ran on `feifei` from clean project
revision `0fb97edd9826dbc6387f5dc90812eaa8b7f116f3`. The final freeze manifest
has contract digest
`f7e3c4e419f8d2167c78eecb58f8e19e2c748e4aff98edcbe12ffb506b86f6d1`
and file SHA-256
`53d80156d9f6eb0451572b29ab7fcf38add08405ffa6ab26c7400899aad47ae2`.
Compact machine-readable evidence is committed at
[`freeze/week2-dataset-v1.json`](freeze/week2-dataset-v1.json).

The freeze contains 14,347 episodes, 2,238,036 frames, 8,395 initial-condition
groups, and 10,300,666 fixed-horizon samples. Episode counts are 11,521 train,
1,396 validation, and 1,430 test, with zero cross-split group overlap. All 11
deterministic artifacts are byte-identical across the two independent builds,
every primary sample contract and foreign key passed validation, and no partial
file remains.

The primary candidate, extraction, and split/sample stages took 28.8, 44.0, and
913.3 seconds, respectively. Split generation sustained 11,278 sample records
per second and decoded 31.4 initial camera frames per second, so no decoded-image
cache was added. Full independent validation took 162.8 seconds.

There are 16,742 weak/unreviewed `gripper_close` intervals and 160 ambiguous
decisions. All weak intervals were excluded from event targets, leaving
`next-event: 0`. The artifact freeze is publishable; event-conditioned training
remains blocked on the pending human quality gate.
