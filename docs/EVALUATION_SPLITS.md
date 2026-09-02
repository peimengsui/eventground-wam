# Reproducible evaluation splits

The committed LIBERO-Plus task manifests live under
`evaluation/splits/libero-plus-v1/`. They are generated from the task
classification file at the benchmark revision pinned by the locked protocol.

Each manifest records:

- the protocol ID and canonical digest;
- the benchmark and classification-file digests;
- the exact deterministic generation algorithm and seed;
- both the one-based classification ID and zero-based runtime task index;
- suite, task name, perturbation category, difficulty, and episode seed;
- a canonical SHA-256 digest covering all task records.

## Split hierarchy

The manifests are strict nested subsets:

```text
sanity-35 ⊂ dev-210 ⊂ stratified-1K ⊂ full-10K
```

- `sanity-35` contains one task from each labeled category×difficulty cell.
- `dev-210` contains six tasks from each of the 35 labeled cells. Within each
  cell, suite quotas follow proportional constrained largest-remainder
  apportionment.
- `stratified-1K` retains every `dev-210` task and apportions the remaining
  capacity proportionally across suite×category×difficulty strata. The source's
  unlabeled-difficulty tasks remain an explicit `unlabeled` stratum.
- `full-10K` contains all 10,030 classified tasks.

Stable SHA-256 ranking selects tasks within every quota. This avoids dependence
on Python or NumPy random-number-generator implementations.

## Generate or verify

Run generation on `feifei`, where the pinned upstream checkout is installed:

```bash
./scripts/run_libero.sh python scripts/generate_eval_splits.py
```

Normal validation never rewrites the manifests:

```bash
./scripts/run_libero.sh python scripts/generate_eval_splits.py \
  --check \
  --verify-runtime
```

`--verify-runtime` compares all 10,030 manifest names against the corresponding
runtime task at `classification_task_id - 1`, preventing an unnoticed one-based
versus zero-based indexing error.

The generator refuses any classification file whose bytes do not match SHA-256
`faa87cce3e3ba434da01df7c77523a391b5f2912e4774330b0aa1be5f6a999e6`.
