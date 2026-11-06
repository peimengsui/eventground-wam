# Research artifacts and optional checks

The default checkout contains code, authored configs, dependency locks and
implementation notes. Generated evidence is retained separately. The cleanup
removes files from Git tracking without deleting the original local copies.

## Files supplied separately

| Ignored path | Role |
| --- | --- |
| `docs/evaluation/` | Evaluation reports, original source snapshots, tables, figures and reproduction packages |
| `docs/freeze/` | Dataset QA/freeze receipts |
| `docs/model/` | Exported normalizers and model metadata |
| `evaluation/splits/` | Generated benchmark panels |
| `configs/datasets/*/` | Generated dataset/order/future-pair manifests and group indexes |
| `docs/blog/eventground-wam/` except `source/*.py` | Website draft, rendered assets and provenance |
| `outputs/`, `runs/`, `checkpoints/`, `data/` | Raw/generated experiment payloads |

Configuration JSON directly under `configs/training/`, `configs/evaluation/`
and `configs/datasets/` remains tracked: these files describe settings and data
contracts rather than storing evaluation results. Do not blanket-ignore every
JSON file. CSV/JSONL exports, array/recording files and portable archives are
ignored as generated artifacts.

No public artifact download is provided here. To run a historical workflow,
restore the appropriate files from your retained local research archive into
the original relative paths. The scripts still check their recorded identities;
do not bypass source, checkpoint or output guards to make a launch proceed.
Historical prose records may refer to files that are deliberately absent from
the code-only checkout.

## Tests

The default command runs self-contained code tests and reports external-evidence
audits as skipped:

```bash
PYTHONPATH=.:src MPLBACKEND=Agg .venv/bin/python -m pytest
```

After restoring the relevant artifact directories, explicitly enable the audits:

```bash
PYTHONPATH=.:src MPLBACKEND=Agg .venv/bin/python -m pytest --research-artifacts
# Or run only the evidence audits:
PYTHONPATH=.:src MPLBACKEND=Agg .venv/bin/python -m pytest --research-artifacts -m research_artifacts
```

Opting in does not suppress missing-file or integrity failures. Each marked test
still runs its original assertions. Unmarked tests do not depend on retained
research outputs. The test runner does not fetch artifacts or start experiment
jobs automatically.

## Reproduce statistics and figures

Use Python 3.11 and the separately pinned analysis dependencies:

```bash
python3.11 -m pip install -r environments/analysis/requirements.txt
python3.11 scripts/verify_week12_reproducibility.py \
  --package /path/to/retained/week12-repro-v1 \
  --extract-to /tmp/eventground-analysis-source
python3.11 /tmp/eventground-analysis-source/scripts/reproduce_week12_analysis.py \
  --output /tmp/eventground-analysis-results
```

The package directory must contain its original archive, manifests and receipts;
both output destinations must be new. Verification uses the original hashes.
The archive contains the matching historical source and inputs, so source files
in this code-only checkout need not be substituted into a sealed past execution.
Training, inference and simulation additionally require model weights,
demonstrations and benchmark assets. The portable metadata rebuild does not
replicate those GPU runs.

## Git history

This cleanup changes the current tracked tree. The previous artifact snapshot
remains in earlier commits; `.gitignore` and `git rm --cached` do not erase it.
A shallow clone (`git clone --depth 1`) avoids downloading that older history.
Removing the artifacts from all historical commits would require a separate
history rewrite and coordination with existing clones.
