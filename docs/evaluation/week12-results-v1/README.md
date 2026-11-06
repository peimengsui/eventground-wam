# Week 12 Steps 2–3 package

**Completed November 1, 2026.** Reporting from existing evidence under the
[frozen scope](../../WEEK12_SCOPE.md). This package supplies the numerical and
figure handoff; project-wide reproducibility packaging and the final technical
report remain later steps.

## Read and reuse

| File | Contents |
| --- | --- |
| [RESULTS.md](RESULTS.md) | Consolidated findings, methods, cost and limitations |
| [gallery.html](gallery.html) | Local figure and qualitative evidence gallery |
| [CAPTIONS.md](CAPTIONS.md) | Figure captions and export links |
| [QUALITATIVE.md](QUALITATIVE.md) | Example selection and interpretation boundaries |
| [tables/TABLES.md](tables/TABLES.md) | All five table exports, with CSV links |
| [consolidated.json](consolidated.json) | Full-precision numerical handoff |
| [figure-index.json](figure-index.json) | Five generated figures and four main existing panels |
| [examples.json](examples.json) | Fourteen case records, frame positions and provenance |
| [numerical-verification.json](numerical-verification.json) | 2,210 numerical cross-checks |
| [source-manifest.json](source-manifest.json) | SHA-256 hashes of consumed local evidence and scripts |
| [verification.json](verification.json) | Final source, artifact and link checks |
| [rebuild-verification.json](rebuild-verification.json) | Fresh-build equality for numerical files, tables and chart PNGs |
| [visual-review.json](visual-review.json) | Figure review, corrected layout issues and inspection scope |
| [package-index.json](package-index.json) | Final package inventory and hashes |

Open `gallery.html` locally in a browser. It is a static document with relative
image links and no external assets or network scripts. The five new figures
have PNG, SVG and PDF exports; the existing qualitative panels and frames retain
their original PNG bytes. Four formal gain/loss panels are designated Figure
4a–d. Eight formal classes, three mechanism cases and three event examples are
included in total.

All 15,000 Week 9 outcomes and all 2,001 formal Week 11 outcomes are reconciled
with the retained metadata. Original paired resampling units/seeds and intervals
are reproduced. Revised event IoUs, source-bootstrap contrasts, continuation
pair counts, fresh H8 variants/task-cluster sensitivity and available timing
arrays are also checked. This package introduces no new statistical endpoint.
The original decisions and qualifications remain in force.

A fresh reporting build reproduced all 2,210 numerical checks and byte-matched
the three numerical/provenance JSON outputs, all five CSV tables and all five
generated PNG charts. All 29 copied qualitative images match their original
bytes. All 34 PNGs decode, and the five SVG/PDF exports parse successfully.

## Rebuild the reporting artifacts

From the repository root, use Python 3.11 with NumPy, Matplotlib, Pillow and
pypdf. The original build used the local Anaconda interpreter. The artifact
verification records library versions. Use a new output directory to preserve
this sealed package:

```bash
python scripts/build_week12_results.py --output /tmp/week12-results-rebuild
python scripts/render_week12_figures.py --output /tmp/week12-results-rebuild
python scripts/write_week12_results.py --output /tmp/week12-results-rebuild
python scripts/verify_week12_results.py --output /tmp/week12-results-rebuild
```

The build requires the retained source paths named in `source-manifest.json`,
including local Week 9 metadata under `outputs/`. It does not load models or
access the simulator, remote machines, full-10K or test. Run the scripts in order:
the numerical builder initializes the source manifest, and rendering/writing
extend it. PDF creation timestamps and SVG element IDs can vary; compare the
full-precision JSON/CSV and PNG outputs for substantive rebuild equality.

The verifier is read-only unless `--write-report` is supplied. If a package
index exists, it checks every sealed file and rejects unindexed files. It
checks source hashes, copied media, exports, tables and local links; the
numerical builder performs the independent resampling and score calculations.

The `source/` directory snapshots all four scripts for provenance. Execute the
repository's `scripts/` copies so their repository-root resolution is correct.
`before/` and `after/` preserve the navigation/status changes. They record this
reporting handoff, not a new scope freeze.

## Evidence limits

Formal means evaluated under a frozen protocol, not an untouched confirmatory
test. One training seed, reused panels/development cases, renderer/input
qualifications, JEPA's research-use exception, unresolved independent event QA
and unresolved on-policy forecasting accuracy remain explicit. Selected
qualitative cases do not estimate prevalence. Historical raw recording/input
audits remain authoritative; no fresh remote byte audit is claimed.

Steps 4–6 remain unstarted in the [current ledger](../../WEEK12_STATUS.md).
