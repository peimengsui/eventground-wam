# Week 11 formal execution child

This child implements Step 5 of the frozen Week 11 experiment. The user's request
to finish launching the evaluation authorizes all 2,001 formal episodes. There
are no training updates or additional qualification episodes. The design,
horizon mapping, endpoints, manifests, seeds, precision and qualified inference,
recording and retry code remain unchanged.

The source archive extends the exact executed pilot archive (source digest
`ca0d0836a6bedf95672ed06554e3a979fb78b0f4a4555f7bf752fe3de82423a9`).
Every inherited file remains byte identical; additions are enumerated and
hashed in `week11-formal-source-manifest.json`. Its archive and manifest are
written under `docs/evaluation/week11-formal-v1/package-v1`.

Run `bash scripts/week9_feifei.sh run_week11_formal.py --output
/home/peimengsui/data/eventground-wam/libero-plus/results/week11-formal-v1` from
the extracted child source on Feifei. `--preflight-only` verifies the full source,
all pilot inventory files, endpoint bytes, numerical qualification, benchmark,
runtime, clean states and projection. Preflight is charged to the same durable
aggregate and first unfinished panel ledgers as execution. Only the child
source directory in PYTHONPATH changes from the qualified runtime.

The serial order is M4-fixed, M4-EG-fixed8, M4-EG-adaptive; each runs targeted-467
then clean-200. Each model's tensors must match the qualification before use
and after each panel. The baseline has no resident event model. The original
scene and metadata builder and the qualified recorder/executor are reused.
Completed receipts are revalidated on resume; attempts and partials remain.
There is one classified infrastructure retry per episode and no policy-failure
retry. A process lock prevents simultaneous writers. There is no automatic
service restart, new-seed extension or outcome-dependent stopping.

The pilot's full-suite step-limit projection is reused with its 1.2 margin:
about 10.17 hours including qualification and 163.49 GiB total output. Current
free space must cover projected additional output plus 50 GiB. The watchdog
checks process-tree RSS and CUDA allocated/reserved peaks every 0.2 seconds,
and output bytes, free disk and competing GPU jobs every five seconds. Limits
remain 28 GiB RSS, strictly below 90% GPU memory, 200 GiB output including pilot,
12 active hours per arm/panel and 48 aggregate active hours including pilot,
preflight, loading, failed attempts, resumes and audits. Crash ledgers retain
their previous active charge plus a conservative ten-second allowance.

Each panel retains full recordings, sealed timings, query/event logs, retry
journals, receipts and a report. Query latency includes failed/interrupted
attempts as well as completed attempts. Required settling and first-query
inputs are audited for every episode and all three pairwise contrasts. A
mismatch remains in the denominator and makes that contrast descriptive;
missing/corrupt evidence cannot support a positive result. The completion
artifact reports audit status and bindings. Step 6 paired bootstrap analysis
and gate interpretation remain a separate task after the fixed matrix finishes.
