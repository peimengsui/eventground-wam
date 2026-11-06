# Week 11 integration and bounded pilot

The Step 3 implementation composes the original M4 inference adapter with an
independent FP32 instance of the selected Week 10 SFT event predictor. The scoped
`routed_horizon` hook replaces only `PixelCondition.horizons` before the existing
future core runs. It always removes its hooks, including after an exception.
Original M4 source, exports, the Step 1 experiment and Step 2 mapping stay sealed.

`EventGroundInferencePolicy` exposes only current front/wrist images, eight-value
proprioception, instruction and suite. Fixed M4 has no event instance during its
rollouts. EG-fixed8 runs, validates and records each prediction but uses eight;
adaptive uses the frozen nearest-onset rule. Both execute eight actions. Invalid
predictions fail the query. There is no event confidence threshold, learned
event embedding, action-side adapter switching or additional training.

The executable entry point is `scripts/run_week11_pilot.py`. It has no formal
evaluation mode. `scripts/package_week11_pilot.py` seals the executed source and
its inherited evidence before launch. Use the existing `scripts/week9_feifei.sh`
launcher to preserve the pinned numerical and simulator environment:

```sh
bash scripts/week9_feifei.sh run_week11_pilot.py --output /home/peimengsui/data/eventground-wam/libero-plus/results/week11-integration-pilot-v1
```

The runner rehashes both M4 exports and the event endpoint, validates completion
chains, checks executed event recipe source, and binds GPU/runtime identities.
Exactly eight optional initial-prefix repeats provide same-observation numerical
tests and renderer audits; they contain no policy actions. Qualification requires
exact normalized and decoded M4/h8 equality for all eight observations, event
agreement with the original FP32 greedy path, all-five-horizon routing, action
dependence on horizon and future tokens, unchanged tensors and endpoint reload.

The 24 pilot episodes follow the frozen arm and pair order. Lossless HDF5 uses
the inherited recorder loop. Durable per-query JSONL records include event and
conditional start/end probabilities, raw and mapped predictions, used horizon,
branch identities, synchronized full-query and component timings, and failures.
For valid `none`, conditional start/end scores are inactive diagnostics, not an
onset prediction. First queries and failed queries remain in latency summaries.
Simulator and recording wall time are recorded separately.

An active-time ledger and process-tree watchdog enforce the frozen three-hour,
28-GiB RSS, 90%-GPU, 200-GiB output and 50-GiB free-disk limits. Resource stops
escape episode retry handlers. Only classified infrastructure failures can use
one retry. Policy errors are final failures; integrity errors stop admission.
Every invocation, partial file and attempt journal remains available. Completed
receipts are reread and validated without extra simulator or model calls.

All settling and first-query fields are compared across all arm pairs and the
renderer repeats. Mismatches remain in the cohort and make the whole affected
contrast descriptive. Pilot success cannot select the checkpoint or mapping.
A conservative 1.2-margin runtime/storage projection is reported before any
later formal launch. Step 5 remains a separate action.
