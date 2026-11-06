# Shared M3/M4 endpoint evaluation preparation

The shared entry point is `scripts/evaluate_coupled_dev.py` with `preflight`, `run` and `analyze` modes and required `--arm M3|M4`, `--training-root`, `--training-source`, and `--output-root`. It uses the sealed `configs/evaluation/week8_coupled_dev_v1.json`. Use separate output roots per arm and execute from the isolated evaluator package, not a live training checkout.

## Admission and inference

Preflight is CPU-only: it checks the full packaged source, original training source, training identity and admitted execution recipe, exact Feifei/B0 software versions, benchmark, dev-210 panel and B0 artifacts. Missing epoch-2 artifacts produce `waiting_for_epoch_2`. Complete endpoints must cross-link the production identity, epoch report, final completion, fixed exposure counters/chains, policy and full-training-predictor file checksums. It never selects epoch 1. M4 retains its separate approved time-budget contract and Lambda training identity; both arms evaluate on Feifei.

At launch, the coupled tensor envelope must match arm, interface, design, exact training identity, epoch, updates, normalizer and future-core configuration. Tensor hashes, exact names, shapes, dtype and finiteness are checked. Restore the pinned base/LoRA/action head, trained future core and trained attention/refinement state. The pixel decoder, JEPA output projection and teacher are absent from inference. Current images use the exact training box reducer, once, on canonical recorder RGB. State stays raw until the shared model normalizes it. The shared action decoder preserves clipping and binary-gripper behavior.

Rollouts use the existing scene construction and HDF5 recorder, 20 Hz and predict-8/execute-8. A durable attempt claim precedes scene imports and rendering, so restarting cannot grant extra attempts. Each failed attempt/partial file is retained; completed artifacts are recovered and verified. Immutable run identities/receipts, nonblocking output/training locks, idle-GPU checks, active-time and disk limits prevent accidental overlap or silent resume into another run.

Analysis keeps every dev task, records per-task exact settling-prefix/first-query input audits against B0, success and suite/category/difficulty counts, query latency and the original paired task bootstrap. Any mismatch qualifies the entire contrast as descriptive; missing/corrupt artifacts additionally make the audit incomplete. The original B0 rendering and JEPA FAIL qualifications persist. This development runner does not claim to replace the Week 9 factorial orchestrator or perform M4−M2 before M2 has its own audited outcomes.

## Packaging and later execution

Build with `PYTHONPATH=src python scripts/package_coupled_evaluation.py --output <new-package-directory> --git <working-git>`. Every byte of the 234-file accepted M3 qualification source is preserved; new evaluator files are additions. The same implementation handles M4 while validating its distinct training source. The tarball includes a sealed file manifest; preflight refuses changed source or contracts.

Run CPU-only preflight on Feifei with the existing B0 environment and the explicit environment wrapper `scripts/coupled_evaluation_feifei.sh`. Example after unpacking the package:

```bash
bash scripts/coupled_evaluation_feifei.sh preflight --arm M3 \
  --training-root /home/peimengsui/data/eventground-wam/libero-plus/training/week8-m3-coupled-v1 \
  --training-source /home/peimengsui/code/eventground-wam-m3-coupled-v1 \
  --output-root /home/peimengsui/data/eventground-wam/libero-plus/evaluation/week8-m3-dev-v1
```

M4 uses a separate transferred training bundle and source directory; preserve original JSON and checkpoint bytes and verify on receipt. A metadata-only staged bundle can pass provenance preflight but must remain `waiting_for_epoch_2`; it cannot launch. Preserve the full-training-predictor export for endpoint checksum verification even though it is never executed for inference. Create an empty `owner.lock` in the transferred bundle after completed transfer, for the evaluator's local bundle lock.

This preparation does not install/start a service, allocate a GPU, or run a rollout. Once an endpoint is complete and Feifei is free, perform the explicit runtime/model/recorder smoke and rendering audit, then launch a separately requested dev evaluation. The future Week 9 launcher additionally needs the cohort-wide GPU synchronization, latency reporting, clean-panel binding and five-arm input audit specified in `docs/WEEK9_EVALUATION_PROTOCOL.md`; these are not silently inherited from historical Week 4 acceptance rules.
