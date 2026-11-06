# Week 6 Step 5 — fresh production JEPA CPT

**Status: complete and independently verified.** Training finished September 27,
2026 at **20:57:28 PDT**, after 20 hours 42 minutes with zero service restarts.
All **221,633 optimizer updates / 8,221,365 unique training pairs** are verified.
The actual final recovery payload and exported `C_jepa-candidate.pt` have identical
model weights; optimizer/scheduler/RNG state and the full independently replayed
sample chain pass. Frozen base and teacher hashes are unchanged. See the
[completion ledger](evaluation/week6-step5-completion-v1.json) and
[independent final audit](evaluation/week6-jepa-cpt-final-v1/verification.json).
Step 6 subsequently completed with a failed front-camera h=2 gate. The
[Week 6 final decision](WEEK6_CLOSEOUT.md) retains this candidate and rejects
promotion to `C_jepa-v1`.

Fresh production training started on Feifei at
**September 27, 2026, 00:15:20 PDT** under the enabled, persistent user service
`eventground-jepa-cpt-v1.service`. At the retained 00:22:21 PDT snapshot it had
completed **1,175 updates / 43,298 pair presentations**, with no service restart.
The [launch ledger](evaluation/week6-step5-launch-v1.json) preserves that historical
startup snapshot. The persistent service subsequently completed successfully.

The run uses the exact [passed Step 4 recipe](WEEK6_JEPA_TRAINING.md), source
snapshot, runtime and common order. No training code or numerical settings were
changed for launch. The previously approved source archive was copied on Feifei
to a separate read-only production directory. Its 69 files were checked against
the pilot inventory, and the new output directory was verified empty before start.

## Fixed run

| Item | Production contract |
| --- | --- |
| Exposure | One complete pass: 8,221,365 train pairs, 221,633 optimizer updates |
| Order | Exact Pixel common order, logical batches and all five valid horizons |
| Initialization | Fresh 4,238,592-parameter Pixel core and 526,336-parameter JEPA head; no trained parent or diagnostic weights |
| Frozen models | Pinned Qwen base and accepted canonical single-image JEPA teacher |
| Loss | FP32 cosine + 10 × anchor-grouped variance; verified logical-batch gradients |
| AdamW | LR 1e-4, weight decay 1e-4, 1,000-update warmup, fixed cosine decay to 1e-5 |
| Target cache | Exact FP32, pool-local host RAM only; 4 GiB cap |
| Recovery | Full model, optimizer, scheduler, Python/NumPy/Torch/CUDA RNG, cursor, pair count and sample chain |
| Saves | Every 500 updates or 900 seconds; three committed generations |
| Resource limits | Six CPU equivalents, 24 GiB host RAM, 5 GiB minimum free disk |
| Runtime | 72 cumulative active hours; 73-hour per-process service backstop |
| Pilot estimate | 34.44 hours including 1.5× margin; validation excluded |
| Endpoint | Fixed final update only; export `C_jepa-candidate.pt` after final integrity checks |

The estimate is a pilot projection, not a completion guarantee. Recovery and
target recomputation can add work. Training loss at startup does not establish
held-out prediction quality. No validation or test payloads are used by training.

## Verified startup

The [CPU startup audit](evaluation/week6-jepa-cpt-launch-v1/startup-verification.json)
loaded preserved copies of the actual update-0 and update-500 checkpoints while
the live trainer continued. It verified:

- Update 0 exactly equals a newly constructed model using the frozen seeds;
  optimizer state is empty, pair count and loss sum are zero, and no pilot or
  diagnostic weights were loaded.
- Production has a distinct training identity from the pilot, with identical
  source, recipe, runtime and common-order pins. The initial frozen base and
  teacher match the accepted pilot/target identities.
- The first periodic checkpoint has **500 optimizer/scheduler steps and 18,277
  unique training pairs**. Independently reading training pair metadata and
  replaying every batch reproduces its complete sample-chain digest.
- Model and optimizer tensors are finite, the model has changed from fresh
  initialization, and all four RNG state families are present. Checkpoint file
  sizes, hashes, identities and sealed commit records match.

The audit reads training metadata and checkpoint copies on CPU; it adds no
optimizer exposure and does not interrupt production or rerun the GPU pilot.
The two preserved payloads occupy 76,388,902 bytes outside the rotating recovery
directory. Their hashes and component digests are retained locally; the payloads
stay on Feifei. The [audit script](../scripts/verify_jepa_launch.py) is separate
from the frozen training source and passed lint and the real-checkpoint audit.
Step 4's 471 passing tests remain the unchanged training-code validation.

The [retained progress snapshot](evaluation/week6-jepa-cpt-launch-v1/progress.json)
shows the service active, enabled and running with zero restarts. At 00:22:21 PDT,
GPU use was 6,124 MiB of 32,607 MiB; service-accounted host memory had peaked at
4,450,906,112 bytes. These are startup observations, not full-run peak claims.

## Artifacts and service

All following paths are on **Feifei**:

| Artifact | Path |
| --- | --- |
| Read-only source | `/home/peimengsui/code/eventground-wam-jepa-cpt-v1` |
| Run and rotating recovery | `/home/peimengsui/data/eventground-wam/libero-plus/training/week6-jepa-cpt-v1` |
| Launch receipts and preserved startup checkpoints | `/home/peimengsui/data/eventground-wam/analysis/week6-jepa-cpt-production-v1` |
| Enabled service | `/home/peimengsui/.config/systemd/user/eventground-jepa-cpt-v1.service` |

The checked-in [service definition](../configs/systemd/eventground-jepa-cpt-v1.service)
uses user lingering, so it survives logout and is enabled at host startup.
Recoverable process failures resume from a committed checkpoint; exit codes 1
and 75 stop automatic retry for review. Retries are bounded, and the runner
retains the cumulative runtime budget across processes. A completed `report.json`
prevents service relaunch from repeating the finished run.

Read-only progress commands:

```bash
ssh -o BatchMode=yes feifei 'systemctl --user status eventground-jepa-cpt-v1.service --no-pager'
ssh -o BatchMode=yes feifei 'cat /home/peimengsui/data/eventground-wam/libero-plus/training/week6-jepa-cpt-v1/status.json'
```

## Verified completion and final disposition

The [final audit script](../scripts/verify_jepa_cpt.py) checked every pair record,
reconstructed the complete sample chain, loaded actual recovery tensors on CPU,
recomputed state-component digests and loaded the exported candidate. It also
reconstructed the fresh seeded model and verified initialization. There were no
production restarts or replayed updates. Earlier Step 3/4 diagnostic exposure
remains separate.

Active training time was **20.7014 hours**. Peak Torch GPU allocation was
**4.63 GiB** (reserved 4.84 GiB), peak process host RSS **6.21 GiB**, and maximum
target-cache storage **1.17 GiB** in RAM. The final full recovery checkpoint is
57,288,907 bytes and the candidate export is 19,086,497 bytes. The completed
training report records zero validation/test payload reads.

Step 6 applied the unchanged
[held-out prediction, retrieval and collapse gate](WEEK6_JEPA_BASELINES.md).
Full validation, the fixed visual review and appearance diagnostics are complete.
The candidate fails the front-camera h=2 criterion and is not accepted as
`C_jepa-v1`; the [final closeout](WEEK6_CLOSEOUT.md) preserves its exact export
and the original gate. No downstream policy training has launched.
