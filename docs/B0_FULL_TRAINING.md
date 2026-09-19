# Full B0 training and recovery

## Declared five-epoch recipe

[`week4_b0_5epoch_v2.json`](../configs/training/week4_b0_5epoch_v2.json) declares
five complete epochs over all 1,715,954 frozen training chunks: **8,579,770
sample exposures**. Every episode's final short batch is included. Batch size is
16, with two CPU input workers and two prefetched batches per worker. Each worker
caches at most one decoded episode, capped at 4,096 source frames. No full-dataset
image or processor cache is materialized.

Recipe v1 remains preserved as the first pilot configuration. That pilot proved
systemd process restart/resume but failed exact GPU next-update equivalence. Recipe v2 adds
deterministic CUDA execution and pins the clean-40 manifest alongside `dev-210`.

- Pinned Qwen3.5-2B initialization; fresh rank-16 LoRA, projectors, and decoder.
  Training RNG seed 17, projector initialization seed 7, decoder seed 11.
- Deterministic Torch algorithms, cuBLAS workspace `:4096:8`, and no cuDNN
  benchmarking. GPU diagnostics verify that the next optimizer update matches
  uninterrupted execution after restoring the complete checkpoint.
- Frozen Week 2 assignments and verified train-only action normalization.
- The existing deterministic episode shuffle, then PCG64 anchor shuffle keyed
  by seed, epoch, and episode. Each epoch's complete batch-order digest is sealed
  before optimization. Worker scheduling does not change batch order.
- AdamW: projector/decoder LR `3e-4`, LoRA LR `1e-4`, betas `(0.9, 0.95)`,
  epsilon `1e-8`, weight decay `0.01`, gradient clipping `1.0`.
- Linear warmup over 2% of all five epochs' optimizer updates, followed by
  cosine decay to 10% of each group's peak LR. No gradient accumulation.
- Continuous-action L1 plus unit-weight gripper BCE; existing Qwen BF16 and
  projector/action-head precision behavior; unchanged camera and action contracts.
- Evaluate **every validation chunk** after each epoch. Select the epoch with
  the lowest sample-weighted validation hybrid objective; ties retain the earlier
  epoch. Test data and rollout scores do not participate in checkpoint selection.
- Preserve all five epoch policy exports. Reload the selected one and repeat
  full validation before publishing `B0.ckpt` and the sealed training report.

The full-training recipe is intentionally a new experiment, separate from the
Week 3 memorization schedules. The five-day launch guard requires a matching
real-data pilot, including exact disk reload and next-update equivalence. Its
projection charges all five training and validation passes at observed training
throughput, with a further 20% margin. This guard checks feasibility; it does not
guarantee wall time or policy quality.

## Checkpoints and interruption behavior

The training process saves every **five minutes or 2,000 updates**, whichever
comes first, at epoch boundaries, and after a requested termination reaches a
completed update. A recovery checkpoint includes:

- every allowlisted trainable model tensor;
- complete AdamW moments and scheduler state;
- Python, NumPy, CPU Torch, and CUDA RNG states;
- epoch, exact next batch, optimizer update and sample counts;
- the rolling consumed-sample digest, epoch loss sums, validation history, and
  selected-checkpoint identity;
- a binding to the source commit, recipe, pinned model, runtime lock, data,
  normalizer, and all epoch-order digests.

Writes use a temporary file, file `fsync`, atomic rename, and directory `fsync`.
The tensor file is committed before its digest-sealed JSON record. The last
three committed recovery generations are retained. An incomplete write is
ignored; a damaged newest checkpoint falls back to the newest intact generation
and records that recovery. If no intact generation exists, or the run identity
has changed, training fails explicitly instead of silently starting over.

Prefetched batches are disposable. Resumption starts at the saved next batch and
decodes only the needed episode, without replaying earlier episodes. Abrupt
termination may lose work since the last committed checkpoint; that work is
recomputed. Validation interrupted mid-pass restarts that validation pass.

One advisory file lock prevents two processes from writing the same run.
`events.jsonl` is an append-only attempt/progress log; checkpoint state is the
authority for completed updates. Replayed updates can appear twice in the log
after an abrupt interruption, distinguished by process ID and timestamp.

## Host-managed execution

The production run launched on Feifei on **2026-09-15 at 07:43:40 PDT** as
`eventground-b0-week4.service`, from source
`fe8e9a9086c36c765fb65a1b8421e4b6d783e965`. The
[passing pilot](evaluation/week4-b0-pilot-v2.json) and
[verified launch receipt](evaluation/week4-training-launch-v1.json) preserve the
launch and recovery evidence. The first periodic checkpoint at update 1,701
was hash-verified and loaded independently while training continued.

Run from a dedicated clean source checkout on `feifei`, using the existing locked
B0 virtual environment. `scripts/manage_b0_job.py` installs a **persistent user
systemd service**, enables it at user-manager startup, and requires user
lingering. The host owns the training process; closing SSH has no effect.

The service restarts failures after 60 seconds, with a limit of four starts per
hour. Recipe/integrity failures (exit 78) are not retried. A clean completion
does not restart. `systemctl --user stop` intentionally stops the job. Lingering
and the enabled unit allow recovery after a host reboot when the home directory,
GPU driver, and local data are available. No machine-wide service or reference
runtime is modified.

Start the pilot through the same launcher, supplying real host paths:

```bash
cd ~/code/eventground-wam-week4-run-v2
environments/b0/.venv/bin/python scripts/manage_b0_job.py start \
  --name eventground-b0-week4-pilot-v2 \
  --recipe configs/training/week4_b0_5epoch_v2.json \
  --repository-root ~/data/eventground-wam/libero-plus/datasets/libero-plus-lerobot/current \
  --frozen-root ~/data/eventground-wam/libero-plus/derived/frozen/week2-dataset-v1 \
  --normalization ~/data/eventground-wam/libero-plus/derived/b0/week3-b0-data-v1/action-statistics.json \
  --cache-dir ~/data/eventground-wam/huggingface-cache \
  --output-root ~/data/eventground-wam/libero-plus/derived/b0/week4-b0-pilot-v2 \
  --pilot-steps 1000
```

Production uses the same arguments with a fresh output root and service name,
replacing `--pilot-steps` with `--pilot-report <pilot-root>/pilot-report.json`.
The production process starts from fresh initialization; it never continues the
pilot weights. A restart of an existing production output automatically resumes
its latest intact checkpoint.

For the declared v2 run, use the same clean source checkout as the pilot:

```bash
cd ~/code/eventground-wam-week4-run-v2
environments/b0/.venv/bin/python scripts/manage_b0_job.py start \
  --name eventground-b0-week4 \
  --recipe configs/training/week4_b0_5epoch_v2.json \
  --repository-root ~/data/eventground-wam/libero-plus/datasets/libero-plus-lerobot/current \
  --frozen-root ~/data/eventground-wam/libero-plus/derived/frozen/week2-dataset-v1 \
  --normalization ~/data/eventground-wam/libero-plus/derived/b0/week3-b0-data-v1/action-statistics.json \
  --cache-dir ~/data/eventground-wam/huggingface-cache \
  --output-root ~/data/eventground-wam/libero-plus/derived/b0/week4-b0-5epoch-v2 \
  --pilot-report ~/data/eventground-wam/libero-plus/derived/b0/week4-b0-pilot-v2/pilot-report.json
```

Consult [`WEEK4_STATUS.md`](WEEK4_STATUS.md) for the last verified job state. Inspect any
existing service/output before starting, and disable the completed pilot service
after validating its report so it does not rerun at the next user-manager startup.

```bash
systemctl --user status eventground-b0-week4 --no-pager
journalctl --user -u eventground-b0-week4 -n 30 --no-pager
systemctl --user stop eventground-b0-week4
systemctl --user start eventground-b0-week4
```

If repeated failures exhaust the rate limit, inspect the logs and repair the
underlying problem, then use `systemctl --user reset-failed eventground-b0-week4`
followed by `start`. Preserve the source checkout and recipe for the duration of
the run; changed code belongs in a new versioned run.

## Evaluation handoff

Five-epoch training completed on **2026-09-16 at 14:56 PDT**. Epoch 5 was selected,
its checkpoint SHA-256 was independently verified, and repeating full validation
after reload reproduced all validation metrics. The sealed
[training report](evaluation/week4-b0-training-report-v2.json) contains all five
epoch results. Clean-40 completed at **20:25:17 PDT** that day with **26/40
successes (65%)**, 14 policy timeouts, and no exceptions/retries. All 40 recorded
episodes were independently validated. See the
[clean report](evaluation/week4-b0-clean40-v1.json),
[verification](evaluation/week4-clean40-verification-v1.json), and
[current work ledger](WEEK4_STATUS.md).

The selected checkpoint bundle contains the full-training manifest, sealed
completion report, normalizer identity, and `B0.ckpt`. The Week 3 cohort review
is not part of this new bundle's acceptance contract. Both Standard LIBERO
clean-40 and LIBERO-Plus `dev-210` are complete. The latter
scored **106/210 (50.48%)** on 2026-09-19, with all 210 artifacts validated and
zero exceptions/retries. See the [failure analysis](WEEK4_DEV210_FAILURE_ANALYSIS.md).
The installed OpenVLA public
reference scored 39/40 (97.5%) on the identical clean-40 panel; the
[paired comparison audit](WEEK4_PUBLIC_REFERENCE.md) documents evaluation parity
and unmatched training/model settings. The [acceptance freeze](WEEK4_CLEAN_ACCEPTANCE.md)
now defines the operational gate against this fixed external reference on 2,000
pairs, retaining a five-point margin and requiring a paired 95% interval. The
larger run completed at **68.25% B0 versus 96.75% OpenVLA** and failed the
frozen clean gate; training and diagnostic completion do not close Week 4.

The frozen [`clean-40` manifest](../evaluation/splits/libero-clean-v1/clean-40.json)
selects all ten canonical tasks in each of four suites, using initial-state
index zero and protocol-derived episode seeds. Names, instructions, BDDL,
initial-state files, and inventory hashes are sealed under
[`libero-clean-v1`](../configs/evaluation/libero_clean_v1.json). This is a
40-rollout clean sanity profile; its result cannot be equated with a public
multi-trial benchmark without matching the evaluation conditions.

`scripts/evaluate_b0_full.py` accepts the full-training completion bundle and
evaluates either complete `clean-40` or complete `dev-210`. It retains actual
runtime instructions, including language perturbations, and uses the shared
recorder, decoding, retry, and resume path without a Week 3 cohort restriction.
Use `--manifest-only` to validate task/source identities before a checkpoint is
available. The completed clean-40 run validates the full-checkpoint rollout path;
dev-210 also completed and passed artifact validation. The clean-performance
gate remains failed, and final Week 4 closure remains pending.
