# Fresh two-epoch all-L1 B0 on trajectory v2

**Milestone disposition:** [Week 4 closed September 24](WEEK4_CLOSEOUT.md) with
user acceptance of this fixed epoch-2 checkpoint as the internal study baseline.
The scientific acceptance statuses below remain unchanged. Additional B0
optimization is deferred; [Week 5 planning has started](WEEK5_STATUS.md).

**Authorized:** September 22, 2026. **Updated:** September 24, 2026.
**Status:** training, validation and both evaluation panels are complete. Final
epoch 2 scored **1,867/2,000 clean (93.35%)** and **162/210 dev (77.14%)**, versus
OpenVLA **1,935/2,000 (96.75%)** and **170/210 (80.95%)**. The historical clean gate
is inconclusive; formal v2 readiness is blocked by one exact-input mismatch and
would remain inconclusive under unchanged outcomes. The renderer discrepancy and
permanent analyzer-stop fix are documented in the
[completed audit and failure review](WEEK4_FRESH_V2_FAILURE_ANALYSIS.md).

Final checkpoint SHA-256:
`7a385fea68fe5d14f642b31e18f8e2e166ccd984d9b4f1ff5b68484191a9b720`.
Final micro validation: all-L1 **0.139842**, motion-L1 **0.145152**, gripper-L1
**0.107982**, decoded MAE **0.139645**, gripper accuracy **94.6698%**.
[Training evidence](../outputs/week4-fresh-v2-training-final-20260924/README.md) ·
[Rollout results](../outputs/week4-fresh-v2-evaluation-20260924/README.md).

**Launch record:** formal pipeline launched at **08:52 PDT**, September 22,
under host-owned `eventground-b0-fresh-v2.service`. Following completion and the
September 24 audit, that service is stopped and disabled for boot startup.

Source commit: `d38968007067cecfcd0dbcd2b1a297a18ad7283d`.
Local verification: **303 tests passed**, Ruff and shell syntax passed.

The 128-update pilot passed after a deliberate SIGTERM at update 22 and one
automatic systemd restart. Saved predictions and the complete next optimizer
update replayed bit-for-bit. Throughput was **21.45 examples/second**, peak
allocated CUDA memory **8.53 GiB**. Projected training plus validation is
**57.26 hours with a 25% training-time margin**; rollout time is additional.
Both simulator smoke episodes produced valid matched-input artifacts with zero
retries (0/2 task successes at only 128 pilot updates; this was a plumbing test).

Pilot evidence: [recovery and throughput](evaluation/week4-fresh-v2-launch/pilot-report.json),
[process resume](evaluation/week4-fresh-v2-launch/pilot-process-resume.json),
[simulator smoke](evaluation/week4-fresh-v2-launch/pilot-smoke.json).

## Fixed experiment

- Base pinned Qwen3.5-2B; fresh broad rank-32 LoRA (alpha 16), OFT action head,
  state projector, Adam and scheduler. No prior B0 weights, optimizer or teacher.
- Run B architecture, native front/wrist inputs, state input, eight actions,
  no augmentation. Mean L1 over all raw 8 x 7 values, including signed gripper.
- The frozen trajectory-v2 assignments: 11,435 train episodes / 1,341 groups,
  1,706,022 valid chunks. Train-only action and state quantiles were refitted.
  All 12,963 train/validation parquet hashes matched the structural audit;
  preparation did not read test payloads.
- Exactly two epochs: **117,132 updates / 3,412,044 presentations**. Maximum
  batch 32 with episode tails retained. Same deterministic 128-update mixing
  and 15k stage fences as Run B, reset each epoch. No new sampling weights.
- Fresh-training optimizer follows the original OFT recipe: both LoRA and head
  peak LR **1e-4**, 2% warmup (2,342 updates), cosine to **1e-5**, AdamW betas
  (0.9, 0.95), epsilon 1e-8, decay 0.01, gradient clipping 1.0. Run B's constant
  1e-5 was continuation-specific and is not used as the fresh peak LR.

## Frozen measurements and decision

1. Full v2 validation after epochs 1 and 2: 230,250 chunks, 1,528 episodes,
   180 groups, 40 tasks. Report motion/gripper/all L1, decoded MAE and gripper
   accuracy, both micro and equal-task/group/variant/chunk hierarchical means.
2. Existing development-200 after each epoch. These results are descriptive;
   there is no early checkpoint selection or outcome-dependent recipe change.
3. Fixed epoch-2 endpoint on clean-2000 and dev-210. Reuse audited clean OpenVLA,
   original B0 and 72k OFT results. Run the installed OpenVLA on identical dev-210
   cases and seeds; verify scene and first-query inputs before paired claims.
4. v2 readiness: every clean suite >=70%, and either clean >=85% with the paired
   dev gap's lower 95% bound >=-5 pp, or clean >=80% with observed dev advantage
   >=5 pp and lower bound >0. Otherwise report BELOW_TARGET or INCONCLUSIVE.
   The dev interval uses 20k stratified paired bootstrap draws within suite x
   perturbation category, PCG64 seed 20260922. It does not capture training-seed
   or independent base-task uncertainty. No automatic extra runs to resolve an
   inconclusive result. Historical clean <=5 pp decisions remain unchanged.
5. The v2 supervised test partition stays locked. This experiment changes both
   split and recipe relative to historical B0, so it cannot isolate loss effects.

## Reliability and budget

- Independent host checkout and output root; persistent systemd user service
  with lingering, restart on failure, no SSH lifetime dependency.
- Atomic checksummed recovery every 300 seconds or 2,000 updates, also after
  update 1 and epoch boundaries. Retain three generations plus epoch policies.
- Recovery includes weights, Adam, scheduler, Python/NumPy/CPU/CUDA RNG, exact
  epoch/mixed-batch cursor and sample chain. Resume rechecks source and recipe.
- Production requires a 128-update pilot, observed process interruption/resume,
  bitwise identical next-update replay, and two valid simulator smoke episodes.
- 72 active hours for pilot/training/validation; 48 for evaluations. Output cap
  100 GiB, free-space reserve 50 GiB. These are stopping limits, not estimates.

## Files

- Recipe: `configs/training/week4_b0_fresh_all_l1_v2.json`
- Acceptance: `configs/evaluation/week4_fresh_v2_acceptance.json`
- Normalizers: `docs/model/week4-fresh-v2/`
- Service: `eventground-b0-fresh-v2.service`
- Host source: `/home/peimengsui/code/eventground-wam-fresh-v2`
- Host output: `/home/peimengsui/data/eventground-wam/libero-plus/derived/b0/week4-fresh-all-l1-v2`
