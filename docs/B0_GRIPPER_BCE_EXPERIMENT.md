# B0 gripper BCE overfit experiment

This versioned experiment tests the hypothesis exposed by the fixed
ten-demonstration L1 runs: the binary gripper decision should be optimized as a
classification output rather than as one continuous regression dimension.

## Locked model and objective

The Qwen3.5-2B backbone, rank-16 LoRA placement, conditioning adapter, and
four-layer action-query Transformer are unchanged. The action projection still
has seven outputs per query. Outputs 0 through 5 are continuous normalized
actions. Output 6 is interpreted as a gripper logit.

For target gripper command `g` in `{-1,+1}`, the binary label is `g > 0`, where
`-1=open` and `+1=close`. The training objective is:

```text
mean_L1(raw_predictions[..., :6], targets[..., :6])
  + 1.0 * BCEWithLogits(raw_predictions[..., 6], targets[..., 6] > 0)
```

There is no label smoothing, class weighting, auxiliary regression loss, or
change to the continuous-action L1 term. At inference, a nonnegative gripper
logit decodes to `+1`; a negative logit decodes to `-1`.

## Optimization and measurement

The experiment initializes from the immutable, reload-verified v2 L1
checkpoint. Because the objective changes, its loss-ratio baseline is recomputed
from the hybrid objective before the first BCE update. Absolute epoch and step
coordinates continue from v2.

The decoder/projector learning rate is `1e-4`, the LoRA learning rate is `3e-5`,
and the deterministic cosine schedule decays both to 1% of their initial values.
AdamW settings, seed 17, batch size four, gradient clipping, and complete-cohort
evaluation each epoch are unchanged. The declared endpoint is epoch 200.

Diagnostics separately report hybrid objective loss, six-dimensional continuous
MAE, gripper BCE, decoded seven-dimensional action MAE, exact gripper sign error
count, and minimum signed gripper-logit margin. The existing cohort gates remain
loss ratio at most `0.20`, decoded action MAE at most `0.10`, and exactly
11,632/11,632 correct gripper signs.

## Predeclared BCE-weight ablation

If the unit-weight run improves classification but plateaus below exact
memorization, one separate run starts again from the same immutable v2 L1
checkpoint with `gripper_bce_weight=10.0`. To bound its effective update size,
decoder/projector learning rate is `1e-5` and LoRA learning rate is `3e-6`;
both use the same 50-epoch cosine schedule down to 1%. All other model, cohort,
optimizer, and acceptance settings remain unchanged. Its hybrid-objective
baseline is recomputed before its first update because the BCE weighting differs.

## Result and reviewed disposition

The unit-weight run completed at epoch 201. Its final decoded action MAE was
`0.02796977` and its hybrid-objective ratio was `0.12608315`, so both continuous
action gates passed. Gripper classification reached 11,580/11,632 correct
output slots (`99.552957%`) at the final checkpoint. The best full-cohort
diagnostic was 11,581/11,632 (`99.561554%`) at epoch 194. Checkpoint reload was
exact.

The original predeclared exact-memorization gate remains recorded as failed;
the evidence was not rewritten to turn 99.55% into 100%. A separate post-hoc
engineering review accepts the overfit exercise with documented residuals for
the limited purpose of advancing to the next Week 3 development step.

The reload-verified checkpoint audit reduced the 52 wrong output slots to 32
unique source action frames because overlapping eight-step chunks repeat source
commands. All 52 errors were within four frames of one of the cohort's 28
gripper-command transitions, 40 were within one frame, and 24 were exactly on a
transition. Only 1,858/11,632 cohort slots are within four frames of a
transition, and there were no errors outside that window.

This is evidence of residual temporal-boundary difficulty rather than a corrupt
label stream: all targets are exact `-1/+1`, 1,504 overlapping source frames
agree across every chunk that contains them, and no exact duplicate conditioning
input has conflicting horizon labels. The labels are direct source control
commands, not derived human annotations. Accordingly, the risk is low for
closing this diagnostic, but this decision does not establish rollout or
generalization quality.

The reviewer stopped the weight-10 ablation. Its partial directory is preserved
with only its sealed cohort manifest and sidecar; it produced no final report or
checkpoint and is not part of the acceptance basis. Compact evidence is in
[`evaluation/week3-b0-overfit-10demo-bce-v1.json`](evaluation/week3-b0-overfit-10demo-bce-v1.json).
