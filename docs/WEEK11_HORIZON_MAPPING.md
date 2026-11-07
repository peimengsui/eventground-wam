# Week 11 Step 2: event-to-horizon contract v1

**October 25, 2026: one deterministic rule is frozen.** The authority is
[week11_horizon_mapping_v1.json](../configs/evaluation/week11_horizon_mapping_v1.json).
It is a child of the immutable Step 1 experiment (local archive: `WEEK11_EXPERIMENT.md`), digest
`5d7e919252104f9814086474f6a92633edc1d86bff984608c0bbb80a7b4c4a0a`.
Arms, checkpoint pins, panels, budgets, execution cadence and success criteria
remain those of the parent. Its historical pending-mapping fields are preserved;
this child supplies their values rather than rewriting the parent.

## Rule and rationale

For a valid positive prediction with onset offset `s`, select the supported
horizon minimizing **absolute distance in control frames**:

`h = argmin over {2,4,8,16,32} of (abs(h - s), h)`.

The second key breaks exact ties toward the **shorter** horizon. This minimizes
quantization displacement from the predicted onset without imposing a universal
early/late bias. Choosing the shorter horizon only at ties limits unnecessary
lookahead where both choices are equally close. It is an engineering convention,
not evidence that earlier futures are better for control. Distance is linear in
frames, not logarithmic or proportional. There is no learned mapping.

| Predicted onset, frames from the current observation | Selected horizon | Lookahead at 20 Hz |
| --- | --- | --- |
| 1–3 | 2 | 0.10 s |
| 4–6 | 4 | 0.20 s |
| 7–12 | 8 | 0.40 s |
| 13–24 | 16 | 0.80 s |
| 25–32 | 32 | 1.60 s |
| Valid `none` | 8 | 0.40 s |

Thus 3 maps to 2, 6 to 4, 12 to 8, 24 to 16, and 30 to 32. Supported starts are
fixed points. Start 1 maps to the smallest supported horizon, 2; starts outside
1–32 are invalid and are **not clamped** into the valid range. The quantizer is
monotone, and maximum displacement for a valid start is eight frames at start 24.
Do not interpolate latent predictions or make multiple future-core calls.

The rule was selected from the interface and absolute-frame-distance objective,
not by optimizing robot success, label agreement or timing error. Week 10's
train/dev/report summaries and diagnostic sensitivity calculations were already
known. This is not a blind preregistration. The new rule and its code are sealed
before recomputing diagnostics on the existing dev predictions. No alternative
mapping sweep or report-panel rescoring is part of this step. Diagnostic results
cannot change this rule after the seal.

## Prediction semantics and validation

Use the selected SFT endpoint's existing FP32, current-observation-only path.
Decode one prediction per action query with `group_size=1`, `greedy=True`, and
`temperature=1.0`. Preserve the existing autoregressive order: event, onset,
interval end. Exact logit ties use the existing first-index argmax behavior:
event IDs are `none`, `gripper_close`, `contact`, `grasp`; start offsets ascend
1–32 and permissible end offsets ascend 2–33. No sampling or beam search.

Pass only the canonical three-field result of `serialize_tokens` to the mapper:

```json
{"next_event": "contact", "interval": [12, 18], "horizon": 12}
```

The half-open interval describes a predicted event; `horizon` is its **start**,
not its midpoint, end or duration. The example maps to supported horizon 8.
Positive types are `gripper_close`, `contact` and `grasp`; all use the same rule.
Require exactly those three keys, an admitted string type, a two-element list
of integer offsets satisfying `1 <= start < end <= 33`, and integer
`horizon == start`. Booleans, floating offsets, stringified numbers and extra
fields are rejected. A clipped end of 33 is valid but does not affect selection.
No target, object identity, remaining episode length or ongoing-event label is
available to the mapper. It cannot correct a confidently wrong event or onset.

A valid negative is exactly:

```json
{"next_event": "none", "interval": null, "horizon": null}
```

`none` means no new onset within the next 32 frames, even if an event is currently
ongoing. It selects **8**, preserving M4's fixed lookahead for this case. Log
`fallback_reason = "no_new_onset"`. Do not infer whether an event is ongoing,
reuse the previous positive prediction, suppress repeated event types, or
translate `none` into stopping the robot.

Malformed predictions raise `InvalidEventPrediction`. Missing/unknown types,
out-of-window starts, inconsistent horizons, reversed/zero-length intervals,
nonfinite offsets and malformed negatives do **not** select horizon 8. The later
wrapper must record these as policy/interface query failures under the parent's
failure rules, without an infrastructure retry. Numerical/model exceptions and
nonfinite active tensors likewise fail; they must not be caught as normal
fallbacks. The event decoder's intentional masked negative-infinity logits remain
permitted under its existing rules. Checkpoint/identity mismatch blocks loading.

## Confidence and temporal behavior

There is **no confidence threshold, entropy gate, class-specific rule, hysteresis,
cooldown or online adaptation**. Log categorical probabilities using the existing
decoder for later analysis, but do not pass them into the mapper. The Week 10
diagnostic contains high-confidence errors; no validated confidence cutoff is
available, and adding one would create a second selection mechanism.

Offsets always refer to the current action-query observation. Recompute after
each executed eight-action chunk. Do not subtract inference latency, shift the
offset by one frame, or carry a countdown across queries. The simulator's existing
step-based timing remains in force; this is not a real-time physical deployment.
The eight-action proposal, eight-action execution and 20 Hz control rate remain
unchanged. For horizons 2 or 4, the future core's existing mask uses only that
proposal prefix; actions after the horizon are masked internally. For 16 or 32,
the inherited normalized zeros after proposal index 7 remain. These masked/padded
inputs are not extra planned actions or a claim of physically stationary motion.

The action coupling was trained at horizon 8, so changing its future input is a
deployment distribution shift even when the predictor supports the selected
horizon. Step 3/4 must verify the wiring and evaluate its consequences; this rule
does not establish robot benefit or accuracy of long-horizon predictions.

## Three-arm application and logging

| Arm | Behavior |
| --- | --- |
| `M4-fixed` | No event call; execute original horizon-8 path. |
| `M4-EG-fixed8` | Decode and validate the event prediction, log this rule's proposed horizon, but force the executed future horizon to 8. |
| `M4-EG-adaptive` | Decode and validate identically, and use this rule's proposed horizon in the future core. |

The fixed8 control does not hide event-branch failures. Mapping is negligible
CPU work and runs in both event arms; both log proposed and actually used future
horizons separately. It never changes the number of executed actions. There is
one future prediction per query, including valid `none` cases.

The pure helper returns rule ID, event type, raw start/end, selected horizon,
signed quantization displacement (`selected - raw_start`, null for none), and
fallback reason (null for positives). The wrapper additionally records arm,
query/anchor identity, actually used horizon, categorical probabilities, branch
and contract identities, and component/end-to-end timing. Preserve the original
prediction separately. A positive start mapping to 8 is not a fallback.

## Verification and recorded-data diagnostic

The [pure helper](../src/eventground_wam/eventground_horizon.py) does not load a
model or implement the robot wrapper. Checks exhaust all 528 valid intervals
for each of three positive classes (1,584 cases), all 32 starts, boundary/tie
behavior, monotonicity, fixed points, invalid inputs and absence of mutation.
The child verifier rechecks the original Step 1 package, the parent digest,
the new code/specification hashes and the complete mapping table.

Only after the child seal, apply it once to the existing 396-prompt dev file.
Report raw/mapped horizon histograms, valid-none counts, out-of-set raw starts,
quantization displacement, and type coverage. Conditional onset errors compare
raw/mapped starts with the same automatic target starts on the same correctly
typed-positive denominator, including lead-bin breakdowns. None fallback has
no onset target and is excluded from those timing errors. Retain errors of event
type in the coverage denominator; a timing improvement does not repair them.

The diagnostic is descriptive reuse of balanced weak-label demonstration data,
not deployed-policy calibration, independent QA, or evidence of robot improvement.
Its records and limitations are packaged under
`docs/evaluation/week11-horizon-v1`. Mapping code is ready for integration;
the policy wrapper, hardware qualification and robot rollouts remain Step 3 onward.
