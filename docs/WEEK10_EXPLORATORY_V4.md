# Week 10 exploratory weak-supervision revision

The user accepted the exploratory path after an assisted review of 20 episodes.
This child preserves the original M4 parent, v3 source partitions, physical
event definitions, conservative prompt eligibility and FP32 numerical recipe.
It changes the evidence claim and accepts the actual, imbalanced prompt count.
The original studies, independent-QA gates and earlier conditional future-onset
proposal remain unchanged. The latter proposal is not activated.

The [sealed protocol](../configs/training/week10_eventground_exploratory_v4.json)
links the immutable review snapshot, inputs and implementation. The saved review
accepts 113 proposed positives (40 contact, 41 grasp, 32 close) in 20 episodes
covering four tasks. There are no saved missed-event checks. Active human review
time is unknown. This supports limited assisted spot-checking; it does not
establish independent precision, recall or validity of unreviewed labels.

All 200 initial train recordings remain the source pool. Exclude incomplete
future windows, anchors inside ongoing events and every window overlapping an
unresolved ambiguity. None means no eligible automatic onset, not a reviewed
negative. No future image, action, contact state or target enters model inputs.

The deterministic rebuild produces 1,332 distinct prompts from 198 recordings
and all 40 tasks: 327 contact, 29 grasp, 476 close and 500 none. All 1,816 ambiguity
flags are retained. Caps remain four anchors per physical event, 16 prompts per
source and 500 per class. Of the selected positive prompts, 129 refer to accepted
event proposals and 703 to pending proposals; these are prompt counts, not counts
of independent reviewed events. None prompts have separate weak-negative
provenance. No class is duplicated to meet a quota.

Grasp coverage is a major limitation: 26 prompts have lead times of 1–4 frames
and three of 5–8 frames; there are none at 9–32 frames. A successful pilot will
not remove this limitation or justify strong grasp forecasting claims.

The authorized next operation is one small qualification on Feifei, frozen
before any optimizer update:

- Deterministically select four prompts per class from the frozen pool, 16 total.
- Initialize from the verified fixed M4 epoch-2 endpoint. Train only the existing
  language LoRA allowlist and new event head; retain frozen vision parameters.
- FP32, logical batch 16, microbatch one; unchanged AdamW, seed 17, learning rates
  (head 3e-4, language LoRA 1e-5), clipping and weight decay.
- Run exactly 64 logical updates. Save after update eight, execute nine, restore
  eight and replay nine. Require bitwise identical parameters, optimizer, RNG and
  counters, plus identical loss and gradient norm. Charge all 65 executions.
- Measure the same training examples at steps 0, 16, 32, 48 and 64. Require final
  loss at most half initial, event agreement at least 14/16, and correct class
  plus both interval boundaries within two frames for at least 12/16. Correct
  none counts as a joint match. These are weak-target fitting checks.
- Require finite updates, unchanged frozen parameters and stored parent,
  memory within 90% of GPU, 50 GiB free disk and outputs within 8 GiB. Limit this
  attempt to 1,800 seconds and include the prior 62.73 diagnostic seconds in the
  original three-hour qualification budget. No automatic retry or tuning.

The fixed update-64 checkpoint is qualification-only. It cannot initialize a
later formal run. Full SFT, RLVR and held-out evaluation have not been launched
or admitted by this revision. If the pilot passes, the next SFT proposal is a
fresh M4 initialization and two passes over these 1,332 unique prompts (168
logical updates at batch 16, retaining each pass's final partial batch). It needs
its own executable weak-label admission and explicit launch approval. RLVR needs
a separate weak-evidence reward contract; the original independent-evidence
reward checker must not be bypassed or presented as satisfied.

Evidence: [dataset](evaluation/week10-exploratory-weak-v4/dataset.json),
[pilot contract](evaluation/week10-exploratory-weak-v4/qualification-contract.json),
[review snapshot evidence](evaluation/week10-assisted-review-v1/review-snapshots/evidence-000177.json).

The [completed qualification](evaluation/week10-exploratory-weak-v4/analysis/RESULTS.md)
passed all predeclared gates at update 64. Loss fell from 2.168946 to 0.022247;
all 16 examples matched the exact weak event/start/end targets. Bitwise recovery
and frozen-parameter/parent integrity passed, within 200.17 seconds and 10.68 GiB
peak reserved memory. These results satisfy the bounded fitting pilot only;
the scope and limitations above remain in force.
