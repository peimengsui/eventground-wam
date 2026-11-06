# Proposed Week 10 source split revision — not active

The production source map now contains **300 exact full-action bindings across
all 40 tasks**: 200 initial review recordings (five per task) and 100 reserves.
[All saved comparisons were independently recomputed](evaluation/week10-production-plan-v2/mapping-audit.json).

The active v2 experiment still requires: “A production manifest needs verified
source-level partition closure before capture.” See
[Source protection](WEEK10_EXPERIMENT_V2.md#source-protection).
The public data does not supply the historical episode-to-original-demo map.
Exact full-action matching identifies known aliases, but cannot rule out a
cropped or resampled copy of the same original demonstration in an old protected
split. For example, removing five initial actions changes the length and whole
sequence hash without changing which original demonstration it came from.
This is an unresolved provenance limitation, not evidence that leakage occurred.

## Concrete proposed change

The [reviewable proposal and exact source IDs](evaluation/week10-production-plan-v2/source-split-v3-proposal.json)
use a new split for the **new canonical event dataset**:

| Partition | Original source demonstrations | Selection |
| --- | ---: | --- |
| Train/review | 300 | The verified, prospectively selected 200 + 100 queue |
| Event-dev | 103 | New metadata-selected source IDs across all 40 tasks |
| Event-report | 77 | Different new metadata-selected source IDs across all 40 tasks |

Every new recording or variant of a source stays in its assigned partition.
The proposed event-dev/report IDs exclude the training queue and all original
sources whose payloads were inspected during this preparation or the earlier
source catalog. They were selected from existing original-demo metadata, with
fixed SHA256 ordering and seed 10. No proposed held-out payload was read.
All other original sources remain unassigned and locked.

These are **new event-dev/report memberships**, not a relabeling of the old
103/77 groups. This proposal replaces the unproved historical-source-closure
requirement with explicit source separation for the new event-supervision task.
It does not claim that the inherited M4 action learner has never seen transformed
aliases of these sources. Conclusions must remain conditional on that fixed M4
parent and cannot inherit a clean historical-source generalization guarantee.
Later robotics evaluation after adaptation must also carry this limitation or
use an independently justified evaluation design.

The M4 endpoint, event definitions, label-quality gates, causal inputs, prompt
budgets, source/event caps, reward rules and optimization authorization gates
remain unchanged. Earlier Week 7–9 models, datasets, split files and outcomes
remain untouched.

## What approval would and would not activate

Approval would authorize a new v3 source-partition contract and completion of the
production manifest under these exact proposed source memberships. It would
not itself launch recording or training. The train-side recording proposal has
200 initial cases and 100 coverage reserves, at most two GPU-hours and 100 GiB
outputs; exact source input hashes, recorder/environment identity and capture
admission must be sealed first. Event-dev/report capture requires its own bounded
child plan. Independent full-episode label QA remains mandatory.

If the original v2 inheritance requirement is retained, production stays blocked
until authoritative conversion/source provenance resolves the relevant historical
aliases. More full-action comparisons alone cannot establish that stronger claim.

**Status: proposed, not approved, not active.** Active v1/v2 contracts are intact.
