# Week 10 prompt coverage diagnosis

The initial 200 recordings are complete, but **the frozen next-event task does
not provide enough eligible grasp prompts to train the planned balanced model**.
This is a structural interaction among the predeclared onset rules, ambiguity
exclusions and sampling caps. It is not evidence that the robot only grasped
29 times, and no quality claim follows from these unreviewed candidate counts.

This read-only diagnosis retains the v3 study, labeler and all candidate/recording
files. It does not admit labels, change rules, capture reserves or train a model.

## Immutable evidence

| Artifact | Digest |
| --- | --- |
| [Active v3 study](../configs/training/week10_eventground_experiment_v3.json) | `080cca5bd1068992556d8fba6c60215d5136f69d8a6106aee47692d209c704fa` |
| [Frozen physical labeler](evaluation/week10-instrumented-pilot-v1/labeler-contract.json) | `a9702609f64b4a424186b57b15034f7dd70097d4f1389714a7a058ce6f210065` |
| [Frozen labeler implementation interpretation](evaluation/week10-production-labels-v1/implementation-contract.json) | `35adba44a3f9d586acbc060215b2324414ca41e82fb2b57d759c1dad0dcbc625` |
| [Coverage report](evaluation/week10-production-labels-v1/run/coverage.json) | `8b7b6d56567045b8ada2f61a83eb8d5889be4c2d4bf944a084b2526d1f2afc8d` |
| [Diagnostic prompt manifest](evaluation/week10-production-labels-v1/run/diagnostic-manifest.json) | `4aeaad05315036315c580c96093b20c4e8574c0768bca5f8a2ab874bb67f0838` |

The executable source inspected was
[instrumented_labeling.py](../src/eventground_wam/instrumented_labeling.py),
SHA-256 `589bae0820fa463e99de38b0f1715f39c64776017a1aeca677736aa703a37867`,
matching the frozen implementation. The manifest uses the unchanged candidate
previews and independently reports a class-specific cap upper bound. Actual
selected counts equal that upper bound for each positive class, so changing the
round-robin ordering cannot repair these shortfalls.

| Class | Extracted candidate intervals | Eligible anchors | Distinct events with an eligible anchor | After four-anchor event cap | Selected / required |
| --- | ---: | ---: | ---: | ---: | ---: |
| Contact | 374 | 1,090 | 99 | 327 | 327 / 500 |
| Grasp | 278 | 32 | 11 | 29 | 29 / 500 |
| Gripper close | 242 | 1,419 | 132 | 476 | 476 / 500 |
| None | — | 6,496 | — | — | 500 / 500 |

The source cap of 16 does not further reduce any positive class's independent
upper bound here. The grasp previews have a maximum lead time of only six
frames: selected grasp prompts cover 1–4 and 5–8, with none in the mandatory
9–16 or 17–32 bins.

## Three distinct effects

**1. Earliest-onset selection plus ongoing-event exclusion hides most grasps.**
The frozen task asks for the earliest next onset of any admitted event type,
and disallows anchors inside any ongoing admitted event. Grasp requires finger
contact, so its onset falls within a supporting contact interval. Of the 278
extracted grasp intervals, **241 start after contact already began**. Before
that contact starts, contact is the earlier target; between contact and grasp,
the anchor lies inside an ongoing event and is excluded. All 241 therefore have
zero eligible grasp anchors. Only 37 grasps share contact's exact onset and can
potentially win the frozen `grasp > contact > gripper_close` tie.

Example: [canonical-0000-8f7a627e893c events](evaluation/week10-production-labels-v1/run/canonical-0000-8f7a627e893c/event-candidates.json)
contain contact with `moka_pot_1` at `[182,242)`, and grasps at `[189,198)` and
`[206,238)`. Anchors 188 and 205 are excluded as `inside_ongoing_event` in the
[preview ledger](evaluation/week10-production-labels-v1/run/canonical-0000-8f7a627e893c/prompt-previews.json).
Before contact onset, an earlier contact/close event wins, or another frozen
exclusion applies. Neither grasp can become the earliest eligible next event.

**2. Unresolved confirmation windows remove additional potential anchors.**
Failed five-frame grasp confirmations remain ambiguous even if a later window
confirms a grasp. Any unresolved candidate intersecting the full 32-frame
future window excludes that anchor. The implementation contract explicitly
preserves those failed windows; they are not silently deleted as false events.
Other preceding events and incomplete tails can also exclude anchors. Of the
37 same-onset grasps, only 11 retain any eligible anchor after all rules.

Example: [canonical-0006-9559f4eaf70c events](evaluation/week10-production-labels-v1/run/canonical-0006-9559f4eaf70c/event-candidates.json)
contain close `[79,85)`, then grasp and contact both starting at 91. Anchors
85–90 might otherwise anticipate that tied grasp. However, there is a one-frame
contact at 88 and a failed grasp confirmation spanning `[88,93)`, with deficient
contact persistence, bilateral support and motion. The
[preview ledger](evaluation/week10-production-labels-v1/run/canonical-0006-9559f4eaf70c/prompt-previews.json)
excludes those anchors as `unresolved_candidate_in_window`. Independent QA can
assess this evidence, but declaring the earlier failed windows resolved would
change the frozen exclusion interpretation if used to regenerate training data.

**3. The four-prompts-per-event cap reduces the surviving 32 anchors to 29.**
This cap prevents multiple nearby anchors from multiplying one physical event's
supervision. It accounts for all three additional losses: one event has five
eligible anchors and another has six. No additional sampler inefficiency is
needed to explain the final grasp count.

Example: [canonical-0005-b7d58405c206 events](evaluation/week10-production-labels-v1/run/canonical-0005-b7d58405c206/event-candidates.json)
contain close `[78,84)` and grasp/contact both starting at 89. Its
[preview ledger](evaluation/week10-production-labels-v1/run/canonical-0005-b7d58405c206/prompt-previews.json)
has eligible grasp anchors 84–88, with lead times 5,4,3,2,1. The diagnostic
manifest selects anchors 84–87; anchor 88 cannot be added without exceeding the
four-per-event cap. Earlier anchors either predict close or fall inside it.

These examples explain the observed coverage using the implemented frozen
semantics. They do not replace independent review of whether the physical
contact/grasp candidates themselves are correct. No extraction or sampling bug
was found in this bounded read-only analysis.

## Next decision

Training remains blocked by both independent label QA and prompt coverage.
Human QA can establish precision, recall and boundary accuracy, but **passing QA
alone cannot turn these 29 allowed grasp prompts into 500**. Even discarding all
ambiguity exclusions would leave at most 37 same-onset grasp events in this
fixed candidate ledger, or 148 prompts under the four-per-event cap.

Within v3, the remaining permitted data expansion is the **100 preselected
coverage reserves**, in their frozen order and cumulative capture budget. Their
outcomes must all be retained and the same labeler, exclusions and caps applied.
There are **471 additional grasp prompts needed**, requiring at least
`ceil(471/4) = 118` additional eligible physical grasp events, plus the missing
lead-time bins. A recording may contain more than one event, so this is not a
proof that the reserves must fail. Their sufficiency cannot be assumed from
the 278 raw grasp intervals or extrapolated as a guaranteed yield. Contact and
close also need 173 and 24 additional eligible prompts, respectively.

The scientific decision is whether to finish that bounded v3 coverage check
or explicitly review a new task definition before spending the remaining
capture/review effort. A possible separate revision would predict **the next
onset of a requested event type**, permitting a future grasp target while
contact is ongoing. That changes the question, target construction, causal
request input and comparison design; it is not a bug fix and cannot be silently
substituted into v3. Another eligibility revision would likewise require a new
reviewed protocol and preserved v3 findings. No such revision is implemented or
frozen by this note.

Recommended next decision: acknowledge this structural coverage failure and
choose explicitly between the remaining fixed reserve check and a separately
reviewed protocol revision. Continue preparing independent QA materials and
model correctness evidence without claiming a training-ready dataset. Do not
duplicate anchors, weaken physical predicates, drop grasp, relabel contact as
grasp, or launch the balanced experiment with the shortfall.

## Read-only counterfactual check

A subsequent [counts-only analysis](evaluation/week10-production-labels-v1/counterfactual-coverage-analysis.json)
retains every candidate file and persists no alternate prompt manifest. Allowing
anchors during an ongoing event, while keeping every ambiguity exclusion and
earliest-future-onset rule, raises the grasp class's independent capped upper
bound from **29 to 233**. It restores candidate anchors in all four lead bins,
but still cannot supply 500 grasp prompts from the initial 200 recordings.
Contact/close ceilings would be 399/497. This simple eligibility change alone
is therefore insufficient; it must not be advertised as a complete resolution.

Ignoring all ambiguity exclusions as a purely mathematical ceiling gives grasp
45 under the existing ongoing-event exclusion, or 1,006 if both exclusions are
removed. Those counts neither justify removing ambiguity checks nor establish
label quality or a jointly feasible balanced manifest. A new scientific design
must decide how independently reviewed evidence resolves genuinely uncertain
windows, which future event is predicted, and what coverage can be obtained
under retained source/event caps. No alternate rule was activated.
