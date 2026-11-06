# Week 12 website blog plan

**November 4, 2026: the [draft](blog/eventground-wam/AUTHORING.md) incorporates the first author review.** Publish one
clear technical story on [peimengsui.github.io](https://peimengsui.github.io/),
using the completed experiments and retained media. The
[publication addendum](WEEK12_PUBLICATION_ADDENDUM_V2.md) replaces the formal
report requirement. This document is the writing and production brief.

## Reader and central story

Write for an ML engineer who understands training and evaluation but may be new
to robot policies. Explain a vision-language-action policy as a model that
turns camera observations and an instruction into robot actions. Establish
that the experiments use simulated manipulation tasks on LIBERO-Plus.

**Working title: Building a Robot Policy That Looks Ahead**

The central story is building pixel and latent future predictors, connecting
them to a policy, and examining the observed gains on perturbed tasks. Lead with
the pixel-coupled policy's 76.8% versus 74.1% baseline result, then distinguish
that comparison from coupling's uncertain incremental value over auxiliary
training. Preserve the clean results, all five arms and the scientific limits.
Event-based routing is a secondary extension: summarize its failed improvement
and possible next experiments, with methods and examples available on expansion.
The article need not use the internal project name. These editorial changes do
not change the results or establish an overall coupling benefit.

The structure and word budgets below record the original planning allocation.
The author's November 4 revision takes precedence for emphasis and ordering:
future prediction and its follow-up questions come first; the shorter extension
summary and expandable detail follow. The evidence and media sources remain
the same.

Use first-person technical prose consistent with the site's project posts.
Explain decisions through concrete examples. Avoid invented personal anecdotes,
a week-by-week diary, a literature survey or a catalogue of experiment IDs.
Spell out unfamiliar terms once; use descriptive model names in the article
and keep B0/M1–M4 mappings in the linked evidence.

Target **1,800–2,200 words of body text**, with a ceiling of 2,500 if needed for
clarity. Keep captions short and references separate. State the outcome within
the opening 150 words. Every visual must answer a reader's question.

## Article structure

| Section | Words | Reader takeaway and evidence | Exhibit |
| --- | ---: | --- | --- |
| Opening | 150 | Why looking ahead seemed useful; what was built; the measured outcome in plain language | A small poster linking to the paired demo, if it improves the opening |
| How the policy uses a predicted future | 350 | Observation and instruction feed an action policy; future prediction can be a training objective or an input to action refinement; an event predictor selects the future horizon while execution remains eight actions per query | Simplified architecture |
| Did future prediction improve control | 300 | Explain the five-arm comparison, pixel versus latent targets, and training-only versus deployed futures; coupling intervals include zero | Compact five-arm result figure |
| What happened when the horizon became adaptive | 400 | Briefly explain weak-label event training and the lack of established additional RLVR benefit; show all three deployment controls and the targeted regression with the clean result beside it | Adaptive result figure |
| What the rollouts reveal | 300 | Walk through one adaptive gain and one loss; connect selected examples to the aggregate evidence; include one diagnostic lesson only if its existing inspection supports it | Two short paired clips or annotated still sequences |
| What I learned and would change | 300 | Discuss the need to test downstream behavior, preserve a matched fixed control, and validate event predictions on deployed trajectories; distinguish observations from hypotheses for future work | No additional figure by default |
| Limits and resources | 200 | Summarize single seed, reused panels, input differences and predictor/label qualifications; link code, full results and reproduction instructions | Three descriptive resource links |
| Total | 2,000 | Captions and resource references are additional | Four core exhibit groups |

## Numerical anchors

Use the [verified results](evaluation/week12-results-v1/RESULTS.md) and
[full tables](evaluation/week12-results-v1/tables/TABLES.md) as the numerical
authority. Preserve denominators and paired intervals in each result caption.

| Claim | Numbers and qualification to preserve |
| --- | --- |
| Coupling has no established overall benefit | Perturbed: +1.35 percentage points, 95% interval [−0.45, +3.15]; clean: −0.15 [−1.85, +1.43]. The study has 1,000 perturbed and 2,000 clean episodes per arm. Affected comparisons remain descriptive because of input differences. |
| Adaptive routing regresses on the targeted panel | Fixed: 311/467; adaptive: 279/467. Difference: −6.85 points [−10.28, −3.43]. There are 19 adaptive-only successes and 51 fixed-only successes. |
| Clean performance also has a lower point estimate | Fixed: 187/200; adaptive: 177/200. Difference: −5.00 points [−10.00, 0.00]; this interval reaches zero. |
| The event-running fixed control matters | Both fixed controls match all 667 binary outcomes and 16,500 identical-input action queries. Full trajectories can differ with rendering. |
| Event forecasting remains exploratory | Additional RLVR benefit is unestablished. SFT continuation report IoU changes by +1.15 points [−1.17, +3.42] on 384 prompts from 80 sources; automatic-label agreement does not establish on-policy accuracy. Include this number only if needed, with full details linked. |

Keep the two robot studies' panels separate. The five-arm comparison does not
declare M4 an overall winner. Evaluation intervals do not measure training-seed
variation. Cover pixel versus latent prediction and event post-training briefly
so the shorter article still answers all four original questions.

## Visual and demo plan

| Asset | Source already available | Planned treatment |
| --- | --- | --- |
| Architecture | [Figure 1](evaluation/week12-results-v1/figures/fig01-architecture.svg) | Re-layout the training and action paths for a narrow page; distinguish the separate studies and show that horizon selection changes future conditioning, not execution length. |
| Future-control comparison | [Figure 2](evaluation/week12-results-v1/figures/fig02-five-arm-comparison.svg) | Re-render from verified tables with readable names, all five arms, separate clean/perturbed panels and the coupling interval. Put the full factorial grid in the linked results. |
| Adaptive-control comparison | [Figure 3](evaluation/week12-results-v1/figures/fig03-adaptive-integration.svg) | Retain three controls, counts and paired differences. Use a stacked mobile layout. Move the latency panel to supporting results and explain its cost briefly in prose if useful. |
| Adaptive gain | [Targeted case 0059](evaluation/week12-results-v1/examples/fig04a-targeted-gain.png) | Bowl from stove to plate under a camera-view perturbation. Adaptive succeeds; both fixed controls fail. Export a paired clip if the retained recording is available; otherwise use labeled stills. |
| Adaptive loss | [Targeted case 0001](evaluation/week12-results-v1/examples/fig04b-targeted-loss.png) | Bowl from between plate and ramekin to plate under an initial-state perturbation. Original and event-running fixed controls succeed; adaptive fails. Present with the gain at equal prominence. |

These cases are the existing first-manifest-ordinal examples in their paired
outcome classes. Preserve that selection rule. The two examples illustrate
different cases, not the frequency of gains or losses. Display the aggregate
19 gains / 51 losses near the pair to prevent a balanced display implying a
balanced result.

The dense nine-panel source images are useful evidence references. For the
article, prepare readable derivatives with the instruction, descriptive arm
names, camera labels and recorded step numbers. Keep the event-running fixed
control in the linked full panels if the main clip shows only original fixed
versus adaptive. Existing still columns are each arm's first, midpoint and last
pre-action observation; they are **not synchronized time points**. Their final
frame is not necessarily a terminal success image. Outcomes come from metadata.

### Video availability and export

Local inspection found the final chart and still-image package, plus 112 older
Week 4 MP4 previews under `outputs/`. It found no Week 9–11 MP4s there and no
local Week 11 raw recordings in the inspected `outputs/` and formal/closeout
packages. The Week 4 clips are not substitutes for the current comparison.

The [example manifest](evaluation/week12-results-v1/examples.json) identifies
the retained Week 11 HDF5 paths for both selected cases and all three arms.
Their current remote availability has not been checked during planning.
Production should locate and verify those existing recordings before promising
motion video. Do not rerun an episode just to obtain a cleaner demo.

Aim for **two 15–25-second silent paired clips**, each covering one case, if
the recording cadence supports useful motion. Timing takes priority over the
duration target. Align arms by recorded simulation step when available, retain
frame gaps, use one clearly stated playback scale, and mark an ended trajectory
instead of stretching it to fit the other arm. If only sparse observations were
recorded, label the result a recorded-observation sequence. Never interpolate
synthetic action frames or describe an unknown speed as real time.

Use native video controls, a meaningful poster, inline playback and a download
fallback. Target small web exports, approximately 5 MB or less per clip when
legibility permits. Avoid autoplay, background music and a mandatory narrated
overview. If recordings are inaccessible, the paired still sequences fully
satisfy the media plan. A lightweight annotated event timeline is optional only
if it clarifies horizon selection better than the architecture already does.

## Supporting material and site fit

Link readers to [full results](evaluation/week12-results-v1/RESULTS.md),
[qualitative provenance](evaluation/week12-results-v1/QUALITATIVE.md) and the
[reproduction guide](WEEK12_REPRODUCIBILITY.md). Retain detailed diagnostics,
all intervals, latency protocols and failed qualification gates there. The
main article must still state qualifications that change its conclusions.
Resolve public URLs during site preparation; local paths are not publication
links, and the large analysis archive need not load with the page.

The site's [_config.yml](https://github.com/peimengsui/peimengsui.github.io/blob/main/_config.yml)
uses Jekyll/Minima. The
[Adaptive Coding Soundtrack post](https://peimengsui.github.io/2026/09/19/Adaptive-Coding-Soundtrack.html)
provides an existing example of an architecture image and an inline demo.
Follow its `layout: post`, title/date/tags/description metadata and existing
image/video asset folders. Choose the actual publication date when integrating
the post. Use the existing site design; no site rebuild is needed.
The user confirmed the local checkout at `/Users/peimengsui/peimengsui.github.io`;
the article is staged in `_drafts` there, with no publication date assigned.

## Production sequence and completion criteria

1. **Write the article.** Draft around the outline and verified anchors. Use
   descriptive exhibit references while media is being adapted. Cut repeated
   background and operational history; check every numerical claim against its
   source. Verify external method references against primary sources.
2. **Prepare the media.** Re-render the three figures for the website; export
   the two retained cases if recordings are available, otherwise prepare the
   still sequences. Record source paths/hashes, case IDs, transformations,
   playback scale, captions and alt text for each delivered asset.
3. **Assemble a site preview.** Prepare the dated post and web assets in the
   site's existing format. Check at phone and desktop widths; ensure labels
   remain readable without horizontal scrolling and media has useful fallbacks.
4. **Finish publication preparation.** Read the article for clarity, verify
   claims and public resource links, check video playback and image sizes, and
   remove drafting placeholders. A complete handoff consists of the post,
   working assets, concise supporting links and a reviewed site preview.

The planning task is complete, and the first draft is available. Author review,
final media adaptation, the actual site build and publication remain subsequent
work; the [status ledger](WEEK12_STATUS.md) records current progress. No new
experiments are needed to complete the article.
