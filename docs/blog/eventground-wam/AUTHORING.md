# EventGround-WAM blog draft handoff

**November 6, 2026: editorial revision after author review.** The target site checkout is
`/Users/peimengsui/peimengsui.github.io`, as specified by the user. The staged
[article](site/_drafts/building-a-robot-policy-that-looks-ahead.md) uses that
site's Jekyll front matter and is intended for its `_drafts` directory. It has
no publication date and is not a published post.
The draft, post layout, and seventeen supporting files have been copied into that checkout
and verified byte-for-byte against this staging directory. The unrelated
untracked `assets/.DS_Store` was left untouched.

The draft embeds model-comparison and event-timing diagrams, two results figures,
two event-extension panels, a saved pixel-prediction panel, and two main-study
comparison videos with posters. Both diagrams have SVG and PNG exports.
Three supporting CSVs and a table guide remain in the authoring package. Result
figures use frozen plotting data with revised titles; qualitative panels are
unchanged. CSVs preserve values and row order with publication-facing labels.
[Asset hashes](asset-manifest.json) and the [source-label map](source-label-map.json)
preserve provenance. No model calls, rollouts, new statistical analyses or
synthesized robot images were needed.

## Editorial revision

The revised opening leads with the pixel-coupled policy's observed 76.8% versus
74.1% baseline result. The main comparison distinguishes that gain from the
smaller, uncertain increment over auxiliary training and retains all five arms,
the clean result, intervals and input/seed qualifications. A positive point
estimate is described as encouraging observed evidence, without suggesting that
more data must produce a significant result.

Event routing is now a separate extension with a short visible summary of both
performance losses and possible future investigations. Its methods, controls,
figure and qualitative examples are retained in an expandable section. The
project name is removed from the article prose and description; internal asset
paths and research provenance keep their existing names. No result or scientific
conclusion has changed. The [first draft](revisions/2026-11-03-first-draft.md)
and its [verification](revisions/2026-11-03-verification.json) are preserved.

## Model diagram and publication labels

The architecture now compares action-only, auxiliary, and coupled paths in
three rows, explicitly identifying the pixel and latent arms. Dashed elements
denote the training-only future branch. The coupled row shows a proposal,
future tokens and one refinement through the same action head. Event routing
stays in its separate extension section. The caption and alt text explain the
model differences instead of the project timeline.

Week-number labels have been removed from all publication content: the article,
figures, table guide and downloaded tables. Original research-package paths and
historical authoring records remain intact. The
[publication renderer](source/render_blog_assets.py) regenerates the three
figures and table labels, checking every non-label table cell against its frozen
source. The results-figure headlines now describe their comparisons without
internal scheduling or gate terminology.

## Results-section revision

The five-policy section now explains the confidence-interval calculation once
in the figure caption, then reports estimates and ranges without repeating
whether each range spans zero. The predictor-qualification paragraph is removed
from the results narrative. A short limitations section explains the one-seed,
development-reuse, RGB-input and multiple-comparison boundaries in plain language.

The lower figure labels and prose distinguish the average coupled-minus-auxiliary
gain from the interaction (latent coupling gain minus pixel coupling gain).
The worked example uses the frozen perturbed estimates: pixel +0.70, latent
+2.00, average +1.35, interaction +1.30 percentage points. All five model
rates, plotted contrasts, uncertainty ranges and downloaded data are retained.

## Results presentation simplification

At the author's request, the interaction comparison is removed from the article,
chart, caption and alt text. The chart retains all five policy rates and the
two main effects: prediction target and coupling. Displayed estimates and
intervals are unchanged; complete audited statistical tables remain available.

## Preview image caching

The browser retained an older results PNG under its unchanged URL even after
the article refreshed. Figure references now include a content-hash version in
both embedded images and full-size links. The publication renderer updates
these versions whenever a figure changes. The corrected chart was verified in
the actual article viewport after refreshing, with two lower-panel rows.

## Closing section removed

November 5: removed the complete “Limits and resources” section, including its
heading, limitations, closing reflection and resource links, at the author's
request. Supporting files remain in the authoring package. No other article
section changed.

## Five-policy metrics table

November 5: added a compact table before the main comparison figure. It reports
perturbed and clean success rates with successful-run counts for all five
policies, and bolds the highest observed rate in each panel. All ten rate/count
cells match the staged `success_rates.csv`; underlying data and figures are
unchanged. The refreshed local preview was visually checked for readable
columns and accurate emphasis. The website draft is synchronized.

## Next-tests section removed

November 5: removed the complete “What I would test next” section, including
all three proposed tests and its closing paragraph, at the author’s request.
The main results now lead directly into the exploratory extension. The website
draft is synchronized, and the removal was verified in the refreshed browser.

## Event-timing explanation rewritten

November 5: replaced the vague extension summary with “Can event timing choose a
better prediction horizon?” The visible section now motivates timing from the
fixed-horizon five-policy study, explains the event-to-horizon rule with a new
schematic, names all three deployment conditions, and reports their success
rates and counts. It distinguishes control preservation, individual rescues,
aggregate regression, inference cost and the lack of an established benefit
from reward-based predictor training. Only the two qualitative examples remain
collapsed; the previous broad follow-up discussion was condensed into the
interpretation of this experiment. The removed next-tests and resources sections
remain absent. The introduction's obsolete next-tests reference was adjusted.

All six deployment rate/count cells and three reported effect intervals were
checked against the staged CSVs. Event forecast scores and latency values were
checked against the retained results. The five-policy comparison and all existing
figure/data bytes remain unchanged. The new diagram follows the frozen mapping
contract and explicitly labels its 14-frame example as illustrative. Website draft
and diagram exports are synchronized; the actual preview diagram and results
table were visually inspected. No publication action was taken.

## Main-study demos, prediction example, and inference cost

November 5: implemented the three approved additions. The visible comparison
clip shows B0/M1 failure and M3 success on perturbed case 0035. The expandable
opposite-outcome clip shows B0/M1 success and M3 failure on case 0034. Each is the
first recorded case in its three-arm outcome class with exact scene, settling
prefix, and first-query input agreement. All six HDF5 hashes match the original
panel reports; downloaded first-frame images also agree exactly across policies.

The 11-second clips retain all 220 comparison control steps at 20 Hz, omit the
initial ten settling transitions, and explicitly hold the last pre-action frame
when a recording ends. Captions distinguish simulation playback from inference
wall time. Success comes from the retained episode metadata. Both MP4s decoded
completely and played to completion in the browser. Their combined size is under
2 MB. Original raw HDF5 inputs stay under ignored `outputs/blog-media-v1` and are
not copied into the website.

The pixel example is the first task of the preselected validation panel, at its
midpoint anchor (115), horizon eight, both cameras. Current and target tiles come
from the checksum-verified original sheet; prediction tiles match the saved
float arrays after the original quantization. The illustration uses enlargement
and labels only. The caption identifies the pretrained module and demonstration
action conditioning, keeping this reconstruction example distinct from M3 policy
rollouts. No new inference, simulation, training, or statistical estimates were
produced.

The five-policy table now reports pooled median / p95 latency from the perturbed
panel on the same RTX 5090, retaining first and failed queries. All ten latency
values and the existing ten success cells were checked. The reported M3/B0
median difference is 3.4 ms, approximately 8%. The title and closing structure
remain unchanged; proposed polish items four and five are not part of this edit.

[Media provenance](media-provenance.json) records selection, source hashes and
exports. [The media renderer](source/render_main_study_media.py) uses Pillow,
numpy, HDF5 2.0-compatible h5py, and imageio-ffmpeg. The latter two were installed
only in `/private/tmp/robot-blog-video-tools`; the original environments were
unchanged. Rebuild with that directory on PYTHONPATH and the retained inputs in
`outputs/blog-media-v1`. The general publication renderer preserves the separate
media manifest and versions both image and video URLs. Website files are synced.

## Title, subtitle, and closing takeaway — November 6, 2026

The title is now “Can Predicted Futures Improve Robot Policies?” with the visible
subtitle “Comparing pixel prediction, latent prediction, and action refinement
across five policies.” The staged post layout adds an escaped subtitle only when
that front-matter field is present. The four-sentence takeaway closes on the
observed pixel-coupled result and its measured inference cost, while distinguishing
the auxiliary-training result and separate event-timing extension. It appears
outside the collapsed qualitative examples. Existing body text, statistics and
media are unchanged. The preview title, subtitle and takeaway were visually
checked; all nineteen installed files match staging. No publication was performed.
The implementation-and-code section remains a proposal for a separate edit.

## Inference-cost explanation — November 6, 2026

Added “What does looking ahead cost?” after the main quantitative results and
before the rollout examples. It explains the auxiliary, pixel-coupled and
latent-coupled latency tradeoffs, distinguishes query generation from execution
and task completion, and identifies the event predictor's larger cost as a
separate evaluation. The existing pixel-latency sentence moved into this section.
All overhead percentages were checked against the existing table. The subsection
was visually checked in the refreshed preview; the website draft matches staging.
Existing metrics, media and the remaining prose are unchanged.

## Claim sources

| Article content | Authoritative local source |
| --- | --- |
| Backbone, action exposure and independent policy initialization | [Shared training protocol](../../PHASE2_TRAINING_PROTOCOL.md) |
| Pixel inputs and outputs | [Pixel implementation](../../WEEK5_PIXEL_PREDICTOR.md) |
| Latent inputs and outputs | [Latent implementation](../../WEEK6_JEPA_PREDICTOR.md) |
| Proposal, internal future tokens and one action refinement | [Coupling interface](../../WEEK8_COUPLING_INTERFACE.md) |
| Predictor acceptance history | [Week 12 scope](../../WEEK12_SCOPE.md) and [latent closeout](../../WEEK6_CLOSEOUT.md) |
| Five-arm effects, counts and all numerical qualifications | [Verified results](../../evaluation/week12-results-v1/RESULTS.md) |
| Event labels, matched continuation and RLVR conclusion | [Event closeout](../../WEEK10_CLOSEOUT.md) |
| Horizon mapping, padding and deployment input shift | [Mapping contract](../../WEEK11_HORIZON_MAPPING.md) |
| Three deployment arms, success counts and control checks | [Verified results](../../evaluation/week12-results-v1/RESULTS.md) |
| Gain/loss outcomes, case selection and timestamps | [Example manifest](../../evaluation/week12-results-v1/examples.json) and [captions](../../evaluation/week12-results-v1/CAPTIONS.md) |
| Later retention, recovery and wrong-drawer observations | [Saved-state inspection](../../evaluation/week11-h8-validation-v1/INSPECTION.md) |
| Follow-up disposition | [Week 11 closeout](../../WEEK11_CLOSEOUT.md) |

External references were checked against the official LIBERO-Plus repository
and the OFT paper. No benchmark-leaderboard numbers or claims of general method
superiority are imported into the article.

## Remaining publication work

- Adapt the two dense results figures and the paired examples for comfortable
  reading at phone width. The new architecture uses scalable SVG; full-size
  images are linked from the draft in the meantime.
- Locate the retained Week 11 recordings if motion clips are wanted. The stills
  already support the prose; do not substitute older Week 4 previews or create
  new rollouts. Preserve actual timing and observed outcomes in any export.
- Prepare the code publication snapshot before adding a source link. The November 6
  [GitHub audit](code-publication-audit.json), using authenticated GitHub CLI access,
  confirms that `peimengsui/eventground-wam` exists and is private. Its default branch
  remains at `eec5da0`; the current local branch is 35 commits ahead, and the later
  predictors, auxiliary/coupled policies, event extension, evaluation and reporting
  implementation are untracked locally and absent from all four GitHub branches.
  Commit and verify a curated source/config/test/environment snapshot, provide
  documented reproduction inputs, and resolve public access before publishing a
  code link. No staging, commit, push or visibility change was performed by the audit.
- Review the prose with the author, complete the actual Jekyll build and final
  mobile media checks, then assign a publication date when moving the article
  into `_posts`. No commit, push or deployment is part of this drafting task.

The [local preview](site/preview.html) is a Markdown reading preview, not a
claim that the Jekyll build has passed. It is excluded from the site copy.
The discussion of blog comments did not authorize adding a comment service;
the site template is unchanged.
