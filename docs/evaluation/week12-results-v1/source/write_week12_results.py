"""Write the results and qualitative handoff from the verified consolidation."""
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def table(headers, rows):
    clean = lambda x: str(x).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |'] +
                     ['| '+' | '.join(clean(x) for x in row)+' |' for row in rows])


def ci(x, low, high):
    def display(value):
        rounded = Decimal(str(round(value, 10))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
        return f'{rounded:+.2f}'
    return f'{display(x)} [{display(low)}, {display(high)}]'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output.resolve()
    d=json.loads((out/'consolidated.json').read_text());w9=d['week9'];w10=d['week10'];w11=d['week11'];f=d['followups']
    manifest=json.loads((out/'source-manifest.json').read_text())
    def link(path,label):
        src=ROOT/path
        manifest['files_sha256'][path]=hashlib.sha256(src.read_bytes()).hexdigest()
        return f'[{label}]({os.path.relpath(src,out)})'
    lines=['# Consolidated results and statistics','',
           '**November 1, 2026: Week 12 Steps 2 and 3 are complete.** This is the numerical and figure handoff for the final report. It consolidates completed experiments under the '+link('docs/WEEK12_SCOPE.md','frozen reporting scope')+'. Project-wide reproduction packaging, report drafting and presentation materials remain later steps.','',
           'The five-arm study does not establish an overall coupling benefit. The weak-label event study does not establish additional RLVR benefit. Formal adaptive routing fails its performance gates, and later exploratory follow-ups establish no replacement for original M4. M4 remains the Week 11 reference, not a declared overall Week 9 winner.','',
           '[Open the figure and example gallery](gallery.html) · [Figure captions and exports](CAPTIONS.md) · [Qualitative evidence](QUALITATIVE.md) · [Full machine-readable results](consolidated.json)','',
           '## Study and model map','',
           table(['Study','Models or conditions','Panel and role','Evidence status'],[
               ['Week 9','B0; M1 pixel auxiliary; M2 JEPA auxiliary; M3 pixel coupled; M4 JEPA coupled','1,000 perturbed cases and 2,000 clean episodes per arm','Frozen planned comparisons; one training seed; affected input comparisons descriptive'],
               ['Week 10 revised task','Starting SFT; primary/calibrated RLVR; preservation hybrid; SFT continuation','396 development prompts / 103 physical sources; 384 report prompts / 80 sources for SFT and continuation','Exploratory automatic weak labels; repeated development and prior report exposure'],
               ['Week 11 formal','Original M4 fixed8; event-running fixed8; adaptive routing','467 targeted and 200 clean episodes per arm','Frozen three-arm deployment evaluation; previously evaluated task pools'],
               ['Week 11 continuation','Original, H8 and mixed endpoints × fixed8/adaptive','27 selected cases per condition; 162 episodes','Exploratory selected-case screen'],
               ['Week 11 adapter','H8 fixed8, motion adapter, H8-information control','First nine-case repeat per condition; 27/81 planned episodes','Partial exploratory screen stopped at latency'],
               ['Week 11 fresh H8','Vanilla M4 and event-free H8 continuation','32 targeted and 32 clean cases per condition; 128 episodes','Exploratory fresh episodes in an existing task pool; targeted variants cover 24 base tasks']]),'',
           'The coupled policies add a future-core path and a 1,187,072-parameter coupling block. M1/M2 omit the future branch at deployment. Matched action-training exposure is not matched total compute or tuning effort. Pixel prediction passed its native gates; JEPA retains its original failed front/h2 guardrail and research-use exception. Native pixel and latent losses cannot directly rank prediction quality. '+link('docs/WEEK6_CLOSEOUT.md','Predictor qualification')+'.','',
           '## Week 9 policy comparison','',
           '### Success rates and paired effects','']
    rows=[]
    for arm in ['B0','M1','M2','M3','M4']:
        row=[arm]
        for panel in ['perturbed','clean']:
            s=w9['panels'][panel]['summaries'][arm];v=w9['panels'][panel]['factorial']['success_rates'][arm]
            row.append(f'{s["successes"]:,}/{s["count"]:,} ({100*s["success_rate"]:.2f}%; 95% CI {100*v["percentile_95"][0]:.2f}–{100*v["percentile_95"][1]:.2f}%)')
        rows.append(row)
    lines += [table(['Arm','Perturbed','Clean'],rows),'',
              'Intervals use 10,000 common PCG64 draws: perturbed variant tasks with seed 7; 40 clean canonical-task clusters, retaining all 50 states, with seed 8. They describe evaluation sampling conditional on one training seed. They are not multiplicity-adjusted confirmatory tests.','']
    rows=[]
    for name,label in [('latent_minus_pixel_main_effect','Latent − pixel'),('coupling_main_effect','Coupling'),('interaction','Interaction')]:
        row=[label]
        for panel in ['perturbed','clean']:
            v=w9['panels'][panel]['factorial']['contrasts_percentage_points'][name];row.append(ci(v['estimate'],*v['percentile_95']))
        rows.append(row)
    lines += [table(['Factorial effect','Perturbed difference, pp [95% CI]','Clean difference, pp [95% CI]'],rows),'',
              'All factorial intervals include zero. Every contributing factorial comparison has RGB mismatches, so these effects remain descriptive. M3−B0 on perturbed cases is the single planned pairwise interval excluding zero (+2.70 pp [0.10, 5.30]); its input mismatches, shared development cases, single seed and multiple comparisons prevent a confirmatory superiority claim. Clean M1−B0 is the only listed pairwise contrast with all initial prefixes exact. No mismatched outcome is removed.','',
              'The 1K panel includes all 210 development tasks; the additional 790 form a prespecified secondary slice. Perturbed/clean success ratios compare different task distributions and are not causal perturbation effects. Full contrasts, paired wins/losses, input qualifications and development slices remain in [consolidated.json](consolidated.json); the tabular contrast export is [contrasts.csv](tables/contrasts.csv). '+link('docs/WEEK9_FINAL_RESULTS.md','Week 9 authority')+'.','',
              '![Week 9 comparison](figures/fig02-five-arm-comparison.png)','',
              '## Week 10 event forecasting','',
              'The revised target is the earliest new onset within 32 frames, including anchors where another event is ongoing. Scores measure agreement with automatic predicates. The earlier 48-prompt task had a different target: its RLVR primary pilot passed its engineering gate, and its scores remain separate in the machine-readable history.','']
    labels={'sft':'Starting SFT','primary':'RLVR primary','calibration':'RLVR calibration','hybrid':'RLVR + supervised preservation','control':'SFT-only continuation'}
    rows=[]
    for key in ['sft','primary','calibration','hybrid','control']:
        m=w10['dev'][key];rows.append([labels[key],f'{100*m["macro_positive_iou"]:.2f}%',f'{m["none_false_positives"]}/{m["none_count"]}',
                                    'Baseline' if key=='sft' else 'Passed existing dev gate' if key=='control' else 'No-go'])
    lines += [table(['Revised development, 396 prompts / 103 sources','Macro positive IoU','None false positives','Original decision'],rows),'']
    pc=w10['dev_comparison_bootstraps']['hybrid_minus_control'];delta=100*(w10['dev']['hybrid']['macro_positive_iou']-w10['dev']['control']['macro_positive_iou'])
    lines += [f'The matched hybrid-minus-SFT-continuation difference is **{ci(delta,*(100*x for x in pc["interval95"]))} pp**. The interval includes zero. Control initialization, prompt order, update count and learning rates were matched; candidate-sampling compute and wall time were not.','',
              'The development-selected SFT continuation was evaluated on the fixed report panel without adding an RLVR report arm.','']
    rows=[]
    for key in ['sft','control']:
        m=w10['report'][key];rows.append([labels[key],f'{100*m["macro_positive_iou"]:.2f}%',f'{m["none_false_positives"]}/{m["none_count"]}',f'{m["matched_positive_start_mae"]:.3f}',f'{100*m["matched_positive_coverage"]:.2f}%'])
    r=w10['report_comparison']
    lines += [table(['Report, 384 prompts / 80 sources','Macro positive IoU','None false positives','Matched onset MAE, frames','Positive match coverage'],rows),'',
              f'Report IoU change is **{ci(100*r["delta_iou"],*(100*x for x in r["bootstrap"]["interval95"]))} pp**, from 10,000 paired physical-source draws with seed 10. All draws are defined. This is a positive point estimate with uncertainty spanning zero. Timing error is conditional on the correctly typed set and must be read with coverage.','',
              'Report labels/counts and baseline results had already been inspected. Independent label QA and deployed-policy event accuracy remain unresolved. These scores do not establish improved robot behavior. '+link('docs/WEEK10_CLOSEOUT.md','Week 10 authority')+'.','',
              '![Event forecasting](figures/figS01-event-forecasting.png)','',
              '## Week 11 formal adaptive integration','']
    rows=[]
    for arm in ['M4-fixed','M4-EG-fixed8','M4-EG-adaptive']:
        row=[arm]
        for panel in ['targeted-467','clean-200']:
            s=w11['summaries'][arm+'/'+panel];row.append(f'{s["successes"]}/{s["count"]} ({100*s["success_rate"]:.2f}%)')
        rows.append(row)
    lines += [table(['Arm','Targeted','Clean'],rows),'']
    rows=[]
    for panel in ['targeted-467','clean-200']:
        for name,s in w11['statistics'][panel].items():
            rows.append([panel,name,ci(s['difference_pp'],*s['paired_95_ci_pp']),f'{s["left_only_success"]} / {s["right_only_success"]}'])
    lines += [table(['Panel','Contrast','Difference, pp [paired 95% CI]','Left-only / right-only successes'],rows),'',
              'Targeted resamples 467 variants; clean resamples 40 canonical-task clusters retaining all five states. Both use 10,000 common draws (seeds 7 and 8). Targeted has an entirely negative interval; clean reaches zero. Both original performance gates fail. Latency, resources and artifact/input checks passed.','',
              'The two fixed controls match all 667 binary outcomes and all 16,500 identical-input action queries. Their full trajectories are not all bitwise equal: small renderer changes precede input/action differences in some pairs. Identical observed control outcomes are not a general equivalence guarantee. Formal panel completion does not create an untouched holdout. '+link('docs/evaluation/week11-closeout-v1/RESULTS.md','Formal authority')+'.','',
              '![Formal integration](figures/fig03-adaptive-integration.png)','',
              '## Exploratory follow-ups and stopping decisions','']
    rows=[]
    for key,label in [('original','Original M4'),('h8-continuation','H8 continuation'),('mixed-continuation','Mixed continuation')]:
        c=f['continuation']['decision']['counts'];rows.append([label,f'{c[key+"/fixed8"]}/27',f'{c[key+"/adaptive"]}/27'])
    lines += [table(['Matched continuation endpoint','Fixed8','Adaptive'],rows),'',
              'Mixed adaptive gains two against H8 adaptive, but mixed fixed8 loses two against H8 fixed8 and fails the frozen preservation rule. Mixed adaptive ties H8 fixed8 at 19/27. The selected screen does not qualify a larger evaluation. All 15 paired contrasts are retained as case-level win/loss lists in the consolidated JSON.','']
    rows=[]
    for panel in ['targeted-467','clean-200']:
        s=f['h8']['panels'][panel]['all_outcomes'];g=f['h8_task_sensitivity']['panels'][panel]
        rows.append([panel,f'{s["vanilla_successes"]}/32',f'{s["h8_successes"]}/32',ci(s['gap_percentage_points'],*s['paired_95_bootstrap_percentage_points']),
                     ci(s['gap_percentage_points'],*g['suite_stratified_task_cluster_bootstrap_95_percentage_points']),g['base_tasks']])
    lines += [table(['Fresh H8 panel','Vanilla','H8','Frozen variant result, pp [95% CI]','Task-cluster sensitivity, pp [95% CI]','Base tasks'],rows),'',
              'The targeted 32 variants cover 24 underlying tasks. The supplementary whole-task bootstrap within suites retains all variants and leaves the negative promotion decision unchanged; its equal-base-task point contrast is −8.13 pp (a different weighting from the −6.25 pp variant contrast). Clean contains 32 base tasks. Both the original and supplementary intervals remain exploratory. The small sample does not establish general inferiority or equivalence.','',
              table(['Workstream','Completed evidence','Disposition'],[
                  ['Closing gate','81 episodes; fixed8 15/27, original adaptive 16/27, gated 16/27','Four rescues offset by four regressions; no promotion'],
                  ['First-chunk interventions','12 episodes across three selected failures','Case-specific causal effects with matched controls; no general routing remedy'],
                  ['Continuation screen','162 episodes; two matched 2,000-update endpoints','Fixed8 preservation gate failed'],
                  ['Placement, phase and channel interventions','12 placement-window plus 24 phase/channel episodes','Narrow matched-case motion/state evidence; retain pre-intervention drift exclusions'],
                  ['Frozen-H8 adapters','Two matched 2,000-update training arms; exact H8 replay preservation','Endpoints retained as research artifacts'],
                  ['Adapter rollout screen','27/81 episodes: H8 6/9, motion 6/9, control 7/9','Stopped at p95 latency; 54 remaining episodes closed without execution'],
                  ['Fresh event-free H8 comparison','128 episodes; all 64 pairs audited','Cost target passed, performance criterion failed; no promotion']]),'',
              'Counts above refer to different, often reused cases and must not be added into an independent trial denominator. The optional four-case fresh H8 recording inspection remains deferred; no new review of those cases is included here. '+link('docs/WEEK11_CLOSEOUT.md','Final Week 11 disposition')+'.','',
              '![Exploratory follow-ups](figures/figS02-exploratory-followups.png)','',
              '## Latency and compute cost','',
              'The complete '+ '[latency table](tables/latency.csv)'+' separates rollout measurements from matched-input timings. Week 9 and formal Week 11 rollout summaries retain every query, including first and failed queries; no warmup or outlier removal is applied. Each query proposes eight actions, so these are not per-action latency or robot-deadline guarantees. Absolute timing differences across separate study executions do not measure a treatment effect.','']
    rows=[]
    with (out/'tables/latency.csv').open() as h:
        for s in csv.DictReader(h):rows.append([s['study'],s['panel'],s['arm'],s['count'],f'{float(s["median_ms"]):.2f}',f'{float(s["p95_ms"]):.2f}'])
    lines += [table(['Study','Panel','Arm','Queries','Median ms','P95 ms'],rows),'',
              'The H8 matched-input benchmark uses 108 retained observations, three rotating-order passes and 324 timed queries per condition after warmup. Event-free H8 p95 is 53.33 ms versus vanilla 54.40 ms; the same H8 with its unused event branch costs 102.67 ms. This supports comparable event-free cost, not a general 2% speedup. The adapter screen used a different frozen 2× ceiling and stopped with motion/control p95 ratios 2.026×/2.023×.','']
    rows=[]
    with (out/'tables/training_cost.csv').open() as h:
        for s in csv.DictReader(h):rows.append([s['arm'],s['hardware'],f'{float(s["active_hours"]):.2f}',s['scope']])
    lines += [table(['Training component','Hardware','Recorded active hours','Accounting scope'],rows),'',
              'Shared pixel and JEPA CPT were each trained once and reused; count each once, not once per downstream arm. Hardware differs across policy runs, so hours cannot be treated as equal-compute comparisons. Full historical tuning time is unavailable and is not zero. The '+link('docs/evaluation/week9-final-v1/cost-ledger.json','original cost ledger')+' retains parameter counts, pilot timing, memory probes and missing-cost qualifications. Later continuation runs each used 2,000 updates and 58,172 presentations; adapter training used the same bounded exposure, not a replacement five-arm training budget.','',
              '## Figures and qualitative selection','',
              'Figure 1 explains the architecture; Figures 2 and 3 summarize the primary robot studies. Figure 4a–d are the unchanged, previously rendered formal gain/loss panels for targeted and clean cases. Supplementary Figures S1 and S2 cover event forecasting and later pilots. Charts are exported as PNG, SVG and PDF. The figure gallery retains all eight formal outcome classes plus three mechanism cases and three weak-label forecast examples.','',
              'Formal examples use the first manifest ordinal in each paired outcome class on each panel. This selection rule is rechecked against the paired matrices. The mechanism and event examples were selected after results and are explicitly diagnostic. Image bytes are copied unchanged and verified against prior media receipts; source IDs, exact recorded steps and captions are in [examples.json](examples.json). No model call, simulator rendering, new rollout or image synthesis is used.','',
              '## Verification and remaining limits','',
              'Effect estimates and interval endpoints in the prose tables use two-decimal half-up rounding after removing floating-point noise at ten decimals. CSV and JSON retain full precision; rounding does not change any decision.','',
              'The [numerical verification record](numerical-verification.json) documents CPU recomputation of all 15,000 Week 9 outcomes, its common-draw rates/contrasts/ratios, all 2,001 formal Week 11 outcomes and paired intervals, revised Week 10 typed IoUs and source-bootstrap comparisons, and the fresh H8 variant and task-cluster sensitivity results. Latency quantiles are checked from retained timing arrays. Continuation pair counts and all 15 win/loss lists are checked against the retained case matrix. Tolerance is 1e-10 absolute and relative.','',
              'This verifies retained local metadata and presentation artifacts. Prior recording/input audits remain authoritative; no fresh remote raw-recording or checkpoint rehash is claimed. The Step 1 freeze and consumed source hashes remain intact. The known prior adapter-report prose/hash discrepancy remains recorded in the Week 11 closeout, and no gate is relabeled.','',
              'Carry forward one training seed, panel/development reuse, historical transformed-source alias exposure, renderer/input qualifications, JEPA’s failed gate/research-use exception, unresolved independent event-label QA and unresolved on-policy forecasting accuracy. Prediction-quality/task-success correlation is not established by the aggregate tables. Full-10K and test remain locked.','']
    (out/'RESULTS.md').write_text('\n'.join(lines))
    examples=json.loads((out/'examples.json').read_text())['examples']
    q=['# Qualitative examples and interpretation','',
       'Fourteen existing examples balance gains and losses across formal outcome classes and exploratory diagnostics. The formal selection is deterministic within each paired class; the mechanism and forecast examples are selected illustrations, not a frequency estimate. [Open the gallery](gallery.html).','',
       '## Formal paired outcomes','',table(['Case ID','Outcome class','Recorded outcomes','Selection'],[
           [e['id'],e['outcome_class'],', '.join(a+': '+('success' if v else 'failure') for a,v in e['outcomes'].items()),e['selection']]
           for e in examples if e['group']=='formal paired outcomes']),'',
       'The panels preserve both cameras and all three arms. Each arm uses its own first query, midpoint and last recorded pre-action observation; columns are not synchronized time points. Episode success is taken from audited metadata, not inferred from a single image. These snapshots do not alone prove a failure mechanism.','',
       '## Mechanism cases','']
    for e in examples:
        if e['group']=='exploratory mechanism cases':q += ['### '+e['id'],'',e['caption'],'',f'![{e["id"]}]({e["media"]})','']
    q += ['The interpretations above reuse '+link('docs/evaluation/week11-h8-validation-v1/INSPECTION.md','the saved physical-state inspection')+'. They concern retention, recovery and fixture engagement in those cases. They do not isolate a universal gripper problem or prove that horizon information alone caused a gain.','',
          '## Event forecasting cases','']
    for e in examples:
        if e['group']=='exploratory event forecasts':q += ['### '+e['id'],'',e['caption'],'',f'Task: {e["row"]["instruction"]}. Anchor frame {e["row"]["anchor"]}. Retained source ID: `{e["row"]["source_id"]}`.','']
    q += ['## Failure taxonomy boundaries','',
          table(['Supported observation','Evidence basis','Limit'],[
              ['Acquisition/retention or recovery loss','Audited physical-state inspections and recorded trajectories','Case-specific; do not infer prevalence from chosen cases'],
              ['Wrong-fixture engagement','Drawer identity in the existing matched adapter inspection','Does not by itself establish that language was ignored'],
              ['Event-type or onset disagreement','Prediction compared with automatic target on retained demonstration windows','Predicate agreement is not independently verified event truth'],
              ['Placement and earlier motion effects','Existing first-chunk and phase/channel interventions','Matched-case causal results only; retain exclusions for pre-intervention drift'],
              ['Hallucinated future / language ignored','No new causal measurement in this package','Do not assign these as proven failure causes']]),'',
          'No four-case fresh H8 investigation, new labeling study or representative failure-frequency analysis is added. The examples preserve the prior scientific boundary.','']
    (out/'QUALITATIVE.md').write_text('\n'.join(q))
    # A compact, complete Markdown companion to the exact-value CSV tables.
    tables=['# Complete tabular exports','', 'Exact unrounded numeric values are in the sibling CSV files. Empty confidence-interval fields mean no interval is asserted; use the paired contrast table where specified.','']
    for path in sorted((out/'tables').glob('*.csv')):
        with path.open() as h:
            r=csv.DictReader(h);rows=list(r);headers=r.fieldnames
        tables += ['## '+path.stem.replace('_',' '),'',f'[CSV]({path.name})','',table(headers,[[row[k] for k in headers] for row in rows]),'']
    (out/'tables/TABLES.md').write_text('\n'.join(tables))
    manifest['files_sha256']['scripts/write_week12_results.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    verifier = ROOT/'scripts/verify_week12_results.py'
    manifest['files_sha256']['scripts/verify_week12_results.py']=hashlib.sha256(verifier.read_bytes()).hexdigest()
    shutil.copyfile(verifier,out/'source/verify_week12_results.py')
    (out/'source-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    shutil.copyfile(__file__,out/'source/write_week12_results.py')
    print(json.dumps({'results_words':len((out/'RESULTS.md').read_text().split()),'qualitative_examples':len(examples)}))


if __name__=='__main__':main()
