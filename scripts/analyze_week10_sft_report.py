"""Paired report analysis; no checkpoint selection or further optimization."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze_week10_weak_pilot import read_eval,paired_bootstrap
from analyze_week10_future_onset import extended,baselines
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_sft_report import verify,initial_base
from eventground_wam.eventground_weak_supervision import sealed


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    base=parser.parse_args().base;p,_,manifests,_=verify(base)
    before,first=read_eval(initial_base(base)/'eval/report/sft')
    after,second=read_eval(base/'eval/report/control')
    if (after['protocol_digest']!=p['digest'] or after['manifest_digest']!=manifests['report']['digest']
        or after['endpoint_completion_digest']!=p['endpoint_completion_digest'] or not after['model_unchanged']
        or after['training_updates']!=0 or after['future_model_input_reads']!=0):
        raise ValueError('Evaluation binding/integrity differs')
    for records in (first,second):
        if ([r['prompt_id'] for r in records]!=[r['prompt_id'] for r in manifests['report']['rows']]
            or [r['target'] for r in records]!=[r['target'] for r in manifests['report']['rows']]):
            raise ValueError('Report order or target changed')
    bootstrap=paired_bootstrap(first,second)
    bootstrap['interpretation']='Paired physical-source bootstrap on the previously fixed, label-informed exploratory report panel; not an untouched confirmatory test.'
    common=[(a,b) for a,b in zip(first,second,strict=True) if a['target']['next_event']!='none' and a['correct_event'] and b['correct_event']]
    common_timing=dict(n=len(common),starting_sft_mae=float(np.mean([a['start_error'] for a,b in common])) if common else None,
                       continuation_mae=float(np.mean([b['start_error'] for a,b in common])) if common else None)
    result=seal_contract(dict(artifact_type='eventground_wam.sft_continuation_report_results',protocol_digest=p['digest'],
        report_manifest_digest=manifests['report']['digest'],endpoint_completion_digest=p['endpoint_completion_digest'],
        evaluations=dict(sft=before['digest'],control=after['digest']),
        metrics=dict(sft=before['metrics'],control=after['metrics']),
        delta_iou=after['metrics']['macro_positive_iou']-before['metrics']['macro_positive_iou'],bootstrap=bootstrap,
        delta_none_fp=after['metrics']['none_false_positive_rate']-before['metrics']['none_false_positive_rate'],
        common_matched_onset=common_timing,extras=dict(sft=extended(first),control=extended(second)),
        baselines=baselines(manifests['train']['rows'],manifests['report']['rows']),
        latency=dict(sft=before['latency'],control=after['latency']),latency_qualification='Baseline was measured in the earlier run, not a contemporaneous matched benchmark.',
        training_updates=0,model_selection_performed=False,automatic_retries=0,independent_qa_passed=False,limitations=p['limitations']))
    out=base/'analysis';out.mkdir(exist_ok=False)
    with (out/'results.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    lines=['# SFT continuation: separate report evaluation','',
        'The final SFT-only continuation checkpoint was selected on dev before this evaluation. No new training, checkpoint search or report-based tuning occurred.',
        'This is the same fixed 384-prompt report panel used for the starting SFT baseline: 96 prompts/class, 80 canonical sources and 40 tasks, disjoint from event training/dev sources.',
        'It is label-informed and exploratory; its labels/counts and baseline results were previously inspected. Automatic weak targets and unresolved inherited-M4 alias exposure limit the claims.', '',
        '| Endpoint | Macro positive IoU | None false positives | Matched onset MAE | Positive match coverage |',
        '|---|---:|---:|---:|---:|']
    for name,records in [('sft',first),('control',second)]:
        m=result['metrics'][name];fp=sum(r['target']['next_event']=='none' and r['prediction']['next_event']!='none' for r in records)
        lines.append(f"| {name} | {100*m['macro_positive_iou']:.2f}% | {fp}/96 | {m['matched_positive_start_mae']:.3f} frames | {100*m['matched_positive_coverage']:.2f}% |")
    ci=bootstrap['interval95'];lines+=['',f"Paired IoU change: **{100*result['delta_iou']:+.2f} percentage points**; 95% source-bootstrap interval "+(f"**[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}] points**." if ci else 'withheld.'),
        f"Bootstrap: {bootstrap['sources']} physical sources, 10,000 draws, PCG64 seed 10. This quantifies evaluation-source sampling, not training-seed variability.", '',
        '| Class | Starting SFT typed IoU | Continued SFT typed IoU | Correct types: before → after |', '|---|---:|---:|---:|']
    for kind in ('contact','grasp','gripper_close'):
        a=before['metrics']['per_class'][kind];b=after['metrics']['per_class'][kind]
        lines.append(f"| {kind} | {100*a['typed_iou']:.2f}% | {100*b['typed_iou']:.2f}% | {a['matched']}/96 → {b['matched']}/96 |")
    lines+=['',f"On the same {len(common)} correctly typed positive prompts, onset MAE changed from {common_timing['starting_sft_mae']:.3f} to {common_timing['continuation_mae']:.3f} frames. Conditional timing on each model's own matched set should be read alongside coverage.",
        '', 'Full confusion matrices, lead-bin metrics, censored/uncensored timing, calibration, baselines, latency and completion bindings are in `results.json`.', '',
        'No RLVR report arm is added after viewing these results. Prior dev failures and the earliest small-panel pass remain separate evidence. No original package is modified.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(delta_iou=result['delta_iou'],bootstrap=bootstrap,none_fp_delta=result['delta_none_fp'],metrics=result['metrics'])))


if __name__=='__main__':main()
