"""Package fixed endpoints, development decisions and source-paired report results."""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
from analyze_week10_weak_pilot import read_eval,paired_bootstrap
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_supervision import sealed,sha
from eventground_wam.eventground_weak_evaluation import per_example,metrics
from eventground_wam.eventground_future_onset import TASK

KINDS=['none','gripper_close','contact','grasp']
BINS=['none','1-4','5-8','9-16','17-32']


def calibration(records,field,classes,target_key):
    probs=np.array([r[field] for r in records],dtype=float)
    if probs.shape!=(len(records),len(classes)) or not np.isfinite(probs).all() or not np.allclose(probs.sum(1),1,atol=1e-5):raise ValueError('Invalid probabilities')
    labels=np.array([classes.index(r['target']['next_event'] if target_key=='event' else r['lead_time_bin']) for r in records])
    truth=np.eye(len(classes))[labels];conf=probs.max(1);correct=probs.argmax(1)==labels;ece=0.;bins=[]
    for i in range(10):
        mask=(conf>=i/10)&(conf<((i+1)/10) if i<9 else conf<=1)
        n=int(mask.sum());acc=float(correct[mask].mean()) if n else None;mean=float(conf[mask].mean()) if n else None
        if n:ece+=n/len(records)*abs(acc-mean)
        bins.append(dict(lower=i/10,upper=(i+1)/10,n=n,accuracy=acc,mean_confidence=mean))
    return dict(multiclass_brier=float(np.mean(((probs-truth)**2).sum(1))),ece_10_equal_width=ece,bins=bins)


def extended(records):
    uncensored=[r['end_error'] for r in records if r['correct_event'] and r['target']['next_event']!='none' and not r['end_censored']]
    return dict(event_calibration=calibration(records,'event_probabilities',KINDS,'event'),
        horizon_calibration=calibration(records,'horizon_bin_probabilities',BINS,'horizon'),
        by_lead_bin={k:metrics([r for r in records if r['lead_time_bin']==k]) for k in BINS},
        uncensored_matched_end_mae=float(np.mean(uncensored)) if uncensored else None,uncensored_matched_end_n=len(uncensored))


def baselines(train,report):
    counts=Counter(r['target']['next_event'] for r in train);total=len(train)
    prior=[counts[k]/total for k in KINDS];hcounts=Counter(r['lead_time_bin'] for r in train);hprior=[hcounts[k]/total for k in BINS]
    kind=max(KINDS,key=lambda k:(counts[k],-KINDS.index(k)))
    if kind=='none':pred=dict(next_event='none',interval=None,horizon=None)
    else:
        intervals=Counter(tuple(r['target']['interval']) for r in train if r['target']['next_event']==kind)
        interval=min(intervals,key=lambda x:(-intervals[x],x))
        pred=dict(next_event=kind,interval=list(interval),horizon=interval[0])
    result={}
    for name,p,event_probs,horizon_probs in [('always_none',dict(next_event='none',interval=None,horizon=None),[1,0,0,0],[1,0,0,0,0]),
                                           ('training_only_prior',pred,prior,hprior)]:
        records=[]
        for row in report:
            r=per_example(row,p);r.update(event_probabilities=event_probs,horizon_bin_probabilities=horizon_probs);records.append(r)
        result[name]=dict(prediction=p,event_probabilities=event_probs,horizon_probabilities=horizon_probs,metrics=metrics(records),extra=extended(records))
    return result


def verify_metadata_run(base,stage):
    c=sealed(base/'runs'/stage/'completion.json');s=sealed(base/'schedules'/f'{stage}.json')
    if c['status']!='complete' or not all(c['gates'].values()) or c['schedule_digest']!=s['digest'] or c['global_step']!=s['updates'] or c['prompts_seen']!=s['presentations']:raise ValueError('Invalid stage '+stage)
    rows=[json.loads(x) for x in (base/'runs'/stage/'metrics.jsonl').read_text().splitlines()]
    normal=[r for r in rows if not r['recovery_replay']]
    if [r['prompt_ids'] for r in normal]!=s['batches']:raise ValueError('Changed prompt exposure')
    for name,h in c['files_sha256'].items():
        if sha(base/'runs'/stage/name)!=h:raise ValueError('Changed stage metadata')
    return c,normal


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);a=parser.parse_args();base=a.base
    p=sealed(base/'protocol.json');execution=sealed(base/'execution-completion.json');decision=sealed(base/'decisions/final.json')
    if execution['status'] not in ('complete','complete_dev_no_go') or execution['protocol_digest']!=p['digest']:raise ValueError('Queue not complete')
    out=base/'analysis';out.mkdir(exist_ok=True)
    runs={};history={}
    for folder in sorted((base/'runs').iterdir()):
        c,rows=verify_metadata_run(base,folder.name);runs[folder.name]=c;history[folder.name]=rows
    train=sealed(base/'train-manifest.json');report=sealed(base/'report-manifest.json')
    development={}
    for arm in ('sft','primary','calibration'):
        if not (base/'eval/dev'/arm/'completion.json').exists(): continue
        c,rows=read_eval(base/'eval/dev'/arm)
        if c['protocol_digest']!=p['digest'] or c['endpoint_completion_digest']!=runs[arm]['digest']: raise ValueError('Development endpoint binding')
        d=sealed(base/'decisions'/f'{arm}.json') if arm!='sft' else None
        development[arm]=dict(completion_digest=c['digest'],metrics=c['metrics'],decision=d)
    evaluations={};predictions={}
    arms=['sft','main','control'] if decision['promoted'] else ['sft']
    for arm in arms:
        c,rows=read_eval(base/'eval/report'/arm)
        if (c['manifest_digest']!=report['digest'] or c['protocol_digest']!=p['digest']
            or c['endpoint_completion_digest']!=runs[arm]['digest']
            or [r['prompt_id'] for r in rows]!=[r['prompt_id'] for r in report['rows']]
            or [r['target'] for r in rows]!=[r['target'] for r in report['rows']]):raise ValueError('Unpaired report identity')
        evaluations[arm]=dict(completion_digest=c['digest'],metrics=c['metrics'],latency=c['latency'],peak_reserved_bytes=c['peak_reserved_bytes'],extra=extended(rows))
        predictions[arm]=rows
    comparisons={}
    for first,second in ([('sft','main'),('sft','control'),('control','main')] if decision['promoted'] else []):
        bootstrap=paired_bootstrap(predictions[first],predictions[second])
        bootstrap['interpretation']='source-paired, label-informed exploratory report IoU difference; no independent truth claim'
        comparisons[f'{second}_minus_{first}']=dict(delta=evaluations[second]['metrics']['macro_positive_iou']-evaluations[first]['metrics']['macro_positive_iou'],bootstrap=bootstrap)
    if decision['promoted']:
        if sealed(base/'schedules/main.json')['batches']!=sealed(base/'schedules/control.json')['batches']:raise ValueError('Unmatched control')
        identities={arm:sealed(base/'runs'/arm/'identity.json') for arm in ('main','control')}
        if identities['main']['initial_trainable_sha256']!=identities['control']['initial_trainable_sha256'] or any(x['sft_completion_digest']!=runs['sft']['digest'] for x in identities.values()):raise ValueError('Initial models differ')
    value=seal_contract(dict(artifact_type='eventground_wam.future_onset_results',task_definition=TASK,protocol_digest=p['digest'],
        execution_completion_digest=execution['digest'],decision_digest=decision['digest'],promoted=decision['promoted'],
        report_manifest_digest=report['digest'],development_results=development,report_results=evaluations,comparisons=comparisons,
        baselines=baselines(train['rows'],report['rows']),
        training={k:dict(completion_digest=v['digest'],updates=v['global_step'],presentations=v['prompts_seen'],
            seconds=v['elapsed_seconds'],peak_reserved_bytes=v['peak_reserved_bytes'],endpoint_sha256=v.get('endpoint_sha256')) for k,v in runs.items()},
        original_experiments_preserved=True,independent_qa_passed=False,limitations=p['limitations'],
        control_matching='Identical starting SFT parameters and prompt presentations; inherited stage-specific rates (RLVR head1e-5/LoRA1e-6, control head3e-4/LoRA1e-5); RLVR draws4 samples/prompt, so this is not a same-rate objective isolation or FLOP-matched comparison'))
    if (out/'results.json').exists():
        if sealed(out/'results.json')!=value:raise ValueError('Previously computed analysis differs')
    else:
        with (out/'results.json').open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    lines=['# Week 10 next-new-onset results','',f"Development gate: **{'passed' if decision['promoted'] else 'no-go'}**. All executed fixed attempts are complete; prior experiments are preserved.",'',
           'The evaluation uses automatic physical labels whose coverage was inspected before this task revision. These are label-informed exploratory agreement results, not independently verified event accuracy or robot success.', '',
           '| Model | Macro positive IoU | None FP | Matched onset MAE | Matched positive coverage | Joint ±2 frames | Forward median / p95 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for arm in arms:
        m=evaluations[arm]['metrics'];l=evaluations[arm]['latency'];mae=m['matched_positive_start_mae']
        lines.append(f"| {arm} | {m['macro_positive_iou']*100:.2f}% | {m['none_false_positive_rate']*100:.2f}% | {mae:.3f} | {m['matched_positive_coverage']*100:.2f}% | {m['joint_two_frame_accuracy']*100:.2f}% | {l['median_ms']:.1f} / {l['p95_ms']:.1f} ms |")
    lines+=['','Report: 384 prompts, 96/class, from 80 sources / 40 tasks; same fixed panel for all report arms. Wrong types and missed positives receive zero IoU. Timing errors are conditional on the correctly predicted event type and must be read with coverage. Forward latency excludes preprocessing, calibration diagnostics and first five warm-up examples.','', '## Source-paired comparisons','']
    for name,c in comparisons.items():
        ci=c['bootstrap']['interval95'];lines.append(f"- {name}: {c['delta']*100:+.2f} pp; 95% source-bootstrap interval "+(f"[{ci[0]*100:+.2f}, {ci[1]*100:+.2f}] pp." if ci else 'withheld due to undefined draws.'))
    if not comparisons:lines.append('Formal RLVR/control were not launched because the fixed development gate did not pass.')
    lines+=['','## Fixed development decisions','','| Dev endpoint | Macro positive IoU | None FP | Matched onset MAE | Gate |','|---|---:|---:|---:|---|']
    for arm,v in development.items():
        m=v['metrics'];d=v['decision'];gate='baseline' if d is None else 'passed' if d['promoted'] else 'no-go: '+', '.join(k for k,z in d['gate']['gates'].items() if not z)
        lines.append(f"| {arm} | {m['macro_positive_iou']*100:.2f}% | {m['none_false_positive_rate']*100:.2f}% | {m['matched_positive_start_mae']:.3f} | {gate} |")
    lines+=['','The fixed dev panel has 396 prompts (99/class) from 103 sources. The primary and optional calibration are separate fresh-SFT runs; calibration uses only its predeclared reward weights. No report outcomes affect the decision.','']
    lines+=['','## Training and integrity','','| Stage | Updates | Presentations | Minutes | Peak GPU GiB |','|---|---:|---:|---:|---:|']
    for k,v in runs.items():lines.append(f"| {k} | {v['global_step']} | {v['prompts_seen']} | {v['elapsed_seconds']/60:.2f} | {v['peak_reserved_bytes']/1024**3:.2f} |")
    lines+=['','Qualification includes one additional recovery replay; its weights are discarded. SFT starts fresh from M4. Each RLVR/control stage starts independently from the completed revised SFT with an empty optimizer. Main/control match prompt order and exposure. They inherit their stage-specific learning rates (RLVR head1e-5/LoRA1e-6; SFT control head3e-4/LoRA1e-5); this is not an equal-learning-rate objective isolation or a FLOP-matched comparison.','',
        'All class-wise metrics, confusion matrices, lead-time bins, censoring, calibration, baselines, endpoint identities and paired uncertainty are in `results.json`. Original physical verification and camera qualifications remain linked through `source-index.json`.','',
        '![Learning curves and report](learning-and-report.png)','', 'Remaining scientific limits: automatic labels; prior label inspection; one training seed; unresolved M4 historical alias exposure; no claim about robot rollout improvement.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(promoted=decision['promoted'],report={k:v['metrics']['macro_positive_iou'] for k,v in evaluations.items()},comparisons=comparisons),indent=2))
if __name__=='__main__':main()
