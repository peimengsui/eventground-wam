"""Apply the prospective development gate and preserve source-paired weak-label comparisons."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_supervision import sealed,sha
from eventground_wam.eventground_weak_evaluation import CLASSES,metrics,promotion


def paired_bootstrap(first,second):
    if [r['prompt_id'] for r in first]!=[r['prompt_id'] for r in second] or [r['source_id'] for r in first]!=[r['source_id'] for r in second]:raise ValueError('Unpaired predictions')
    sources=sorted({r['source_id'] for r in first});kinds=[k for k in CLASSES if k!='none']
    nums=np.zeros((len(sources),3));dens=np.zeros_like(nums);index={s:i for i,s in enumerate(sources)}
    for a,b in zip(first,second,strict=True):
        if a['target']!=b['target']:raise ValueError('Changed paired target')
        if a['target']['next_event'] in kinds:
            i=index[a['source_id']];j=kinds.index(a['target']['next_event'])
            nums[i,j]+=b['positive_iou']-a['positive_iou'];dens[i,j]+=1
    rng=np.random.Generator(np.random.PCG64(10));values=[];undefined=0
    for _ in range(10000):
        selected=rng.integers(0,len(sources),size=len(sources));d= dens[selected].sum(0)
        if (d==0).any():undefined+=1
        else:values.append(float(np.mean(nums[selected].sum(0)/d)))
    return dict(unit='physical_source_id',sources=len(sources),draws=10000,seed=10,rng='PCG64',
        undefined_draws=undefined,undefined_fraction=undefined/10000,
        interval95=np.quantile(values,[.025,.975],method='linear').tolist() if undefined<=100 and values else None,
        interval_withheld=undefined>100,interpretation='paired exploratory weak-label IoU difference; development panel, not final report')


def read_eval(path):
    c=sealed(path/'completion.json')
    if c['status']!='complete' or sha(path/'predictions.jsonl')!=c['predictions_sha256']:raise ValueError('Incomplete/changed evaluation')
    rows=[json.loads(x) for x in (path/'predictions.jsonl').read_text().splitlines()]
    if metrics(rows)!=c['metrics']:raise ValueError('Metric reconstruction differs')
    return c,rows


def main():
    p=argparse.ArgumentParser()
    for k in ('protocol','sft','pilot','attempt','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--stage',choices=['primary','calibration'],required=True);p.add_argument('--primary-decision',type=Path)
    a=p.parse_args();protocol=sealed(a.protocol);sft,first=read_eval(a.sft);pilot,second=read_eval(a.pilot);attempt=sealed(a.attempt/'completion.json')
    if any(c['protocol_digest']!=protocol['digest'] for c in (sft,pilot,attempt)) or sft['manifest_digest']!=pilot['manifest_digest']:raise ValueError('Study/panel binding differs')
    if attempt['status']!='complete' or attempt['global_step']!=26 or attempt['prompts_seen']!=404 or not all(attempt['gates'].values()):raise ValueError('Wrong pilot endpoint')
    gate=promotion(sft['metrics'],pilot['metrics'],all(attempt['gates'].values()))
    result=dict(artifact_type='eventground_wam.weak_rlvr_development_decision',protocol_digest=protocol['digest'],stage=a.stage,
        sft_evaluation_digest=sft['digest'],pilot_evaluation_digest=pilot['digest'],attempt_digest=attempt['digest'],manifest_digest=sft['manifest_digest'],
        primary_promoted=gate['promoted'] if a.stage=='primary' else False,promoted=gate['promoted'],gate=gate,
        calibration_permitted=a.stage=='primary' and not gate['promoted'],bootstrap=paired_bootstrap(first,second),
        independent_qa_passed=False,full_training_launched=False,
        conditional_scope='fresh SFT -> one-pass 1332-prompt RLVR and matched SFT continuation' if gate['promoted'] else 'one predefined calibration' if a.stage=='primary' else 'omit RLVR main/control; preserve negative result',
        all_training_decisions_sealed=gate['promoted'] or a.stage=='calibration')
    if a.stage=='calibration':
        primary=sealed(a.primary_decision)
        if primary['protocol_digest']!=protocol['digest'] or not primary['calibration_permitted']:raise ValueError('Calibration decision mismatch')
        result['primary_decision_digest']=primary['digest']
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(seal_contract(result),f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
