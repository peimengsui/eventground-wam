"""Supplemental fixed dev summaries; no fitting, selection or report-model inference."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_supervision import sealed,sha
from eventground_wam.eventground_weak_evaluation import per_example,metrics


def calibration(rows):
    bins=['none','1-4','5-8','9-16','17-32'];p=np.asarray([r['horizon_bin_probabilities'] for r in rows])
    y=np.eye(5)[[bins.index(r['lead_time_bin']) for r in rows]];reliability={}
    for i,name in enumerate(bins):
        bucket=[]
        for lo in np.arange(10)/10:
            mask=(p[:,i]>=lo)&(p[:,i]<(lo+.1) if lo<.9 else p[:,i]<=1)
            bucket.append(dict(lower=float(lo),upper=float(lo+.1),n=int(mask.sum()),
                mean_prediction=float(p[mask,i].mean()) if mask.any() else None,
                observed_frequency=float(y[mask,i].mean()) if mask.any() else None))
        reliability[name]=bucket
    return dict(brier_score=float(np.square(p-y).sum(1).mean()),reliability=reliability,weak_targets=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();b=a.base
    protocol=sealed(Path('configs/training/week10_weak_rlvr_evaluation_v1.json'))
    data=sealed(Path('docs/evaluation/week10-exploratory-weak-v4/dataset.json'));manifest=sealed(b/'event-dev/manifest.json')
    rows=manifest['rows'];lookup={r['prompt_id']:r for r in rows};arms={}
    for folder in sorted((b/'evaluations').iterdir()):
        if not (folder/'completion.json').exists():continue
        c=sealed(folder/'completion.json')
        if c['status']!='complete' or c['manifest_digest']!=manifest['digest']:raise ValueError('Invalid dev result')
        r=[json.loads(x) for x in (folder/'predictions.jsonl').read_text().splitlines()]
        if sha(folder/'predictions.jsonl')!=c['predictions_sha256']:raise ValueError('Changed prediction file')
        arms[folder.name]=dict(metrics=metrics(r),horizon_calibration=calibration(r),
            by_lead_bin={k:metrics([x for x in r if x['lead_time_bin']==k]) for k in ['none','1-4','5-8','9-16','17-32']},
            by_task={k:metrics([x for x in r if lookup[x['prompt_id']]['task_key']==k]) for k in sorted({x['task_key'] for x in rows})},
            latency=c['latency'],evaluation_digest=c['digest'])
    none=dict(next_event='none',interval=None,horizon=None)
    prior=[]
    for r in rows:
        index=int(hashlib.sha256(f"17:prior:{r['prompt_id']}".encode()).hexdigest(),16)%len(data['main_rows'])
        prior.append(per_example(r,data['main_rows'][index]['target']))
    baselines={'always_none':metrics([per_example(r,none) for r in rows]),'train_only_empirical_prior':metrics(prior)}
    output=seal_contract(dict(artifact_type='eventground_wam.weak_pilot_dev_supplement',protocol_digest=protocol['digest'],
        manifest_digest=manifest['digest'],arms=arms,baselines=baselines,independent_qa_passed=False,report_predictions_opened=False))
    path=b/'analysis/dev-supplement.json';path.parent.mkdir(exist_ok=True)
    with path.open('x') as f:json.dump(output,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
if __name__=='__main__':main()
