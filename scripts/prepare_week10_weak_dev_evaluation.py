"""Bind a completed pilot/SFT to the fixed dev manifest and frozen reporting code."""
import argparse
import json
from pathlib import Path
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_weak_supervision import sealed,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=['SFT','primary','calibration'],required=True)
    p.add_argument('--prior-gpu-seconds',type=float,required=True);a=p.parse_args()
    b=Path('docs/evaluation/week10-weak-rlvr-v1');remote='/home/peimengsui/data/eventground-wam/libero-plus/results/week10-weak-rlvr-v1'
    proto=sealed('configs/training/week10_weak_rlvr_evaluation_v1.json');manifest=sealed(b/'event-dev/manifest.json');labels=sealed(b/'event-dev/label-completion.json')
    if labels['manifest_digest']!=manifest['digest'] or manifest['selected_count']==0:raise ValueError('No completed balanced dev manifest')
    q=sealed(b/'qualification-launch.json');details=sealed(b/'evaluation-reporting-details.json')
    files={**q['code_sha256'],**details['code_sha256']}
    for name,h in files.items():
        if sha(b/'source'/name)!=h:raise ValueError('Changed frozen source '+name)
    c=dict(artifact_type='eventground_wam.weak_dev_inference_launch',protocol_digest=proto['digest'],arm=a.arm,
        output=f'{remote}/evaluations/{a.arm}',sft_root='/home/peimengsui/data/eventground-wam/libero-plus/results/week10-weak-sft-v1',
        panel_root=f'{remote}/event-dev',manifest_digest=manifest['digest'],label_completion_digest=labels['digest'],
        code_sha256=files,reporting_details_digest=details['digest'],resources=dict(wall_seconds=1800,output_bytes=1024**3,
            minimum_free_bytes=50*1024**3,prior_active_gpu_seconds=a.prior_gpu_seconds),independent_qa_passed=False)
    if a.arm!='SFT':
        done=sealed(b/a.arm/'completion.json')
        if done['status']!='complete' or done['protocol_digest']!=proto['digest']:raise ValueError('Incomplete pilot')
        c.update(attempt_root=f'{remote}/{a.arm}',attempt_digest=done['digest'])
    with (b/f'eval-{a.arm}-launch.json').open('x') as f:json.dump(seal_contract(c),f,indent=2,sort_keys=True);f.write('\n')
if __name__=='__main__':main()
