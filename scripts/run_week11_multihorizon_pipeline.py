"""Sequential guarded qualification, matched training and fixed-endpoint screen."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARMS=('h8-continuation','mixed-continuation')


def seal(v):
    v=dict(v)
    v['digest']=hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
    return v


def read(path):
    v=json.loads(path.read_text()); digest=v.pop('digest')
    if seal(v)['digest']!=digest: raise ValueError('changed artifact: '+str(path))
    return {**v,'digest':digest}


def write(path,v):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(seal(v),indent=2,sort_keys=True)+'\n');temp.replace(path)


def run_worker(root,arm,phase,source,cfg):
    stage='training' if phase in ('train','verify-endpoint') else 'qualification'
    result=root/stage/arm/(phase+'.json')
    if result.exists():
        previous=read(result)
        if previous['source_digest']!=source['digest'] or previous['design_digest']!=cfg['digest']:
            raise ValueError('existing phase belongs to a different execution source')
        if previous['passed']: return
        raise ValueError('prior failed phase retained; explicit source/attempt repair required')
    marker=root/stage/arm/'interrupt-marker.json'
    if phase=='interrupt' and marker.exists():
        record=read(root/stage/arm/'interrupt-exit.json')
        if record['source_digest']!=source['digest'] or record['exit_code']!=-signal.SIGKILL:
            raise ValueError('interruption receipt changed')
        return
    logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    path=logs/f'{arm}-{phase}.log'
    start=time.monotonic()
    write(root/'progress.json',{'stage':stage,'arm':arm,'phase':phase,'status':'running','time_unix':time.time()})
    print(json.dumps({'stage':stage,'arm':arm,'phase':phase,'status':'running'}),flush=True)
    ledger=root/stage/'active-budget.json'
    spent=read(ledger)['active_seconds'] if ledger.exists() else 0
    limit=cfg['resources'][stage+'_hours_total']*3600-spent
    if limit<=0: raise ValueError('stage time budget exhausted')
    with path.open('x') as log:
        done=subprocess.run([sys.executable,str(ROOT/'scripts/run_week11_multihorizon_training.py'),
            '--root',str(root),'--arm',arm,'--phase',phase],stdout=log,stderr=subprocess.STDOUT,timeout=limit,check=False)
    expected=-signal.SIGKILL if phase=='interrupt' else 0
    write(root/stage/arm/(phase+'-exit.json'),{'source_digest':source['digest'],'phase':phase,
          'exit_code':done.returncode,'elapsed_seconds':time.monotonic()-start})
    if done.returncode!=expected: raise ValueError(f'{arm}/{phase} exited {done.returncode}; see {path}')
    if phase=='interrupt':
        if not read(marker)['partial_update']: raise ValueError('interruption did not exercise partial update')
    elif not read(result)['passed']: raise ValueError('worker did not pass')


def qualify(root,source,cfg):
    path=root/'qualification/qualification.json'
    if path.exists():
        q=read(path)
        if q['source_digest']!=source['digest'] or not q['passed']: raise ValueError('qualification source changed')
        return q
    initial, runtime = None, None
    phases={}; projection=0.; mean=0.
    for arm in ARMS:
        for phase in ('correctness','uninterrupted','interrupt','resume','resources'):
            run_worker(root,arm,phase,source,cfg)
            if phase=='interrupt': continue
            r=read(root/'qualification'/arm/(phase+'.json'))
            phases[arm+'/'+phase]=r['digest']
            if initial is None: initial=r['initial_tensors'];runtime=r['runtime']['digest']
            if r['initial_tensors']!=initial or r['runtime']['digest']!=runtime:
                raise ValueError('qualification starting tensors or runtime differ')
        r=read(root/'qualification'/arm/'resources.json')
        # Count a fresh startup for training AND the later independent endpoint
        # reload phase, plus one measured full save as finalization allowance.
        overhead=2*r['initialization_seconds']+r['checkpoint_seconds']
        projection+=r['conservative_projection']['total_seconds']+overhead
        mean+=r['mean_projection']['total_seconds']+overhead
    passed=projection<=cfg['resources']['maximum_training_projection_hours_including_startup_and_finalization']*3600
    write(path,{'artifact_type':'eventground_wam.week11_multihorizon_qualification',
        'source_digest':source['digest'],'design_digest':cfg['digest'],'runtime_digest':runtime,
        'initial_tensors':initial,'phases':phases,'projected_training_seconds':projection,
        'mean_projected_training_seconds':mean,'passed':passed,'pilot_weights_discarded':True})
    if not passed: raise ValueError('training resource projection exceeds frozen gate')
    return read(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--qualification-only',action='store_true')
    parser.add_argument('--training-only',action='store_true')
    args=parser.parse_args();root=args.root;root.mkdir(parents=True,exist_ok=True)
    source=read(ROOT/'week11-multihorizon-source.json')
    cfg=read(ROOT/'configs/training/week11_multihorizon_pilot_v1.json')
    with (root/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            q=qualify(root,source,cfg)
            print(json.dumps({'status':'qualified','mean_training_hours':q['mean_projected_training_seconds']/3600,
                              'conservative_training_hours':q['projected_training_seconds']/3600}),flush=True)
            if args.qualification_only:return
            for arm in ARMS:
                for phase in ('train','verify-endpoint'):run_worker(root,arm,phase,source,cfg)
            endpoints={arm:read(root/'training'/arm/'endpoint/endpoint.json') for arm in ARMS}
            a,b=endpoints.values()
            if a['sample_chain']!=b['sample_chain'] or a['future_chain']!=b['future_chain']:
                raise ValueError('trained arms had different data exposures')
            write(root/'training/completion.json',{'passed':True,'design_digest':cfg['digest'],
                  'source_digest':source['digest'],'endpoints':{k:v['digest'] for k,v in endpoints.items()},
                  'matched_sample_and_future_chains':True,'updates_per_arm':2000})
            if args.training_only:return
            with (root/'logs/screen.log').open('x') as log:
                done=subprocess.run([sys.executable,str(ROOT/'scripts/run_week11_multihorizon_screen.py'),
                    '--root',str(root)],stdout=log,stderr=subprocess.STDOUT,timeout=cfg['resources']['screen_hours_total']*3600,check=False)
            if done.returncode:raise ValueError('screen stopped; see screen log')
            write(root/'completion.json',{'status':'complete','source_digest':source['digest'],
                  'design_digest':cfg['digest'],'training':read(root/'training/completion.json')['digest'],
                  'screen':read(root/'screen/completion.json')['digest']})
            write(root/'progress.json',{'status':'complete','time_unix':time.time()})
        except BaseException as exc:
            write(root/'progress.json',{'status':'stopped','error':f'{type(exc).__name__}: {exc}','time_unix':time.time()})
            raise


if __name__=='__main__':main()
