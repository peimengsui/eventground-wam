"""Launch both authorized training arms only after all qualification phases pass."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from week11_replay_common import read, sha, write

ARMS=('horizon-motion','h8-information-control')


def run(bundle, root):
    lock=(root/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=read(bundle/'contract.json')
    for name,h in cfg['code_sha256'].items(): assert sha(bundle/name)==h
    phases=[('qualify',ARMS[0],root/'qualification')]
    phases += [('restart',arm,root/'restart'/arm) for arm in ARMS]
    phases += [(phase,arm,root/('training' if phase=='train' else 'endpoint-verification')/arm)
               for arm in ARMS for phase in ('train','endpoint')]
    result={'status':'incomplete','contract_digest':cfg['digest'],'started_unix':time.time(),
            'new_evaluation_episodes':0,'stages':[]}
    try:
        for phase,arm,dest in phases:
            if (dest/'completion.json').exists() and read(dest/'completion.json')['status']=='complete':
                done=read(dest/'completion.json')
            else:
                write(root/'progress.json',{'phase':phase,'arm':arm,'time_unix':time.time()})
                with (root/f'{phase}-{arm}.log').open('a') as log:
                    command=[sys.executable,str(bundle/'scripts/run_week11_motion_training.py'),
                             '--bundle',str(bundle),'--root',str(root),'--phase',phase,'--arm',arm]
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
                    write(root/'worker.json',{'phase':phase,'arm':arm,'pid':child.pid,'command':command})
                    code=child.wait()
                    if code: raise RuntimeError(f'{phase}/{arm} exited {code}; see its preserved log')
                done=read(dest/'completion.json')
            assert done['status']=='complete' and done['passed'] and done['contract_digest']==cfg['digest']
            result['stages'].append({'phase':phase,'arm':arm,'digest':done['digest'],'seconds':done['seconds']})
            write(root/'pipeline-progress.json',result)
        endpoints={arm:read(root/'training'/arm/'endpoint/endpoint.json') for arm in ARMS}
        for k in ('sample_chain','horizon_chain','samples_seen','updates','reference_sha256'):
            assert endpoints[ARMS[0]][k]==endpoints[ARMS[1]][k],k
        result.update(status='complete',endpoints={a:e['digest'] for a,e in endpoints.items()},
                      production_updates=4000,passed=True,matched_training_exposure=True)
    except BaseException as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['finished_unix']=time.time(); write(root/'completion.json',result)
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.bundle,a.root)
