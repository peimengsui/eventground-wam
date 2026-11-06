"""Sequential bounded queue for the approved revision, preserving each fixed attempt."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_future_runtime import verify,validate_training_completion
from eventground_wam.eventground_weak_supervision import sealed


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);a=parser.parse_args();base=a.base
    p,_,_,_=verify(base);start=time.monotonic();done=[]
    def save(name,value):
        with (base/name).open('x') as f:json.dump(seal_contract(value),f,indent=2,sort_keys=True);f.write('\n')
    save('execution-start.json',dict(protocol_digest=p['digest'],started_at=datetime.now(timezone.utc).isoformat(),automatic_retries=0))
    def command(script,*args,limit):
        if time.monotonic()-start>p['queue_wall_seconds']:raise TimeoutError('Aggregate queue deadline')
        key=script+':'+':'.join(args)
        state=dict(protocol_digest=p['digest'],current=key,completed=done,elapsed_seconds=time.monotonic()-start)
        temp=base/'queue-status.tmp';temp.write_text(json.dumps(state,indent=2)+'\n');os.replace(temp,base/'queue-status.json')
        subprocess.run([sys.executable,str(base/'source/scripts'/script),'--base',str(base),*args],check=True,timeout=limit)
        done.append(key)
    def train(stage):
        command('train_week10_future_onset.py','--stage',stage,limit=p['training_stages'][stage]['wall_seconds']+120)
        validate_training_completion(base,stage)
    def evaluate(arm,panel):command('evaluate_week10_future_onset.py','--arm',arm,'--panel',panel,limit=1920)
    status='failed_preserved';error=None
    try:
        train('sft-qualification');train('sft');evaluate('sft','dev')
        train('rlvr-qualification');train('primary');evaluate('primary','dev')
        command('decide_week10_future_onset.py','--stage','primary',limit=300)
        if not sealed(base/'decisions/primary.json')['promoted']:
            train('calibration');evaluate('calibration','dev')
            command('decide_week10_future_onset.py','--stage','calibration',limit=300)
        final=sealed(base/'decisions/final.json')
        if final['promoted']:
            train('main');train('control')
            for arm in ('sft','main','control'):evaluate(arm,'report')
        else:evaluate('sft','report')
        status='complete' if final['promoted'] else 'complete_dev_no_go'
    except Exception as e:error=repr(e);raise
    finally:
        save('execution-completion.json',dict(protocol_digest=p['digest'],status=status,error=error,completed=done,
            elapsed_seconds=time.monotonic()-start,finished_at=datetime.now(timezone.utc).isoformat(),automatic_retries=0))
if __name__=='__main__':main()
