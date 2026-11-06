#!/usr/bin/env python3
"""Sequential qualified Week 9 panels; completed work is verified and preserved."""
from pathlib import Path
import argparse,subprocess
from eventground_wam.b0_training import read_sealed
from eventground_wam.auxiliary_evaluation import require
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,required=True)
p.add_argument('--arms',nargs='+',choices=['B0','M1','M2','M3','M4'],default=['B0','M1','M3'])
a=p.parse_args()
def call(script,*args):subprocess.run(['bash',str(ROOT/'scripts/week9_feifei.sh'),script,*map(str,args)],check=True)
# Every initially available arm qualifies before the first formal episode.
for arm in a.arms:
    if not (a.root/'qualification'/arm/'qualification.json').exists():
        call('evaluate_week9.py','qualify','--root',a.root,'--arm',arm)
for arm in a.arms:
    for panel in ('perturbed','clean'):
        complete=a.root/'panels'/arm/panel/'evaluation-report.json'
        if complete.exists():
            require(read_sealed(complete)['mode']=='formal','nonformal report in queue')
        else:
            call('evaluate_week9.py','run','--root',a.root,'--arm',arm,'--panel',panel)
        call('analyze_week9.py','--root',a.root)
print('Authorized Week 9 queue completed.',flush=True)
