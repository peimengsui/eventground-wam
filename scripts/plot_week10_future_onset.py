"""Plot verified fixed-run metadata; no training or scientific decision changes."""
import argparse
import json
from pathlib import Path
import numpy as np

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);a=parser.parse_args();base=a.base
    out=base/'analysis';results=json.loads((out/'results.json').read_text());evaluations=results['report_results'];arms=list(evaluations)
    history={d.name:[json.loads(x) for x in (d/'metrics.jsonl').read_text().splitlines() if not json.loads(x)['recovery_replay']] for d in (base/'runs').iterdir()}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False,'font.size':10})
    fig,axs=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for stage in ('sft','control'):
        if stage in history:
            r=history[stage];x=np.array([z['global_step'] for z in r]);y=np.array([z['loss'] for z in r])
            axs[0].plot(x,y,alpha=.18);w=min(10,len(y));axs[0].plot(x[w-1:],np.convolve(y,np.ones(w)/w,'valid'),label=stage)
    axs[0].set(title='SFT losses (10-update mean)',xlabel='Updates within stage',ylabel='Structured loss');axs[0].legend()
    for stage in ('primary','calibration','main'):
        if stage in history:
            r=history[stage];axs[1].plot([z['global_step'] for z in r],[z['kl'] for z in r],label=stage)
    axs[1].set(title='RLVR reference KL',xlabel='Updates within stage',ylabel='Logged KL');axs[1].legend()
    vals=[evaluations[k]['metrics']['macro_positive_iou']*100 for k in arms]
    axs[2].bar(arms,vals,color=['#3d74b5','#e78442','#57a98d'][:len(arms)])
    for i,v in enumerate(vals):axs[2].text(i,v+.5,f'{v:.2f}%',ha='center')
    axs[2].set(title='Separate report: weak-label agreement',ylabel='Macro positive IoU (%)',ylim=(0,max(20,max(vals)+8)))
    fig.suptitle('Week 10 — next new event onset • exploratory, single seed',fontweight='bold');fig.savefig(out/'learning-and-report.png');plt.close(fig)

if __name__=='__main__':main()
