"""Scientific comparison of frozen dev deltas and report class outcomes."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    base=parser.parse_args().base
    def read(p):return json.loads(p.read_text())
    report=read(base/'analysis/results.json');pilot=read(base.parent/'week10-preservation-pilot-v1/analysis/results.json')
    future=base.parent/'week10-future-onset-v1';starting=read(future/'eval/dev/sft/completion.json')['metrics']['macro_positive_iou']
    names=['RLVR primary','RLVR reward calibration','RLVR + preservation','SFT-only continuation']
    deltas=[];intervals=[]
    for arm in ('primary','calibration'):
        deltas.append(100*(read(future/'eval/dev'/arm/'completion.json')['metrics']['macro_positive_iou']-starting))
        intervals.append(np.array(read(future/'decisions'/f'{arm}.json')['bootstrap']['interval95'])*100)
    for arm in ('hybrid','control'):
        deltas.append(100*pilot['changes_from_sft'][arm]['macro_positive_iou'])
        intervals.append(np.array(pilot['comparisons'][arm+'_minus_sft']['interval95'])*100)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12.8,5.2),gridspec_kw={'width_ratios':[1.2,1]},constrained_layout=True)
    colors=['#667085','#667085','#2374ab','#c36a24']
    for i,(d,ci) in enumerate(zip(deltas,intervals,strict=True)):
        axes[0].plot(ci,[i,i],color=colors[i],lw=3);axes[0].scatter([d],[i],color=colors[i],s=55,zorder=3)
        axes[0].text(d,i-.18,f'{d:+.2f}',ha='center',fontsize=10,color=colors[i])
    axes[0].set(yticks=range(4),yticklabels=names,ylim=(3.5,-.6),xlabel='IoU gain over starting SFT (percentage points)',title='Dev: 396 prompts / 103 sources')
    axes[0].axvline(0,color='#475467',lw=1);axes[0].axvline(1,color='#9a3545',ls='--',lw=1,label='Dev IoU gate: +1 point')
    axes[0].legend(frameon=False,fontsize=9,loc='lower right');axes[0].grid(axis='x',alpha=.15)
    kinds=['contact','grasp','gripper_close'];x=np.arange(3)
    for offset,arm,label,color in [(-.18,'sft','Starting SFT','#667085'),(.18,'control','SFT continuation','#c36a24')]:
        values=[report['metrics'][arm]['per_class'][k]['typed_iou']*100 for k in kinds]
        bars=axes[1].bar(x+offset,values,width=.34,label=label,color=color)
        axes[1].bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
    ci=report['bootstrap']['interval95'];delta=100*report['delta_iou']
    axes[1].set(xticks=x,xticklabels=['Contact','Grasp','Gripper close'],ylabel='Typed positive IoU (%)',ylim=(0,60),title='Report: 384 prompts / 80 sources')
    axes[1].legend(frameon=False,fontsize=9);axes[1].grid(axis='y',alpha=.15);axes[1].set_axisbelow(True)
    fig.suptitle(f'SFT continuation report gain: {delta:+.2f} points · 95% interval [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]',fontsize=16)
    fig.text(.5,-.025,'Intervals: paired physical-source bootstrap · single training seed · exploratory weak labels · report interval includes zero',ha='center',fontsize=10,color='#475467')
    fig.savefig(base/'analysis/dev-and-report.png',dpi=180,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':main()
