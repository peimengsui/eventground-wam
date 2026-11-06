"""Render a static scientific figure from completed immutable pilot evidence."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--base', type=Path, required=True)
    base = parser.parse_args().base
    result = json.loads((base/'analysis/results.json').read_text())
    logs = {arm:[json.loads(x) for x in (base/'runs'/arm/'metrics.jsonl').read_text().splitlines()]
            for arm in ('hybrid', 'control')}
    palette = dict(sft='#667085', hybrid='#2374ab', control='#c36a24')
    plt.rcParams.update({'font.size':11, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axs = plt.subplots(2, 2, figsize=(12,8.5), constrained_layout=True)
    for arm, rows in logs.items():
        steps = [r['global_step'] for r in rows]
        axs[0,0].plot(steps, [r['sft_loss'] for r in rows], label=arm, color=palette[arm], alpha=.35, lw=1)
        smooth = [np.mean([r['sft_loss'] for r in rows[max(0,i-4):i+1]]) for i in range(len(rows))]
        axs[0,0].plot(steps, smooth, color=palette[arm], lw=2)
    axs[0,0].set(title='Supervised loss on the same training prompts', xlabel='Optimizer update', ylabel='Structured SFT loss')
    axs[0,0].legend(frameon=False); axs[0,0].text(.03,.04,'Faint: raw batch; solid: trailing 5 updates',transform=axs[0,0].transAxes, fontsize=9)
    hybrid = logs['hybrid']; steps=[r['global_step'] for r in hybrid]
    axs[0,1].plot(steps,[r['kl'] for r in hybrid], color=palette['hybrid'])
    axs[0,1].set(title='Hybrid: divergence from frozen SFT', xlabel='Optimizer update', ylabel='Mean sampled KL estimate (unweighted)')
    arms=['sft','hybrid','control']; names=['Starting SFT','RLVR + preservation','SFT-only control']
    metrics=result['metrics']; values=[metrics[a]['macro_positive_iou']*100 for a in arms]
    bars=axs[1,0].bar(names,values,color=[palette[a] for a in arms],width=.6)
    axs[1,0].bar_label(bars,fmt='%.2f%%',padding=4)
    axs[1,0].axhline(values[0]+1,color='#9a3545',ls='--',lw=1,label='Required SFT + 1 point')
    axs[1,0].set(title='Fixed dev panel: macro positive IoU',ylabel='IoU (%)',ylim=(0,max(values)+10)); axs[1,0].legend(frameon=False,fontsize=9)
    fp=[metrics[a]['none_false_positive_rate']*100 for a in arms]
    bars=axs[1,1].bar(names,fp,color=[palette[a] for a in arms],width=.6)
    axs[1,1].bar_label(bars,labels=[f'{round(x*.99)}/99 ({x:.2f}%)' for x in fp],padding=4)
    axs[1,1].axhline(fp[0]+2,color='#9a3545',ls='--',lw=1,label='Maximum SFT + 2 points')
    axs[1,1].set(title='Fixed dev panel: none false positives',ylabel='False-positive rate (%)',ylim=(0,max(fp+[fp[0]+2])+2)); axs[1,1].legend(frameon=False,fontsize=9)
    for ax in axs.flat:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    for ax in axs[1]:ax.tick_params(axis='x',labelsize=9)
    fig.suptitle(f"Week 10 preservation pilot · λ = {result['selected_weight']:.4f}\n500 matched prompts · 32 updates · identical initialization and learning rates",fontsize=16)
    fig.text(.5,-.018,'Exploratory weak-label agreement · reused 396-prompt dev panel · one seed · no full training launched',ha='center',fontsize=10,color='#475467')
    fig.savefig(base/'analysis/learning-and-dev.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
