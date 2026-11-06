"""Static scientific figures for the fixed exploratory dev pilot."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();b=a.base
s=json.loads((b/'evaluations/SFT/completion.json').read_text());r=json.loads((b/'evaluations/primary/completion.json').read_text())
fig,ax=plt.subplots(1,2,figsize=(10,4.2),gridspec_kw={'width_ratios':[1.7,1]})
classes=['contact','grasp','gripper_close'];x=np.arange(3);width=.35
for shift,c,name,color in [(-width/2,s,'SFT','#3b6ba5'),(width/2,r,'RLVR pilot','#d17c36')]:
 vals=[100*c['metrics']['per_class'][k]['typed_iou'] for k in classes]
 bars=ax[0].bar(x+shift,vals,width,label=name,color=color)
 ax[0].bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
ax[0].set_xticks(x,['Contact','Grasp','Gripper close']);ax[0].set_ylim(0,100);ax[0].set_ylabel('Typed interval IoU (%)');ax[0].legend(frameon=False)
vals=[100*s['metrics']['macro_positive_iou'],100*r['metrics']['macro_positive_iou']]
bars=ax[1].bar(['SFT','RLVR pilot'],vals,color=['#3b6ba5','#d17c36']);ax[1].bar_label(bars,fmt='%.2f',padding=3)
ax[1].set_ylim(0,65);ax[1].set_ylabel('Macro positive-event IoU (%)')
for a0 in ax:a0.spines[['top','right']].set_visible(False);a0.grid(axis='y',alpha=.2);a0.set_axisbelow(True)
fig.suptitle('Exploratory weak-label dev comparison • fixed endpoints',fontsize=13)
fig.text(.5,.015,'48 prompts (12/class), 23 source groups. Gain +1.20 pp; paired 95% interval −0.20 to +2.81 pp.\nDevelopment gate passed; no independent label QA or final-report claim.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.12,1,.94));out=b/'analysis';out.mkdir(exist_ok=True)
for ext in ('png','pdf'):fig.savefig(out/f'dev-comparison.{ext}',dpi=180,bbox_inches='tight')
plt.close(fig)
log=[json.loads(x) for x in (b/'primary/metrics.jsonl').read_text().splitlines()];groups=json.loads((b/'primary/rewards.json').read_text())['groups']
means=[];cursor=0
for row in log:
 count=len(row['prompt_ids']);means.append(float(np.mean([g['rewards'] for g in groups[cursor:cursor+count]])));cursor+=count
fig,axes=plt.subplots(1,2,figsize=(10,3.8));steps=[r['global_step'] for r in log]
axes[0].plot(steps,means,marker='.',color='#d17c36');axes[0].set_ylabel('Mean sampled weak-label reward');axes[0].set_ylim(0,1)
axes[1].plot(steps,[r['kl'] for r in log],marker='.',color='#3b6ba5');axes[1].set_ylabel('Sampled reference KL')
for a0 in axes:a0.set_xlabel('Logical update');a0.spines[['top','right']].set_visible(False);a0.grid(alpha=.2)
fig.suptitle('404 training prompts • four candidates/prompt • one fixed pass')
fig.text(.5,.01,'Different prompt batches at each update; training reward variation is not a held-out learning curve.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.94))
for ext in ('png','pdf'):fig.savefig(out/f'pilot-training.{ext}',dpi=180,bbox_inches='tight')
