#!/usr/bin/env python3
"""Plot the completed, artifact-audited M3/M1 development comparison."""
from pathlib import Path
import argparse,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--analysis',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();r=json.loads(a.analysis.read_text());a.output.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(12.5,4.6),gridspec_kw={'width_ratios':[1,1.2,1]})
colors=['#758a9c','#167f8d'];arms=['M1','M3']
rates=[r['summary'][m]['success_rate']*100 for m in arms]
axes[0].bar(arms,rates,color=colors,width=.52)
for i,m in enumerate(arms):
    s=r['summary'][m];axes[0].text(i,rates[i]+2,f"{s['successes']}/{s['count']}\n{rates[i]:.1f}%",ha='center')
axes[0].set_ylim(0,100);axes[0].set_ylabel('Success rate (%)');axes[0].set_title('Fixed dev-210 endpoint')
x=r['M3_minus_M1'];gap=x['gap_percentage_points'];lo,hi=x['paired_95_interval_percentage_points']
axes[1].axvline(0,color='#adb4bd',ls='--',lw=1)
axes[1].errorbar([gap],[0],xerr=[[gap-lo],[hi-gap]],fmt='o',color=colors[1],capsize=6,ms=8)
axes[1].set_yticks([]);axes[1].set_xlabel('M3 − M1 (percentage points)');axes[1].set_title('Paired task bootstrap · 95%')
axes[1].set_xlim(min(-3,lo-2),max(3,hi+2));axes[1].set_ylim(-1,1)
axes[1].text(gap,.3,f'{gap:+.2f} pp\n[{lo:+.2f}, {hi:+.2f}]',ha='center')
lat=[r['summary'][m]['latency']['all_queries'] for m in arms]
positions=np.arange(2);axes[2].bar(positions-.16,[l['median_ms'] for l in lat],width=.3,color=colors,label='Median')
axes[2].bar(positions+.16,[l['p95_ms'] for l in lat],width=.3,color=colors,alpha=.4,label='p95')
axes[2].set_xticks(positions,arms);axes[2].set_ylabel('Milliseconds / 8-action query');axes[2].set_title('Deployed query latency');axes[2].legend(frameon=False,fontsize=9)
axes[2].set_ylim(0,65)
fig.suptitle('M3 (pixel coupled) versus M1 (pixel auxiliary)',fontsize=16,fontweight='bold')
exact=r['input_comparison']['exact_count'];total=r['input_comparison']['count']
fig.text(.5,.02,f'{exact}/{total} exact input pairs · all 210 tasks retained · descriptive development comparison · single training seed',ha='center',fontsize=9,color='#4b5563')
fig.tight_layout(rect=[0,.07,1,.92]);fig.savefig(a.output/'M3-versus-M1-dev210.png',dpi=180,facecolor='white');plt.close(fig)
