"""Render report figures and curate existing, unmodified evidence images."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/eventground-week12-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAVY, TEAL, CORAL, GRAY = '#18344A', '#117C83', '#BE5545', '#687785'
COLORS = ['#687785', '#3974AD', '#117C83', '#8A649C', '#C27A32']
SOURCE = {}
FIGURES = []


def read(path):
    p = ROOT / path
    SOURCE[path] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def save(fig, out, stem, caption, sources):
    for ext in ['png', 'svg', 'pdf']:
        fig.savefig(out/'figures'/f'{stem}.{ext}', dpi=220, facecolor='white', metadata={'Creator': 'EventGround-WAM Week 12'})
    FIGURES.append({'id': stem, 'caption': caption, 'sources': sources,
                    'files': [f'figures/{stem}.{ext}' for ext in ['png', 'svg', 'pdf']]})
    plt.close(fig)


def decorate(ax, xlabel=None):
    ax.grid(axis='x', color='#E2E8EC', linewidth=.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ['top', 'right', 'left']:
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color('#BCC7CF')
    ax.tick_params(axis='y', length=0)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=9)


def header(fig, title, subtitle):
    fig.text(.055, .955, title, fontsize=19, fontweight='bold', color=NAVY, va='top')
    fig.text(.055, .906, subtitle, fontsize=10, color=GRAY, va='top')


def architecture(out):
    fig = plt.figure(figsize=(13, 6.9)); ax = fig.add_axes([.035, .13, .93, .71])
    ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis('off')
    header(fig, 'Future representations and temporal routing',
           'Figure 1  |  Five-arm policy study, followed by a separate three-arm M4 deployment study')
    def box(x,y,w,h,title,body,color=TEAL):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.12',
                                  facecolor='#F5F8FA',edgecolor=color,lw=1.2))
        ax.text(x+w/2,y+h-.22,title,ha='center',va='top',fontsize=11,fontweight='bold',color=color)
        ax.text(x+w/2,y+.17,body,ha='center',va='bottom',fontsize=9,color=NAVY,linespacing=1.45)
    def arrow(a,b,color=GRAY,style='-'):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.3,color=color,linestyle=style))
    box(.1,3.2,2.4,1.1,'Current observation','Instruction · front/wrist RGB\nProprioception')
    box(3.0,3.2,2.5,1.1,'Policy backbone + head','Qwen3.5-2B features\nEight-action proposal')
    box(6.15,3.2,2.5,1.1,'Future core','Proposal conditioned · horizon h\nPixel or JEPA future tokens')
    box(9.35,3.2,3.1,1.1,'Coupling + same action head','Attend to predicted future\nRefine and execute eight actions')
    arrow((2.56,3.75),(2.94,3.75));arrow((5.56,3.75),(6.09,3.75));arrow((8.71,3.75),(9.29,3.75))
    ax.text(7.4,4.55,'M3 / M4 inference path',ha='center',fontsize=12,fontweight='bold',color=NAVY)
    box(.1,.65,5.4,1.55,'Week 9 comparison','B0: action only\nM1 / M2: pixel / JEPA auxiliary training; no future core at inference\nM3 / M4: pixel / JEPA coupling; fixed h = 8',NAVY)
    box(6.15,.65,2.5,1.55,'Week 11 event branch','Selected SFT predictor\nNext new event + onset\nNearest h ∈ {2, 4, 8, 16, 32}',CORAL)
    box(9.35,.65,3.1,1.55,'Week 11 deployment arms','Original M4: no event branch, h = 8\nEvent-running control: h = 8\nAdaptive: event-selected h',CORAL)
    arrow((7.4,2.27),(7.4,3.14),CORAL)
    ax.text(7.58,2.68,'h only',fontsize=9,color=CORAL)
    ax.text(10.9,4.58,'One refinement per query',ha='center',fontsize=10,color=GRAY)
    fig.text(.055,.065,'Coupled policies use detached executable proposals padded to 32 steps. Replanning remains every eight actions at 20 Hz.',fontsize=9,color=GRAY)
    fig.text(.055,.035,'Separate event inference uses the same current observations. No demonstration future action or target enters the rollout action path.',fontsize=9,color=GRAY)
    save(fig,out,'fig01-architecture','Architecture schematic of the implemented controls. Week 9 and Week 11 are distinct studies. Auxiliary futures are training-only; coupled futures enter action refinement. The event branch changes the selected horizon, not the eight-action execution interval.',
         ['docs/WEEK8_COUPLING_INTERFACE.md','docs/WEEK11_EXPERIMENT.md','docs/WEEK11_HORIZON_MAPPING.md'])


def week9(data,out):
    d=data['week9']; fig,axs=plt.subplots(2,2,figsize=(12.8,9.2))
    fig.subplots_adjust(left=.16,right=.85,top=.82,bottom=.16,hspace=.55,wspace=.65)
    header(fig,'No overall coupling benefit is established','Figure 2  |  Week 9: 15,000 outcomes across five fixed policy endpoints')
    arms=['B0','M1','M2','M3','M4']
    for col,(panel,title,lim) in enumerate([('perturbed','Perturbed · n = 1,000 / arm',(68,81)),('clean','Clean · n = 2,000 / arm',(86,100))]):
        p=d['panels'][panel];ax=axs[0,col]
        for y,arm,c in zip(range(4,-1,-1),arms,COLORS):
            r=p['factorial']['success_rates'][arm];x=r['estimate']*100;lo,hi=np.array(r['percentile_95'])*100
            ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt='o',color=c,capsize=4,lw=1.7,ms=7)
            s=p['summaries'][arm]
            ax.text(1.04,y,f'{x:.2f}%\n{s["successes"]:,}/{s["count"]:,}',transform=ax.get_yaxis_transform(),va='center',fontsize=8.5,color=c)
        ax.set(yticks=range(4,-1,-1),yticklabels=arms,xlim=lim,ylim=(-.6,4.6))
        ax.set_title(title,loc='left',fontweight='bold',pad=12);decorate(ax,'Success rate (%) · 95% interval')
        ax=axs[1,col]
        for y,key,label in [(2,'latent_minus_pixel_main_effect','Latent − pixel'),(1,'coupling_main_effect','Coupling'),(0,'interaction','Interaction')]:
            r=p['factorial']['contrasts_percentage_points'][key];x=r['estimate'];lo,hi=r['percentile_95']
            ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt='o',color=NAVY,capsize=4,lw=1.7,ms=7)
            ax.text(1.04,y,f'{x:+.2f} pp',transform=ax.get_yaxis_transform(),va='center',fontsize=9)
        ax.axvline(0,color=GRAY,ls='--',lw=1);ax.set(yticks=[2,1,0],yticklabels=['Latent − pixel','Coupling','Interaction'],xlim=(-6,8),ylim=(-.6,2.6))
        ax.set_title('Paired factorial effects',loc='left',fontweight='bold',pad=12);decorate(ax,'Difference (percentage points)')
    fig.text(.055,.075,'All factorial intervals include zero. Contributing RGB mismatches keep these effects descriptive.',fontsize=10,color=NAVY)
    fig.text(.055,.045,'10,000 shared PCG64 draws: variant tasks (seed 7); 40 clean task clusters × 50 states (seed 8). One training seed; dev overlap retained.',fontsize=8.5,color=GRAY)
    save(fig,out,'fig02-five-arm-comparison','Week 9 success rates and factorial effects with frozen 95% bootstrap intervals. Clean uncertainty resamples canonical tasks, retaining all 50 states. Affected comparisons remain descriptive; one training seed, development overlap and the JEPA research-use exception limit generalization.',[d['source']])


def formal(data,out):
    d=data['week11'];fig,axs=plt.subplots(2,2,figsize=(12.6,8.8))
    fig.subplots_adjust(left=.14,right=.95,top=.82,bottom=.17,wspace=.45,hspace=.6)
    header(fig,'Adaptive routing fails the original performance gates','Figure 3  |  Week 11 formal integration: 2,001 episodes; three contemporaneous arms')
    arms=['M4-fixed','M4-EG-fixed8','M4-EG-adaptive'];labels=['Original fixed8','Event-running fixed8','Adaptive'];cols=[GRAY,TEAL,CORAL]
    for i,(panel,title) in enumerate([('targeted-467','Targeted · n = 467 / arm'),('clean-200','Clean · n = 200 / arm')]):
        ax=axs[0,i]
        for y,arm,label,c in zip([2,1,0],arms,labels,cols):
            s=d['summaries'][arm+'/'+panel];rate=s['success_rate']*100
            ax.barh(y,rate,color=c,height=.5)
            ax.text(rate+1,y,f'{rate:.2f}%  ({s["successes"]}/{s["count"]})',va='center',fontsize=9)
        ax.set(yticks=[2,1,0],yticklabels=labels,xlim=(0,124),ylim=(-.6,2.6),xticks=[0,25,50,75,100])
        ax.set_title(title,loc='left',fontweight='bold',pad=12);decorate(ax,'Observed success rate (%)')
    ax=axs[1,0]
    for y,panel,label in [(1,'targeted-467','Targeted'),(0,'clean-200','Clean')]:
        s=d['statistics'][panel]['M4-EG-adaptive minus M4-fixed'];x=s['difference_pp'];lo,hi=s['paired_95_ci_pp']
        ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt='o',color=CORAL,ms=8,lw=2,capsize=5)
        ax.text(x,y+.23,f'{x:+.2f} [{lo:+.2f}, {hi:+.2f}]',ha='center',fontsize=9)
    ax.axvline(0,ls='--',color=GRAY,lw=1);ax.set(yticks=[1,0],yticklabels=['Targeted','Clean'],xlim=(-13,3),ylim=(-.5,1.7))
    ax.set_title('Adaptive minus original fixed8',loc='left',fontweight='bold');decorate(ax,'Paired difference (pp) · 95% interval')
    ax=axs[1,1]
    for offset,panel,marker in [(.14,'targeted-467','o'),(-.14,'clean-200','s')]:
        vals=[d['summaries'][a+'/'+panel]['latency']['all_queries']['p95_ms'] for a in arms]
        ax.scatter(vals,np.array([2,1,0])+offset,c=cols,s=45,marker=marker,label='Targeted' if offset>0 else 'Clean')
        for v,y in zip(vals,np.array([2,1,0])+offset):ax.text(v+2,y,f'{v:.2f}',fontsize=8,va='center')
    ax.set(yticks=[2,1,0],yticklabels=labels,xlim=(0,137),ylim=(-.5,2.65));ax.legend(loc='upper left',fontsize=8,frameon=False)
    ax.set_title('Rollout p95 latency',loc='left',fontweight='bold');decorate(ax,'Milliseconds per eight-action query')
    fig.text(.055,.085,'Fixed controls match all 667 binary outcomes and 16,500 identical-input action queries; full trajectories can differ with rendering.',fontsize=9,color=NAVY)
    fig.text(.055,.055,'Targeted bootstrap: 467 variants. Clean: 40 task clusters × 5 states. The clean interval reaches zero. These are reused panels, one training seed.',fontsize=8.5,color=GRAY)
    fig.text(.055,.028,'Latency includes every recorded rollout query, including first and failed queries. Engineering cost gates pass; performance gates fail.',fontsize=8.5,color=GRAY)
    save(fig,out,'fig03-adaptive-integration','Formal three-arm comparison. The event-running fixed8 control matches fixed M4 binary outcomes; adaptive routing loses 32 targeted and 10 clean successes net. Paired intervals use the frozen resampling units. Clean includes zero; latency does not compensate for the failed performance gates.',['docs/evaluation/week11-closeout-v1/remote/analysis.json'])


def event(data,out):
    d=data['week10'];fig,axs=plt.subplots(1,3,figsize=(14,5.7),gridspec_kw={'width_ratios':[1.5,1,1.05]})
    fig.subplots_adjust(left=.20,right=.96,top=.76,bottom=.25,wspace=.65)
    header(fig,'Event forecasting remains exploratory weak-label evidence','Figure S1  |  Revised next-new-onset task; demonstration forecasts are separate from deployed robot success')
    ax=axs[0];keys=['sft','primary','calibration','hybrid','control'];labels=['Starting SFT','RLVR primary','RLVR calibration','RLVR + preservation','SFT continuation']
    for y,k in zip(range(4,-1,-1),keys):
        m=d['dev'][k];c=TEAL if k=='control' else GRAY
        ax.scatter(m['macro_positive_iou']*100,y,color=c,s=55)
        ax.text(m['macro_positive_iou']*100+.25,y,f'{m["macro_positive_iou"]*100:.2f}% · FP {m["none_false_positives"]}/{m["none_count"]}',fontsize=8,va='center')
    ax.set(yticks=range(4,-1,-1),yticklabels=labels,xlim=(39,50),ylim=(-.6,4.6));ax.set_title('Development · 396 prompts\n103 physical sources',loc='left',fontweight='bold',fontsize=11);decorate(ax,'Macro positive IoU (%)')
    ax=axs[1]
    for y,k in [(1,'sft'),(0,'control')]:
        m=d['report'][k];v=m['macro_positive_iou']*100
        ax.scatter(v,y,color=GRAY if k=='sft' else TEAL,s=55);ax.text(v+.3,y,f'{v:.2f}%',fontsize=9,va='center')
    ax.set(yticks=[1,0],yticklabels=['Starting SFT','Continuation'],xlim=(32,41),ylim=(-.6,1.6));ax.set_title('Report · 384 prompts\n80 physical sources',loc='left',fontweight='bold',fontsize=11);decorate(ax,'Macro positive IoU (%)')
    ax=axs[2];r=d['report_comparison'];x=r['delta_iou']*100;lo,hi=np.array(r['bootstrap']['interval95'])*100
    ax.errorbar(x,0,xerr=[[x-lo],[hi-x]],fmt='o',color=TEAL,ms=8,capsize=5,lw=2);ax.axvline(0,ls='--',lw=1,color=GRAY)
    ax.text(x,.25,f'{x:+.2f} [{lo:+.2f}, {hi:+.2f}]',ha='center',fontsize=9)
    ax.set(yticks=[],ylim=(-.8,.8),xlim=(-3,5));ax.set_title('Report change\nContinuation − starting SFT',loc='left',fontweight='bold',fontsize=11);decorate(ax,'IoU difference (pp) · 95% CI')
    fig.text(.055,.13,'The report interval spans zero. Report none false positives stay at 3/96. SFT continuation was selected on development before report scoring.',fontsize=9,color=NAVY)
    fig.text(.055,.085,'No RLVR report arm was added. Automatic targets, adaptive development reuse, prior report exposure and unresolved independent label QA qualify these results.',fontsize=8.5,color=GRAY)
    fig.text(.055,.045,'Report interval: 10,000 paired physical-source draws, PCG64 seed 10. The earlier 48-prompt task has different targets and remains separate.',fontsize=8.5,color=GRAY)
    save(fig,out,'figS01-event-forecasting','Revised-task development scores and the separately evaluated SFT continuation. The report comparison uses the existing source-bootstrap interval. Point scores on development do not establish superiority; old-task scores and robot outcomes are not pooled.',['docs/evaluation/week10-sft-continuation-report-v1/analysis/results.json','docs/evaluation/week10-preservation-pilot-v1/analysis/results.json'])


def followups(data,out):
    d=data['followups'];fig,axs=plt.subplots(1,3,figsize=(13.6,6.2),gridspec_kw={'width_ratios':[1.1,1,1.3]})
    fig.subplots_adjust(left=.09,right=.97,top=.77,bottom=.28,wspace=.7)
    header(fig,'Later pilots do not establish a replacement for original M4','Figure S2  |  Exploratory follow-ups shown as separate cohorts; no pooled success rate')
    ax=axs[0];x=np.arange(3)
    for off,route,c in [(-.16,'fixed8',TEAL),(.16,'adaptive',CORAL)]:
        vals=[d['continuation']['decision']['counts'][k+'/'+route] for k in ['original','h8-continuation','mixed-continuation']]
        ax.bar(x+off,vals,width=.3,color=c,label=route)
        for xx,v in zip(x+off,vals):ax.text(xx,v+.35,str(v),ha='center',fontsize=9)
    ax.set(xticks=x,xticklabels=['Original','H8','Mixed'],ylim=(0,27),ylabel='Successful cases');ax.set_title('Matched continuation\n27 selected cases per condition',fontsize=11,loc='left',fontweight='bold');ax.legend(frameon=False,fontsize=8)
    ax=axs[1];keys=['h8-fixed8','horizon-motion','h8-information-control'];vals=[d['adapter']['counts'][k]['successes'] for k in keys]
    ax.bar(range(3),vals,color=[GRAY,CORAL,TEAL],width=.55)
    for i,v in enumerate(vals):ax.text(i,v+.2,f'{v}/9',ha='center',fontsize=10)
    ax.set(xticks=range(3),xticklabels=['H8','Motion','Control'],ylim=(0,9),ylabel='Successful cases');ax.set_title('Adapter screen · stopped\n27 / 81 planned episodes',fontsize=11,loc='left',fontweight='bold')
    ax=axs[2]
    for y,panel,label in [(1,'targeted-467','Targeted'),(0,'clean-200','Clean')]:
        p=d['h8']['panels'][panel]['all_outcomes'];g=d['h8_task_sensitivity']['panels'][panel]
        x=p['gap_percentage_points'];lo,hi=g['suite_stratified_task_cluster_bootstrap_95_percentage_points']
        ax.errorbar(x,y,xerr=[[x-lo],[hi-x]],fmt='o',color=NAVY,ms=7,lw=1.8,capsize=4)
        ax.text(x,y+.24,f'{x:+.2f} [{lo:+.2f}, {hi:+.2f}]',ha='center',fontsize=8)
    ax.axvline(0,ls='--',color=GRAY,lw=1);ax.set(yticks=[1,0],yticklabels=['Targeted','Clean'],xlim=(-22,13),ylim=(-.5,1.7));decorate(ax,'H8 − vanilla (pp) · grouped 95% CI')
    ax.set_title('Fresh H8 comparison\n32 cases per panel, two policies',fontsize=11,loc='left',fontweight='bold')
    fig.text(.055,.165,'Continuation: mixed fixed8 loses two vs H8, failing preservation. Adapter: p95 exceeds the frozen 2× ceiling; remaining repeats stay unrun.',fontsize=9,color=NAVY)
    fig.text(.055,.115,'Fresh H8: 22 vs 24 targeted successes; 29 vs 29 clean. Targeted contains 24 base tasks; plotted intervals retain the task-cluster sensitivity analysis.',fontsize=8.5,color=GRAY)
    fig.text(.055,.065,'Fresh rollouts use previously evaluated task pools and one training seed. Case-specific gains do not establish general improvement, inferiority or equivalence.',fontsize=8.5,color=GRAY)
    save(fig,out,'figS02-exploratory-followups','Exploratory follow-ups retain their distinct controls and panels. Selected continuation counts and the incomplete adapter screen are not estimates of general benchmark performance. Fresh H8 effects use the supplementary whole-base-task bootstrap within suites; the frozen variant intervals remain in the tables.',['docs/evaluation/week11-multihorizon-results-v1/summary.json','docs/evaluation/week11-motion-screen-v1/status.json','docs/evaluation/week11-h8-validation-v1/base-task-sensitivity.json'])


def curate_examples(out):
    examples=[];base='docs/evaluation/week11-closeout-v1/'
    selected=read(base+'remote/selected-case-index.json');presentation=read(base+'presentation-v2/presentation.json')
    matrix=read(base+'remote/paired-matrices.json')
    main={('targeted-467','adaptive_only_success'):'fig04a-targeted-gain',('targeted-467','fixed_only_success'):'fig04b-targeted-loss',
          ('clean-200','adaptive_only_success'):'fig04c-clean-gain',('clean-200','fixed_only_success'):'fig04d-clean-loss'}
    for p in presentation['snapshots']:
        panel,kind=p['panel'],p['outcome_class'];row=selected['panels'][panel][kind];ordinal=row['ordinal']
        values=matrix[panel]['outcomes'];pred=lambda r: ('both_success' if r[0] and r[2] else 'fixed_only_success' if r[0] else 'adaptive_only_success' if r[2] else 'both_failure')
        assert ordinal==next(i for i,r in enumerate(values) if pred(r)==kind)
        path=base+'presentation-v2/figures/'+p['file'];src=ROOT/path
        digest=hashlib.sha256(src.read_bytes()).hexdigest();assert digest==presentation['files_sha256'][p['file']]
        SOURCE[path]=digest
        name=main.get((panel,kind),f'{panel}-{kind}')+'.png';dest=out/'examples'/name;shutil.copyfile(src,dest)
        caption=(f'{panel}, ordinal {ordinal:04d}: {kind.replace("_", " ")}. First manifest ordinal in this paired-outcome class. '
                 'Rows show adaptive, event-running fixed8 and original fixed8. Columns show each trajectory’s first query, midpoint and last recorded pre-action observation; times differ across arms. '
                 'Each image pairs front and wrist cameras. Outcomes come from the audited episode record; snapshots alone do not identify failure causes.')
        e={'id':f'{panel}-{ordinal:04d}','group':'formal paired outcomes','selection':selected['selection'],
           'outcome_class':kind,'task':row['task'],'outcomes':dict(zip(matrix[panel]['arms'],values[ordinal])),
           'media':'examples/'+name,'source':path,'sha256':digest,'recorded_steps':p['records'],'caption':caption}
        examples.append(e)
        if (panel,kind) in main:FIGURES.append({'id':main[panel,kind],'caption':caption,'sources':[path,base+'remote/selected-case-index.json'],'files':[e['media']]})
    mech='docs/evaluation/week11-h8-validation-v1/inspection-mirror/inspection-focus/'
    receipt=read(mech+'completion.json')
    captions={
        'clean-200-0107':'Bowl retention gain: both adapters retain the bowl and succeed, while fixed H8 loses it despite close commands. The H8-information control also reproduces the gain, so this does not isolate non-8 future information.',
        'clean-200-0151':'Recovery regression: fixed H8 recovers a late pot grasp and succeeds. Neither adapter obtains bilateral pot contact; stove activation alone is insufficient.',
        'targeted-467-0231':'Wrong-fixture engagement: the motion adapter pulls the bottom drawer while the matched control opens the requested middle drawer. Gripper commands agree through their shared 181-frame span.'}
    for key,caption in captions.items():
        name=key+'-matched-window.png';path=mech+name;digest=hashlib.sha256((ROOT/path).read_bytes()).hexdigest();assert digest==receipt['figures_sha256'][name]
        SOURCE[path]=digest;shutil.copyfile(ROOT/path,out/'examples'/name)
        examples.append({'id':key,'group':'exploratory mechanism cases','selection':receipt['selection'],
                         'media':'examples/'+name,'source':path,'sha256':digest,'recorded_steps':receipt['windows'][key],
                         'caption':caption+' Post-hoc, case-specific inspection; not a representative failure frequency.',
                         'interpretation_source':'docs/evaluation/week11-h8-validation-v1/INSPECTION.md'})
    evbase='docs/evaluation/week10-closeout-v1/';events=read(evbase+'examples.json');media=read(evbase+'media-verification.json')
    for key in ['case-05','case-09','case-13']:
        c=next(c for c in events['cases'] if c['case_id']==key);frames=[]
        for offset in [0,16,32]:
            for camera in ['front','wrist']:
                name=f'{key}-{camera}-{offset:02d}.png';relative='frames/'+name;path=evbase+relative
                digest=hashlib.sha256((ROOT/path).read_bytes()).hexdigest();assert digest==media['frame_files_sha256'][relative]
                SOURCE[path]=digest;shutil.copyfile(ROOT/path,out/'examples'/name)
                frames.append({'media':'examples/'+name,'camera':camera,'offset':offset,'absolute_frame':c['row']['anchor']+offset,'sha256':digest,'source':path})
        pred=c['predictions']['control'];target=c['row']['target']
        caption=(f'{c["categories"][0]}. Selected SFT continuation predicts {pred["prediction"]["next_event"]} {pred["prediction"]["interval"]}; '
                 f'automatic target is {target["next_event"]} {target["interval"]}. Typed IoU {pred["positive_iou"]:.3f}. '
                 'Intervals are relative to the anchor; endpoint 33 denotes the clipped window boundary. Future frames are retrospective display only. '
                 'Automatic predicate agreement is not independent event truth or deployed-policy accuracy.')
        examples.append({'id':key,'group':'exploratory event forecasts','selection':'One previously selected type-confusion case, one timing-error case and one correct-interval case; chosen after results, not representative.',
                         'row':c['row'],'prediction':pred,'frames':frames,'caption':caption,'source':evbase+'examples.json'})
    for path in ['docs/evaluation/week11-h8-validation-v1/INSPECTION.md','docs/WEEK11_EXPERIMENT.md','docs/WEEK11_HORIZON_MAPPING.md','docs/WEEK8_COUPLING_INTERFACE.md']:
        SOURCE[path]=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
    (out/'examples.json').write_text(json.dumps({'examples':examples,'image_policy':'Copied existing source PNG bytes unchanged; no image synthesis, crops, retouching or simulator rendering.'},indent=2,sort_keys=True)+'\n')
    return examples


def gallery(out,examples):
    esc=html.escape
    parts=['<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
           '<title>EventGround-WAM final figures and examples</title><style>body{font:16px/1.6 system-ui,sans-serif;color:#18344a;max-width:1180px;margin:auto;padding:36px 28px;background:#fbfcfd}h1{font-size:32px;line-height:1.2}h2{margin-top:48px;border-top:1px solid #d9e3e8;padding-top:24px}h3{margin-bottom:10px}img{max-width:100%;height:auto;background:white;border:1px solid #e3e9ed}figure{margin:24px 0 42px}figcaption{max-width:1000px;margin-top:12px;color:#475a68}.frames{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}.frames figure{margin:0}a{color:#08757c}nav{display:flex;gap:24px;flex-wrap:wrap}small{color:#687785}.note{background:#edf4f6;padding:18px 24px;border-left:4px solid #117c83}code{font-size:13px}details{margin:12px 0} @media(max-width:700px){.frames{grid-template-columns:1fr 1fr}}</style>',
           '<h1>EventGround-WAM<br>Final figures and qualitative evidence</h1><p>Week 12 · Steps 2–3 · Existing audited evidence</p>',
           '<nav><a href="#figures">Figures</a><a href="#formal">Formal examples</a><a href="#mechanisms">Mechanism cases</a><a href="#events">Event forecasts</a><a href="RESULTS.md">Consolidated results</a></nav>',
           '<p class="note">Studies and panels remain separate. Formal means evaluation under a frozen protocol, not an untouched confirmatory test. All studies use one training seed. Selected examples illustrate behavior; they do not estimate its frequency.</p><h2 id="figures">Statistical figures and architecture</h2>']
    for f in FIGURES:
        if f['id'].startswith('fig04'):continue
        png=next(x for x in f['files'] if x.endswith('.png'))
        parts.append(f'<figure><h3>{esc(f["id"])}</h3><a href="{png}"><img src="{png}" alt="{esc(f["caption"])}"></a><figcaption>{esc(f["caption"])}<br>')
        parts.append(' · '.join(f'<a href="{p}">{Path(p).suffix[1:].upper()}</a>' for p in f['files'])+'</figcaption></figure>')
    for group,anchor,title in [('formal paired outcomes','formal','Formal paired outcomes'),('exploratory mechanism cases','mechanisms','Exploratory mechanism cases'),('exploratory event forecasts','events','Exploratory event forecasts')]:
        parts.append(f'<h2 id="{anchor}">{title}</h2>')
        for e in examples:
            if e['group']!=group:continue
            parts.append(f'<figure id="{esc(e["id"])}"><h3>{esc(e["id"])}</h3>')
            if 'media' in e:parts.append(f'<a href="{e["media"]}"><img loading="lazy" src="{e["media"]}" alt="{esc(e["caption"])}"></a>')
            else:
                parts.append('<div class="frames">')
                for frame in sorted(e['frames'],key=lambda f:(f['camera'],f['offset'])):
                    parts.append(f'<figure><img loading="lazy" src="{frame["media"]}" alt="{esc(e["id"])} {frame["camera"]} frame {frame["absolute_frame"]}"><small>{frame["camera"]} · +{frame["offset"]} · frame {frame["absolute_frame"]}</small></figure>')
                parts.append('</div>')
            parts.append(f'<figcaption>{esc(e["caption"])}<details><summary>Selection and provenance</summary><p>{esc(e["selection"])}</p><code>{esc(e["source"])}</code></details></figcaption></figure>')
    parts.append('<footer><p>Exact source paths, SHA-256 hashes, case identities and recorded steps are preserved in <a href="examples.json">examples.json</a> and <a href="source-manifest.json">source-manifest.json</a>.</p></footer></html>')
    (out/'gallery.html').write_text('\n'.join(parts))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);out=p.parse_args().output.resolve()
    assert (out/'week12-build.json').exists()
    d=json.loads((out/'consolidated.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'text.color':NAVY,'axes.labelcolor':NAVY,
                         'xtick.color':GRAY,'ytick.color':NAVY,'svg.fonttype':'none','pdf.fonttype':42})
    architecture(out);week9(d,out);formal(d,out);event(d,out);followups(d,out)
    examples=curate_examples(out);gallery(out,examples)
    (out/'figure-index.json').write_text(json.dumps({'figures':FIGURES},indent=2,sort_keys=True)+'\n')
    captions=['# Final figure captions','']
    for f in FIGURES:
        captions.extend([f'## {f["id"]}','',f['caption'],'', ' · '.join(f'[{Path(p).suffix[1:].upper()}]({p})' for p in f['files']),''])
    (out/'CAPTIONS.md').write_text('\n'.join(captions))
    manifest=json.loads((out/'source-manifest.json').read_text());manifest['files_sha256'].update(SOURCE)
    manifest['files_sha256']['scripts/render_week12_figures.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'source-manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    shutil.copyfile(__file__,out/'source/render_week12_figures.py')
    print(json.dumps({'figures':len(FIGURES),'examples':len(examples),'media_source_files':len(SOURCE),'gallery':str(out/'gallery.html')}))


if __name__=='__main__':main()
