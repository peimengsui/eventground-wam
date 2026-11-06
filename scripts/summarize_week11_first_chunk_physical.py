"""Dense saved-state process evidence and existing camera views for four-arm cases."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from review_week11_event_windows import contacts_by_frame
from week11_replay_common import read, sha, write

from eventground_wam.week11_first_chunk import ARMS

matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS=('Fixed8','After closing','Skip first change','First change only')
COLORS=('#246b91','#bd5838','#329364','#785bb0')


def runs(mask):
    d=np.diff(np.r_[False,np.asarray(mask,bool),False].astype(int))
    return [[int(a),int(b)] for a,b in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1),strict=True)]


def frames(case,loaded,steps,title,output):
    fig,axes=plt.subplots(4,6,figsize=(18,14.4),layout='constrained')
    for i,arm in enumerate(ARMS):
        status='success' if case['outcomes'][arm] else 'failure'
        with h5py.File(case['arms'][arm]['artifact']) as f:
            for ax,s in zip(axes[i],steps[i],strict=True):
                ax.imshow(np.concatenate([f['steps/agent_rgb'][s],f['steps/wrist_rgb'][s]],axis=0));ax.axis('off')
                cmd=loaded[arm]['ev']['actions'][s,6]
                ax.set_title(f'{LABELS[i]} · {status}\nstep {s} · {(s-10)/20:.2f}s · '+('close' if cmd==1 else 'open'),fontsize=9)
    fig.suptitle(case['id']+' · '+title,fontsize=14)
    fig.supxlabel('Existing cameras: agent above wrist. Saved frame precedes labeled command. Overview rows sample their own durations; activation rows share absolute steps.',fontsize=9)
    fig.savefig(output/f'{case["id"]}-{title}.png',dpi=130);plt.close(fig)


def plot(case,loaded,output):
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True,layout='constrained')
    chosen=[(i,o['object_id']) for i,o in enumerate(loaded[ARMS[0]]['g']['objects'])
            if any(k in o['object_id'] for k in ('black_bowl_1','alphabet_soup','tomato_sauce'))]
    for ai,arm in enumerate(ARMS):
        ev=loaded[arm]['ev'];t=(np.arange(10,len(ev['actions']))-10)/20
        axes[0].plot(t,ev['gripper_position'][10:,2],color=COLORS[ai],label=LABELS[ai])
        axes[1].plot(t,1000*np.abs(ev['state'][10:,-2:]).sum(1),color=COLORS[ai])
        for oi,(index,name) in enumerate(chosen):
            axes[2].plot(t,1000*(ev['object_position'][10:,index,2]-ev['object_position'][10,index,2]),
                color=COLORS[ai],linestyle=('-','--')[oi%2],label=LABELS[ai]+' '+name)
    for ax in axes:
        ax.axvline((case['intervention_step']-10)/20,color='#444',linestyle=':');ax.grid(alpha=.2)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('Gripper z (m)');axes[0].legend(fontsize=9,ncol=2)
    axes[1].set_ylabel('Finger aperture (mm)');axes[2].set_ylabel('Object height change (mm)')
    axes[2].legend(fontsize=8,ncol=2);axes[2].set_xlabel('Seconds after first policy query')
    fig.suptitle(case['id']+' · four-arm physical traces',fontsize=14)
    fig.savefig(output/f'{case["id"]}-motion.png',dpi=140);plt.close(fig)


def run(summary,annotations,output):
    start=time.monotonic();summary=read(summary);ann=read(annotations/'annotations.json')
    indexed={x['artifact']:x for x in ann['recordings']};output.mkdir(parents=True,exist_ok=False);results=[]
    for case in summary['rows']:
        loaded={};arms={}
        for arm in ARMS:
            path=case['arms'][arm]['artifact'];row=indexed[path];root=annotations/row['recording_id']
            assert sha(path)==row['source_sha256']
            for name,digest in row['evidence_sha256'].items():assert sha(root/name)==digest
            ev=dict(np.load(root/'evidence.npz'));g=read(root/'geometry.json');candidate=read(root/'candidate.json')
            contact=contacts_by_frame(g,[json.loads(l) for l in (root/'contacts.jsonl').read_text().splitlines()])
            loaded[arm]={'ev':ev,'g':g};objects={};names=[o['object_id'] for o in g['objects']]
            for i,name in enumerate(names):
                p=ev['object_position'][:,i];above=p[:,2]-p[10,2]>.03;above[:10]=False
                obj={'initial_position_m':p[10].tolist(),'last_position_m':p[-1].tolist(),
                    'max_height_change_mm':float(1000*(p[10:,2]-p[10,2]).max()),
                    'max_height_step':int(np.argmax(p[10:,2])+10),
                    'above_30mm_intervals':runs(above),'gripper_contact_intervals':runs([name in c['movable_objects'] for c in contact])}
                if name=='akita_black_bowl_1' and 'plate_1' in names:
                    plate=ev['object_position'][:,names.index('plate_1')];dist=np.linalg.norm(p[:,:2]-plate[:,:2],axis=1)*1000
                    obj['plate_xy_distance_mm']={'minimum_after_settling':float(dist[10:].min()),'last':float(dist[-1]),
                        'at_104':float(dist[104]) if len(dist)>104 else None,
                        'frames_below_30mm':np.flatnonzero((np.arange(len(dist))>=10)&(dist<30)).tolist()}
                objects[name]=obj
            commands=ev['actions'][:,6]
            arms[arm]={'success':case['outcomes'][arm],'frames':len(commands),'objects':objects,
                'closing_steps':(np.flatnonzero((commands[:-1]==-1)&(commands[1:]==1))+1).tolist(),
                'opening_steps':(np.flatnonzero((commands[:-1]==1)&(commands[1:]==-1))+1).tolist(),
                'candidate_events':candidate['events'],'automatic_labels_unreviewed':True}
        results.append({'id':case['id'],'intervention_step':case['intervention_step'],'arms':arms})
        overview=[np.rint(np.linspace(10,len(loaded[a]['ev']['actions'])-1,12)).astype(int).tolist() for a in ARMS]
        for page in range(2):frames(case,loaded,[s[page*6:(page+1)*6] for s in overview],f'overview-{page+1}',output)
        end=min(len(loaded[a]['ev']['actions'])-1 for a in ARMS);d=case['intervention_step']
        steps=[min(d+x,end) for x in (0,4,8,16,24,32)]
        frames(case,loaded,[steps]*4,'activation',output);plot(case,loaded,output)
    write(output/'physical-summary.json',{'rollout_summary_digest':summary['digest'],'annotation_digest':ann['digest'],
        'cases':results,'height_metric_start_step':10,'contacts_half_open_intervals':True,
        'model_calls':0,'simulator_steps':0,'independent_human_qa':False})
    write(output/'completion.json',{'status':'complete','seconds':time.monotonic()-start,'source_sha256':sha(__file__),
        'physical_summary_digest':read(output/'physical-summary.json')['digest'],'figures_sha256':{p.name:sha(p) for p in output.glob('*.png')}})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--summary',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.summary,a.annotations,a.output)
