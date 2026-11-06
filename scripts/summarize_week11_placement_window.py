"""Bowl retention, release, containment and drawer closure in all twelve runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from review_week11_event_windows import contacts_by_frame
from week11_replay_common import read, sha, write

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ARMS=('fixed8','placement-only','late-only','adaptive')
COLORS=('#777777','#d88617','#326ca8','#2b8c68')


def intervals(mask):
    m=np.asarray(mask,bool).copy();m[:10]=False
    d=np.diff(np.r_[False,m,False].astype(int))
    return [[int(a),int(b)] for a,b in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1),strict=True)]


def run(audit,annotations,output):
    output.mkdir(parents=True,exist_ok=False)
    summary=read(audit/'summary.json');ann=read(annotations/'annotations.json')
    assert read(annotations/'completion.json')['status']=='complete'
    indexed={r['artifact']:r for r in ann['recordings']};rows=[];loaded={}
    for row in summary['rows']:
        rec=indexed[row['artifact']];root=annotations/rec['recording_id']
        for n,h in rec['evidence_sha256'].items():assert sha(root/n)==h
        assert sha(row['artifact'])==rec['source_sha256']
        ev=dict(np.load(root/'evidence.npz'));g=read(root/'geometry.json')
        objects=[o['object_id'] for o in g['objects']]
        bowl_name=next(n for n in objects if 'black_bowl' in n);bi=objects.index(bowl_name)
        inside_index=next(i for i,goal in enumerate(g['goal_atoms']) if goal[0]=='in' and goal[1]==bowl_name)
        joints=[j for j in g['joint_map'] if 'bottom' in (j['name'] or '') and j['type']==2]
        assert len(joints)==1, joints
        qpos=ev['joint_qpos'][:,joints[0]['qpos_address']]
        contacts=contacts_by_frame(g,[json.loads(l) for l in (root/'contacts.jsonl').read_text().splitlines()])
        bilateral=np.array([len(c['finger_sides'].get(bowl_name,[]))==2 for c in contacts])
        command=ev['actions'][:,6];bowl=ev['object_position'][:,bi]
        height=(bowl[:,2]-bowl[10,2])*1000;gripper=ev['gripper_position']
        snapshots={}
        for t in (138,218,298,len(command)-1):
            if t>=len(command):continue
            snapshots[str(t)]={'bowl_position_m':bowl[t].tolist(),'gripper_position_m':gripper[t].tolist(),
                'bowl_height_change_mm':float(height[t]),'bowl_gripper_distance_mm':float(1000*np.linalg.norm(bowl[t]-gripper[t])),
                'aperture_mm':float(1000*np.abs(ev['state'][t,-2:]).sum()),'bilateral_contact':bool(bilateral[t]),
                'command':int(command[t]),'drawer_qpos_m':float(qpos[t]),'goals':ev['goal_values'][t].tolist()}
        openings=(np.flatnonzero((command[:-1]==1)&(command[1:]==-1))+1).tolist()
        closings=(np.flatnonzero((command[:-1]==-1)&(command[1:]==1))+1).tolist()
        rows.append({'arm':row['arm'],'repeat':row['repeat'],'success':row['success'],
            'artifact':row['artifact'],'recording_id':rec['recording_id'],'frames':len(command),
            'opening_steps':openings,'closing_steps':closings,'bilateral_bowl_intervals':intervals(bilateral),
            'goal_atoms':g['goal_atoms'],'goal_true_intervals':[intervals(ev['goal_values'][:,i]) for i in range(len(g['goal_atoms']))],
            'all_goals_true_intervals':intervals(ev['goal_values'].all(1)),
            'drawer_joint':joints[0],'final_drawer_qpos_m':float(qpos[-1]),
            'snapshots':snapshots,'source_sha256':rec['source_sha256'],
            'max_roundtrip_error':rec['max_state_roundtrip_error'],
            'max_fresh_eef_vs_recorded_m':rec['max_fresh_eef_vs_recorded_m'],
            'max_fresh_finger_vs_recorded_m':rec['max_fresh_finger_vs_recorded_m']})
        loaded[row['repeat'],row['arm']]={'row':rows[-1],'ev':ev,'height':height,'drawer':qpos,
                                         'bilateral':bilateral,'inside_index':inside_index}
    for rep in (1,2,3):
        fig,axs=plt.subplots(5,1,figsize=(12,11),sharex=True,layout='constrained')
        for arm,color in zip(ARMS,COLORS,strict=True):
            d=loaded[rep,arm];ev=d['ev'];x=np.arange(10,len(d['height']))
            axs[0].plot(x,d['height'][10:],color=color,label=arm)
            axs[1].plot(x,ev['gripper_position'][10:,2]*1000,color=color)
            axs[2].plot(x,-1000*d['drawer'][10:],color=color)
            i=ARMS.index(arm)
            axs[3].step(x,(ev['actions'][10:,6]+1)/2*.6+i,where='post',color=color)
            axs[4].step(x,ev['goal_values'][10:,d['inside_index']].astype(int)*.6+i,where='post',color=color)
        for ax in axs:
            for x in (138,218,298):ax.axvline(x,color='#555555',alpha=.3,linestyle=':')
            ax.grid(alpha=.2);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        axs[0].set_ylabel('Bowl height change (mm)');axs[0].legend(ncol=4)
        axs[1].set_ylabel('Gripper z (mm)');axs[2].set_ylabel('Drawer opening (mm)')
        for ax in axs[3:]:ax.set_yticks(np.arange(4)+.3,ARMS)
        axs[3].set_ylabel('Gripper command\nupper=close, lower=open')
        axs[4].set_ylabel('Bowl inside drawer\nupper=true, lower=false')
        axs[4].set_xlabel('Recorded pre-action frame (10 settling frames included)')
        fig.suptitle(f'Clean 155 · repeat {rep} · placement 138–217 / late 218–297')
        fig.savefig(output/f'repeat-{rep}-traces.png',dpi=130);plt.close(fig)
        fig,axs=plt.subplots(4,5,figsize=(18,9),layout='constrained')
        for i,arm in enumerate(ARMS):
            row=loaded[rep,arm]['row'];frames=[138,218,min(250,row['frames']-1),min(298,row['frames']-1),row['frames']-1]
            with h5py.File(row['artifact']) as f:
                for ax,t in zip(axs[i],frames,strict=True):
                    ax.imshow(np.concatenate([f['steps/agent_rgb'][t],f['steps/wrist_rgb'][t]],axis=1));ax.axis('off')
                    cmd='close' if f['steps/executed_action'][t,6]==1 else 'open'
                    ax.set_title(f'{arm} · '+('success' if row['success'] else 'failure')+f'\nframe {t} · {cmd}',fontsize=10)
        fig.suptitle(f'Clean 155 · repeat {rep} · saved front and wrist cameras',fontsize=14)
        fig.supxlabel('Frames precede the labeled command. Terminal post-action state is absent. Last column is each run’s final saved frame.',fontsize=10)
        fig.savefig(output/f'repeat-{rep}-cameras.png',dpi=120);plt.close(fig)
    write(output/'physical-summary.json',{'rows':rows,'annotation_digest':ann['digest'],
        'summary_digest':summary['digest'],'simulator_steps':0,'model_calls':0,
        'frame_convention':'Zero-based pre-action frames including settling; half-open intervals; terminal post-action state absent.',
        'qualification':'Supporting reconstructed geometry; original model XML unavailable. Recorder terminal success is authoritative.'})
    write(output/'completion.json',{'status':'complete','physical_summary_digest':read(output/'physical-summary.json')['digest'],
        'figures_sha256':{p.name:sha(p) for p in output.glob('*.png')}})
    print(json.dumps([{k:r[k] for k in ('arm','repeat','success','opening_steps','bilateral_bowl_intervals','goal_true_intervals','final_drawer_qpos_m')} for r in rows]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('audit','annotations','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.audit,a.annotations,a.output)
