"""Compare nine saved motion-adapter recordings with unchanged physical metrics."""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np
from review_week11_event_windows import contacts_by_frame
from week11_replay_common import read, sha, write

matplotlib.use('Agg')
import matplotlib.pyplot as plt

CONDITIONS = ('h8-fixed8', 'horizon-motion', 'h8-information-control')
LABELS = ('H8 fixed8', 'Horizon-aware', 'H8-information control')
COLORS = ('#246b91','#b75222','#28835d')
STYLES = ('-','-','-')


def runs(mask):
    m=np.asarray(mask,bool).copy();m[:10]=False
    d=np.diff(np.r_[False,m,False].astype(int))
    return [[int(a),int(b)] for a,b in zip(np.flatnonzero(d==1),np.flatnonzero(d==-1),strict=True)]


def first_difference(a,b):
    n=min(len(a),len(b));a,b=a[:n],b[:n]
    d=np.flatnonzero(np.any(a!=b,axis=tuple(range(1,a.ndim))) if a.ndim>1 else a!=b)
    return int(d[0]) if len(d) else None


def sheet(case,loaded,steps,name,output):
    columns=len(steps[0]);fig,axes=plt.subplots(len(CONDITIONS),columns,figsize=(3.6*columns,6.7),layout='constrained')
    for i,condition in enumerate(CONDITIONS):
        with h5py.File(case['artifacts'][condition]) as f:
            for ax,s in zip(axes[i],steps[i],strict=True):
                ax.imshow(np.concatenate([f['steps/agent_rgb'][s],f['steps/wrist_rgb'][s]],axis=1))
                ax.axis('off');cmd=loaded[condition]['ev']['actions'][s,6]
                ax.set_title(f'{LABELS[i]} · '+('S' if case['outcomes'][condition] else 'F')+
                    f' · step {s} · {(s-10)/20:.2f}s\n'+('close' if cmd==1 else 'open'),fontsize=9)
    fig.suptitle(case['id']+' · '+name+'\n'+loaded[CONDITIONS[0]]['meta']['instruction'],fontsize=13)
    fig.supxlabel('Saved cameras: front left, wrist right. Frame precedes labeled executed command. Overview samples each run independently; matched sheets share absolute steps. S/F are recorded episode outcomes.',fontsize=9)
    fig.savefig(output/f'{case["id"]}-{name}.png',dpi=120);plt.close(fig)


def trace(case,loaded,output):
    fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True,layout='constrained')
    geom=loaded[CONDITIONS[0]]['g']
    chosen=[(i,o['object_id']) for i,o in enumerate(geom['objects']) if o['movable_free_object']
            and any(s in o['object_id'] for s in ('black_bowl','tomato_sauce','moka_pot','wine_bottle'))]
    for i,c in enumerate(CONDITIONS):
        ev=loaded[c]['ev'];x=(np.arange(10,len(ev['actions']))-10)/20
        kw={'color':COLORS[i],'linestyle':STYLES[i]}
        axes[0].plot(x,ev['gripper_position'][10:,2],label=LABELS[i],**kw)
        axes[1].plot(x,1000*np.abs(ev['state'][10:,-2:]).sum(1),**kw)
        for oi,obj in chosen:
            axes[2].plot(x,1000*(ev['object_position'][10:,oi,2]-ev['object_position'][10,oi,2]),label=LABELS[i]+' '+obj,**kw)
        # Each condition has its own row; goal atoms retain separate offsets.
        for j,atom in enumerate(geom['goal_atoms']):
            y=ev['goal_values'][10:,j].astype(float)*.55+i*1.5+j*.15
            axes[3].step(x,y,where='post',label=LABELS[i]+' '+atom[0],**kw)
    for ax in axes:
        ax.grid(alpha=.2);ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    axes[0].set_ylabel('Gripper z (m)');axes[0].legend(ncol=2,fontsize=8)
    axes[1].set_ylabel('Finger aperture (mm)');axes[2].set_ylabel('Target height change (mm)')
    axes[2].legend(fontsize=7,ncol=2);axes[3].set_ylabel('Saved-state goal flags\n(row offsets)')
    axes[3].legend(fontsize=7,ncol=2);axes[3].set_xlabel('Seconds after first policy query')
    fig.suptitle(case['id']+' · physical traces and reconstructed task components',fontsize=13)
    fig.savefig(output/f'{case["id"]}-traces.png',dpi=130);plt.close(fig)


def run(bundle,annotations,output):
    start=time.monotonic();cfg=read(bundle/'contract.json');ann=read(annotations/'annotations.json')
    assert read(annotations/'contract.json')['replay_contract_digest']==cfg['digest']
    indexed={r['artifact']:r for r in ann['recordings']};output.mkdir(parents=True,exist_ok=False);results=[]
    for case in cfg['selection']['pairs']:
        loaded={};arms={}
        for c,path in case['artifacts'].items():
            assert sha(path)==cfg['input_sha256'][path]
            qpath=str(Path(path).with_suffix('.queries.jsonl'));assert sha(qpath)==cfg['input_sha256'][qpath]
            receipt=indexed[path];root=annotations/receipt['recording_id']
            for n,h in receipt['evidence_sha256'].items():assert sha(root/n)==h
            ev=dict(np.load(root/'evidence.npz'));g=read(root/'geometry.json')
            contacts=contacts_by_frame(g,[json.loads(line) for line in (root/'contacts.jsonl').read_text().splitlines()])
            queries=[json.loads(line) for line in Path(qpath).read_text().splitlines()]
            with h5py.File(path) as f:
                meta=json.loads(f['episode/metadata_json'][()]);raw={k:f['steps'][k][:] for k in ('proprioception','mujoco_state')}
            loaded[c]={'ev':ev,'g':g,'meta':meta,'queries':queries,'raw':raw}
            objects={};names=[o['object_id'] for o in g['objects']]
            for i,obj in enumerate(g['objects']):
                p=ev['object_position'][:,i];name=obj['object_id']
                metrics={'movable':obj['movable_free_object'],'initial_position_m':p[10].tolist(),'last_position_m':p[-1].tolist(),
                    'max_height_change_mm':float(1000*(p[10:,2]-p[10,2]).max()),
                    'max_height_step':int(np.argmax(p[10:,2])+10),'last_height_change_mm':float(1000*(p[-1,2]-p[10,2])),
                    'above_initial_30mm_intervals':runs(p[:,2]-p[10,2]>.03),
                    'gripper_contact_intervals':runs([name in r['movable_objects'] for r in contacts]),
                    'two_finger_contact_intervals':runs([len(r['finger_sides'].get(name,[]))==2 for r in contacts])}
                if 'black_bowl' in name and 'plate_1' in names:
                    dist=1000*np.linalg.norm(p[:,:2]-ev['object_position'][:,names.index('plate_1'),:2],axis=1)
                    metrics['plate_xy_mm']={'minimum':float(dist[10:].min()),'last':float(dist[-1]),'below_30mm_intervals':runs(dist<30)}
                objects[name]=metrics
            joints={}
            for joint in g['joint_map']:
                if joint['type'] in (2,3) and any(s in (joint['name'] or '') for s in ('stove','cabinet','drawer')):
                    x=ev['joint_qpos'][:,joint['qpos_address']]
                    joints[joint['name']]={**joint,'initial':float(x[10]),'minimum':float(x[10:].min()),'maximum':float(x[10:].max()),'last':float(x[-1])}
            command=ev['actions'][:,6]
            arms[c]={'success':case['outcomes'][c],'frames':len(command),'objects':objects,'fixture_joints':joints,
                'goal_atoms':g['goal_atoms'],'goal_true_intervals':[runs(ev['goal_values'][:,i]) for i in range(len(g['goal_atoms']))],
                'all_goals_true_intervals':runs(ev['goal_values'].all(axis=1)),
                'closing_steps':(np.flatnonzero((command[:-1]==-1)&(command[1:]==1))+1).tolist(),
                'opening_steps':(np.flatnonzero((command[:-1]==1)&(command[1:]==-1))+1).tolist(),
                'last_aperture_mm':float(1000*np.abs(ev['state'][-1,-2:]).sum()),
                'annotation_receipt_digest':read(root/'receipt.json')['digest'],'source_sha256':receipt['source_sha256']}
        comparisons={}
        for left,right in itertools.combinations(CONDITIONS,2):
            a,b=loaded[left],loaded[right];step=first_difference(a['ev']['actions'],b['ev']['actions'])
            assert step is None or step>=10
            row={'first_action_difference_step':step,'first_gripper_difference_step':first_difference(a['ev']['actions'][:,6],b['ev']['actions'][:,6])}
            if step is not None:
                anchor=10+8*((step-10)//8);q=(anchor-10)//8
                with h5py.File(case['artifacts'][left]) as fa,h5py.File(case['artifacts'][right]) as fb:
                    match={k:bool(np.array_equal(fa['steps'][k][anchor],fb['steps'][k][anchor])) for k in ('agent_rgb','wrist_rgb','mujoco_state')}
                    match['proprioception_float32']=bool(np.array_equal(fa['steps/proprioception'][anchor].astype('f4'),fb['steps/proprioception'][anchor].astype('f4')))
                    delta=fb['steps/predicted_action_chunk'][anchor]-fa['steps/predicted_action_chunk'][anchor]
                row.update(query_anchor=anchor,inputs_at_first_changed_query=match,
                    left_prediction=a['queries'][q]['policy']['prediction'],right_prediction=b['queries'][q]['policy']['prediction'],
                    left_horizon=a['queries'][q]['policy']['used_horizon'],right_horizon=b['queries'][q]['policy']['used_horizon'],
                    first_chunk_translation_rms_delta=float(np.sqrt(np.mean(delta[:,:3]**2))),
                    first_chunk_rotation_rms_delta=float(np.sqrt(np.mean(delta[:,3:6]**2))))
            comparisons[left+' versus '+right]=row
        results.append({'id':case['id'],'task':case['task'],'arms':arms,'comparisons':comparisons})
        steps=[np.rint(np.linspace(10,len(loaded[c]['ev']['actions'])-1,6)).astype(int).tolist() for c in CONDITIONS]
        sheet(case,loaded,steps,'overview',output);trace(case,loaded,output)
        print(json.dumps({'completed_cases':len(results),'total':len(cfg['selection']['pairs'])}),flush=True)
    write(output/'inspection.json',{'contract_digest':cfg['digest'],'annotation_digest':ann['digest'],'cases':results,
        'model_calls':0,'simulator_steps':0,'new_rollouts':0,'independent_human_qa':False,
        'frame_convention':'Zero-based saved pre-action frames; policy seconds=(frame-10)/20; intervals half-open; terminal post-action state absent.',
        'goal_qualification':'Goal flags reconstructed from each saved state; recorded terminal outcome remains authoritative.'})
    write(output/'completion.json',{'status':'complete','seconds':time.monotonic()-start,
        'source_sha256':sha(__file__),'inspection_digest':read(output/'inspection.json')['digest'],
        'figures_sha256':{p.name:sha(p) for p in sorted(output.glob('*.png'))},'model_calls':0,'simulator_steps':0})


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('bundle','annotations','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();run(args.bundle,args.annotations,args.output)
