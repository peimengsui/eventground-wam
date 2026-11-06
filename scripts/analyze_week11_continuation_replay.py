"""Factor checkpoint and horizon effects on identical saved observations."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from week11_replay_common import HORIZONS, distance, read, sha, write

MODELS=('h8-continuation','mixed-continuation')


def compare(a,b):
    return {'translation':distance(a[:,:3],b[:,:3]),'rotation':distance(a[:,3:6],b[:,3:6]),
        'gripper_disagree_actions':int(np.count_nonzero(a[:,6]!=b[:,6]))}


def aggregate(rows,key):
    selected=[key(r) for r in rows];out={'observations':len(selected)}
    for kind in ('translation','rotation'):
        out[kind]={}
        for metric in ('rms_delta','relative_l2_delta'):
            v=[x[kind][metric] for x in selected if x[kind][metric] is not None]
            out[kind][metric]={'median':float(np.median(v)),'p90':float(np.quantile(v,.9)),'n':len(v)} if v else None
    out['gripper_changed_chunks']=sum(x['gripper_disagree_actions']>0 for x in selected)
    out['gripper_changed_actions']=sum(x['gripper_disagree_actions'] for x in selected)
    return out


def run(replay,output):
    start=time.monotonic();cfg=read(replay/'contract.json');completed=read(replay/'completion.json')
    assert completed['status']=='complete' and completed['model_calls']==cfg['expected_model_calls']
    data=read(replay/'replay.json');assert data['digest']==completed['replay_digest']
    qualification=read(replay/'qualification.json');assert len(qualification['rows'])==len(cfg['observations'])
    assert all(x['exact'] and x['max_abs_delta']==0 for x in qualification['rows'])
    bymodel={(r['evaluated_model'],r['id']):r for r in data['observations']}
    output.mkdir(parents=True,exist_ok=False);rows=[];examples=[]
    anchors={
        'targeted-467-0129':[66,74,82,90], 'targeted-467-0231':[106,114,122,130,138,146],
        'targeted-467-0365':[98,106,114,122], 'targeted-467-0362':[170,178,186,194,202],
        'clean-200-0006':[90,98,106,114], 'clean-200-0030':[66,74,82,90],
        'clean-200-0107':[138,146,154,162,170], 'clean-200-0155':[202,210,218,226,250,258,266,274],
        'clean-200-0151':[194,202,218,282,290,298]}
    unique=set();saved={}
    for e in cfg['observations']:
        modelrows={m:bymodel[m,e['id']] for m in MODELS};arrays={}
        for m,r in modelrows.items():
            assert sha(replay/r['tensor_file'])==r['tensor_sha256']
            z=dict(np.load(replay/r['tensor_file']));arrays[m]=z
            assert all(np.isfinite(a).all() for a in z.values())
            if r['own_recorded_parity']:assert np.array_equal(z[f'h{r["recorded_horizon"]}_decoded_actions'],z['recorded_actions'])
        left,right=[modelrows[m] for m in MODELS]
        assert left['input_digest']==right['input_digest']
        for h in HORIZONS:
            assert arrays[MODELS[0]][f'h{h}_decoded_actions'].shape==(8,7)
        actions={m:{h:z[f'h{h}_decoded_actions'] for h in HORIZONS} for m,z in arrays.items()}
        row={**e,'input_digest':left['input_digest'],'recorded_horizon':left['recorded_horizon'],
            'weight_effect':{str(h):compare(actions[MODELS[0]][h],actions[MODELS[1]][h]) for h in HORIZONS},
            'horizon_effect':{m:{str(h):compare(a[8],a[h]) for h in HORIZONS} for m,a in actions.items()},
            'interaction':{str(h):compare(actions[MODELS[0]][h]-actions[MODELS[0]][8],actions[MODELS[1]][h]-actions[MODELS[1]][8]) for h in HORIZONS},
            'commands':{m:{str(h):''.join('C' if v==1 else 'O' for v in a[h][:,6]) for h in HORIZONS} for m,a in actions.items()}}
        key=(e['case_id'],left['input_digest']);row['duplicate_input_in_case']=key in unique;unique.add(key)
        rows.append(row)
        if e['step'] in anchors[e['case_id']]:
            examples.append({**e,'recorded_horizon':left['recorded_horizon'],'commands':row['commands'],
                'raw_gripper_score':{m:{str(h):arrays[m][f'h{h}_raw_actions'][0,:,6].tolist() for h in HORIZONS} for m in MODELS},
                'mean_translation':{m:{str(h):actions[m][h][:,:3].mean(0).tolist() for h in HORIZONS} for m in MODELS},
                'mean_rotation':{m:{str(h):actions[m][h][:,3:6].mean(0).tolist() for h in HORIZONS} for m in MODELS}})
        saved[e['id']]=actions
    selected=[r for r in rows if not r['duplicate_input_in_case'] and r['phase']!='initial_control']
    grouped={c['id']:[r for r in selected if r['case_id']==c['id']] for c in cfg['cases']}
    bycase={cid:{'n':len(rs),'weight_h8':aggregate(rs,lambda r:r['weight_effect']['8']),
        'horizon_32':{m:aggregate(rs,lambda r,m=m:r['horizon_effect'][m]['32']) for m in MODELS}} for cid,rs in grouped.items()}
    overall={'weight_by_horizon':{str(h):aggregate(selected,lambda r,h=h:r['weight_effect'][str(h)]) for h in HORIZONS},
        'horizon_by_checkpoint':{m:{str(h):aggregate(selected,lambda r,m=m,h=h:r['horizon_effect'][m][str(h)]) for h in HORIZONS} for m in MODELS}}
    write(output/'analysis.json',{'contract_digest':cfg['digest'],'replay_digest':data['digest'],
        'entry_count':len(rows),'unique_case_inputs':len(unique),'interaction_window_unique_inputs':len(selected),
        'duplicate_inputs':sum(r['duplicate_input_in_case'] for r in rows),'cases':bycase,'overall':overall,'rows':rows,
        'interpretation':'Decoded translation and rotation controller commands, not physical displacement or action correctness. Correlated, outcome-selected observations; no population inference.'})
    write(output/'examples.json',{'entries':examples,'legend':'C=close (+1); O=open (-1); eight commands per query; candidate horizon never changes execution chunk length.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    colors=('#246b91','#b75222')
    fig,ax=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for j,kind in enumerate(('translation','rotation')):
        for m,col in zip(MODELS,colors,strict=True):
            y=[100*overall['horizon_by_checkpoint'][m][str(h)][kind]['relative_l2_delta']['median'] for h in HORIZONS]
            ax[j].plot(range(5),y,'o-',color=col,label=m.replace('-continuation',''))
        ax[j].set_xticks(range(5),HORIZONS);ax[j].set_xlabel('Forced candidate horizon');ax[j].set_ylabel('Median relative L2 change from horizon 8 (%)')
        ax[j].set_title(kind.capitalize()+' commands');ax[j].grid(alpha=.2);ax[j].legend()
    fig.suptitle(f'Horizon sensitivity on {len(selected)} identical-input interaction-window observations')
    fig.supxlabel('Lower sensitivity is not a measure of correctness or rollout success.',fontsize=10)
    fig.savefig(output/'horizon-sensitivity.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,5),layout='constrained');ids=list(bycase)
    for j,kind in enumerate(('translation','rotation')):
        y=[bycase[c]['weight_h8'][kind]['rms_delta']['median'] for c in ids]
        ax[j].barh(range(len(ids)),y,color='#6b6688');ax[j].set_yticks(range(len(ids)),[c.replace('targeted-467-','T').replace('clean-200-','C') for c in ids]);ax[j].invert_yaxis()
        ax[j].set_xlabel('Median RMS difference in decoded controller commands');ax[j].set_title(kind.capitalize()+' at fixed horizon 8')
    fig.suptitle('Training changes the action mapping on the same observations')
    fig.savefig(output/'checkpoint-effects.png',dpi=140);plt.close(fig)
    chosen=[('clean-200-0030','mixed-continuation','fixed8',74),('clean-200-0107','h8-continuation','fixed8',154),
        ('clean-200-0155','mixed-continuation','fixed8',218),('clean-200-0155','mixed-continuation','adaptive',218),
        ('clean-200-0006','h8-continuation','adaptive',106),('targeted-467-0129','h8-continuation','adaptive',74)]
    fig,axes=plt.subplots(3,2,figsize=(12,12),layout='constrained')
    for ax,(cid,origin,route,step) in zip(axes.flat,chosen,strict=True):
        entry=next(e for e in cfg['observations'] if e['case_id']==cid and e['origin_model']==origin and e['origin_route']==route and e['step']==step)
        values=np.array([saved[entry['id']][m][h][:,6] for m in MODELS for h in HORIZONS])
        ax.imshow(values,aspect='auto',vmin=-1,vmax=1,cmap=ListedColormap(['#e8e8e8','#345e80']))
        ax.set_yticks(range(10),[('H8' if m==MODELS[0] else 'Mixed')+f' / h{h}' for m in MODELS for h in HORIZONS],fontsize=8)
        ax.set_xticks(range(8),range(step,step+8),fontsize=8);ax.axhline(4.5,color='white',linewidth=2)
        ax.set_title(cid+f' · frame {step}\nInput from '+origin.replace('-continuation','')+'/'+route,fontsize=10)
    fig.suptitle('Identical-input gripper commands: blue = close; grey = open')
    fig.supxlabel('Counterfactual commands only; downstream motion and success were not simulated.',fontsize=10)
    fig.savefig(output/'gripper-comparisons.png',dpi=140);plt.close(fig)
    write(output/'completion.json',{'status':'complete','seconds':time.monotonic()-start,'source_sha256':sha(__file__),
        'analysis_digest':read(output/'analysis.json')['digest'],'examples_digest':read(output/'examples.json')['digest'],
        'figures_sha256':{p.name:sha(p) for p in sorted(output.glob('*.png'))},'model_calls':0,'simulator_steps':0})
    print(json.dumps({'status':'complete','entries':len(rows),'unique':len(unique),'interaction_inputs':len(selected),'seconds':time.monotonic()-start}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.replay,a.output)
