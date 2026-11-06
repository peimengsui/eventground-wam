"""Distinguish unused saved-frame render differences from consumed input drift."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from week11_replay_common import read, sha, write


def changes(a,b,frames):
    left,right=a[frames],b[frames];delta=left.astype('f8')-right.astype('f8')
    flat=delta.reshape(len(frames),-1);counts=np.count_nonzero(flat,axis=1);ix=np.flatnonzero(counts)
    return {'frames':[int(frames[i]) for i in ix], 'changed_values':[int(counts[i]) for i in ix],
            'max_abs_delta':[float(np.abs(flat[i]).max()) for i in ix]}


def pair(left,right,stop):
    with h5py.File(left) as a,h5py.File(right) as b:
        x,y=a['steps'],b['steps'];frames=np.arange(stop+1);qs=np.arange(10,stop+1,8)
        images={k:changes(x[k],y[k],frames) for k in ('agent_rgb','wrist_rgb')}
        query_images={k:changes(x[k],y[k],qs) for k in images}
        states={k:bool(np.array_equal(x[k][:stop+1],y[k][:stop+1])) for k in ('proprioception','mujoco_state')}
        actions=bool(np.array_equal(x['executed_action'][:stop],y['executed_action'][:stop]))
        chunks=bool(np.array_equal(x['predicted_action_chunk'][10:stop:8],y['predicted_action_chunk'][10:stop:8]))
        d=x['predicted_action_chunk'][stop]-y['predicted_action_chunk'][stop]
        motion={'translation_rms_delta':float(np.sqrt(np.mean(d[:,:3]**2))),
                'rotation_rms_delta':float(np.sqrt(np.mean(d[:,3:6]**2))),
                'gripper_changed_actions':int(np.count_nonzero(d[:,6]))}
    q=[[json.loads(line)['policy'] for line in Path(p).with_suffix('.queries.jsonl').read_text().splitlines()] for p in (left,right)]
    forecasts=all(all(q[0][i][k]==q[1][i][k] for k in ('prediction','decision'))
        and q[0][i]['event']['categorical_probabilities']==q[1][i]['event']['categorical_probabilities']
        for i in range(len(qs)))
    return {'frame':stop,'all_saved_image_differences':images,'query_image_differences':query_images,
        'prior_and_intervention_states_equal':states,'prior_actions_equal':actions,'prior_chunks_equal':chunks,
        'prior_and_intervention_forecasts_equal':forecasts,'intervention_motion':motion,
        'intervention_horizons':[qq[(stop-10)//8]['used_horizon'] for qq in q],
        'consumed_inputs_physics_actions_forecasts_match':bool(all(not v['frames'] for v in query_images.values())
            and all(states.values()) and actions and chunks and forecasts)}


def first_changed_query(left,right):
    with h5py.File(left) as a,h5py.File(right) as b:
        x,y=a['steps'],b['steps'];n=min(len(x['executed_action']),len(y['executed_action']))
        for frame in range(10,n,8):
            equal={k:bool(np.array_equal(x[k][frame],y[k][frame])) for k in ('agent_rgb','wrist_rgb','proprioception','mujoco_state')}
            chunk=np.array_equal(x['predicted_action_chunk'][frame],y['predicted_action_chunk'][frame])
            if not all(equal.values()) or not chunk:
                return {'frame':frame,'input_equal':equal,'chunk_equal':bool(chunk),
                    'prior_actions_equal':bool(np.array_equal(x['executed_action'][:frame],y['executed_action'][:frame])),
                    'camera_delta':{k:changes(x[k],y[k],np.array([frame])) for k in ('agent_rgb','wrist_rgb')}}
    return None


def run(raw,audit,output):
    summary=read(audit/'summary.json');cfg=read(raw/'contract.json')
    paths={(r['repeat'],r['arm']):r['artifact'] for r in summary['rows']}
    inputs={p:sha(p) for p in {*paths.values(),*cfg['historical'].values()}}
    pairs=[]
    for rep in (1,2,3):
        for arm,frame in [('placement-only',138),('late-only',218)]:
            pairs.append({'repeat':rep,'arm':arm,**pair(paths[rep,'fixed8'],paths[rep,arm],frame)})
    comparisons=[('fixed8-repeat1-vs-repeat3',paths[1,'fixed8'],paths[3,'fixed8']),
        ('fixed8-repeat1-vs-historical',paths[1,'fixed8'],cfg['historical']['fixed8']),
        ('fixed8-repeat3-vs-historical',paths[3,'fixed8'],cfg['historical']['fixed8'])]
    comparisons += [(f'fixed8-vs-adaptive-repeat{rep}',paths[rep,'fixed8'],paths[rep,'adaptive']) for rep in (1,2,3)]
    differences={label:first_changed_query(a,b) for label,a,b in comparisons}
    write(output,{'summary_digest':summary['digest'],'source_sha256':sha(__file__),'input_sha256':inputs,
        'pairs':pairs,'first_changed_queries':differences,
        'consumed_input_matched_pairs':sum(r['consumed_inputs_physics_actions_forecasts_match'] for r in pairs),
        'strict_all_saved_frames_matched_pairs':sum(r['all_prefix_and_inputs_exact'] for r in summary['prefixes']),
        'qualification':'Supplementary recording-only audit; original strict audit is unchanged. No replacement repeats or new rollouts.',
        'model_calls':0,'simulator_steps':0})
    print(json.dumps({'consumed_input_matched_pairs':sum(r['consumed_inputs_physics_actions_forecasts_match'] for r in pairs),
        'first_changed_queries':differences}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('raw','audit','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();run(a.raw,a.audit,a.output)
