"""Independent routing, pairing, control-repeat and outcome audit for 12 episodes."""
from __future__ import annotations

import argparse
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
from week11_replay_common import read, sha, write

from eventground_wam.episode_recorder import read_episode_metadata
from eventground_wam.week11_first_chunk import ARMS, FIXED, GATED, ONLY, SKIP, audit_first_chunk

KEYS = ('agent_rgb','wrist_rgb','proprioception','mujoco_state','executed_action')


def equal(x,y,key):
    if key=='proprioception':x,y=x.astype(np.float32),y.astype(np.float32)
    return bool(np.array_equal(x,y))


def prefix(left,right,step):
    with h5py.File(left) as l,h5py.File(right) as r:
        ls,rs=l['steps'],r['steps']
        prior={k:equal(ls[k][:step],rs[k][:step],k) for k in KEYS}
        current={k:equal(ls[k][step],rs[k][step],k) for k in KEYS if k!='executed_action'}
        chunks=bool(np.array_equal(ls['predicted_action_chunk'][10:step:8],rs['predicted_action_chunk'][10:step:8]))
        at=bool(np.array_equal(ls['predicted_action_chunk'][step],rs['predicted_action_chunk'][step]))
        return {'prior_frames':prior,'input_at_intervention':current,'prior_query_chunks_equal':chunks,
                'all_prefix_and_inputs_exact':all(prior.values()) and all(current.values()) and chunks,
                'intervention_chunks_equal':at}


def repeat(current,old):
    with h5py.File(current) as c,h5py.File(old) as o:
        cs,os=c['steps'],o['steps'];n=min(len(cs['executed_action']),len(os['executed_action']))
        whole={k:equal(cs[k][:n],os[k][:n],k) for k in KEYS}
        first=None
        for step in range(10,n,8):
            flags={k:equal(cs[k][step],os[k][step],k) for k in KEYS if k!='executed_action'}
            flags['chunk']=bool(np.array_equal(cs['predicted_action_chunk'][step],os['predicted_action_chunk'][step]))
            flags['prior_actions']=equal(cs['executed_action'][:step],os['executed_action'][:step],'executed_action')
            if not all(flags.values()):first={'query':(step-10)//8,'step':step,'equal':flags};break
        same_length=len(cs['executed_action'])==len(os['executed_action'])
    success=read_episode_metadata(Path(current))['success'];old_success=read_episode_metadata(Path(old))['success']
    return {'current_success':success,'previous_success':old_success,'outcome_reproduced':success==old_success,
            'same_length':same_length,'common_frames':n,'all_common_frames':whole,'first_query_difference':first,
            'full_recorded_trajectory_exact':same_length and all(whole.values()) and first is None}


def run(raw,output):
    start=time.monotonic();output.mkdir(parents=True,exist_ok=False)
    cfg=read(raw/'contract.json');completion=read(raw/'completion.json');admission=read(raw/'admission.json')
    assert completion['status']=='complete' and completion['completed_rollouts']==12
    assert completion['contract_digest']==cfg['digest']
    reports={}
    for key,digest in completion['reports'].items():
        report=read(raw/'panels'/key/'report.json');assert report['digest']==digest;reports[key]=report
    rows=[];queries_total=0
    for case in cfg['cases']:
        row={'id':case['id'],'task':case['task'],'panel':case['panel'],'ordinal':case['ordinal'],'outcomes':{},'arms':{}}
        paths={}
        for arm in ARMS:
            report=reports[arm+'/'+case['panel']];ix=report['case_ids'].index(case['id']);result=report['episodes'][ix]
            assert not report['audits'][ix]['error'] and result['artifact']
            path=Path(result['artifact']);paths[arm]=path
            assert sha(path)==result['artifact_sha256'] and sha(Path(result['timing_artifact']))==result['timing_sha256']
            audit=audit_first_chunk(result,arm,admission)
            assert audit==report['gate_audits'][ix] and audit['valid'] and not audit['query_exceptions']
            queries=[json.loads(l) for l in path.with_suffix('.queries.jsonl').read_text().splitlines()]
            queries_total+=len(queries)
            assert read_episode_metadata(path)['success']==result['success']
            row['outcomes'][arm]=result['success']
            row['arms'][arm]={'artifact':str(path),'success':result['success'],'policy_steps':result['policy_step_count'],
                'retry_count':result['retry_count'],'termination_reason':result['termination_reason'],'routing_audit':audit,
                'horizon_counts':dict(Counter(str(q['policy']['used_horizon']) for q in queries))}
        d=10+8*case['gate_query'];row['intervention_step']=d
        row['activation_matches_previous']={a:row['arms'][a]['routing_audit']['first_closing_step']==case['first_closing_step']
            and row['arms'][a]['routing_audit']['first_query_after_closing']==case['gate_query'] for a in ARMS}
        row['pairing']={a+' vs '+b:prefix(paths[a],paths[b],d) for a,b in itertools.combinations(ARMS,2)}
        row['all_intervention_inputs_exact']=all(v['all_prefix_and_inputs_exact'] for v in row['pairing'].values())
        row['first_chunk_groups_exact']={a+' vs '+b:prefix(paths[a],paths[b],d+8)
                                        for a,b in ((FIXED,SKIP),(GATED,ONLY))}
        row['control_repeats']={a:repeat(paths[a],case['artifacts'][a]) for a in (FIXED,GATED)}
        rows.append(row)
    summary={'contract_digest':cfg['digest'],'completion_digest':completion['digest'],'rows':rows,
        'successes':{a:sum(r['outcomes'][a] for r in rows) for a in ARMS},'queries_audited':queries_total,
        'all_intervention_inputs_exact':all(r['all_intervention_inputs_exact'] for r in rows),
        'all_activation_times_reproduced':all(all(r['activation_matches_previous'].values()) for r in rows),
        'all_first_chunk_group_continuations_exact':all(v['all_prefix_and_inputs_exact'] for r in rows for v in r['first_chunk_groups_exact'].values()),
        'control_outcomes_reproduced':sum(v['outcome_reproduced'] for r in rows for v in r['control_repeats'].values()),
        'controls_full_trajectory_exact':sum(v['full_recorded_trajectory_exact'] for r in rows for v in r['control_repeats'].values()),
        'selected_diagnostic_cases':True,'population_estimate':False,'baseline_changed':False}
    write(output/'summary.json',summary)
    files={str(p.relative_to(raw)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(raw.rglob('*')) if p.is_file()}
    write(output/'raw-inventory.json',{'root':str(raw),'files':files})
    write(output/'completion.json',{'status':'complete','summary_digest':read(output/'summary.json')['digest'],
        'inventory_digest':read(output/'raw-inventory.json')['digest'],'raw_files_hashed':len(files),
        'source_sha256':sha(__file__),'seconds':time.monotonic()-start,'model_calls':0,'simulator_steps':0})
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'}),flush=True)
    print(json.dumps([{'case':r['id'],'outcomes':r['outcomes']} for r in rows]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.raw,a.output)
