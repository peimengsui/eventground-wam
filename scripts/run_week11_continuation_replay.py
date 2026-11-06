"""Matched-observation replay of both continuation endpoints, without rollouts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from run_week11_horizon_replay import capture
from week11_replay_common import read, sha, write


def observation(entry):
    path=Path(entry['artifact']);step=entry['step']
    with h5py.File(path) as f:
        meta=json.loads(f['episode/metadata_json'][()])
        obs={k:f['steps'][k][step] for k in ('agent_rgb','wrist_rgb','proprioception')}
        recorded=f['steps/predicted_action_chunk'][step]
    obs.update(instruction=meta['instruction'],suite=meta['suite'])
    q=json.loads(path.with_suffix('.queries.jsonl').read_text().splitlines()[entry['query']])['policy']
    dig=hashlib.sha256()
    for k in ('agent_rgb','wrist_rgb','proprioception'):
        x=obs[k].astype('f4') if k=='proprioception' else obs[k]
        dig.update(k.encode());dig.update(str((x.shape,str(x.dtype))).encode());dig.update(x.tobytes())
    dig.update(json.dumps([meta['instruction'],meta['suite']]).encode())
    return obs,recorded,q,dig.hexdigest()


def run(bundle,output):
    import run_week11_multihorizon_screen as screen
    import run_week11_pilot as pilot
    from week11_multihorizon_runtime import ROOT, admission, runtime_identity

    from eventground_wam.b0_training import configure_deterministic_cuda, seed_training
    from eventground_wam.pixel_predictor import tensor_state_digest
    from eventground_wam.week11_resources import ResourceGuard, require_idle_gpu

    cfg=read(bundle/'contract.json')
    for p,h in cfg['code_sha256'].items():assert sha(bundle/p)==h,p
    output.mkdir(parents=True,exist_ok=False);write(output/'contract.json',cfg)
    guard=ResourceGuard(output,cfg['digest'],cfg['resources'])
    started=time.monotonic();result={'status':'incomplete','model_calls':0,'training_updates':0,'simulator_steps':0,'event_model_calls':0}
    def call(policy,obs,horizon):
        guard.check();assert result['model_calls']<cfg['maximum_model_calls']
        value=capture(policy,obs,horizon);result['model_calls']+=1;guard.check();return value
    def progress(phase,**fields):
        row={'phase':phase,'model_calls':result['model_calls'],'seconds':time.monotonic()-started,**fields}
        write(output/'progress.json',row);print(json.dumps(row),flush=True)
    try:
        require_idle_gpu();os.environ['TOKENIZERS_PARALLELISM']='false'
        configure_deterministic_cuda();seed_training(17);torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.9);torch.cuda.reset_peak_memory_stats()
        training,source,_=admission();ss=screen.source_check(training,source)
        assert source['digest']==cfg['training_source_digest'] and ss['digest']==cfg['screen_source_digest']
        assert training['digest']==cfg['training_design_digest']
        runtime=runtime_identity();run_root=Path(cfg['run_root'])
        for arm,pin in cfg['endpoints'].items():
            end=read(run_root/'training'/arm/'endpoint/endpoint.json')
            assert end['digest']==pin['digest'] and end['policy_sha256']==pin['policy_sha256']
            assert sha(run_root/'training'/arm/'endpoint/policy.pt')==pin['policy_sha256']
        for path,h in cfg['input_sha256'].items():assert sha(path)==h,path
        _,_,original,_=pilot.endpoint_check(read(ROOT/'configs/evaluation/week11_integration_v1.json'))
        actions=screen.Actions(training,run_root,original)
        write(output/'preflight.json',{'training_source_digest':source['digest'],'screen_source_digest':ss['digest'],
            'runtime':runtime,'input_files_verified':len(cfg['input_sha256']),'endpoints':cfg['endpoints']})
        # Every originating-model output must reproduce before interpreting forced alternatives.
        parity=[];states={}
        for arm in cfg['models']:
            policy=actions.select(arm);states[arm]=tensor_state_digest(actions.model.state_dict())
            for entry in cfg['observations']:
                if entry['origin_model']!=arm:continue
                obs,recorded,q,digest=observation(entry)
                got=call(policy,obs,q['used_horizon'])['decoded_actions']
                row={'id':entry['id'],'model':arm,'used_horizon':q['used_horizon'],'input_digest':digest,
                    'exact':bool(np.array_equal(got,recorded)),'max_abs_delta':float(np.abs(got-recorded).max())}
                parity.append(row)
                if not row['exact']:
                    write(output/'parity-failure.json',row)
                    raise AssertionError('Recorded-action parity failed: '+entry['id'])
            assert tensor_state_digest(actions.model.state_dict())==states[arm]
            progress('qualification',model=arm,qualified_observations=len(parity))
        assert len(parity)==len(cfg['observations'])
        write(output/'qualification.json',{'status':'pass','rows':parity,'all_recorded_outputs_exact':True})
        rows=[];input_digests={};state_checks={}
        for arm in cfg['models']:
            policy=actions.select(arm)
            assert tensor_state_digest(actions.model.state_dict())==states[arm]
            for index,entry in enumerate(cfg['observations']):
                obs,recorded,q,digest=observation(entry)
                if entry['id'] in input_digests:assert input_digests[entry['id']]==digest
                input_digests[entry['id']]=digest
                values={}
                for h in (8,2,4,16,32):
                    values[h]=call(policy,obs,h)
                    assert np.array_equal(values[h]['proposal'],values[8]['proposal'])
                    assert np.array_equal(values[h]['query_hidden'],values[8]['query_hidden'])
                repeat=call(policy,obs,8)
                assert all(np.array_equal(repeat[k],values[8][k]) for k in repeat),'repeat differs'
                owned=entry['origin_model']==arm
                if owned:assert np.array_equal(values[q['used_horizon']]['decoded_actions'],recorded)
                dest=output/'tensors'/arm/f'{index:04d}.npz';dest.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(dest,**{f'h{h}_{k}':v[k] for h,v in values.items()
                    for k in ('decoded_actions','raw_actions','normalized_actions')},recorded_actions=recorded)
                row={**entry,'evaluated_model':arm,'input_digest':digest,'recorded_horizon':q['used_horizon'],
                    'recorded_prediction':q['prediction'],'own_recorded_parity':True if owned else None,
                    'h8_repeat_all_tensors_exact':True,'proposal_and_query_horizon_invariant':True,
                    'tensor_file':str(dest.relative_to(output)),'tensor_sha256':sha(dest)}
                rows.append(row);write(output/'observations'/arm/f'{index:04d}.json',row)
                if index%25==0:progress('replay',model=arm,completed_model_observations=index+1,total_per_model=len(cfg['observations']))
            after=tensor_state_digest(actions.model.state_dict());assert after==states[arm]
            state_checks[arm]={'before':states[arm],'after':after}
        for path,h in cfg['input_sha256'].items():assert sha(path)==h,path
        for arm,pin in cfg['endpoints'].items():assert sha(run_root/'training'/arm/'endpoint/policy.pt')==pin['policy_sha256']
        assert admission()[1]['digest']==cfg['training_source_digest']
        assert screen.source_check(training,source)['digest']==cfg['screen_source_digest']
        assert result['model_calls']==len(cfg['observations'])*13
        write(output/'replay.json',{'observations':rows,'model_state_checks':state_checks,
            'input_files_rehashed_after':len(cfg['input_sha256']),'endpoints_unchanged':True,'source_unchanged':True,
            'contract_digest':cfg['digest']})
        result.update(status='complete',distinct_observation_entries=len(cfg['observations']),model_observations=len(rows),
            replay_digest=read(output/'replay.json')['digest'],qualification_digest=read(output/'qualification.json')['digest'])
    except BaseException as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['resources']=guard.close();result['seconds']=time.monotonic()-started
        write(output/'completion.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.bundle,a.output)
