"""One fresh-SFT preservation pilot or matched-rate SFT control; no automatic retries."""
from __future__ import annotations
import argparse
import copy
import json
import signal
import socket
import time
from datetime import datetime,timezone
from pathlib import Path
import torch
from train_week10_eventground import check_gpu_exclusive,require_output_capacity,write_completion,json_bytes
from train_week10_weak_sft import save_payload
from qualify_week10_weak_supervision import recovery_digest
from eventground_wam.b0_training import configure_deterministic_cuda,seed_training,capture_rng,cpu_tree
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_data import load_current_observation,collate_current_observations,target_tokens
from eventground_wam.eventground_model import serialize_tokens
from eventground_wam.eventground_runtime import build_eventground_from_m4,verify_eventground_parent
from eventground_wam.eventground_training import CHECKPOINT_FORMAT
from eventground_wam.eventground_preservation import PreservationTrainer
from eventground_wam.eventground_future_onset import TASK,audit_row,weak_reward
from eventground_wam.eventground_future_runtime import load_endpoint
from eventground_wam.eventground_preservation_runtime import verify,require_stage,parent_base,selected_weight
from eventground_wam.eventground_weak_supervision import sealed,sha
from eventground_wam.pixel_predictor import tensor_state_digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True);parser.add_argument('--stage',required=True)
    a=parser.parse_args();base=a.base;stage=a.stage;root=base/'source'
    p,index,manifests,recipe=verify(base)
    weight,diagnostic=selected_weight(base)
    if socket.gethostname()!='feifei' or str(base.resolve())!=p['remote_base'] or stage not in p['training_stages']:raise ValueError('Host/base/stage changed')
    require_stage(base,stage);s=sealed(base/'schedules'/f'{stage}.json');resources=p['training_stages'][stage]
    rows={r['prompt_id']:r for r in manifests['train']['rows']};batches=[[rows[pid] for pid in batch] for batch in s['batches']]
    candidates={r['candidate']['recording_id']:r['candidate'] for r in index['panels']['train']}
    check_gpu_exclusive();out=base/'runs'/stage;out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();trainer=None;executed=0;replay=False
    report=dict(artifact_type='eventground_wam.preservation_training',task_definition=TASK,status='failed_preserved',stage=stage,
        protocol_digest=p['digest'],schedule_digest=s['digest'],independent_qa_passed=False,automatic_retries=0,
        started_at=datetime.now(timezone.utc).isoformat(),checkpoints=[])
    def budget():
        if time.monotonic()-start>resources['wall_seconds']:raise TimeoutError('Frozen stage wall ceiling')
        require_output_capacity(out,resources,0)
        if torch.cuda.is_initialized() and torch.cuda.max_memory_reserved()>.9*torch.cuda.get_device_properties(0).total_memory:raise RuntimeError('GPU memory ceiling')
    def stop(*_):raise TimeoutError('Bounded service stopped; preserve failed attempt')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(resources['wall_seconds'])
    try:
        budget();configure_deterministic_cuda();torch.set_num_threads(4);seed_training(17)
        torch.cuda.set_per_process_memory_fraction(.9);torch.cuda.reset_peak_memory_stats();device=torch.device('cuda:0')
        verified={};observations={};cache_bytes=0
        for pid in dict.fromkeys(pid for batch in s['batches'] for pid in batch):
            row=rows[pid];audit_row(row,candidates[row['recording_id']]);o=load_current_observation(row,verified_files=verified)
            observations[pid]=o;cache_bytes+=o.front.nbytes+o.wrist.nbytes+o.proprioception.nbytes
            if cache_bytes>resources['maximum_cached_observation_bytes']:raise ValueError('CPU observation cache cap')
            budget()
        model,builder,allowlist,parent=build_eventground_from_m4(root,recipe,Path(p['cache']),device)
        if {k:list(v) for k,v in allowlist.items()}!=recipe['exact_allowlist']:raise ValueError('Changed allowlist')
        mode=stage.split('-')[0];is_rlvr=mode=='hybrid'
        parent_sft=load_endpoint(model,allowlist,parent_base(base),'sft')
        reference=copy.deepcopy(model) if is_rlvr else None
        if reference is not None:reference.requires_grad_(False);reference.eval()
        trainer=PreservationTrainer(model,allowlist,recipe)
        if trainer.optimizer.state or trainer.global_step or trainer.prompts_seen:raise ValueError('Nonempty optimizer')
        params=dict(model.named_parameters());frozen={n:t for n,t in params.items() if not t.requires_grad}
        frozen_sha=tensor_state_digest(frozen);initial_sha=tensor_state_digest({n:params[n] for n in trainer.names})
        reference_sha=tensor_state_digest(dict(reference.named_parameters())) if reference is not None else None
        if parent_sft and initial_sha!=parent_sft['inference_header']['parameter_sha256']:raise ValueError('Not initialized from exact fresh SFT')
        identity=seal_contract(dict(artifact_type='eventground_wam.preservation_training_identity',task_definition=TASK,
            mode=mode,preservation_weight=weight,diagnostic_digest=diagnostic['digest'],
            protocol_digest=p['digest'],schedule_digest=s['digest'],stage=stage,parent_identity=parent,
            sft_completion_digest=parent_sft['digest'] if parent_sft else None,initial_trainable_sha256=initial_sha,
            initial_frozen_sha256=frozen_sha,reference_sha256=reference_sha,fresh_optimizer=True,pilot_weights_reused=False,seed=17))
        (out/'identity.json').write_bytes(json_bytes(identity));report['identity_digest']=identity['digest']
        print(json.dumps(dict(phase='fresh_initialization_verified',stage=stage,seconds=time.monotonic()-start)),flush=True)
        weights=p['reward_weights']['primary']
        reward_records=[]
        def reward_function(evidence,tokens):
            predictions=serialize_tokens(tokens)
            if len(predictions)!=4*len(evidence):raise ValueError('Split candidate group')
            values=[]
            for i,row in enumerate(evidence):
                group=predictions[4*i:4*i+4];scores=[weak_reward(pred,row,candidates[row['recording_id']],weights) for pred in group]
                values.append(scores);reward_records.append(dict(prompt_id=row['prompt_id'],predictions=group,rewards=scores,recovery_replay=replay))
            return torch.tensor(values,dtype=torch.float32)
        def update(batch):
            nonlocal executed
            budget();tick=time.monotonic();micro=[]
            for row in batch:
                enc,state=collate_current_observations([observations[row['prompt_id']]],builder,device)
                micro.append((enc,state,[row],target_tokens([row],device)))
            result=trainer.update(micro,prompt_count=len(batch),mode=mode,weight=weight,reference_model=reference,reward_function=reward_function)
            torch.cuda.synchronize();executed+=1
            result.update(step_seconds=time.monotonic()-tick,prompt_ids=[r['prompt_id'] for r in batch],prompt_count=len(batch),recovery_replay=replay)
            budget();return result
        def checkpoint():
            payload=dict(format=CHECKPOINT_FORMAT,identity=identity,recipe_digest=recipe['digest'],stage=trainer.stage,
                exact_names=trainer.names,parameters={n:params[n].detach().cpu().clone() for n in trainer.names},
                optimizer=cpu_tree(trainer.optimizer.state_dict()),rng=capture_rng(),global_step=trainer.global_step,prompts_seen=trainer.prompts_seen)
            name=f'update-{trainer.global_step:06d}.pt';h=save_payload(out/name,payload,output=out,resources=resources)
            report['checkpoints'].append(dict(file=name,sha256=h,step=trainer.global_step))
        for i,batch in enumerate(batches):
            result=update(batch)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k!='prompt_ids'}),flush=True)
            if trainer.global_step in resources['checkpoint_steps']:checkpoint()
            if stage.endswith('qualification') and i==1:
                expected=recovery_digest(trainer);expected_metrics={k:v for k,v in result.items() if k not in ('step_seconds','recovery_replay')}
                trainer.resume(out/'update-000001.pt',identity=identity);replay=True;actual_result=update(batch)
                actual=recovery_digest(trainer);actual_metrics={k:v for k,v in actual_result.items() if k not in ('step_seconds','recovery_replay')}
                report['recovery']=dict(expected_digest=expected,actual_digest=actual,metrics_identical=expected_metrics==actual_metrics)
                if expected!=actual or expected_metrics!=actual_metrics:raise ValueError('Bitwise recovery failed')
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(actual_result,allow_nan=False)+'\n')
        if trainer.global_step!=s['updates'] or trainer.prompts_seen!=s['presentations']:raise ValueError('Fixed exposure differs')
        for path,(_,fingerprint) in verified.items():
            st=Path(path).stat()
            if (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)!=fingerprint:raise ValueError('Input changed')
        if tensor_state_digest(frozen)!=frozen_sha:raise ValueError('Frozen parameters changed')
        if reference is not None and tensor_state_digest(dict(reference.named_parameters()))!=reference_sha:raise ValueError('Reference changed')
        verify_eventground_parent(sealed(root/'configs/training/week10_eventground_experiment_v3.json'))
        if parent_sft and sha(parent_base(base)/'runs/sft'/parent_sft['endpoint_file'])!=parent_sft['endpoint_sha256']:raise ValueError('Stored SFT changed')
        state={n:params[n].detach().cpu().clone() for n in trainer.names};digest=tensor_state_digest(state)
        if digest==initial_sha:raise ValueError('No parameter update')
        h=seal_contract(dict(artifact_type='eventground_wam.future_onset_endpoint',task_definition=TASK,protocol_digest=p['digest'],
            identity_digest=identity['digest'],schedule_digest=s['digest'],stage=stage,exact_names=list(trainer.names),parameter_sha256=digest,
            global_step=trainer.global_step,prompts_seen=trainer.prompts_seen,selection='fixed_final_update',independent_qa_passed=False))
        if not stage.endswith('qualification'):
            endpoint_sha=save_payload(out/'endpoint.pt',dict(header=h,parameters=state),output=out,resources=resources)
            report.update(endpoint_file='endpoint.pt',endpoint_sha256=endpoint_sha,inference_header=h)
        if is_rlvr:(out/'rewards.json').write_bytes(json_bytes(dict(groups=reward_records,group_size=4,weights=weights)))
        gates=dict(current_observation_only=True,all_targets_reaudited=True,exact_parameter_allowlist=True,finite_updates=True,
            frozen_parameters_unchanged=True,reference_unchanged=True,stored_parents_unchanged=True,source_files_unchanged=True,
            parameters_changed=True,exact_prompt_exposure=True,bitwise_recovery=not stage.endswith('qualification') or report['recovery']['metrics_identical'])
        report.update(status='complete',gates=gates,current_observations=len(observations),source_files_verified=len(verified),cached_observation_bytes=cache_bytes)
        budget()
    except Exception as e:report.update(status='failed_preserved',error=repr(e));raise
    finally:
        signal.alarm(0);elapsed=time.monotonic()-start
        report.update(elapsed_seconds=elapsed,global_step=trainer.global_step if trainer else 0,prompts_seen=trainer.prompts_seen if trainer else 0,
            partial_update_in_progress=trainer._in_update if trainer else False,executed_updates=executed,finished_at=datetime.now(timezone.utc).isoformat())
        if torch.cuda.is_initialized():report.update(peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        report['files_sha256']={str(x.relative_to(out)):sha(x) for x in out.rglob('*') if x.is_file() and x.suffix!='.pt'}
        write_completion(out,report,resources);print(json.dumps(dict(stage=stage,status=report['status'],seconds=elapsed)),flush=True)
if __name__=='__main__':main()
