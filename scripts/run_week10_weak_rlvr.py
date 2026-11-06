"""Bounded weak-evidence qualification/pilot only; never launches main RLVR or control."""
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
from eventground_wam.eventground_data import load_current_observation,collate_current_observations
from eventground_wam.eventground_model import serialize_tokens
from eventground_wam.eventground_runtime import build_eventground_from_m4,verify_eventground_parent
from eventground_wam.eventground_training import EventGroundTrainer,CHECKPOINT_FORMAT
from eventground_wam.eventground_weak_supervision import sealed,sha,build_dataset,audit_row
from eventground_wam.eventground_weak_evaluation import pilot_rows,weak_reward
from eventground_wam.eventground_weak_rlvr import load_completed_sft
from eventground_wam.pixel_predictor import tensor_state_digest


def checkpoint(trainer,path,identity,output,resources):
    if trainer._in_update:raise ValueError('Partial update')
    params=dict(trainer.model.named_parameters())
    payload=dict(format=CHECKPOINT_FORMAT,identity=identity,recipe_digest=trainer.recipe['digest'],stage=trainer.stage,
        exact_names=trainer.names,parameters={n:params[n].detach().cpu().clone() for n in trainer.names},
        optimizer=cpu_tree(trainer.optimizer.state_dict()),rng=capture_rng(),global_step=trainer.global_step,prompts_seen=trainer.prompts_seen)
    return save_payload(path,payload,output=output,resources=resources)


def main():
    p=argparse.ArgumentParser()
    for k in ('root','launch','output','cache'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();launch=sealed(a.launch);root=a.root;protocol=sealed(root/'configs/training/week10_weak_rlvr_evaluation_v1.json')
    if socket.gethostname()!='feifei' or str(a.output.resolve())!=launch['output'] or launch['protocol_digest']!=protocol['digest']:raise ValueError('Host/output/protocol mismatch')
    stage=launch['stage']
    if stage not in ('qualification','primary','calibration'):raise ValueError('Only qualification and pilot authorized')
    for mapping in (launch['code_sha256'],protocol['code_sha256']):
        for name,h in mapping.items():
            if sha(root/name)!=h:raise ValueError('Code changed '+name)
    recipe=sealed(root/'configs/training/week10_eventground_numerical_v2.json')
    weak=sealed(root/'configs/training/week10_eventground_exploratory_v4.json')
    data=sealed(root/'docs/evaluation/week10-exploratory-weak-v4/dataset.json')
    if recipe['digest']!=protocol['numerical_recipe_digest'] or data['digest']!=protocol['dataset_digest'] or build_dataset(root,weak)!=data:raise ValueError('Dataset/recipe changed')
    if stage!='qualification':
        q=sealed(Path(launch['qualification'])/'completion.json')
        if q['status']!='passed_qualification' or not all(q['gates'].values()) or q['protocol_digest']!=protocol['digest']:raise ValueError('Qualification not passed')
    if stage=='calibration':
        decision=sealed(launch['primary_decision'])
        if decision['protocol_digest']!=protocol['digest'] or decision['primary_promoted'] or not decision['calibration_permitted']:raise ValueError('Calibration not admitted')
    allpilot=pilot_rows(data)
    rows=([r for pid in data['qualification_prompt_ids'] for r in data['main_rows'] if r['prompt_id']==pid]
          if stage=='qualification' else allpilot)
    expected_ids=[r['prompt_id'] for r in rows]
    if expected_ids!=launch['prompt_ids']:raise ValueError('Frozen prompt schedule differs')
    batches=[rows,rows] if stage=='qualification' else [rows[i:i+16] for i in range(0,len(rows),16)]
    weights=(.3,.35,.2,.15) if stage=='calibration' else (.4,.25,.2,.15)
    check_gpu_exclusive();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();trainer=None;executed=0
    resources=launch['resources'];report=dict(artifact_type='eventground_wam.weak_rlvr_attempt',stage=stage,status='failed_preserved',
        launch_digest=launch['digest'],protocol_digest=protocol['digest'],independent_qa_passed=False,
        started_at=datetime.now(timezone.utc).isoformat(),automatic_retries=0,heldout_payload_reads=0,reward_weights=list(weights),checkpoints=[])
    def budget():
        if time.monotonic()-start>resources['wall_seconds']:raise TimeoutError('Attempt deadline')
        require_output_capacity(a.output,resources,0)
        if torch.cuda.is_initialized() and torch.cuda.max_memory_reserved()>.9*torch.cuda.get_device_properties(0).total_memory:raise RuntimeError('GPU memory ceiling')
    def stop(*_):raise TimeoutError('Attempt stopped by bounded service')
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGALRM,stop);signal.alarm(resources['wall_seconds'])
    try:
        budget();configure_deterministic_cuda();torch.set_num_threads(4);seed_training(17)
        torch.cuda.set_per_process_memory_fraction(.9);torch.cuda.reset_peak_memory_stats();device=torch.device('cuda:0')
        verified={};observations={};candidates={}
        for row in rows:
            observations[row['prompt_id']]=load_current_observation(row,verified_files=verified)
            rid=row['recording_id']
            if rid not in candidates:candidates[rid]=sealed(root/weak['inputs']['candidate_root']/rid/'event-candidates.json')
            audit_row(row,candidates[rid])
            if candidates[rid]['digest']!=data['source_candidates'][row['source_id']]['candidate_digest']:raise ValueError('Candidate binding differs')
        model,builder,allowlist,parent=build_eventground_from_m4(root,recipe,a.cache,device)
        sft_header=load_completed_sft(model,allowlist,protocol,Path(launch['sft_root']))
        reference=copy.deepcopy(model);reference.requires_grad_(False);reference.eval()
        trainer=EventGroundTrainer(model,allowlist,recipe,stage='rlvr')
        if trainer.optimizer.state or trainer.global_step:raise ValueError('Nonempty fresh optimizer')
        params=dict(model.named_parameters());frozen={n:t for n,t in params.items() if not t.requires_grad}
        frozen_sha=tensor_state_digest(frozen);reference_sha=tensor_state_digest(dict(reference.named_parameters()))
        identity=seal_contract(dict(artifact_type='eventground_wam.weak_rlvr_identity',launch_digest=launch['digest'],
            protocol_digest=protocol['digest'],stage=stage,sft_header_digest=sft_header['digest'],reference_sha256=reference_sha,
            initial_frozen_sha256=frozen_sha,fresh_optimizer=True,reference_independent=True,parent_identity=parent))
        (a.output/'identity.json').write_bytes(json_bytes(identity));report['identity_digest']=identity['digest']
        print(json.dumps(dict(phase='verified_SFT_and_reference',seconds=time.monotonic()-start)),flush=True)
        reward_records=[];replay=False
        def reward_function(evidence,tokens):
            predictions=serialize_tokens(tokens)
            if len(predictions)!=4*len(evidence):raise ValueError('Split candidate group')
            values=[]
            for i,row in enumerate(evidence):
                group=predictions[4*i:4*i+4];scores=[weak_reward(p,row,candidates[row['recording_id']],weights) for p in group]
                values.append(scores);reward_records.append(dict(prompt_id=row['prompt_id'],predictions=group,rewards=scores,recovery_replay=replay))
            return torch.tensor(values,dtype=torch.float32)
        def update(batch):
            nonlocal executed
            budget();tick=time.monotonic();micros=[]
            for row in batch:
                enc,state=collate_current_observations([observations[row['prompt_id']]],builder,device)
                micros.append((enc,state,[row]))
            result=trainer.rlvr_update(micros,prompt_count=len(batch),reference_model=reference,reward_function=reward_function)
            torch.cuda.synchronize();executed+=1
            result.update(step_seconds=time.monotonic()-tick,prompt_ids=[r['prompt_id'] for r in batch],recovery_replay=replay)
            budget();return result
        for index,batch in enumerate(batches):
            result=update(batch)
            with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(result,allow_nan=False)+'\n')
            print(json.dumps({k:v for k,v in result.items() if k!='prompt_ids'}),flush=True)
            if (stage=='qualification' and index==0) or (stage!='qualification' and trainer.global_step in (13,26)):
                ck=a.output/f'update-{trainer.global_step:06d}.pt';h=checkpoint(trainer,ck,identity,a.output,resources)
                report['checkpoints'].append(dict(file=ck.name,sha256=h,step=trainer.global_step))
            if stage=='qualification' and index==1:
                expected=recovery_digest(trainer);expected_metrics={k:v for k,v in result.items() if k not in ('step_seconds','recovery_replay')}
                trainer.resume(a.output/'update-000001.pt',identity=identity);replay=True
                actual_result=update(batch);actual=recovery_digest(trainer)
                actual_metrics={k:v for k,v in actual_result.items() if k not in ('step_seconds','recovery_replay')}
                report['recovery']=dict(expected_digest=expected,actual_digest=actual,metrics_identical=expected_metrics==actual_metrics)
                if expected!=actual or expected_metrics!=actual_metrics:raise ValueError('Recovery replay differs')
                with (a.output/'metrics.jsonl').open('a') as f:f.write(json.dumps(actual_result,allow_nan=False)+'\n')
        for path,(_,fingerprint) in verified.items():
            s=Path(path).stat()
            if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)!=fingerprint:raise ValueError('Input changed during run')
        if tensor_state_digest(frozen)!=frozen_sha or tensor_state_digest(dict(reference.named_parameters()))!=reference_sha:raise ValueError('Frozen/reference changed')
        verify_eventground_parent(sealed(root/'configs/training/week10_eventground_experiment_v3.json'))
        if sha(Path(launch['sft_root'])/'run/eventground-sft-epoch-2.pt')!=protocol['sft_endpoint_sha256']:raise ValueError('Stored SFT parent changed')
        state={n:params[n].detach().cpu().clone() for n in trainer.names};tensor_digest=tensor_state_digest(state)
        if tensor_digest==protocol['sft_endpoint_tensor_digest']:raise ValueError('No trainable parameters changed')
        header=seal_contract(dict(artifact_type='eventground_wam.weak_rlvr_inference',protocol_digest=protocol['digest'],identity_digest=identity['digest'],
            sft_header_digest=sft_header['digest'],stage=stage,exact_names=list(trainer.names),parameter_sha256=tensor_digest,
            global_step=trainer.global_step,prompts_seen=trainer.prompts_seen,independent_qa_passed=False,selection='fixed_final_update'))
        if stage!='qualification':
            h=save_payload(a.output/'endpoint.pt',dict(header=header,parameters=state),output=a.output,resources=resources)
            report.update(endpoint_file='endpoint.pt',endpoint_sha256=h,inference_header=header)
        encoded=json_bytes(dict(groups=reward_records,group_size=4,weights=list(weights)));require_output_capacity(a.output,resources,len(encoded))
        (a.output/'rewards.json').write_bytes(encoded)
        gates=dict(current_observation_only=True,group_four_rewards=True,exact_parameter_allowlist=True,finite_losses_gradients_parameters=True,
            frozen_parameters_unchanged=True,reference_unchanged=True,stored_parents_unchanged=True,parameters_changed=True,
            memory_below_90_percent=True,bitwise_checkpoint_recovery=stage!='qualification' or report['recovery']['metrics_identical'])
        if stage=='qualification':
            steps=[json.loads(x)['step_seconds'] for x in (a.output/'metrics.jsonl').read_text().splitlines()]
            report['projected_26_update_seconds']=max(steps)*26+120
            gates['pilot_feasible_within_1800_seconds']=report['projected_26_update_seconds']<1800
            if not all(gates.values()):raise ValueError('Qualification resource gate failed')
        report.update(gates=gates,status='passed_qualification' if stage=='qualification' else 'complete',final_parameter_sha256=tensor_digest,
            source_files_verified=len(verified),current_observations=len(observations))
    except Exception as e:
        report['error']=f'{type(e).__name__}: {e}'[:4000];raise
    finally:
        signal.alarm(0);elapsed=time.monotonic()-start
        report.update(elapsed_seconds=elapsed,executed_updates=executed,global_step=trainer.global_step if trainer else 0,
            prompts_seen=trainer.prompts_seen if trainer else 0,partial_update_in_progress=trainer._in_update if trainer else False,
            cumulative_active_gpu_seconds=resources['prior_active_gpu_seconds']+elapsed,finished_at=datetime.now(timezone.utc).isoformat())
        if torch.cuda.is_initialized():report.update(peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        write_completion(a.output,report,resources);print(json.dumps(dict(status=report['status'],elapsed_seconds=elapsed)),flush=True)
if __name__=='__main__':main()
