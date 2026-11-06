"""Fixed greedy predictions on a sealed weak event-dev manifest; no optimizer."""
from __future__ import annotations
import argparse
import json
import signal
import time
from pathlib import Path
import torch
from train_week10_eventground import check_gpu_exclusive,require_output_capacity,write_completion,json_bytes
from eventground_wam.b0_training import configure_deterministic_cuda,seed_training
from eventground_wam.demonstration_contracts import contract_digest
from eventground_wam.eventground_data import load_current_observation,collate_current_observations
from eventground_wam.eventground_model import serialize_tokens
from eventground_wam.eventground_runtime import build_eventground_from_m4,verify_eventground_parent
from eventground_wam.eventground_weak_supervision import sealed,sha
from eventground_wam.eventground_weak_evaluation import per_example,metrics
from eventground_wam.eventground_weak_rlvr import load_completed_sft
from eventground_wam.pixel_predictor import tensor_state_digest


def main():
    p=argparse.ArgumentParser()
    for k in ('root','launch','output','cache'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();launch=sealed(a.launch);root=a.root;protocol=sealed(root/'configs/training/week10_weak_rlvr_evaluation_v1.json')
    if launch['protocol_digest']!=protocol['digest'] or str(a.output.resolve())!=launch['output']:raise ValueError('Protocol/output mismatch')
    for name,h in launch['code_sha256'].items():
        if sha(root/name)!=h:raise ValueError('Changed implementation '+name)
    panel=Path(launch['panel_root']);manifest=sealed(panel/'manifest.json');labels=sealed(panel/'label-completion.json')
    if manifest['digest']!=launch['manifest_digest'] or labels['manifest_digest']!=manifest['digest'] or manifest['panel']!='event-dev':raise ValueError('Only fixed dev panel admitted')
    if manifest['selected_count']==0:raise ValueError('No complete balanced dev panel; gate cannot be evaluated')
    for name,h in labels['output_sha256'].items():
        if sha(panel/name)!=h:raise ValueError('Label evidence changed')
    check_gpu_exclusive();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();resources=launch['resources']
    report=dict(status='failed_preserved',artifact_type='eventground_wam.weak_event_dev_evaluation',launch_digest=launch['digest'],
        protocol_digest=protocol['digest'],manifest_digest=manifest['digest'],arm=launch['arm'],independent_qa_passed=False,training_updates=0)
    def budget():
        if time.monotonic()-start>resources['wall_seconds']:raise TimeoutError('Inference deadline')
        require_output_capacity(a.output,resources,0)
        if torch.cuda.is_initialized() and torch.cuda.max_memory_reserved()>.9*torch.cuda.get_device_properties(0).total_memory:raise RuntimeError('Memory ceiling')
    def stop(*_):raise TimeoutError('Bounded inference stopped')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(resources['wall_seconds'])
    try:
        budget();configure_deterministic_cuda();torch.set_num_threads(4);seed_training(17)
        torch.cuda.set_per_process_memory_fraction(.9);torch.cuda.reset_peak_memory_stats();device=torch.device('cuda:0')
        recipe=sealed(root/'configs/training/week10_eventground_numerical_v2.json')
        model,builder,allowlist,parent=build_eventground_from_m4(root,recipe,a.cache,device)
        load_completed_sft(model,allowlist,protocol,Path(launch['sft_root']))
        if launch['arm']!='SFT':
            attempt=Path(launch['attempt_root']);done=sealed(attempt/'completion.json')
            if done['status']!='complete' or done['global_step']!=26 or done['prompts_seen']!=404 or done['protocol_digest']!=protocol['digest'] or not all(done['gates'].values()):raise ValueError('Incomplete pilot endpoint')
            path=attempt/done['endpoint_file']
            if sha(path)!=done['endpoint_sha256'] or done['digest']!=launch['attempt_digest']:raise ValueError('Changed pilot endpoint')
            artifact=torch.load(path,map_location='cpu',weights_only=True);h=artifact['header'];state=artifact['parameters'];params=dict(model.named_parameters())
            names=sorted((*allowlist['head'],*allowlist['language_lora']))
            if h!=done['inference_header'] or h['digest']!=contract_digest(h) or sorted(state)!=names or tensor_state_digest(state)!=h['parameter_sha256']:raise ValueError('Export identity mismatch')
            with torch.no_grad():
                for n,t in state.items():
                    if t.shape!=params[n].shape or t.dtype!=params[n].dtype or not torch.isfinite(t).all():raise ValueError('Invalid pilot tensor')
                    params[n].copy_(t.to(device))
            del artifact,state
        model.requires_grad_(False);model.eval();before=tensor_state_digest(dict(model.named_parameters()))
        verified={};records=[];latencies=[]
        for index,row in enumerate(manifest['rows']):
            budget();observation=load_current_observation(row,verified_files=verified)
            encoded,state=collate_current_observations([observation],builder,device)
            torch.cuda.synchronize();tick=time.monotonic()
            with torch.no_grad():
                features=model(encoded,state)
                scores=model.head.sample(features,group_size=1,greedy=True)
                pred=serialize_tokens(scores.tokens)[0]
            torch.cuda.synchronize();elapsed=time.monotonic()-tick
            record=per_example(row,pred);record['forward_seconds']=elapsed
            # Marginalize all three positive event types, not the observed target.
            with torch.no_grad():
                event_probs=scores.event_logits.softmax(-1)[0]
                hypothetical=torch.tensor([[1,1,2],[2,1,2],[3,1,2]],dtype=torch.int64,device=device)
                start_probs=model.head.teacher_forced(features.repeat_interleave(3,0),hypothetical).start_logits.softmax(-1)
                onset=(event_probs[1:,None]*start_probs).sum(0)
                horizon_probs=torch.stack([event_probs[0],onset[:4].sum(),onset[4:8].sum(),onset[8:16].sum(),onset[16:].sum()])
                if not torch.isfinite(horizon_probs).all() or abs(float(horizon_probs.sum())-1)>1e-5:raise ValueError('Invalid marginal horizon probabilities')
            record['event_probabilities']=event_probs.cpu().tolist()
            record['horizon_bin_probabilities']=horizon_probs.cpu().tolist()
            records.append(record)
            if index>=5:latencies.append(elapsed)
            with (a.output/'predictions.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        if tensor_state_digest(dict(model.named_parameters()))!=before:raise ValueError('Inference mutated parameters')
        for path,(_,fingerprint) in verified.items():
            s=Path(path).stat()
            if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)!=fingerprint:raise ValueError('Input changed')
        verify_eventground_parent(sealed(root/'configs/training/week10_eventground_experiment_v3.json'))
        import numpy as np
        report.update(status='complete',metrics=metrics(records),predictions_sha256=sha(a.output/'predictions.jsonl'),
            input_files_verified=len(verified),current_observation_reads=len(records),future_model_input_reads=0,
            model_tensor_sha256=before,model_unchanged=True,
            latency=dict(excluded_first=5,n=len(latencies),median_ms=float(np.median(latencies)*1000) if latencies else None,
                p95_ms=float(np.quantile(latencies,.95)*1000) if latencies else None,scope='synchronized model forward only; preprocessing excluded'))
    except Exception as e:report['error']=repr(e);raise
    finally:
        signal.alarm(0);elapsed=time.monotonic()-start;report.update(elapsed_seconds=elapsed,cumulative_active_gpu_seconds=resources['prior_active_gpu_seconds']+elapsed)
        if torch.cuda.is_initialized():report.update(peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        write_completion(a.output,report,resources);print(json.dumps(report,allow_nan=False),flush=True)
if __name__=='__main__':main()
