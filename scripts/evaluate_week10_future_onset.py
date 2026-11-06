"""Current-observation-only evaluation with the versioned next-onset target audit."""
from __future__ import annotations
import argparse
import json
import signal
import time
from pathlib import Path
import numpy as np
import torch
from train_week10_eventground import check_gpu_exclusive,require_output_capacity,write_completion
from eventground_wam.b0_training import configure_deterministic_cuda,seed_training
from eventground_wam.eventground_data import load_current_observation,collate_current_observations
from eventground_wam.eventground_model import serialize_tokens
from eventground_wam.eventground_runtime import build_eventground_from_m4,verify_eventground_parent
from eventground_wam.eventground_future_onset import TASK,audit_row
from eventground_wam.eventground_future_runtime import verify,load_endpoint
from eventground_wam.eventground_weak_supervision import sealed,sha
from eventground_wam.eventground_weak_evaluation import per_example,metrics
from eventground_wam.pixel_predictor import tensor_state_digest


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--arm',choices=['sft','primary','calibration','main','control'],required=True)
    parser.add_argument('--panel',choices=['dev','report'],required=True);a=parser.parse_args();base=a.base;root=base/'source'
    p,index,manifests,recipe=verify(base);manifest=manifests[a.panel]
    if a.panel=='report':
        decision=sealed(base/'decisions/final.json')
        if decision['protocol_digest']!=p['digest'] or a.arm not in (['sft','main','control'] if decision['promoted'] else ['sft']):raise ValueError('Report gate/arm differs')
    candidates={r['candidate']['recording_id']:r['candidate'] for r in index['panels'][a.panel]}
    for row in manifest['rows']:audit_row(row,candidates[row['recording_id']])
    check_gpu_exclusive();out=base/'eval'/a.panel/a.arm;out.mkdir(parents=True,exist_ok=False)
    resources=p['inference_resources'];start=time.monotonic()
    report=dict(status='failed_preserved',artifact_type='eventground_wam.future_onset_evaluation',task_definition=TASK,
        protocol_digest=p['digest'],manifest_digest=manifest['digest'],arm=a.arm,panel=a.panel,independent_qa_passed=False,training_updates=0)
    def budget():
        if time.monotonic()-start>resources['wall_seconds']:raise TimeoutError('Inference ceiling')
        require_output_capacity(out,resources,0)
        if torch.cuda.is_initialized() and torch.cuda.max_memory_reserved()>.9*torch.cuda.get_device_properties(0).total_memory:raise RuntimeError('Memory ceiling')
    def stop(*_):raise TimeoutError('Inference stopped')
    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop);signal.alarm(resources['wall_seconds'])
    try:
        configure_deterministic_cuda();torch.set_num_threads(4);seed_training(17);torch.cuda.set_per_process_memory_fraction(.9)
        torch.cuda.reset_peak_memory_stats();device=torch.device('cuda:0')
        model,builder,allowlist,parent=build_eventground_from_m4(root,recipe,Path(p['cache']),device)
        completion=load_endpoint(model,allowlist,base,a.arm);report['endpoint_completion_digest']=completion['digest']
        model.requires_grad_(False);model.eval();before=tensor_state_digest(dict(model.named_parameters()))
        verified={};records=[];latencies=[]
        for i,row in enumerate(manifest['rows']):
            budget();observation=load_current_observation(row,verified_files=verified)
            encoded,state=collate_current_observations([observation],builder,device)
            torch.cuda.synchronize();tick=time.monotonic()
            with torch.no_grad():
                features=model(encoded,state);scores=model.head.sample(features,group_size=1,greedy=True);pred=serialize_tokens(scores.tokens)[0]
            torch.cuda.synchronize();elapsed=time.monotonic()-tick
            record=per_example(row,pred);record['forward_seconds']=elapsed
            with torch.no_grad():
                event_probs=scores.event_logits.softmax(-1)[0]
                hypothetical=torch.tensor([[1,1,2],[2,1,2],[3,1,2]],dtype=torch.int64,device=device)
                start_probs=model.head.teacher_forced(features.repeat_interleave(3,0),hypothetical).start_logits.softmax(-1)
                onset=(event_probs[1:,None]*start_probs).sum(0)
                horizon_probs=torch.stack([event_probs[0],onset[:4].sum(),onset[4:8].sum(),onset[8:16].sum(),onset[16:].sum()])
                if not torch.isfinite(horizon_probs).all() or abs(float(horizon_probs.sum())-1)>1e-5:raise ValueError('Invalid probabilities')
            record['event_probabilities']=event_probs.cpu().tolist();record['horizon_bin_probabilities']=horizon_probs.cpu().tolist()
            records.append(record)
            if i>=5:latencies.append(elapsed)
            with (out/'predictions.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
        if tensor_state_digest(dict(model.named_parameters()))!=before:raise ValueError('Inference mutated parameters')
        for path,(_,fingerprint) in verified.items():
            s=Path(path).stat()
            if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)!=fingerprint:raise ValueError('Input changed')
        verify_eventground_parent(sealed(root/'configs/training/week10_eventground_experiment_v3.json'))
        report.update(status='complete',metrics=metrics(records),predictions_sha256=sha(out/'predictions.jsonl'),
            input_files_verified=len(verified),current_observation_reads=len(records),future_model_input_reads=0,
            model_tensor_sha256=before,model_unchanged=True,targets_reaudited=True,
            latency=dict(excluded_first=5,n=len(latencies),median_ms=float(np.median(latencies)*1000),
                p95_ms=float(np.quantile(latencies,.95)*1000),scope='synchronized forward only, preprocessing and calibration diagnostics excluded'))
        budget()
    except Exception as e:report.update(status='failed_preserved',error=repr(e));raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.monotonic()-start
        if torch.cuda.is_initialized():report.update(peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory)
        write_completion(out,report,resources);print(json.dumps({k:v for k,v in report.items() if k!='confusion'},allow_nan=False),flush=True)
if __name__=='__main__':main()
