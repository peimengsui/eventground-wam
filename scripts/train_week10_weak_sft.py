"""A fresh M4 child, exactly two passes, explicit exploratory weak-label admission."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
from train_week10_eventground import (
    BoundedWriter,
    CountingWriter,
    check_gpu_exclusive,
    json_bytes,
    require_output_capacity,
    write_completion,
)

from eventground_wam.b0_training import (
    capture_rng,
    configure_deterministic_cuda,
    cpu_tree,
    seed_training,
    sha256_file,
)
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.eventground_data import (
    collate_current_observations,
    load_current_observation,
    target_tokens,
)
from eventground_wam.eventground_runtime import build_eventground_from_m4, verify_eventground_parent
from eventground_wam.eventground_training import CHECKPOINT_FORMAT, EventGroundTrainer
from eventground_wam.eventground_weak_sft import admission, cursor, load_inputs, validate_launch
from eventground_wam.eventground_weak_supervision import sealed, sha
from eventground_wam.pixel_predictor import tensor_state_digest


def save_payload(path, payload, *, output, resources):
    """Use the previously tested exact-size bounded writer for each fresh payload."""
    path=Path(path)
    if path.exists():
        raise FileExistsError(path)
    counting=CountingWriter(resources["output_bytes"])
    torch.save(payload,counting)
    require_output_capacity(output,resources,counting.count+65)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb",buffering=0) as handle:
            writer=BoundedWriter(handle,counting.count,output=output,resources=resources,sidecar_reservation=65)
            torch.save(payload,writer);writer.flush();os.fsync(handle.fileno())
            if writer.count != counting.count:
                raise ValueError("Checkpoint serialization changed size")
        digest=sha256_file(temporary)
        if path.exists():
            raise FileExistsError(path)
        os.replace(temporary,path)
        with path.with_suffix(path.suffix+".sha256").open("x") as stream:
            stream.write(digest+"\n")
        return digest
    finally:
        temporary.unlink(missing_ok=True)


def save_checkpoint(trainer,path,identity,schedule,output,resources):
    if trainer._in_update:
        raise ValueError("Cannot checkpoint a partial update")
    position=cursor(trainer.global_step,trainer.prompts_seen,schedule)
    params=dict(trainer.model.named_parameters())
    payload={"format":CHECKPOINT_FORMAT,"identity":identity,"recipe_digest":trainer.recipe["digest"],
             "stage":trainer.stage,"exact_names":trainer.names,
             "parameters":{n:params[n].detach().cpu().clone() for n in trainer.names},
             "optimizer":cpu_tree(trainer.optimizer.state_dict()),"rng":capture_rng(),
             "global_step":trainer.global_step,"prompts_seen":trainer.prompts_seen}
    require_output_capacity(output,resources,16384)
    digest=save_payload(path,payload,output=output,resources=resources)
    marker=seal_contract({"artifact_type":"eventground_wam.week10_weak_sft_cursor",
                         "identity_digest":identity["digest"],"checkpoint_sha256":digest,**position})
    encoded=json_bytes(marker)
    require_output_capacity(output,resources,len(encoded))
    with Path(path).with_suffix(".pt.cursor.json").open("xb") as stream:
        stream.write(encoded)
    return digest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("root","launch","admission","cache","output"):
        parser.add_argument(f"--{name}",type=Path,required=True)
    args=parser.parse_args()
    launch=sealed(args.launch);data=load_inputs(args.root,launch)
    batches=validate_launch(args.root,launch,data)
    if sealed(args.admission) != admission(launch):
        raise ValueError("Wrong explicit weak-label SFT admission")
    if socket.gethostname()!=launch["host"] or str(args.output.resolve())!=launch["output"]:
        raise ValueError("Formal host or unique output destination differs")
    if sha(Path(__file__))!=launch["code_sha256"]["scripts/train_week10_weak_sft.py"]:
        raise ValueError("Executing runner differs from frozen implementation")
    check_gpu_exclusive()
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();trainer=None;latest=None
    resources=launch["resources"]
    report={"artifact_type":"eventground_wam.week10_weak_sft_completion","status":"failed_preserved",
            "launch_digest":launch["digest"],"admission_digest":sealed(args.admission)["digest"],
            "protocol_digest":launch["protocol_digest"],"dataset_digest":launch["dataset_digest"],
            "recipe_digest":launch["recipe_digest"],"schedule_digest":launch["schedule_digest"],
            "selection":launch["checkpoint_selection"],"independent_qa_passed":False,
            "heldout_payload_reads":0,"automatic_retries":0,"pilot_weights_reused":False,
            "initialization":launch["initialization"],"started_at":datetime.now(timezone.utc).isoformat(),
            "checkpoints":[],"completed_epochs":[]}

    def budget():
        if time.monotonic()-start>resources["wall_seconds"]:
            raise TimeoutError("Formal SFT wall ceiling")
        require_output_capacity(args.output,resources,0)
        if torch.cuda.is_initialized() and torch.cuda.max_memory_reserved()>.9*torch.cuda.get_device_properties(0).total_memory:
            raise RuntimeError("Formal SFT memory ceiling")

    def stop(*_):
        raise TimeoutError("Formal SFT stopped by bounded service; no automatic restart")

    signal.signal(signal.SIGALRM,stop);signal.signal(signal.SIGTERM,stop)
    signal.alarm(resources["wall_seconds"])
    try:
        budget()
        qualification_root=Path(launch["qualification_root"])
        if sealed(qualification_root/"completion.json")!=data["qualification"]:
            raise ValueError("Live qualification completion differs")
        for name,expected in data["qualification"]["files_sha256"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts or sha256_file(qualification_root/name)!=expected:
                raise ValueError(f"Qualification evidence changed: {name}")
        print(json.dumps({"phase":"qualification_verified","elapsed_seconds":time.monotonic()-start}),flush=True)
        verified_files={};observations={};cache_bytes=0
        for index,row in enumerate(data["dataset"]["main_rows"]):
            budget()
            observation=load_current_observation(row,verified_files=verified_files)
            cache_bytes+=observation.front.nbytes+observation.wrist.nbytes+observation.proprioception.nbytes
            if cache_bytes>resources["maximum_cached_observation_bytes"]:
                raise ValueError("CPU observation cache ceiling")
            observations[row["prompt_id"]]=observation
            if (index+1)%200==0:
                print(json.dumps({"phase":"current_observation_preload","prompts":index+1,
                                  "elapsed_seconds":time.monotonic()-start}),flush=True)
        report.update(current_observation_reads=len(observations),source_files_verified=len(verified_files),
                      cached_observation_bytes=cache_bytes,preload_seconds=time.monotonic()-start)
        configure_deterministic_cuda();torch.set_num_threads(resources["cpu_threads"]);seed_training(17)
        device=torch.device("cuda:0")
        torch.cuda.set_per_process_memory_fraction(resources["maximum_gpu_fraction"],device)
        torch.cuda.reset_peak_memory_stats()
        recipe=data["recipe"]
        model,builder,allowlist,parent=build_eventground_from_m4(args.root,recipe,args.cache,device)
        if {k:list(v) for k,v in allowlist.items()}!=recipe["exact_allowlist"]:
            raise ValueError("Exact trainable allowlist differs")
        trainer=EventGroundTrainer(model,allowlist,recipe,stage="sft")
        if trainer.global_step or trainer.prompts_seen or trainer.optimizer.state:
            raise ValueError("Formal SFT must start from a fresh optimizer")
        parameters=dict(model.named_parameters())
        frozen={n:p for n,p in parameters.items() if not p.requires_grad}
        initial_frozen=tensor_state_digest(frozen)
        initial_trainable=tensor_state_digest({n:parameters[n] for n in trainer.names})
        identity=seal_contract({"artifact_type":"eventground_wam.week10_weak_sft_identity","stage":"weak-sft",
                               "launch_digest":launch["digest"],"schedule_digest":launch["schedule_digest"],
                               "parent_identity":parent,"seed":17,"independent_qa_passed":False,
                               "initial_frozen_sha256":initial_frozen,"initial_trainable_sha256":initial_trainable,
                               "fresh_optimizer_verified":True,"pilot_weights_reused":False})
        require_output_capacity(args.output,resources,len(json_bytes(identity)))
        with (args.output/"identity.json").open("xb") as stream:stream.write(json_bytes(identity))
        report["identity_digest"]=identity["digest"]
        print(json.dumps({"phase":"fresh_M4_initialized","optimizer_updates":0,
                          "initial_trainable_sha256":initial_trainable,"elapsed_seconds":time.monotonic()-start}),flush=True)
        epoch_loss=0.;epoch_prompts=0
        for index,batch in enumerate(batches):
            budget();tick=time.monotonic();micro=[]
            for row in batch:
                encoded,state=collate_current_observations([observations[row["prompt_id"]]],builder,device)
                micro.append((encoded,state,target_tokens([row],device)))
            prepared=time.monotonic()
            metrics=trainer.sft_update(micro,prompt_count=len(batch))
            torch.cuda.synchronize();elapsed=time.monotonic()-tick
            del micro
            if trainer.global_step!=index+1:
                raise ValueError("Unexpected optimizer exposure")
            epoch_loss+=metrics["loss"]*len(batch);epoch_prompts+=len(batch)
            record={**metrics,"epoch":index//84+1,"batch_index":index%84,
                    "prompt_count":len(batch),"prompt_ids":[r["prompt_id"] for r in batch],
                    "preprocess_seconds":prepared-tick,"step_seconds":elapsed,
                    "elapsed_seconds":time.monotonic()-start,
                    "peak_reserved_bytes":torch.cuda.max_memory_reserved()}
            encoded_log=(json.dumps(record,allow_nan=False)+"\n").encode()
            require_output_capacity(args.output,resources,len(encoded_log))
            with (args.output/"metrics.jsonl").open("ab") as stream:
                stream.write(encoded_log);stream.flush()
            print(json.dumps({k:v for k,v in record.items() if k!="prompt_ids"}),flush=True)
            if trainer.global_step in launch["checkpoint_steps"]:
                latest=args.output/"checkpoints"/f"update-{trainer.global_step:06d}.pt"
                checksum=save_checkpoint(trainer,latest,identity,data["schedule"],args.output,resources)
                report["checkpoints"].append({"global_step":trainer.global_step,"file":str(latest.relative_to(args.output)),"sha256":checksum})
            if trainer.global_step%84==0:
                if epoch_prompts!=1332:
                    raise ValueError("Epoch prompt count changed")
                report["completed_epochs"].append({"epoch":trainer.global_step//84,"prompts":epoch_prompts,
                                                    "mean_training_loss":epoch_loss/epoch_prompts})
                epoch_loss=0.;epoch_prompts=0
        if trainer.global_step!=168 or trainer.prompts_seen!=2664:
            raise ValueError("Fixed SFT endpoint exposure differs")
        if tensor_state_digest(frozen)!=initial_frozen:
            raise ValueError("Frozen parameters changed")
        # Cached inputs retain a verified file identity for the duration of this run.
        for path,(_,fingerprint) in verified_files.items():
            s=Path(path).stat()
            if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)!=fingerprint:
                raise ValueError("Cached source changed during SFT")
        study=sealed(args.root/data["protocol"]["inputs"]["parent_study"])
        verify_eventground_parent(study)
        state={n:parameters[n].detach().cpu().clone() for n in trainer.names}
        state_digest=tensor_state_digest(state)
        if state_digest==initial_trainable:
            raise ValueError("No trainable parameters changed")
        header=seal_contract({"artifact_type":"eventground_wam.week10_weak_sft_inference",
            "identity_digest":identity["digest"],"launch_digest":launch["digest"],
            "recipe_digest":recipe["digest"],"dataset_digest":launch["dataset_digest"],
            "parent_policy_sha256":parent["parent_policy_sha256"],"parameter_sha256":state_digest,
            "exact_names":list(trainer.names),"epoch":2,"global_step":168,"prompts_seen":2664,
            "selection":launch["checkpoint_selection"],"independent_qa_passed":False})
        endpoint=args.output/"eventground-sft-epoch-2.pt"
        endpoint_sha=save_payload(endpoint,{"header":header,"parameters":state},output=args.output,resources=resources)
        budget()
        report.update(status="complete",endpoint_file=endpoint.name,endpoint_sha256=endpoint_sha,
                      inference_header=header,frozen_parameters_unchanged=True,stored_M4_parent_unchanged=True,
                      source_files_unchanged=True,finite_updates=True,final_parameter_sha256=state_digest)
    except Exception as error:
        report["error"]=f"{type(error).__name__}: {error}"[:4096]
        raise
    finally:
        signal.alarm(0)
        report.update(global_step=trainer.global_step if trainer else 0,prompts_seen=trainer.prompts_seen if trainer else 0,
                      partial_update_in_progress=trainer._in_update if trainer else False,
                      elapsed_seconds=time.monotonic()-start,
                      cumulative_active_gpu_seconds=resources["prior_active_gpu_seconds"]+time.monotonic()-start,
                      prior_output_bytes=resources["prior_output_bytes"],
                      last_recovery_checkpoint=str(latest) if latest else None,
                      finished_at=datetime.now(timezone.utc).isoformat())
        if torch.cuda.is_initialized():
            report["peak_reserved_bytes"]=torch.cuda.max_memory_reserved()
            report["gpu_total_bytes"]=torch.cuda.get_device_properties(0).total_memory
        report["metadata_files_sha256"]={str(p.relative_to(args.output)):sha(p) for p in args.output.rglob("*") if p.is_file() and p.suffix!=".pt"}
        write_completion(args.output,report,resources)
        print(json.dumps({"status":report["status"],"global_step":report["global_step"],
                          "elapsed_seconds":report["elapsed_seconds"]}),flush=True)


if __name__=="__main__":
    main()
