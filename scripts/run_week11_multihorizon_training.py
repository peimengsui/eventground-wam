"""One guarded worker phase of the frozen Week 11 matched continuation."""
from __future__ import annotations

import argparse
import gc
import os
import signal
import statistics
import time
from dataclasses import replace
from pathlib import Path

import torch
from week11_multihorizon_runtime import (
    StageGuard,
    admission,
    clear,
    data_and_models,
    emit,
    engine_for,
    export_endpoint,
    initial_digest,
    require,
    runtime_identity,
    targets,
    write,
)

from eventground_wam.auxiliary_data import AuxiliaryTargetCache
from eventground_wam.auxiliary_policy import ObservationFeatures
from eventground_wam.b0_training import (
    CheckpointStore,
    configure_deterministic_cuda,
    move_batch,
    read_sealed,
)
from eventground_wam.coupled_checkpoint import export_coupled_policy, load_coupled_policy
from eventground_wam.coupled_policy import LOSS_WEIGHTS, CoupledPolicy, coupled_full_batch_backward
from eventground_wam.pixel_predictor import PixelCondition, tensor_state_digest
from eventground_wam.pixel_training import tree_digest
from eventground_wam.week11_multihorizon_plan import projected_seconds
from eventground_wam.week11_multihorizon_training import ARMS
from eventground_wam.week11_resources import require_idle_gpu


def forward(engine, batch, horizons=None, *, original=False):
    inputs, state, actions = move_batch(batch, engine.device)
    images = batch['future_inputs'].current_images.to(engine.device)
    if original:
        return CoupledPolicy.forward_with_features(engine.model, inputs, state, images,
                                                    normalized_action_targets=actions)
    engine.model.candidate_horizons = horizons or engine.horizons(batch['index'])
    try:
        return engine.model.forward_with_features(inputs, state, images, normalized_action_targets=actions)
    finally:
        engine.model.candidate_horizons = None


def gradient_reference(engine, result, future, truth):
    from check_auxiliary_training import objective
    loss = objective(engine.predictor, 'jepa', future.condition(
        ObservationFeatures(result.features.values.float(), result.features.padding)), truth.detach(), future.anchor_indices)
    above = (list(engine.model.policy.action_decoder.parameters())+list(engine.model.coupling.parameters())
             +list(engine.predictor.parameters()))
    with torch.backends.cudnn.flags(allow_tf32=False):
        gradients = torch.autograd.grad(result.action_output.loss+LOSS_WEIGHTS['M4']*loss,
                                        [result.features.values, *above])
        for p,g in zip(above, gradients[1:], strict=True): p.grad = g.detach().clone()
    result.features.values.backward(gradients[0])


def correctness(engine, batches, cache, output):
    b = batches[0]
    truth = targets(b, cache).detach().requires_grad_(True)
    future = b['future_inputs'].to('cuda')
    with torch.no_grad():
        original = forward(engine, b, original=True)
        raw = original.action_output.raw_action_predictions.detach().cpu()
        h8 = forward(engine, b, [8]*len(b['tokens']))
        require(torch.equal(raw, h8.action_output.raw_action_predictions.cpu()), 'h8 forward changed')
        require(torch.equal(original.future_tokens, h8.future_tokens), 'h8 future changed')
    del original, h8
    comparisons = {}
    # Compare the inherited full-objective reference against the inherited
    # production backward under each routed condition, including h8/no hook.
    for name, selected in [('h8', [8]*len(b['tokens'])), ('scheduled', engine.horizons(0))]:
        clear(engine)
        result = forward(engine, b, selected, original=name=='h8')
        gradient_reference(engine, result, future, truth)
        expected = [p.grad.detach().cpu().clone() for p in engine.parameters]
        del result
        clear(engine)
        result = forward(engine, b, selected)
        coupled_full_batch_backward(engine.model, result, engine.predictor, future, truth)
        from check_auxiliary_training import compare_gradients
        comparison = compare_gradients(expected, [p.grad.detach().cpu() for p in engine.parameters])
        require(comparison['relative_l2'] <= .01, 'full objective gradient mismatch')
        if name == 'h8':
            require(all(torch.equal(a,p.grad.cpu()) for a,p in zip(expected,engine.parameters,strict=True)),
                    'h8 backward does not reproduce exactly')
        require(truth.grad is None and not result.candidate_actions.requires_grad, 'target/proposal gradient leakage')
        require(not engine.model.future_core._forward_pre_hooks, 'candidate hook leaked into backward')
        require(all(p.grad is None for p in engine.model.policy.parameters() if not p.requires_grad), 'frozen base gradient')
        comparisons[name] = comparison
        del result, expected
    clear(engine)
    with torch.no_grad():
        result = forward(engine, b)
        reference = result.action_output.raw_action_predictions.cpu()
        # Independently build predictor conditions with the scheduled horizon
        # vector to verify that all candidate microbatch slices use the plan.
        cond = PixelCondition(result.features.values, result.features.padding, future.current_images,
            future.normalized_state, result.candidate_actions,
            torch.tensor(engine.horizons(0), device='cuda', dtype=torch.int64))
        with torch.backends.cudnn.flags(allow_tf32=False):
            direct = torch.cat([engine.predictor(PixelCondition(**{k:v[a:a+10] for k,v in vars(cond).items()})).future_tokens
                                for a in range(0, len(b['tokens']), 10)])
        require(torch.equal(direct, result.future_tokens), 'mixed vector is misaligned with candidate microbatches')
        altered = dict(b, targets=-b['targets'], future_mask=~b['future_mask'])
        require(torch.equal(forward(engine, altered).action_output.raw_action_predictions.cpu(), reference), 'target or mask leakage')
        engine.predictor(replace(future, actions=-future.actions,
            horizons=torch.full_like(future.horizons,2)).condition(result.features))
        require(torch.equal(forward(engine,b).action_output.raw_action_predictions.cpu(),reference), 'teacher branch affects actions')
    del result, direct, cond, future, truth
    # Real short-batch edge probes retain missing h16/h32 targets as-is.
    probes = sorted({0, min(range(len(batches)), key=lambda i:len(batches.plan[i])),
                     next(i for i,b in enumerate(batches.plan) if len(b)==23)})
    edges=[]
    for index in probes:
        b=batches[index]; truth=targets(b,cache); clear(engine)
        result=forward(engine,b)
        coupled_full_batch_backward(engine.model,result,engine.predictor,b['future_inputs'].to('cuda'),truth)
        require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in engine.parameters),'edge gradient invalid')
        edges.append({'index':index,'anchors':b['sample_count'],'future_pairs':len(truth),
                      'missing_future_slots':int((~b['future_mask']).sum()),'finite_gradients':True})
        del result,truth
    clear(engine)
    engine.model.eval()
    b=batches[0]
    with torch.no_grad(): expected=forward(engine,b).action_output.raw_action_predictions.cpu()
    path=output/'diagnostic-export.pt'
    export_coupled_policy(path,engine.model,engine.exact_lora_names,metadata={'diagnostic_only':True})
    with torch.no_grad(): engine.model.coupling.output.weight.zero_()
    load_coupled_policy(path,engine.model,engine.exact_lora_names)
    with torch.no_grad(): require(torch.equal(forward(engine,b).action_output.raw_action_predictions.cpu(),expected),'export/reload differs')
    return {'h8_forward_exact':True,'h8_backward_exact':True,'gradient_comparisons':comparisons,
            'scheduled_vector_token_parity':True,'target_mask_isolation':True,'detached_proposal':True,
            'edge_probes':edges,'export_reload_exact':True,'passed':True}


def recovery(engine,batches,cache,output,phase):
    store=CheckpointStore(output/'process-recovery',engine.identity_digest,keep=3)
    if phase=='resume':
        engine.restore(store.load())
        require(engine.global_step==2,'wrong process interruption cursor')
    for i in range(engine.global_step,4):
        b=batches[i]; truth=targets(b,cache)
        if phase=='interrupt' and i==2:
            def kill(*_):
                write(output/'interrupt-marker.json',{'global_step':engine.global_step,'partial_update':True})
                os.kill(os.getpid(),signal.SIGKILL)
            engine.step(b,truth,hook=kill)
        else: engine.step(b,truth)
        if phase=='interrupt' and i==1: store.save(engine.snapshot())
    snapshot=engine.snapshot()
    digest=tree_digest(snapshot)
    if phase=='resume':
        expected=read_sealed(output/'uninterrupted.json')['state_digest']
        require(digest==expected,'fresh-process resume differs from uninterrupted state')
        damaged=store.save(snapshot)
        (store.root/damaged['checkpoint']).write_bytes(b'intentionally torn diagnostic checkpoint')
        intact=store.load(); require(intact['global_step']==2 and store.recovery_notes,'fallback failed')
        engine.restore(intact)
        cache=AuxiliaryTargetCache(cache.teacher,maximum_bytes=8*2**30)
        for i in range(2,4):
            b=batches[i]; engine.step(b,targets(b,cache))
        require(tree_digest(engine.snapshot())==expected,'fallback replay differs')
    return {'state_digest':digest,'global_step':4,'process_resume_exact':phase=='resume',
            'damaged_latest_fallback_exact':phase=='resume','passed':True}


def timing(engine,batches,cache,output,guard):
    batches.pairs._cache.clear(); gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    rows=[]
    for i in range(128):
        guard.check(); torch.cuda.synchronize(); start=time.monotonic()
        b=batches[i]; truth=targets(b,cache)
        row=engine.step(b,truth); torch.cuda.synchronize()
        rows.append({**row,'seconds':time.monotonic()-start})
        if (i+1)%8==0: emit(output,'timing',step=i+1,seconds=rows[-1]['seconds'])
        del b,truth
    started=time.monotonic()
    store=CheckpointStore(output/'timing-recovery',engine.identity_digest,keep=3)
    store.save(engine.snapshot())
    save_seconds=time.monotonic()-started
    times=[r['seconds'] for r in rows]; mean=statistics.mean(times)
    ordered=sorted(times); pos=.95*(len(times)-1); low=int(pos)
    p95=ordered[low]+(pos-low)*(ordered[low+1]-ordered[low])
    return {'rows':rows,'mean_step_seconds':mean,'p95_step_seconds':p95,'checkpoint_seconds':save_seconds,
            'conservative_projection':projected_seconds(2000,max(mean,p95),save_seconds,margin=1.2),
            'mean_projection':projected_seconds(2000,mean,save_seconds),'pilot_weights_discarded':True,'passed':True}


def train(engine,batches,cache,output,guard,cfg,identity,qualification):
    require(qualification['passed'] and qualification['design_digest']==cfg['digest'],'unqualified training')
    store=CheckpointStore(output/'recovery',engine.identity_digest,keep=3)
    recovered=store.load()
    if recovered:
        engine.restore(recovered); del recovered
    else:
        require(not (output/'recovery-created.json').exists(),'all recovery records missing')
        store.save(engine.snapshot()); write(output/'recovery-created.json',{'identity_digest':identity['digest']})
    timer=time.monotonic()
    reference=read_sealed(output.parents[1]/'qualification'/engine.continuation_arm/'resources.json')['rows']
    with (output/'updates.jsonl').open('a') as log:
        while engine.global_step<2000:
            guard.check(); start=time.monotonic(); b=batches[engine.global_step]
            row=engine.step(b,targets(b,cache)); torch.cuda.synchronize()
            row['seconds']=time.monotonic()-start
            if row['global_step']<=128:
                old=reference[row['global_step']-1]
                require(all(row[k]==old[k] for k in ('action_loss','future_loss','gradient_norm','horizon_chain','learning_rates')),
                        'production differs from its qualification stream')
            import json
            log.write(json.dumps(row)+'\n'); log.flush()
            if engine.global_step%8==0: emit(output,'training',step=engine.global_step,total=2000,
                action_loss=row['action_loss'],future_loss=row['future_loss'],seconds=row['seconds'])
            if time.monotonic()-timer>=300 or engine.global_step%2000==0:
                store.save(engine.snapshot());timer=time.monotonic()
            del b
    endpoint=export_endpoint(output/'endpoint',engine,identity,cfg)
    return {'passed':True,'endpoint_digest':endpoint['digest'],'continuation_updates':2000,
            'sample_chain':engine.sample_chain,'future_chain':engine.future_chain,'horizon_chain':engine.horizon_chain}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--arm',choices=ARMS,required=True)
    parser.add_argument('--phase',choices=['correctness','uninterrupted','interrupt','resume','resources','train','verify-endpoint'],required=True)
    args=parser.parse_args()
    configure_deterministic_cuda();torch.set_num_threads(1);require_idle_gpu()
    cfg,source,schedule=admission(); rt=runtime_identity()
    stage='training' if args.phase in ('train','verify-endpoint') else 'qualification'
    stage_root=args.root/stage;output=stage_root/args.arm;output.mkdir(parents=True,exist_ok=True)
    guard=StageGuard(stage_root,cfg['digest'],cfg,stage)
    started=time.monotonic();report={'passed':False,'phase':args.phase,'arm':args.arm,'runtime':rt,
                                   'design_digest':cfg['digest'],'source_digest':source['digest']}
    try:
        emit(output,'initializing',worker_phase=args.phase)
        values=data_and_models(cfg,schedule)
        _base,batches,model,_predictor,names,teacher,teacher_hash,cache,setup_seconds=values
        engine,identity=engine_for(values,cfg,source,rt,args.arm,stage)
        report.update(identity=identity,initial_tensors=initial_digest(engine),initialization_seconds=setup_seconds)
        if args.phase=='correctness': report.update(correctness(engine,batches,cache,output))
        elif args.phase in ('uninterrupted','interrupt','resume'): report.update(recovery(engine,batches,cache,output,args.phase))
        elif args.phase=='resources': report.update(timing(engine,batches,cache,output,guard))
        elif args.phase=='train':
            qualified=read_sealed(args.root/'qualification/qualification.json')
            require(qualified['source_digest']==source['digest'] and qualified['runtime_digest']==rt['digest'],'qualification runtime/source differs')
            require(report['initial_tensors']==qualified['initial_tensors'],'wrong training starting tensors')
            report.update(train(engine,batches,cache,output,guard,cfg,identity,qualified))
        else:
            endpoint=read_sealed(output/'endpoint/endpoint.json')
            from eventground_wam.b0_training import sha256_file
            require(endpoint['identity']==identity and endpoint['metadata']['continuation_updates']==2000,'endpoint identity differs')
            require(sha256_file(output/'endpoint/policy.pt')==endpoint['policy_sha256'],'endpoint bytes differ')
            loaded=load_coupled_policy(output/'endpoint/policy.pt',model,names)
            require(loaded['digest']==endpoint['header_digest'],'endpoint header differs')
            store=CheckpointStore(output/'recovery',engine.identity_digest,keep=3)
            saved=store.load();require(saved['global_step']==2000,'final recovery endpoint missing')
            model.eval()
            with torch.no_grad(): expected=forward(engine,batches[0],[8]*32).action_output.raw_action_predictions.cpu()
            engine.restore(saved);model.eval()
            with torch.no_grad(): require(torch.equal(expected,forward(engine,batches[0],[8]*32).action_output.raw_action_predictions.cpu()),'fresh endpoint differs from full training state')
            report.update(passed=True,fresh_process_endpoint_reload_exact=True,endpoint_digest=endpoint['digest'])
        require(tensor_state_digest(teacher.visual.state_dict())==teacher_hash,'teacher changed')
        report['teacher_unchanged']=True
    except BaseException as exc:
        report.update(error=f'{type(exc).__name__}: {exc}');raise
    finally:
        report.update(elapsed_seconds=time.monotonic()-started,resources=guard.close())
        write(output/(args.phase+'.json'),report)
        emit(output,'phase_complete' if report['passed'] else 'stopped',worker_phase=args.phase,passed=report['passed'])


if __name__=='__main__': main()
