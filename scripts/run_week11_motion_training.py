"""Qualification, recoverable adapter training, and independent endpoint checks."""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from run_week11_continuation_replay import observation
from week11_motion_adapter import ARMS, FORMAT, FrozenMotionPolicy, MotionEngine, MotionInference
from week11_motion_runtime import load_batches, verify_training_payloads
from week11_replay_common import read, sha
from week11_replay_common import write as sealed_write

from eventground_wam.auxiliary_data import AuxiliaryBatches
from eventground_wam.b0_training import (
    CheckpointStore,
    atomic_torch_save,
    configure_deterministic_cuda,
    seed_training,
)
from eventground_wam.coupled_inference import CoupledInferencePolicy
from eventground_wam.pixel_predictor import tensor_state_digest
from eventground_wam.pixel_training import tree_digest
from eventground_wam.week11_resources import ResourceGuard, require_idle_gpu


def write(path, value):
    sealed_write(path, value)
    return read(path)


def emit(output, phase, **fields):
    row = {'phase': phase, 'time_unix': time.time(), **fields}
    write(output/'progress.json', row); print(json.dumps(row), flush=True)


def action_adapter(action, model):
    return CoupledInferencePolicy(policy=model, input_builder=action._input_builder,
        normalizer=action._normalizer, device=action._device, checkpoint=action.checkpoint)


def forced(action, obs, horizon):
    action._policy.horizons = [horizon]
    try:
        return action.predict_chunk(**obs)
    finally:
        action._policy.horizons = None


def replay_inputs(cfg):
    root = Path(cfg['replay_root'])
    contract = read(root/'bundle/contract.json')
    assert contract['digest'] == cfg['replay_contract_digest']
    result = read(root/'replay/replay.json')
    assert result['digest'] == cfg['replay_result_digest']
    for name, digest in contract['input_sha256'].items():
        assert sha(name) == digest, name
    rows = []
    for row in result['observations']:
        if row['evaluated_model'] != 'h8-continuation':
            continue
        obs, _, q, digest = observation(row)
        assert digest == row['input_digest']
        path = root/'replay'/row['tensor_file']; assert sha(path) == row['tensor_sha256']
        with np.load(path) as f:
            expected = f['h8_decoded_actions'].copy()
        rows.append((row['id'], obs, expected, q))
    assert len(rows) == 456
    return rows


def parity(action, rows, output, guard, *, zero):
    checks = []; calls = 0
    for i, (name, obs, expected, _) in enumerate(rows):
        hs = (8, 2, 4, 16, 32) if zero else (8, (2, 4, 16, 32)[i % 4])
        for horizon in hs:
            guard.check(); got = forced(action, obs, horizon); calls += 1
            assert np.array_equal(got[:, 6], expected[:, 6]), ('gripper', name, horizon)
            if horizon == 8 or zero:
                assert np.array_equal(got, expected), ('H8/zero parity', name, horizon)
        checks.append({'id': name, 'h8_exact': True, 'gripper_exact': True,
                       'all_horizons_zero_parity': zero})
        if i % 100 == 0:
            emit(output.parent, 'parity', file=output.name, completed=i+1, total=456)
    return write(output, {'passed': True, 'calls': calls, 'rows': checks})


def latency(cfg, actions, action, event_recipe, rows, output, guard):
    import run_week11_pilot as pilot

    from eventground_wam.eventground_inference import require_disjoint_models
    # A separate original allocation permits interleaved fresh controls without checkpoint swapping.
    _, original_contract, original_endpoint, _ = pilot.endpoint_check(
        read(pilot.ROOT/'configs/evaluation/week11_integration_v1.json'))
    original = pilot.load_action(original_contract, original_endpoint)
    event = pilot.load_event(event_recipe)
    require_disjoint_models(action._policy, event.model)
    event_hash = tensor_state_digest(event.model.state_dict())
    adaptive = MotionInference(action, event, guard=guard.check)
    values = {'original-fixed': [], ARMS[0]: [], ARMS[1]: []}; details = []
    indices = cfg['latency_observation_indices']
    labels = list(values)
    for index in [-1]*12 + list(range(len(indices))):
        slot = max(index, 0)
        name, obs, _, recorded = rows[indices[slot]]
        for k in range(3):
            label = labels[(slot+k) % 3]
            guard.check(); torch.cuda.synchronize(); tick = time.perf_counter()
            if label == 'original-fixed':
                original.predict_chunk(**obs)
            else:
                action._policy.arm = label; adaptive.predict_chunk(**obs)
                assert adaptive.last_query['prediction'] == recorded['prediction']
                assert adaptive.last_query['event']['categorical_probabilities'] == recorded['event']['categorical_probabilities']
            torch.cuda.synchronize(); ms = (time.perf_counter()-tick)*1000
            if index >= 0:
                values[label].append(ms); details.append({'id': name, 'condition': label, 'ms': ms})
        if index >= 0 and index % 32 == 0:
            emit(output, 'latency', completed=index+1, total=len(indices))
    assert tensor_state_digest(event.model.state_dict()) == event_hash
    summary = {k:{'queries':len(v), 'median_ms':float(np.median(v)), 'p95_ms':float(np.percentile(v,95))}
               for k,v in values.items()}
    ratios = {arm:summary[arm]['p95_ms']/summary['original-fixed']['p95_ms'] for arm in ARMS}
    passed = all(r <= 2 for r in ratios.values())
    result = write(output/'latency.json', {'passed':passed, 'summary':summary, 'ratios':ratios,
        'rows':details, 'warmup_rounds':12, 'event_unchanged':True, 'event_state_sha256':event_hash,
        'condition_order':'rotating-interleaved-on-identical-observations'})
    del adaptive, event, original; gc.collect(); torch.cuda.empty_cache()
    assert passed, ('two-times latency ceiling', ratios)
    action._policy.arm = ARMS[0]
    return result


def make_engine(model, batches, cfg, arm, mode):
    identity = {'artifact_type': FORMAT, 'contract_digest':cfg['digest'], 'arm':arm,
                'mode':mode, 'reference_sha256':cfg['h8_endpoint']['policy_sha256']}
    from eventground_wam.demonstration_contracts import seal_contract
    identity = seal_contract(identity)
    model.arm = arm
    seed_training(17)
    return MotionEngine(model, batches.plan, identity['digest'], learning_rate=cfg['optimizer']['learning_rate']), identity


def qualify(cfg, actions, action, batches, rows, recipe, output, guard, frozen):
    model = action._policy
    initial = {k:v.detach().cpu().clone() for k,v in model.adapter.state_dict().items()}
    assert sum(p.numel() for p in model.adapter.parameters()) == cfg['adapter']['parameters']
    # Compare the lighter loader with the inherited collator before using any training updates.
    for i in (0, 1):
        old = AuxiliaryBatches.__getitem__(batches, i); new = batches[i]
        assert all(torch.equal(old['inputs'][k],v) for k,v in new['inputs'].items())
        for k in ('targets','proprioception'):
            assert torch.equal(old[k], new[k])
        assert old['identity_digest'] == new['identity_digest']
        assert torch.equal(old['future_inputs'].current_images, new['current_images'])
    write(output/'collator-parity.json', {'passed':True,'batches':[0,1], 'train_only':True,
        'inputs_actions_proprioception_images_identities_exact':True})
    lat = latency(cfg, actions, action, recipe, rows, output, guard)
    results = {}
    for arm in ARMS:
        dest = output/arm; dest.mkdir(exist_ok=True)
        model.adapter.load_state_dict(initial); seed_training(17)
        engine, identity = make_engine(model,batches,cfg,arm,'qualification')
        write(dest/'identity.json',identity)
        parity(action, rows, dest/'initial-parity.json', guard, zero=True)
        start = engine.snapshot()
        engine.step(batches[0]); engine.step(batches[1]); saved = engine.snapshot()
        store = CheckpointStore(dest/'restart',engine.identity)
        store.save(saved)
        engine.step(batches[2]); expected = tree_digest(engine.snapshot())
        engine.restore(store.load()); engine.step(batches[2])
        assert tree_digest(engine.snapshot()) == expected
        after = engine.snapshot()
        def interrupt(): raise RuntimeError('intentional-before-optimizer')
        try:
            engine.step(batches[3],hook=interrupt)
        except RuntimeError as exc:
            assert str(exc) == 'intentional-before-optimizer'
        else:
            raise AssertionError('missing interruption')
        try:
            engine.snapshot()
        except ValueError:
            pass
        else:
            raise AssertionError('partial checkpoint accepted')
        engine.restore(after); assert tree_digest(engine.snapshot()) == expected
        write(dest/'recovery-reference.json',{'expected_after_next_step':expected, 'saved_step':2,
            'next_step':3, 'in_process_restart_exact':True, 'partial_update_rejected':True})
        parity(action, rows, dest/'trained-parity.json', guard, zero=False)
        assert tensor_state_digest(model.reference.state_dict()) == frozen
        engine.restore(start)
        timing = []
        for i in range(cfg['timing_updates']):
            guard.check(); tick = time.perf_counter(); record = engine.step(batches[i])
            torch.cuda.synchronize(); record['seconds'] = time.perf_counter()-tick; timing.append(record)
            if (i+1) % 8 == 0: emit(dest,'timing',step=i+1,total=cfg['timing_updates'])
        mean = float(np.mean([r['seconds'] for r in timing])); p95 = float(np.percentile([r['seconds'] for r in timing],95))
        result = write(dest/'timing.json',{'passed':True,'rows':timing,'mean_seconds':mean,'p95_seconds':p95,
            'projected_seconds':2000*mean+120, 'conservative_seconds':2000*max(mean,p95)*1.2+180,
            'weights_discarded':True})
        results[arm] = result
        engine.restore(start)
        assert tensor_state_digest(model.reference.state_dict()) == frozen
    assert sum(r['conservative_seconds'] for r in results.values()) < cfg['training_total_seconds']
    return {'passed':True, 'latency_digest':lat['digest'], 'arms':{a:r['digest'] for a,r in results.items()},
        'projected_training_seconds':sum(r['projected_seconds'] for r in results.values()),
        'conservative_training_seconds':sum(r['conservative_seconds'] for r in results.values()),
        'frozen_reference_unchanged':True, 'training_ready_requires_fresh_restart_checks':True}


def train(cfg, engine, identity, batches, root, output, guard):
    qualified = read(root/'qualification/completion.json')
    assert qualified['status']=='complete' and qualified['passed'] and qualified['contract_digest']==cfg['digest']
    for arm in ARMS:
        rec = read(root/'restart'/arm/'completion.json')
        assert rec['status']=='complete' and rec['passed'] and rec['contract_digest']==cfg['digest']
    reference = read(root/'qualification'/engine.model.arm/'timing.json')['rows']
    store = CheckpointStore(output/'recovery',engine.identity,keep=3)
    saved = store.load()
    if saved:
        engine.restore(saved)
    else:
        assert not (output/'recovery-created.json').exists(), 'missing committed recovery'
        store.save(engine.snapshot()); write(output/'recovery-created.json',{'identity_digest':engine.identity})
    timer = time.monotonic()
    with (output/'updates.jsonl').open('a') as log:
        while engine.global_step < 2000:
            guard.check(); tick = time.perf_counter(); row = engine.step(batches[engine.global_step])
            torch.cuda.synchronize(); row['seconds'] = time.perf_counter()-tick
            if row['global_step']<=len(reference):
                old = reference[row['global_step']-1]
                assert all(row[k]==old[k] for k in ('motion_l1','gradient_norm','sample_chain','horizon_chain','learning_rate'))
            log.write(json.dumps(row)+'\n'); log.flush()
            if engine.global_step % 16 == 0:
                emit(output,'training',**row,total=2000)
            if engine.global_step % 128 == 0 or time.monotonic()-timer>=300 or engine.global_step==2000:
                store.save(engine.snapshot()); timer = time.monotonic()
    assert engine.samples_seen == cfg['action_presentations']
    endpoint = output/'endpoint'; endpoint.mkdir(exist_ok=True)
    checkpoint = engine.snapshot(); atomic_torch_save(endpoint/'adapter.pt',checkpoint)
    end = write(endpoint/'endpoint.json',{'format':FORMAT,'identity':identity,'contract_digest':cfg['digest'],
        'arm':engine.model.arm,'updates':engine.global_step,'samples_seen':engine.samples_seen,
        'sample_chain':engine.sample_chain,'horizon_chain':engine.horizon_chain,
        'adapter_state_sha256':tensor_state_digest(engine.model.adapter.state_dict()),
        'checkpoint_sha256':sha(endpoint/'adapter.pt'), 'reference_sha256':cfg['h8_endpoint']['policy_sha256'],
        'selection':'final-2000-update-endpoint','gripper':'frozen-current-observation-H8', 'h8_bypass':True})
    return {'passed':True,'endpoint_digest':end['digest'],'updates':2000,'samples_seen':engine.samples_seen}


def run(bundle, root, phase, arm):
    import run_week11_multihorizon_screen as screen
    import run_week11_pilot as pilot
    from week11_multihorizon_runtime import CACHE, DATA, PLAN, ROOT, admission, runtime_identity
    cfg = read(bundle/'contract.json')
    for name,h in cfg['code_sha256'].items(): assert sha(bundle/name)==h, name
    output = (root/'qualification' if phase=='qualify' else root/'restart'/arm if phase=='restart'
              else root/'training'/arm if phase=='train' else root/'endpoint-verification'/arm)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'completion.json').exists():
        previous = read(output/'completion.json')
        assert phase=='train' and previous['status']!='complete', 'preserve completed or failed attempts'
        attempts=output/'attempts'; attempts.mkdir(exist_ok=True)
        (output/'completion.json').rename(attempts/f'completion-{time.time_ns()}.json')
    result = {'status':'incomplete','contract_digest':cfg['digest'],'phase':phase,'arm':arm}
    limits = dict(cfg['resources'], qualification_active_seconds=(cfg['qualification_seconds']
        if phase in ('qualify','restart','endpoint') else cfg['training_total_seconds']/2))
    guard = ResourceGuard(output,cfg['digest'],limits); started=time.monotonic()
    try:
        require_idle_gpu(); configure_deterministic_cuda(); seed_training(17); torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.9); torch.cuda.reset_peak_memory_stats()
        training, source, schedule = admission(); source2=screen.source_check(training,source)
        assert source['digest']==cfg['parent_training_source_digest']
        assert source2['digest']==cfg['parent_screen_source_digest']
        assert schedule['digest']==cfg['schedule_digest']; runtime=runtime_identity()
        audit=read(PLAN/'verified-input-audit.json'); checked=verify_training_payloads(audit)
        _,_,original,recipe=pilot.endpoint_check(read(ROOT/'configs/evaluation/week11_integration_v1.json'))
        actions=screen.Actions(training,Path(cfg['parent_run_root']),original)
        original_action=actions.select('h8-continuation')
        end=read(Path(cfg['parent_run_root'])/'training/h8-continuation/endpoint/endpoint.json')
        assert all(end[k]==v for k,v in cfg['h8_endpoint'].items())
        frozen=tensor_state_digest(actions.model.state_dict())
        model=FrozenMotionPolicy(actions.model,arm); action=action_adapter(original_action,model)
        batches=load_batches(action._normalizer,schedule,audit,root=ROOT,data=DATA,cache=CACHE)
        write(output/'preflight.json',{'runtime':runtime,'training_files_verified':len(checked),
            'frozen_reference_state':frozen,'adapter_parameters':sum(p.numel() for p in model.adapter.parameters()),
            'source_digest':source['digest'],'schedule_digest':schedule['digest']})
        emit(output,'preflight_passed',phase_to_run=phase,arm=arm)
        if phase=='qualify':
            result.update(qualify(cfg,actions,action,batches,replay_inputs(cfg),recipe,output,guard,frozen))
        elif phase=='restart':
            engine,identity=make_engine(model,batches,cfg,arm,'qualification')
            source_dir=root/'qualification'/arm
            expected=read(source_dir/'recovery-reference.json')
            engine.restore(CheckpointStore(source_dir/'restart',engine.identity).load())
            engine.step(batches[engine.global_step])
            assert tree_digest(engine.snapshot())==expected['expected_after_next_step']
            p=parity(action,replay_inputs(cfg),output/'parity.json',guard,zero=False)
            result.update(passed=True,fresh_process_next_update_exact=True,parity_digest=p['digest'])
        else:
            engine,identity=make_engine(model,batches,cfg,arm,'production')
            write(output/'identity.json',identity)
            if phase=='train':
                result.update(train(cfg,engine,identity,batches,root,output,guard))
            else:
                enddir=root/'training'/arm/'endpoint'; end=read(enddir/'endpoint.json')
                assert sha(enddir/'adapter.pt')==end['checkpoint_sha256'] and end['identity']==identity
                engine.restore(torch.load(enddir/'adapter.pt',map_location='cpu',weights_only=True))
                assert engine.global_step==2000 and engine.samples_seen==58172
                assert tensor_state_digest(model.adapter.state_dict())==end['adapter_state_sha256']
                p=parity(action,replay_inputs(cfg),output/'parity.json',guard,zero=False)
                result.update(passed=True,endpoint_digest=end['digest'],fresh_process_endpoint_loaded=True,
                              h8_and_gripper_parity_digest=p['digest'])
        assert tensor_state_digest(model.reference.state_dict())==frozen
        parent_endpoint=Path(cfg['parent_run_root'])/'training/h8-continuation/endpoint'
        assert read(parent_endpoint/'endpoint.json')['digest']==cfg['h8_endpoint']['digest']
        assert sha(parent_endpoint/'policy.pt')==cfg['h8_endpoint']['policy_sha256']
        for name,h in cfg['code_sha256'].items(): assert sha(bundle/name)==h
        assert admission()[1]['digest']==cfg['parent_training_source_digest']
        result.update(status='complete',frozen_reference_unchanged=True)
    except BaseException as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['seconds']=time.monotonic()-started; result['resources']=guard.close()
        write(output/'completion.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=('qualify','restart','train','endpoint'),required=True)
    p.add_argument('--arm',choices=ARMS,default=ARMS[0]); a=p.parse_args()
    run(a.bundle,a.root,a.phase,a.arm)
