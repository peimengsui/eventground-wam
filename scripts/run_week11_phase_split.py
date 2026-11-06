"""Twelve fresh, same-seed clean155 rollouts with predeclared routing windows."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from week11_phase_split import ARMS, WindowPolicy, audit_window
from week11_replay_common import read, sha
from week11_replay_common import write as write_json


def write(path, value):
    write_json(path, value)
    return read(path)


def qualify(output, cfg, fixed, adaptive):
    rows = []; calls = 0
    for arm, path in cfg['historical'].items():
        queries = [json.loads(line) for line in Path(path).with_suffix('.queries.jsonl').read_text().splitlines()]
        with h5py.File(path) as f:
            meta = json.loads(f['episode/metadata_json'][()])
            for frame in (10, 138, 218):
                obs = {k: f['steps'][k][frame] for k in ('agent_rgb', 'wrist_rgb', 'proprioception')}
                obs.update(instruction=meta['instruction'], suite=meta['suite'])
                policy = fixed if arm == 'fixed8' else adaptive
                got = policy.predict_chunk(**obs); calls += 1
                assert np.array_equal(got, f['steps/predicted_action_chunk'][frame])
                old = queries[(frame-10)//8]['policy']
                assert all(policy.last_query[k] == old[k] for k in ('prediction', 'decision', 'used_horizon'))
                assert policy.last_query['event']['categorical_probabilities'] == old['event']['categorical_probabilities']
                rows.append({'route': arm, 'frame': frame, 'action_and_event_exact': True})
    # Same real observation at every boundary; only the executed-frame count is synthetic.
    for frame in (10, 130, 138, 146, 218, 298):
        outputs = [p.predict_chunk(**obs).copy() for p in (fixed, adaptive)]; calls += 2
        forecast = fixed.last_query['prediction']; assert adaptive.last_query['prediction'] == forecast
        for arm in ARMS:
            p = WindowPolicy(fixed, adaptive, arm); p.history.executed_steps = frame
            got = p.predict_chunk(**obs); calls += 1
            active = (arm == 'adaptive' or arm == 'early-only' and 10 <= frame < 138
                      or arm == 'late-sustained' and frame >= 138)
            assert np.array_equal(got, outputs[int(active)]) and p.last_query['prediction'] == forecast
            assert p.last_query['window_active'] == active
            p.reset(); assert p.history.executed_steps == 0 and p.last_query is None
    assert calls == 42
    return write(output/'qualification.json', {'passed': True, 'historical_parity': rows,
        'boundary_frames': [10,130,138,146,218,298], 'all_four_routes_exact': True,
        'reset_checked': True, 'full_policy_queries': calls, 'simulator_steps': 0})


def run(bundle, output):
    import run_week11_formal as formal
    import run_week11_multihorizon_screen as screen
    import run_week11_pilot as pilot
    from evaluate_clean_comparison import seed_episode
    from evaluate_week9 import runtime, scene
    from week11_multihorizon_runtime import ROOT, admission, runtime_identity

    from eventground_wam.b0_training import configure_deterministic_cuda, seed_training
    from eventground_wam.evaluation_comparisons import audit_panel
    from eventground_wam.eventground_inference import EventGroundInferencePolicy
    from eventground_wam.pixel_predictor import tensor_state_digest
    from eventground_wam.week11_closing_gate import run_gated_episode
    from eventground_wam.week11_execution import execute_episode, immutable
    from eventground_wam.week11_formal import relocated_runtime
    from eventground_wam.week11_resources import ResourceGuard, require_idle_gpu

    cfg = read(bundle/'contract.json')
    for p,h in cfg['code_sha256'].items(): assert sha(bundle/p) == h, p
    output.mkdir(parents=True, exist_ok=False); write(output/'contract.json', cfg)
    guard = ResourceGuard(output, cfg['digest'], cfg['resources'])
    started = time.monotonic(); result = {'status':'incomplete', 'contract_digest':cfg['digest'],
        'training_updates':0, 'planned_rollouts':12}
    try:
        require_idle_gpu(); os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        configure_deterministic_cuda(); seed_training(17); torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.9); torch.cuda.reset_peak_memory_stats()
        training, source, _ = admission(); ss = screen.source_check(training, source)
        assert training['digest'] == cfg['training_design_digest'] and source['digest'] == cfg['training_source_digest']
        assert ss['digest'] == cfg['screen_source_digest']; rt_identity = runtime_identity()
        for path,h in cfg['input_sha256'].items(): assert sha(path) == h, path
        run_root = Path(cfg['run_root']); end = read(run_root/'training/mixed-continuation/endpoint/endpoint.json')
        assert end['digest'] == cfg['endpoint']['digest'] and end['policy_sha256'] == cfg['endpoint']['policy_sha256']
        integration = read(ROOT/'configs/evaluation/week11_integration_v1.json')
        original_admission, _, original_endpoint, recipe = pilot.endpoint_check(integration)
        parent_source = formal.package_check(); rt = runtime()
        prior = read(pilot.DATA/'results/week11-formal-v1/cohort.json')
        assert rt == relocated_runtime(prior['runtime'], ROOT, pilot.LIBERO)
        from check_reference_policy import verify_libero
        assert verify_libero(pilot.LIBERO) == prior['benchmark']
        tasks, protocol, suites, states = formal.inventory(integration, 'clean-200')
        task = tasks[155]; assert all(task[k] == v for k,v in cfg['task'].items())
        assert protocol['simulation']['settling_steps'] == 10
        max_frames = 10+protocol['benchmark']['suites'][task['suite']]['max_policy_steps']
        rates = []
        for path in cfg['historical'].values():
            with h5py.File(path) as f: n = len(f['steps/executed_action'])
            rates.append((read(Path(path).with_suffix('.timing.json'))['episode_wall_time_seconds']/n,
                          Path(path).stat().st_size/n))
        seconds = 180+12*max_frames*max(x[0] for x in rates)*1.2
        storage = 128*2**20+12*max_frames*max(x[1] for x in rates)*1.2
        assert seconds+guard.budget.seconds < cfg['resources']['qualification_active_seconds']
        assert storage < cfg['resources']['output_limit_bytes']
        assert shutil.disk_usage(output).free > storage+cfg['resources']['minimum_free_bytes']
        write(output/'preflight.json', {'passed':True, 'runtime':rt, 'runtime_identity':rt_identity,
            'projected_seconds_1p2':seconds, 'projected_bytes_1p2':storage, 'max_frames':max_frames})
        from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur
        install_codec_independent_motion_blur()
        actions = screen.Actions(training, run_root, original_endpoint); action = actions.select('mixed-continuation')
        event = pilot.load_event(recipe); action_hash = tensor_state_digest(actions.model.state_dict())
        event_hash = tensor_state_digest(event.model.state_dict())
        assert event_hash == read(formal.EVIDENCE/'remote/numerical-qualification.json')['event_state_sha256']
        # Use the previously admitted branch identity unchanged.
        historical_identity = read(Path(cfg['historical']['fixed8']).parent/'identity.json')
        adm = json.loads(Path(cfg['historical']['fixed8']).with_suffix('.queries.jsonl').read_text().splitlines()[0])['policy']['branch_identities']
        assert adm['continuation_endpoint_digest'] == end['digest'] and adm['m4_sha256'] == end['policy_sha256']
        assert adm['parent_original_admission_digest'] == original_admission['digest']
        assert historical_identity['admission_digest'] == adm['digest']; immutable(output/'admission.json', adm)
        fixed = EventGroundInferencePolicy(action, screen.ROUTES['fixed8'], event, identities=adm, guard=guard.check)
        adaptive = EventGroundInferencePolicy(action, screen.ROUTES['adaptive'], event, identities=adm, guard=guard.check)
        qualify(output, cfg, fixed, adaptive)
        print(json.dumps({'phase':'qualified', 'planned_rollouts':12, 'projected_seconds':seconds}), flush=True)
        cohort = write(output/'cohort.json', {'artifact_type':'eventground_wam.week11_phase_split',
            'parent_digest':training['digest'], 'contract_digest':cfg['digest'], 'runtime':rt,
            'posthoc_diagnostic':True, 'same_seed_repetitions':True, 'training_updates':0})
        rows = []
        for item in cfg['schedule']:
            guard.check(); arm, rep = item['arm'], item['repeat']
            root = output/'panels'/f'repeat-{rep}'/arm
            identity = formal.identity_for(cohort, adm, arm, 'clean-200', protocol, parent_source)
            identity.pop('digest'); identity.update(mode='selected_causal_diagnostic', repeat=rep,
                intervention_contract_digest=cfg['digest'], training_endpoint=end['digest'])
            identity = write(root/'identity.json', identity)
            metadata, initial, factory = scene('clean', task, protocol, suites, states, identity)
            row = execute_episode(root=root, ordinal=155, task=task, identity=identity, protocol=protocol,
                metadata=metadata, initial_state=initial, policy=WindowPolicy(fixed, adaptive, arm),
                environment_factory=factory, seed_episode=seed_episode, episode_runner=run_gated_episode)
            audit = audit_window(row, arm, adm)
            artifacts = audit_panel([row], identity, [task], protocol); assert artifacts[0]['error'] is None
            report = write(root/'report.json', {'arm':arm, 'repeat':rep, 'result':row,
                'routing_audit':audit, 'artifact_audit':artifacts[0]})
            rows.append({'arm':arm, 'repeat':rep, 'report':str(root/'report.json'), 'report_digest':report['digest'],
                         'success':row['success'], 'artifact':row['artifact']})
            progress = {'completed':len(rows), 'total':12, 'arm':arm, 'repeat':rep,
                        'success':row['success'], 'seconds':time.monotonic()-started}
            write(output/'progress.json', progress); print(json.dumps(progress), flush=True)
        assert tensor_state_digest(actions.model.state_dict()) == action_hash
        assert tensor_state_digest(event.model.state_dict()) == event_hash
        assert pilot.endpoint_check(integration)[0] == original_admission
        assert sha(actions.paths['mixed-continuation']) == cfg['endpoint']['policy_sha256']
        assert admission()[1]['digest'] == cfg['training_source_digest']
        assert screen.source_check(training, source)['digest'] == cfg['screen_source_digest'] and runtime() == rt
        for path,h in cfg['input_sha256'].items(): assert sha(path) == h, path
        result.update(status='complete', completed_rollouts=len(rows), rows=rows,
            action_state_sha256=action_hash, event_state_sha256=event_hash,
            model_states_unchanged=True, frozen_inputs_and_source_unchanged=True)
    except BaseException as exc:
        result.update(status='stopped', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        result.update(seconds=time.monotonic()-started, resources=guard.close())
        write(output/'completion.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a = p.parse_args()
    with (a.bundle/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); run(a.bundle,a.output)
