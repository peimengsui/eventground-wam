"""Bounded four-arm first-chunk intervention on three selected failures."""
from __future__ import annotations

import argparse
import fcntl
import itertools
import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import run_week11_formal as formal
import run_week11_pilot as pilot
import torch
from evaluate_week9 import runtime, scene

from eventground_wam.auxiliary_evaluation import require
from eventground_wam.b0_training import (
    configure_deterministic_cuda,
    read_sealed,
    seed_training,
    sha256_file,
)
from eventground_wam.clean_comparison import atomic_json, utc_now
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.episode_recorder import read_episode_metadata
from eventground_wam.evaluation_comparisons import audit_panel, compare_inputs, summarize_rows
from eventground_wam.eventground_inference import EventGroundInferencePolicy
from eventground_wam.pixel_predictor import tensor_state_digest
from eventground_wam.week11_closing_gate import ADAPTIVE, run_gated_episode
from eventground_wam.week11_execution import execute_episode, immutable
from eventground_wam.week11_first_chunk import (
    ARMS,
    FIXED,
    GATED,
    ONLY,
    SKIP,
    FirstChunkPolicy,
    audit_first_chunk,
)
from eventground_wam.week11_formal import relocated_runtime
from eventground_wam.week11_resources import ResourceGuard, require_idle_gpu

ROOT = Path(__file__).resolve().parents[1]


def write(path, value):
    sealed = seal_contract(value)
    atomic_json(path, sealed)
    return sealed


def qualify(output, cases, policies):
    """Test actual recorded initial/activation inputs, with synthetic histories only."""
    rows = []
    for case in cases:
        paths = case['artifacts']
        step = 10+8*case['gate_query']
        meta = read_episode_metadata(Path(paths[FIXED]))
        with h5py.File(paths[FIXED]) as f, h5py.File(paths[GATED]) as g:
            initial = {k:f['steps'][k][10] for k in ('agent_rgb','wrist_rgb','proprioception')}
            obs = {k:f['steps'][k][step] for k in ('agent_rgb','wrist_rgb','proprioception')}
            require(all(np.array_equal(obs[k],g['steps'][k][step]) for k in ('agent_rgb','wrist_rgb')),
                    'Historical first-intervention images mismatch')
            initial_recorded = f['steps/predicted_action_chunk'][10]
            recorded_fixed, recorded_gated = f['steps/predicted_action_chunk'][step],g['steps/predicted_action_chunk'][step]
        for x in (initial,obs):x.update(instruction=meta['instruction'],suite=meta['suite'])
        for p in policies.values():
            p.reset()
            require(np.array_equal(p.predict_chunk(**initial),initial_recorded), 'Initial fixed8 path changed')
        fixed = policies[FIXED].fixed.predict_chunk(**obs)
        adaptive = policies[FIXED].adaptive.predict_chunk(**obs)
        require(np.array_equal(fixed,recorded_fixed) and np.array_equal(adaptive,recorded_gated),
                'Recorded first-intervention chunks failed to reproduce')
        prediction = policies[FIXED].adaptive.last_query['prediction']
        for arm,p in policies.items():
            p.history.observe(np.array([0.,0.,0.,0.,0.,0.,-1.]))
            p.history.observe(np.array([0.,0.,0.,0.,0.,0.,1.]))
            first = p.predict_chunk(**obs)
            require(np.array_equal(first,adaptive if arm in (GATED,ONLY) else fixed),
                    'First chunk routing changed')
            require(p.last_query['prediction']==prediction, 'First chunk forecast changed')
            second = p.predict_chunk(**obs)
            require(np.array_equal(second,adaptive if arm in (GATED,SKIP) else fixed),
                    'Later routing changed')
            require(p.last_query['prediction']==prediction, 'Later forecast changed')
        rows.append({'case':case['id'],'initial_all_four_fixed8_exact':True,
                     'recorded_activation_chunks_exact':True,'first_and_later_routing_exact':True,
                     'forecast_exact':True,'used_activation_horizons':[8,4,8,4]})
    return write(output/'qualification.json', {'passed':True,'cases':rows,
        'synthetic_history_only':True,'simulator_steps':0,'full_policy_queries':14*len(cases)})


def run(output, *, preflight_only=False):
    cfg = read_sealed(ROOT/'week11-first-chunk-contract.json')
    output.mkdir(parents=True, exist_ok=True)
    limits = {**cfg['resources'], 'qualification_active_seconds':
              cfg['resources']['qualification_active_seconds']-cfg.get('prior_attempt_seconds', 0.)}
    guard = ResourceGuard(output, cfg['digest'], limits)
    invocation = output/'invocations'/utc_now().replace(':', '-')
    invocation.mkdir(parents=True, exist_ok=False)
    result = {'status': 'incomplete', 'started_at': utc_now(), 'contract_digest': cfg['digest'],
              'training_updates': 0, 'planned_rollouts': len(cfg['cases'])*len(ARMS)}
    started = time.monotonic()
    try:
        require_idle_gpu()
        parent = formal.package_check()
        require(parent['digest'] == cfg['formal_source_digest'], 'Parent source changed')
        for name, digest in cfg['code_sha256'].items():
            require(sha256_file(ROOT/name) == digest, 'Diagnostic code changed: '+name)
        immutable(output/'contract.json', cfg)
        configure_deterministic_cuda()
        seed_training(17)
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.9)
        torch.cuda.reset_peak_memory_stats()
        integration = read_sealed(ROOT/'configs/evaluation/week11_integration_v1.json')
        admission, contract, endpoint, recipe = pilot.endpoint_check(integration)
        require(admission == read_sealed(Path(cfg['raw_root'])/'admission.json'), 'Endpoints changed')
        numerical = read_sealed(formal.EVIDENCE/'remote/numerical-qualification.json')
        runtime_before = runtime()
        parent_cohort = read_sealed(Path(cfg['raw_root'])/'cohort.json')
        require(runtime_before == relocated_runtime(parent_cohort['runtime'], ROOT, pilot.LIBERO),
                'Runtime differs beyond child source path')
        from check_reference_policy import verify_libero
        require(verify_libero(pilot.LIBERO) == parent_cohort['benchmark'], 'Benchmark changed')
        for path, digest in cfg['input_sha256'].items():
            require(sha256_file(Path(path)) == digest, 'Prior input changed: '+path)
        inputs = {p: formal.inventory(integration, p) for p in ('targeted-467', 'clean-200')}
        for case in cfg['cases']:
            task = inputs[case['panel']][0][case['ordinal']]
            require(all(task[k] == v for k, v in case['task'].items()), 'Selected task changed')
        # A worst-step-length projection from the prior recordings, charging fresh controls too.
        projected_seconds, projected_bytes = 180., 0.
        for case in cfg['cases']:
            protocol = inputs[case['panel']][1]
            max_steps = protocol['benchmark']['suites'][case['task']['suite']]['max_policy_steps']+10
            costs = []
            for old in case['artifacts'].values():
                old = Path(old)
                with h5py.File(old) as handle:
                    steps = len(handle['steps/executed_action'])
                elapsed = read_sealed(old.with_suffix('.timing.json'))['episode_wall_time_seconds']
                costs.append((elapsed/max(1, steps), old.stat().st_size/max(1, steps)))
            projected_seconds += len(ARMS)*max_steps*max(c[0] for c in costs)*1.2
            projected_bytes += len(ARMS)*max_steps*max(c[1] for c in costs)*1.2
        projected_bytes += 128*2**20
        free = shutil.disk_usage(output).free
        require(projected_seconds+guard.budget.seconds < cfg['resources']['qualification_active_seconds'],
                'Projected active time exceeds diagnostic cap')
        require(projected_bytes < cfg['resources']['output_limit_bytes']
                and free > projected_bytes+cfg['resources']['minimum_free_bytes'],
                'Projected disk budget/reserve unavailable')
        write(invocation/'preflight.json', {'passed': True, 'source_digest': parent['digest'],
            'runtime': runtime_before, 'admission': admission, 'input_files_hashed': len(cfg['input_sha256']),
            'projected_seconds_1p2': projected_seconds, 'projected_bytes_1p2': projected_bytes,
            'free_bytes': free, 'prior_packages_modified': False})
        immutable(output/'admission.json', admission)
        if preflight_only:
            result['status'] = 'preflight_passed'
            return
        from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur
        install_codec_independent_motion_blur()
        action, event = pilot.load_action(contract, endpoint), pilot.load_event(recipe)
        action_hash = tensor_state_digest(action._policy.state_dict())
        event_hash = tensor_state_digest(event.model.state_dict())
        require(action_hash == numerical['action_state_sha256'] and event_hash == numerical['event_state_sha256'],
                'Model tensors differ from qualification')
        fixed = EventGroundInferencePolicy(action, FIXED, event, identities=admission, guard=guard.check)
        adaptive = EventGroundInferencePolicy(action, ADAPTIVE, event, identities=admission, guard=guard.check)
        policies = {arm: FirstChunkPolicy(fixed, adaptive, arm) for arm in ARMS}
        # Qualification has no episodes, and is performed once per immutable output.
        if not (output/'qualification.json').exists():
            qualify(output, cfg['cases'], policies)
        else:
            require(read_sealed(output/'qualification.json')['passed'], 'Prior qualification failed')
        print(json.dumps({'phase': 'qualified', 'rollouts': cfg['planned_rollouts'],
                          'projected_seconds_1p2': projected_seconds}), flush=True)
        cohort = seal_contract({'artifact_type': 'eventground_wam.week11_first_chunk',
            'contract_digest': cfg['digest'], 'parent_digest': integration['digest'],
            'runtime': runtime_before, 'admission_digest': admission['digest'],
            'fresh_rollouts': True, 'posthoc_diagnostic_panel': True, 'training_updates': 0})
        immutable(output/'cohort.json', cohort)
        identities = {}
        for arm in ARMS:
            for panel in inputs:
                base_identity = formal.identity_for(cohort, admission, arm, panel,
                    inputs[panel][1], {**parent, 'digest': cfg['digest']})
                base_identity.pop('digest')
                identities[arm, panel] = seal_contract({**base_identity,
                    'artifact_type': 'eventground_wam.week11_first_chunk_panel',
                    'mode': 'posthoc_diagnostic'})
                immutable(output/'panels'/arm/panel/'identity.json', identities[arm, panel])
        from evaluate_clean_comparison import seed_episode
        rows, gate_audits = {}, {}
        for schedule_index, item in enumerate(cfg['schedule']):
            case = cfg['cases'][item['case_index']]
            arm, panel, ordinal = item['arm'], case['panel'], case['ordinal']
            tasks, protocol, suites, states = inputs[panel]
            task, identity = tasks[ordinal], identities[arm, panel]
            guard.check()
            metadata, initial, factory = scene('clean' if panel == 'clean-200' else 'perturbed',
                task, protocol, suites, states, identity)
            row = execute_episode(root=output/'panels'/arm/panel, ordinal=ordinal, task=task,
                identity=identity, protocol=protocol, metadata=metadata, initial_state=initial,
                policy=policies[arm], environment_factory=factory, seed_episode=seed_episode,
                episode_runner=run_gated_episode)
            gate = audit_first_chunk(row, arm, admission)
            rows[case['id'], arm], gate_audits[case['id'], arm] = row, gate
            progress = {'completed': schedule_index+1, 'total': len(cfg['schedule']),
                        'case': case['id'], 'arm': arm, 'updated_at': utc_now(),
                        'elapsed_seconds': time.monotonic()-started}
            atomic_json(output/'progress.json', progress)
            print(json.dumps(progress), flush=True)
        reports = {}
        for arm in ARMS:
            for panel in inputs:
                cases = [c for c in cfg['cases'] if c['panel'] == panel]
                selected_rows = [rows[c['id'], arm] for c in cases]
                tasks = [inputs[panel][0][c['ordinal']] for c in cases]
                audits = audit_panel(selected_rows, identities[arm, panel], tasks, inputs[panel][1])
                reports[arm, panel] = write(output/'panels'/arm/panel/'report.json', {
                    'identity_digest': identities[arm, panel]['digest'], 'episodes': selected_rows,
                    'case_ids': [c['id'] for c in cases], 'audits': audits,
                    'summary': summarize_rows(selected_rows),
                    'gate_audits': [gate_audits[c['id'], arm] for c in cases]})
        comparisons = {panel: {a+' vs '+b: compare_inputs(reports[a, panel]['audits'],
            reports[b, panel]['audits']) for a, b in itertools.combinations(ARMS, 2)} for panel in inputs}
        require(tensor_state_digest(action._policy.state_dict()) == action_hash
                and tensor_state_digest(event.model.state_dict()) == event_hash, 'Model tensors mutated')
        require(pilot.endpoint_check(integration)[0] == admission, 'Endpoint changed during run')
        require(runtime() == runtime_before, 'Runtime changed during run')
        valid = all(a['error'] is None for r in reports.values() for a in r['audits'])
        valid &= all(g['valid'] and not g.get('query_exceptions') for g in gate_audits.values())
        result.update(status='complete' if valid else 'audit_incomplete', completed_rollouts=len(rows),
            all_artifacts_and_gates_valid=valid, comparisons=comparisons,
            reports={a+'/'+p: r['digest'] for (a, p), r in reports.items()},
            model_state_unchanged=True, action_state_sha256=action_hash, event_state_sha256=event_hash,
            all_initial_inputs_exact=all(c['all_exact'] for ps in comparisons.values() for c in ps.values()))
    except BaseException as exc:
        result.update(status='stopped', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result.update(finished_at=utc_now(), invocation_seconds=time.monotonic()-started,
                      resources=guard.close())
        write(invocation/'result.json', result)
        write(output/'completion.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.output, preflight_only=args.preflight_only)
