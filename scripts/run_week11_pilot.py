"""Frozen Week 11 integration qualification and exactly 24 smoke episodes.

No formal-panel execution entry point is provided. Each invocation and attempt
is retained; resume verifies receipts and cannot buy additional episode retries.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
from verify_week11_horizon import verify as verify_mapping

from eventground_wam.auxiliary_evaluation import require
from eventground_wam.b0_training import (
    configure_deterministic_cuda,
    read_sealed,
    seed_training,
    sha256_file,
)
from eventground_wam.clean_comparison import atomic_json, metadata_for_pair, utc_now
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.episode_recorder import sha256_array
from eventground_wam.evaluation_comparisons import audit_panel, compare_inputs, summarize_rows
from eventground_wam.evaluation_protocol import load_evaluation_protocol, protocol_sha256
from eventground_wam.eventground_data import CurrentObservation, collate_current_observations
from eventground_wam.eventground_inference import (
    ARMS,
    EventBranch,
    EventGroundInferencePolicy,
    routed_horizon,
)
from eventground_wam.eventground_model import serialize_tokens
from eventground_wam.pixel_predictor import tensor_state_digest
from eventground_wam.week9_execution import bind_metadata, immutable
from eventground_wam.week11_resources import ResourceGuard, require_idle_gpu

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('/home/peimengsui/data/eventground-wam/libero-plus')
BASE = DATA/'results/week10-preservation-pilot-v1'
CACHE = Path('/home/peimengsui/data/eventground-wam/huggingface-cache')
LIBERO = Path('/home/peimengsui/code/vendor/LIBERO-plus')


def write(path, value):
    value = seal_contract(value)
    atomic_json(path, value)
    return value


def package_check():
    source = read_sealed(ROOT/'week11-source-manifest.json')
    for name, expected in source['files'].items():
        require(sha256_file(ROOT/name) == expected, 'Execution source changed: '+name)
    verify_mapping(ROOT)
    child = read_sealed(ROOT/'docs/evaluation/week11-horizon-v1/package-index.json')
    require(child['digest'] == source['mapping_package_digest'], 'Mapping package changed')
    for name, expected in child['files_sha256'].items():
        require(sha256_file(ROOT/name) == expected, 'Mapping evidence changed: '+name)
    return source


def endpoint_check(cfg):
    from eventground_wam.coupled_evaluation import load_contract, verify_endpoint
    from eventground_wam.eventground_preservation_runtime import validate_training_completion
    from eventground_wam.eventground_runtime import verify_eventground_parent
    training = Path(cfg['endpoints']['m4']['checkpoint_path']).parents[2]
    contract = load_contract(ROOT, 'M4')
    identity, endpoint, completion = verify_endpoint(training, contract)
    parent = read_sealed(ROOT/'configs/training/week10_eventground_experiment_v3.json')
    verify_eventground_parent(parent)
    event = validate_training_completion(BASE, 'control')
    pin = cfg['endpoints']['event']
    require(event['digest'] == pin['completion_digest']
            and event['identity_digest'] == pin['identity_digest']
            and event['inference_header']['digest'] == pin['inference_header_digest']
            and sha256_file(Path(pin['remote_endpoint'])) == pin['endpoint_sha256'],
            'Selected event completion or checkpoint differs')
    recipe = read_sealed(BASE/'source/configs/training/week10_eventground_numerical_v2.json')
    require(recipe['digest'] == read_sealed(BASE/'protocol.json')['recipe_digest'], 'Event recipe differs')
    # The loader validates frozen source, and these checks bind executed modules
    # to that source instead of merely trusting an unused copy on disk.
    for name, expected in recipe['code_sha256'].items():
        require(sha256_file(ROOT/name) == expected, 'Executed event recipe code differs: '+name)
    return seal_contract({'m4_identity': identity['digest'], 'm4_endpoint': endpoint['digest'],
        'm4_completion': completion['digest'], 'm4_sha256': cfg['endpoints']['m4']['policy_sha256'],
        'predictor_sha256': cfg['endpoints']['m4']['predictor_sha256'],
        'event_completion': event['digest'], 'event_sha256': pin['endpoint_sha256'],
        'recipe_digest': recipe['digest']}), contract, endpoint, recipe


def load_action(contract, endpoint):
    from types import SimpleNamespace

    from evaluate_coupled_dev import load_policy
    policy = load_policy(SimpleNamespace(training_root=DATA/'training/week8-m4-lambda-v1',
        cache_dir=CACHE), {**contract, 'checkpoint_id': 'eventground-wam/week11-m4-fixed-epoch-2'}, endpoint)
    policy._policy.requires_grad_(False).eval()
    return policy


def load_event(recipe):
    from eventground_wam.eventground_future_runtime import load_endpoint
    from eventground_wam.eventground_preservation_runtime import validate_training_completion
    from eventground_wam.eventground_runtime import build_eventground_from_m4
    validate_training_completion(BASE, 'control')
    model, builder, allowed, _ = build_eventground_from_m4(BASE/'source', recipe, CACHE, 'cuda:0')
    load_endpoint(model, allowed, BASE, 'control')
    return EventBranch(model, builder, 'cuda:0')


def inventory(cfg):
    from check_reference_policy import verify_libero

    from eventground_wam.clean_evaluation import clean_manifest
    benchmark = verify_libero(LIBERO)
    canonical, suites = clean_manifest(LIBERO, load_evaluation_protocol(ROOT/'configs/evaluation/libero_clean_v1.json'))
    require(canonical == read_sealed(ROOT/'evaluation/splits/libero-clean-v1/clean-40.json'), 'Clean inventory changed')
    indexed = {(r['suite'], r['runtime_task_index']): r for r in canonical['tasks']}
    tasks = [{**p, 'perturbation_category': 'Clean', 'difficulty_level': None}
             for p in read_sealed(ROOT/cfg['panels']['qualification-8']['path'])['pairs']]
    states = {}
    for task in tasks:
        key = task['suite'], task['runtime_task_index']
        for name in ('task_name', 'instruction', 'bddl_sha256', 'initial_states_file_sha256'):
            require(task[name] == indexed[key][name], 'Qualification task changed: '+name)
        if key not in states:
            states[key] = np.asarray(suites[key].get_task_init_states(key[1]))
        require(len(states[key]) == 50 and sha256_array(states[key][task['initial_state_index']])
                == task['initial_state_sha256'], 'Qualification state changed')
    protocol = load_evaluation_protocol(ROOT/'configs/evaluation/week4_clean_comparison_v1.json')
    return tasks, protocol, suites, states, benchmark


def scene(task, protocol, suites, states, identity):
    from eventground_wam.libero_bddl import resolve_libero_bddl_paths
    key = task['suite'], task['runtime_task_index']
    paths = resolve_libero_bddl_paths(suites[key].get_task_bddl_file_path(key[1]))
    require(sha256_file(paths.source) == task['bddl_sha256'], 'BDDL changed')
    metadata = bind_metadata(metadata_for_pair(task, protocol, identity, paths.source), identity)
    def factory():
        from libero.libero.envs import OffScreenRenderEnv
        size = protocol['observation']['capture_resolution']
        return OffScreenRenderEnv(bddl_file_name=str(paths.runtime), camera_heights=size[0],
            camera_widths=size[1], control_freq=protocol['simulation']['control_frequency_hz'])
    return metadata, states[key][task['initial_state_index']], factory


def prefix_observations(output, tasks, protocol, suites, states, identity, guard):
    """Exactly one optional renderer prefix repeat per pair; no policy execution."""
    from evaluate_clean_comparison import seed_episode
    from evaluate_reference_policy import proprioception_from_observation, transform_rgb
    audits, observations = [], []
    for ordinal, task in enumerate(tasks):
        guard.check()
        path = output/f'prefix-{ordinal:04d}.hdf5'
        receipt = path.with_suffix('.json')
        if receipt.exists():
            saved = read_sealed(receipt)
            require(saved['task'] == task and saved['cohort_digest'] == identity['cohort_digest']
                    and sha256_file(path) == saved['sha256'], 'Renderer prefix changed')
        else:
            # A claimed but interrupted prefix is not silently repeated again.
            claim = path.with_suffix('.claim.json')
            require(not claim.exists() and not path.exists(), 'Interrupted prefix; fixed repeat allowance consumed')
            immutable(claim, seal_contract({'task': task, 'cohort_digest': identity['cohort_digest']}))
            metadata, initial, factory = scene(task, protocol, suites, states, identity)
            seed_episode(task['episode_seed'])
            env = factory()
            values = {k: [] for k in ('agent_rgb', 'wrist_rgb', 'proprioception', 'mujoco_state', 'executed_action')}
            try:
                env.seed(protocol['seeding']['environment_seed'])
                env.reset()
                obs = env.set_init_state(initial)
                for i in range(11):
                    for camera, key in (('agent', 'agent_rgb'), ('wrist', 'wrist_rgb')):
                        values[key].append(transform_rgb(obs, protocol['observation']['cameras'][camera],
                                                        protocol['observation']['capture_resolution']))
                    values['proprioception'].append(proprioception_from_observation(obs))
                    values['mujoco_state'].append(np.asarray(env.get_sim_state(), dtype=np.float64))
                    if i < 10:
                        action = np.asarray(protocol['simulation']['settling_action'], dtype=np.float64)
                        values['executed_action'].append(action)
                        obs, _, done, _ = env.step(action)
                        require(not done, 'Qualification prefix terminated before first query')
            finally:
                env.close()
            with h5py.File(str(path)+'.partial', 'x') as handle:
                for key, rows in values.items():
                    handle.create_dataset(key, data=np.stack(rows), compression='gzip')
                handle.attrs['complete'] = True
            Path(str(path)+'.partial').rename(path)
            inputs = {'first_query_step': 10,
                'prefix': {f'{i}:{key}': sha256_array(value) for key, rows in values.items() for i, value in enumerate(rows)},
                'scene': {key: getattr(metadata, key) for key in ('bddl_sha256', 'initial_state_sha256', 'instruction', 'episode_seed')}}
            saved = write(receipt, {'task': task, 'cohort_digest': identity['cohort_digest'], 'sha256': sha256_file(path), 'inputs': inputs})
        with h5py.File(path) as handle:
            observations.append({'agent_rgb': handle['agent_rgb'][10], 'wrist_rgb': handle['wrist_rgb'][10],
                'proprioception': handle['proprioception'][10], 'instruction': task['instruction'], 'suite': task['suite']})
        audits.append({'ordinal': ordinal, 'inputs': saved['inputs'], 'error': None})
    return observations, audits


def qualify(output, action, event, observations, identities, guard):
    """Real-device exact parity, endpoint reload and controlled dependency checks."""
    before = tensor_state_digest(action._policy.state_dict())
    event_before = tensor_state_digest(event.model.state_dict())
    fixed = EventGroundInferencePolicy(action, 'M4-EG-fixed8', event, identities=identities, guard=guard.check)
    fixed.capture = True
    results = []
    for ordinal, obs in enumerate(observations):
        guard.check()
        captured = {}
        def save(_m, _args, value, captured=captured):
            captured['normalized'] = value.normalized_actions.detach().float().cpu().clone()
        handle = action._policy.register_forward_hook(save)
        try:
            original = action.predict_chunk(**obs)
        finally:
            handle.remove()
        current = CurrentObservation(obs['agent_rgb'], obs['wrist_rgb'], np.asarray(obs['proprioception'], np.float32), obs['instruction'])
        encoded, state = collate_current_observations([current], event.builder, event.device)
        with torch.no_grad():
            reference = serialize_tokens(event.model.head.sample(event.model(encoded, state), group_size=1, greedy=True).tokens)[0]
        actual = fixed.predict_chunk(**obs)
        require(np.array_equal(actual, original), 'Exact decoded M4 fixed8 parity failed')
        require(torch.equal(captured['normalized'], fixed.capture_record['normalized_actions']), 'Exact normalized M4 fixed8 parity failed')
        require(fixed.last_query['prediction'] == reference, 'Pinned FP32 event prediction parity failed')
        repeat, _ = event.predict(current)
        require(repeat == reference, 'Greedy event repeat differs')
        results.append({'ordinal': ordinal, 'exact_normalized': True, 'exact_decoded': True,
                        'event_reference_and_repeat_exact': True, 'prediction': reference})
    # Force all five permitted horizons on the same current observation. This
    # engineering perturbation is not used to choose the deployed mapping.
    outputs = {}
    for horizon in (2, 4, 8, 16, 32):
        record = {}
        with routed_horizon(action._policy, horizon, record, device=action._device, capture=True):
            outputs[horizon] = action.predict_chunk(**observations[0])
        require(record['used_horizon'] == horizon, 'Horizon did not reach core')
    differences = {str(h): float(np.max(np.abs(outputs[h]-outputs[8]))) for h in (2, 4, 16, 32)}
    require(all(d > 0 for d in differences.values()), 'Action output does not depend on horizon')
    token_hook = action._policy.future_core.register_forward_hook(lambda _m, _a, tokens: tokens.flip(-1))
    try:
        altered = action.predict_chunk(**observations[0])
    finally:
        token_hook.remove()
    token_delta = float(np.max(np.abs(altered-outputs[8])))
    require(token_delta > 0, 'Action output does not depend on future tokens')
    require(tensor_state_digest(action._policy.state_dict()) == before, 'Event/coupling execution mutated M4 tensors')
    require(tensor_state_digest(event.model.state_dict()) == event_before, 'Event inference mutated tensors')
    # Reload both actual endpoints in place, then repeat exact output checks.
    from eventground_wam.coupled_checkpoint import load_coupled_policy
    from eventground_wam.eventground_future_runtime import load_endpoint
    from eventground_wam.eventground_model import configure_eventground_trainables
    from eventground_wam.eventground_preservation_runtime import validate_training_completion
    # The exported policy itself supplies the exact admitted trainable key set;
    # language/vision LoRA names are read, not inferred from a broad substring.
    artifact = torch.load(DATA/'training/week8-m4-lambda-v1/epochs/epoch-02/M4.ckpt', map_location='cpu', weights_only=True)
    inherited = tuple(n for n in artifact['states']['policy'] if n.startswith('conditioning_adapter.backbone.'))
    for name, parameter in action._policy.policy.named_parameters():
        parameter.requires_grad_(name in artifact['states']['policy'])
    del artifact
    load_coupled_policy(DATA/'training/week8-m4-lambda-v1/epochs/epoch-02/M4.ckpt', action._policy, inherited)
    allowed = configure_eventground_trainables(event.model, inherited)
    validate_training_completion(BASE, 'control')
    load_endpoint(event.model, allowed, BASE, 'control')
    action._policy.requires_grad_(False).eval()
    event.model.requires_grad_(False).eval()
    require(tensor_state_digest(action._policy.state_dict()) == before, 'M4 reload differs')
    require(tensor_state_digest(event.model.state_dict()) == event_before, 'Event reload differs')
    require(np.array_equal(fixed.predict_chunk(**observations[0]), outputs[8]), 'Reload decoded parity failed')
    require(fixed.last_query['prediction'] == results[0]['prediction'], 'Reload event differs')
    fixed.capture_record = None
    return write(output/'numerical-qualification.json', {'passed': True, 'pairs': results,
        'exact_same_device': True, 'horizon_action_max_abs_delta': differences, 'future_token_action_max_abs_delta': token_delta,
        'original_m4_unchanged': True, 'event_unchanged': True, 'reload_exact': True,
        'action_state_sha256': before, 'event_state_sha256': event_before, 'future_inputs_used': False,
        'separate_model_storage': True, 'branch_identities': identities})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    source = package_check()
    cfg = read_sealed(ROOT/'configs/evaluation/week11_integration_v1.json')
    output.mkdir(parents=True, exist_ok=True)
    configure_deterministic_cuda()
    torch.set_num_threads(4)
    seed_training(17)
    torch.cuda.set_per_process_memory_fraction(.9)
    torch.cuda.reset_peak_memory_stats()
    guard = ResourceGuard(output, cfg['digest'], cfg['resources'])
    invocation = output/'invocations'/utc_now().replace(':', '-')
    invocation.mkdir(parents=True, exist_ok=False)
    result = {'status': 'incomplete', 'source_digest': source['digest'], 'training_updates': 0,
              'formal_episodes': 0, 'user_request': 'Can you do 3 Integrate M4-EG and 4 Verify correctness and run a bounded pilot'}
    try:
        gpu_processes = require_idle_gpu()
        write(invocation/'gpu-process-admission.json', {'processes': gpu_processes})
        admission, contract, endpoint, recipe = endpoint_check(cfg)
        immutable(output/'admission.json', admission)
        from evaluate_week9 import runtime
        baseline_runtime = runtime()
        tasks, protocol, suites, states, benchmark = inventory(cfg)
        cohort = seal_contract({'artifact_type': 'eventground_wam.week11_qualification_cohort',
            'source_digest': source['digest'], 'parent_digest': cfg['digest'], 'mapping_digest': source['mapping_digest'],
            'runtime': baseline_runtime, 'benchmark': benchmark, 'admission_digest': admission['digest'],
            'arm_order': list(ARMS), 'episodes_per_arm': 8, 'optional_prefix_repeats': 8, 'resources': cfg['resources']})
        immutable(output/'cohort.json', cohort)
        immutable(output/'tasks.json', seal_contract({'tasks': tasks}))
        identities = {}
        for arm in ARMS:
            identities[arm] = seal_contract({'artifact_type': 'eventground_wam.week11_pilot_arm',
                'arm': arm, 'mode': 'qualification', 'cohort_digest': cohort['digest'],
                'week9_protocol_digest': cfg['digest'], 'policy_objective': 'all-action-l1-binary-gripper-v2',
                'admission_digest': admission['digest'], 'project_git_revision': source['base_git_revision'],
                'checkpoint_id': 'eventground-wam/week11-'+arm.lower(), 'checkpoint_revision': admission['m4_sha256'],
                'protocol_sha256': protocol_sha256(protocol), 'acceptance_rule_digest': protocol['comparison']['acceptance_rule_digest']})
            immutable(output/arm/'identity.json', identities[arm])
        observations, prefix_audits = prefix_observations(output/'renderer-prefixes', tasks, protocol,
            suites, states, identities[ARMS[0]], guard)
        load_started = time.monotonic()
        action = load_action(contract, endpoint)
        event = load_event(recipe)
        model_load_seconds = time.monotonic()-load_started
        require(runtime() == baseline_runtime, 'Model load changed runtime')
        numerical_path = output/'numerical-qualification.json'
        if numerical_path.exists():
            numerical = read_sealed(numerical_path)
            require(numerical['passed'] and numerical['branch_identities'] == admission, 'Numerical qualification binding differs')
            require(tensor_state_digest(action._policy.state_dict()) == numerical['action_state_sha256']
                    and tensor_state_digest(event.model.state_dict()) == numerical['event_state_sha256'], 'Reload differs from qualification')
        else:
            numerical = qualify(output, action, event, observations, admission, guard)
        print(json.dumps({'phase': 'numerical_qualification', 'passed': True, 'resources': guard.snapshot()}), flush=True)
        # Baseline carries no resident event model. Both EG arms use separate
        # branch instances, then the event instance is released again.
        del event
        gc.collect()
        torch.cuda.empty_cache()
        reports = {}
        from evaluate_clean_comparison import seed_episode

        from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur
        from eventground_wam.week11_execution import execute_episode, validate_query_artifact
        install_codec_independent_motion_blur()
        for arm in ARMS:
            guard.check()
            require(runtime() == baseline_runtime, 'Cross-arm runtime changed')
            tick = time.monotonic()
            event = None if arm == 'M4-fixed' else load_event(recipe)
            load_seconds = time.monotonic()-tick
            policy = EventGroundInferencePolicy(action, arm, event, identities=admission, guard=guard.check)
            rows = []
            for ordinal, task in enumerate(tasks):
                guard.check()
                metadata, initial, factory = scene(task, protocol, suites, states, identities[arm])
                kwargs = {'root': output/arm, 'ordinal': ordinal, 'task': task, 'identity': identities[arm],
                    'protocol': protocol, 'metadata': metadata, 'initial_state': initial, 'policy': policy,
                    'environment_factory': factory, 'seed_episode': seed_episode}
                row = execute_episode(**kwargs)
                # Actual completed-artifact resume is read-only and must be exact.
                require(execute_episode(**kwargs) == row, 'Completed receipt resume differs')
                validate_query_artifact(row, arm, admission)
                rows.append(row)
                print(json.dumps({'phase': 'pilot', 'arm': arm, 'completed': len(rows), 'total': 8,
                                  'successes': sum(r['success'] for r in rows)}), flush=True)
            require(tensor_state_digest(action._policy.state_dict()) == numerical['action_state_sha256'], 'Rollout mutated M4')
            if event is not None:
                require(tensor_state_digest(event.model.state_dict()) == numerical['event_state_sha256'], 'Rollout mutated event model')
            report = {'identity_digest': identities[arm]['digest'], 'episodes': rows, 'summary': summarize_rows(rows),
                          'audits': audit_panel(rows, identities[arm], tasks, protocol), 'event_model_load_seconds': load_seconds}
            report['renderer_repeat_audit'] = compare_inputs(report['audits'], prefix_audits)
            reports[arm] = write(output/arm/'report.json', report)
            del policy, event
            gc.collect()
            torch.cuda.empty_cache()
        comparisons = {a+' vs '+b: compare_inputs(reports[a]['audits'], reports[b]['audits'])
                       for a, b in itertools.combinations(ARMS, 2)}
        audits_complete = all(not x['error'] for r in reports.values() for x in r['audits'])
        all_inputs_exact = all(c['all_exact'] for c in comparisons.values())
        horizon_counts = {}
        for arm in ARMS:
            queries = [q for p in (output/arm).glob('*.queries.jsonl') for q in map(json.loads, p.read_text().splitlines())]
            horizon_counts[arm] = dict(Counter(str(q['policy']['used_horizon']) for q in queries))
        projection = {}
        for arm, report in reports.items():
            rows = report['episodes']
            # Conservative common bound: longest observed complete episode and
            # largest recording, applied to every formal episode with 1.2 margin.
            seconds = max(r['episode_wall_time_seconds'] or 0 for r in rows)
            size = max(Path(r['artifact']).stat().st_size if r['artifact'] else 0 for r in rows)
            projection[arm] = {'targeted_seconds_1p2': 1.2*467*seconds,
                'clean_seconds_1p2': 1.2*200*seconds, 'bytes_1p2': 1.2*667*size,
                'method': 'longest observed qualification episode and largest HDF5, 1.2 margin'}
        projected_seconds = guard.budget.seconds + sum(p['targeted_seconds_1p2']+p['clean_seconds_1p2'] for p in projection.values())
        projected_bytes = guard.output_bytes+sum(p['bytes_1p2'] for p in projection.values())
        fits = (projected_seconds <= cfg['resources']['aggregate_active_seconds']
                and projected_bytes <= cfg['resources']['output_limit_bytes']
                and projected_bytes <= guard.free_bytes-cfg['resources']['minimum_free_bytes']
                and all(p[k] <= cfg['resources']['per_formal_arm_panel_active_seconds']
                        for p in projection.values() for k in ('targeted_seconds_1p2', 'clean_seconds_1p2')))
        endpoint_check(cfg)
        guard.check()
        result.update(status='complete' if audits_complete else 'incomplete_artifact_audit',
            qualification_episodes=24, numerical_qualification_digest=numerical['digest'],
            model_load_seconds=model_load_seconds, arm_report_digests={k: v['digest'] for k, v in reports.items()},
            summaries={k: v['summary'] for k, v in reports.items()}, comparisons=comparisons,
            horizon_counts=horizon_counts, all_artifacts_valid=audits_complete,
            all_cross_arm_inputs_exact=all_inputs_exact, positive_gate_eligible=all_inputs_exact and audits_complete,
            projection=projection, projected_total_active_seconds=projected_seconds,
            projected_total_bytes=projected_bytes, projected_formal_fits=fits,
            formal_launch_authorized=False, success_used_to_tune=False)
    except BaseException as exc:
        result.update(status='stopped_preserved', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result['resources'] = guard.close()
        result['completed_at_utc'] = utc_now()
        write(invocation/'completion.json', result)
        write(output/'latest-completion.json', result)
        print(json.dumps({k: result[k] for k in ('status', 'error', 'qualification_episodes', 'resources') if k in result}), flush=True)


if __name__ == '__main__':
    main()
