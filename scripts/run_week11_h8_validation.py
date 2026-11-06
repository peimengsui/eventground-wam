"""Qualify event-free fixed H8 and run a frozen, fresh 64-case paired screen."""
from __future__ import annotations

import argparse
import fcntl
import gc
import json
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from week11_h8_validation import CONDITIONS, paired_summary, prefix_pair, select_cases
from week11_replay_common import read, seal, sha
from week11_replay_common import write as sealed_write

from eventground_wam.pixel_predictor import tensor_state_digest


def write(path, value):
    sealed_write(path, value)
    return read(path)


def emit(output, phase, **fields):
    value = {'phase': phase, 'time_unix': time.time(), **fields}
    write(output/'progress.json', value); print(json.dumps(value), flush=True)


def load_observation(path, frame):
    with h5py.File(path) as f:
        meta = json.loads(f['episode/metadata_json'][()])
        obs = {k: f['steps'][k][frame] for k in ('agent_rgb', 'wrist_rgb', 'proprioception')}
        expected = f['steps/predicted_action_chunk'][frame]
    obs.update(instruction=meta['instruction'], suite=meta['suite'])
    return obs, expected


def qualify(cfg, output, policies, event_control, guard):
    parity = []; observations = {}
    for artifact in cfg['h8_recordings']:
        path = Path(artifact['artifact'])
        queries = [json.loads(s) for s in path.with_suffix('.queries.jsonl').read_text().splitlines()]
        for i in range(len(queries)):
            guard.check(); obs, expected = load_observation(path, 10+8*i)
            actual = policies['h8-continuation'].predict_chunk(**obs)
            assert np.array_equal(actual, expected), (str(path), i, 'event-free H8 parity')
            observations[str(path), i] = (obs, expected, queries[i]['policy'])
        parity.append({'case': artifact['case'], 'recorded_h8_queries_exact': len(queries)})
        emit(output, 'h8_event_removal_parity', completed_cases=len(parity), total=9)
    for path in cfg['original_recordings']:
        guard.check(); obs, expected = load_observation(path, 10)
        assert np.array_equal(policies['vanilla-m4'].predict_chunk(**obs), expected), path
    write(output/'recorded-parity.json', {'passed': True, 'h8_recordings': parity,
        'h8_queries': sum(r['recorded_h8_queries_exact'] for r in parity),
        'vanilla_original_recordings': len(cfg['original_recordings']), 'new_simulator_steps': 0})
    labels = ['vanilla-m4', 'h8-continuation', 'h8-with-event']
    timing_policies = {**policies, 'h8-with-event': event_control}
    values = {a: [] for a in labels}; rows = []
    anchors = cfg['timing_anchors']
    for round_index in range(-12, len(anchors)*3):
        slot = max(round_index, 0) % len(anchors); block = max(round_index, 0)//len(anchors)
        anchor = anchors[slot]; obs, expected, saved = observations[anchor['artifact'], anchor['query']]
        outputs = {}
        for k in range(3):
            label = labels[(slot+block+k) % 3]; policy = timing_policies[label]
            guard.check(); torch.cuda.synchronize(); tick = time.perf_counter()
            outputs[label] = policy.predict_chunk(**obs)
            torch.cuda.synchronize(); elapsed = (time.perf_counter()-tick)*1000
            if round_index >= 0:
                values[label].append(elapsed)
                rows.append({'condition': label, 'block': block, 'anchor': slot, 'ms': elapsed})
        assert np.array_equal(outputs['h8-continuation'], expected)
        assert np.array_equal(outputs['h8-with-event'], expected)
        assert event_control.last_query['prediction'] == saved['prediction']
        assert event_control.last_query['event']['categorical_probabilities'] == saved['event']['categorical_probabilities']
        if round_index >= 0 and round_index % 54 == 0:
            emit(output, 'matched_input_latency', completed=round_index+1, total=3*len(anchors))
    summary = {a: {'queries': len(v), 'median_ms': float(np.median(v)), 'p95_ms': float(np.percentile(v, 95))}
               for a, v in values.items()}
    ratio = summary['h8-continuation']['p95_ms']/summary['vanilla-m4']['p95_ms']
    passed = ratio <= cfg['hard_latency_ratio_ceiling']
    result = write(output/'latency.json', {'passed': passed, 'summary': summary, 'rows': rows,
        'h8_to_vanilla_p95_ratio': ratio, 'within_10_percent': ratio <= 1.1,
        'unique_anchors': len(anchors), 'blocks': 3, 'warmup_rounds': 12,
        'all_h8_event_on_off_actions_exact': True, 'all_saved_event_forecasts_exact': True,
        'outliers_removed': False, 'order': 'rotating within anchor and across three blocks'})
    assert passed, ('fixed H8 latency ceiling', ratio)
    return result


def projection(cfg, inputs):
    rates = {}
    for entry in cfg['h8_recordings']:
        path = Path(entry['artifact']); timing = read(path.with_suffix('.timing.json'))
        with h5py.File(path) as f:
            n = len(f['steps/executed_action']); meta = json.loads(f['episode/metadata_json'][()])
        rates.setdefault(meta['suite'], []).append((timing['episode_wall_time_seconds']/n, path.stat().st_size/n))
    seconds, storage = 180., 256*2**20
    for c in cfg['cases']:
        suite = c['task']['suite']; maximum = 10+inputs[c['panel']][1]['benchmark']['suites'][suite]['max_policy_steps']
        seconds += 2*maximum*max(x[0] for x in rates[suite])*1.2
        storage += 2*maximum*max(x[1] for x in rates[suite])*1.2
    return seconds, storage


def run(bundle, output):
    import run_week11_formal as formal
    import run_week11_multihorizon_screen as screen
    import run_week11_pilot as pilot
    from evaluate_clean_comparison import seed_episode
    from evaluate_week9 import runtime, scene
    from week11_multihorizon_runtime import ROOT, admission, runtime_identity

    from eventground_wam.b0_training import configure_deterministic_cuda, seed_training
    from eventground_wam.evaluation_comparisons import audit_panel, latency_summary
    from eventground_wam.eventground_inference import EventGroundInferencePolicy
    from eventground_wam.week11_execution import execute_episode, validate_query_artifact
    from eventground_wam.week11_formal import relocated_runtime
    from eventground_wam.week11_resources import ResourceGuard, require_idle_gpu

    cfg = read(bundle/'contract.json')
    for p, h in cfg['code_sha256'].items():
        assert sha(bundle/p) == h, p
    manifests = {p: read(ROOT/entry['path']) for p, entry in cfg['panel_manifests'].items()}
    assert select_cases(manifests, set(cfg['excluded_case_ids'])) == cfg['cases']
    assert tuple(cfg['conditions']) == CONDITIONS and len(cfg['schedule']) == 128
    assert {(x['case_index'], x['condition']) for x in cfg['schedule']} == {
        (i, a) for i in range(64) for a in CONDITIONS}
    output.mkdir(parents=True, exist_ok=False); write(output/'contract.json', cfg)
    guard = ResourceGuard(output, cfg['digest'], cfg['resources'])
    started = time.monotonic(); rows = []
    result = {'status': 'incomplete', 'contract_digest': cfg['digest'], 'planned_rollouts': 128,
              'training_updates': 0, 'new_checkpoint_selection': False}
    try:
        require_idle_gpu(); os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        configure_deterministic_cuda(); seed_training(17); torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.9); torch.cuda.reset_peak_memory_stats()
        training, source, _ = admission(); source2 = screen.source_check(training, source)
        assert source['digest'] == cfg['training_source_digest'] and source2['digest'] == cfg['screen_source_digest']
        for p, h in cfg['input_sha256'].items():
            assert sha(p) == h, p
        integration = read(ROOT/'configs/evaluation/week11_integration_v1.json')
        original_admission, original_contract, original_endpoint, recipe = pilot.endpoint_check(integration)
        parent_source = formal.package_check(); rt = runtime(); rt_identity = runtime_identity()
        prior = read(pilot.DATA/'results/week11-formal-v1/cohort.json')
        assert rt == relocated_runtime(prior['runtime'], ROOT, pilot.LIBERO)
        from check_reference_policy import verify_libero
        assert verify_libero(pilot.LIBERO) == prior['benchmark']
        inputs = {p: formal.inventory(integration, p) for p in cfg['panel_manifests']}
        for c in cfg['cases']:
            assert all(inputs[c['panel']][0][c['ordinal']][k] == v for k, v in c['task'].items())
        seconds, storage = projection(cfg, inputs)
        assert seconds+guard.budget.seconds < cfg['resources']['qualification_active_seconds']
        assert storage < cfg['resources']['output_limit_bytes']
        assert shutil.disk_usage(output).free > storage+cfg['resources']['minimum_free_bytes']
        from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur
        install_codec_independent_motion_blur()
        actions = screen.Actions(training, Path(cfg['parent_run_root']), original_endpoint)
        h8 = actions.select('h8-continuation')
        vanilla = pilot.load_action(original_contract, original_endpoint)
        action_policies = dict(zip(CONDITIONS, (vanilla, h8), strict=True))
        states = {a: tensor_state_digest(p._policy.state_dict()) for a, p in action_policies.items()}
        assert states['h8-continuation'] == cfg['h8_state_sha256']
        assert states['vanilla-m4'] == read(formal.EVIDENCE/'remote/numerical-qualification.json')['action_state_sha256']
        event = pilot.load_event(recipe); event_hash = tensor_state_digest(event.model.state_dict())
        assert event_hash == read(formal.EVIDENCE/'remote/numerical-qualification.json')['event_state_sha256']
        admissions, policies = {}, {}
        for a, p in action_policies.items():
            p._policy.requires_grad_(False).eval()
            fields = dict(original_admission)
            if a == 'h8-continuation':
                end = actions.headers[a]
                fields.update(m4_identity=end['identity']['digest'], m4_endpoint=end['digest'],
                    m4_completion=read(Path(cfg['parent_run_root'])/'training'/a/'train.json')['digest'],
                    predictor_sha256=end['predictor_sha256'])
            admissions[a] = seal({**fields, 'model': a, 'm4_sha256': p.checkpoint.checkpoint_revision,
                'action_checkpoint': p.checkpoint.training_report_digest, 'event_branch_used': False,
                'parent_original_admission_digest': original_admission['digest'],
                'validation_contract': cfg['digest']})
            policies[a] = EventGroundInferencePolicy(p, 'M4-fixed', identities=admissions[a], guard=guard.check)
        write(output/'admissions.json', admissions)
        event_control = EventGroundInferencePolicy(h8, 'M4-EG-fixed8', event, identities=admissions['h8-continuation'], guard=guard.check)
        measured = qualify(cfg, output, policies, event_control, guard)
        assert tensor_state_digest(event.model.state_dict()) == event_hash
        del event_control, event; gc.collect(); torch.cuda.empty_cache()
        for a, p in action_policies.items():
            assert tensor_state_digest(p._policy.state_dict()) == states[a]
        assert seconds+guard.budget.seconds < cfg['resources']['qualification_active_seconds']
        write(output/'preflight.json', {'passed': True, 'runtime': rt, 'runtime_identity': rt_identity,
            'projected_remaining_seconds_1p2': seconds, 'projected_bytes_1p2': storage,
            'latency_digest': measured['digest'], 'action_state_sha256': states,
            'event_state_sha256': event_hash, 'event_unloaded_before_rollouts': True})
        emit(output, 'qualified', projected_remaining_seconds=seconds, projected_bytes=storage, total=128)
        cohort = write(output/'cohort.json', {'artifact_type': 'eventground_wam.week11_h8_validation',
            'parent_digest': training['digest'], 'contract_digest': cfg['digest'], 'runtime': rt,
            'case_selection_uses_outcomes': False, 'training_updates': 0})
        identities = {}
        for item in cfg['schedule']:
            guard.check(); case = cfg['cases'][item['case_index']]; arm, panel = item['condition'], case['panel']
            tasks, protocol, suites, initial_states = inputs[panel]; task = tasks[case['ordinal']]
            root = output/'panels'/arm/panel; key = (arm, panel)
            if key not in identities:
                identity = formal.identity_for(cohort, admissions[arm], arm, panel, protocol, parent_source)
                identity.pop('digest'); identity.update(mode='prespecified_continuation_validation',
                    evaluation_source_digest=cfg['digest'], training_endpoint=policies[arm].checkpoint.training_report_digest)
                identities[key] = write(root/'identity.json', identity)
            identity = identities[key]
            metadata, initial, factory = scene('clean' if panel == 'clean-200' else 'perturbed',
                task, protocol, suites, initial_states, identity)
            row = execute_episode(root=root, ordinal=case['ordinal'], task=task, identity=identity, protocol=protocol,
                metadata=metadata, initial_state=initial, policy=policies[arm], environment_factory=factory,
                seed_episode=seed_episode)
            artifacts = audit_panel([row], identity, [task], protocol)
            assert artifacts[0]['error'] is None, artifacts[0]
            assert validate_query_artifact(row, 'M4-fixed', admissions[arm])
            assert not any(row['query_exceptions'])
            report = write(root/f'report-{case["ordinal"]:04d}.json', {'case': case['id'], 'condition': arm,
                'result': row, 'artifact_audit': artifacts[0]})
            rows.append({'case': case['id'], 'condition': arm, 'success': row['success'], 'artifact': row['artifact'],
                'report': str(root/f'report-{case["ordinal"]:04d}.json'), 'report_digest': report['digest'],
                'policy_latencies_ms': row['policy_latencies_ms']})
            emit(output, 'validation', completed=len(rows), total=128, case=case['id'], condition=arm,
                 success=row['success'], seconds=time.monotonic()-started)
            if len(rows) % 32 == 0:
                lat = {a: latency_summary([r['policy_latencies_ms'] for r in rows if r['condition'] == a]) for a in CONDITIONS}
                ratio = lat['h8-continuation']['all_queries']['p95_ms']/lat['vanilla-m4']['all_queries']['p95_ms']
                write(output/f'latency-{len(rows):03d}.json', {'summary': lat, 'ratio': ratio,
                    'passed': ratio <= cfg['hard_latency_ratio_ceiling'], 'warmup_queries_removed': False})
                for a, p in action_policies.items():
                    assert tensor_state_digest(p._policy.state_dict()) == states[a]
                assert ratio <= cfg['hard_latency_ratio_ceiling'], ('rollout latency ceiling', ratio)
        indexed = {(r['case'], r['condition']): r for r in rows}; pairs = []; panels = {}
        for case in cfg['cases']:
            a, b = [indexed[case['id'], arm] for arm in CONDITIONS]
            pair = {'case': case['id'], 'panel': case['panel'], 'suite': case['task']['suite'],
                    CONDITIONS[0]: a['success'], CONDITIONS[1]: b['success'],
                    **prefix_pair(a['artifact'], b['artifact'])}
            pairs.append(pair)
        for panel in cfg['panel_manifests']:
            selected = [p for p in pairs if p['panel'] == panel]; matched = [p for p in selected if p['causally_matched']]
            panels[panel] = {'all_outcomes': paired_summary(selected), 'matched_only': paired_summary(matched) if matched else None,
                             'unmatched_cases': [p['case'] for p in selected if not p['causally_matched']]}
        clean = panels['clean-200']['all_outcomes']; targeted = panels['targeted-467']['all_outcomes']
        matched_all = all(p['causally_matched'] for p in pairs)
        directional = matched_all and targeted['net_successes'] >= 2 and clean['net_successes'] >= 0 and measured['within_10_percent']
        analysis = write(output/'analysis.json', {'pairs': pairs, 'panels': panels,
            'all_pairs_matched': matched_all, 'directional_support': directional,
            'stronger_targeted_statistical_support': directional and targeted['paired_95_bootstrap_percentage_points'][0] > 0
                and targeted['exact_two_sided_mcnemar_p'] < .05,
            'baseline_promotion': False, 'training_seed_replicates': 1,
            'selection_qualification': cfg['selection_qualification'], 'latency_digest': measured['digest']})
        for a, p in action_policies.items():
            assert tensor_state_digest(p._policy.state_dict()) == states[a]
        for p, h in cfg['input_sha256'].items():
            assert sha(p) == h, p
        for p, h in cfg['code_sha256'].items():
            assert sha(bundle/p) == h, p
        assert runtime() == rt and admission()[1]['digest'] == cfg['training_source_digest']
        assert screen.source_check(training, source)['digest'] == cfg['screen_source_digest']
        result.update(status='complete', analysis_digest=analysis['digest'], models_unchanged=True,
                      all_pairs_matched=matched_all, directional_support=directional)
    except BaseException as exc:
        result.update(status='stopped', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result.update(seconds=time.monotonic()-started, completed_rollouts=len(rows), rows=rows, resources=guard.close())
        write(output/'completion.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    with (a.bundle/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); run(a.bundle, a.output)
