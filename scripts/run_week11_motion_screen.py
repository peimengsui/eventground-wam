"""Bounded 81-episode motion-adapter screen; no training or automatic expansion."""
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
from run_week11_motion_training import action_adapter, parity, replay_inputs
from week11_motion_adapter import ARMS, FORMAT, FrozenMotionPolicy, MotionInference
from week11_motion_screen import CONDITIONS, all_pairs, audit_queries, decision
from week11_replay_common import read, seal, sha
from week11_replay_common import write as sealed_write

from eventground_wam.pixel_predictor import tensor_state_digest


def write(path, value):
    sealed_write(path, value)
    return read(path)


def emit(output, phase, **fields):
    value = {'phase': phase, 'time_unix': time.time(), **fields}
    write(output/'progress.json', value)
    print(json.dumps(value), flush=True)


class RecordedMotion(MotionInference):
    def __init__(self, action, event, admission, guard):
        super().__init__(action, event, guard=guard)
        self.admission = admission

    def predict_chunk(self, **kwargs):
        try:
            return super().predict_chunk(**kwargs)
        finally:
            if self.last_query is not None:
                self.last_query.update(arm=self.action_policy._policy.arm, branch_identities=self.admission)


def latency(output, cfg, original, policies, rows, guard):
    labels = ['original-fixed', *ARMS]
    values = {a: [] for a in labels}
    details = []
    indices = cfg['latency_observation_indices']
    for i in [-1]*12 + list(range(len(indices))):
        slot = max(i, 0)
        name, obs, _, recorded = rows[indices[slot]]
        for k in range(3):
            arm = labels[(slot+k) % 3]
            policy = original if arm == 'original-fixed' else policies[arm]
            guard.check(); torch.cuda.synchronize(); tick = time.perf_counter()
            policy.predict_chunk(**obs)
            torch.cuda.synchronize(); ms = (time.perf_counter()-tick)*1000
            if arm != 'original-fixed':
                assert policy.last_query['prediction'] == recorded['prediction']
                assert policy.last_query['event']['categorical_probabilities'] == recorded['event']['categorical_probabilities']
            if i >= 0:
                values[arm].append(ms)
                details.append({'observation': name, 'condition': arm, 'ms': ms})
        if i >= 0 and i % 32 == 0:
            emit(output, 'latency', completed=i+1, total=len(indices))
    summary = {a: {'queries': len(v), 'median_ms': float(np.median(v)),
                   'p95_ms': float(np.percentile(v, 95))} for a, v in values.items()}
    ratios = {a: summary[a]['p95_ms']/summary['original-fixed']['p95_ms'] for a in ARMS}
    passed = all(v <= cfg['latency_ceiling_ratio'] for v in ratios.values())
    result = write(output/'latency.json', {'passed': passed, 'summary': summary, 'ratios': ratios,
        'rows': details, 'warmup_rounds': 12, 'endpoints': cfg['endpoints'],
        'order': 'rotating-interleaved-identical-observations', 'outliers_removed': False})
    assert passed, ('two-times original-fixed p95 latency ceiling', ratios)
    return result


def projection(cfg, inputs, measured):
    seconds, storage = 120., 128*2**20
    slowest = max(v['p95_ms'] for v in measured['summary'].values())
    for c in cfg['cases']:
        path = Path(c['artifacts']['h8-continuation/fixed8'])
        timing = read(path.with_suffix('.timing.json'))
        with h5py.File(path) as f:
            n = len(f['steps/executed_action'])
        maximum = 10+inputs[c['panel']][1]['benchmark']['suites'][c['task']['suite']]['max_policy_steps']
        # Scale historical simulator/recording cost by measured adapter inference cost.
        prior_p95 = float(np.percentile([q['latency_ms'] for q in timing['queries']], 95))
        seconds += 9*maximum*timing['episode_wall_time_seconds']/n*max(1., slowest/prior_p95)*1.2
        storage += 9*maximum*path.stat().st_size/n*1.2
    return seconds, storage


def run(bundle, output):
    import run_week11_formal as formal
    import run_week11_multihorizon_screen as screen
    import run_week11_pilot as pilot
    from evaluate_b0_full import FullCheckpointIdentity
    from evaluate_clean_comparison import seed_episode
    from evaluate_week9 import runtime, scene
    from week11_multihorizon_runtime import ROOT, admission, runtime_identity

    from eventground_wam.b0_training import configure_deterministic_cuda, seed_training
    from eventground_wam.evaluation_comparisons import audit_panel, latency_summary
    from eventground_wam.eventground_inference import EventGroundInferencePolicy
    from eventground_wam.week11_execution import execute_episode
    from eventground_wam.week11_formal import relocated_runtime
    from eventground_wam.week11_resources import ResourceGuard, require_idle_gpu

    cfg = read(bundle/'contract.json')
    for p, h in cfg['code_sha256'].items():
        assert sha(bundle/p) == h, p
    assert tuple(cfg['conditions']) == CONDITIONS and len(cfg['schedule']) == 81
    assert {(s['case_index'], s['repeat'], s['condition']) for s in cfg['schedule']} == {
        (i, r, a) for i in range(9) for r in (1, 2, 3) for a in CONDITIONS}
    output.mkdir(parents=True, exist_ok=False)
    write(output/'contract.json', cfg)
    guard = ResourceGuard(output, cfg['digest'], cfg['resources'])
    started = time.monotonic(); rows = []
    result = {'status': 'incomplete', 'contract_digest': cfg['digest'], 'planned_rollouts': 81,
              'training_updates': 0, 'automatic_extension': False}
    try:
        require_idle_gpu(); os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        configure_deterministic_cuda(); seed_training(17); torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.9); torch.cuda.reset_peak_memory_stats()
        training, source, _ = admission(); source2 = screen.source_check(training, source)
        assert source['digest'] == cfg['training_source_digest'] and source2['digest'] == cfg['screen_source_digest']
        rt_identity = runtime_identity()
        for p, h in cfg['input_sha256'].items():
            assert sha(p) == h, p
        train_root = Path(cfg['motion_training_root'])
        trained = read(train_root/'completion.json')
        assert trained['passed'] and trained['digest'] == cfg['training_completion_digest']
        audit = read(train_root/'independent-audit.json')
        assert audit['digest'] == cfg['training_audit_digest'] and audit['status'] == 'pass'
        train_cfg = read(train_root/'bundle/contract.json')
        assert train_cfg['digest'] == cfg['motion_contract_digest']
        integration = read(ROOT/'configs/evaluation/week11_integration_v1.json')
        original_admission, original_contract, original_endpoint, recipe = pilot.endpoint_check(integration)
        parent_source = formal.package_check(); rt = runtime()
        prior = read(pilot.DATA/'results/week11-formal-v1/cohort.json')
        assert rt == relocated_runtime(prior['runtime'], ROOT, pilot.LIBERO)
        from check_reference_policy import verify_libero
        assert verify_libero(pilot.LIBERO) == prior['benchmark']
        inputs = {p: formal.inventory(integration, p) for p in ('targeted-467', 'clean-200')}
        for c in cfg['cases']:
            task = inputs[c['panel']][0][c['ordinal']]
            assert all(task[k] == v for k, v in c['task'].items())
            assert inputs[c['panel']][1]['simulation']['settling_steps'] == 10
        from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur
        install_codec_independent_motion_blur()
        actions = screen.Actions(training, Path(cfg['parent_run_root']), original_endpoint)
        reference = actions.select('h8-continuation')
        reference_hash = tensor_state_digest(actions.model.state_dict())
        assert reference_hash == cfg['reference_state_sha256']
        event = pilot.load_event(recipe); event_hash = tensor_state_digest(event.model.state_dict())
        assert event_hash == read(formal.EVIDENCE/'remote/numerical-qualification.json')['event_state_sha256']
        historical = Path(cfg['cases'][0]['artifacts']['h8-continuation/fixed8'])
        reference_adm = json.loads(historical.with_suffix('.queries.jsonl').read_text().splitlines()[0])['policy']['branch_identities']
        assert reference_adm['m4_sha256'] == cfg['h8_endpoint']['policy_sha256']
        assert reference_adm['parent_original_admission_digest'] == original_admission['digest']
        admissions = {CONDITIONS[0]: seal({**reference_adm, 'motion_screen_contract': cfg['digest']})}
        policies = {CONDITIONS[0]: EventGroundInferencePolicy(reference, 'M4-EG-fixed8', event,
                    identities=admissions[CONDITIONS[0]], guard=guard.check)}
        adapted = {}
        for arm in ARMS:
            endpoint_dir = train_root/'training'/arm/'endpoint'
            end = read(endpoint_dir/'endpoint.json')
            assert end == cfg['endpoints'][arm]
            assert sha(endpoint_dir/'adapter.pt') == end['checkpoint_sha256']
            loaded = torch.load(endpoint_dir/'adapter.pt', map_location='cpu', weights_only=True)
            assert loaded['adapter_format'] == FORMAT and loaded['identity_digest'] == end['identity']['digest']
            assert loaded['global_step'] == 2000 and loaded['arm'] == arm
            model = FrozenMotionPolicy(actions.model, arm)
            model.adapter.load_state_dict(loaded['state_dict'], strict=True)
            model.requires_grad_(False).eval()
            assert tensor_state_digest(model.adapter.state_dict()) == end['adapter_state_sha256']
            action = action_adapter(reference, model)
            action.checkpoint = FullCheckpointIdentity(checkpoint_id='eventground-wam/week11-'+arm,
                checkpoint_revision=end['checkpoint_sha256'], training_manifest_digest=end['identity']['digest'],
                training_report_digest=end['digest'], selected_epoch=None, selected_step=2000)
            admissions[arm] = seal({**reference_adm, 'model': arm, 'm4_sha256': end['checkpoint_sha256'],
                'reference_sha256': end['reference_sha256'], 'adapter_endpoint': end['digest'],
                'adapter_identity': end['identity']['digest'], 'motion_screen_contract': cfg['digest']})
            adapted[arm] = action
            policies[arm] = RecordedMotion(action, event, admissions[arm], guard.check)
        write(output/'admissions.json', admissions)
        emit(output, 'endpoint_admission_passed', completed_rollouts=0, planned_rollouts=81)
        replay = replay_inputs(train_cfg)
        for arm in ARMS:
            parity(adapted[arm], replay, output/f'parity-{arm}.json', guard, zero=False)
        original = pilot.load_action(original_contract, original_endpoint)
        measured = latency(output, cfg, original, policies, replay, guard)
        del original; gc.collect(); torch.cuda.empty_cache()
        seconds, storage = projection(cfg, inputs, measured)
        assert seconds+guard.budget.seconds < cfg['resources']['qualification_active_seconds']
        assert storage < cfg['resources']['output_limit_bytes']
        assert shutil.disk_usage(output).free > storage+cfg['resources']['minimum_free_bytes']
        write(output/'preflight.json', {'passed': True, 'runtime': rt, 'runtime_identity': rt_identity,
            'projected_remaining_seconds_1p2': seconds, 'projected_bytes_1p2': storage,
            'latency_digest': measured['digest'], 'reference_state_sha256': reference_hash,
            'event_state_sha256': event_hash, 'all_456_h8_and_gripper_checks_passed': True})
        emit(output, 'qualified', projected_remaining_seconds=seconds, projected_bytes=storage, total=81)
        cohort = write(output/'cohort.json', {'artifact_type': 'eventground_wam.week11_motion_screen',
            'parent_digest': training['digest'], 'contract_digest': cfg['digest'], 'runtime': rt,
            'posthoc_selected_diagnostic': True, 'same_seed_repeats': True, 'training_updates': 0})
        identities = {}
        for item in cfg['schedule']:
            guard.check(); c = cfg['cases'][item['case_index']]
            arm, rep, panel = item['condition'], item['repeat'], c['panel']
            tasks, protocol, suites, states = inputs[panel]; task = tasks[c['ordinal']]
            root = output/'panels'/f'repeat-{rep}'/arm/panel
            key = (rep, arm, panel)
            if key not in identities:
                identity = formal.identity_for(cohort, admissions[arm], arm, panel, protocol, parent_source)
                identity.pop('digest')
                identity.update(mode='selected_diagnostic_repeated', repeat=rep,
                    evaluation_source_digest=cfg['digest'], training_endpoint=policies[arm].checkpoint.training_report_digest)
                identities[key] = write(root/'identity.json', identity)
            identity = identities[key]
            metadata, initial, factory = scene('clean' if panel == 'clean-200' else 'perturbed',
                task, protocol, suites, states, identity)
            row = execute_episode(root=root, ordinal=c['ordinal'], task=task, identity=identity,
                protocol=protocol, metadata=metadata, initial_state=initial, policy=policies[arm],
                environment_factory=factory, seed_episode=seed_episode)
            # Artifact/contract faults stop; task failures remain scored and never trigger replacement.
            artifacts = audit_panel([row], identity, [task], protocol)
            assert artifacts[0]['error'] is None, artifacts[0]
            query_audit = audit_queries(row, arm, admissions[arm])
            report = write(root/f'report-{c["ordinal"]:04d}.json', {'case': c['id'], 'condition': arm,
                'repeat': rep, 'result': row, 'artifact_audit': artifacts[0], 'query_audit': query_audit})
            rows.append({'case': c['id'], 'condition': arm, 'repeat': rep, 'success': row['success'],
                'artifact': row['artifact'], 'report': str(root/f'report-{c["ordinal"]:04d}.json'),
                'report_digest': report['digest'], 'policy_latencies_ms': row['policy_latencies_ms']})
            emit(output, 'evaluation', completed=len(rows), total=81, case=c['id'], condition=arm,
                 repeat=rep, success=row['success'], seconds=time.monotonic()-started)
            if len(rows) % 27 == 0:
                lat = {a: latency_summary([r['policy_latencies_ms'] for r in rows if r['condition'] == a])
                       for a in CONDITIONS}
                ratios = {a: lat[a]['all_queries']['p95_ms']/measured['summary']['original-fixed']['p95_ms']
                          for a in ARMS}
                passed = all(v <= cfg['latency_ceiling_ratio'] for v in ratios.values())
                write(output/f'rollout-latency-{len(rows):02d}.json',
                      {'passed': passed, 'summary': lat, 'ratios': ratios, 'no_queries_removed': True})
                assert passed, ('rollout p95 latency ceiling', ratios)
                assert tensor_state_digest(actions.model.state_dict()) == reference_hash
                assert tensor_state_digest(event.model.state_dict()) == event_hash
                for a in ARMS:
                    assert tensor_state_digest(adapted[a]._policy.adapter.state_dict()) == cfg['endpoints'][a]['adapter_state_sha256']
        pairs = all_pairs(cfg['cases'], rows)
        write(output/'prefix-audit.json', {'pairs': pairs, 'all_outcomes_retained': True,
              'unmatched_pairs': sum(not p['causally_matched'] for p in pairs)})
        outcome = decision(cfg['cases'], rows, pairs, integrity_valid=True)
        write(output/'analysis.json', outcome)
        for p, h in cfg['input_sha256'].items():
            assert sha(p) == h, p
        for p, h in cfg['code_sha256'].items():
            assert sha(bundle/p) == h, p
        assert runtime() == rt and admission()[1]['digest'] == cfg['training_source_digest']
        assert screen.source_check(training, source)['digest'] == cfg['screen_source_digest']
        result.update(status='complete', decision=outcome, model_states_unchanged=True)
    except BaseException as exc:
        result.update(status='stopped', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result.update(seconds=time.monotonic()-started, completed_rollouts=len(rows), rows=rows,
                      resources=guard.close())
        write(output/'completion.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    with (a.bundle/'run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(a.bundle, a.output)
