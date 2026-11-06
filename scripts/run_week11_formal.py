"""Execute the frozen 2,001-episode Week 11 matrix using qualified pilot code."""
from __future__ import annotations

import argparse
import fcntl
import gc
import itertools
import json
import shutil
import time
from pathlib import Path

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
from eventground_wam.evaluation_comparisons import audit_panel, compare_inputs, summarize_rows
from eventground_wam.evaluation_protocol import load_evaluation_protocol, protocol_sha256
from eventground_wam.eventground_inference import EventGroundInferencePolicy
from eventground_wam.pixel_predictor import tensor_state_digest
from eventground_wam.week11_execution import execute_episode, immutable, validate_query_artifact
from eventground_wam.week11_formal import (
    FormalResourceGuard,
    all_attempt_latencies,
    projection_admission,
    relocated_runtime,
    schedule,
)
from eventground_wam.week11_resources import require_idle_gpu

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'docs/evaluation/week11-integration-v1'
PILOT_OUTPUT = pilot.DATA/'results/week11-integration-pilot-v1'


def package_check():
    source = read_sealed(ROOT/'week11-formal-source-manifest.json')
    for name, expected in source['files'].items():
        require(sha256_file(ROOT/name) == expected, 'Formal source changed: '+name)
    parent = pilot.package_check()
    require(parent['digest'] == source['qualified_source_digest'], 'Qualified source changed')
    return source


def qualified_evidence(source):
    analysis = read_sealed(EVIDENCE/'analysis.json')
    inventory = read_sealed(EVIDENCE/'remote/artifact-inventory.json')
    require(analysis['source_digest'] == source['qualified_source_digest']
            and analysis['remote_inventory_digest'] == inventory['digest'], 'Pilot evidence binding changed')
    require(read_sealed(PILOT_OUTPUT/'artifact-inventory.json') == inventory, 'Pilot inventory changed')
    for name, expected in inventory['files_sha256'].items():
        require(sha256_file(PILOT_OUTPUT/name) == expected, 'Pilot artifact changed: '+name)
    numerical = read_sealed(EVIDENCE/'remote/numerical-qualification.json')
    cohort = read_sealed(EVIDENCE/'remote/cohort.json')
    require(numerical['passed'] and numerical['digest'] == analysis['numerical_digest']
            and cohort['digest'] == analysis['cohort_digest'], 'Pilot qualification changed')
    return analysis, numerical, cohort


def inventory(cfg, panel):
    entry = cfg['panels'][panel]
    path = ROOT/entry['path']
    require(sha256_file(path) == entry['sha256'], 'Panel manifest changed')
    manifest = read_sealed(path)
    require(manifest['digest'] == entry['digest'] and manifest['count'] == entry['count'], 'Panel binding changed')
    if panel == 'clean-200':
        # Qualified clean inventory/state validation accepts the frozen pairs.
        adapted = {**cfg, 'panels': {**cfg['panels'], 'qualification-8': entry}}
        tasks, protocol, suites, states, _ = pilot.inventory(adapted)
    else:
        tasks = manifest['tasks']
        protocol = load_evaluation_protocol(ROOT/'configs/evaluation/libero_plus_v1.json')
        suites, states = {}, {}
    require(len(tasks) == entry['count'], 'Panel task count changed')
    return tasks, protocol, suites, states


def identity_for(cohort, admission, arm, panel, protocol, source):
    return seal_contract({'artifact_type': 'eventground_wam.week11_formal_panel',
        'arm': arm, 'panel': panel, 'mode': 'formal', 'cohort_digest': cohort['digest'],
        'week9_protocol_digest': cohort['parent_digest'], 'policy_objective': 'all-action-l1-binary-gripper-v2',
        'admission_digest': admission['digest'], 'project_git_revision': source['base_git_revision'],
        'evaluation_source_digest': source['digest'],
        'checkpoint_id': 'eventground-wam/week11-'+arm.lower(), 'checkpoint_revision': admission['m4_sha256'],
        'protocol_sha256': protocol_sha256(protocol),
        'acceptance_rule_digest': protocol.get('comparison', {}).get('acceptance_rule_digest'),
        'legacy_profile_and_acceptance_fields_are_format_only': True})


def run_panel(output, arm, panel, inputs, identity, admission, policy, guard, load_seconds):
    from evaluate_clean_comparison import seed_episode
    tasks, protocol, suites, states = inputs
    target = output/'panels'/arm/panel
    immutable(target/'identity.json', identity)
    immutable(target/'tasks.json', seal_contract({'tasks': tasks, 'selected_ordinals': list(range(len(tasks)))}))
    rows = []
    for ordinal, task in enumerate(tasks):
        guard.check()
        metadata, initial, factory = scene('clean' if panel == 'clean-200' else 'perturbed',
                                          task, protocol, suites, states, identity)
        row = execute_episode(root=target, ordinal=ordinal, task=task, identity=identity,
            protocol=protocol, metadata=metadata, initial_state=initial, policy=policy,
            environment_factory=factory, seed_episode=seed_episode)
        validate_query_artifact(row, arm, admission)
        rows.append(row)
        progress = {'arm': arm, 'panel': panel, 'completed': len(rows), 'total': len(tasks),
                    'successes': sum(r['success'] for r in rows), 'updated_at': utc_now()}
        atomic_json(target/'progress.json', progress)
        atomic_json(output/'progress.json', progress)
        print(json.dumps(progress), flush=True)
    guard.check()
    audits = audit_panel(rows, identity, tasks, protocol)
    attempt_timing = all_attempt_latencies(target, identity)
    report_path = target/'report.json'
    if report_path.exists():
        prior = read_sealed(report_path)
        require(prior['identity_digest'] == identity['digest'] and prior['episodes'] == rows
                and prior['audits'] == audits and prior['all_attempt_query_timing'] == attempt_timing,
                'Sealed panel changed on resume')
        return prior
    report = seal_contract({'artifact_type': 'eventground_wam.week11_formal_panel_report',
        'identity_digest': identity['digest'], 'episodes': rows, 'summary': summarize_rows(rows),
        'audits': audits, 'all_attempt_query_timing': attempt_timing,
        'model_load_seconds_current_invocation': load_seconds,
        'panel_active_seconds_all_invocations': guard.panel_budget.seconds,
        'resources': guard.snapshot(), 'no_outcome_selection': True,
        'status': 'audited' if all(a['error'] is None for a in audits) else 'audit_incomplete'})
    immutable(report_path, report)
    return report


def run(output, *, preflight_only=False):
    cfg = read_sealed(ROOT/'configs/evaluation/week11_integration_v1.json')
    source = read_sealed(ROOT/'week11-formal-source-manifest.json')
    analysis = read_sealed(EVIDENCE/'analysis.json')
    queue = schedule(cfg)
    guard = FormalResourceGuard(output, source['digest'], cfg['resources'], analysis['resources'])
    # Include preflight, model loading and audit in both the aggregate ledger
    # and the first unfinished panel's ledger, including failed invocations.
    pending = [(a, p) for a, p in queue if not (output/'panels'/a/p/'report.json').exists()]
    guard.set_panel(*(pending or queue)[0])
    invocation = output/'invocations'/utc_now().replace(':', '-')
    invocation.mkdir(parents=True, exist_ok=False)
    result = {'status': 'incomplete', 'started_at': utc_now(), 'source_digest': source['digest'],
              'training_updates': 0, 'qualification_episodes_added': 0, 'formal_episode_count': 2001}
    try:
        require(package_check() == source, 'Formal package differs')
        analysis, numerical, parent = qualified_evidence(source)
        require_idle_gpu()
        configure_deterministic_cuda()
        torch.set_num_threads(4)
        seed_training(17)
        torch.cuda.set_per_process_memory_fraction(.9)
        torch.cuda.reset_peak_memory_stats()
        admission, contract, endpoint, recipe = pilot.endpoint_check(cfg)
        require(admission == numerical['branch_identities'], 'Endpoint admission differs from qualification')
        baseline_runtime = runtime()
        require(baseline_runtime == relocated_runtime(parent['runtime'], ROOT, pilot.LIBERO),
                'Runtime differs from qualified pilot beyond child source path')
        from check_reference_policy import verify_libero
        require(verify_libero(pilot.LIBERO) == parent['benchmark'], 'Benchmark changed')
        inputs = {panel: inventory(cfg, panel) for panel in cfg['execution']['formal_panel_order']}
        cohort = seal_contract({'artifact_type': 'eventground_wam.week11_formal_cohort',
            'source_digest': source['digest'], 'qualified_source_digest': parent['source_digest'],
            'parent_digest': cfg['digest'], 'mapping_digest': parent['mapping_digest'],
            'qualification_cohort_digest': parent['digest'], 'numerical_digest': numerical['digest'],
            'runtime': baseline_runtime, 'runtime_difference_from_pilot': 'immutable child PYTHONPATH only',
            'benchmark': parent['benchmark'], 'admission_digest': admission['digest'],
            'schedule': [list(p) for p in queue], 'formal_episodes': 2001, 'resources': cfg['resources'],
            'fresh_formal_rollouts': True, 'training_updates': 0,
            'authorization': 'User: Can you continue to finish launching the eval'})
        immutable(output/'cohort.json', cohort)
        immutable(output/'admission.json', admission)
        completed = []
        for arm, panel in queue:
            path = output/'panels'/arm/panel/'report.json'
            if path.exists():
                expected = identity_for(cohort, admission, arm, panel, inputs[panel][1], source)
                require(read_sealed(path)['identity_digest'] == expected['digest'], 'Completed panel identity changed')
                completed.append((arm, panel))
        current_bytes = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        projection = projection_admission(cfg, analysis, shutil.disk_usage(output).free,
            completed=completed, active_seconds=guard.budget.seconds, output_bytes=current_bytes)
        immutable(invocation/'preflight.json', seal_contract({'passed': True, 'projection': projection,
            'cohort_digest': cohort['digest'], 'admission_digest': admission['digest'],
            'numerical_qualification_digest': numerical['digest'], 'pilot_inventory_rehashed': True,
            'resources': guard.snapshot()}))
        print(json.dumps({'phase': 'preflight_passed', 'projection': projection}), flush=True)
        if preflight_only:
            result['status'] = 'preflight_passed'
            return
        from eventground_wam.libero_sensor_noise import install_codec_independent_motion_blur
        install_codec_independent_motion_blur()
        tick = time.monotonic()
        action = pilot.load_action(contract, endpoint)
        action_load_seconds = time.monotonic()-tick
        require(tensor_state_digest(action._policy.state_dict()) == numerical['action_state_sha256'],
                'Loaded action model differs from qualified tensors')
        reports = {}
        for arm in cfg['execution']['arm_order']:
            event = None
            for panel_index, panel in enumerate(cfg['execution']['formal_panel_order']):
                guard.set_panel(arm, panel)
                tick = time.monotonic()
                if panel_index == 0 and arm != 'M4-fixed':
                    event = pilot.load_event(recipe)
                    require(tensor_state_digest(event.model.state_dict()) == numerical['event_state_sha256'],
                            'Loaded event model differs from qualified tensors')
                load_seconds = time.monotonic()-tick + (action_load_seconds if (arm, panel) == queue[0] else 0.)
                require(runtime() == baseline_runtime, 'Model loading or cross-panel runtime changed')
                policy = EventGroundInferencePolicy(action, arm, event, identities=admission, guard=guard.check)
                identity = identity_for(cohort, admission, arm, panel, inputs[panel][1], source)
                reports[(arm, panel)] = run_panel(output, arm, panel, inputs[panel], identity,
                    admission, policy, guard, load_seconds)
                require(tensor_state_digest(action._policy.state_dict()) == numerical['action_state_sha256'],
                        'Rollout mutated action model')
                if event is not None:
                    require(tensor_state_digest(event.model.state_dict()) == numerical['event_state_sha256'],
                            'Rollout mutated event model')
                del policy
            del event
            gc.collect()
            torch.cuda.empty_cache()
        comparisons = {panel: {a+' vs '+b: compare_inputs(reports[(a, panel)]['audits'], reports[(b, panel)]['audits'])
            for a, b in itertools.combinations(cfg['execution']['arm_order'], 2)}
            for panel in cfg['execution']['formal_panel_order']}
        artifacts_valid = all(r['status'] == 'audited' for r in reports.values())
        require(pilot.endpoint_check(cfg)[0] == admission, 'Endpoint changed during evaluation')
        guard.check()
        completion = seal_contract({'artifact_type': 'eventground_wam.week11_formal_completion',
            'cohort_digest': cohort['digest'], 'source_digest': source['digest'], 'completed_at': utc_now(),
            'status': 'complete' if artifacts_valid else 'audit_incomplete', 'formal_episodes': 2001,
            'all_artifacts_valid': artifacts_valid, 'comparisons': comparisons,
            'all_cross_arm_inputs_exact': all(c['all_exact'] for ps in comparisons.values() for c in ps.values()),
            'reports': {a+'/'+p: r['digest'] for (a, p), r in reports.items()},
            'resources': guard.snapshot(), 'step6_analysis_pending': True, 'training_updates': 0})
        if (output/'completion.json').exists():
            prior = read_sealed(output/'completion.json')
            require(prior['reports'] == completion['reports'] and prior['comparisons'] == comparisons,
                    'Completed evaluation changed')
        else:
            immutable(output/'completion.json', completion)
        result['status'] = completion['status']
    except BaseException as exc:
        result.update(status='stopped', exception=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        result.update(finished_at=utc_now(), resources=guard.close())
        immutable(invocation/'result.json', seal_contract(result))
        print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output/'.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.output, preflight_only=args.preflight_only)


if __name__ == '__main__':
    main()
