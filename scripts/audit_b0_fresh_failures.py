#!/usr/bin/env python3
"""Read-only audit and saved-state diagnostics for the completed fresh B0 run.

Writes new analysis artifacts only. Never calls a policy or alters recorded outcomes.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import h5py
import numpy as np

from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.clean_comparison import utc_now
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.episode_recorder import read_episode_metadata
from eventground_wam.evaluation_protocol import load_evaluation_protocol
from eventground_wam.failure_analysis import inspect_trajectory


def save(args, name, value):
    result = seal_contract({
        'artifact_type': f'eventground_wam.fresh_v2_{name}',
        'created_at_utc': utc_now(), 'analysis_script_sha256': sha256_file(Path(__file__)),
        **value,
    })
    atomic_json(args.output_root / f'{name}.json', result)
    return result


def audit(args):
    from analyze_b0_fresh import checked_dev, paired_dev_interval

    tasks = json.loads((args.project_root / 'evaluation/splits/libero-plus-v1/dev-210.json').read_text())['tasks']
    reports = {p: read_sealed(args.work_root / 'dev-210' / p / 'evaluation-report.json')
               for p in ('b0', 'reference')}
    identities = {p: read_sealed(args.work_root / 'dev-210' / p / 'identity.json') for p in reports}
    rule = read_sealed(args.project_root / 'configs/evaluation/week4_fresh_v2_acceptance.json')
    for p in reports:
        assert len(reports[p]['episodes']) == reports[p]['episode_count'] == len(tasks) == 210
        assert reports[p]['identity_digest'] == identities[p]['digest']
        assert identities[p]['acceptance_rule_digest'] == rule['digest']
    protocol = load_evaluation_protocol()
    pairs, differences, strata = [], [], []
    fields = ('agent_rgb', 'wrist_rgb', 'proprioception', 'mujoco_state', 'executed_action')
    for i, task in enumerate(tasks):
        rows = [reports[p]['episodes'][i] for p in reports]
        checked = [checked_dev(Path(r['artifact']), r, identities[p], task, protocol)
                   for p, r in zip(reports, rows, strict=True)]
        record = {'ordinal': i, **task, 'exact_match': checked[0] == checked[1],
                  'scene_metadata_matched': checked[0][0] == checked[1][0],
                  'b0_success': rows[0]['success'], 'reference_success': rows[1]['success']}
        if not record['exact_match']:
            detail = []
            with h5py.File(rows[0]['artifact']) as a, h5py.File(rows[1]['artifact']) as b:
                first = [int(np.flatnonzero(np.isfinite(f['steps/policy_latency_ms'][:]))[0]) for f in (a, b)]
                record['first_query_steps'] = first
                for step in range(min(first) + 1):
                    for field in fields:
                        if field == 'executed_action' and step == min(first):
                            continue  # Policy actions are expected to differ after the shared prefix.
                        x, y = a[f'steps/{field}'][step], b[f'steps/{field}'][step]
                        delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
                        indices = np.argwhere(x != y)
                        if len(indices):
                            detail.append({'step': step, 'field': field, 'count': len(indices),
                                'max_absolute_difference': float(delta.max()),
                                'mean_absolute_difference': float(delta.mean()),
                                'coordinates': indices.tolist() if len(indices) < 100 else None,
                                'b0_values': [x[tuple(k)].item() for k in indices] if len(indices) < 100 else None,
                                'reference_values': [y[tuple(k)].item() for k in indices] if len(indices) < 100 else None})
                record['prefix_differences'] = detail
            record['artifacts'] = {p: {'path': r['artifact'], 'sha256': r['artifact_sha256']}
                                   for p, r in zip(reports, rows, strict=True)}
        pairs.append(record)
        differences.append(int(rows[0]['success']) - int(rows[1]['success']))
        strata.append((task['suite'], task['perturbation_category']))
        if (i + 1) % 30 == 0:
            print(json.dumps({'audited_pairs': i + 1}), flush=True)
    save(args, 'input-audit', {
        'individual_artifacts_validated': 420, 'pair_count': 210,
        'exact_matches': sum(r['exact_match'] for r in pairs),
        'report_digests': {p: r['digest'] for p, r in reports.items()},
        'acceptance_rule_digest': rule['digest'], 'pairs': pairs,
        'provisional_gap_pp': 100 * sum(differences) / len(differences),
        'provisional_interval_95_pp': paired_dev_interval(differences, strata),
        'scope': 'Full original artifact validation and exact-input audit; no tolerance, discarded pair, or replacement rollout. Interval is descriptive while the input gate fails.',
    })


def clean_rows(args):
    root = args.work_root / 'epochs/epoch-02/rollouts-clean-2000'
    report = read_sealed(root / 'evaluation-report.json')
    panel = read_sealed(args.project_root / 'evaluation/splits/week4-clean-comparison-v1/paired-2000.json')
    roots = {'b0': root, 'reference': args.reference_clean_root}
    rows = []
    for p, location in roots.items():
        for file in sorted((location / 'receipts').glob('*-libero_10-*.json')):
            receipt = read_sealed(file)
            i = receipt['ordinal']
            check = report['comparisons']['openvla'][i]
            assert receipt['digest'] == check[f'{p}_receipt_digest']
            assert receipt['pair'] == panel['pairs'][i]
            assert receipt['assessable'] and check['initial_inputs_exactly_matched']
            rows.append({**receipt['pair'], **receipt, 'policy': p, 'panel': 'clean-long'})
    assert len(rows) == 1000
    return rows


def language_rows(args):
    tasks = json.loads((args.project_root / 'evaluation/splits/libero-plus-v1/dev-210.json').read_text())['tasks']
    rows = []
    for p in ('b0', 'reference'):
        report = read_sealed(args.work_root / 'dev-210' / p / 'evaluation-report.json')
        for i, task in enumerate(tasks):
            if task['perturbation_category'] == 'Language Instructions':
                rows.append({**task, **report['episodes'][i], 'ordinal': i,
                             'policy': p, 'panel': 'dev-language'})
    assert len(rows) == 60
    return rows


def traces(args):
    from libero.libero.envs.env_wrapper import ControlEnv

    dense = args.mode == 'dense'
    stride = 1 if dense else 10
    if dense:
        source = read_sealed(args.output_root / 'saved-state-analysis.json')
        assert source['status'] == 'complete'
        rows = [r for r in source['episodes'] if r['policy'] == 'b0' and not r['success']
                and (r['panel'] == 'dev-language' or r['runtime_task_index'] in (1, 2, 7))]
        assert len(rows) == 78
    else:
        rows = clean_rows(args) + language_rows(args)
    groups = {}
    for r in rows:
        meta = read_episode_metadata(r['artifact'])
        r['instruction'] = meta['instruction']
        r['bddl_sha256'] = meta['bddl_sha256']
        r['bddl_path'] = meta['bddl_path']
        groups.setdefault(meta['bddl_path'], []).append(r)
    completed = []
    for bddl, entries in groups.items():
        assert sha256_file(Path(bddl)) == entries[0]['bddl_sha256']
        with contextlib.redirect_stdout(io.StringIO()):
            env = ControlEnv(bddl_file_name=bddl, use_camera_obs=False,
                has_renderer=False, has_offscreen_renderer=False, control_freq=20)
            env.seed(0)
            env.reset()
        try:
            goals = env.env.parsed_problem['goal_state']
            for row in entries:
                assert sha256_file(Path(row['artifact'])) == row['artifact_sha256']
                frames = []
                max_roundtrip_error = 0.
                max_eef_delta = 0.
                with h5py.File(row['artifact']) as f:
                    mask = f['steps/phase'][:] == 1
                    indices = np.flatnonzero(mask)
                    selected = sorted(set(indices[::stride].tolist()) | {int(indices[-1])})
                    states = f['steps/mujoco_state'][selected]
                    eef = f['steps/proprioception'][selected, :3]
                    for step, state, recorded_eef in zip(selected, states, eef, strict=True):
                        env.set_state(state)
                        env.env.sim.forward()
                        env._post_process()
                        env._update_observables(force=True)
                        error = float(np.max(np.abs(env.get_sim_state() - state)))
                        assert error < 1e-10
                        max_roundtrip_error = max(max_roundtrip_error, error)
                        obs = env.env._get_observations()
                        max_eef_delta = max(max_eef_delta, float(np.max(np.abs(obs['robot0_eef_pos'] - recorded_eef))))
                        contacts = [name for name, obj in env.env.objects_dict.items()
                            if env.env._check_grasp(gripper=env.env.robots[0].gripper, object_geoms=obj.contact_geoms)]
                        frames.append({'step': step,
                            'goal_predicates': [bool(env.env._eval_predicate(g)) for g in goals],
                            'grasp_contacts': contacts})
                row.update({'goals': goals, 'sampled_saved_states': frames,
                    'sample_stride': stride, 'final_saved_state_included': True,
                    'max_physics_roundtrip_error': max_roundtrip_error,
                    'max_fresh_eef_vs_recorded_delta_m': max_eef_delta,
                    'ever_sampled_goal_predicates': np.any([x['goal_predicates'] for x in frames], axis=0).tolist(),
                    'final_sampled_goal_predicates': frames[-1]['goal_predicates'],
                    'ever_sampled_grasp_contacts': sorted({n for x in frames for n in x['grasp_contacts']}),
                    'motion': inspect_trajectory(row['artifact'])})
                completed.append(row)
        finally:
            env.close()
        save(args, 'dense-failure-state-analysis' if dense else 'saved-state-analysis', {
            'status': 'complete' if len(completed) == len(rows) else 'running',
            'completed': len(completed), 'expected': len(rows), 'episodes': completed,
            'scope': ('Every recorded policy state in all 65 B0 failures on clean Long tasks 1,2,7 and all 13 B0 language failures; no inference or new rollout.' if dense else
                      'All 500 clean Long trials for both policies, and all 30 language dev trials for both policies. Every tenth recorded policy state plus final recorded state; no inference or new rollout.'),
            'limits': 'Pre-action states omit the final success-producing transition. Sampling can miss short events. Physics states round-trip; freshly recomputed sensors can differ from cached observations. Grasp contacts and motion flags are descriptive, not causal failure labels. Privileged predicates are used only for offline diagnostics.',
        })
        print(json.dumps({'completed': len(completed), 'expected': len(rows), 'bddl': bddl}), flush=True)


def previews(args):
    from analyze_week4_failures import render_preview

    analysis = read_sealed(args.output_root / 'saved-state-analysis.json')
    assert analysis['status'] == 'complete'
    rows = analysis['episodes']
    by_key = {(r['panel'], r['ordinal'], r['policy']): r for r in rows}
    selected = []
    for task in (1, 7, 2):
        candidates = sorted((r for r in rows if r['panel'] == 'clean-long'
            and r['policy'] == 'b0' and r['runtime_task_index'] == task), key=lambda r: r['ordinal'])
        failures = [r for r in candidates if not r['success'] and by_key[('clean-long', r['ordinal'], 'reference')]['success']]
        successes = [r for r in candidates if r['success'] and by_key[('clean-long', r['ordinal'], 'reference')]['success']]
        selected.extend(failures[:2] + successes[:1])
    selected.extend(sorted((r for r in rows if r['panel'] == 'dev-language'
        and r['policy'] == 'b0' and not r['success']), key=lambda r: r['ordinal']))
    result = []
    for r in selected:
        for policy in ('b0', 'reference'):
            row = by_key[(r['panel'], r['ordinal'], policy)]
            name = f"{r['panel']}-{r['ordinal']:04d}-{policy}"
            item = render_preview(row, args.output_root / 'previews' / f'{name}.mp4',
                                  f"{policy} | {r['panel']} | ordinal {r['ordinal']} | success={row['success']}")
            item.update({'panel': r['panel'], 'ordinal': r['ordinal'], 'policy': policy})
            result.append(item)
        print(json.dumps({'preview_pair': r['ordinal'], 'panel': r['panel']}), flush=True)
    save(args, 'preview-manifest', {'analysis_digest': analysis['digest'], 'previews': result,
        'selection': 'For clean tasks 1,7,2: first two B0-failure/reference-success pairs and first both-success pair by frozen ordinal. All 13 B0 language failures paired with reference. Illustrative clean selection; exhaustive B0 language-failure coverage. Preview images are lossy; HDF5 is source of record.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--reference-clean-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--mode', choices=('audit', 'traces', 'dense', 'previews'), required=True)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    {'audit': audit, 'traces': traces, 'dense': traces, 'previews': previews}[args.mode](args)


if __name__ == '__main__':
    main()
