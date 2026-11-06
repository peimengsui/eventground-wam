"""Fixed-endpoint provenance and prospective Week 7 analysis, without model loading."""
from __future__ import annotations

import numpy as np

from eventground_wam.b0_training import read_sealed, sha256_file

CONTRACT_DIGEST = '4cef2ab85220f8185ab34adc249434546684efe9cf4e9bd38d467e21f93957de'
DESIGN_DIGEST = '287bba079fd5cbac60577b69e3abbe566553f5fdacf00d2041db2c1b53a04315'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_contract(root):
    contract = read_sealed(root / 'configs/evaluation/week7_m1_dev_v1.json')
    require(contract['digest'] == CONTRACT_DIGEST, 'evaluation contract changed')
    for name, digest in contract['source_files_sha256'].items():
        require(sha256_file(root / name) == digest, f'frozen evaluation input changed: {name}')
    design = read_sealed(root / 'configs/training/week8_coupled_design_v1.json')
    require(design['digest'] == DESIGN_DIGEST, 'coupled design changed')
    require(sha256_file(root / design['specification']['path']) == design['specification']['sha256'],
            'coupled design specification changed')
    for name, digest in design['parents'].items():
        require(read_sealed(root / name)['digest'] == digest, f'design parent changed: {name}')
    return contract


def verify_training_identity(training_root, contract):
    identity = read_sealed(training_root / 'training-manifest.json')
    require(identity['digest'] == contract['training_identity_digest'], 'wrong training identity')
    for key in ('recipe_digest', 'production_source_digest', 'arm'):
        require(identity[key] == contract[key], f'training {key} changed')
    require(identity['fixed_endpoint_updates'] == contract['updates'], 'wrong training endpoint')
    return identity


def verify_endpoint(training_root, contract):
    """Only accept the completed, cross-linked epoch-2 export; never fall back to epoch 1."""
    identity = verify_training_identity(training_root, contract)
    endpoint = training_root / 'epochs' / 'epoch-02'
    report = read_sealed(endpoint / 'training-report.json')
    completion = read_sealed(training_root / 'training-report.json')
    require(read_sealed(endpoint / 'training-manifest.json') == identity, 'epoch identity changed')
    require(report['artifact_type'] == 'eventground_wam.week7_auxiliary_epoch', 'wrong epoch report')
    require(completion['artifact_type'] == 'eventground_wam.week7_auxiliary_training_complete',
            'missing final completion record')
    require(completion['status'] == 'training_complete_evaluation_pending', 'training not complete')
    require(report['epoch'] == 2 and report['selection'] == 'fixed-epoch-2-endpoint',
            'only fixed epoch 2 may be evaluated')
    require(report['action_only_export_roundtrip_equal'] is True, 'export roundtrip failed')
    for value in (report, completion):
        for key, expected in {'identity_digest': identity['digest'], 'global_step': contract['updates'],
                              'samples_seen': contract['action_presentations'],
                              'future_pairs_seen': contract['future_pairs']}.items():
            require(value[key] == expected, f'incomplete endpoint: {key}')
    require(completion['recipe_digest'] == contract['recipe_digest'], 'completion recipe changed')
    require(completion['endpoint_report_digest'] == report['digest'], 'completion endpoint changed')
    for key in ('sample_chain', 'future_chain', 'checkpoint_sha256'):
        require(completion[key] == report[key], f'completion chain changed: {key}')
    for filename, key in ((contract['checkpoint_name'], 'checkpoint_sha256'),
                          ('auxiliary-predictor.pt', 'predictor_sha256')):
        require(sha256_file(endpoint / filename) == report[key], f'changed export: {filename}')
    return identity, report, completion


def verify_envelope(state, contract):
    from eventground_wam.b0_model import QWEN_REVISION
    from eventground_wam.b0_training import POLICY_FORMAT

    require(set(state) == {'format', 'qwen_revision', 'identity_digest', 'objective', 'state_dict'},
            'unexpected checkpoint envelope')
    expected = {'format': POLICY_FORMAT, 'qwen_revision': QWEN_REVISION,
                'identity_digest': contract['training_identity_digest'],
                'objective': 'all-action-l1-binary-gripper-v2'}
    require(all(state[k] == v for k, v in expected.items()), 'wrong action-only checkpoint envelope')


def paired_summary(left, right):
    """All tasks retained, unstratified common-index PCG64 seed-7 task bootstrap."""
    left, right = np.asarray(left), np.asarray(right)
    require(left.shape == right.shape and left.ndim == 1 and len(left) > 0, 'invalid paired outcomes')
    require(np.isin(left, [0, 1]).all() and np.isin(right, [0, 1]).all(), 'outcomes must be binary')
    delta = left.astype(np.int64) - right.astype(np.int64)
    indices = np.random.Generator(np.random.PCG64(7)).integers(0, len(left), (10000, len(left)))
    interval = np.quantile(delta[indices].mean(axis=1) * 100, [.025, .975], method='linear')
    return {'task_count': len(left), 'gap_percentage_points': float(delta.mean() * 100),
            'paired_95_interval_percentage_points': interval.tolist(),
            'left_only_success': int((delta == 1).sum()), 'right_only_success': int((delta == -1).sum()),
            'both_success': int(((left == 1) & (right == 1)).sum()),
            'both_failure': int(((left == 0) & (right == 0)).sum()),
            'bootstrap': {'draws': 10000, 'rng': 'PCG64', 'seed': 7, 'quantile_method': 'linear'}}


def artifact_inputs(path, row, identity, task, protocol):
    """Audit every canonical prefix frame, including first-query observations/state."""
    import h5py

    from eventground_wam.episode_recorder import (
        read_episode_metadata,
        sha256_array,
        validate_episode_file,
    )

    require(bool(row['artifact']) and sha256_file(path) == row['artifact_sha256'],
            'missing or changed episode artifact')
    summary = validate_episode_file(path, protocol)
    metadata = read_episode_metadata(path)
    require(bool(summary['success']) == row['success'], 'outcome differs from recorder')
    for key in ('checkpoint_id', 'checkpoint_revision', 'project_git_revision'):
        require(metadata[key] == identity[key], f'episode identity changed: {key}')
    for key in ('suite', 'runtime_task_index', 'classification_task_id', 'episode_seed',
                'perturbation_category', 'difficulty_level'):
        require(metadata[key] == task[key], f'episode task changed: {key}')
    with h5py.File(path) as handle:
        steps = handle['steps']
        queries = np.flatnonzero(np.isfinite(steps['policy_latency_ms'][:]))
        # A policy error before its first query remains a scored failure, with pairing unverified.
        if not len(queries):
            return None
        first = int(queries[0])
        require(first == protocol['simulation']['settling_steps'], 'noncanonical settling prefix')
        fields = ('agent_rgb', 'wrist_rgb', 'proprioception', 'mujoco_state', 'executed_action')
        inputs = {f'{i}:{key}': sha256_array(steps[key][i])
                  for i in range(first + 1) for key in fields
                  if not (key == 'executed_action' and i == first)}
    return {'first_query_step': first, 'prefix': inputs,
            'scene': {key: metadata[key] for key in
                      ('bddl_sha256', 'initial_state_sha256', 'instruction', 'episode_seed')}}
