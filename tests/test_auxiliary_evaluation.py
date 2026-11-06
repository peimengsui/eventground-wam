from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from eventground_wam.auxiliary_evaluation import (
    load_contract,
    paired_summary,
    verify_endpoint,
    verify_envelope,
)
from eventground_wam.b0_model import QWEN_REVISION
from eventground_wam.b0_training import POLICY_FORMAT, atomic_json, read_sealed, sha256_file
from eventground_wam.demonstration_contracts import seal_contract


def bundle(root):
    identity = seal_contract({'arm': 'M1', 'recipe_digest': 'recipe', 'production_source_digest': 'source',
                              'fixed_endpoint_updates': 117132})
    contract = {'arm': 'M1', 'recipe_digest': 'recipe', 'production_source_digest': 'source',
                'training_identity_digest': identity['digest'], 'updates': 117132,
                'action_presentations': 3412044, 'future_pairs': 16259770, 'checkpoint_name': 'M1.ckpt'}
    endpoint = root / 'epochs/epoch-02'
    endpoint.mkdir(parents=True)
    (endpoint / 'M1.ckpt').write_bytes(b'policy fixture')
    (endpoint / 'auxiliary-predictor.pt').write_bytes(b'predictor fixture')
    report = seal_contract({'artifact_type': 'eventground_wam.week7_auxiliary_epoch',
        'epoch': 2, 'selection': 'fixed-epoch-2-endpoint', 'action_only_export_roundtrip_equal': True,
        'identity_digest': identity['digest'], 'global_step': 117132, 'samples_seen': 3412044,
        'future_pairs_seen': 16259770, 'sample_chain': 'samples', 'future_chain': 'future',
        'checkpoint_sha256': sha256_file(endpoint / 'M1.ckpt'),
        'predictor_sha256': sha256_file(endpoint / 'auxiliary-predictor.pt')})
    completion = seal_contract({**{k: report[k] for k in ('identity_digest', 'global_step', 'samples_seen',
                                'future_pairs_seen', 'sample_chain', 'future_chain', 'checkpoint_sha256')},
        'artifact_type': 'eventground_wam.week7_auxiliary_training_complete',
        'status': 'training_complete_evaluation_pending', 'recipe_digest': 'recipe',
        'endpoint_report_digest': report['digest']})
    for path, value in [(root / 'training-manifest.json', identity),
                         (endpoint / 'training-manifest.json', identity),
                         (endpoint / 'training-report.json', report),
                         (root / 'training-report.json', completion)]:
        atomic_json(path, value)
    return contract


def test_frozen_contract_and_design_parents_are_intact():
    contract = load_contract(Path(__file__).resolve().parents[1])
    assert contract['task_count'] == 210


def test_accepts_only_complete_cross_linked_endpoint(tmp_path):
    contract = bundle(tmp_path)
    assert verify_endpoint(tmp_path, contract)[1]['global_step'] == 117132
    (tmp_path / 'training-report.json').unlink()
    with pytest.raises(FileNotFoundError):
        verify_endpoint(tmp_path, contract)


@pytest.mark.parametrize(('file', 'key', 'value'), [
    ('epochs/epoch-02/training-report.json', 'epoch', 1),
    ('epochs/epoch-02/training-report.json', 'global_step', 117131),
    ('epochs/epoch-02/training-report.json', 'samples_seen', 3412043),
    ('epochs/epoch-02/training-report.json', 'future_pairs_seen', 16259769),
    ('epochs/epoch-02/training-report.json', 'action_only_export_roundtrip_equal', False),
    ('training-report.json', 'future_chain', 'changed'),
    ('training-report.json', 'endpoint_report_digest', 'wrong'),
    ('training-report.json', 'status', 'running'),
    ('training-manifest.json', 'production_source_digest', 'changed'),
])
def test_rejects_incomplete_or_wrong_provenance(tmp_path, file, key, value):
    contract = bundle(tmp_path)
    path = tmp_path / file
    artifact = read_sealed(path)
    artifact[key] = value
    atomic_json(path, seal_contract({k: v for k, v in artifact.items() if k != 'digest'}))
    with pytest.raises(ValueError):
        verify_endpoint(tmp_path, contract)


@pytest.mark.parametrize('filename', ['M1.ckpt', 'auxiliary-predictor.pt'])
def test_rejects_export_corruption(tmp_path, filename):
    contract = bundle(tmp_path)
    (tmp_path / 'epochs/epoch-02' / filename).write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='changed export'):
        verify_endpoint(tmp_path, contract)


def test_envelope_rejects_foreign_and_auxiliary_weights():
    contract = {'training_identity_digest': 'identity'}
    state = {'format': POLICY_FORMAT, 'qwen_revision': QWEN_REVISION, 'identity_digest': 'identity',
             'objective': 'all-action-l1-binary-gripper-v2', 'state_dict': {}}
    verify_envelope(state, contract)
    for key, value in [('identity_digest', 'foreign'), ('qwen_revision', 'other-base'),
                       ('predictor', {}), ('objective', 'different-loss')]:
        changed = copy.deepcopy(state)
        changed[key] = value
        with pytest.raises(ValueError):
            verify_envelope(changed, contract)


def test_bootstrap_uses_frozen_common_indices_and_keeps_failures():
    left = np.array([1, 0, 1, 1, 0, 0])
    right = np.array([0, 1, 1, 0, 0, 0])
    summary = paired_summary(left, right)
    indices = np.random.Generator(np.random.PCG64(7)).integers(0, 6, (10000, 6))
    expected = np.quantile((left - right)[indices].mean(axis=1) * 100, [.025, .975], method='linear')
    assert summary['paired_95_interval_percentage_points'] == expected.tolist()
    assert summary['left_only_success'] == 2 and summary['right_only_success'] == 1
    assert summary['both_failure'] == 2 and summary['task_count'] == 6
    assert paired_summary(left, left)['paired_95_interval_percentage_points'] == [0, 0]


def test_input_audit_checks_intermediate_settling_frames(tmp_path, monkeypatch):
    import h5py

    from eventground_wam import episode_recorder
    from eventground_wam.auxiliary_evaluation import artifact_inputs

    task = {'suite': 'suite', 'runtime_task_index': 0, 'classification_task_id': 0, 'episode_seed': 7,
                'perturbation_category': 'test', 'difficulty_level': 'Long'}
    identity = {'checkpoint_id': 'M1', 'checkpoint_revision': 'revision', 'project_git_revision': 'git'}
    meta = dict(**task, **identity, bddl_sha256='bddl', initial_state_sha256='initial', instruction='task')
    monkeypatch.setattr(episode_recorder, 'read_episode_metadata', lambda p: meta)
    monkeypatch.setattr(episode_recorder, 'validate_episode_file', lambda p, protocol: {'success': False})
    path = tmp_path / 'episode.hdf5'
    with h5py.File(path, 'w') as handle:
        steps = handle.create_group('steps')
        steps['policy_latency_ms'] = [np.nan, np.nan, 20.0]
        for field in ('agent_rgb', 'wrist_rgb', 'proprioception', 'mujoco_state', 'executed_action'):
            steps[field] = np.zeros((3, 2))
    protocol = {'simulation': {'settling_steps': 2}}
    def audit():
        return artifact_inputs(path, {'artifact': str(path), 'artifact_sha256': sha256_file(path),
                                      'success': False}, identity, task, protocol)
    before = audit()
    with h5py.File(path, 'r+') as handle:
        handle['steps/executed_action'][2] = 1  # First policy action may differ.
    assert audit() == before
    with h5py.File(path, 'r+') as handle:
        handle['steps/agent_rgb'][1] = 1  # Interior settling frame must match too.
    assert audit() != before
