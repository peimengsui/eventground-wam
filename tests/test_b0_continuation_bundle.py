"""The new objective must never enter evaluation as a legacy or partial endpoint."""
import sys
from pathlib import Path

import pytest

from eventground_wam.b0_continuation import OBJECTIVE
from eventground_wam.b0_model import QWEN_REVISION
from eventground_wam.b0_training import atomic_json, sha256_file
from eventground_wam.clean_comparison import ComparisonIntegrityError
from eventground_wam.demonstration_contracts import seal_contract

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from evaluate_b0_continuation import verify_bundle


def fixture_bundle(root):
    normalizer = root / 'normalizer.json'
    normalizer.write_text('pinned normalizer')
    (root / 'B0.ckpt').write_bytes(b'checkpoint-content')
    config = {'digest': 'new-experiment'}
    manifest = seal_contract({
        'artifact_type': 'eventground_wam.b0_all_l1_continuation_manifest',
        'mode': 'train', 'arm': 'B', 'experiment_digest': config['digest'],
        'recipe': {'recipe_id': 'all-l1-b', 'objective': OBJECTIVE, 'qwen_revision': QWEN_REVISION,
                   'policy_family': 'qwen35-oft-b0-v1', 'lora_rank': 32},
        'source': {'normalization_file_sha256': sha256_file(normalizer)},
    })
    report = {
        'artifact_type': 'eventground_wam.all_l1_continuation_report', 'status': 'complete',
        'identity_digest': manifest['digest'], 'experiment_digest': config['digest'],
        'selected_additional_step': 15000, 'additional_updates': 15000,
        'additional_samples': 435945, 'cumulative_updates': 87000,
        'reload_validation_passed': True, 'validation': {'metrics': {'sample_count': 2560}},
        'checkpoint_sha256': sha256_file(root / 'B0.ckpt'),
    }
    atomic_json(root / 'training-manifest.json', manifest)
    atomic_json(root / 'training-report.json', seal_contract(report))
    return normalizer, config, report


def test_only_exact_complete_endpoint_can_load(tmp_path):
    normalizer, config, report = fixture_bundle(tmp_path)
    _, checkpoint = verify_bundle(tmp_path, normalizer, config)
    assert checkpoint.selected_step == 87000
    for key, value in [('selected_additional_step', 14000), ('additional_updates', 14000),
                       ('additional_samples', 435944), ('reload_validation_passed', False),
                       ('status', 'running')]:
        changed = {**report, key: value}
        atomic_json(tmp_path / 'training-report.json', seal_contract(changed))
        with pytest.raises(ComparisonIntegrityError):
            verify_bundle(tmp_path, normalizer, config)
    atomic_json(tmp_path / 'training-report.json', seal_contract(report))
    (tmp_path / 'B0.ckpt').write_bytes(b'changed-checkpoint')
    with pytest.raises(ComparisonIntegrityError, match='bytes changed'):
        verify_bundle(tmp_path, normalizer, config)


def test_pilot_and_legacy_objective_are_not_formal_candidates(tmp_path):
    normalizer, config, _ = fixture_bundle(tmp_path)
    from eventground_wam.b0_training import read_sealed

    original = read_sealed(tmp_path / 'training-manifest.json')
    for updates in [{'mode': 'pilot'}, {'recipe': {
            **original['recipe'], 'objective': 'continuous-l1-plus-gripper-bce-v1'}}]:
        changed = {k: v for k, v in {**original, **updates}.items() if k != 'digest'}
        atomic_json(tmp_path / 'training-manifest.json', seal_contract(changed))
        with pytest.raises(ComparisonIntegrityError):
            verify_bundle(tmp_path, normalizer, config)
