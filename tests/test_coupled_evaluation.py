from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eventground_wam.b0_data import ActionNormalizer
from eventground_wam.b0_inference import B0InferencePolicy
from eventground_wam.b0_training import atomic_json, read_sealed, sha256_file
from eventground_wam.coupled_checkpoint import export_coupled_policy, load_coupled_policy
from eventground_wam.coupled_evaluation import load_contract, verify_endpoint, verify_envelope
from eventground_wam.coupled_inference import CoupledInferencePolicy
from eventground_wam.coupled_rollouts import claim_attempt
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.pixel_baselines import box_rgb


def bundle(root, arm):
    identity = seal_contract({'artifact_type': 'eventground_wam.coupled_production', 'arm': arm,
        'contract_digest': 'contract', 'design_digest': 'design', 'source_digest': 'source',
        'fixed_endpoint_updates': 117132, 'runtime': {'digest': 'runtime'}})
    recipe = seal_contract({'contract_digest': 'contract', 'production_source_digest': 'source',
                            'runtime_digest': 'runtime'})
    contract = {'arm': arm, 'training_contract_digest': 'contract', 'design_freeze_digest': 'design',
        'production_source_digest': 'source', 'training_identity_digest': identity['digest'],
        'updates': 117132, 'action_presentations': 3412044, 'future_pairs': 16259770,
        'checkpoint_name': f'{arm}.ckpt', 'execution_recipe_digest': recipe['digest']}
    endpoint = root / 'epochs/epoch-02'
    endpoint.mkdir(parents=True)
    (endpoint / f'{arm}.ckpt').write_bytes(b'policy fixture')
    (endpoint / 'full-training-predictor.pt').write_bytes(b'predictor fixture')
    report = seal_contract({'artifact_type': 'eventground_wam.coupled_epoch', 'epoch': 2,
        'selection': 'fixed-epoch-2-endpoint', 'decoder_free_coupled_export': True,
        'identity_digest': identity['digest'], 'global_step': 117132, 'samples_seen': 3412044,
        'future_pairs_seen': 16259770, 'sample_chain': 'samples', 'future_chain': 'future',
        'checkpoint_sha256': sha256_file(endpoint / f'{arm}.ckpt'),
        'predictor_sha256': sha256_file(endpoint / 'full-training-predictor.pt')})
    completion = seal_contract({**{k: report[k] for k in ('identity_digest', 'global_step', 'samples_seen',
        'future_pairs_seen', 'sample_chain', 'future_chain', 'checkpoint_sha256')},
        'artifact_type': 'eventground_wam.week7_auxiliary_training_complete',
        'status': 'training_complete_evaluation_pending', 'recipe_digest': 'contract',
        'endpoint_report_digest': report['digest']})
    for path, value in [(root / 'training-manifest.json', identity),
        (root / 'execution-recipe.json', recipe), (endpoint / 'training-manifest.json', identity),
        (endpoint / 'training-report.json', report), (root / 'training-report.json', completion)]:
        atomic_json(path, value)
    return contract


@pytest.mark.parametrize('arm', ['M3', 'M4'])
def test_endpoint_and_real_contract(arm, tmp_path):
    contract = load_contract(Path(__file__).resolve().parents[1], arm)
    assert contract['arm'] == arm and contract['updates'] == 117132
    fixture = bundle(tmp_path, arm)
    assert verify_endpoint(tmp_path, fixture)[1]['epoch'] == 2
    (tmp_path / 'training-report.json').unlink()
    with pytest.raises(FileNotFoundError):
        verify_endpoint(tmp_path, fixture)


@pytest.mark.parametrize(('file', 'key', 'value'), [
    ('epochs/epoch-02/training-report.json', 'epoch', 1),
    ('epochs/epoch-02/training-report.json', 'global_step', 117131),
    ('epochs/epoch-02/training-report.json', 'decoder_free_coupled_export', False),
    ('training-report.json', 'future_pairs_seen', 1),
    ('training-report.json', 'endpoint_report_digest', 'wrong'),
    ('training-report.json', 'sample_chain', 'wrong'),
    ('execution-recipe.json', 'runtime_digest', 'wrong'),
    ('training-manifest.json', 'arm', 'M4'),
])
def test_rejects_resealed_incomplete_or_mislinked_endpoint(tmp_path, file, key, value):
    contract = bundle(tmp_path, 'M3')
    path = tmp_path / file
    atomic_json(path, seal_contract({**{k: v for k, v in read_sealed(path).items() if k != 'digest'}, key: value}))
    with pytest.raises(ValueError):
        verify_endpoint(tmp_path, contract)


@pytest.mark.parametrize('filename', ['M4.ckpt', 'full-training-predictor.pt'])
def test_rejects_changed_export(tmp_path, filename):
    contract = bundle(tmp_path, 'M4')
    (tmp_path / 'epochs/epoch-02' / filename).write_bytes(b'changed')
    with pytest.raises(ValueError):
        verify_endpoint(tmp_path, contract)


@pytest.mark.parametrize('arm', ['M3', 'M4'])
def test_real_coupled_export_metadata_and_reload(arm, tmp_path):
    from test_coupled_policy import LORA_NAMES, activate, setup
    torch.set_num_threads(1)
    model, _, batch, images = setup(arm)
    activate(model)
    model.eval()
    metadata = {'identity_digest': 'training', 'global_step': 117132, 'epoch': 2,
                'checkpoint_selection': 'fixed-epoch-2-endpoint'}
    path = tmp_path / f'{arm}.ckpt'
    export_coupled_policy(path, model, LORA_NAMES, metadata=metadata)
    artifact = torch.load(path, weights_only=True)
    contract = {'arm': arm, 'updates': 117132, 'training_identity_digest': 'training'}
    verify_envelope(artifact, contract, model.proposal.normalizer)
    restored, _, _, _ = setup(arm)
    restored.eval()
    load_coupled_policy(path, restored, LORA_NAMES)
    with torch.inference_mode():
        assert torch.equal(model(batch['inputs'], batch['proprioception'], images).normalized_actions,
                           restored(batch['inputs'], batch['proprioception'], images).normalized_actions)
    artifact['header'] = seal_contract({**{k: v for k, v in artifact['header'].items() if k != 'digest'}, 'metadata': {**metadata, 'epoch': 1}})
    with pytest.raises(ValueError):
        verify_envelope(artifact, contract, model.proposal.normalizer)


@pytest.mark.parametrize('arm', ['M3', 'M4'])
def test_observations_only_adapter_matches_decoder_and_exact_training_images(arm):
    from test_coupled_policy import setup
    torch.set_num_threads(1)
    model, _, batch, _ = setup(arm)
    inputs = {k: v[:1] for k, v in batch['inputs'].items()}
    front = np.arange(256*256*3, dtype=np.uint8).reshape(256, 256, 3)
    wrist = front[::-1].copy()
    seen = []
    hook = model.register_forward_pre_hook(lambda m, args: seen.append(args))
    kwargs = dict(input_builder=lambda *a: inputs, normalizer=ActionNormalizer.identity(),
                  device=torch.device('cpu'), checkpoint=SimpleNamespace())
    coupled = CoupledInferencePolicy(policy=model, **kwargs)
    baseline = B0InferencePolicy(policy=model.policy, **kwargs)
    query = dict(agent_rgb=front, wrist_rgb=wrist, proprioception=batch['proprioception'][0].numpy(),
                 instruction='pick the cup', suite='libero_spatial')
    assert np.array_equal(coupled.predict_chunk(**query), baseline.predict_chunk(**query))
    assert np.array_equal(seen[0][2].numpy(), np.stack([box_rgb(front), box_rgb(wrist)])[None])
    assert seen[0][2].dtype == torch.float32 and not seen[0][2].requires_grad
    assert coupled.inference_summary()['decoded_action_count'] == 8
    with pytest.raises(TypeError):
        coupled.predict_chunk(**query, future_images=front)
    hook.remove()


def test_crash_before_artifact_consumes_attempt_and_different_identity_rejected(tmp_path):
    path = tmp_path / 'attempts.json'
    assert claim_attempt(path, {'checkpoint': 'one'}, 1) == 0
    assert claim_attempt(path, {'checkpoint': 'one'}, 1) == 1
    for _ in range(3):
        assert claim_attempt(path, {'checkpoint': 'one'}, 1) is None
    with pytest.raises(ValueError):
        claim_attempt(path, {'checkpoint': 'two'}, 1)


def test_real_training_metadata_provenance_is_accepted():
    from eventground_wam.coupled_evaluation import verify_training_identity
    root = Path(__file__).resolve().parents[1]
    for arm, directory in [('M3', 'week8-m3-feifei-launch-v1'), ('M4', 'week8-m4-lambda-launch-v1')]:
        contract = load_contract(root, arm)
        assert verify_training_identity(root / 'docs/evaluation' / directory / 'run', contract)[
            'digest'] == contract['training_identity_digest']


def test_recording_attempts_exhausted_before_render_do_not_retry_on_restart(tmp_path, monkeypatch):
    import sys
    import evaluate_clean_comparison as clean
    import evaluate_reference_policy as ref
    from eventground_wam.coupled_rollouts import evaluate_task
    from eventground_wam.evaluation_protocol import load_evaluation_protocol

    task = {'suite': 'libero_spatial', 'runtime_task_index': 0, 'classification_task_id': 1,
            'task_name': 'fixture', 'perturbation_category': 'Light Conditions',
            'difficulty_level': 1, 'episode_seed': 7}
    attempts = []
    def scene_failure(*args):
        attempts.append(1)
        raise OSError('fixture scene unavailable before recorder opens')
    suite = SimpleNamespace(get_task=lambda i: SimpleNamespace(name='fixture', language='instruction'),
                            get_task_bddl_file_path=scene_failure)
    monkeypatch.setitem(sys.modules, 'libero.libero.envs', SimpleNamespace(OffScreenRenderEnv=object))
    monkeypatch.setattr(clean, 'seed_episode', lambda seed: None)
    kwargs = dict(ordinal=0, manifest_task=task, suite=suite, policy=None,
        protocol=load_evaluation_protocol(), output_dir=tmp_path, project_revision='project',
        resume=True, checkpoint_id='M3', checkpoint_revision='revision')
    first = evaluate_task(**kwargs)
    assert not first.success and first.termination_reason == 'infrastructure_failure'
    assert first.retry_count == 1 and first.artifact is None and len(attempts) == 2
    second = evaluate_task(**kwargs)
    assert not second.success and len(attempts) == 2
    assert len(list(tmp_path.glob('*.attempt-*-error.json'))) == 2


def test_week9_actual_panel_freeze():
    from verify_week9_protocol import verify
    assert verify()['clean_trials'] == 2000
