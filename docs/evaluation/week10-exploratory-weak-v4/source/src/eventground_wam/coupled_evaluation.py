"""Strict completed-endpoint and source checks for shared M3/M4 evaluation."""
from __future__ import annotations

from dataclasses import asdict

from eventground_wam.auxiliary_evaluation import DESIGN_DIGEST, require
from eventground_wam.b0_training import read_sealed, sha256_file
from eventground_wam.demonstration_contracts import canonical_json_bytes, contract_digest


def load_contract(root, arm):
    contract = read_sealed(root / 'configs/evaluation/week8_coupled_dev_v1.json')
    require(arm in ('M3', 'M4'), 'unsupported coupled arm')
    for name, digest in contract['source_files_sha256'].items():
        require(sha256_file(root / name) == digest, f'frozen input changed: {name}')
    design = read_sealed(root / 'configs/training/week8_coupled_design_v1.json')
    require(design['digest'] == DESIGN_DIGEST == contract['design_freeze_digest'], 'design changed')
    require(sha256_file(root / design['specification']['path']) == design['specification']['sha256'],
            'design specification changed')
    for name, digest in design['parents'].items():
        require(read_sealed(root / name)['digest'] == digest, f'design parent changed: {name}')
    # Keep the shared contract digest; arm-specific values are frozen inside it.
    return {**contract, **contract['arms'][arm], 'arm': arm}


def verify_training_identity(training_root, contract):
    identity = read_sealed(training_root / 'training-manifest.json')
    require(identity['digest'] == contract['training_identity_digest'], 'wrong training identity')
    expected = {'artifact_type': 'eventground_wam.coupled_production', 'arm': contract['arm'],
                'design_digest': contract['design_freeze_digest'],
                'contract_digest': contract['training_contract_digest'],
                'source_digest': contract['production_source_digest'],
                'fixed_endpoint_updates': contract['updates']}
    for key, value in expected.items():
        require(identity[key] == value, f'training {key} changed')
    recipe = read_sealed(training_root / 'execution-recipe.json')
    require(recipe['digest'] == contract['execution_recipe_digest'], 'execution recipe changed')
    require(recipe['contract_digest'] == identity['contract_digest']
            and recipe['production_source_digest'] == identity['source_digest']
            and recipe['runtime_digest'] == identity['runtime']['digest'], 'admission changed')
    return identity


def verify_source(source_root, contract):
    source = read_sealed(source_root / contract['training_source_manifest'])
    require(source['digest'] == contract['production_source_digest'], 'training source changed')
    for name, digest in source['files'].items():
        require(sha256_file(source_root / name) == digest, f'training source changed: {name}')
    return source


def endpoint_paths(training_root, contract):
    endpoint = training_root / 'epochs/epoch-02'
    return [training_root / 'training-report.json', endpoint / 'training-report.json',
            endpoint / 'training-manifest.json', endpoint / contract['checkpoint_name'],
            endpoint / 'full-training-predictor.pt']


def verify_endpoint(training_root, contract):
    """Fail closed on incomplete exposure, wrong identity, or either changed export."""
    identity = verify_training_identity(training_root, contract)
    endpoint = training_root / 'epochs/epoch-02'
    report = read_sealed(endpoint / 'training-report.json')
    completion = read_sealed(training_root / 'training-report.json')
    require(read_sealed(endpoint / 'training-manifest.json') == identity, 'epoch identity changed')
    require(report['artifact_type'] == 'eventground_wam.coupled_epoch', 'wrong epoch report')
    require(completion['artifact_type'] == 'eventground_wam.week7_auxiliary_training_complete'
            and completion['status'] == 'training_complete_evaluation_pending', 'training not complete')
    require(report['epoch'] == 2 and report['selection'] == 'fixed-epoch-2-endpoint'
            and report['decoder_free_coupled_export'] is True, 'wrong fixed endpoint export')
    for value in (report, completion):
        for key, expected in {'identity_digest': identity['digest'], 'global_step': contract['updates'],
                              'samples_seen': contract['action_presentations'],
                              'future_pairs_seen': contract['future_pairs']}.items():
            require(value[key] == expected, f'incomplete endpoint: {key}')
    require(completion['recipe_digest'] == contract['training_contract_digest'], 'completion contract changed')
    require(completion['endpoint_report_digest'] == report['digest'], 'completion endpoint changed')
    for key in ('sample_chain', 'future_chain', 'checkpoint_sha256'):
        require(completion[key] == report[key], f'completion chain changed: {key}')
    for filename, key in ((contract['checkpoint_name'], 'checkpoint_sha256'),
                          ('full-training-predictor.pt', 'predictor_sha256')):
        require(sha256_file(endpoint / filename) == report[key], f'changed export: {filename}')
    return identity, report, completion


def verify_envelope(artifact, contract, normalizer):
    """Bind tensor envelope to the intended training; loader separately checks model shapes."""
    import torch
    from eventground_wam.coupled_checkpoint import FORMAT
    from eventground_wam.coupled_policy import INTERFACE_VERSION
    from eventground_wam.pixel_predictor import PixelPredictorConfig, tensor_state_digest

    require(set(artifact) == {'header', 'states'}, 'unexpected checkpoint envelope')
    header, states = artifact['header'], artifact['states']
    require(header['digest'] == contract_digest(header), 'export header changed')
    expected = {'format': FORMAT, 'interface': INTERFACE_VERSION, 'design_digest': DESIGN_DIGEST,
                'arm': contract['arm'], 'future_config': asdict(PixelPredictorConfig()),
                'metadata': {'identity_digest': contract['training_identity_digest'],
                             'global_step': contract['updates'], 'epoch': 2,
                             'checkpoint_selection': 'fixed-epoch-2-endpoint'}}
    require(all(header[k] == v for k, v in expected.items()), 'wrong coupled envelope identity')
    require(canonical_json_bytes(header['normalizer']) == canonical_json_bytes(asdict(normalizer)),
            'export normalizer changed')
    require(set(states) == set(header['state_sha256']) == {'policy', 'future_core', 'coupling'},
            'unexpected inference tensor groups')
    for group, tensors in states.items():
        require(tensor_state_digest(tensors) == header['state_sha256'][group], 'export tensors changed')
        require(all(torch.isfinite(t).all() for t in tensors.values()), 'nonfinite export tensors')
    require(not any(n.startswith(('image_decoder.', 'residual_head.', 'output_projection.'))
                    for n in states['future_core']), 'training outputs in inference export')
    return header
