"""Construct fresh coupled arms from the sealed design and original CPT modules."""
from __future__ import annotations

from pathlib import Path

import torch

from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_fresh import load_fresh_recipe
from eventground_wam.b0_model import b0_trainable_state_dict
from eventground_wam.b0_oft import build_oft_policy
from eventground_wam.b0_training import read_sealed, seed_training, sha256_file
from eventground_wam.coupled_policy import DESIGN_DIGEST, METHODS, CoupledPolicy
from eventground_wam.jepa_predictor import JepaPredictorConfig, load_jepa_checkpoint
from eventground_wam.pixel_predictor import (
    PixelPredictorConfig,
    load_pixel_checkpoint,
    tensor_state_digest,
)


def load_coupled_design(root: Path):
    design = read_sealed(root/'configs/training/week8_coupled_design_v1.json')
    if (design['digest'] != DESIGN_DIGEST or
            sha256_file(root/design['specification']['path']) != design['specification']['sha256']):
        raise ValueError('frozen coupled design or specification changed')
    for name, digest in design['parents'].items():
        if read_sealed(root/name)['digest'] != digest:
            raise ValueError(f'coupled design parent changed: {name}')
    return design


def load_original_future(design, arm, device):
    if arm not in METHODS:
        raise ValueError('coupled arm must be M3 or M4')
    initial = design['arms'][arm]['initialization']
    path = Path(initial['future_module_path'])
    if sha256_file(path) != initial['future_module_sha256']:
        raise ValueError('original CPT module checksum differs from frozen design')
    # The inherited predictor constructors seed Torch globally inside a CPU fork.
    # Save CUDA streams as well here, so loading future tensors cannot reset them.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        predictor, header = (load_pixel_checkpoint(path) if arm == 'M3' else load_jepa_checkpoint(path))
    expected = PixelPredictorConfig() if arm == 'M3' else JepaPredictorConfig()
    if predictor.config != expected:
        raise ValueError('CPT predictor architecture differs from accepted original')
    return predictor.to(device), header


def build_fresh_coupled_policy(root: Path, cache_dir: Path, device, *, arm: str):
    design = load_coupled_design(root)
    if arm not in METHODS:
        raise ValueError('coupled arm must be M3 or M4')
    recipe = load_fresh_recipe(root/'configs/training/week4_b0_fresh_all_l1_v2.json')
    normalization = root/'docs/model/week4-fresh-v2/action-statistics.json'
    if sha256_file(normalization) != recipe['normalization_sha256']:
        raise ValueError('action normalization changed')
    seed_training(17)
    policy, names = build_oft_policy(recipe, str(cache_dir), device)
    initial = tensor_state_digest(b0_trainable_state_dict(policy, allowed_exact_names=names))
    if initial != design['arms'][arm]['initialization']['shared_tensor_digest']:
        raise ValueError('fresh shared initialization differs from B0/M1/M2')
    predictor, header = load_original_future(design, arm, device)
    model = CoupledPolicy(policy, predictor, load_b0_action_normalizer(normalization), arm=arm).to(device)
    model.train()
    predictor.train()
    return model, predictor, names, {
        'design_digest': design['digest'], 'shared_initialization_digest': initial,
        'original_future_module_sha256': design['arms'][arm]['initialization']['future_module_sha256'],
        'original_future_header_digest': header['digest'], 'base_recipe_digest': recipe['digest'],
        'normalization_sha256': recipe['normalization_sha256'],
        'production_training_authorized_by_builder': False,
    }
