"""Strict, decoder-free M3/M4 interface exports; not optimizer recovery state."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from eventground_wam.b0_model import b0_trainable_state_dict, load_b0_trainable_state_dict
from eventground_wam.b0_training import atomic_torch_save, cpu_tree
from eventground_wam.coupled_policy import DESIGN_DIGEST, INTERFACE_VERSION
from eventground_wam.demonstration_contracts import (
    canonical_json_bytes,
    contract_digest,
    seal_contract,
)
from eventground_wam.pixel_predictor import tensor_state_digest

FORMAT = 'eventground-wam-coupled-inference-state-v1'


def export_coupled_policy(path: Path, model, exact_lora_names, *, metadata):
    """Preserve inference tensors only; original pinned base weights stay external."""
    if path.exists():
        raise FileExistsError(path)
    states = {
        'policy': cpu_tree(b0_trainable_state_dict(model.policy, allowed_exact_names=exact_lora_names)),
        'future_core': cpu_tree(model.future_core.state_dict()),
        'coupling': cpu_tree(model.coupling.state_dict()),
    }
    header = seal_contract({
        'format': FORMAT, 'interface': INTERFACE_VERSION, 'design_digest': DESIGN_DIGEST,
        'arm': model.arm, 'future_config': asdict(model.future_core.config),
        'normalizer': asdict(model.proposal.normalizer), 'metadata': metadata,
        'state_sha256': {name: tensor_state_digest(state) for name, state in states.items()},
        'scope': 'Inference interface export, not production admission or optimizer-resume state.',
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(path, {'header': header, 'states': states})
    return header


def load_coupled_policy(path: Path, model, exact_lora_names):
    """Load into a caller-provided pinned base, checking every tensor before mutation."""
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    header, states = artifact['header'], artifact['states']
    if (header.get('digest') != contract_digest(header) or header.get('format') != FORMAT
            or header.get('design_digest') != DESIGN_DIGEST
            or header.get('interface') != INTERFACE_VERSION or header.get('arm') != model.arm
            or header.get('future_config') != asdict(model.future_core.config)
            or canonical_json_bytes(header.get('normalizer')) !=
            canonical_json_bytes(asdict(model.proposal.normalizer))):
        raise ValueError('coupled export identity, configuration or normalizer changed')
    expected = {
        'policy': b0_trainable_state_dict(model.policy, allowed_exact_names=exact_lora_names),
        'future_core': model.future_core.state_dict(), 'coupling': model.coupling.state_dict(),
    }
    if set(states) != set(expected) or set(header['state_sha256']) != set(expected):
        raise ValueError('coupled export tensor groups changed')
    for group, reference in expected.items():
        if (set(states[group]) != set(reference)
                or tensor_state_digest(states[group]) != header['state_sha256'][group]):
            raise ValueError(f'coupled export weights changed: {group}')
        for name, value in states[group].items():
            target = reference[name]
            if (value.shape != target.shape or value.dtype != target.dtype
                    or not torch.isfinite(value).all()):
                raise ValueError(f'coupled export tensor shape, precision or values changed: {name}')
    load_b0_trainable_state_dict(model.policy, states['policy'], allowed_exact_names=exact_lora_names)
    model.future_core.load_state_dict(states['future_core'], strict=True)
    model.coupling.load_state_dict(states['coupling'], strict=True)
    return header
