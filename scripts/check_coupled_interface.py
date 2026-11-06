#!/usr/bin/env python3
"""Bounded real-backbone interface smoke check; no optimization or rollouts."""
from __future__ import annotations

import argparse
import gc
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from check_pixel_predictor import verify_base_files
from train_auxiliary_policy import other_gpu_jobs

from eventground_wam.b0_model import b0_trainable_state_dict, load_pinned_qwen_processor
from eventground_wam.b0_oft import OFTInputBuilder, normalize_state
from eventground_wam.b0_training import (
    atomic_json,
    configure_deterministic_cuda,
    read_sealed,
    sha256_file,
)
from eventground_wam.coupled_policy import COUPLING_PARAMETERS, coupled_parameter_groups
from eventground_wam.coupled_setup import build_fresh_coupled_policy
from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.pixel_baselines import box_rgb
from eventground_wam.pixel_predictor import PixelCondition, tensor_state_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    signal.alarm(15*60)
    started = time.monotonic()
    if args.output.exists():
        raise FileExistsError(args.output)
    source = read_sealed(args.source/'coupled-source-manifest.json')
    for name, digest in source['files'].items():
        assert sha256_file(args.source/name) == digest, name
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                         '--format=csv,noheader,nounits'], text=True)
    assert not other_gpu_jobs(processes, os.getpid()), 'GPU has another compute owner'
    configure_deterministic_cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    verify_base_files(args.cache, read_sealed(args.source/'docs/model/Qwen3.5-2B-manifest.json'))
    builder = OFTInputBuilder(load_pinned_qwen_processor(cache_dir=str(args.cache), local_files_only=True))
    rgb = np.random.default_rng(17).integers(0, 256, (2, 256, 256, 3), dtype=np.uint8)
    encoding = builder(['pick up the black bowl and place it on the plate'],
                       [rgb[0].transpose(2, 0, 1)], [rgb[1].transpose(2, 0, 1)])
    encoding = {k: v.to(device='cuda', dtype=torch.bfloat16) if v.is_floating_point() else v.cuda()
                for k, v in encoding.items()}
    images = torch.from_numpy(np.stack([box_rgb(frame) for frame in rgb])[None]).cuda()
    state = torch.zeros(1, 8, device='cuda')
    targets = torch.zeros(1, 8, 7, device='cuda')
    targets[..., 6] = -1
    rows = []
    for arm in ('M3', 'M4'):
        model, predictor, names, provenance = build_fresh_coupled_policy(
            args.source, args.cache, torch.device('cuda'), arm=arm)
        groups = coupled_parameter_groups(model, predictor, names)
        original = tensor_state_digest(b0_trainable_state_dict(model.policy, allowed_exact_names=names))
        with torch.no_grad():
            baseline = model.policy(encoding, state)
            result = model.forward_with_features(encoding, state, images)
            assert torch.equal(result.action_output.raw_action_predictions, baseline.raw_action_predictions)
            adapter = model.policy.conditioning_adapter
            normalized_state = normalize_state(state, adapter.state_lower, adapter.state_upper)
            condition = PixelCondition(result.features.values, result.features.padding, images,
                normalized_state, result.candidate_actions, torch.tensor([8], device='cuda'))
            # Compare with the same inherited predictor execution context. In the
            # pinned Torch build this context also disables cuDNN; setting only
            # its global TF32 flag leaves a different convolution implementation.
            with torch.backends.cudnn.flags(allow_tf32=False):
                expected_tokens = predictor(condition).future_tokens
            assert torch.equal(result.future_tokens, expected_tokens)
            with torch.random.fork_rng(devices=[0]):
                torch.manual_seed(99)
                model.coupling.output.weight.normal_(std=.01)
        result = model.forward_with_features(encoding, state, images, normalized_action_targets=targets)
        assert not torch.equal(result.action_output.raw_action_predictions, baseline.raw_action_predictions)
        result.proposal_raw.retain_grad()
        result.action_output.loss.backward()
        assert result.proposal_raw.grad is None
        activity = {}
        for name, params in [('head', model.policy.action_decoder.parameters()),
                            ('state', adapter.proprioception_projector.parameters()),
                            ('lora', (dict(model.policy.named_parameters())[n] for n in names)),
                            ('future_core', model.future_core.parameters()),
                            ('coupling', model.coupling.parameters())]:
            values = [p.grad for p in params if p.grad is not None]
            assert values and all(torch.isfinite(v).all() for v in values)
            activity[name] = sum(float(v.float().square().sum()) for v in values)**.5
            assert activity[name] > 0
        assert all(p.grad is None for p in model.policy.parameters() if not p.requires_grad)
        output = predictor.residual_head if arm == 'M3' else predictor.output_projection
        assert output.weight.grad is None
        assert tensor_state_digest(b0_trainable_state_dict(model.policy, allowed_exact_names=names)) == original
        row = {'arm': arm, **provenance, 'zero_coupling_actions_bitwise_equal': True,
               'tokens_equal_original_predictor': True, 'active_coupling_changes_actions': True,
               'proposal_detached': True, 'action_gradient_norms': activity,
               'training_output_gradients_absent': True, 'frozen_base_gradients_absent': True,
               'shared_weights_unchanged': True, 'coupling_parameters': COUPLING_PARAMETERS,
               'parameter_groups': {g['name']: sum(p.numel() for p in g['params']) for g in groups}}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model, predictor, groups, result, baseline, condition, expected_tokens, adapter, output
        gc.collect()
        torch.cuda.empty_cache()
    result = seal_contract({'artifact_type': 'eventground_wam.week8_coupling_real_backbone_smoke',
        'status': 'passed', 'source_digest': source['digest'], 'arms': rows,
        'seconds': time.monotonic()-started, 'torch': str(torch.__version__),
        'gpu': torch.cuda.get_device_name(0), 'optimizer_updates': 0,
        'demonstration_payload_reads': 0, 'validation_payload_reads': 0, 'rollouts': 0,
        'input_scope': 'One seeded synthetic dual-camera observation and instruction per arm.',
        'limitations': 'Interface integration only; no process-recovery, full-batch device equivalence, 128-update timing or resource admission claim.'})
    atomic_json(args.output, result)
    print(json.dumps({'status': result['status'], 'digest': result['digest']}), flush=True)


if __name__ == '__main__':
    main()
