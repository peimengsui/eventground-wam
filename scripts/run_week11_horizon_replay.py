"""Frozen-observation, five-horizon inference with no environment execution."""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from week11_replay_common import HORIZONS, distance, read, sha, write


def capture(policy, obs, horizon):
    from eventground_wam.eventground_inference import routed_horizon
    values, handles = {}, []

    def save(name):
        def hook(_m, _a, output):
            values[name] = output.detach().float().cpu().numpy().copy()
        return hook

    def coupling(_m, args, output):
        values['query_hidden'] = args[0].detach().float().cpu().numpy().copy()
        values['residual'] = (output-args[0]).detach().float().cpu().numpy().copy()

    def final(_m, _a, output):
        values['normalized_actions'] = output.normalized_actions.detach().float().cpu().numpy().copy()
        values['raw_actions'] = output.raw_action_predictions.detach().float().cpu().numpy().copy()

    model = policy._policy
    handles.extend([model.future_core.register_forward_hook(save('future_tokens')),
        model.coupling.future_norm.register_forward_hook(save('normalized_future_tokens')),
        model.proposal.register_forward_hook(save('proposal')),
        model.coupling.register_forward_hook(coupling), model.register_forward_hook(final)])
    try:
        with routed_horizon(model, horizon, {}, device=policy._device):
            values['decoded_actions'] = policy.predict_chunk(**obs)
    finally:
        for handle in handles:
            handle.remove()
    assert all(np.isfinite(v).all() for v in values.values())
    return values


def run(args):
    import run_week11_formal as formal
    import run_week11_pilot as pilot

    from eventground_wam.b0_training import configure_deterministic_cuda, seed_training
    from eventground_wam.episode_recorder import read_episode_metadata
    from eventground_wam.pixel_predictor import tensor_state_digest
    from eventground_wam.week11_resources import ResourceGuard, require_idle_gpu

    cfg = read(args.bundle/'contract.json')
    for name, digest in cfg['code_sha256'].items():
        assert sha(args.bundle/name) == digest, name
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.bundle/'contract.json', args.output/'contract.json')
    limits = {'qualification_active_seconds': 1800, 'host_process_tree_rss_limit_bytes': 28*2**30,
        'gpu_peak_fraction_strictly_below': .9, 'output_limit_bytes': 2*2**30, 'minimum_free_bytes': 50*2**30}
    guard = ResourceGuard(args.output, cfg['digest'], limits)
    started = time.monotonic(); result = {'status': 'incomplete', 'model_calls': 0, 'training_updates': 0, 'simulator_steps': 0}
    try:
        require_idle_gpu()
        source = formal.package_check()
        assert source['digest'] == cfg['formal_source_digest']
        configure_deterministic_cuda(); seed_training(17); torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.9); torch.cuda.reset_peak_memory_stats()
        admission, contract, endpoint, _ = pilot.endpoint_check(read(pilot.ROOT/'configs/evaluation/week11_integration_v1.json'))
        assert admission == read(Path(cfg['raw_root'])/'admission.json')
        verified = {}
        for o in cfg['selection']['observations']:
            for p in (o['artifact'], o['paired_adaptive_artifact']):
                if p and p not in verified:
                    assert sha(p) == cfg['input_sha256'][p], p
                    assert sha(Path(p).with_suffix('.queries.jsonl')) == cfg['input_sha256'][str(Path(p).with_suffix('.queries.jsonl'))]
                    verified[p] = [Path(p).stat().st_size, Path(p).stat().st_mtime_ns]
        write(args.output/'preflight.json', {'source_digest': source['digest'], 'admission': admission,
            'recordings_verified': len(verified), 'contract_digest': cfg['digest'],
            'torch': torch.__version__, 'numpy': np.__version__, 'h5py': h5py.__version__})
        policy = pilot.load_action(contract, endpoint)
        before = tensor_state_digest(policy._policy.state_dict())
        results = []
        for index, entry in enumerate(cfg['selection']['observations']):
            guard.check()
            p = Path(entry['artifact']); meta = read_episode_metadata(p)
            with h5py.File(p, 'r') as f:
                s = f['steps']; step = entry['step']
                obs = {k: s[k][step] for k in ('agent_rgb', 'wrist_rgb', 'proprioception')}
                recorded = s['predicted_action_chunk'][step]
            obs.update(instruction=meta['instruction'], suite=meta['suite'])
            query = json.loads(p.with_suffix('.queries.jsonl').read_text().splitlines()[entry['query']])
            values = {}
            for h in (8, 2, 4, 16, 32):
                values[h] = capture(policy, obs, h); result['model_calls'] += 1
                assert np.array_equal(values[h]['proposal'], values[8]['proposal']), 'Proposal changed with horizon'
                assert np.array_equal(values[h]['query_hidden'], values[8]['query_hidden']), 'Backbone query changed with horizon'
                guard.check()
            used = query['policy']['used_horizon']
            assert np.array_equal(values[used]['decoded_actions'], recorded), f'Recorded-action parity failed: {entry["id"]}'
            repeat = capture(policy, obs, 8); result['model_calls'] += 1
            assert all(np.array_equal(repeat[k], values[8][k]) for k in repeat), 'Horizon-8 repeat parity failed'
            paired = None
            if entry['paired_adaptive_artifact']:
                paired_path = Path(entry['paired_adaptive_artifact'])
                paired_query = json.loads(paired_path.with_suffix('.queries.jsonl').read_text().splitlines()[entry['query']])
                with h5py.File(paired_path, 'r') as f:
                    for k in ('agent_rgb', 'wrist_rgb'):
                        assert np.array_equal(obs[k], f['steps'][k][entry['step']])
                    assert np.array_equal(obs['proprioception'].astype(np.float32), f['steps/proprioception'][entry['step']].astype(np.float32))
                    paired = np.array_equal(values[paired_query['policy']['used_horizon']]['decoded_actions'],
                                            f['steps/predicted_action_chunk'][entry['step']])
                    assert paired, 'Shared-prefix adaptive parity failed'
            metrics = {}
            for h in HORIZONS:
                metrics[str(h)] = {k: distance(values[8][k], values[h][k]) for k in
                    ('future_tokens', 'normalized_future_tokens', 'residual', 'raw_actions')}
                a, b = values[8]['decoded_actions'], values[h]['decoded_actions']
                metrics[str(h)].update(translation=distance(a[:, :3], b[:, :3]),
                    rotation=distance(a[:, 3:6], b[:, 3:6]),
                    gripper_disagree_actions=int(np.count_nonzero(a[:, 6] != b[:, 6])),
                    translation_prefix_rms=float(np.sqrt(np.mean((a[:min(h, 8), :3]-b[:min(h, 8), :3])**2))),
                    translation_suffix_rms=float(np.sqrt(np.mean((a[h:, :3]-b[h:, :3])**2))) if h < 8 else None)
            dest = args.output/'tensors'/f'{index:03d}.npz'; dest.parent.mkdir(exist_ok=True)
            np.savez_compressed(dest, **{f'h{h}_{k}': v for h, vals in values.items() for k, v in vals.items()})
            row = {**entry, 'used_horizon': used, 'prediction': query['policy']['prediction'],
                'recorded_action_parity': True, 'h8_repeat_all_tensors_exact': True,
                'shared_prefix_adaptive_parity': None if paired is None else bool(paired),
                'metrics': metrics, 'tensor_file': str(dest.relative_to(args.output)), 'tensor_sha256': sha(dest)}
            results.append(row); write(args.output/'observations'/f'{index:03d}.json', row)
            if index % 10 == 0:
                print(json.dumps({'completed_observations': index+1, 'total': len(cfg['selection']['observations']),
                    'seconds': time.monotonic()-started}), flush=True)
        after = tensor_state_digest(policy._policy.state_dict())
        assert after == before, 'Model state changed'
        assert all([Path(p).stat().st_size, Path(p).stat().st_mtime_ns] == st for p, st in verified.items())
        write(args.output/'replay.json', {'observations': results, 'contract_digest': cfg['digest'],
            'model_state_before': before, 'model_state_after': after, 'original_recordings_unchanged': True})
        result.update(status='complete', observations=len(results), pairs=len(cfg['selection']['pairs']),
                      model_state_unchanged=True, replay_digest=read(args.output/'replay.json')['digest'])
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        result['resources'] = guard.close()
        result['seconds'] = time.monotonic()-started
        write(args.output/'completion.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args())
