#!/usr/bin/env python3
"""Frozen, group-separated training-only comparison of absolute/residual JEPA."""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import resource
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from check_pixel_predictor import verify_base_files
from run_jepa_baselines import encode
from train_jepa_cpt import configure

from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_model import load_pinned_qwen_model, load_pinned_qwen_processor
from eventground_wam.b0_training import atomic_json, atomic_torch_save
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.future_pairs import CAMERAS, HORIZONS, read_sealed, sha256_file
from eventground_wam.jepa_baselines import collapse_diagnostics
from eventground_wam.jepa_features import FrozenQwenVisualTargets
from eventground_wam.jepa_followup import (
    ResidualJepaPredictor,
    backward_batch,
    errors,
    from_tokens,
    head_span,
    motion_bin,
    predict,
    select_groups,
    span_lower_bound,
    training_plan,
)
from eventground_wam.jepa_predictor import (
    JepaPredictor,
    jepa_prediction_loss,
    load_jepa_checkpoint,
)
from eventground_wam.pixel_baselines import group_weights
from eventground_wam.pixel_data import FutureInputBuilder, FuturePairDataset
from eventground_wam.pixel_predictor import FrozenQwenFeatures, PixelCondition, tensor_state_digest
from eventground_wam.pixel_training import rate_factor

ROOT = Path(__file__).resolve().parents[1]


def save(path, kind, **values):
    result = seal_contract({'artifact_type': kind, **values})
    atomic_json(path, result)
    return result


def progress(output, phase, **values):
    value = {'phase': phase, 'time_unix': time.time(), **values}
    atomic_json(output/'progress.json', value)
    print(json.dumps(value), flush=True)


def assemble(bank, indices, device='cuda'):
    indices = list(map(int, indices))
    features = [bank['features'][i] for i in indices for _ in HORIZONS]
    values = torch.nn.utils.rnn.pad_sequence(features, batch_first=True).to(device)
    mask = torch.arange(values.shape[1], device=device)[None] >= torch.tensor(
        [len(x) for x in features], device=device)[:, None]
    condition = PixelCondition(values, mask, bank['images'][indices].repeat_interleave(5, 0).to(device),
        bank['state'][indices].repeat_interleave(5, 0).to(device), bank['actions'][indices].flatten(0, 1).to(device),
        torch.tensor(HORIZONS, device=device).repeat(len(indices)))
    return (condition, bank['current'][indices].repeat_interleave(5, 0).to(device),
            bank['targets'][indices].flatten(0, 1).to(device),
            torch.arange(len(indices), device=device).repeat_interleave(5))


def prepare(args, contract):
    dataset = FuturePairDataset(args.pairs, args.data_root/
        f'datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo', split='train',
        expected_manifest_sha256=contract['pair_manifest_sha256'], cache_episodes=1,
        normalizer=load_b0_action_normalizer(ROOT/'docs/model/week4-fresh-v2/action-statistics.json'),
        state_bounds=read_sealed(ROOT/'configs/training/week4_b0_fresh_all_l1_v2.json')['state_normalization'])
    selected = contract['selection']
    groups = [json.loads(x) for x in (ROOT/'configs/datasets/libero-plus-trajectory-v2/groups.jsonl').read_text().splitlines()]
    assert selected == select_groups(groups, contract['seed'])
    for s in selected:
        source = dataset.sources[s['episode_index']]
        assert all(source[k] == s[k] for k in ('task_key', 'group_digest', 'split', 'episode_length'))
    model_files = verify_base_files(args.cache, read_sealed(ROOT/'docs/model/Qwen3.5-2B-manifest.json'))
    full = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True).cuda()
    processor = load_pinned_qwen_processor(cache_dir=str(args.cache), local_files_only=True)
    encoder, builder = FrozenQwenFeatures(full.model), FutureInputBuilder(processor)
    teacher = FrozenQwenVisualTargets(full.model.visual, processor.image_processor)
    accepted = read_sealed(ROOT/'docs/evaluation/week6-jepa-targets-v2/report.json')
    before, vision_before = tensor_state_digest(encoder.state_dict()), tensor_state_digest(teacher.visual.state_dict())
    assert vision_before == accepted['vision_state_after']
    for name, obj in [('vision', type(teacher.visual)), ('processor', type(teacher.image_processor))]:
        assert sha256_file(Path(inspect.getfile(obj))) == accepted['implementation_files'][name]['sha256']
    bank = {k: [] for k in ['features', 'images', 'state', 'actions', 'current', 'targets', 'motion', 'records']}
    current_seconds = 0.; teacher_seconds = 0.; repeat = 0.
    for n, selection in enumerate(selected):
        source = dataset.sources[selection['episode_index']]
        samples = [dataset[dataset.ordinal(source['episode_index'], t, h)]
                   for t in selection['anchors'] for h in HORIZONS]
        observations = [samples[i].observation for i in range(0, 20, 5)]
        start = time.monotonic()
        encoding = builder([x.instruction for x in observations], [x.current_rgb[0] for x in observations],
                           [x.current_rgb[1] for x in observations])
        encoding = {k: v.cuda() for k, v in encoding.items()}
        features, padding = encoder(encoding)
        current_seconds += time.monotonic()-start
        rgb = dataset._episode(source)[2]
        frames = sorted({s.identity['anchor'] for s in samples}|{s.identity['target'] for s in samples})
        positions = {frame: i for i, frame in enumerate(frames)}
        start = time.monotonic(); targets = torch.from_numpy(encode(teacher, rgb[frames]))
        teacher_seconds += time.monotonic()-start
        if n in (0, 2):
            repeated_features, _ = encoder(encoding)
            repeat = max(repeat, (repeated_features-features).abs().max().item(),
                         float(np.max(np.abs(encode(teacher, rgb[frames])-targets.numpy()))))
        for i, t in enumerate(selection['anchors']):
            sample = samples[i*5]
            bank['features'].append(features[i, ~padding[i]].cpu().clone())
            bank['images'].append(sample.observation.current_images)
            bank['state'].append(sample.observation.state)
            bank['actions'].append(torch.stack([s.observation.actions for s in samples[i*5:(i+1)*5]]))
            bank['current'].append(targets[positions[t]].clone())
            bank['targets'].append(torch.stack([targets[positions[t+h]] for h in HORIZONS]))
            bank['motion'].append(torch.stack([(s.observation.current_images-s.target_images).abs().mean((1, 2, 3))
                                               for s in samples[i*5:(i+1)*5]]))
            bank['records'].append({**sample.identity, 'role': selection['role'],
                'pair_identities': [s.identity for s in samples[i*5:(i+1)*5]],
                'current_rgb_sha256': hashlib.sha256(rgb[t].tobytes()).hexdigest(),
                'future_rgb_sha256': [hashlib.sha256(rgb[t+h].tobytes()).hexdigest() for h in HORIZONS]})
        if (n+1) % 10 == 0:
            progress(args.output, 'encode_train_only', episodes=n+1, total=len(selected))
    assert before == tensor_state_digest(encoder.state_dict())
    assert vision_before == tensor_state_digest(teacher.visual.state_dict()) and repeat == 0
    del full, encoder, teacher, processor, builder, encoding, features, repeated_features, targets, dataset, rgb, samples
    gc.collect(); torch.cuda.empty_cache()
    for key in ['images', 'state', 'actions', 'current', 'targets', 'motion']:
        bank[key] = torch.stack(bank[key])
    assert len(bank['records']) == 640
    atomic_torch_save(args.output/'train-only-bank.pt', bank)
    receipt = save(args.output/'bank-receipt.json', 'eventground_wam.jepa_followup_bank',
        contract_digest=contract['digest'], bank_sha256=sha256_file(args.output/'train-only-bank.pt'),
        records=bank['records'], source_sha256={str(ROOT/'configs/datasets/libero-plus-trajectory-v2/groups.jsonl'):
            contract['input_sha256']['configs/datasets/libero-plus-trajectory-v2/groups.jsonl']},
        base_unchanged=True, teacher_unchanged=True, feature_repeat_max_error=repeat,
        base_state_sha256=before, teacher_state_sha256=vision_before, verified_base_files=model_files,
        current_base_encoding_seconds=current_seconds, all_teacher_encoding_seconds=teacher_seconds,
        extra_residual_current_teacher_frames=640, extra_residual_current_teacher_camera_images=1280,
        validation_payload_reads=0, test_payload_reads=0, anchors=640, dual_camera_pairs=3200)
    return bank, receipt


def summaries(rows):
    result = []
    for role in ('fit', 'pilot_eval'):
        active = [r for r in rows if r['role'] == role]
        for camera, horizon, motion in [('both', None, 'all'), ('both', 'medium_long', 'all')]+[
                (c, h, 'all') for c in CAMERAS for h in HORIZONS]+[
                (c, h, m) for c in CAMERAS for h in HORIZONS for m in ('low', 'medium', 'high')]:
            chosen = [r for r in active if (camera == 'both' or r['camera'] == camera)
                and (horizon is None or r['horizon'] == horizon or (horizon == 'medium_long' and r['horizon'] >= 8))
                and (motion == 'all' or r['motion_bin'] == motion)]
            if not chosen:
                continue
            grouped = defaultdict(list)
            for r in chosen:
                grouped[(r['task_key'], r['group_digest'])].append(r)
            keys = sorted(grouped)
            point, boot = group_weights(keys, replicates=10000, seed=20261002)
            a = np.array([[np.mean([r[field] for r in grouped[k]]) for field in ('cosine', 'copy')]
                          for k in keys])
            gain = (a[:, 1]-a[:, 0])
            ratio = (boot@gain)/(boot@a[:, 1])
            result.append({'role': role, 'camera': camera, 'horizon': horizon, 'motion_bin': motion,
                'camera_examples': len(chosen), 'groups': len(keys), 'tasks': len({k[0] for k in keys}),
                'cosine': float(point@a[:, 0]), 'copy': float(point@a[:, 1]),
                'relative_gain_vs_copy': float(point@gain/(point@a[:, 1])),
                'paired_group_bootstrap_95': np.quantile(ratio, [.025, .975]).tolist()})
    return result


@torch.no_grad()
def evaluate(model, bank, indices, output, name, *, retain):
    model.eval(); rows = []; predicted_means = defaultdict(list); target_means = defaultdict(list)
    basis = None if isinstance(model, ResidualJepaPredictor) else head_span(model)
    prediction_file = output/f'{name}-predictions.npy'; token_file = output/f'{name}-tokens.npy'
    predictions = np.lib.format.open_memmap(prediction_file, mode='w+', dtype='float32',
                    shape=(len(indices)*5, 128, 2048)) if retain else None
    tokens = np.lib.format.open_memmap(token_file, mode='w+', dtype='float32',
                    shape=(len(indices)*5, 128, 256)) if retain else None
    for start in range(0, len(indices), 8):
        ids = indices[start:start+8]; condition, current, target, _ = assemble(bank, ids)
        value = predict(model, condition, current)
        error = errors(value.features, target).cpu(); copied = errors(current, target).cpu()
        change = errors(value.features, current).cpu()
        bounds = span_lower_bound(target, basis).reshape(-1, 2, 64).mean(-1).cpu() if basis is not None else None
        current_bounds = span_lower_bound(current, basis).reshape(-1, 2, 64).mean(-1).cpu() if basis is not None else None
        if retain:
            predictions[start*5:(start+len(ids))*5] = value.features.cpu().numpy()
            tokens[start*5:(start+len(ids))*5] = value.future_tokens.cpu().numpy()
        for i, anchor_id in enumerate(ids):
            record = bank['records'][anchor_id]; key = record['task_key'], record['group_digest']
            if retain:
                predicted_means[key].append(value.features[i*5:(i+1)*5].mean(0).cpu().numpy())
                target_means[key].append(target[i*5:(i+1)*5].mean(0).cpu().numpy())
            for hi, h in enumerate(HORIZONS):
                for ci, camera in enumerate(CAMERAS):
                    j = i*5+hi; motion = float(bank['motion'][anchor_id, hi, ci])
                    rows.append({'task_key': record['task_key'], 'group_digest': record['group_digest'],
                        'episode_index': record['episode_index'], 'anchor': record['anchor'],
                        'role': record['role'], 'horizon': h, 'camera': camera,
                        'cosine': float(error[j, ci]), 'copy': float(copied[j, ci]),
                        'predicted_change_from_current': float(change[j, ci]),
                        'rgb_change': motion, 'motion_bin': motion_bin(motion),
                        'head_span_optimistic_error_bound': None if bounds is None else float(bounds[j, ci]),
                        'current_head_span_optimistic_error_bound': None if current_bounds is None else float(current_bounds[j, ci])})
    result = {'rows': rows, 'head_span_rank': None if basis is None else basis.shape[1]}
    if retain:
        predictions.flush(); tokens.flush()
        groups = sorted(predicted_means)
        result['collapse'] = {}
        for label, means in [('prediction', predicted_means), ('target', target_means)]:
            values = np.stack([np.mean(means[k], 0) for k in groups])
            np.save(output/f'{name}-{label}-group-means.npy', values, allow_pickle=False)
            result['collapse'][label] = collapse_diagnostics(values, groups)
        result['collapse_groups'] = groups
    save(output/f'{name}-metrics.json', 'eventground_wam.jepa_followup_measurements', **result)
    return result


def gradient_diagnostics(model, bank, indices, loss_rule):
    model.eval(); condition, current, target, groups = assemble(bank, indices)
    predicted = predict(model, condition, current).features
    loss = jepa_prediction_loss(predicted, target, groups, **loss_rule)
    h2_front = (1-(predicted[::5, :64]*target[::5, :64]).sum(-1)).mean()
    params = list(model.named_parameters())
    gradients = [torch.autograd.grad(term, [p for _, p in params], retain_graph=True, allow_unused=True)
                 for term in (loss.cosine, loss_rule['variance_weight']*loss.variance, h2_front)]
    result = {'loss': float(loss.total.detach()), 'cosine': float(loss.cosine.detach()),
              'weighted_variance': float(loss_rule['variance_weight']*loss.variance.detach())}
    for group in ('all', 'core', 'head'):
        positions = [i for i, (n, _) in enumerate(params) if group == 'all'
                     or n.startswith('output_projection.') == (group == 'head')]
        norms = [sum(float(g[i].double().square().sum()) for i in positions if g[i] is not None)**.5
                 for g in gradients]
        dot = lambda a, b, positions=positions: sum(float((gradients[a][i].double()*gradients[b][i].double()).sum())
                              for i in positions if gradients[a][i] is not None and gradients[b][i] is not None)
        result[group] = {'cosine_gradient_norm': norms[0], 'weighted_variance_gradient_norm': norms[1],
            'front_h2_gradient_norm': norms[2],
            'variance_to_cosine_norm_ratio': norms[1]/norms[0] if norms[0] else None,
            'cosine_variance_alignment': dot(0, 1)/(norms[0]*norms[1]) if norms[0]*norms[1] > 1e-20 else None,
            'front_h2_variance_alignment': dot(2, 1)/(norms[2]*norms[1]) if norms[2]*norms[1] > 1e-20 else None}
    model.zero_grad(set_to_none=True)
    return result


def train_arm(model, bank, fit_ids, contract, output, name):
    initial_state = tensor_state_digest(model.state_dict()); model.train()
    plan = training_plan(len(fit_ids), contract['updates_per_arm'], contract['seed'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .999), eps=1e-8,
                                 weight_decay=1e-4, foreach=False, fused=False)
    started = time.monotonic(); chain = '0'*64; activity = defaultdict(float)
    for step, local_indices in enumerate(plan):
        indices = [fit_ids[i] for i in local_indices]
        assert all(bank['records'][i]['role'] == 'fit' for i in indices)
        condition, current, target, groups = assemble(bank, indices)
        optimizer.zero_grad(set_to_none=True)
        factor = rate_factor(step, len(plan), contract['warmup_updates'], .1)
        for group in optimizer.param_groups:
            group['lr'] = 1e-4*factor
        loss = backward_batch(model, condition, current, target, groups, contract['loss'])
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        assert torch.isfinite(norm) and torch.isfinite(loss.total)
        if step in (0, 1, len(plan)-1):
            for key in ('action_projection', 'action_sequence_projection', 'horizon_projection', 'current_patches', 'output_projection'):
                activity[key] = max(activity[key], sum(float(p.grad.abs().sum()) for p in getattr(model, key).parameters()))
        optimizer.step()
        chain = hashlib.sha256((chain+json.dumps(indices, separators=(',', ':'))).encode()).hexdigest()
        if (step+1) % 100 == 0:
            row = {'arm': name, 'step': step+1, 'cosine': float(loss.cosine.detach()),
                   'variance': float(loss.variance.detach()), 'total': float(loss.total.detach()),
                   'elapsed_seconds': time.monotonic()-started, 'sample_chain': chain}
            with (output/f'{name}-trace.jsonl').open('a') as handle:
                handle.write(json.dumps(row)+'\n')
            progress(output, 'train', **row)
        if time.monotonic()-started > contract['limits']['per_arm_seconds']:
            raise RuntimeError('sealed per-arm time cap exceeded; no endpoint extension')
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    header = seal_contract({'artifact_type': 'eventground_wam.jepa_followup_diagnostic_checkpoint',
        'arm': name, 'config': asdict(model.config), 'contract_digest': contract['digest'],
        'state_sha256': tensor_state_digest(state), 'updates': len(plan), 'sample_chain': chain,
        'initial_state_sha256': initial_state, 'purpose': 'training_only_pilot_not_accepted_not_production'})
    atomic_torch_save(output/f'{name}.pt', {'header': header, 'state_dict': state})
    return {'updates': len(plan), 'pair_presentations': len(plan)*40, 'unique_fit_pairs': len(fit_ids)*5,
        'sample_chain': chain, 'initial_state_sha256': initial_state, 'final_state_sha256': header['state_sha256'],
        'checkpoint_sha256': sha256_file(output/f'{name}.pt'), 'gradient_activity': dict(activity),
        'elapsed_seconds': time.monotonic()-started}


@torch.no_grad()
def ablations(model, bank, ids, output, name):
    normal = np.load(output/f'{name}-eval-predictions.npy', mmap_mode='r')
    tokens = np.load(output/f'{name}-eval-tokens.npy', mmap_mode='r')
    tasks = defaultdict(list)
    for j, i in enumerate(ids):
        tasks[bank['records'][i]['task_key']].append(j)
    donors = {}
    for positions in tasks.values():
        assert len(positions) == 8
        donors.update({p: positions[(j+4) % 8] for j, p in enumerate(positions)})
    rows = []
    for kind in ('zero-actions', 'reverse-actions', 'zero-horizon', 'swap-tokens'):
        for start in range(0, len(ids), 8):
            chosen = ids[start:start+8]; condition, current, target, _ = assemble(bank, chosen)
            original_tokens = torch.from_numpy(np.array(tokens[start*5:(start+8)*5])).cuda()
            if kind == 'swap-tokens':
                replaced = torch.from_numpy(np.concatenate([tokens[donors[i]*5:(donors[i]+1)*5]
                                                            for i in range(start, start+8)])).cuda()
                features = from_tokens(model, replaced, current); altered_tokens = replaced
            else:
                value = predict(model, condition, current, ablation=kind)
                features, altered_tokens = value.features, value.future_tokens
            changed = errors(features, target).cpu().numpy()
            base = torch.from_numpy(np.array(normal[start*5:(start+8)*5])).cuda()
            original_error = errors(base, target).cpu().numpy()
            feature_change = (features-base).abs().mean((1, 2)).cpu().numpy()
            token_change = (altered_tokens-original_tokens).abs().mean((1, 2)).cpu().numpy()
            for j, i in enumerate(chosen):
                r = bank['records'][i]
                for hi, h in enumerate(HORIZONS):
                    for ci, camera in enumerate(CAMERAS):
                        k = j*5+hi
                        rows.append({'kind': kind, 'task_key': r['task_key'], 'group_digest': r['group_digest'],
                            'episode_index': r['episode_index'], 'anchor': r['anchor'], 'horizon': h, 'camera': camera,
                            'cosine': float(changed[k, ci]), 'normal_cosine': float(original_error[k, ci]),
                            'mean_absolute_feature_change': float(feature_change[k]),
                            'mean_absolute_token_change': float(token_change[k])})
    save(output/f'{name}-ablations.json', 'eventground_wam.jepa_followup_ablations', rows=rows)
    return {kind: {'medium_long_cosine_increase': float(np.mean([r['cosine']-r['normal_cosine'] for r in rows if r['kind'] == kind and r['horizon'] >= 8])),
                   'mean_absolute_feature_change': float(np.mean([r['mean_absolute_feature_change'] for r in rows if r['kind'] == kind])),
                   'mean_absolute_token_change': float(np.mean([r['mean_absolute_token_change'] for r in rows if r['kind'] == kind]))}
            for kind in ('zero-actions', 'reverse-actions', 'zero-horizon', 'swap-tokens')}


def run(args):
    started = time.monotonic(); runtime = configure()
    contract = read_sealed(args.contract)
    assert runtime['torch'] == '2.13.0+cu130' and runtime['transformers'] == '5.17.0'
    for name, digest in contract['input_sha256'].items():
        assert sha256_file(ROOT/name) == digest, name
    assert sha256_file(args.candidate) == contract['reference_candidate_sha256']
    args.output.mkdir(parents=True, exist_ok=False)
    identity = save(args.output/'identity.json', 'eventground_wam.jepa_followup_identity',
                    contract_digest=contract['digest'], runtime=runtime, input_sha256=contract['input_sha256'])
    progress(args.output, 'prepare')
    bank, receipt = prepare(args, contract)
    fit_ids = [i for i, r in enumerate(bank['records']) if r['role'] == 'fit']
    eval_ids = [i for i, r in enumerate(bank['records']) if r['role'] == 'pilot_eval']
    assert len(fit_ids) == len(eval_ids) == 320
    diagnostic_ids = [fit_ids[i] for i in (0, 8, 16, 24, 32, 40, 48, 56)]
    outcomes = {}
    for name in ('production_v1_reference', 'absolute_control', 'copy_residual'):
        progress(args.output, 'start_arm', arm=name)
        if name == 'production_v1_reference':
            model, _ = load_jepa_checkpoint(args.candidate)
        else:
            model = ResidualJepaPredictor() if name == 'copy_residual' else JepaPredictor()
            other = JepaPredictor()
            assert all(torch.equal(v, other.state_dict()[k]) for k, v in model.state_dict().items()
                       if not k.startswith('output_projection.'))
            del other
        model = model.cuda()
        initial = gradient_diagnostics(model, bank, diagnostic_ids, contract['loss'])
        if name == 'copy_residual':
            with torch.no_grad():
                x, cur, _, _ = assemble(bank, diagnostic_ids)
                assert (model(x, cur).features-cur).abs().max() <= 1e-6
        training = None if name == 'production_v1_reference' else train_arm(model, bank, fit_ids, contract, args.output, name)
        after = gradient_diagnostics(model, bank, diagnostic_ids, contract['loss'])
        fit = evaluate(model, bank, fit_ids, args.output, name+'-fit', retain=False)
        evaluation = evaluate(model, bank, eval_ids, args.output, name+'-eval', retain=True)
        interventions = None if name == 'production_v1_reference' else ablations(model, bank, eval_ids, args.output, name)
        outcomes[name] = {'training': training, 'initial_gradients': initial, 'final_gradients': after,
                         'summaries': summaries(fit['rows']+evaluation['rows']),
                         'collapse': evaluation['collapse'], 'ablations': interventions,
                         'head_span_rank': evaluation['head_span_rank']}
        save(args.output/f'{name}-report.json', 'eventground_wam.jepa_followup_arm', **outcomes[name])
        del model, fit, evaluation
        gc.collect(); torch.cuda.empty_cache()
    assert outcomes['absolute_control']['training']['sample_chain'] == outcomes['copy_residual']['training']['sample_chain']
    assert sha256_file(args.candidate) == contract['reference_candidate_sha256']
    report = save(args.output/'report.json', 'eventground_wam.jepa_followup_report',
        status='complete', purpose='training_only_diagnostic_no_checkpoint_acceptance',
        contract_digest=contract['digest'], identity_digest=identity['digest'], bank_receipt_digest=receipt['digest'],
        arms=outcomes, elapsed_seconds=time.monotonic()-started,
        gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(), host_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        optimizer_updates=2*contract['updates_per_arm'], validation_payload_reads=0, test_payload_reads=0,
        original_candidate_unchanged=True, independent_audit='pending', pilot_decision='pending_independent_audit',
        accepted_checkpoint_created=False)
    files = {p.name: sha256_file(p) for p in args.output.iterdir() if p.is_file() and p.suffix in ('.json', '.npy', '.pt', '.jsonl') and p.name != 'progress.json'}
    save(args.output/'completion.json', 'eventground_wam.jepa_followup_completion', report_digest=report['digest'], files_sha256=files)
    progress(args.output, 'complete', report_digest=report['digest'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--pairs', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=Path('/home/peimengsui/data/eventground-wam/libero-plus'))
    p.add_argument('--cache', type=Path, default=Path('/home/peimengsui/data/eventground-wam/huggingface-cache'))
    run(p.parse_args())
