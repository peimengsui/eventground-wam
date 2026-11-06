#!/usr/bin/env python3
"""Render the fixed review panel and run sealed, matched appearance diagnostics."""
from __future__ import annotations

import argparse
import json
import resource
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from run_jepa_baselines import atomic_array, encode
from train_jepa_cpt import configure

from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_model import load_pinned_qwen_model, load_pinned_qwen_processor
from eventground_wam.b0_training import atomic_json
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.future_pairs import CAMERAS, read_sealed, sha256_file
from eventground_wam.jepa_features import FrozenQwenVisualTargets
from eventground_wam.jepa_predictor import load_jepa_checkpoint
from eventground_wam.pixel_baselines import box_rgb
from eventground_wam.pixel_data import FutureInputBuilder, FuturePairDataset
from eventground_wam.pixel_predictor import (
    FrozenQwenFeatures,
    load_pixel_checkpoint,
    tensor_state_digest,
)
from eventground_wam.pixel_training import encode_training_batch

TRANSFORMS = [{'name': 'identity', 'kind': 'identity'},
              {'name': 'brightness_0.7', 'kind': 'brightness', 'factor': .7},
              {'name': 'brightness_1.3', 'kind': 'brightness', 'factor': 1.3},
              {'name': 'contrast_0.7', 'kind': 'contrast', 'factor': .7},
              {'name': 'blur_1.5', 'kind': 'gaussian_blur', 'radius': 1.5}]


def transform_rgb(rgb, spec):
    if rgb.dtype != np.uint8 or rgb.shape[-3:] != (256, 256, 3):
        raise ValueError('appearance input must be native uint8 RGB')
    kind = spec['kind']
    if kind == 'identity':
        return rgb.copy()
    if kind == 'brightness':
        return np.rint(np.clip(rgb.astype(np.float64)*spec['factor'], 0, 255)).astype(np.uint8)
    if kind == 'contrast':
        return np.rint(np.clip((rgb.astype(np.float64)-127.5)*spec['factor']+127.5, 0, 255)).astype(np.uint8)
    if kind == 'gaussian_blur':
        images = [np.asarray(Image.fromarray(x).filter(ImageFilter.GaussianBlur(spec['radius'])))
                  for x in rgb.reshape(-1, 256, 256, 3)]
        return np.stack(images).reshape(rgb.shape)
    raise ValueError('unknown sealed appearance transform')


def pixel_to_native(images):
    """Explicit decoder-output bridge; never compare pixel L1 directly to JEPA loss."""
    if images.shape[1:] != (2, 3, 128, 128) or not np.isfinite(images).all():
        raise ValueError('invalid decoded pixel predictions')
    values = np.rint(np.clip(images.transpose(0, 1, 3, 4, 2), 0, 1)*255).astype(np.uint8)
    return np.stack([np.asarray(Image.fromarray(x).resize((256, 256), Image.Resampling.BILINEAR))
                     for x in values.reshape(-1, 128, 128, 3)]).reshape(-1, 2, 256, 256, 3)


def errors(a, b):
    if a.shape != b.shape or a.ndim != 3 or a.shape[1] != 128:
        raise ValueError('unaligned feature fields')
    return np.clip(1-(a.astype(np.float64)*b.astype(np.float64)).sum(-1), 0, 2).reshape(-1, 2, 8, 8)


def token_units(value):
    x = value.cpu().numpy().astype(np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    assert (norm > 1e-12).all()
    return x/norm


def summarize(rows):
    """Paired task bootstrap: only one selected episode/group per task in this panel."""
    fields = ['target_cosine', 'common_feature_drift', 'native_token_drift', 'target_teacher_drift', 'excess_target_error']
    output = []
    for name in [t['name'] for t in TRANSFORMS]:
        for method in ('copy', 'pixel', 'jepa'):
            for camera in ('both', *CAMERAS):
                selected = [r for r in rows if r['transform'] == name and r['method'] == method and (camera == 'both' or r['camera'] == camera)]
                tasks = sorted({r['task_key'] for r in selected})
                values = np.array([[np.mean([r[f] for r in selected if r['task_key'] == task]) for f in fields] for task in tasks])
                rng = np.random.default_rng(20260928)
                boot = values[rng.integers(len(tasks), size=(10000, len(tasks)))].mean(1)
                output.append({'transform': name, 'method': method, 'camera': camera, 'examples': len(selected),
                               'tasks': len(tasks), 'metrics': {f: {'value': float(values[:, j].mean()),
                               'task_bootstrap_95': np.quantile(boot[:, j], [.025, .975]).tolist()} for j, f in enumerate(fields)}})
    differences = []
    for name in [t['name'] for t in TRANSFORMS]:
        paired = {}
        for r in rows:
            if r['transform'] == name and r['method'] in ('pixel', 'jepa'):
                key = r['task_key'], r['camera'], r['anchor'], r['horizon']
                paired.setdefault(key, {})[r['method']] = r
        tasks = sorted({key[0] for key in paired})
        values = np.array([[np.mean([v['jepa'][f]-v['pixel'][f] for k, v in paired.items() if k[0] == task])
                            for f in fields] for task in tasks])
        rng = np.random.default_rng(20260928)
        boot = values[rng.integers(len(tasks), size=(10000, len(tasks)))].mean(1)
        differences.append({'transform': name, 'jepa_minus_pixel': {f: {'value': float(values[:, j].mean()),
                            'paired_task_bootstrap_95': np.quantile(boot[:, j], [.025, .975]).tolist()} for j, f in enumerate(fields)}})
    return {'by_transform_method_camera': output, 'paired_differences': differences}


def heat(value):
    """Fixed error scale 0..0.5: navy -> cyan -> yellow -> red."""
    stops = np.array([[20, 30, 80], [0, 155, 190], [250, 225, 90], [220, 35, 35]])
    x = np.clip(value/.5, 0, 1)*3
    lo = np.minimum(x.astype(int), 2); alpha = x-lo
    rgb = np.rint(stops[lo]*(1-alpha[..., None])+stops[lo+1]*alpha[..., None]).astype(np.uint8)
    return Image.fromarray(rgb).resize((116, 116), Image.Resampling.NEAREST)


def render(path, source, examples, rgb, pixels, targets, copied, predicted, pixel_features, keys, *, supplemental=False):
    font = ImageFont.truetype('DejaVuSans.ttf', 12)
    width, height = 920, 114+158*len(examples)
    canvas = Image.new('RGB', (width, height), '#f8fafc'); draw = ImageDraw.Draw(canvas)
    camera = examples[0]['camera'].split('.')[-1]
    draw.text((10, 8), f"{source['task_key']} | ep {source['episode_index']} | {camera} | {'SUPPLEMENTAL h=2' if supplemental else 'FIXED REVIEW'}", fill='#172b46', font=font)
    draw.text((10, 27), source['instruction'][:130], fill='#172b46', font=font)
    draw.text((10, 46), 'Heatmaps: cosine error in common teacher space. Fixed 0..0.50 scale: navy=low, red=high.', fill='#172b46', font=font)
    draw.text((10, 64), 'JEPA has no RGB decoder. Pixel RGB is its actual decoded output; error uses resize+teacher bridge.', fill='#172b46', font=font)
    labels = ['Current RGB', 'Future RGB', 'Pixel RGB', 'Copy error', 'JEPA error', 'Pixel error']
    for j, label in enumerate(labels):
        draw.text((10+j*151, 89), label, fill='#172b46', font=font)
    for n, ex in enumerate(examples):
        i = keys.index((ex['anchor'], ex['horizon'])); c = CAMERAS.index(ex['camera']); y = 111+n*158
        ce, je, pe = [errors(v[i:i+1], targets[i:i+1])[0, c] for v in (copied, predicted, pixel_features)]
        draw.text((10, y), f"h={ex['horizon']} t={ex['anchor']}->{ex['target']} | copy={ce.mean():.3f}, JEPA={je.mean():.3f}, Pixel={pe.mean():.3f}", fill='#172b46', font=font)
        values = [Image.fromarray(rgb[ex['anchor'], c]), Image.fromarray(rgb[ex['target'], c]),
                  Image.fromarray(np.rint(np.clip(pixels[i, c].transpose(1, 2, 0), 0, 1)*255).astype(np.uint8)), heat(ce), heat(je), heat(pe)]
        for j, value in enumerate(values):
            canvas.paste(value.resize((116, 116), Image.Resampling.NEAREST), (10+j*151, y+20))
    canvas.save(path)


@torch.inference_mode()
def run(args):
    started = time.monotonic(); runtime = configure()
    contract = read_sealed(args.contract)
    assert contract['transforms'] == TRANSFORMS and contract['script_sha256'] == sha256_file(Path(__file__))
    assert sha256_file(args.jepa) == contract['jepa_sha256'] and sha256_file(args.pixel) == contract['pixel_sha256']
    report = read_sealed(args.validation/'report.json'); identity = read_sealed(args.validation/'identity.json')
    assert report['digest'] == contract['validation_report_digest'] and runtime == identity['runtime']
    for name, digest in identity['source_sha256'].items():
        assert sha256_file(args.source/name) == digest, name
    examples = read_sealed(args.source/'configs/evaluation/week5_pixel_qualitative_v2.json')
    assert examples['digest'] == contract['qualitative_panel_digest']
    for directory in ('episodes', 'panels', 'h2-panels', 'arrays'):
        (args.output/directory).mkdir(parents=True, exist_ok=True)
    if (args.output/'report.json').exists():
        raise FileExistsError('completed diagnostics are immutable')
    if (args.output/'contract.json').exists():
        assert read_sealed(args.output/'contract.json') == contract
    else:
        atomic_json(args.output/'contract.json', contract)
    full = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True).to('cuda')
    encoder = FrozenQwenFeatures(full.model).to('cuda'); processor = load_pinned_qwen_processor(cache_dir=str(args.cache), local_files_only=True)
    teacher = FrozenQwenVisualTargets(full.model.visual, processor.image_processor).eval(); builder = FutureInputBuilder(processor)
    jepa, _ = load_jepa_checkpoint(args.jepa); pixel, _ = load_pixel_checkpoint(args.pixel)
    models = [m.eval().requires_grad_(False).to('cuda') for m in (jepa, pixel)]
    before = [tensor_state_digest(m.state_dict()) for m in (*models, encoder)]
    root = args.source
    protocol = read_sealed(root/'configs/evaluation/week6_jepa_baselines_v1.json')
    dataset = FuturePairDataset(args.pairs, args.data_root/f'datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo',
                               split='validation', expected_manifest_sha256=protocol['pair_manifest_sha256'],
                               normalizer=load_b0_action_normalizer(root/'docs/model/week4-fresh-v2/action-statistics.json'),
                               state_bounds=read_sealed(root/'configs/training/week4_b0_fresh_all_l1_v2.json')['state_normalization'])
    byep = defaultdict(list)
    for ex in examples['examples']:
        byep[ex['episode_index']].append(ex)
    allrows, receipts = [], []
    for ep, selected in sorted(byep.items(), key=lambda item: item[1][0]['task_key']):
        path = args.output/f'episodes/{ep:06d}.json'
        if path.exists():
            saved = read_sealed(path); assert saved['contract_digest'] == contract['digest']
            for name, digest in saved['files_sha256'].items():
                assert sha256_file(args.output/name) == digest
            allrows.extend(saved['rows']); receipts.append(saved); continue
        source = dataset.sources[ep]; rgb = dataset._episode(source)[2]
        fixed_keys = sorted({(e['anchor'], e['horizon']) for e in selected})
        supplements = [{**selected[0], 'camera': camera, 'horizon': 2, 'anchor_fraction': f,
                        'anchor': int(f*(len(rgb)-3)), 'target': int(f*(len(rgb)-3))+2}
                       for f in (.25, .5, .75) for camera in CAMERAS]
        keys = sorted(set(fixed_keys)|{(e['anchor'], 2) for e in supplements})
        samples = [dataset[dataset.ordinal(ep, t, h)] for t, h in keys]
        frames = sorted({t for t, h in keys}|{t+h for t, h in keys}); positions = {f: i for i, f in enumerate(frames)}
        baseline = None; rows = []; files = {}; replay_error = 0.; clean_comparisons = []
        original = read_sealed(args.validation/f'episodes/{ep:06d}.json')
        bank = np.load(args.validation/f'qualitative/{ep:06d}-predictions.npy')
        assert sha256_file(args.validation/f'qualitative/{ep:06d}-predictions.npy') == original['arrays'][f'qualitative/{ep:06d}-predictions.npy']
        saved_predictions = dict(zip(map(tuple, original['qualitative_keys']), bank, strict=True))
        for spec in TRANSFORMS:
            native = transform_rgb(rgb[frames], spec)
            target_bank = encode(teacher, native)
            active_keys = keys if spec['kind'] == 'identity' else fixed_keys
            active_samples = []
            for key in active_keys:
                sample = samples[keys.index(key)]
                current = native[positions[key[0]]]
                observation = replace(sample.observation, current_rgb=current,
                                      current_images=torch.from_numpy(np.stack([box_rgb(x) for x in current])))
                # Targets never enter either predictor; identity and action/state fields stay fixed.
                active_samples.append(replace(sample, observation=observation))
            condition, _, _ = encode_training_batch(active_samples, encoder, builder, torch.device('cuda'), feature_microbatch=4)
            jp, pp = models[0](condition), models[1](condition)
            jf = jp.features.cpu().numpy(); pi = pp.images.cpu().numpy(); pf = encode(teacher, pixel_to_native(pi))
            jt, pt = token_units(jp.future_tokens), token_units(pp.future_tokens)
            targets = np.stack([target_bank[positions[t+h]] for t, h in active_keys])
            copied = np.stack([target_bank[positions[t]] for t, h in active_keys])
            if spec['kind'] == 'identity':
                baseline = {'jepa': jf.copy(), 'pixel': pf.copy(), 'copy': copied.copy(),
                            'targets': targets.copy(), 'jt': jt.copy(), 'pt': pt.copy()}
                replay_error = max(float(abs(jf[keys.index(k)]-saved_predictions[k]).max()) for k in fixed_keys)
                assert replay_error < 1e-5
                for n, key in enumerate(keys):
                    for c, camera in enumerate(CAMERAS):
                        clean_comparisons.append({'camera': camera, 'anchor': key[0], 'horizon': key[1],
                                                  'fixed_example': key in fixed_keys,
                                                  **{method: float(errors(value[n:n+1], targets[n:n+1])[0, c].mean())
                                                     for method, value in [('copy', copied), ('jepa', jf), ('pixel', pf)]}})
                for camera in CAMERAS:
                    for prefix, exs, supplemental in [('panels', [e for e in selected if e['camera'] == camera], False),
                                                     ('h2-panels', [e for e in supplements if e['camera'] == camera], True)]:
                        name = f'{prefix}/{ep:06d}-{camera.split(".")[-1]}.png'
                        render(args.output/name, source, exs, rgb, pi, targets, copied, jf, pf, keys, supplemental=supplemental)
                        files[name] = sha256_file(args.output/name)
            names = {'jepa': (jf, jt), 'pixel': (pf, pt), 'copy': (copied, copied)}
            for method, (field, tokens) in names.items():
                references = baseline['jt' if method == 'jepa' else 'pt'] if method != 'copy' else baseline['copy']
                idx = [keys.index(k) for k in active_keys]
                loss = errors(field, targets).mean((2, 3)); drift = errors(field, baseline[method][idx]).mean((2, 3))
                native_drift = errors(tokens, references[idx]).mean((2, 3)); teacher_drift = errors(targets, baseline['targets'][idx]).mean((2, 3))
                clean_loss = errors(baseline[method][idx], baseline['targets'][idx]).mean((2, 3))
                for n, key in enumerate(active_keys):
                    if key not in fixed_keys:
                        continue
                    for c, camera in enumerate(CAMERAS):
                        rows.append({'episode_index': ep, 'source_digest': source['digest'], 'task_key': source['task_key'],
                                     'group_digest': source['group_digest'], 'camera': camera, 'anchor': key[0], 'horizon': key[1],
                                     'transform': spec['name'], 'method': method, 'target_cosine': float(loss[n, c]),
                                     'common_feature_drift': float(drift[n, c]), 'native_token_drift': float(native_drift[n, c]),
                                     'target_teacher_drift': float(teacher_drift[n, c]), 'excess_target_error': float(loss[n, c]-clean_loss[n, c])})
            # Retain exact fields on a fixed one-task pilot panel for independent arithmetic replay.
            if source['task_key'] == contract['arithmetic_replay_task']:
                for name, value in {'jepa': jf, 'pixel': pf, 'copy': copied, 'targets': targets, 'jepa_tokens': jt,
                                    'pixel_tokens': pt, 'keys': np.asarray(active_keys)}.items():
                    rel = f'arrays/{ep:06d}-{spec["name"]}-{name}.npy'; files[rel] = atomic_array(args.output/rel, value)
            del condition, jp, pp, jf, pi, pf, jt, pt, target_bank
        saved = seal_contract({'artifact_type': 'eventground_wam.jepa_review_episode', 'contract_digest': contract['digest'],
                               'episode_index': ep, 'task_key': source['task_key'], 'source_digest': source['digest'],
                               'fixed_examples': selected, 'supplemental_h2_examples': supplements, 'rows': rows,
                               'clean_comparisons': clean_comparisons,
                               'clean_prediction_replay_max_error': replay_error, 'files_sha256': files})
        atomic_json(path, saved); receipts.append(saved); allrows.extend(rows)
        progress = {'completed_tasks': len(receipts), 'total_tasks': 40, 'elapsed_seconds': time.monotonic()-started}
        atomic_json(args.output/'progress.json', progress); print(json.dumps(progress), flush=True)
        assert time.monotonic()-started < 3600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 < 16*1024**3
    assert len(receipts) == 40 and len(allrows) == 480*len(TRANSFORMS)*3
    assert [tensor_state_digest(m.state_dict()) for m in (*models, encoder)] == before
    assert sha256_file(args.jepa) == contract['jepa_sha256'] and sha256_file(args.pixel) == contract['pixel_sha256']
    report = seal_contract({'artifact_type': 'eventground_wam.jepa_appearance_diagnostics', 'status': 'complete',
                            'contract_digest': contract['digest'], 'validation_report_digest': report['digest'],
                            **summarize(allrows), 'fixed_camera_examples': 480, 'fixed_dual_camera_pairs': 240,
                            'supplemental_h2_camera_examples': 240, 'tasks': 40,
                            'clean_prediction_replay_max_error': max(r['clean_prediction_replay_max_error'] for r in receipts),
                            'models_unchanged': True, 'optimizer_updates': 0, 'test_payload_reads': 0,
                            'elapsed_seconds': time.monotonic()-started, 'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                            'host_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                            'visual_review': 'pending_actual_inspection', 'acceptance_gate_changed': False,
                            'interpretation_limits': contract['interpretation_limits'],
                            'episode_receipts': [{'episode_index': r['episode_index'], 'digest': r['digest']} for r in receipts]})
    atomic_json(args.output/'measurements.json', seal_contract({'rows': allrows}))
    atomic_json(args.output/'report.json', report)
    print(json.dumps({'status': 'complete', 'report_digest': report['digest']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'validation', 'pairs', 'jepa', 'pixel', 'contract', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path.home()/'data/eventground-wam/libero-plus')
    parser.add_argument('--cache', type=Path, default=Path.home()/'data/eventground-wam/huggingface-cache')
    run(parser.parse_args())
