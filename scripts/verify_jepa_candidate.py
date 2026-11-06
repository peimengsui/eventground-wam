#!/usr/bin/env python3
"""Independent arithmetic, provenance, saved-array and fresh-inference candidate audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from run_jepa_baselines import decode, encode
from train_jepa_cpt import configure
from verify_jepa_baselines import FIELDS, draws, scalars, tie_scores

from eventground_wam.b0_data import load_b0_action_normalizer
from eventground_wam.b0_model import load_pinned_qwen_model, load_pinned_qwen_processor
from eventground_wam.demonstration_contracts import SOURCE_DATASET_REVISION, seal_contract
from eventground_wam.future_pairs import CAMERAS, HORIZONS, read_sealed, sha256_file, write_json
from eventground_wam.jepa_features import FrozenQwenVisualTargets
from eventground_wam.jepa_predictor import load_jepa_checkpoint
from eventground_wam.pixel_data import FutureInputBuilder, FuturePairDataset
from eventground_wam.pixel_predictor import FrozenQwenFeatures, tensor_state_digest
from eventground_wam.pixel_training import encode_training_batch, slice_condition


def grouped(rows, methods):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r['task_key'], r['group_digest'], r['camera'], r['horizon']].append(r)
    groups = sorted({(r['task_key'], r['group_digest']) for r in rows})
    array = np.empty((len(groups), len(methods), 2, 5, 5), np.float64)
    for g, (task, group) in enumerate(groups):
        for c, camera in enumerate(CAMERAS):
            for h, horizon in enumerate(HORIZONS):
                for m, method in enumerate(methods):
                    array[g, m, c, h] = np.mean([[r[method][f] for f in FIELDS]
                                               for r in buckets[task, group, camera, horizon]], axis=0)
    return groups, array


def verify_aggregates(baseline_rows, rows, baseline_queries, queries, report):
    """Explicit group means and resampled arrays, independent of production reducers."""
    def key(r):
        return r['episode_index'], r['camera'], r['horizon']
    originals = {key(r): r for r in baseline_rows}
    assert len(originals) == len(rows) == len({key(r) for r in rows})
    joined = []
    for r in rows:
        b = originals[key(r)]
        assert all(r[k] == b[k] for k in ('source_digest', 'group_digest', 'task_key', 'pairs'))
        assert all(abs(r['prediction'][f]-b['copy'][f]) < 1e-10 for f in FIELDS[3:])
        joined.append({**b, 'prediction': r['prediction']})
    groups, array = grouped(joined, ('copy', 'mean', 'prediction'))
    point, boot = draws(groups, array)
    errors = []

    def summary(saved, p, b):
        pv, bv = scalars(p), scalars(b)
        for f, value in pv.items():
            errors.append(abs(float(value)-saved[f]['value']))
            errors.extend(abs(np.quantile(bv[f], [.025, .975])-saved[f]['group_bootstrap_95']))

    summary(report['overall'], point[2].mean((0, 1)), boot[:, 2].mean((1, 2)))
    summary(report['medium_long'], point[2, :, 2:].mean((0, 1)), boot[:, 2, :, 2:].mean((1, 2)))
    cells = []
    for c, camera in enumerate(CAMERAS):
        for h, horizon in enumerate(HORIZONS):
            saved = next(r for r in report['by_camera_horizon'] if (r['camera'], r['horizon']) == (camera, horizon))
            summary(saved['prediction'], point[2, c, h], boot[:, 2, c, h])
            gain = 1-point[2, c, h, 0]/point[0, c, h, 0]
            ci = np.quantile(1-boot[:, 2, c, h, 0]/boot[:, 0, c, h, 0], [.025, .975])
            cells.append({'camera': camera, 'horizon': horizon, 'copy': float(point[0, c, h, 0]),
                          'prediction': float(point[2, c, h, 0]), 'relative_gain': float(gain),
                          'paired_95': ci.tolist(), 'passes': bool(gain >= -.02)})
    micro = np.average([[r['prediction'][f] for f in FIELDS] for r in rows], axis=0,
                       weights=[r['pairs'] for r in rows])
    for f, value in scalars(micro).items():
        errors.append(abs(float(value)-report['micro'][f]))
    effects = {}
    for name, base, motion in [('cosine_vs_copy', 0, False), ('cosine_vs_mean', 1, False), ('motion_vs_copy', 0, True)]:
        p, b = (point[:, :, 2:], boot[:, :, :, 2:]) if motion else (point, boot)
        p, b = p.mean((1, 2)), b.mean((2, 3))
        field = 1 if motion else 0
        value = 1-p[2, field]/p[base, field]
        samples = 1-b[:, 2, field]/b[:, base, field]
        effects[name] = {'value': float(value), 'paired_95': np.quantile(samples, [.025, .975]).tolist()}

    def qkey(q):
        return (*key(q), q['anchor'])
    baseq = {qkey(q): q for q in baseline_queries}
    assert len(baseq) == len(queries) == len({qkey(q) for q in queries})
    recomputed = []
    rank_error = 0.
    for q in queries:
        b = baseq[qkey(q)]
        assert all(q[k] == b[k] for k in ('source_digest', 'group_digest', 'task_key', 'target', 'pool'))
        assert q['target'] == q['anchor']+q['horizon']
        if q['pool']['eligible']:
            pool = q['pool']; assert len(set(pool['frames'])) == 8 and pool['frames'][pool['target_slot']] == q['target']
            scores = tie_scores(q['prediction']['scores'], pool['target_slot'])
            rank_error = max(rank_error, max(abs(scores[f]-q['prediction'][f]) for f in scores))
            recomputed.append({**q, 'prediction': scores, 'copy': b['copy']})

    def retrieval(selected, saved):
        bucket = defaultdict(list)
        for q in selected:
            bucket[q['task_key'], q['group_digest']].append([[q[m][f] for f in ('recall1', 'recall3', 'mrr')]
                                                           for m in ('copy', 'prediction')])
        order = sorted(bucket)
        p, b = draws(order, np.array([np.mean(bucket[g], axis=0) for g in order]))
        assert saved['groups'] == len(order) and saved['tasks'] == len({g[0] for g in order})
        assert saved['eligible_queries'] == len(selected)
        for j, f in enumerate(('recall1', 'recall3', 'mrr')):
            errors.append(abs(p[1, j]-saved['metrics'][f]['value']))
            errors.extend(abs(np.quantile(b[:, 1, j], [.025, .975])-saved['metrics'][f]['group_bootstrap_95']))
        return {'value': float(p[1, 0]-p[0, 0]), 'paired_95': np.quantile(b[:, 1, 0]-b[:, 0, 0], [.025, .975]).tolist()}

    effects['retrieval_recall1'] = retrieval(recomputed, report['retrieval'])
    for cell in report['retrieval_by_camera_horizon']:
        retrieval([q for q in recomputed if (q['camera'], q['horizon']) == (cell['camera'], cell['horizon'])], cell)
    for name, effect in effects.items():
        saved = report['gate_with_review_pending']['effects'][name]
        errors.extend([abs(effect['value']-saved['value']), *abs(np.array(effect['paired_95'])-saved['paired_95'])])
    assert max(errors) < 1e-10 and rank_error < 1e-12
    task_cells = []
    for task in sorted({g[0] for g in groups}):
        value = array[[i for i, g in enumerate(groups) if g[0] == task]].mean(0)[:, 0, 0, 0]
        task_cells.append({'task': task, 'copy': float(value[0]), 'prediction': float(value[2]),
                           'relative_regression': float(value[2]/value[0]-1)})
    return {'primary_bootstrap_max_error': float(max(errors)), 'retrieval_rank_max_error': rank_error,
            'effects': effects, 'camera_horizon_cells': cells, 'front_h2_by_task': task_cells,
            'motion_groups': int((array[:, 0, :, 2:, 3].mean((1, 2)) > 0).sum()), 'groups': groups}


def independent_collapse(values, groups):
    tasks = sorted({g[0] for g in groups})
    w = np.array([1/(len(tasks)*sum(x[0] == g[0] for x in groups)) for g in groups])
    result = {}
    for c, camera in enumerate(CAMERAS):
        x = values[:, c*64:(c+1)*64].astype(np.float64)
        mean = sum(w[i]*x[i] for i in range(len(x)))
        variance = sum(w[i]*np.mean((x[i]-mean)**2) for i in range(len(x)))
        pooled = x.mean(1); center = sum(w[i]*pooled[i] for i in range(len(x)))
        energy = torch.linalg.svdvals(torch.from_numpy((pooled-center)*np.sqrt(w[:, None]))).numpy()**2
        probabilities = energy/energy.sum(); active = probabilities[probabilities > 1e-15]
        rank = float(np.exp(-sum(active*np.log(active))))
        unit = pooled/np.linalg.norm(pooled, axis=1, keepdims=True)
        similarities = []
        for task in tasks:
            ix = [i for i, g in enumerate(groups) if g[0] == task]
            if len(ix) > 1:
                similarities.append(np.mean([np.dot(unit[i], unit[j]) for pos, i in enumerate(ix) for j in ix[pos+1:]]))
        result[camera] = {'fixed_position_std_rms': float(np.sqrt(variance)), 'pooled_effective_rank': rank,
                          'within_task_group_cosine': float(np.mean(similarities)),
                          'nonzero_pooled_vectors': bool(np.all(np.linalg.norm(pooled, axis=1) > 1e-12))}
    return result


def scalar_row(prediction, target, current, future):
    a, b = np.rint(current*1020).astype(np.int32), np.rint(future*1020).astype(np.int32)
    mask = abs(b-a).sum(0) > 120
    if mask.sum()*100 < 128*128:
        mask[:] = False
    w = mask.reshape(8, 16, 8, 16).mean((1, 3)).reshape(64)
    error = np.clip(1-(prediction.astype(np.float64)*target.astype(np.float64)).sum(-1), 0, 2)
    return np.array([error.mean(), (error*w).mean(), (error*(1-w)).mean(), w.mean(), (1-w).mean()])


@torch.inference_mode()
def audit(args):
    started = time.monotonic(); runtime = configure()
    root, out = args.source, args.run
    report = read_sealed(out/'report.json'); complete = read_sealed(out/'completion.json'); identity = read_sealed(out/'identity.json')
    assert report['identity_digest'] == identity['digest'] == complete['identity_digest']
    assert complete['report_digest'] == report['digest'] and report['status'] == complete['status'] == 'complete'
    assert runtime == identity['runtime'] and sha256_file(args.candidate) == report['candidate_sha256'] == identity['candidate_sha256']
    for n, d in identity['source_sha256'].items():
        assert sha256_file(root/n) == d, n
    for n, d in complete['files_sha256'].items():
        assert sha256_file(out/n) == d, n
    protocol = read_sealed(root/'configs/evaluation/week6_jepa_baselines_v1.json')
    assert protocol['digest'] == report['protocol_digest']
    for n, d in protocol['input_sha256'].items():
        assert sha256_file(root/n) == d, n
    base = read_sealed(args.baselines/'report.json')
    assert base['digest'] == report['baseline_report_digest']
    base_complete = read_sealed(args.baselines/'completion.json')
    for n, d in base_complete['files_sha256'].items():
        assert sha256_file(args.baselines/n) == d, n
    rows = read_sealed(out/'episode-metrics.json')['rows']; queries = read_sealed(out/'retrieval-panel.json')['queries']
    baserows = read_sealed(args.baselines/'episode-metrics.json')['rows']; basequeries = read_sealed(args.baselines/'retrieval-panel.json')['queries']
    manifest = read_sealed(args.pairs/'manifest.json')
    assert sha256_file(args.pairs/'manifest.json') == protocol['pair_manifest_sha256']
    assert sha256_file(args.pairs/'sources.jsonl') == manifest['outputs']['sources.jsonl']['sha256']
    sources = {s['episode_index']: s for s in map(json.loads, (args.pairs/'sources.jsonl').read_text().splitlines()) if s['split'] == 'validation'}
    expected = {(ep, c, h) for ep in sources for c in CAMERAS for h in HORIZONS}
    byep = defaultdict(list)
    for r in rows:
        expected.remove((r['episode_index'], r['camera'], r['horizon']))
        s = sources[r['episode_index']]
        assert r['source_digest'] == s['digest'] and r['pairs'] == s['episode_length']-r['horizon']
        byep[r['episode_index']].append(r)
    assert not expected and len(sources) == len(complete['episode_receipts']) == 1528
    assert sum(r['pairs'] for r in rows)//2 == 1109994
    panels, panelgroups, receipt_by_ep, qualitative = [], [], {}, []
    for rec in complete['episode_receipts']:
        ep = rec['episode_index']; path = out/'episodes'/f'{ep:06d}.json'; saved = read_sealed(path)
        assert sha256_file(path) == rec['sha256'] and saved['digest'] == rec['digest']
        assert saved['rows'] == byep[ep] and saved['identity_digest'] == identity['digest']
        assert saved['source_digest'] == sources[ep]['digest'] and saved['split'] == 'validation'
        for n, d in saved['arrays'].items():
            assert sha256_file(out/n) == d, n
        receipt_by_ep[ep] = saved; qualitative.extend(saved['qualitative_examples'])
        if saved['query_keys']:
            bank = np.load(out/f'panels/{ep:06d}-query-predictions.npy')
            means = np.load(out/f'panels/{ep:06d}-means.npy')
            assert np.array_equal(bank.mean(0, dtype=np.float64).astype(np.float32), means[0])
            assert np.array_equal(means[1], np.load(args.baselines/f'panels/{ep:06d}.npy')[1])
            panels.append(means[0]); panelgroups.append((sources[ep]['task_key'], sources[ep]['group_digest']))
    fixed = read_sealed(root/'configs/evaluation/week5_pixel_qualitative_v2.json')['examples']
    assert sorted(map(json.dumps, qualitative)) == sorted(map(json.dumps, fixed)) and len(qualitative) == 480
    result = verify_aggregates(baserows, rows, basequeries, queries, report)
    print(json.dumps({'phase': 'all-receipts-and-aggregates', 'maximum_error': result['primary_bootstrap_max_error']}), flush=True)
    collapse = independent_collapse(np.stack(panels), panelgroups)
    collapse_error = max(abs(collapse[c][f]-report['collapse'][c][f]) for c in CAMERAS for f in collapse[c])
    assert collapse_error < 1e-8
    effects = result['effects']
    checks = {'overall_cosine': effects['cosine_vs_copy']['value'] >= .05 and effects['cosine_vs_copy']['paired_95'][0] > 0,
              'mean_control': effects['cosine_vs_mean']['value'] > 0 and effects['cosine_vs_mean']['paired_95'][0] > 0,
              'motion_cosine': effects['motion_vs_copy']['value'] >= .1 and effects['motion_vs_copy']['paired_95'][0] > 0,
              'camera_horizon': all(c['passes'] for c in result['camera_horizon_cells']),
              'retrieval': effects['retrieval_recall1']['value'] >= .05 and effects['retrieval_recall1']['paired_95'][0] > 0 and report['retrieval']['metrics']['recall1']['value'] >= .225,
              'qualitative_review': False}
    for c in CAMERAS:
        t, v = base['collapse']['target'][c], collapse[c]
        checks['collapse/'+c] = v['fixed_position_std_rms'] >= .5*t['fixed_position_std_rms'] and v['pooled_effective_rank'] >= max(5, .5*t['pooled_effective_rank']) and v['within_task_group_cosine'] <= min(.999, t['within_task_group_cosine']+.05) and v['nonzero_pooled_vectors']
    assert checks == report['gate_with_review_pending']['checks']
    assert result['motion_groups'] >= 120 and len(result['groups']) == 180 and len({g[0] for g in result['groups']}) == 40
    replay_error = 0.
    for ep in complete['replay_episode_indices']:
        features = np.load(out/f'replay/{ep:06d}-features.npy', mmap_mode='r')
        boxes = np.load(out/f'replay/{ep:06d}-boxes.npy', mmap_mode='r')
        predictions = np.load(out/f'replay/{ep:06d}-predictions.npy', mmap_mode='r')
        keys = np.load(out/f'replay/{ep:06d}-pair-keys.npy')
        assert len(set(map(tuple, keys))) == len(predictions) == sum(sources[ep]['episode_length']-h for h in HORIZONS)
        sums = np.zeros((2, 5, 5))
        for n, (t, h) in enumerate(keys):
            for c in range(2):
                sums[c, HORIZONS.index(h)] += scalar_row(predictions[n, c*64:(c+1)*64], features[t+h, c*64:(c+1)*64], boxes[t, c], boxes[t+h, c])
        for r in byep[ep]:
            error = abs(sums[CAMERAS.index(r['camera']), HORIZONS.index(r['horizon'])]/r['pairs']-np.array([r['prediction'][f] for f in FIELDS]))
            replay_error = max(replay_error, float(error.max()))
    assert replay_error < 1e-10
    print(json.dumps({'phase': 'collapse-and-saved-replay', 'maximum_error': replay_error}), flush=True)
    full = load_pinned_qwen_model(cache_dir=str(args.cache), local_files_only=True).to('cuda')
    encoder = FrozenQwenFeatures(full.model).to('cuda'); processor = load_pinned_qwen_processor(cache_dir=str(args.cache), local_files_only=True)
    builder = FutureInputBuilder(processor); teacher = FrozenQwenVisualTargets(full.model.visual, processor.image_processor).eval()
    model, _ = load_jepa_checkpoint(args.candidate); model = model.eval().requires_grad_(False).to('cuda')
    states = [tensor_state_digest(x.state_dict()) for x in (model, encoder)]
    repository = args.data_root/f'datasets/libero-plus-lerobot/snapshots/{SOURCE_DATASET_REVISION}/repo'
    dataset = FuturePairDataset(args.pairs, repository, split='validation', expected_manifest_sha256=protocol['pair_manifest_sha256'],
                                normalizer=load_b0_action_normalizer(root/'docs/model/week4-fresh-v2/action-statistics.json'),
                                state_bounds=read_sealed(root/'configs/training/week4_b0_fresh_all_l1_v2.json')['state_normalization'])
    fresh_prediction_error, fresh_teacher_error, replayed_pairs = 0., 0., 0
    for ep in complete['replay_episode_indices']:
        source = sources[ep]; length = source['episode_length']
        predictions = np.load(out/f'replay/{ep:06d}-predictions.npy', mmap_mode='r')
        keys = {tuple(key): i for i, key in enumerate(np.load(out/f'replay/{ep:06d}-pair-keys.npy'))}
        for begin in sorted({0, ((length-2)//16)*8, ((length-3)//8)*8}):
            samples = [dataset[dataset.ordinal(ep, t, h)] for t in range(begin, min(begin+8, length-2)) for h in HORIZONS if t+h < length]
            condition, _, _ = encode_training_batch(samples, encoder, builder, torch.device('cuda'), feature_microbatch=4)
            for n in range(0, len(samples), 10):
                actual = model(slice_condition(condition, n, min(n+10, len(samples)))).features.cpu().numpy()
                saved = np.stack([predictions[keys[s.identity['anchor'], s.identity['horizon']]] for s in samples[n:n+10]])
                fresh_prediction_error = max(fresh_prediction_error, float(abs(actual-saved).max())); replayed_pairs += len(actual)
        frames = sorted({0, length//2, length-1})
        actual = encode(teacher, dataset._episode(source)[2][frames])
        saved = np.load(out/f'replay/{ep:06d}-features.npy', mmap_mode='r')[frames]
        fresh_teacher_error = max(fresh_teacher_error, float(abs(actual-saved).max()))
    assert fresh_prediction_error < 1e-5 and fresh_teacher_error == 0
    panel_sources = [sources[ep] for ep, rec in receipt_by_ep.items() if rec['query_keys']]
    selected = [min(panel_sources, key=lambda s: (s['task_key'], s['group_digest'])), max(panel_sources, key=lambda s: s['episode_length'])]
    real_score_error = 0.
    for s in selected:
        ep = s['episode_index']; qs = [q for q in queries if q['episode_index'] == ep]
        frames = sorted({f for q in qs for f in q['pool']['frames']})
        rgb = decode(repository, s, 'validation', frames); features = encode(teacher, rgb)
        positions = {f: i for i, f in enumerate(frames)}
        predictions = dict(zip(map(tuple, receipt_by_ep[ep]['query_keys']), np.load(out/f'panels/{ep:06d}-query-predictions.npy'), strict=True))
        for q in qs:
            c = CAMERAS.index(q['camera']); p = predictions[q['anchor'], q['horizon']][c*64:(c+1)*64].astype(np.float64)
            assert [hashlib.sha256(rgb[positions[f], c].tobytes()).hexdigest() for f in q['pool']['frames']] == q['pool']['image_sha256']
            scores = [float((p*features[positions[f], c*64:(c+1)*64].astype(np.float64)).sum()/64) for f in q['pool']['frames']]
            real_score_error = max(real_score_error, float(abs(np.asarray(scores)-q['prediction']['scores']).max()))
    assert real_score_error < 1e-10
    assert [tensor_state_digest(x.state_dict()) for x in (model, encoder)] == states
    result.pop('groups')
    return seal_contract({'artifact_type': 'eventground_wam.jepa_candidate_independent_verification', 'status': 'passed',
                          'report_digest': report['digest'], 'completion_digest': complete['digest'], 'candidate_sha256': sha256_file(args.candidate),
                          'auditor_sha256': sha256_file(Path(__file__)), 'arithmetic_helper_sha256': sha256_file(Path(__file__).with_name('verify_jepa_baselines.py')),
                          **result, 'independently_recomputed_checks': checks, 'collapse': collapse, 'collapse_max_error': collapse_error,
                          'saved_full_episode_replay_max_error': replay_error, 'saved_replay_episodes': complete['replay_episode_indices'],
                          'fresh_prediction_max_error': fresh_prediction_error, 'fresh_prediction_pairs': replayed_pairs,
                          'fresh_teacher_max_error': fresh_teacher_error, 'fresh_retrieval_score_max_error': real_score_error,
                          'fresh_retrieval_episodes': [s['episode_index'] for s in selected], 'full_pair_coverage': 1109994,
                          'retrieval_queries': 3240, 'qualitative_identities_saved': 480, 'models_unchanged': True,
                          'optimizer_updates': 0, 'test_payload_reads': 0, 'elapsed_seconds': time.monotonic()-started,
                          'candidate_accepted': False, 'limits': 'All receipts, arrays, aggregates and score ranks verified; two complete saved-array episode replays, fresh start/middle/tail predictor batches and teacher observations, and two fresh retrieval-panel replays. Not a second full GPU evaluation.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'run', 'baselines', 'pairs', 'candidate', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path.home()/'data/eventground-wam/libero-plus')
    parser.add_argument('--cache', type=Path, default=Path.home()/'data/eventground-wam/huggingface-cache')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, audit(args))
    print('Independent candidate audit passed', flush=True)
