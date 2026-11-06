#!/usr/bin/env python3
"""Verify all diagnostic identities/summaries and independently replay saved fields."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.future_pairs import CAMERAS, read_sealed, sha256_file, write_json

FIELDS = ('target_cosine', 'common_feature_drift', 'native_token_drift', 'target_teacher_drift', 'excess_target_error')


def verify(root):
    report = read_sealed(root/'report.json'); contract = read_sealed(root/'contract.json')
    rows = read_sealed(root/'measurements.json')['rows']
    assert report['contract_digest'] == contract['digest']
    all_rows, examples, supplemental = [], [], []
    for rec in report['episode_receipts']:
        episode = read_sealed(root/f"episodes/{rec['episode_index']:06d}.json")
        assert episode['digest'] == rec['digest'] and episode['contract_digest'] == contract['digest']
        for name, digest in episode['files_sha256'].items():
            assert sha256_file(root/name) == digest, name
        all_rows.extend(episode['rows']); examples.extend(episode['fixed_examples']); supplemental.extend(episode['supplemental_h2_examples'])
    assert all_rows == rows and len(examples) == 480 and len(supplemental) == 240
    expected = {(ex['episode_index'], ex['camera'], ex['anchor'], ex['horizon'], t['name'], m)
                for ex in examples for t in contract['transforms'] for m in ('copy', 'pixel', 'jepa')}
    for r in rows:
        expected.remove(tuple(r[k] for k in ('episode_index', 'camera', 'anchor', 'horizon', 'transform', 'method')))
        assert all(np.isfinite(r[f]) for f in FIELDS)
    assert not expected
    errors = []

    def check(saved, values, paired=False):
        # Explicit accumulation over random draws, not vectorized sampled means.
        rng = np.random.default_rng(20260928)
        indices = rng.integers(len(values), size=(10000, len(values)))
        bootstrap = np.zeros((10000, len(FIELDS)))
        for col in range(len(values)):
            bootstrap += values[indices[:, col]]/len(values)
        for j, f in enumerate(FIELDS):
            errors.append(abs(float(sum(values[:, j])/len(values))-saved[f]['value']))
            errors.extend(abs(np.quantile(bootstrap[:, j], [.025, .975])-saved[f]['paired_task_bootstrap_95' if paired else 'task_bootstrap_95']))

    for summary in report['by_transform_method_camera']:
        selected = [r for r in rows if r['transform'] == summary['transform'] and r['method'] == summary['method']
                    and (summary['camera'] == 'both' or r['camera'] == summary['camera'])]
        tasks = sorted({r['task_key'] for r in selected})
        assert len(tasks) == summary['tasks'] == 40 and len(selected) == summary['examples']
        values = np.array([[sum(r[f] for r in selected if r['task_key'] == task)/sum(r['task_key'] == task for r in selected)
                            for f in FIELDS] for task in tasks])
        check(summary['metrics'], values)
    for summary in report['paired_differences']:
        values = []
        for task in sorted({r['task_key'] for r in rows}):
            selected = [r for r in rows if r['transform'] == summary['transform'] and r['task_key'] == task]
            means = {m: np.mean([[r[f] for f in FIELDS] for r in selected if r['method'] == m], axis=0) for m in ('jepa', 'pixel')}
            values.append(means['jepa']-means['pixel'])
        check(summary['jepa_minus_pixel'], np.asarray(values), True)
    assert max(errors) < 1e-10
    episode = next(read_sealed(root/f"episodes/{r['episode_index']:06d}.json") for r in report['episode_receipts']
                   if read_sealed(root/f"episodes/{r['episode_index']:06d}.json")['task_key'] == contract['arithmetic_replay_task'])
    ep = episode['episode_index']
    def array(name, field):
        return np.load(root/f'arrays/{ep:06d}-{name}-{field}.npy')
    baseline = {f: array('identity', f) for f in ('copy', 'jepa', 'pixel', 'targets', 'jepa_tokens', 'pixel_tokens')}
    lookup = {tuple(key): n for n, key in enumerate(array('identity', 'keys'))}
    replay_error = 0.
    for t in contract['transforms']:
        name = t['name']; targets = array(name, 'targets'); keys = array(name, 'keys')
        loaded = {f: array(name, f) for f in ('copy', 'jepa', 'pixel', 'jepa_tokens', 'pixel_tokens')}
        for r in [x for x in rows if x['episode_index'] == ep and x['transform'] == name]:
            n = next(i for i, key in enumerate(keys) if tuple(key) == (r['anchor'], r['horizon']))
            old = lookup[r['anchor'], r['horizon']]; c = CAMERAS.index(r['camera']); ix = slice(c*64, (c+1)*64)
            def distance(a, b, ix=ix):
                return float(np.mean([max(0., min(2., 1-float(np.dot(u.astype(np.float64), v.astype(np.float64))))) for u, v in zip(a[ix], b[ix], strict=True)]))
            m = r['method']; token = m+'_tokens' if m != 'copy' else 'copy'
            target_error = distance(loaded[m][n], targets[n])
            actual = [target_error, distance(loaded[m][n], baseline[m][old]),
                      distance(loaded[token][n], baseline[token][old]), distance(targets[n], baseline['targets'][old]),
                      target_error-distance(baseline[m][old], baseline['targets'][old])]
            replay_error = max(replay_error, max(abs(actual[j]-r[f]) for j, f in enumerate(FIELDS)))
    assert replay_error < 1e-10
    return seal_contract({'artifact_type': 'eventground_wam.jepa_appearance_independent_verification', 'status': 'passed',
                          'report_digest': report['digest'], 'contract_digest': contract['digest'],
                          'auditor_sha256': sha256_file(Path(__file__)), 'metric_rows': len(rows),
                          'fixed_examples': len(examples), 'supplemental_examples': len(supplemental),
                          'summary_bootstrap_max_error': float(max(errors)), 'saved_field_replay_max_error': replay_error,
                          'arithmetic_replay_task': contract['arithmetic_replay_task'],
                          'limits': 'All identities, file hashes and reported aggregates verified; one fixed task independently replayed from saved fields. This is not the visual review.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, verify(args.run))
