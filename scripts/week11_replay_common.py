"""Portable metrics and selection for recording-based horizon replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ('M4-EG-fixed8', 'M4-EG-adaptive')
HORIZONS = (2, 4, 8, 16, 32)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def seal(value):
    v = dict(value); v.pop('digest', None)
    v['digest'] = hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    return v


def read(path):
    v = json.loads(Path(path).read_text())
    assert seal(v)['digest'] == v['digest'], str(path)
    return v


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(seal(value), indent=2, sort_keys=True, allow_nan=False)+'\n')
    tmp.replace(path)


def distance(a, b):
    """b relative to baseline a, evaluated in float64 without scale assumptions."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    na, nb = np.linalg.norm(a.ravel()), np.linalg.norm(b.ravel())
    return {'rms_delta': float(np.sqrt(np.mean((b-a)**2))),
        'relative_l2_delta': float(np.linalg.norm((b-a).ravel())/na) if na else None,
        'cosine': float(np.clip(np.dot(a.ravel(), b.ravel())/(na*nb), -1, 1)) if na and nb else None,
        'baseline_rms': float(np.sqrt(np.mean(a*a))), 'changed_rms': float(np.sqrt(np.mean(b*b)))}


def select(rows):
    pairs, empty = [], []
    for panel in ('targeted-467', 'clean-200'):
        for suite in ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10'):
            for outcome in ('adaptive_loss', 'adaptive_win', 'both_success', 'both_failure'):
                candidates = [r for r in rows if r['panel'] == panel and r['task']['suite'] == suite
                    and r['outcome_class'] == outcome and r['divergence']['first_non8_at_shared_prefix']
                    and r['divergence']['first_non8_at_shared_prefix']['inputs_equal']]
                if candidates:
                    pairs.append(min(candidates, key=lambda r: r['ordinal']))
                else:
                    empty.append({'panel': panel, 'suite': suite, 'outcome': outcome})
    observations, omitted = [], []
    for r in pairs:
        selected = {}
        q = r['divergence']['first_non8_query']
        selected[ARMS[0], q] = ['shared_first_departure']
        for arm in ARMS:
            onsets = [x for x in r['arms'][arm]['closing_proxy']['confirmed_onsets'] if x > 10]
            if not onsets:
                omitted.append({'panel': r['panel'], 'ordinal': r['ordinal'], 'arm': arm,
                                'reason': 'no_confirmed_closing_after_first_query'})
                continue
            onset = onsets[0]
            for role, qi in [('before_first_closing', (onset-11)//8),
                             ('after_first_closing', (onset-10)//8+1)]:
                if qi < len(r['arms'][arm]['queries']):
                    selected.setdefault((arm, qi), []).append(role)
                else:
                    omitted.append({'panel': r['panel'], 'ordinal': r['ordinal'], 'arm': arm,
                                    'reason': 'after_closing_query_not_recorded'})
        for (arm, qi), roles in selected.items():
            observations.append({'id': f'{r["panel"]}-{r["ordinal"]:04d}-{arm}-q{qi:03d}',
                'panel': r['panel'], 'ordinal': r['ordinal'], 'task': r['task'],
                'outcomes': r['outcomes'], 'outcome_class': r['outcome_class'], 'roles': roles,
                'arm': arm, 'query': qi, 'step': 10+8*qi, 'artifact': r['artifacts'][arm],
                'paired_adaptive_artifact': r['artifacts'][ARMS[1]] if 'shared_first_departure' in roles else None})
    return {'pairs': [{k: r[k] for k in ('panel', 'ordinal', 'task', 'outcome_class', 'artifacts')} for r in pairs],
            'observations': observations, 'empty_strata': empty, 'omitted_anchors': omitted}
