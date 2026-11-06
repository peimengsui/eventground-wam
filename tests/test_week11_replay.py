"""Selection, metric, and instrumentation invariants before GPU replay."""
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from run_week11_horizon_replay import capture
from summarize_week11_replay import annotate
from torch import nn
from week11_replay_common import distance, select


def test_distances_identical_and_scaled():
    a = np.arange(1, 8)
    assert distance(a, a)['relative_l2_delta'] == 0
    d = distance(a, 2*a)
    assert d['relative_l2_delta'] == 1 and abs(d['cosine']-1) < 1e-12


def test_zero_baseline_is_not_reported_as_no_change():
    d = distance(np.zeros(4), np.ones(4))
    assert d['relative_l2_delta'] is None and d['cosine'] is None and d['rms_delta'] == 1


def test_event_type_and_horizon_agreement_are_distinct():
    event = {'event_type': 'contact', 'start': 4, 'end': 8, 'left_censored': False,
             'right_censored': False, 'object_id': 'object'}
    c = {'digest': 'candidate', 'frames': 40, 'events': [event], 'ambiguities': []}
    r = {'step': 0, 'prediction': {'next_event': 'grasp', 'horizon': 5, 'interval': [5, 12]}}
    a = annotate(r, c)
    assert a['eligible'] and not a['type_match'] and a['mapped_horizon_match']
    assert a['signed_onset_error_if_type_match'] is None


def test_ambiguous_or_censored_event_windows_are_not_negatives():
    r = {'step': 0, 'prediction': {'next_event': 'none', 'horizon': None, 'interval': None}}
    c = {'digest': 'candidate', 'frames': 40, 'events': [], 'ambiguities': [{'start': 9, 'end': 10}]}
    a = annotate(r, c)
    assert not a['eligible'] and a['exclusion'] == 'unresolved_ambiguity' and 'target' not in a
    a = annotate({**r, 'step': 8}, {**c, 'ambiguities': []})
    assert not a['eligible'] and a['exclusion'] == 'incomplete_future_window'


def test_selection_order_and_anchors():
    def row(i):
        return {'panel': 'targeted-467', 'ordinal': i, 'task': {'suite': 'libero_spatial'},
            'outcomes': [1, 1, 0], 'outcome_class': 'adaptive_loss',
            'divergence': {'first_non8_query': 1, 'first_non8_at_shared_prefix': {'inputs_equal': True}},
            'artifacts': {'M4-EG-fixed8': 'fixed', 'M4-EG-adaptive': 'adaptive'},
            'arms': {a: {'closing_proxy': {'confirmed_onsets': [26]}, 'queries': [None]*10}
                     for a in ('M4-EG-fixed8', 'M4-EG-adaptive')}}
    s = select([row(7), row(2)])
    assert s['pairs'][0]['ordinal'] == 2 and len(s['pairs']) == 1
    assert len(s['observations']) == 4  # before closing shares first-departure anchor
    assert s['observations'][0]['roles'] == ['shared_first_departure', 'before_first_closing']
    assert {x['query'] for x in s['observations']} == {1, 3}


@dataclass
class Condition:
    actions: torch.Tensor
    horizons: torch.Tensor


class Core(nn.Module):
    def forward(self, c):
        x = torch.arange(128*256, dtype=torch.float32).reshape(1, 128, 256)/100
        return x + c.horizons.float().reshape(1, 1, 1)*torch.sin(x)


class Coupling(nn.Module):
    def __init__(self):
        super().__init__(); self.future_norm = nn.LayerNorm(256)

    def forward(self, q, tokens):
        return q+self.future_norm(tokens).mean(dim=1, keepdim=True).repeat(1, 8, 8)


class Model(nn.Module):
    def __init__(self):
        super().__init__(); self.proposal = nn.Identity(); self.future_core = Core()
        self.coupling = Coupling()

    def forward(self):
        actions = self.proposal(torch.zeros(1, 32, 7))
        tokens = self.future_core(Condition(actions, torch.tensor([8])))
        out = self.coupling(torch.ones(1, 8, 2048), tokens)[:, :, :7]
        return SimpleNamespace(normalized_actions=out, raw_action_predictions=out)


class Policy:
    def __init__(self):
        self._policy, self._device = Model(), 'cpu'

    def predict_chunk(self, **_):
        with torch.inference_mode():
            return self._policy().normalized_actions.numpy()[0].astype(np.float64)


def test_capture_preserves_baseline_and_removes_hooks():
    p = Policy(); baseline = p.predict_chunk()
    a = capture(p, {}, 8); b = capture(p, {}, 32); c = capture(p, {}, 8)
    assert np.array_equal(a['decoded_actions'], baseline)
    assert all(np.array_equal(a[k], c[k]) for k in a)
    assert not np.array_equal(a['future_tokens'], b['future_tokens'])
    assert np.array_equal(a['proposal'], b['proposal'])
    assert np.array_equal(a['query_hidden'], b['query_hidden'])
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in p._policy.modules())


def test_capture_cleans_hooks_on_error():
    p = Policy()
    def fail(**_):
        raise ValueError('test')
    p.predict_chunk = fail
    try:
        capture(p, {}, 8)
    except ValueError:
        pass
    else:
        raise AssertionError('exception swallowed')
    assert all(not m._forward_hooks and not m._forward_pre_hooks for m in p._policy.modules())
