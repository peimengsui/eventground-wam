from dataclasses import replace

import pytest
import torch

from eventground_wam.jepa_followup import (
    ResidualJepaPredictor,
    backward_batch,
    head_span,
    select_groups,
    span_lower_bound,
    training_plan,
    validate_selection,
)
from eventground_wam.jepa_predictor import (
    JepaPredictor,
    JepaPredictorConfig,
    jepa_prediction_loss,
    normalize_predictions,
)
from eventground_wam.pixel_predictor import PixelCondition


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def make_batch():
    torch.manual_seed(13)
    condition = PixelCondition(torch.randn(2, 5, 8), torch.zeros(2, 5, dtype=torch.bool),
                              torch.rand(2, 2, 3, 128, 128), torch.randn(2, 8),
                              torch.randn(2, 32, 7), torch.tensor([2, 32]))
    current = normalize_predictions(torch.randn(2, 128, 2048)).detach().requires_grad_()
    return condition, current


def test_copy_initialization_same_core_and_no_teacher_gradient():
    config = JepaPredictorConfig(feature_dimension=8, dimension=32, heads=4, layers=1)
    residual, baseline = ResidualJepaPredictor(config), JepaPredictor(config)
    for name, value in baseline.state_dict().items():
        if not name.startswith('output_projection.'):
            assert torch.equal(value, residual.state_dict()[name])
    assert residual.parameter_budget() == baseline.parameter_budget()
    x, current = make_batch()
    torch.testing.assert_close(residual(x, current).features, current, atol=1e-6, rtol=0)
    optimizer = torch.optim.AdamW(residual.parameters(), lr=.001)
    truth = normalize_predictions(torch.randn_like(current)).detach()
    for _ in range(3):
        optimizer.zero_grad()
        (1-(residual(x, current).features*truth).sum(-1)).mean().backward()
        optimizer.step()
    assert current.grad is None
    assert residual.action_projection[0].weight.grad.abs().sum() > 0
    normal = residual(x, current)
    reversed_prediction = residual(x, current, ablation='reverse-actions')
    assert (normal.features-reversed_prediction.features).abs().max() > 1e-6
    assert (normal.future_tokens-reversed_prediction.future_tokens).abs().max() > 1e-6
    with pytest.raises(TypeError):
        residual(x, current, targets=truth)
    with pytest.raises(ValueError):
        residual(x, torch.zeros_like(current))
    poisoned = x.actions.clone(); poisoned[0, 2:] = float('nan')
    torch.testing.assert_close(residual(replace(x, actions=poisoned), current).features,
                               normal.features, atol=0, rtol=0)


def test_group_split_reproducibility_and_leak_rejection():
    groups = [{'task_key': f't{t}', 'group_digest': f'{t}-{g}', 'split': split,
               'episode_indices': [100*t+g*2, 100*t+g*2+1], 'episode_length': 70}
              for t in range(2) for g in range(6) for split in ['train']]
    groups.append({**groups[0], 'split': 'test', 'group_digest': 'test', 'episode_indices': [999]})
    selected = select_groups(groups)
    assert selected == select_groups(list(reversed(groups))) and len(selected) == 8
    assert all(r['split'] == 'train' and r['episode_index'] != 999 for r in selected)
    with pytest.raises(ValueError):
        validate_selection(selected+[selected[0]])
    with pytest.raises(ValueError):
        validate_selection([{**selected[0], 'split': 'validation'}, *selected[1:]])
    plan = training_plan(16, 7)
    assert plan == training_plan(16, 7)
    assert sorted(plan[0]+plan[1]) == list(range(16))


def test_head_bound_is_optimistic_for_actual_predictions():
    model = JepaPredictor(JepaPredictorConfig(feature_dimension=8, dimension=32, heads=4, layers=1))
    tokens = torch.randn(4, 128, 32)
    predicted = normalize_predictions(model.output_projection(tokens)).detach()
    basis = head_span(model)
    assert span_lower_bound(predicted, basis).max() < 1e-6
    target = normalize_predictions(torch.randn_like(predicted)).detach()
    actual = (1-(predicted.double()*target.double()).sum(-1)).clamp(0, 2)
    assert (span_lower_bound(target, basis) <= actual+1e-8).all()


def test_residual_logical_batch_gradient_matches_full_graph():
    config = JepaPredictorConfig(feature_dimension=8, dimension=32, heads=4, layers=1)
    a, b = ResidualJepaPredictor(config), ResidualJepaPredictor(config)
    with torch.no_grad():
        a.output_projection.weight.normal_(0, .01)
    b.load_state_dict(a.state_dict())
    x, current = make_batch()
    target = normalize_predictions(torch.randn_like(current)).detach()
    groups = torch.tensor([0, 1]); rule = {'variance_weight': 10.}
    full = jepa_prediction_loss(a(x, current).features, target, groups, **rule)
    full.total.backward()
    replay = backward_batch(b, x, current, target, groups, rule, microbatch=1)
    torch.testing.assert_close(full.total, replay.total, atol=1e-6, rtol=1e-6)
    for p, q in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=1e-4)
    assert current.grad is None
