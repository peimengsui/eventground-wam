from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from test_b0_oft import batch as action_batch
from test_b0_oft import policy as base_policy

from eventground_wam.auxiliary_policy import AuxiliaryInputs, ObservationFeatures
from eventground_wam.b0_data import ActionNormalizer
from eventground_wam.b0_inference import decode_b0_action_chunk
from eventground_wam.b0_model import B0ObjectiveConfig
from eventground_wam.coupled_checkpoint import export_coupled_policy, load_coupled_policy
from eventground_wam.coupled_policy import (
    COUPLING_PARAMETERS,
    LOSS_WEIGHTS,
    CoupledPolicy,
    ExecutableProposal,
    FutureCrossAttention,
    coupled_joint_backward,
    coupled_parameter_groups,
)
from eventground_wam.jepa_predictor import (
    JepaPredictor,
    jepa_prediction_loss,
    normalize_predictions,
)
from eventground_wam.jepa_training import LOSS_RULE
from eventground_wam.pixel_predictor import (
    PixelCondition,
    PixelPredictor,
    pixel_reconstruction_loss,
)

LORA_NAMES = ('conditioning_adapter.backbone.lora_scale',)


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def setup(arm):
    torch.manual_seed(17)
    policy = base_policy()
    policy.objective_config = B0ObjectiveConfig('all-action-l1-binary-gripper-v2')
    predictor = PixelPredictor() if arm == 'M3' else JepaPredictor()
    if arm == 'M3':
        with torch.no_grad():
            predictor.residual_head.weight.normal_(std=.01)
    model = CoupledPolicy(policy, predictor, ActionNormalizer.identity(), arm=arm)
    b = action_batch(0)
    return model, predictor, b, torch.rand(2, 2, 3, 128, 128)


def run(model, batch, images, **kwargs):
    return model.forward_with_features(batch['inputs'], batch['proprioception'], images, **kwargs)


def activate(model):
    with torch.no_grad():
        # Exercise all K/V/core paths after the expected zero-initialized first update.
        model.coupling.output.weight.normal_(std=.01)


@pytest.mark.parametrize('arm', ['M3', 'M4'])
def test_core_tokens_match_original_and_do_not_execute_training_outputs(arm, monkeypatch):
    model, predictor, b, images = setup(arm)
    result = run(model, b, images)
    condition = PixelCondition(result.features.values, result.features.padding, images,
        torch.zeros(2, 8), result.candidate_actions, torch.tensor([8, 8]))
    original = predictor(condition).future_tokens
    expected_state = {n: v.clone() for n, v in predictor.state_dict().items()}
    def forbidden(*args, **kwargs):
        raise AssertionError('training output executed during action inference')
    monkeypatch.setattr(predictor, 'decode', forbidden)
    if arm == 'M4':
        monkeypatch.setattr(predictor.output_projection, 'forward', forbidden)
    assert torch.equal(model.future_core(condition), original)
    assert all(torch.equal(v, predictor.state_dict()[n]) for n, v in expected_state.items())
    assert not any(n.startswith(('image_decoder.', 'residual_head.', 'output_projection.'))
                   for n in model.future_core.state_dict())
    assert set(map(id, model.future_core.parameters())) < set(map(id, predictor.parameters()))


def test_coupling_exact_parameter_inventory_initialization_and_rng_preservation():
    torch.manual_seed(98)
    rng = torch.get_rng_state().clone()
    block = FutureCrossAttention()
    assert torch.equal(torch.get_rng_state(), rng)
    assert sum(p.numel() for p in block.parameters()) == COUPLING_PARAMETERS
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261005)
        expected = [torch.nn.Linear(2048, 256), torch.nn.Linear(256, 256), torch.nn.Linear(256, 256)]
    for actual, reference in zip((block.query, block.key, block.value), expected, strict=True):
        assert torch.equal(actual.weight, reference.weight)
        assert torch.equal(actual.bias, reference.bias)
    queries, futures = torch.randn(2, 8, 2048), torch.randn(2, 128, 256)
    assert torch.equal(block(queries, futures), queries)
    assert block.output.weight.count_nonzero() == block.output.bias.count_nonzero() == 0


def test_proposal_matches_real_decoder_in_simulator_units_and_is_detached():
    normalizer = ActionNormalizer(center=(.7, -.8, .1, 0., .3, -.4, 0.),
        scale=(.2, .3, 2., .01, .9, 1.5, 1.), lower=(-1.,)*7, upper=(1.,)*7)
    raw = torch.linspace(-12, 12, 2*8*7).reshape(2, 8, 7).requires_grad_()
    raw.data[0, 0, 6] = 0
    actual = ExecutableProposal(normalizer)(raw)
    for i in range(2):
        decoded = raw[i].detach().numpy().copy()
        decoded[:, 6] = np.where(decoded[:, 6] >= 0, 1, -1)
        executable, _ = decode_b0_action_chunk(decoded, normalizer)
        expected = torch.from_numpy(normalizer.normalize(executable).astype(np.float32))
        assert torch.equal(actual[i, :8], expected)
    assert not actual.requires_grad and actual[:, 8:].count_nonzero() == 0
    assert actual[0, 0, 6] == 1 and actual[:, :8, :6].abs().max() > 1


@pytest.mark.parametrize('arm', ['M3', 'M4'])
def test_one_backbone_two_same_head_calls_zero_parity_and_target_isolation(arm):
    model, _predictor, b, images = setup(arm)
    original = model.policy(b['inputs'], b['proprioception'], normalized_action_targets=b['targets'])
    before = {n: p.clone() for n, p in model.policy.state_dict().items()}
    calls = {'backbone': 0, 'head': 0, 'core': 0}
    handles = []
    for name, module in [('backbone', model.policy.conditioning_adapter.backbone),
                         ('head', model.policy.action_decoder), ('core', model.future_core)]:
        def count(_m, _a, _o, key=name):
            calls[key] += 1
        handles.append(module.register_forward_hook(count))
    result = run(model, b, images, normalized_action_targets=b['targets'])
    for handle in handles:
        handle.remove()
    assert calls == {'backbone': 1, 'head': 2, 'core': 1}
    assert torch.equal(result.action_output.raw_action_predictions, original.raw_action_predictions)
    assert torch.equal(result.action_output.normalized_actions, original.normalized_actions)
    assert torch.equal(result.action_output.loss, original.loss)
    assert all(torch.equal(v, model.policy.state_dict()[n]) for n, v in before.items())
    changed = run(model, b, images, normalized_action_targets=-b['targets'])
    assert torch.equal(result.action_output.raw_action_predictions, changed.action_output.raw_action_predictions)
    with pytest.raises(TypeError):
        model(b['inputs'], b['proprioception'], images, future_mask=torch.ones(2, 5))
    assert not model.policy.conditioning_adapter.backbone._forward_hooks
    assert not any('image_decoder' in n or 'output_projection' in n for n in model.state_dict())


@pytest.mark.parametrize('arm', ['M3', 'M4'])
def test_action_gradient_boundaries_at_zero_and_nonzero_coupling(arm):
    model, predictor, b, images = setup(arm)
    for nonzero in (False, True):
        if nonzero:
            activate(model)
        model.zero_grad(set_to_none=True)
        truth = b['targets'].clone().requires_grad_()
        result = run(model, b, images, normalized_action_targets=truth)
        result.proposal_raw.retain_grad()
        result.action_output.loss.backward()
        assert result.proposal_raw.grad is None and not result.candidate_actions.requires_grad
        assert truth.grad is None
        assert model.coupling.output.weight.grad.abs().sum() > 0
        assert model.policy.action_decoder.network[-1].weight.grad.abs().sum() > 0
        assert model.policy.conditioning_adapter.proprioception_projector[0].weight.grad.abs().sum() > 0
        assert model.policy.conditioning_adapter.backbone.lora_scale.grad.abs().sum() > 0
        assert model.policy.conditioning_adapter.backbone.embedding.weight.grad is None
        for parameter in (model.coupling.key.weight, model.coupling.value.weight,
                          model.future_core.action_projection[0].weight):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert bool(parameter.grad.abs().sum() > 0) == nonzero
        output_head = predictor.residual_head if arm == 'M3' else predictor.output_projection
        assert output_head.weight.grad is None
    with torch.no_grad():
        hidden, tokens = result.action_output.conditioning_memory.tokens, result.future_tokens
        assert not torch.equal(model.coupling(hidden, tokens), model.coupling(hidden, tokens.flip(2)))


def auxiliary(result, images):
    # No horizon-8 target exists, and anchor 1 has just one available target.
    return AuxiliaryInputs(torch.tensor([0, 0, 1]), images, torch.zeros(2, 8),
                           torch.randn(3, 32, 7), torch.tensor([2, 4, 2]))


@pytest.mark.parametrize('arm', ['M3', 'M4'])
@pytest.mark.parametrize('microbatch', [1, 2, 3])
def test_joint_whole_batch_gradient_parity_and_supervision_isolation(arm, microbatch):
    model, predictor, b, images = setup(arm)
    activate(model)
    groups = coupled_parameter_groups(model, predictor, LORA_NAMES)
    parameters = [p for group in groups for p in group['params']]
    assert len(parameters) == len(set(map(id, parameters)))
    result = run(model, b, images, normalized_action_targets=b['targets'])
    future = auxiliary(result, images)
    truth = (torch.rand(3, 2, 3, 128, 128) if arm == 'M3' else
             normalize_predictions(torch.randn(3, 128, 2048))).requires_grad_()
    pred = predictor(future.condition(result.features))
    loss = (pixel_reconstruction_loss(pred.images, truth.detach(), ssim_weight=.1) if arm == 'M3'
            else jepa_prediction_loss(pred.features, truth, future.anchor_indices, **LOSS_RULE).total)
    (result.action_output.loss + LOSS_WEIGHTS[arm]*loss).backward()
    expected = [p.grad.clone() for p in parameters]
    for p in parameters:
        p.grad = None
    other = run(model, b, images, normalized_action_targets=b['targets'])
    metrics = coupled_joint_backward(model, other, predictor, future, truth, microbatch=microbatch)
    error = torch.cat([(p.grad-g).flatten() for p, g in zip(parameters, expected, strict=True)])
    reference = torch.cat([g.flatten() for g in expected])
    assert error.abs().max() < 1e-5
    assert error.norm()/reference.norm() < (1e-3 if arm == 'M3' else 1e-5)
    assert abs(metrics['future_loss']-float(loss.detach())) < 1e-6
    assert truth.grad is None and other.future_tokens.shape == (2, 128, 256)
    # A different teacher-forced request does not mutate the action path or its core.
    changed = replace(future, actions=-future.actions, horizons=torch.tensor([32, 16, 2]))
    predictor(changed.condition(ObservationFeatures(other.features.values.detach(), other.features.padding)))
    after = run(model, b, images)
    assert torch.equal(other.action_output.raw_action_predictions, after.action_output.raw_action_predictions)


@pytest.mark.parametrize('arm', ['M3', 'M4'])
def test_short_batch_empty_supervision_and_export_reload(arm, tmp_path):
    model, predictor, b, images = setup(arm)
    activate(model)
    small = {**b, 'inputs': {k: v[:1] for k, v in b['inputs'].items()},
             'proprioception': b['proprioception'][:1], 'targets': b['targets'][:1]}
    result = run(model, small, images[:1], normalized_action_targets=small['targets'])
    empty = AuxiliaryInputs(torch.empty(0, dtype=torch.int64), images[:1], torch.zeros(1, 8),
                            torch.empty(0, 32, 7), torch.empty(0, dtype=torch.int64))
    metrics = coupled_joint_backward(model, result, predictor, empty, torch.empty(0))
    assert metrics['future_pairs'] == 0 and result.future_tokens.shape == (1, 128, 256)
    file = tmp_path/'policy.pt'
    header = export_coupled_policy(file, model, LORA_NAMES, metadata={'scope': 'synthetic interface test'})
    reloaded, _, _, _ = setup(arm)
    load_coupled_policy(file, reloaded, LORA_NAMES)
    after = run(reloaded, small, images[:1])
    assert torch.equal(after.action_output.raw_action_predictions, result.action_output.raw_action_predictions)
    assert torch.equal(after.future_tokens, result.future_tokens)
    assert header['arm'] == arm
    with pytest.raises(FileExistsError):
        export_coupled_policy(file, model, LORA_NAMES, metadata={})
    artifact = torch.load(file, weights_only=True)
    artifact['states']['coupling']['output.bias'][0] += 1
    torch.save(artifact, file)
    with pytest.raises(ValueError, match='weights changed'):
        load_coupled_policy(file, reloaded, LORA_NAMES)


def test_reject_unshared_teacher_forced_core_and_wrong_objective():
    model, _, _, _ = setup('M4')
    with pytest.raises(ValueError, match='share'):
        coupled_parameter_groups(model, JepaPredictor(), LORA_NAMES)
    with pytest.raises(ValueError, match='all-action-L1'):
        CoupledPolicy(base_policy(), PixelPredictor(), ActionNormalizer.identity(), arm='M3')


def test_attention_stays_fp32_inside_outer_autocast_and_rejects_bad_shapes():
    block = FutureCrossAttention()
    with torch.no_grad():
        block.output.weight.normal_(std=.01)
    query, tokens = torch.randn(1, 8, 2048), torch.randn(1, 128, 256)
    reference = block(query, tokens)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = block(query, tokens)
    assert actual.dtype == torch.float32 and torch.equal(actual, reference)
    for bad in (tokens[:, :64], tokens*float('nan')):
        with pytest.raises(ValueError):
            block(query, bad)
    block.bfloat16()
    with pytest.raises(ValueError, match='FP32'):
        block(query, tokens)


@pytest.mark.parametrize('arm', ['M3', 'M4'])
def test_invalid_supervision_leaves_no_hooks_and_export_rejects_wrong_normalizer(arm, tmp_path):
    model, _, b, images = setup(arm)
    with pytest.raises(ValueError, match='gripper'):
        run(model, b, images, normalized_action_targets=torch.zeros_like(b['targets']))
    assert not model.policy.conditioning_adapter.backbone._forward_hooks
    path = tmp_path/'inference.pt'
    export_coupled_policy(path, model, LORA_NAMES, metadata={})
    model.proposal = ExecutableProposal(ActionNormalizer(center=(.1,)*6+(0.,),
        scale=(1.,)*7, lower=(-1.,)*7, upper=(1.,)*7))
    with pytest.raises(ValueError, match='normalizer'):
        load_coupled_policy(path, model, LORA_NAMES)


@pytest.mark.parametrize('arm', ['M3', 'M4'])
def test_candidate_microbatch_keeps_all_anchors_and_matches_full_core_gradients(arm):
    model, _predictor, b, images = setup(arm)
    expanded = {**b, 'inputs': {k: v[:1].repeat(11, *([1]*(v.ndim-1)))
                               for k, v in b['inputs'].items()},
                'proprioception': b['proprioception'][:1].repeat(11, 1)}
    images = images[:1].repeat(11, 1, 1, 1, 1)
    calls = []
    handle = model.future_core.register_forward_hook(lambda m, args, out: calls.append(len(out)))
    result = run(model, expanded, images)
    handle.remove()
    assert calls == [10, 1] and result.future_tokens.shape == (11, 128, 256)
    from eventground_wam.b0_oft import normalize_state
    adapter = model.policy.conditioning_adapter
    condition = PixelCondition(result.features.values, result.features.padding, images,
        normalize_state(expanded['proprioception'], adapter.state_lower, adapter.state_upper),
        result.candidate_actions, torch.full((11,), 8, dtype=torch.int64))
    direct = model.future_core(condition)
    torch.testing.assert_close(result.future_tokens, direct, atol=1e-5, rtol=1e-5)
    derivative = torch.randn_like(direct)
    parameters = tuple(model.future_core.parameters())
    expected = torch.autograd.grad(direct, parameters, derivative, retain_graph=True)
    actual = torch.autograd.grad(result.future_tokens, parameters, derivative)
    error = torch.cat([(a-e).flatten() for a, e in zip(actual, expected, strict=True)])
    reference = torch.cat([e.flatten() for e in expected])
    assert error.norm()/reference.norm() < 1e-5


def test_candidate_and_backward_preserve_inherited_predictor_precision_context():
    model, predictor, b, images = setup('M4')
    observed = []
    handle = model.future_core.register_forward_pre_hook(
        lambda m, args: observed.append((torch.backends.cudnn.enabled, torch.backends.cudnn.allow_tf32)))
    backward = predictor.current_patches.weight.register_hook(
        lambda grad: observed.append((torch.backends.cudnn.enabled, torch.backends.cudnn.allow_tf32)))
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=True):
        result = run(model, b, images, normalized_action_targets=b['targets'])
        future = auxiliary(result, images)
        truth = normalize_predictions(torch.randn(3, 128, 2048))
        coupled_joint_backward(model, result, predictor, future, truth)
        assert torch.backends.cudnn.enabled and torch.backends.cudnn.allow_tf32
    handle.remove()
    backward.remove()
    assert len(observed) >= 3 and all(flags == (False, False) for flags in observed)


def test_bf16_backbone_shares_one_live_fp32_future_boundary():
    model, predictor, b, images = setup('M4')
    model.policy.conditioning_adapter.backbone.bfloat16()
    activate(model)
    result = run(model, b, images, normalized_action_targets=b['targets'])
    assert result.features.values.dtype == torch.float32 and result.features.values.requires_grad
    seen = []
    handle = result.features.values.register_hook(lambda grad: seen.append(grad.dtype))
    future = auxiliary(result, images)
    truth = normalize_predictions(torch.randn(3, 128, 2048))
    coupled_joint_backward(model, result, predictor, future, truth)
    handle.remove()
    assert seen == [torch.float32]
    gradient = model.policy.conditioning_adapter.backbone.lora_scale.grad
    assert gradient.dtype == torch.bfloat16 and torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_zero_coupling_action_queries_also_use_shared_fp32_boundary():
    model, _, b, images = setup('M4')
    model.policy.conditioning_adapter.backbone.bfloat16()
    result = run(model, b, images, normalized_action_targets=b['targets'])
    result.features.values.retain_grad()
    result.action_output.loss.backward()
    # At zero coupling, the candidate branch contributes zero. A nonzero
    # gradient here proves the direct action-query route shares this boundary.
    grad = result.features.values.grad
    assert grad is not None and grad.dtype == torch.float32 and grad.abs().sum() > 0


@pytest.mark.parametrize('arm', ['M3', 'M4'])
@pytest.mark.parametrize('nonzero', [False, True])
def test_full_batch_execution_matches_complete_objective_and_base_precision(arm, nonzero):
    from eventground_wam.coupled_policy import coupled_full_batch_backward, FULL_BATCH_EXECUTION
    model, predictor, b, images = setup(arm)
    model.policy.conditioning_adapter.backbone.bfloat16()
    if nonzero:
        activate(model)
    parameters = [p for g in coupled_parameter_groups(model, predictor, LORA_NAMES) for p in g['params']]
    result = run(model, b, images, normalized_action_targets=b['targets'])
    future = auxiliary(result, images)
    truth = (torch.rand(3, 2, 3, 128, 128) if arm == 'M3' else
             normalize_predictions(torch.randn(3, 128, 2048))).requires_grad_()
    with torch.backends.cudnn.flags(allow_tf32=False):
        predicted = predictor(future.condition(result.features))
        loss = (pixel_reconstruction_loss(predicted.images, truth.detach(), ssim_weight=.1) if arm == 'M3'
                else jepa_prediction_loss(predicted.features, truth.detach(), future.anchor_indices, **LOSS_RULE).total)
        (result.action_output.loss + LOSS_WEIGHTS[arm]*loss).backward()
    expected = [p.grad.clone() for p in parameters]
    model.zero_grad(set_to_none=True)
    predictor.zero_grad(set_to_none=True)
    base_context, future_context = [], []
    handles = [model.policy.conditioning_adapter.backbone.lora_scale.register_hook(
        lambda grad: base_context.append((torch.backends.cudnn.enabled, torch.backends.cudnn.allow_tf32))),
        predictor.current_patches.weight.register_hook(
        lambda grad: future_context.append((torch.backends.cudnn.enabled, torch.backends.cudnn.allow_tf32)))]
    with torch.backends.cudnn.flags(enabled=True, allow_tf32=True):
        actual = run(model, b, images, normalized_action_targets=b['targets'])
        actual.proposal_raw.retain_grad()
        metrics = coupled_full_batch_backward(model, actual, predictor, future, truth)
    for handle in handles:
        handle.remove()
    assert base_context == [(True, True)] and future_context == [(False, False)]
    assert metrics['backward_execution'] == FULL_BATCH_EXECUTION
    assert abs(metrics['future_loss'] - float(loss.detach())) < 1e-6
    assert actual.proposal_raw.grad is None and truth.grad is None
    for parameter, gradient in zip(parameters, expected, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('arm', ['M3', 'M4'])
def test_full_batch_execution_handles_empty_auxiliary_supervision(arm):
    from eventground_wam.coupled_policy import coupled_full_batch_backward
    model, predictor, b, images = setup(arm)
    result = run(model, b, images, normalized_action_targets=b['targets'])
    empty = AuxiliaryInputs(torch.empty(0, dtype=torch.int64), images, torch.zeros(2, 8),
                            torch.empty(0, 32, 7), torch.empty(0, dtype=torch.int64))
    metrics = coupled_full_batch_backward(model, result, predictor, empty, torch.empty(0))
    assert metrics['future_loss'] == 0. and metrics['future_pairs'] == 0
    assert model.coupling.output.weight.grad.abs().sum() > 0
