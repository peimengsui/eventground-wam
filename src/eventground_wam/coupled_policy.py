"""Frozen M3/M4 proposal → future tokens → attention → shared-head refinement."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from eventground_wam.auxiliary_losses import joint_backward
from eventground_wam.auxiliary_policy import ObservationFeatures, policy_with_features
from eventground_wam.b0_data import ActionNormalizer
from eventground_wam.b0_model import (
    B0ActionOutput,
    ConditioningMemory,
    _validate_normalized_action_targets,
    normalized_action_l1_loss,
)
from eventground_wam.b0_oft import (
    OFTActionHead,
    OFTConditioningAdapter,
    marker_positions,
    normalize_state,
)
from eventground_wam.future_core import FutureTokenCore
from eventground_wam.jepa_predictor import JepaPredictor
from eventground_wam.pixel_predictor import PixelCondition, PixelPredictor

INTERFACE_VERSION = 'oft-propose-predict-attend-refine-v1'
DESIGN_DIGEST = '287bba079fd5cbac60577b69e3abbe566553f5fdacf00d2041db2c1b53a04315'
COUPLING_PARAMETERS = 1187072
METHODS = {'M3': 'pixel', 'M4': 'jepa'}
LOSS_WEIGHTS = {'M3': 4.836645309900765, 'M4': 1.5140534185764003}
CANDIDATE_MICROBATCH = 10
FULL_BATCH_EXECUTION = 'coupled-whole-logical-future-batch-v1'


class FutureCrossAttention(nn.Module):
    """Exact 1,187,072-parameter FP32 residual block, with a zero output map."""

    def __init__(self):
        super().__init__()
        # Construct on CPU without disturbing either the CPU or CUDA RNG streams.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(20261005)
            self.query_norm = nn.LayerNorm(2048, eps=1e-5, dtype=torch.float32, device='cpu')
            self.future_norm = nn.LayerNorm(256, eps=1e-5, dtype=torch.float32, device='cpu')
            self.query = nn.Linear(2048, 256, dtype=torch.float32, device='cpu')
            self.key = nn.Linear(256, 256, dtype=torch.float32, device='cpu')
            self.value = nn.Linear(256, 256, dtype=torch.float32, device='cpu')
            self.output = nn.Linear(256, 2048, dtype=torch.float32, device='cpu')
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        assert sum(p.numel() for p in self.parameters()) == COUPLING_PARAMETERS

    def forward(self, action_hidden: Tensor, future_tokens: Tensor):
        batch = len(action_hidden)
        if (batch == 0 or action_hidden.shape != (batch, 8, 2048)
                or future_tokens.shape != (batch, 128, 256)
                or action_hidden.device != future_tokens.device
                or not torch.isfinite(action_hidden).all()
                or not torch.isfinite(future_tokens).all()):
            raise ValueError('coupling requires finite [B,8,2048] queries and [B,128,256] futures')
        if any(p.dtype != torch.float32 for p in self.parameters()):
            raise ValueError('coupling parameters must remain FP32')
        with torch.autocast(device_type=action_hidden.device.type, enabled=False):
            queries = self.query(self.query_norm(action_hidden.float()))
            tokens = self.future_norm(future_tokens.float())
            keys, values = self.key(tokens), self.value(tokens)
            q = queries.reshape(batch, 8, 8, 32).transpose(1, 2)
            k = keys.reshape(batch, 128, 8, 32).transpose(1, 2)
            v = values.reshape(batch, 128, 8, 32).transpose(1, 2)
            weights = ((q @ k.transpose(-1, -2)) * (32 ** -.5)).softmax(dim=-1)
            context = (weights @ v).transpose(1, 2).reshape(batch, 8, 256)
            return action_hidden.float() + self.output(context)


class ExecutableProposal(nn.Module):
    """Detach, decode in simulator units, then return FP32 normalized conditioning.

    The affine boundary uses FP64 to match the existing NumPy action decoder.
    All future-network and coupling arithmetic stays FP32. Clipping normalized
    values directly would give a different physical action under nonidentity stats.
    """

    def __init__(self, normalizer: ActionNormalizer):
        super().__init__()
        self.normalizer = normalizer
        self.register_buffer('center', torch.tensor(normalizer.center, dtype=torch.float64))
        self.register_buffer('scale', torch.tensor(normalizer.scale, dtype=torch.float64))

    def forward(self, raw: Tensor):
        if raw.ndim != 3 or raw.shape[1:] != (8, 7) or not torch.isfinite(raw).all():
            raise ValueError('candidate must be finite [B,8,7]')
        if self.center.dtype != torch.float64 or self.scale.dtype != torch.float64:
            raise ValueError('candidate affine buffers must retain decoder precision')
        with torch.no_grad(), torch.autocast(device_type=raw.device.type, enabled=False):
            values = raw.detach().double()
            motion = (values[..., :6]*self.scale[:6]+self.center[:6]).clamp(-1, 1)
            motion = (motion-self.center[:6])/self.scale[:6]
            gripper = torch.where(values[..., 6:] >= 0, 1., -1.)
            normalized = torch.cat((motion, gripper), dim=-1).float()
            padded = normalized.new_zeros((len(raw), 32, 7))
            padded[:, :8] = normalized
        return padded


@dataclass(frozen=True)
class CoupledForward:
    action_output: B0ActionOutput
    features: ObservationFeatures
    proposal_raw: Tensor
    candidate_actions: Tensor
    future_tokens: Tensor


class CoupledPolicy(nn.Module):
    """Inference owns the base policy, future core, coupling and action statistics.

    No target, teacher, demonstrated-action or availability input is accepted.
    A full predictor passed at construction shares core tensors with this model,
    but its training-only output heads are not registered in the action policy.
    """

    def __init__(self, policy, predictor, normalizer: ActionNormalizer, *, arm: str):
        super().__init__()
        if arm not in METHODS:
            raise ValueError('coupled arm must be M3 or M4')
        if (not isinstance(policy.conditioning_adapter, OFTConditioningAdapter)
                or not isinstance(policy.action_decoder, OFTActionHead)
                or policy.objective_config.objective != 'all-action-l1-binary-gripper-v2'):
            raise ValueError('coupling requires the unchanged OFT all-action-L1 policy')
        if ((arm == 'M4' and not isinstance(predictor, JepaPredictor)) or
                (arm == 'M3' and type(predictor) not in (PixelPredictor, FutureTokenCore))):
            raise ValueError('predictor does not match the coupled arm')
        if (predictor.config.feature_dimension != 2048 or predictor.config.dimension != 256):
            raise ValueError('coupling requires the frozen 2048 → 256 future interface')
        self.arm = arm
        self.policy = policy
        self.future_core = FutureTokenCore(predictor)
        self.coupling = FutureCrossAttention()
        self.proposal = ExecutableProposal(normalizer)

    def forward_with_features(self, backbone_inputs, proprioception, current_images, *,
                              normalized_action_targets=None):
        batch = len(proprioception)
        if not 1 <= batch <= 32 or current_images.shape != (batch, 2, 3, 128, 128):
            raise ValueError('all 1..32 current action anchors require both 128×128 images')
        first, features = policy_with_features(self.policy, backbone_inputs, proprioception)
        # Refinement queries and both future branches share one FP32 boundary.
        # Separate casts sum rounded BF16 adjoints in a different order from
        # the whole-batch objective, even though forward values are identical.
        features = ObservationFeatures(features.values.float(), features.padding)
        positions = marker_positions(backbone_inputs['input_ids'], backbone_inputs['attention_mask'],
                                     self.policy.conditioning_adapter.marker_id)
        queries = features.values.gather(
            1, positions[:, 1:, None].expand(-1, -1, features.values.shape[-1]))
        candidate = self.proposal(first.raw_action_predictions)
        adapter = self.policy.conditioning_adapter
        # State/image tensors are observations, not trainable target proxies.
        state = normalize_state(proprioception.detach(), adapter.state_lower, adapter.state_upper)
        condition = PixelCondition(features.values, features.padding, current_images.detach(),
                                   state, candidate,
                                   torch.full((batch,), 8, dtype=torch.int64, device=state.device))
        if any(p.dtype != torch.float32 for p in self.future_core.parameters()):
            raise ValueError('future-core parameters must remain FP32')
        with torch.autocast(device_type=state.device.type, enabled=False), \
                torch.backends.cudnn.flags(allow_tf32=False):
            # Keep the inherited predictor execution size. A deployed query is
            # one anchor and therefore still makes exactly one future-core call.
            chunks = []
            for start in range(0, batch, CANDIDATE_MICROBATCH):
                stop = min(start+CANDIDATE_MICROBATCH, batch)
                chunk = PixelCondition(**{key: value[start:stop]
                                          for key, value in vars(condition).items()})
                chunks.append(self.future_core(chunk))
            tokens = torch.cat(chunks, dim=0)
            refined = self.coupling(queries, tokens)
        memory = ConditioningMemory(refined, first.conditioning_memory.key_padding_mask)
        raw = self.policy.action_decoder(memory)
        if not torch.isfinite(raw).all():
            raise ValueError('nonfinite refined action prediction')
        ones = torch.ones_like(raw[..., 6:])
        decoded = torch.cat((raw[..., :6], torch.where(raw[..., 6:] >= 0, ones, -ones)), dim=-1)
        loss = continuous = gripper = None
        if normalized_action_targets is not None:
            truth = normalized_action_targets.detach()
            _validate_normalized_action_targets(truth, raw)
            loss = normalized_action_l1_loss(raw, truth)
            continuous = F.l1_loss(raw[..., :6], truth[..., :6])
            gripper = F.l1_loss(raw[..., 6], truth[..., 6])
        output = B0ActionOutput(decoded, memory, loss=loss, raw_action_predictions=raw,
                                continuous_l1_loss=continuous, gripper_l1_loss=gripper)
        return CoupledForward(output, features, first.raw_action_predictions, candidate, tokens)

    def forward(self, backbone_inputs, proprioception, current_images, *,
                normalized_action_targets=None):
        return self.forward_with_features(backbone_inputs, proprioception, current_images,
            normalized_action_targets=normalized_action_targets).action_output


def require_shared_predictor(model: CoupledPolicy, predictor):
    expected = dict(model.future_core.named_parameters())
    actual = dict(predictor.named_parameters())
    if any(actual.get(name) is not parameter for name, parameter in expected.items()):
        raise ValueError('candidate and teacher-forced branches must share the same core tensors')
    if isinstance(predictor, JepaPredictor) != (model.arm == 'M4'):
        raise ValueError('teacher-forced predictor method differs from coupled arm')
    if type(predictor) not in (PixelPredictor, JepaPredictor):
        raise ValueError('teacher-forced branch requires its complete training output head')


def coupled_joint_backward(model, result, predictor, future_inputs, future_targets, *,
                           microbatch=10, hook=None):
    """Final-action loss plus the unchanged, separate teacher-forced auxiliary loss.

    Existing whole-batch JEPA variance/replay and hidden-state VJP are reused.
    The candidate-conditioned prediction is never compared with demonstrated futures.
    Caller zeroes gradients first, then jointly clips/steps once per logical batch.
    """
    require_shared_predictor(model, predictor)
    with torch.autocast(device_type=result.features.values.device.type, enabled=False), \
            torch.backends.cudnn.flags(allow_tf32=False):
        return joint_backward(result.action_output, result.features, predictor, METHODS[model.arm],
                              future_inputs, future_targets, microbatch=microbatch,
                              loss_weight=LOSS_WEIGHTS[model.arm], hook=hook)


def coupled_full_batch_backward(model, result, predictor, future_inputs, future_targets, *,
                                hook=None):
    """Differentiate the complete logical objective before entering the BF16 base.

    Full-batch execution avoids small microbatch accumulation differences that
    the BF16 recurrent backbone can amplify. It requires separate actual-device
    memory/time admission. Candidate conditioning still uses microbatches of ten.
    """
    from eventground_wam.jepa_predictor import jepa_prediction_loss
    from eventground_wam.jepa_training import LOSS_RULE
    from eventground_wam.pixel_predictor import pixel_reconstruction_loss

    require_shared_predictor(model, predictor)
    count = len(future_inputs.horizons)
    if count != len(future_targets):
        raise ValueError('target and future-pair cardinalities differ')
    action_loss = result.action_output.loss
    if action_loss is None or not torch.isfinite(action_loss):
        raise ValueError('joint training requires a finite action loss')
    parameters = (list(model.policy.action_decoder.parameters())
                  + list(model.coupling.parameters())
                  + list((predictor if count else model.future_core).parameters()))
    if len(parameters) != len(set(map(id, parameters))):
        raise ValueError('above-boundary parameters must be disjoint')
    with torch.autocast(device_type=result.features.values.device.type, enabled=False), \
            torch.backends.cudnn.flags(allow_tf32=False):
        metrics = {'future_loss': 0., 'future_pairs': count}
        total = action_loss
        if count:
            prediction = predictor(future_inputs.condition(result.features))
            truth = future_targets.detach()
            if model.arm == 'M3':
                loss = pixel_reconstruction_loss(prediction.images, truth, ssim_weight=.1)
            else:
                components = jepa_prediction_loss(prediction.features, truth,
                    future_inputs.anchor_indices, **LOSS_RULE)
                loss = components.total
                metrics.update(cosine=float(components.cosine.detach()),
                    variance=float(components.variance.detach()),
                    independent_anchors=components.independent_anchors)
            metrics['future_loss'] = float(loss.detach())
            total = total + LOSS_WEIGHTS[model.arm]*loss
        if not torch.isfinite(total):
            raise ValueError('nonfinite coupled objective')
        gradients = torch.autograd.grad(total, [result.features.values, *parameters])
        for parameter, gradient in zip(parameters, gradients[1:], strict=True):
            if not torch.isfinite(gradient).all():
                raise ValueError('nonfinite coupled gradient')
            if parameter.grad is None:
                parameter.grad = gradient.detach()
            else:
                parameter.grad.add_(gradient.detach())
    # Backpropagate into the base only after restoring its inherited context.
    # Checkpoint recomputation must use the same context as its original forward.
    derivative = gradients[0].detach()
    if not torch.isfinite(derivative).all():
        raise ValueError('nonfinite shared-feature gradient')
    if hook is not None:
        hook(0, count)
    result.features.values.backward(derivative)
    return {**metrics, 'action_loss': float(action_loss.detach()),
            'joint_loss': float(total.detach()),
            'joint_feature_gradient_norm': float(derivative.norm()),
            'backward_execution': FULL_BATCH_EXECUTION}


def coupled_parameter_groups(model, predictor, exact_lora_names):
    """Four disjoint groups; include the shared future core exactly once."""
    from eventground_wam.b0_model import validate_b0_trainable_parameters

    require_shared_predictor(model, predictor)
    names = validate_b0_trainable_parameters(model.policy, allowed_exact_names=exact_lora_names)
    policy_parameters = dict(model.policy.named_parameters())
    lora = set(exact_lora_names)
    groups = [
        {'name': 'head_and_state', 'params': [policy_parameters[n] for n in names if n not in lora]},
        {'name': 'lora', 'params': [policy_parameters[n] for n in names if n in lora]},
        {'name': 'future', 'params': list(predictor.parameters())},
        {'name': 'coupling', 'params': list(model.coupling.parameters())},
    ]
    parameters = [p for group in groups for p in group['params']]
    if (any(not group['params'] for group in groups)
            or len(set(map(id, parameters))) != len(parameters)
            or not all(p.requires_grad for p in parameters)):
        raise ValueError('optimizer groups must be nonempty, trainable and disjoint')
    return [{**group, 'lr': 1e-4} for group in groups]
