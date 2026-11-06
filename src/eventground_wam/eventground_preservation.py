"""A separately versioned GRPO + supervised-preservation pilot.

Both arms use the original RLVR optimizer rates. Each component is normalized
by the actual logical batch, including the last partial batch. The supervised
targets remain separate from the causal model inputs.
"""
import math
import statistics

import torch

from eventground_wam.eventground_objectives import grpo_loss, structured_sft_loss
from eventground_wam.eventground_training import EventGroundTrainer


def preservation_weight(grpo_norms, sft_norms):
    """Prospective rule: supervised gradient is 25% of the GRPO gradient.

    Two fixed training-only batches, no optimizer updates or dev metrics.
    Clamp to [0.01, 1.0]; never repeat this diagnostic to select another value.
    """
    if len(grpo_norms) != 2 or len(sft_norms) != 2:
        raise ValueError("Exactly two fixed diagnostic batches required")
    if any(not math.isfinite(x) or x <= 0 for x in [*grpo_norms, *sft_norms]):
        raise ValueError("Nonpositive or nonfinite diagnostic gradient")
    return max(.01, min(1., .25 * statistics.median(grpo_norms) / statistics.median(sft_norms)))


class PreservationTrainer(EventGroundTrainer):
    def __init__(self, model, allowlist, recipe):
        super().__init__(model, allowlist, recipe, stage="rlvr")

    def _accumulate(self, batches, *, prompt_count, mode, weight, reference_model, reward_function):
        if mode not in ("hybrid", "control", "rlvr") or not math.isfinite(weight) or weight < 0:
            raise ValueError("Invalid preservation objective")
        if sum(len(state) for _, state, _, _ in batches) != prompt_count:
            raise ValueError("Incomplete/duplicated logical batch")
        if any(len(tokens) != len(state) or len(evidence) != len(state)
               for _, state, evidence, tokens in batches):
            raise ValueError("Unpaired supervised targets and evidence")
        if mode != "control":
            if reference_model is None or reference_model is self.model:
                raise ValueError("Separate frozen reference required")
            pointers = {p.data_ptr() for p in self.model.parameters()}
            if any(p.requires_grad or p.data_ptr() in pointers for p in reference_model.parameters()):
                raise ValueError("Reference must be independently frozen")
            reference_model.eval()
        self._begin(prompt_count)
        totals = dict(loss=0., policy_loss=0., kl=0., sft_loss=0., equal_reward_groups=0)
        numerical = self.recipe["grpo"]
        for encoded, state, evidence, tokens in batches:
            features = self.model(encoded, state)
            supervised = structured_sft_loss(self.model.head.teacher_forced(features, tokens),
                                             logical_batch_size=prompt_count)
            if mode == "control":
                loss = supervised
            else:
                with torch.no_grad():
                    sampled = self.model.head.sample(features.detach(), group_size=4,
                                                     temperature=numerical["temperature"])
                    rewards = reward_function(evidence, sampled.tokens).to(features.device)
                    reference = reference_model.head.teacher_forced(
                        reference_model(encoded, state).repeat_interleave(4, 0), sampled.tokens,
                        temperature=numerical["temperature"])
                current = self.model.head.teacher_forced(features.repeat_interleave(4, 0), sampled.tokens,
                                                        temperature=numerical["temperature"])
                result = grpo_loss(
                    current.log_probs.reshape(-1, 4, 3), sampled.log_probs.reshape(-1, 4, 3),
                    reference.log_probs.reshape(-1, 4, 3), current.active.reshape(-1, 4, 3), rewards,
                    clip_epsilon=numerical["clip_epsilon"], kl_beta=numerical["kl_beta"],
                    advantage_epsilon=numerical["advantage_epsilon"], logical_prompt_count=prompt_count)
                loss = result.total + (weight * supervised if mode == "hybrid" else 0.)
                totals["policy_loss"] += float(result.policy.detach())
                totals["kl"] += float(result.kl.detach())
                totals["equal_reward_groups"] += result.equal_reward_groups
            if not torch.isfinite(loss) or not torch.isfinite(supervised):
                raise FloatingPointError("Nonfinite preservation loss")
            loss.backward()
            totals["loss"] += float(loss.detach())
            totals["sft_loss"] += float(supervised.detach())
        return totals

    def update(self, batches, *, prompt_count, mode, weight=0., reference_model=None, reward_function=None):
        totals = self._accumulate(list(batches), prompt_count=prompt_count, mode=mode, weight=weight,
                                  reference_model=reference_model, reward_function=reward_function)
        norm = self._finish(prompt_count)
        return dict(totals, gradient_norm=norm, global_step=self.global_step, prompts_seen=self.prompts_seen)

    def gradient_diagnostic(self, batches, *, prompt_count, mode, reference_model=None, reward_function=None):
        if self.optimizer.state or self.global_step or self.prompts_seen or mode not in ("rlvr", "control"):
            raise ValueError("Diagnostic requires fresh optimizer and a single component")
        totals = self._accumulate(list(batches), prompt_count=prompt_count, mode=mode, weight=0.,
                                  reference_model=reference_model, reward_function=reward_function)
        params = dict(self.model.named_parameters())
        if any(params[n].grad is None for n in self.names):
            raise ValueError("Detached diagnostic parameter")
        if any(p.grad is not None for n, p in params.items() if n not in self.names):
            raise ValueError("Frozen diagnostic parameter has a gradient")
        norm = math.sqrt(sum(float(params[n].grad.detach().double().square().sum()) for n in self.names))
        if not math.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid diagnostic gradient norm")
        self.optimizer.zero_grad(set_to_none=True)
        self._in_update = False
        return dict(totals, gradient_norm=norm, optimizer_updates=0)
