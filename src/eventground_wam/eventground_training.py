"""Logical-batch EventGround updates and identity-bound recovery checkpoints."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from eventground_wam.b0_training import (
    atomic_torch_save,
    capture_rng,
    cpu_tree,
    read_sealed,
    restore_rng,
    sha256_file,
)
from eventground_wam.demonstration_contracts import contract_digest
from eventground_wam.eventground_objectives import grpo_loss, structured_sft_loss

CHECKPOINT_FORMAT = "eventground-week10-optimizer-recovery-v1"


def exact_trainable_names(model):
    return tuple(n for n, p in model.named_parameters() if p.requires_grad)


def require_training_admission(admission, *, study_digest, recipe_digest, manifest_digest):
    """Diagnostic manifests, stale reviews and resource probes cannot launch SFT."""
    if (admission.get("digest") != contract_digest(admission)
            or admission.get("artifact_type") != "eventground_wam.week10_training_admission"
            or admission.get("status") != "passed"
            or admission.get("study_digest") != study_digest
            or admission.get("recipe_digest") != recipe_digest
            or admission.get("manifest_digest") != manifest_digest
            or admission.get("labels_admitted") is not True
            or admission.get("human_review_gate_passed") is not True
            or admission.get("correctness_passed") is not True
            or admission.get("actual_host_resources_passed") is not True
            or admission.get("formal_launch_approved") is not True):
        raise ValueError("Formal training requires linked label, numerical, resource and launch gates")
    for key in ("qa_digest", "qualification_digest", "launch_authorization_digest"):
        value = admission.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Missing linked admission evidence: {key}")


class EventGroundTrainer:
    """No hidden data loading: callers provide causal inputs and separate offline targets.

    Every microbatch contributes its sum divided by the complete logical batch.
    Group-of-four sampling and advantages are never split across prompts.
    """

    def __init__(self, model, allowlist, recipe, *, stage="sft"):
        if recipe.get("digest") != contract_digest(recipe) or stage not in ("sft", "rlvr", "control"):
            raise ValueError("A sealed numerical recipe and declared stage are required")
        self.model, self.recipe, self.stage = model, copy.deepcopy(recipe), stage
        self.allowlist = {k: tuple(v) for k, v in allowlist.items()}
        self.names = tuple(sorted((*allowlist["head"], *allowlist["language_lora"])))
        if (len(self.names) != len(set(self.names))
                or set(exact_trainable_names(model)) != set(self.names)):
            raise ValueError("Trainable parameter set differs from exact allowlist")
        parameters = dict(model.named_parameters())
        numerical = recipe["optimization"]
        rates = numerical["rlvr_learning_rates" if stage == "rlvr" else "sft_learning_rates"]
        self.optimizer = torch.optim.AdamW([
            {"params": [parameters[n] for n in allowlist[group]], "lr": rates[group],
             "group_name": group}
            for group in ("head", "language_lora")
        ], betas=tuple(numerical["betas"]), eps=numerical["epsilon"],
            weight_decay=numerical["weight_decay"])
        self.global_step = self.prompts_seen = 0
        self._in_update = False

    def _begin(self, prompt_count):
        if (self._in_update or type(prompt_count) is not int
                or not 0 < prompt_count <= self.recipe["logical_batch_size"]
                or set(exact_trainable_names(self.model)) != set(self.names)):
            raise ValueError("Partial failed update, logical batch or parameter allowlist changed")
        self._in_update = True
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

    def _finish(self, prompt_count):
        parameters = dict(self.model.named_parameters())
        for name, parameter in parameters.items():
            if name not in self.names and parameter.grad is not None:
                raise ValueError(f"Frozen parameter received a gradient: {name}")
            if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
                raise FloatingPointError(f"Nonfinite gradient: {name}")
        if any(parameters[n].grad is None for n in self.names):
            raise ValueError("An allowlisted parameter is detached from the loss")
        norm = torch.nn.utils.clip_grad_norm_(
            [parameters[n] for n in self.names],
            self.recipe["optimization"]["maximum_gradient_norm"], error_if_nonfinite=True,
        )
        self.optimizer.step()
        if not all(torch.isfinite(parameters[n]).all() for n in self.names):
            raise FloatingPointError("Nonfinite parameter after update")
        self.global_step += 1
        self.prompts_seen += prompt_count
        self._in_update = False
        return float(norm)

    def sft_update(self, microbatches, *, prompt_count):
        """Each entry is (backbone encoding, current proprioception, target tokens)."""
        if self.stage not in ("sft", "control"):
            raise ValueError("SFT update in the wrong stage")
        batches = list(microbatches)
        if sum(len(tokens) for _, _, tokens in batches) != prompt_count:
            raise ValueError("Incomplete/duplicated logical batch")
        self._begin(prompt_count)
        total = 0.
        for encoded, state, tokens in batches:
            scores = self.model.score(encoded, state, tokens)
            loss = structured_sft_loss(scores, logical_batch_size=prompt_count)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite SFT loss")
            loss.backward()
            total += float(loss.detach())
        norm = self._finish(prompt_count)
        return {"loss": total, "gradient_norm": norm, "global_step": self.global_step,
                "prompts_seen": self.prompts_seen}

    def rlvr_update(self, microbatches, *, prompt_count, reference_model, reward_function):
        """Each entry is (encoding, current state, independent offline reward evidence).

        reward_function(evidence, sampled_tokens) returns [prompts,4] rewards.
        The caller must verify the fixed SFT reference identity before constructing
        this stage. A reference may never share tensors or trainable parameters.
        """
        if self.stage != "rlvr" or reference_model is self.model:
            raise ValueError("RLVR requires its own stage and a separate frozen reference")
        current_params = {p.data_ptr() for p in self.model.parameters()}
        if any(p.requires_grad or p.data_ptr() in current_params for p in reference_model.parameters()):
            raise ValueError("Reference must be independently frozen")
        batches = list(microbatches)
        if sum(len(state) for _, state, _ in batches) != prompt_count:
            raise ValueError("Incomplete/duplicated logical batch")
        self._begin(prompt_count)
        reference_model.eval()
        numerical = self.recipe["grpo"]
        total = policy_loss = kl = 0.
        equal = 0
        for encoded, state, evidence in batches:
            features = self.model(encoded, state)
            with torch.no_grad():
                sampled = self.model.head.sample(
                    features.detach(), group_size=4, temperature=numerical["temperature"],
                )
                rewards = reward_function(evidence, sampled.tokens).to(features.device)
                reference_features = reference_model(encoded, state)
                reference = reference_model.head.teacher_forced(
                    reference_features.repeat_interleave(4, 0), sampled.tokens,
                    temperature=numerical["temperature"],
                )
            current = self.model.head.teacher_forced(
                features.repeat_interleave(4, 0), sampled.tokens,
                temperature=numerical["temperature"],
            )
            result = grpo_loss(
                current.log_probs.reshape(-1, 4, 3), sampled.log_probs.reshape(-1, 4, 3),
                reference.log_probs.reshape(-1, 4, 3), current.active.reshape(-1, 4, 3), rewards,
                clip_epsilon=numerical["clip_epsilon"], kl_beta=numerical["kl_beta"],
                advantage_epsilon=numerical["advantage_epsilon"],
                logical_prompt_count=prompt_count,
            )
            if not torch.isfinite(result.total):
                raise FloatingPointError("Nonfinite RLVR loss")
            result.total.backward()
            total += float(result.total.detach())
            policy_loss += float(result.policy.detach())
            kl += float(result.kl.detach())
            equal += result.equal_reward_groups
        norm = self._finish(prompt_count)
        return {"loss": total, "policy_loss": policy_loss, "kl": kl,
                "equal_reward_groups": equal, "gradient_norm": norm,
                "global_step": self.global_step, "prompts_seen": self.prompts_seen}

    def save(self, path: Path, *, identity):
        path = Path(path)
        if self._in_update:
            raise ValueError("Cannot save a partial update; restore a verified checkpoint first")
        if path.exists() or identity.get("digest") != contract_digest(identity):
            raise ValueError("Checkpoint must have a fresh path and sealed identity")
        parameters = dict(self.model.named_parameters())
        state = {n: parameters[n].detach().cpu().clone() for n in self.names}
        payload = {"format": CHECKPOINT_FORMAT, "identity": identity,
                   "recipe_digest": self.recipe["digest"], "stage": self.stage,
                   "exact_names": self.names, "parameters": state,
                   "optimizer": cpu_tree(self.optimizer.state_dict()), "rng": capture_rng(),
                   "global_step": self.global_step, "prompts_seen": self.prompts_seen}
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_torch_save(path, payload)
        with path.with_suffix(path.suffix + ".sha256").open("x") as stream:
            stream.write(sha256_file(path) + "\n")

    def resume(self, path: Path, *, identity):
        path = Path(path)
        if sha256_file(path) != path.with_suffix(path.suffix + ".sha256").read_text().strip():
            raise ValueError("Recovery checkpoint checksum differs")
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (saved["format"] != CHECKPOINT_FORMAT or saved["identity"] != identity
                or identity.get("digest") != contract_digest(identity)
                or saved["recipe_digest"] != self.recipe["digest"] or saved["stage"] != self.stage
                or tuple(saved["exact_names"]) != self.names
                or set(saved["parameters"]) != set(self.names)):
            raise ValueError("Recovery identity, stage, recipe or tensor set differs")
        parameters = dict(self.model.named_parameters())
        for name, value in saved["parameters"].items():
            if (value.shape != parameters[name].shape or value.dtype != parameters[name].dtype
                    or not torch.isfinite(value).all()):
                raise ValueError(f"Recovery tensor invalid: {name}")
        for key in ("global_step", "prompts_seen"):
            if type(saved[key]) is not int or saved[key] < 0:
                raise ValueError("Recovery counters must be nonnegative integers")
        if not saved["global_step"] <= saved["prompts_seen"] <= (
                saved["global_step"] * self.recipe["logical_batch_size"]):
            raise ValueError("Recovery exposure counters disagree")
        expected_optimizer = self.optimizer.state_dict()
        if len(saved["optimizer"]["param_groups"]) != len(expected_optimizer["param_groups"]):
            raise ValueError("Recovery optimizer groups changed")
        ordered_parameters = []
        for actual, expected, live in zip(
            saved["optimizer"]["param_groups"], expected_optimizer["param_groups"],
            self.optimizer.param_groups, strict=True,
        ):
            if actual != expected:
                raise ValueError("Recovery optimizer recipe or parameter order changed")
            ordered_parameters.extend(zip(actual["params"], live["params"], strict=True))
        expected_ids = {key for key, _ in ordered_parameters}
        states = saved["optimizer"]["state"]
        if set(states) != (expected_ids if saved["global_step"] else set()):
            raise ValueError("Recovery optimizer moment coverage differs")
        for key, parameter in ordered_parameters:
            if not saved["global_step"]:
                continue
            state = states[key]
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Recovery optimizer fields differ")
            step = state["step"]
            if not isinstance(step, torch.Tensor) or step.numel() != 1 or float(step) != saved["global_step"]:
                raise ValueError("Recovery optimizer step disagrees")
            for moment in (state["exp_avg"], state["exp_avg_sq"]):
                if (moment.shape != parameter.shape or moment.dtype != parameter.dtype
                        or not torch.isfinite(moment).all()):
                    raise ValueError("Recovery optimizer moments malformed")
            if (state["exp_avg_sq"] < 0).any():
                raise ValueError("Negative optimizer second moment")
        original_rng = capture_rng()
        try:
            restore_rng(saved["rng"])
        except Exception as exc:
            raise ValueError("Recovery random state is malformed") from exc
        finally:
            restore_rng(original_rng)
        # Load optimizer state into a disposable optimizer first, so malformed
        # groups fail before the model's parameter values are changed.
        probe = copy.deepcopy(self.optimizer)
        probe.load_state_dict(saved["optimizer"])
        del probe
        with torch.no_grad():
            for name, value in saved["parameters"].items():
                parameters[name].copy_(value.to(parameters[name].device))
        self.optimizer.load_state_dict(saved["optimizer"])
        self.global_step, self.prompts_seen = saved["global_step"], saved["prompts_seen"]
        restore_rng(saved["rng"])
        self._in_update = False


def load_admitted_manifest(path, admission_path, recipe):
    manifest, admission = read_sealed(Path(path)), read_sealed(Path(admission_path))
    require_training_admission(admission, study_digest=recipe["study_digest"],
                               recipe_digest=recipe["digest"], manifest_digest=manifest["digest"])
    if manifest.get("labels_admitted") is not True:
        raise ValueError("Unadmitted diagnostic prompts cannot train a model")
    return manifest, admission
