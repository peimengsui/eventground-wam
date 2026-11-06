"""Week 11 composition: isolated FP32 event branch, unchanged M4 action path.

The scoped pre-hook replaces only PixelCondition.horizons. Calling the original
inference adapter retains its proposal, precision boundaries, refinement and
decoder verbatim. Hooks are removed on every exit, including failed queries.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from threading import Lock

import numpy as np
import torch

from eventground_wam.eventground_data import CurrentObservation, collate_current_observations
from eventground_wam.eventground_horizon import SUPPORTED_HORIZONS, select_horizon
from eventground_wam.eventground_model import serialize_tokens

ARMS = ("M4-fixed", "M4-EG-fixed8", "M4-EG-adaptive")


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def routed_horizon(model, horizon, record, *, device, capture=False):
    """One inference query; no parameter replacement or adapter switching."""
    if type(horizon) is not int or horizon not in SUPPORTED_HORIZONS:
        raise ValueError("Unsupported deployment horizon")
    handles = []
    calls = 0
    starts = {}
    record["component_ms"] = {}

    def begin(name):
        def hook(_module, _args):
            synchronize(device)
            starts[name] = time.perf_counter()
        return hook

    def end(name):
        def hook(_module, _args, output):
            synchronize(device)
            record["component_ms"][name] = (time.perf_counter() - starts[name]) * 1000
            if name == "action_forward" and capture:
                record["normalized_actions"] = output.normalized_actions.detach().float().cpu().clone()
        return hook

    def route(_module, args):
        nonlocal calls
        calls += 1
        condition, = args
        if (calls != 1 or condition.horizons.shape != (1,)
                or condition.horizons.item() != 8
                or condition.actions.shape != (1, 32, 7)
                or torch.count_nonzero(condition.actions[:, 8:]).item() != 0):
            raise ValueError("Frozen one-query proposal/core contract changed")
        condition = replace(condition, horizons=torch.full_like(condition.horizons, horizon))
        record["used_horizon"] = int(condition.horizons.item())
        record["future_calls"] = calls
        record["proposal_actions"] = 8
        record["zero_padded_actions"] = 24
        if capture:
            record["condition"] = condition
        return (condition,)

    try:
        handles.append(model.future_core.register_forward_pre_hook(route))
        for name, module in (("action_forward", model), ("future_core", model.future_core),
                             ("refinement_attention", model.coupling)):
            handles.append(module.register_forward_pre_hook(begin(name)))
            handles.append(module.register_forward_hook(end(name)))
        yield record
        if calls != 1:
            raise ValueError("Exactly one future-core call is required")
    finally:
        for handle in handles:
            handle.remove()


class EventBranch:
    def __init__(self, model, builder, device):
        self.model, self.builder, self.device = model, builder, device
        if getattr(builder, "eventground_compute_dtype", None) != torch.float32:
            raise ValueError("Selected event recipe must remain FP32")
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("Event model must remain FP32")
        model.requires_grad_(False).eval()

    def predict(self, observation):
        synchronize(self.device)
        started = time.perf_counter()
        encoded, state = collate_current_observations([observation], self.builder, self.device)
        synchronize(self.device)
        prepared = time.perf_counter()
        # Match the pinned evaluator's no_grad + greedy categorical path exactly.
        with torch.no_grad():
            features = self.model(encoded, state)
            scores = self.model.head.sample(features, group_size=1, greedy=True)
            prediction = serialize_tokens(scores.tokens)[0]
            probabilities = {}
            for name in ("event", "start", "end"):
                values = getattr(scores, name + "_logits").softmax(-1)
                if not torch.isfinite(values).all():
                    raise ValueError("Nonfinite event probabilities")
                probabilities[name] = values[0].detach().cpu().tolist()
        synchronize(self.device)
        return prediction, {"categorical_probabilities": probabilities,
                            "event_preprocess_ms": (prepared-started)*1000,
                            "event_forward_and_decode_ms": (time.perf_counter()-prepared)*1000}


def require_disjoint_models(action_model, event_model):
    action = {p.untyped_storage().data_ptr() for p in action_model.parameters()}
    event = {p.untyped_storage().data_ptr() for p in event_model.parameters()}
    if action & event:
        raise ValueError("Action and event branches share parameter storage")


class EventGroundInferencePolicy:
    """Recorder interface accepting only current cameras, state and instruction."""
    def __init__(self, action_policy, arm, event_branch=None, *, identities=None, guard=None):
        if arm not in ARMS or (event_branch is None) != (arm == "M4-fixed"):
            raise ValueError("Wrong event branch for deployment arm")
        if event_branch is not None:
            require_disjoint_models(action_policy._policy, event_branch.model)
        self.action_policy, self.arm, self.event_branch = action_policy, arm, event_branch
        self.identities, self.guard = identities or {}, guard or (lambda: None)
        self.checkpoint = action_policy.checkpoint
        self.last_query = None
        self.capture = False
        self.capture_record = None
        self._lock = Lock()

    def predict_chunk(self, *, agent_rgb, wrist_rgb, proprioception, instruction, suite):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Concurrent policy queries are unsupported")
        record = {"arm": self.arm, "branch_identities": self.identities,
                  "prediction": None, "decision": None, "used_horizon": None,
                  "event": None, "exception": None, "future_calls": 0,
                  "execute_actions": 8}
        self.last_query = record
        started = time.perf_counter()
        try:
            self.guard()
            observation = CurrentObservation(agent_rgb, wrist_rgb,
                np.asarray(proprioception, dtype=np.float32), instruction)
            if not isinstance(suite, str) or not suite:
                raise ValueError("suite must be a nonempty string")
            horizon = 8
            if self.event_branch is not None:
                prediction, details = self.event_branch.predict(observation)
                record.update(prediction=prediction, event=details)
                decision = select_horizon(prediction)
                record["decision"] = asdict(decision)
                if self.arm == "M4-EG-adaptive":
                    horizon = decision.selected_horizon
            synchronize(self.action_policy._device)
            tick = time.perf_counter()
            with routed_horizon(self.action_policy._policy, horizon, record,
                                device=self.action_policy._device, capture=self.capture):
                actions = self.action_policy.predict_chunk(agent_rgb=agent_rgb, wrist_rgb=wrist_rgb,
                    proprioception=proprioception, instruction=instruction, suite=suite)
            synchronize(self.action_policy._device)
            record["action_branch_ms"] = (time.perf_counter()-tick)*1000
            record["action_preprocess_and_cpu_decode_ms"] = (
                record["action_branch_ms"]-record["component_ms"]["action_forward"])
            if actions.shape != (8, 7) or not np.isfinite(actions).all():
                raise ValueError("Nonfinite or malformed decoded action chunk")
            self.guard()
            return actions
        except BaseException as exc:
            record["exception"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["wrapper_wall_ms"] = (time.perf_counter()-started)*1000
            if self.capture:
                self.capture_record = {k: record.pop(k) for k in ("condition", "normalized_actions") if k in record}
            self._lock.release()
