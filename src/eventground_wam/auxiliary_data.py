"""Attach valid future pairs to the original action batches without filtering anchors."""
from __future__ import annotations

import hashlib
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import Dataset

from eventground_wam.auxiliary_policy import AuxiliaryInputs
from eventground_wam.b0_data import B0Sample, B0SampleIdentity, collate_b0_samples
from eventground_wam.b0_model import load_pinned_qwen_processor
from eventground_wam.b0_oft import OFTInputBuilder, normalize_state
from eventground_wam.continuation_freeze import batches_digest
from eventground_wam.demonstration_contracts import canonical_json_bytes
from eventground_wam.future_pairs import HORIZONS
from eventground_wam.jepa_data import JepaPairDataset
from eventground_wam.pixel_baselines import box_rgb


def future_references(tokens, sources, *, split="train"):
    """Action order is untouched; future rows are anchor-major then horizon-major."""
    if split not in {"train", "validation"}:
        raise ValueError("test payloads are locked")
    if not tokens or len(tokens) > 32 or len(set(tokens)) != len(tokens):
        raise ValueError("invalid logical action batch")
    mask = np.zeros((len(tokens), len(HORIZONS)), dtype=bool)
    references = []
    for index, (_, episode, anchor) in enumerate(tokens):
        source = sources.get(episode)
        if (source is None or source["split"] != split or type(anchor) is not int
                or not 0 <= anchor <= source["episode_length"]-8):
            raise ValueError("action anchor is outside its frozen split or eight-action interval")
        for column, horizon in enumerate(HORIZONS):
            if anchor+horizon < source["episode_length"]:
                mask[index, column] = True
                references.append((index, episode, anchor, horizon))
    return references, mask


class AuxiliaryPairDataset(JepaPairDataset):
    """Reuse canonical source/index validation with an explicit decoded-frame LRU.

    Action mixing spans more than the CPT four-episode pool. This version bounds
    the cache by total frames, retaining exactly the same decode/target bytes.
    """
    def __init__(self, *args, maximum_cache_frames=8192, **kwargs):
        super().__init__(*args, cache_episodes=4, **kwargs)
        if type(maximum_cache_frames) is not int or not 1 <= maximum_cache_frames <= 8192:
            raise ValueError("decoded-frame cache must be bounded to 1..8192 frames")
        self.maximum_cache_frames = maximum_cache_frames
        # A nonempty episode contains at least one frame; the frame bound below
        # is stricter than this count bound in the inherited loader.
        self.cache_episodes = maximum_cache_frames
        self.peak_cached_frames = 0

    def _episode(self, source):
        if source["episode_length"] > self.maximum_cache_frames:
            raise ValueError("one episode exceeds the decoded-frame budget")
        if source["episode_index"] not in self._cache:
            while self._cache and sum(len(v[0]) for v in self._cache.values()) + source[
                    "episode_length"] > self.maximum_cache_frames:
                self._cache.popitem(last=False)
        value = super()._episode(source)
        self.peak_cached_frames = max(self.peak_cached_frames,
                                     sum(len(v[0]) for v in self._cache.values()))
        return value


class AuxiliaryBatches(Dataset):
    def __init__(self, action_dataset, pairs, plan, cache_dir, *, expected_epoch=0, builder=None):
        if action_dataset.config.camera_orientation != "identity":
            raise ValueError("auxiliary data requires the frozen training image orientation")
        if action_dataset.config.split != pairs.split:
            raise ValueError("action and future split differ")
        self.action_dataset, self.pairs, self.plan = action_dataset, pairs, plan
        self.cache_dir, self.expected_epoch, self.builder = str(cache_dir), expected_epoch, builder
        self.assignments = {s.episode_index: s for s in action_dataset.split_index.assignments}
        self.episodes = {e.episode_index: e for e in action_dataset.episodes}
        if set(self.episodes) != set(pairs.sources):
            raise ValueError("action and future episode sets differ")
        for episode, meta in self.episodes.items():
            source = pairs.sources[episode]
            if (meta.length != source["episode_length"] or meta.instruction != source["instruction"]
                    or meta.task_index != source["task_index"]):
                raise ValueError("action and future source metadata differ")

    def __len__(self):
        return len(self.plan)

    def __getitem__(self, index):
        tokens = self.plan[index]
        if any(epoch != self.expected_epoch for epoch, _, _ in tokens):
            raise ValueError("action batch epoch changed")
        refs, mask = future_references(tokens, self.pairs.sources, split=self.pairs.split)
        action_samples, future_samples = {}, {}
        # Decode an episode once per batch even if a batch exceeds the LRU size.
        for episode in dict.fromkeys(episode for _, episode, _ in tokens):
            source = self.pairs.sources[episode]
            actions, states, videos = self.pairs._episode(source)
            assignment = self.assignments[episode]
            for slot, (_, ep, anchor) in enumerate(tokens):
                if ep != episode:
                    continue
                action_samples[slot] = B0Sample(
                    B0SampleIdentity(self.pairs.split, ep, source["task_index"], anchor,
                                     anchor+8, assignment.digest),
                    source["instruction"], videos[anchor, 0].transpose(2, 0, 1).copy(),
                    videos[anchor, 1].transpose(2, 0, 1).copy(), states[anchor].copy(),
                    self.pairs.normalizer.normalize(actions[anchor:anchor+8]).astype(np.float32))
            for pos, (_, ep, anchor, horizon) in enumerate(refs):
                if ep == episode:
                    future_samples[pos] = self.pairs[self.pairs.ordinal(ep, anchor, horizon)]
        action = collate_b0_samples([action_samples[i] for i in range(len(tokens))])
        samples = [future_samples[i] for i in range(len(refs))]
        if self.builder is None:
            self.builder = OFTInputBuilder(load_pinned_qwen_processor(
                cache_dir=self.cache_dir, local_files_only=True))
        encoding = self.builder(action.instructions, list(action.front_rgb), list(action.wrist_rgb))
        current = torch.from_numpy(np.stack([
            np.stack([box_rgb(frame.transpose(1, 2, 0)) for frame in (front, wrist)])
            for front, wrist in zip(action.front_rgb, action.wrist_rgb, strict=True)]))
        state = torch.from_numpy(action.proprioception)
        bounds = self.pairs.state_bounds
        normalized_state = normalize_state(state, torch.tensor(bounds["lower_q01"]),
                                            torch.tensor(bounds["upper_q99"]))
        future = AuxiliaryInputs(
            torch.tensor([r[0] for r in refs], dtype=torch.int64), current, normalized_state,
            torch.stack([s.observation.actions for s in samples]),
            torch.tensor([r[3] for r in refs], dtype=torch.int64))
        return {
            "index": index, "tokens": tokens,
            "inputs": {k: v.to(torch.bfloat16) if v.is_floating_point() else v
                       for k, v in encoding.items()},
            "proprioception": state, "targets": torch.from_numpy(action.actions),
            "sample_count": len(tokens), "unique_task_count": len(set(action.instructions)),
            "token_digest": batches_digest([tokens]),
            "identity_digest": hashlib.sha256(canonical_json_bytes(
                [tokens, [i.digest for i in action.identities]])).hexdigest(),
            "future_inputs": future, "future_mask": torch.from_numpy(mask),
            "pixel_targets": torch.stack([s.target_images for s in samples]),
            "target_rgb": np.stack([s.target_rgb for s in samples]),
            "future_identities": tuple(s.identity for s in samples),
            "future_digest": hashlib.sha256(canonical_json_bytes(
                [s.identity for s in samples])).hexdigest(),
        }


class AuxiliaryTargetCache:
    """Bounded FP32 CPU target LRU; independent frozen teacher, no persistent cache."""
    def __init__(self, teacher, *, maximum_bytes=8*1024**3):
        if type(maximum_bytes) is not int or maximum_bytes < 128*2048*4:
            raise ValueError("target cache must hold at least one target")
        self.teacher, self.maximum_bytes = teacher, maximum_bytes
        self.values = OrderedDict()
        self.peak_bytes = 0

    def encode(self, batch, device):
        keys = [(s["source_digest"], s["target"]) for s in batch["future_identities"]]
        if any(s["split"] != "train" for s in batch["future_identities"]):
            raise ValueError("pilot target cache must remain training-only")
        result, hits, misses = [], 0, 0
        item_bytes = 128*2048*4
        for key, rgb in zip(keys, batch["target_rgb"], strict=True):
            if key not in self.values:
                value = self.teacher(rgb[None])[0].detach().cpu().clone()
                if (value.dtype != torch.float32 or value.shape != (128, 2048)
                        or not torch.isfinite(value).all() or (value.norm(dim=-1)-1).abs().max() > 1e-5):
                    raise ValueError("invalid canonical JEPA target")
                while self.values and (len(self.values)+1)*item_bytes > self.maximum_bytes:
                    self.values.popitem(last=False)
                self.values[key] = value
                misses += 1
            else:
                hits += 1
            self.values.move_to_end(key)
            result.append(self.values[key])
        self.peak_bytes = max(self.peak_bytes, len(self.values)*item_bytes)
        return torch.stack(result).to(device), {"target_cache_hits": hits,
            "target_cache_misses": misses, "target_cache_peak_bytes": self.peak_bytes}
