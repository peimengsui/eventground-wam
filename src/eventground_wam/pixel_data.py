"""Leakage-separated model inputs and targets resolved from the frozen future index."""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from eventground_wam.b0_data import ActionNormalizer
from eventground_wam.b0_model import QwenB0InputBuilder
from eventground_wam.demonstration_contracts import contract_digest
from eventground_wam.demonstration_reader import iter_parquet_rows, iter_video_frames
from eventground_wam.future_pairs import (
    CAMERAS,
    HORIZONS,
    RECORD,
    SPLITS,
    pair_bytes,
    pair_rule,
    read_sealed,
    sha256_file,
)
from eventground_wam.pixel_baselines import box_rgb
from eventground_wam.pixel_predictor import PixelCondition

FEATURE_PROMPT = "Represent the current scene and instruction for predicting robot action consequences."


class FutureInputBuilder(QwenB0InputBuilder):
    """Reuse pinned dual-camera processing, with a versioned current-scene request."""

    def build_messages(self, instructions, front_images, wrist_images):
        messages = super().build_messages(instructions, front_images, wrist_images)
        for conversation, instruction in zip(messages, instructions, strict=True):
            conversation[0]["content"][-1]["text"] = f"Instruction: {instruction}\n{FEATURE_PROMPT}"
        return messages


@dataclass(frozen=True)
class FutureObservation:
    instruction: str
    current_rgb: np.ndarray  # [front/wrist, 256,256,3] for pinned Qwen processing
    current_images: torch.Tensor  # [front/wrist,3,128,128] for spatial query/skip
    state: torch.Tensor
    actions: torch.Tensor  # [32,7], real actions [t,t+h), then zeros
    horizon: int


@dataclass(frozen=True)
class FutureTrainingSample:
    observation: FutureObservation
    target_images: torch.Tensor
    identity: dict


@dataclass(frozen=True)
class PixelTrainingBatch:
    condition: PixelCondition
    targets: torch.Tensor
    identities: tuple[dict, ...]


def make_future_sample(source, anchor, horizon, actions, states, videos, normalizer, state_bounds):
    if (source["split"] not in SPLITS or horizon not in HORIZONS or type(anchor) is not int
            or not 0 <= anchor < source["episode_length"]-horizon):
        raise ValueError("forbidden split or out-of-episode future request")
    length = source["episode_length"]
    if (actions.shape != (length, 7) or states.shape != (length, 8)
            or videos.shape != (length, 2, 256, 256, 3) or videos.dtype != np.uint8):
        raise ValueError("episode payload shape differs from the frozen source")
    padded = np.zeros((32, 7), np.float32)
    padded[:horizon] = normalizer.normalize(actions[anchor:anchor+horizon]).astype(np.float32)
    state = torch.as_tensor(states[anchor].copy(), dtype=torch.float32)
    lower, upper = (torch.tensor(state_bounds[key], dtype=torch.float32)
                    for key in ("lower_q01", "upper_q99"))
    if lower.shape != (8,) or upper.shape != (8,) or not torch.isfinite(state).all():
        raise ValueError("invalid current state or normalization bounds")
    span = upper-lower
    scaled = 2 * (state-lower) / span.clamp_min(1e-6) - 1
    normalized_state = torch.where(span > 1e-6, scaled.clamp(-1, 1), torch.zeros_like(scaled))
    current = np.stack([box_rgb(videos[anchor, camera]) for camera in range(2)])
    target = np.stack([box_rgb(videos[anchor+horizon, camera]) for camera in range(2)])
    observation = FutureObservation(source["instruction"], videos[anchor].copy(),
                                    torch.from_numpy(current), normalized_state,
                                    torch.from_numpy(padded), horizon)
    digest = RECORD.unpack(pair_bytes(source, anchor, horizon))[-1].hex()
    identity = {"episode_index": source["episode_index"], "group_digest": source["group_digest"],
                "task_key": source["task_key"], "split": source["split"],
                "source_digest": source["digest"], "pair_digest": digest,
                "anchor": anchor, "target": anchor+horizon, "horizon": horizon,
                "action_start": anchor, "action_stop_exclusive": anchor+horizon}
    return FutureTrainingSample(observation, torch.from_numpy(target), identity)


def collate_future_samples(samples, features, feature_padding):
    if not samples or features.shape[0] != len(samples) or feature_padding.shape != features.shape[:2]:
        raise ValueError("future batch/features are not aligned")
    observations = [sample.observation for sample in samples]
    condition = PixelCondition(
        features=features, feature_padding=feature_padding,
        current_images=torch.stack([x.current_images for x in observations]),
        state=torch.stack([x.state for x in observations]),
        actions=torch.stack([x.actions for x in observations]),
        horizons=torch.tensor([x.horizon for x in observations], dtype=torch.int64))
    return PixelTrainingBatch(condition, torch.stack([x.target_images for x in samples]),
                              tuple(x.identity for x in samples))


class FuturePairDataset:
    """Random-access references with bounded full-episode decoding; profile in Step 4."""

    def __init__(self, pairs: Path, repository: Path, *, split: str,
                 expected_manifest_sha256: str, normalizer: ActionNormalizer,
                 state_bounds: dict, cache_episodes=1):
        # Reject test before any filesystem access, even if its payload exists.
        if split not in SPLITS or type(cache_episodes) is not int or not 1 <= cache_episodes <= 4:
            raise ValueError("only bounded train/validation future datasets are allowed")
        self.pairs, self.repository, self.split = Path(pairs), Path(repository), split
        if sha256_file(self.pairs / "manifest.json") != expected_manifest_sha256:
            raise ValueError("future manifest differs from approved freeze")
        self.manifest = read_sealed(self.pairs / "manifest.json")
        if self.manifest["rule"] != pair_rule():
            raise ValueError("future pair rules changed")
        for name in ("sources.jsonl", f"{split}.bin"):
            pin = self.manifest["outputs"][name]
            path = self.pairs / name
            if path.stat().st_size != pin["bytes"] or sha256_file(path) != pin["sha256"]:
                raise ValueError("future pair/source index changed")
        sources = [json.loads(line) for line in (self.pairs / "sources.jsonl").read_text().splitlines()]
        if any(s["split"] not in SPLITS or s["digest"] != contract_digest(s) for s in sources):
            raise ValueError("unsealed or forbidden future source")
        self.sources = {s["episode_index"]: s for s in sources if s["split"] == split}
        self._offsets = {}
        total = 0
        for source in sorted(self.sources.values(), key=lambda s: s["episode_index"]):
            for h in HORIZONS:
                self._offsets[(source["episode_index"], h)] = total
                total += max(0, source["episode_length"]-h)
        if total != self.manifest["counts"][split]["pairs"]:
            raise ValueError("future source/record coverage mismatch")
        if (normalizer.source_identity is None
                or normalizer.source_identity["split_manifest_sha256"] != self.manifest["provenance"]["split_manifest_sha256"]
                or state_bounds.get("digest") != contract_digest(state_bounds)
                or state_bounds.get("split") != "train"
                or state_bounds["source"] != normalizer.source_identity):
            raise ValueError("future normalizers must match the frozen train-only source")
        self.normalizer, self.state_bounds = normalizer, state_bounds
        self._length, self.cache_episodes = total, cache_episodes
        self._cache = OrderedDict()
        self.decoded_episodes = []

    def __len__(self):
        return self._length

    def ordinal(self, episode, anchor, horizon):
        source = self.sources.get(episode)
        if (source is None or horizon not in HORIZONS or type(anchor) is not int
                or not 0 <= anchor < source["episode_length"]-horizon):
            raise ValueError("request is not a valid pair in the selected split")
        return self._offsets[(episode, horizon)] + anchor

    def __getitem__(self, ordinal):
        if type(ordinal) is not int or not 0 <= ordinal < len(self):
            raise IndexError(ordinal)
        with (self.pairs / f"{self.split}.bin").open("rb") as handle:
            handle.seek(ordinal*RECORD.size)
            raw = handle.read(RECORD.size)
        ep, anchor, target, h, code, source_sha, _ = RECORD.unpack(raw)
        source = self.sources.get(ep)
        if (source is None or code != SPLITS[self.split] or source["digest"] != source_sha.hex()
                or target != anchor+h or pair_bytes(source, anchor, h) != raw
                or self.ordinal(ep, anchor, h) != ordinal):
            raise ValueError("pair source, boundaries, identity or ordering changed")
        payload = self._episode(source)
        return make_future_sample(source, anchor, h, *payload, self.normalizer, self.state_bounds)

    def _episode(self, source):
        ep = source["episode_index"]
        if ep in self._cache:
            self._cache.move_to_end(ep)
            return self._cache[ep]
        if source["split"] != self.split:
            raise ValueError("cross-split payload access is forbidden")
        for pin in source["files"].values():
            path = self.repository / pin["path"]
            if path.stat().st_size != pin["bytes"] or sha256_file(path) != pin["sha256"]:
                raise ValueError("selected source payload changed")
        rows = list(iter_parquet_rows(self.repository / source["files"]["parquet"]["path"]))
        if len(rows) != source["episode_length"]:
            raise ValueError("telemetry length changed")
        for frame, row in enumerate(rows):
            if (row["episode_index"] != ep or row["frame_index"] != frame
                    or row["task_index"] != source["task_index"]
                    or abs(row["timestamp"]-frame/20) > 1e-4):
                raise ValueError("telemetry frame/camera alignment changed")
        actions = np.asarray([row["action"] for row in rows], dtype=np.float32)
        states = np.asarray([row["observation.state"] for row in rows], dtype=np.float32)
        if not np.isfinite(actions).all() or not np.isfinite(states).all():
            raise ValueError("non-finite source telemetry")
        videos = []
        for camera in CAMERAS:
            frames = [frame.payload.to_ndarray(format="rgb24") for frame in
                      iter_video_frames(self.repository / source["files"][camera]["path"])]
            if len(frames) != len(rows):
                raise ValueError("camera/telemetry frame counts differ")
            videos.append(np.stack(frames))
        payload = actions, states, np.stack(videos, axis=1)
        while len(self._cache) >= self.cache_episodes:
            self._cache.popitem(last=False)
        self._cache[ep] = payload
        self.decoded_episodes.append(ep)
        return payload
