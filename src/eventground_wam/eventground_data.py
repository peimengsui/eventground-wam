"""Current observation and offline target records have separate, strict APIs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch

from eventground_wam.b0_training import sha256_file


@dataclass(frozen=True)
class CurrentObservation:
    front: np.ndarray
    wrist: np.ndarray
    proprioception: np.ndarray
    instruction: str

    def __post_init__(self):
        for pixels in (self.front, self.wrist):
            if pixels.shape != (256, 256, 3) or pixels.dtype != np.uint8:
                raise ValueError("Current cameras must be RGB uint8 256x256")
        if (self.proprioception.shape != (8,) or not np.isfinite(self.proprioception).all()
                or not isinstance(self.instruction, str) or not self.instruction):
            raise ValueError("Invalid current state or instruction")


def load_current_observation(row, *, verified_files=None):
    """Read exactly one current camera/state row; never read action or future arrays."""
    path = Path(row["recording_path"])
    digest = row["recording_sha256"]
    def fingerprint(stat):
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)

    before = fingerprint(path.stat())
    identity = (digest, before)
    if verified_files is None or verified_files.get(str(path)) != identity:
        if sha256_file(path) != digest:
            raise ValueError("Current observation source changed")
        if fingerprint(path.stat()) != before:
            raise ValueError("Current observation source changed during verification")
        if verified_files is not None:
            verified_files[str(path)] = identity
    anchor = row["anchor"]
    if type(anchor) is not int or anchor < 0:
        raise ValueError("Anchor must be a nonnegative integer")
    with h5py.File(path, "r") as h:
        if fingerprint(os.fstat(h.id.get_vfd_handle())) != before:
            raise ValueError("Current observation file was replaced before reading")
        if (not h.attrs["complete"] or h.attrs["source_id"] != row["source_id"]
                or anchor >= len(h["frame_index"]) or h["frame_index"][anchor] != anchor):
            raise ValueError("Current observation identity/frame differs")
        observation = CurrentObservation(
            h["observation/front"][anchor], h["observation/wrist"][anchor],
            h["observation/state"][anchor].astype(np.float32), row["instruction"],
        )
        if (fingerprint(os.fstat(h.id.get_vfd_handle())) != before
                or fingerprint(path.stat()) != before):
            raise ValueError("Current observation source changed during reading")
        return observation


def collate_current_observations(observations, builder, device):
    if not observations or any(type(o) is not CurrentObservation for o in observations):
        raise ValueError("Only strict current observation records enter the model")
    encoding = builder(
        [o.instruction for o in observations], [o.front for o in observations],
        [o.wrist for o in observations],
    )
    compute_dtype = getattr(builder, "eventground_compute_dtype", torch.bfloat16)
    if compute_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("Unsupported EventGround input precision")
    encoded = {k: v.to(device=device, dtype=compute_dtype if v.is_floating_point() else v.dtype)
               for k, v in encoding.items()}
    state = torch.from_numpy(np.stack([o.proprioception for o in observations])).to(device)
    return encoded, state


def target_tokens(rows, device="cpu"):
    """Offline structured supervision; never passed to the observation collator."""
    classes = ("none", "gripper_close", "contact", "grasp")
    tokens = []
    for row in rows:
        target = row["target"]
        if set(target) != {"next_event", "interval", "horizon"}:
            raise ValueError("Unexpected target fields")
        event = target["next_event"]
        if event not in classes:
            raise ValueError("Unknown event")
        if event == "none":
            if target["interval"] is not None or target["horizon"] is not None:
                raise ValueError("none has no interval/horizon")
            tokens.append([0, 0, 0])
        else:
            interval = target["interval"]
            if (not isinstance(interval, (list, tuple)) or len(interval) != 2
                    or any(type(x) is not int for x in interval)
                    or not 1 <= interval[0] < interval[1] <= 33
                    or type(target["horizon"]) is not int or target["horizon"] != interval[0]):
                raise ValueError("Invalid event interval/horizon")
            tokens.append([classes.index(event), *interval])
    return torch.tensor(tokens, dtype=torch.long, device=device)
