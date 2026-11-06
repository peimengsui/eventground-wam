"""Synthetic HDF5 checks for the causal current-frame API and offline targets."""

import hashlib

import h5py
import numpy as np
import pytest
import torch

import eventground_wam.eventground_data as data


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def recording(tmp_path):
    path = tmp_path / "synthetic-recording.h5"
    source = "a" * 64
    with h5py.File(path, "x") as handle:
        handle.attrs["complete"] = True
        handle.attrs["source_id"] = source
        handle["frame_index"] = np.arange(4)
        handle["observation/front"] = np.stack([
            np.full((256, 256, 3), index, np.uint8) for index in range(4)
        ])
        handle["observation/wrist"] = np.stack([
            np.full((256, 256, 3), 20 + index, np.uint8) for index in range(4)
        ])
        handle["observation/state"] = np.arange(32, dtype=np.float64).reshape(4, 8)
        # These are deliberately invalid as usable training inputs. Loading a
        # current observation must not inspect or validate offline/future fields.
        handle["action"] = np.full((4, 7), np.nan)
        handle["future_contact"] = np.full((4, 8), np.nan)
    return path, {
        "recording_path": str(path), "recording_sha256": sha(path), "source_id": source,
        "anchor": 2, "instruction": "move the synthetic object",
        "target": {"next_event": "grasp", "interval": [3, 6], "horizon": 3},
        "future_state": "must not enter the model",
    }


def test_loads_exact_current_frame_and_no_future_fields(recording):
    _, row = recording
    current = data.load_current_observation(row)
    assert type(current) is data.CurrentObservation
    assert set(current.__dataclass_fields__) == {"front", "wrist", "proprioception", "instruction"}
    assert np.all(current.front == 2) and np.all(current.wrist == 22)
    np.testing.assert_array_equal(current.proprioception, np.arange(16, 24, dtype=np.float32))
    assert current.proprioception.dtype == np.float32
    assert not hasattr(current, "target") and not hasattr(current, "future_state")


def test_hdf5_access_trace_contains_only_current_camera_and_state_rows(recording, monkeypatch):
    _, row = recording
    actual_file = h5py.File
    reads = []

    class TracedDataset:
        def __init__(self, dataset, key):
            self.dataset, self.key = dataset, key

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            reads.append((self.key, index))
            assert index == row["anchor"]
            return self.dataset[index]

    class TracedFile:
        def __init__(self, *args, **kwargs):
            self.handle = actual_file(*args, **kwargs)
            self.attrs = self.handle.attrs
            self.id = self.handle.id

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.handle.close()

        def __getitem__(self, key):
            assert key in {"frame_index", "observation/front", "observation/wrist", "observation/state"}
            return TracedDataset(self.handle[key], key)

    monkeypatch.setattr(data.h5py, "File", TracedFile)
    data.load_current_observation(row)
    assert reads == [
        ("frame_index", 2), ("observation/front", 2),
        ("observation/wrist", 2), ("observation/state", 2),
    ]


def test_sha_mismatch_rejected_before_hdf5_parsing(recording, monkeypatch):
    _, row = recording
    row["recording_sha256"] = "0" * 64
    def fail_if_opened(*_args, **_kwargs):
        raise AssertionError("Corrupt recording must not be opened")
    monkeypatch.setattr(data.h5py, "File", fail_if_opened)
    with pytest.raises(ValueError, match="changed"):
        data.load_current_observation(row)


def test_verification_cache_populated_without_exposing_targets(recording):
    path, row = recording
    cache = {}
    first = data.load_current_observation(row, verified_files=cache)
    assert str(path) in cache
    second = data.load_current_observation(row, verified_files=cache)
    np.testing.assert_array_equal(first.front, second.front)
    assert set(cache) == {str(path)}


def test_cached_file_mutation_is_rejected(recording):
    path, row = recording
    cache = {}
    data.load_current_observation(row, verified_files=cache)
    with h5py.File(path, "r+") as handle:
        handle["observation/front"][2, 0, 0, 0] = 250
    with pytest.raises(ValueError, match="source changed"):
        data.load_current_observation(row, verified_files=cache)


def test_file_changed_during_hashing_is_rejected(recording, monkeypatch):
    path, row = recording
    original_digest = row["recording_sha256"]

    def hash_then_mutate(_path):
        with h5py.File(path, "r+") as handle:
            handle["observation/front"][2, 0, 0, 0] = 250
        return original_digest

    monkeypatch.setattr(data, "sha256_file", hash_then_mutate)
    with pytest.raises(ValueError, match="during verification"):
        data.load_current_observation(row)


@pytest.mark.parametrize("anchor", [-1, 4, True, 1.0, "1"])
def test_invalid_anchors_rejected(recording, anchor):
    _, row = recording
    row["anchor"] = anchor
    with pytest.raises(ValueError):
        data.load_current_observation(row)


@pytest.mark.parametrize("mutation", ["source", "incomplete", "frame_order", "nan_state"])
def test_hdf5_identity_and_numeric_failures_rejected(recording, mutation):
    path, row = recording
    with h5py.File(path, "r+") as handle:
        if mutation == "source":
            handle.attrs["source_id"] = "b" * 64
        elif mutation == "incomplete":
            handle.attrs["complete"] = False
        elif mutation == "frame_order":
            handle["frame_index"][2] = 3
        else:
            handle["observation/state"][2, 0] = np.nan
    row["recording_sha256"] = sha(path)
    with pytest.raises(ValueError):
        data.load_current_observation(row)


@pytest.mark.parametrize("key,value", [
    ("front", np.zeros((128, 128, 3), dtype=np.uint8)),
    ("front", np.zeros((256, 256, 3), dtype=np.float32)),
    ("wrist", np.zeros((256, 256, 4), dtype=np.uint8)),
    ("proprioception", np.zeros(7)), ("proprioception", np.full(8, np.inf)),
    ("instruction", ""), ("instruction", None),
])
def test_current_observation_shape_dtype_and_content_validation(key, value):
    fields = {"front": np.zeros((256, 256, 3), np.uint8),
              "wrist": np.zeros((256, 256, 3), np.uint8),
              "proprioception": np.zeros(8), "instruction": "move"}
    fields[key] = value
    with pytest.raises(ValueError):
        data.CurrentObservation(**fields)


@pytest.mark.parametrize("compute_dtype", [torch.bfloat16, torch.float32])
def test_collator_routes_only_current_observations_to_builder(recording, compute_dtype):
    _, row = recording
    current = data.load_current_observation(row)
    calls = []

    def builder(instructions, fronts, wrists):
        calls.append((instructions, fronts, wrists))
        assert instructions == [row["instruction"]]
        assert np.all(fronts[0] == 2) and np.all(wrists[0] == 22)
        return {"input_ids": torch.tensor([[1, 2]]), "pixel_values": torch.ones(1, 3)}

    builder.eventground_compute_dtype = compute_dtype
    encoded, state = data.collate_current_observations([current], builder, "cpu")
    assert len(calls) == 1
    assert encoded["input_ids"].dtype == torch.int64
    assert encoded["pixel_values"].dtype == compute_dtype
    assert state.shape == (1, 8) and state.dtype == torch.float32
    assert "target" not in encoded and "future_state" not in encoded and "action" not in encoded


def test_future_and_action_records_cannot_be_routed_through_collator(recording):
    _, row = recording

    def forbidden_builder(*_args):
        raise AssertionError("Invalid record reached the encoder")

    for invalid in ([], [row], [{"action": np.zeros(7)}], [{"future_state": np.zeros(8)}]):
        with pytest.raises(ValueError, match="current observation"):
            data.collate_current_observations(invalid, forbidden_builder, "cpu")


def test_dataclass_rejects_future_keyword_fields():
    with pytest.raises(TypeError):
        data.CurrentObservation(
            front=np.zeros((256, 256, 3), np.uint8), wrist=np.zeros((256, 256, 3), np.uint8),
            proprioception=np.zeros(8), instruction="move", future_contact=np.zeros(8),
        )


def test_offline_target_support_is_exact_and_separate():
    targets = [
        {"next_event": "none", "interval": None, "horizon": None},
        {"next_event": "gripper_close", "interval": [1, 2], "horizon": 1},
        {"next_event": "contact", "interval": [32, 33], "horizon": 32},
        {"next_event": "grasp", "interval": [17, 33], "horizon": 17},
    ]
    result = data.target_tokens([{"target": target} for target in targets])
    assert result.dtype == torch.int64
    assert result.tolist() == [[0, 0, 0], [1, 1, 2], [2, 32, 33], [3, 17, 33]]


@pytest.mark.parametrize("target", [
    {"next_event": "other", "interval": [1, 2], "horizon": 1},
    {"next_event": "none", "interval": [1, 2], "horizon": 1},
    {"next_event": "none", "interval": None, "horizon": None, "future_state": []},
    {"next_event": "grasp", "interval": [0, 2], "horizon": 0},
    {"next_event": "grasp", "interval": [1, 34], "horizon": 1},
    {"next_event": "grasp", "interval": [2, 2], "horizon": 2},
    {"next_event": "grasp", "interval": [1, 2], "horizon": 2},
    {"next_event": "grasp", "interval": [1.0, 2], "horizon": 1},
    {"next_event": "grasp", "interval": [True, 2], "horizon": 1},
    {"next_event": "grasp", "interval": [1, 2], "horizon": True},
    {"next_event": "grasp", "interval": None, "horizon": 1},
])
def test_invalid_offline_targets_rejected(target):
    with pytest.raises(ValueError):
        data.target_tokens([{"target": target}])
