import copy
import importlib.util
from pathlib import Path

import pytest
import torch

_path = Path(__file__).parents[1] / "scripts/diagnose_week10_batch_numerics.py"
_spec = importlib.util.spec_from_file_location("week10_batch_numerics", _path)
diagnostic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diagnostic)


def encoding():
    # Four different patch counts make a wrong batch-axis slice detectable.
    grid = torch.tensor([[1, 2, 2], [1, 2, 3], [1, 2, 4], [2, 2, 2]])
    pixels = torch.arange(26 * 3).reshape(26, 3).float()
    return {"input_ids": torch.tensor([[10, 11, 12, 0, 0], [20, 21, 22, 23, 24]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]]),
            "mm_token_type_ids": torch.tensor([[0, 1, 1, 0, 0], [0, 1, 1, 0, 0]]),
            "image_grid_thw": grid, "pixel_values": pixels}


def test_slicing_preserves_padding_and_two_complete_camera_patch_blocks():
    source = encoding()
    first = diagnostic.slice_encoding(source, 0, 1)
    second = diagnostic.slice_encoding(source, 1, 2)
    assert first["input_ids"].shape == second["input_ids"].shape == (1, 5)
    assert torch.equal(first["attention_mask"], source["attention_mask"][:1])
    assert torch.equal(first["pixel_values"], source["pixel_values"][:10])
    assert torch.equal(second["pixel_values"], source["pixel_values"][10:])
    assert torch.equal(second["image_grid_thw"], source["image_grid_thw"][2:])
    for key in source:
        assert torch.equal(torch.cat([first[key], second[key]]), source[key])
        assert first[key].data_ptr() != source[key].data_ptr()


@pytest.mark.parametrize("change", ["patch_count", "missing_view", "missing_key", "bad_grid"])
def test_malformed_flattened_layout_fails_closed(change):
    source = encoding()
    if change == "patch_count":
        source["pixel_values"] = source["pixel_values"][:-1]
    elif change == "missing_view":
        source["image_grid_thw"] = source["image_grid_thw"][:-1]
    elif change == "missing_key":
        source.pop("mm_token_type_ids")
    else:
        source["image_grid_thw"][0, 0] = 0
    with pytest.raises(ValueError):
        diagnostic.slice_encoding(source, 0, 1)


@pytest.mark.parametrize("start,stop", [(0, 0), (1, 3), (-1, 1), (True, 2)])
def test_invalid_example_range_rejected(start, stop):
    with pytest.raises(ValueError):
        diagnostic.slice_encoding(encoding(), start, stop)


def test_gradient_comparison_reports_global_and_groups():
    reference = {"head.x": torch.tensor([3., 4.]), "language.y": torch.tensor([12.])}
    actual = copy.deepcopy(reference)
    actual["head.x"] += torch.tensor([1., 0.])
    metrics = diagnostic.gradient_comparison(reference, actual)
    assert metrics["relative_l2_by_group"]["all"] == pytest.approx(1 / 13)
    assert metrics["relative_l2_by_group"]["head"] == pytest.approx(1 / 5)
    assert metrics["relative_l2_by_group"]["language_lora"] == 0
    with pytest.raises(ValueError):
        diagnostic.gradient_comparison(reference, {"head.x": actual["head.x"]})


def test_actual_torch_version_metadata_can_be_sealed(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "diagnostic test device")
    report = diagnostic.backend_metadata("cuda:0", [torch.bfloat16, torch.float32])
    sealed = diagnostic.seal_contract({"backend": report})
    assert type(sealed["backend"]["torch"]) is str
    assert sealed["backend"]["torch"] == str(torch.__version__)
