from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from eventground_wam.jepa_features import (
    FrozenQwenVisualTargets,
    anchor_feature_diagnostics,
    normalize_visual_targets,
)


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Processor:
    patch_size, merge_size, temporal_patch_size = 16, 2, 2
    backend, rescale_factor = "torchvision", 1/255
    image_mean = image_std = (.5, .5, .5)

    def __call__(self, *, images, **kwargs):
        assert kwargs == {"do_resize": False, "do_rescale": True, "do_normalize": True,
                          "do_convert_rgb": True, "return_tensors": "pt"}
        pixels = torch.stack([torch.full((256, 1536), float(x[0, 0, 0])/255)
                              for x in images]).flatten(0, 1)
        return {"pixel_values": pixels, "image_grid_thw": torch.tensor([[1, 16, 16]]*len(images))}


class Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(2048, dtype=torch.bfloat16))
        self.config = SimpleNamespace(patch_size=16, spatial_merge_size=2,
                                      temporal_patch_size=2, out_hidden_size=2048,
                                      hidden_size=1024, depth=24)

    def forward(self, pixels, *, grid_thw, return_dict):
        assert not self.training and not torch.is_grad_enabled() and return_dict
        marker = pixels.reshape(len(grid_thw), 256, 1536)[:, 0, 0]
        base = self.weight.expand(len(grid_thw), 64, 2048).clone()
        base[:, :, 0] = marker[:, None]*100
        return SimpleNamespace(pooler_output=base.flatten(0, 1))


def test_target_layout_camera_order_normalization_and_freezing():
    teacher = FrozenQwenVisualTargets(Visual(), Processor()).train()
    rgb = np.zeros((2, 2, 256, 256, 3), np.uint8)
    rgb[0, 1] = 255
    rgb[1, 0] = 128
    raw = teacher.encode_raw(rgb)
    assert raw.shape == (2, 128, 2048)
    assert raw[0, 0, 0] == 0 and raw[0, 64, 0] == 100
    assert 49 < raw[1, 0, 0] < 51 and raw[1, 64, 0] == 0
    result = teacher(rgb)
    assert result.dtype == torch.float32 and not result.requires_grad
    torch.testing.assert_close(result.norm(dim=-1), torch.ones(2, 128))
    assert not any(p.requires_grad for p in teacher.parameters())
    teacher.visual.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="unfrozen"):
        teacher(rgb)


@pytest.mark.parametrize("bad", ["float", "resolution", "cameras", "empty", "large", "text"])
def test_forbidden_target_inputs(bad):
    teacher = FrozenQwenVisualTargets(Visual(), Processor())
    rgb = np.zeros((1, 2, 256, 256, 3), np.uint8)
    if bad == "float":
        rgb = rgb.astype(np.float32)
    elif bad == "resolution":
        rgb = rgb[:, :, :128, :128]
    elif bad == "cameras":
        rgb = rgb[:, :1]
    elif bad == "empty":
        rgb = rgb[:0]
    elif bad == "large":
        rgb = np.repeat(rgb, 17, axis=0)
    else:
        rgb = {"images": rgb, "instruction": "future instruction"}
    with pytest.raises(ValueError):
        teacher(rgb)


def test_rejects_adapted_or_wrong_encoder():
    visual = Visual()
    visual.lora_A = nn.Parameter(torch.ones(2))
    with pytest.raises(ValueError, match="unadapted"):
        FrozenQwenVisualTargets(visual, Processor())
    visual = Visual()
    visual.config.spatial_merge_size = 1
    with pytest.raises(ValueError, match="configuration"):
        FrozenQwenVisualTargets(visual, Processor())


def test_no_fitted_or_batch_statistics_and_stop_gradient():
    g = torch.Generator().manual_seed(7)
    raw = torch.randn(3, 128, 2048, generator=g, requires_grad=True)
    result = normalize_visual_targets(raw)
    torch.testing.assert_close(result[:1], normalize_visual_targets(raw[:1]), rtol=0, atol=0)
    assert not result.requires_grad
    with pytest.raises(ValueError, match="degenerate"):
        normalize_visual_targets(torch.ones_like(raw))
    raw = raw.detach()
    raw[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        normalize_visual_targets(raw)


def test_spatial_diversity_cannot_hide_cross_example_collapse():
    g = torch.Generator().manual_seed(8)
    same = torch.randn(1, 64, 2048, generator=g).repeat(10, 1, 1)
    collapsed = anchor_feature_diagnostics(same)
    assert collapsed["pooled_effective_rank"] == 0
    assert collapsed["fixed_position_across_example_std_rms"] == 0
    assert collapsed["within_image_spatial_std_rms"] > 0
    distinct = anchor_feature_diagnostics(torch.randn(10, 64, 2048, generator=g))
    assert distinct["pooled_effective_rank"] > 5


def test_runtime_receipt_serializes_loaded_torch_build(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"scripts"))
    from check_jepa_targets import runtime_versions

    from eventground_wam.demonstration_contracts import seal_contract

    monkeypatch.setattr("importlib.metadata.version", lambda name: "test-version")
    versions = runtime_versions()
    assert type(versions["torch_loaded_build"]) is str
    receipt = seal_contract({"versions": versions})
    assert receipt["versions"]["torch_loaded_build"] == str(torch.__version__)


def test_vision_execution_is_single_image_even_for_many_observations():
    class BatchSensitiveVisual(Visual):
        def forward(self, pixels, *, grid_thw, return_dict):
            assert len(grid_thw) == 1
            return super().forward(pixels, grid_thw=grid_thw, return_dict=return_dict)
    teacher = FrozenQwenVisualTargets(BatchSensitiveVisual(), Processor())
    rgb = np.zeros((3, 2, 256, 256, 3), np.uint8)
    rgb[1] = 127
    rgb[2] = 255
    together = teacher(rgb)
    separate = torch.cat([teacher(rgb[i:i+1]) for i in range(3)])
    torch.testing.assert_close(together, separate, rtol=0, atol=0)
