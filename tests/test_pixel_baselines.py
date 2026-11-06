from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch

from eventground_wam.demonstration_contracts import seal_contract
from eventground_wam.future_pairs import CAMERAS, HORIZONS, read_sealed
from eventground_wam.pixel_baselines import (
    FIELDS,
    PixelMetrics,
    assess_candidate,
    box_rgb,
    fit_templates,
    fixed_examples,
    group_weights,
    grouped_arrays,
    measure_video,
    measurement_rule,
    motion_mask,
    summarize_arrays,
)

ROOT = Path(__file__).parents[1]


@pytest.fixture(autouse=True)
def one_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def source(index, group="g", task="t", split="train", length=40):
    return {"episode_index": index, "group_digest": group, "task_key": task,
            "split": split, "episode_length": length, "digest": f"source-{index}"}


def metrics(error, ssim=.8, area=.2):
    return dict(zip(FIELDS, [error, ssim, error * area, error * (1 - area),
                             area, 1 - area], strict=True))


def rows_for(index, group, task, error=.1):
    return [{"episode_index": index, "source_digest": f"source-{index}",
             "group_digest": group, "task_key": task, "camera": camera,
             "horizon": h, "pairs": 40-h, "copy": metrics(error),
             "background": metrics(.3, ssim=.5)} for camera in CAMERAS for h in HORIZONS]


@pytest.fixture()
def gate_rows():
    rows = []
    for task in range(40):
        for group in range(3):
            index = task * 3 + group
            rows += rows_for(index, f"g-{index}", f"task-{task}", .09 + .01 * group)
    return rows


def candidate(rows, scale):
    result = deepcopy(rows)
    for row in result:
        row["prediction"] = metrics(row["copy"]["l1"] * scale, ssim=.85)
    return result


def test_box_uses_all_four_pixels_without_uint8_overflow():
    rgb = np.full((256, 256, 3), 255, dtype=np.uint8)
    rgb[0, 0] = [0, 127, 255]
    result = box_rgb(rgb)
    np.testing.assert_allclose(result[:, 0, 0], [765/1020, 892/1020, 1])
    assert result.shape == (3, 128, 128) and result.dtype == np.float32
    assert result.flags.c_contiguous and np.all(result[:, 1:, 1:] == 1)
    with pytest.raises(ValueError):
        box_rgb(rgb.astype(np.float32))


def test_ssim_analytic_constants_and_identity():
    scorer = PixelMetrics(dtype=torch.float64)
    black = torch.zeros(2, 3, 16, 16, dtype=torch.float64)
    white = torch.ones_like(black)
    torch.testing.assert_close(scorer.ssim(black, white),
                               torch.full((2,), .0001/1.0001, dtype=torch.float64))
    generator = torch.Generator().manual_seed(13)
    values = torch.rand(2, 3, 16, 16, generator=generator, dtype=torch.float64)
    torch.testing.assert_close(scorer.ssim(values, values), torch.ones(2, dtype=torch.float64))


def test_mask_is_ground_truth_defined_and_discards_tiny_changes():
    current = torch.zeros(2, 3, 16, 16)
    target = current.clone()
    target[0, :, :8, :] = 1
    target[1, :, 0, 0] = 1  # Only 1/256 of the image changes: excluded.
    scorer = PixelMetrics()
    copy = scorer.paired(current, target, current)
    oracle = scorer.paired(current, target, target)
    torch.testing.assert_close(copy["motion_area"], torch.tensor([.5, 0]))
    torch.testing.assert_close(copy["motion_area"], oracle["motion_area"])
    torch.testing.assert_close(oracle["motion_error_mass"], torch.zeros(2))
    assert copy["static_error_mass"][1].item() == 1/256


def test_ssim_keeps_low_variance_moments_in_double_precision():
    ramp = .73 + torch.linspace(0, .0001, 16)[None, None, None, :]
    image = ramp.expand(1, 3, 16, 16).contiguous()
    target = torch.full_like(image, .73005)
    scorer = PixelMetrics()
    mean, variance = scorer.moments(image)
    assert variance.dtype == torch.float64 and torch.all(variance > 0)
    # For a constant target, its variance and covariance are exactly zero.
    constant = target.double()[0, 0, 0, 0]
    expected = ((2*mean*constant + .0001)*.0009
                / ((mean.square()+constant.square()+.0001)*(variance+.0009))).mean()
    assert scorer.ssim(image, target).dtype == torch.float64
    assert scorer.ssim(image, target).item() == pytest.approx(expected.item(), abs=1e-11)


def test_motion_threshold_excludes_exact_boundary_using_integer_box_sums():
    current = torch.full((4, 3, 128, 128), 100/1020)
    target = current.clone()
    target[0] = 140/1020  # Exactly 10/255 average RGB difference.
    target[1] = 141/1020  # Above the threshold.
    target[2, 0] = 220/1020  # Same boundary, change concentrated in red.
    target[3, 0] = 221/1020
    mask = motion_mask(current, target)
    assert mask.sum(dim=(1, 2)).tolist() == [0, 16384, 0, 16384]


def test_motion_area_eligibility_uses_exact_pixel_count():
    current = torch.zeros(2, 3, 128, 128)
    target = current.clone().reshape(2, 3, -1)
    target[0, :, :163] = 1
    target[1, :, :164] = 1
    mask = motion_mask(current, target.reshape_as(current))
    assert mask.sum(dim=(1, 2)).tolist() == [0, 164]


def test_optimized_all_anchor_metrics_match_direct_pair_computation():
    rng = torch.Generator().manual_seed(29)
    images = torch.rand(37, 3, 16, 16, generator=rng)
    background = torch.rand(3, 16, 16, generator=rng)
    actual = measure_video(images, background, batch_size=7)
    scorer = PixelMetrics()
    for h in HORIZONS:
        for method, prediction in (("copy", images[:-h]),
                                   ("background", background.expand_as(images[h:]))):
            expected = scorer.paired(images[:-h], images[h:], prediction)
            for field in FIELDS:
                assert actual[h][method][field] == pytest.approx(
                    expected[field].double().mean().item(), abs=2e-7)


def test_template_group_weights_prevent_variant_count_bias():
    sources = [source(0), source(1), source(2, group="h")]
    means = [(s, np.full((2, 3, 128, 128), value, np.float32))
             for s, value in zip(sources, [0., .2, .9])]
    templates, tasks, counts = fit_templates(sources, iter(means))
    np.testing.assert_allclose(templates, .5, atol=1e-7)
    assert tasks == ["t"] and counts == {"t": {"groups": 2, "episodes": 3}}


@pytest.mark.parametrize("split", ["validation", "test"])
def test_fit_rejects_nontrain_before_consuming_images(split):
    def forbidden():
        raise AssertionError("read forbidden payload")
        yield
    with pytest.raises(ValueError, match="training-only"):
        fit_templates([source(0, split=split)], forbidden())


def test_fit_requires_complete_ordered_coverage():
    a, b = source(0), source(1)
    mean = np.zeros((2, 3, 128, 128), np.float32)
    with pytest.raises(ValueError, match="order/identity"):
        fit_templates([a, b], [(b, mean), (a, mean)])
    with pytest.raises(ValueError, match="incomplete"):
        fit_templates([a, b], [(a, mean)])


def test_group_and_task_weights_are_not_frame_or_variant_counts():
    rows = rows_for(0, "g", "t", .1) + rows_for(1, "g", "t", .3)
    rows += rows_for(2, "h", "t", .6) + rows_for(3, "i", "u", .8)
    groups, array = grouped_arrays(rows, "copy")
    point, bootstrap = group_weights(groups, replicates=100)
    report = summarize_arrays(array, point, bootstrap)
    assert report["l1"]["value"] == pytest.approx(.6)  # ((.2+.6)/2+.8)/2
    np.testing.assert_allclose(bootstrap.sum(axis=1), 1)
    # Duplicating both variants within a group does not change the estimate or uncertainty.
    rows += rows_for(4, "g", "t", .1) + rows_for(5, "g", "t", .3)
    again_groups, again = grouped_arrays(rows, "copy")
    assert groups == again_groups
    np.testing.assert_allclose(array, again)


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "identity", "mass"])
def test_grouping_rejects_corrupt_measurements(corruption):
    rows = rows_for(0, "g", "t")
    if corruption == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif corruption == "missing":
        rows.pop()
    elif corruption == "identity":
        rows[0]["group_digest"] = "other"
    else:
        rows[0]["copy"]["motion_error_mass"] = .99
    with pytest.raises(ValueError):
        grouped_arrays(rows, "copy")


def test_frozen_qualitative_panel_is_reproducible_and_in_bounds():
    sources = [source(i, f"g-{i//2}", f"t-{i//4}", "validation", 40+i)
               for i in range(160)]
    examples = fixed_examples(sources)
    assert examples == fixed_examples(list(reversed(sources)))
    assert len(examples) == 480
    assert len({row["group_digest"] for row in examples}) == 40
    for row in examples:
        assert 0 <= row["anchor"] < row["target"] < sources[row["episode_index"]]["episode_length"]
    with pytest.raises(ValueError, match="validation only"):
        fixed_examples([source(0, split="test")])


def test_identical_copy_fails_and_true_improvement_passes(gate_rows):
    copied = candidate(gate_rows, 1)
    failed = assess_candidate(gate_rows, copied)
    assert failed["status"] == "FAIL" and not failed["checks"]["overall_l1"]
    passed = assess_candidate(gate_rows, candidate(gate_rows, .8))
    assert passed["status"] == "PASS"
    assert passed["effects"]["l1_vs_copy"]["value"] == pytest.approx(.2)
    assert passed["effects"]["motion_l1_vs_copy"]["value"] == pytest.approx(.2)


def test_good_overall_score_cannot_hide_camera_horizon_regression(gate_rows):
    predicted = candidate(gate_rows, .5)
    for row in predicted:
        if row["camera"] == CAMERAS[0] and row["horizon"] == 2:
            row["prediction"] = metrics(row["copy"]["l1"] * 1.03, ssim=.85)
    result = assess_candidate(gate_rows, predicted)
    assert result["status"] == "FAIL"
    assert result["checks"]["overall_l1"]
    assert not result["checks"]["all_camera_horizon_cells"]


def test_large_point_gain_without_positive_confidence_bound_fails(gate_rows):
    predicted = candidate(gate_rows, .5)
    for row in predicted:
        error = [.001, .001, .261][row["episode_index"] % 3]
        row["prediction"] = metrics(error, ssim=.85)
    result = assess_candidate(gate_rows, predicted)
    assert result["effects"]["l1_vs_copy"]["value"] > .1
    assert result["effects"]["l1_vs_copy"]["paired_95"][0] < 0
    assert result["status"] == "FAIL" and not result["checks"]["overall_l1"]


def test_ssim_uncertainty_guard_rejects_small_unstable_improvement(gate_rows):
    predicted = candidate(gate_rows, .5)
    for row in predicted:
        row["prediction"]["ssim"] = [.9, .9, .603][row["episode_index"] % 3]
    result = assess_candidate(gate_rows, predicted)
    assert result["effects"]["ssim_vs_copy"]["value"] > 0
    assert result["effects"]["ssim_vs_copy"]["paired_95"][0] < -.005
    assert result["status"] == "FAIL" and not result["checks"]["ssim"]


def test_insufficient_groups_is_not_assessed(gate_rows):
    rows = [row for row in gate_rows if row["episode_index"] != 0]
    assert assess_candidate(rows, candidate(rows, .5))["status"] == "NOT_ASSESSED_MOTION_COVERAGE"


@pytest.mark.parametrize("corruption", ["source", "mask", "coverage"])
def test_candidate_cannot_change_evaluation_pairs_or_masks(gate_rows, corruption):
    predicted = candidate(gate_rows, .5)
    if corruption == "source":
        predicted[0]["source_digest"] = "wrong"
    elif corruption == "mask":
        predicted[0]["prediction"]["motion_area"] = .9
    else:
        predicted.pop()
    with pytest.raises(ValueError):
        assess_candidate(gate_rows, predicted)


def test_no_motion_is_inconclusive_not_a_pass(gate_rows):
    for row in gate_rows:
        row["copy"] = metrics(.1, area=0)
        row["background"] = metrics(.3, area=0)
    predicted = deepcopy(gate_rows)
    for row in predicted:
        row["prediction"] = metrics(0, ssim=1, area=0)
    assert assess_candidate(gate_rows, predicted)["status"].startswith("NOT_ASSESSED")


def test_runner_rejects_test_payload_before_filesystem_access(tmp_path):
    spec = importlib.util.spec_from_file_location("pixel_runner", ROOT / "scripts/run_pixel_baselines.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    with pytest.raises(ValueError, match="forbidden"):
        list(runner.decoded_camera(tmp_path, source(0, split="test"), CAMERAS[0], "validation"))


def test_runtime_receipt_accepts_torch_version_wrapper():
    spec = importlib.util.spec_from_file_location("pixel_runner", ROOT / "scripts/run_pixel_baselines.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    sealed = seal_contract({"runtime": runner.runtime_metadata(torch.device("cpu"))})
    assert sealed["runtime"]["torch"] == str(torch.__version__)
    assert type(sealed["runtime"]["torch"]) is str


def test_audit_receipt_normalizes_numpy_scalar_estimates():
    spec = importlib.util.spec_from_file_location("pixel_audit", ROOT / "scripts/validate_pixel_baselines.py")
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    result = audit.primary_estimates(np.array([.1, .8, .04, .06, .2, .8]))
    sealed = seal_contract({"estimates": result})
    assert sealed["estimates"]["motion_l1"] == pytest.approx(.2)
    assert all(type(value) is float for value in result.values())


def test_executable_measurement_rule_matches_sealed_pretraining_protocol():
    protocol = read_sealed(ROOT / "configs/evaluation/week5_pixel_baselines_v2.json")
    assert protocol["measurement"] == measurement_rule()
    assert protocol["frozen_before_predictor_training"] is True
