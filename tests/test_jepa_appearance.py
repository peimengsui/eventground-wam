from pathlib import Path

import numpy as np
import pytest


def test_matched_appearance_transform_is_deterministic_and_geometry_preserving(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from review_jepa_candidate import TRANSFORMS, transform_rgb
    rng = np.random.default_rng(29)
    rgb = rng.integers(0, 256, (2, 2, 256, 256, 3), dtype=np.uint8)
    original = rgb.copy()
    for spec in TRANSFORMS:
        a = transform_rgb(rgb, spec)
        assert a.shape == rgb.shape and a.dtype == np.uint8
        np.testing.assert_array_equal(a, transform_rgb(rgb, spec))
        for i in range(2):
            np.testing.assert_array_equal(a[i], transform_rgb(rgb[i], spec))
    np.testing.assert_array_equal(rgb, original)
    np.testing.assert_array_equal(transform_rgb(rgb, TRANSFORMS[0]), rgb)
    constant = np.full_like(rgb, 100)
    assert (transform_rgb(constant, TRANSFORMS[1]) == 70).all()
    assert (transform_rgb(constant, TRANSFORMS[2]) == 130).all()
    with pytest.raises(ValueError):
        transform_rgb(rgb.astype(np.float32), TRANSFORMS[0])


def test_common_feature_bridge_keeps_cameras_and_identity_error(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from review_jepa_candidate import errors, pixel_to_native
    images = np.zeros((2, 2, 3, 128, 128), np.float32)
    images[:, 1] = 1
    rgb = pixel_to_native(images)
    assert rgb.shape == (2, 2, 256, 256, 3)
    assert (rgb[:, 0] == 0).all() and (rgb[:, 1] == 255).all()
    x = np.zeros((2, 128, 256)); x[..., 0] = 1
    np.testing.assert_array_equal(errors(x, x), 0)
    y = x.copy(); y[:, 64:, 0] = 0; y[:, 64:, 1] = 1
    assert (errors(x, y)[:, 0] == 0).all() and (errors(x, y)[:, 1] == 1).all()
