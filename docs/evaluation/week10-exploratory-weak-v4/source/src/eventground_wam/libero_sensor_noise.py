"""Host-independent compatibility helpers for LIBERO-Plus sensor noise."""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np

MOTION_BLUR_PARAMETERS = (
    (5, 2),
    (8, 3),
    (10, 4),
    (12, 5),
    (15, 6),
    (18, 8),
    (20, 10),
    (25, 12),
    (30, 15),
    (35, 20),
)


def codec_independent_motion_blur(image: Any, severity: int = 1) -> np.ndarray:
    """Apply LIBERO-Plus motion blur without an ImageMagick image codec.

    The upstream implementation round-trips every RGB observation through a
    PNG blob before calling ``MagickMotionBlurImage``. That makes benchmark
    execution depend on host-level ImageMagick PNG delegates. Importing and
    exporting raw pixels calls the same blur primitive with the same parameter
    table and random angle while removing that undeclared system dependency.
    """

    if not 1 <= severity <= len(MOTION_BLUR_PARAMETERS):
        raise ValueError(f"Motion-blur severity must be in [1, 10], found {severity}")
    rgb = np.asarray(image, dtype=np.uint8)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 RGB image, found shape {rgb.shape}")

    from wand.api import library
    from wand.image import Image as WandImage

    library.MagickMotionBlurImage.argtypes = (
        ctypes.c_void_p,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
    )
    radius, sigma = MOTION_BLUR_PARAMETERS[severity - 1]
    height, width = rgb.shape[:2]
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = 255
    with WandImage(width=width, height=height) as wand_image:
        wand_image.import_pixels(
            channel_map="RGBA",
            storage="char",
            data=rgba.reshape(-1).tolist(),
        )
        succeeded = library.MagickMotionBlurImage(
            wand_image.wand,
            float(radius),
            float(sigma),
            float(np.random.uniform(-45, 45)),
        )
        if not succeeded:
            wand_image.raise_exception()
            raise RuntimeError("ImageMagick motion blur failed without an exception")
        output = wand_image.export_pixels(channel_map="RGB", storage="char")
    return np.asarray(output, dtype=np.uint8).reshape(height, width, 3)


def install_codec_independent_motion_blur() -> None:
    """Install the reproducible motion-blur path into pinned LIBERO-Plus."""

    from libero.libero.envs import env_wrapper

    env_wrapper.motion_blur = codec_independent_motion_blur
