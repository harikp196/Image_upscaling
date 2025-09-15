# tests/test_fsrcnn.py
import os
import importlib
import numpy as np
import torch
import pytest

# Adjust this import if your file isn't named fsrcnn_app.py
import app as app


@pytest.fixture(autouse=True)
def _reset_cache_between_tests():
    # Ensure cache isolation between tests
    app.MODEL_CACHE.clear()
    yield
    app.MODEL_CACHE.clear()


def test_fsrcnn_forward_output_shape_cpu_only():
    """FSRCNN forward should upscale 1-channel input by its scale factor."""
    model = app.FSRCNN(scale_factor=3).eval()
    x = torch.randn(1, 1, 10, 12)  # (N, C, H, W)
    with torch.inference_mode():
        y = model(x)
    assert y.shape == (1, 1, 30, 36), "Output shape must be (H*scale, W*scale)"


def test_run_fsrcnn_on_y_shape_and_dtype():
    """run_fsrcnn_on_y should return uint8 image with upscaled spatial dims."""
    y = np.random.randint(0, 256, (9, 7), dtype=np.uint8)
    model = app.FSRCNN(scale_factor=2).eval()
    out = app.run_fsrcnn_on_y(y, model)
    assert out.dtype == np.uint8
    assert out.shape == (9 * 2, 7 * 2)


def test_bicubic_upscale_rgb_shape_and_dtype():
    rgb = np.random.randint(0, 256, (16, 24, 3), dtype=np.uint8)
    out = app.bicubic_upscale_rgb(rgb, scale=4)
    assert out.dtype == np.uint8
    assert out.shape == (16 * 4, 24 * 4, 3)


def test_rgb_ycbcr_roundtrip_close():
    """RGB -> YCrCb -> RGB roundtrip should be close (small max diff)."""
    rgb = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
    ycrcb = app.rgb_to_ycbcr(rgb)
    back = app.ycbcr_to_rgb(ycrcb)
    # Allow small numerical differences from color conversion
    assert np.max(np.abs(back.astype(int) - rgb.astype(int))) <= 2


def test_fsrcnn_upscale_falls_back_to_bicubic_when_no_weights(tmp_path):
    """When no valid weights are provided, FSRCNN code must return bicubic result."""
    rgb = np.random.randint(0, 256, (12, 10, 3), dtype=np.uint8)
    scale = 3

    # Ensure a fresh cache so the "no-weights" path is exercised
    app.MODEL_CACHE.clear()

    out_fallback = app.fsrcnn_upscale_rgb(rgb, scale=scale, weights=None)
    out_bicubic = app.bicubic_upscale_rgb(rgb, scale=scale)

    assert out_fallback.shape == out_bicubic.shape
    # Code path returns bicubic directly; should be byte-identical
    assert np.array_equal(out_fallback, out_bicubic)


def test_ui_accepts_grayscale_and_rgba_and_clips():
    """The UI helper should handle grayscale, RGBA, and non-uint8 inputs."""
    # Grayscale -> stacked to RGB
    gray = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
    out_gray = app.upscale_ui(gray, 2, "Bicubic", "", "", "")
    assert out_gray.shape == (16, 16, 3)
    assert out_gray.dtype == np.uint8

    # RGBA -> drop alpha
    rgba = np.random.randint(0, 256, (8, 8, 4), dtype=np.uint8)
    out_rgba = app.upscale_ui(rgba, 2, "Bicubic", "", "", "")
    assert out_rgba.shape == (16, 16, 3)
    assert out_rgba.dtype == np.uint8

    # Float input -> should clip/convert to uint8 internally
    f_rgb = np.random.randn(8, 8, 3).astype(np.float32) * 1000.0  # intentionally wild
    out_float = app.upscale_ui(f_rgb, 2, "Bicubic", "", "", "")
    assert out_float.dtype == np.uint8
    assert out_float.shape == (16, 16, 3)


def test_maybe_downscale_for_memory_respects_limit():
    big = np.random.randint(0, 256, (4000, 4000, 3), dtype=np.uint8)  # 16M px
    capped = app.maybe_downscale_for_memory(big, max_pixels=1_000_000)
    assert capped.shape[0] * capped.shape[1] <= 1_000_000


def test_get_model_cache_per_scale():
    m2, w2 = app.get_model(2, weights_path=None)
    m3, w3 = app.get_model(3, weights_path=None)

    # Cache populated for both scales
    assert 2 in app.MODEL_CACHE and 3 in app.MODEL_CACHE
    assert isinstance(m2, app.FSRCNN) and isinstance(m3, app.FSRCNN)
    assert m2 is not m3, "Different scales should use different model instances"
