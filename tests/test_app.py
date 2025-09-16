import pytest
import numpy as np
import cv2
import torch
from app import FSRCNN, rgb_to_ycbcr, ycbcr_to_rgb, bicubic_upscale_rgb, upscale_ui, try_load_weights

def test_fsrcnn_model_initialization():
    for scale in [2, 3, 4]:
        model = FSRCNN(scale_factor=scale)
        assert model is not None
        assert hasattr(model, 'first_part')
        assert hasattr(model, 'mid_part')
        assert hasattr(model, 'last_part')

def test_color_conversion():
    test_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    ycbcr = rgb_to_ycbcr(test_img)
    reconstructed = ycbcr_to_rgb(ycbcr)
    
    assert test_img.shape == reconstructed.shape
    assert np.mean(np.abs(test_img.astype(float) - reconstructed.astype(float))) < 2.0

def test_bicubic_upscaling():
    test_img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    
    for scale in [1, 2, 3, 4]:
        upscaled = bicubic_upscale_rgb(test_img, scale)
        expected_shape = (16 * scale, 16 * scale, 3)
        assert upscaled.shape == expected_shape

def test_try_load_weights_error(tmp_path):
    model = FSRCNN(scale_factor=2)
    fake_weights = {
        "last_part.bias": torch.zeros(1),
        "a_fake_key.weight": torch.randn(10)
    }
    fake_weights_file = "test.pth"
    torch.save(fake_weights, fake_weights_file)
    result = try_load_weights(model, str(fake_weights_file))
    assert result == True

    model = FSRCNN(scale_factor=2)
    assert try_load_weights(model, None) == False


    model = FSRCNN(scale_factor=2)
    corrupted_file = tmp_path/"corrupted.pth"
    corrupted_file.write_text("this is just a text file, not a model!")
    result = try_load_weights(model, str(corrupted_file))
    assert result == False

def test_try_load_weights():
    model = FSRCNN(scale_factor=2)
    assert try_load_weights(model, "../models/fsrcnn_x2.pth") == False


def test_model_forward_pass():
    for scale in [2, 3, 4]:
        model = FSRCNN(scale_factor=scale)
        dummy_input = np.random.rand(1, 1, 32, 32).astype(np.float32)
        output = model(torch.from_numpy(dummy_input))
        
        expected_height = 32 * scale
        expected_width = 32 * scale
        assert output.shape[2] == expected_height
        assert output.shape[3] == expected_width

def test_upscale_ui_noimage():
    assert upscale_ui(None, 2, "FSRCNN (Y channel)") == (None, 'Please upload an image.')

def test_upscale_ui():
    # Float input
    float_image = np.random.rand(32, 32, 3).astype(np.float32)
    
    result = upscale_ui(
        image=float_image,
        scale_factor=2,
        method="Bicubic"
    )
    
    assert result[0] is not None
    assert result[0].dtype == np.uint8
    assert result[0].shape == (64, 64, 3)

    """Test upscale_ui with grayscale (2D) input"""
    grayscale_image = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
    result = upscale_ui(
        image=grayscale_image,
        scale_factor=2,
        method="FSRCNN (Y channel)",
    )
    
    assert result[0] is not None
    assert result[0].dtype == np.uint8
    assert result[0].shape == (64, 64, 3)

    """Test upscale_ui with RGBA input"""
    rgba_image = np.random.randint(0, 255, (32, 32, 4), dtype=np.uint8)
    
    result = upscale_ui(
        image=rgba_image,
        scale_factor=2,
        method="Bicubic",
    )
    assert result[0] is not None
    assert result[0].dtype == np.uint8
    assert result[0].shape == (64, 64, 3)

    downloadscale_image = np.random.randint(0, 255, (4000, 4000, 3), dtype=np.uint8)
    
    result = upscale_ui(
        image=downloadscale_image,
        scale_factor=2,
        method="FSRCNN (Y channel)",
    )
    
    assert result[0] is not None
    assert result[0].dtype == np.uint8
    assert result[0].shape == (5656, 5656, 3)



    test_img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    for scale in [2, 3, 4]:
        upscaled = upscale_ui(test_img, scale, "FSRCNN (Y channel)")
        expected_shape = (16 * scale, 16 * scale, 3)
        assert upscaled[0].shape == expected_shape

def test_upscale_ui_bicubic():
    test_img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    for scale in [2, 3, 4]:
        upscaled = upscale_ui(test_img, scale, "Bicubic")
        expected_shape = (16 * scale, 16 * scale, 3)
        assert upscaled[0].shape == expected_shape

if __name__ == "__main__":
    pytest.main([__file__])