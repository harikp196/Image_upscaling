import pytest
import numpy as np
import cv2
from app import FSRCNN, rgb_to_ycbcr, ycbcr_to_rgb, bicubic_upscale_rgb

def test_fsrcnn_model_initialization():
    """Test that FSRCNN models can be initialized for different scales"""
    for scale in [2, 3, 4]:
        model = FSRCNN(scale_factor=scale)
        assert model is not None
        # Check model architecture components
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
    
    for scale in [2, 3, 4]:
        upscaled = bicubic_upscale_rgb(test_img, scale)
        expected_shape = (16 * scale, 16 * scale, 3)
        assert upscaled.shape == expected_shape

def test_model_forward_pass():
    for scale in [2, 3, 4]:
        model = FSRCNN(scale_factor=scale)
        dummy_input = np.random.rand(1, 1, 32, 32).astype(np.float32)
        output = model(torch.from_numpy(dummy_input))
        
        expected_height = 32 * scale
        expected_width = 32 * scale
        assert output.shape[2] == expected_height
        assert output.shape[3] == expected_width

if __name__ == "__main__":
    pytest.main([__file__])