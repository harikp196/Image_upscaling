# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#       http://www.apache.org/licenses/LICENSE-2.0
# ============================================================================
# Combined FSRCNN model + Gradio UI for image upscaling (x2/x3/x4).
# Run:  python fsrcnn_gradio_app.py
# Reqs: pip install torch torchvision opencv-python gradio numpy

from math import sqrt
import typing as tp

import cv2
import numpy as np
import torch
from torch import nn
import gradio as gr


# ----------------------------
# FSRCNN model (1-channel Y)
# ----------------------------
class FSRCNN(nn.Module):
    """
    FSRCNN for single-channel (Y) super-resolution.
    Args:
        upscale_factor (int): 2, 3, or 4
    """

    def __init__(self, upscale_factor: int) -> None:
        super(FSRCNN, self).__init__()
        # Feature extraction
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(1, 56, (5, 5), (1, 1), (2, 2)),
            nn.PReLU(56)
        )
        # Shrink
        self.shrink = nn.Sequential(
            nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(12)
        )
        # Mapping
        self.map = nn.Sequential(
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12),
            nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
            nn.PReLU(12)
        )
        # Expand
        self.expand = nn.Sequential(
            nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)),
            nn.PReLU(56)
        )
        # Deconvolution (learned upsampling)
        self.deconv = nn.ConvTranspose2d(
            56, 1, (9, 9),
            (upscale_factor, upscale_factor),
            (4, 4),
            (upscale_factor - 1, upscale_factor - 1)
        )

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.feature_extraction(x)
        out = self.shrink(out)
        out = self.map(out)
        out = self.expand(out)
        out = self.deconv(out)
        return out

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.deconv.bias.data)


# ----------------------------
# Helpers
# ----------------------------
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache one model per scale to avoid re-init each call
MODEL_CACHE: dict[int, FSRCNN] = {}

def get_model(scale: int, weights_path: tp.Optional[str] = None) -> FSRCNN:
    if scale not in MODEL_CACHE:
        model = FSRCNN(scale).to(Device).eval()
        if weights_path:  # Optional: load pretrained weights
            state = torch.load(weights_path, map_location=Device)
            # Supports state dict under "state_dict" or raw
            if isinstance(state, dict) and "state_dict" in state:
                state = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
            else:
                state = {k.replace("module.", ""): v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
        MODEL_CACHE[scale] = model
    return MODEL_CACHE[scale]


def rgb_to_ycbcr(img_rgb: np.ndarray) -> np.ndarray:
    # Expect uint8 RGB
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)  # OpenCV uses CrCb order
    # Note: returns (Y, Cr, Cb)


def ycbcr_to_rgb(img_ycrcb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)


def run_fsrcnn_on_y(y: np.ndarray, scale: int, model: FSRCNN) -> np.ndarray:
    """
    y: HxW uint8
    returns: (H*scale, W*scale) uint8, clipped
    """
    # Normalize to 0..1 and shape to NCHW
    y_f = y.astype(np.float32) / 255.0
    tens = torch.from_numpy(y_f).unsqueeze(0).unsqueeze(0).to(Device)  # 1x1xH xW
    with torch.no_grad():
        out = model(tens)
    out_np = out.squeeze(0).squeeze(0).clamp(0.0, 1.0).cpu().numpy()
    out_u8 = (out_np * 255.0 + 0.5).astype(np.uint8)
    return out_u8


def fsrcnn_upscale_rgb(img_rgb: np.ndarray, scale: int,
                       weights: tp.Optional[str] = None) -> np.ndarray:
    """
    Process Y with FSRCNN, upscale Cr/Cb by bicubic, merge back to RGB.
    """
    h, w = img_rgb.shape[:2]
    model = get_model(scale, weights)

    # Convert to YCrCb
    ycrcb = rgb_to_ycbcr(img_rgb)
    y = ycrcb[..., 0]
    cr = ycrcb[..., 1]
    cb = ycrcb[..., 2]

    # Super-resolve Y channel
    y_sr = run_fsrcnn_on_y(y, scale, model)

    # Bicubic upscale Cr/Cb to match
    new_w, new_h = w * scale, h * scale
    cr_up = cv2.resize(cr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    cb_up = cv2.resize(cb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    ycrcb_up = np.stack([y_sr, cr_up, cb_up], axis=-1)
    rgb_up = ycbcr_to_rgb(ycrcb_up)
    return rgb_up


def bicubic_upscale_rgb(img_rgb: np.ndarray, scale: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    return cv2.resize(img_rgb, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


# ----------------------------
# Gradio UI
# ----------------------------
SCALE_OPTIONS = [2, 3, 4]

def upscale_ui(image: np.ndarray, scale_factor: int, method: str, weights_2x: str, weights_3x: str, weights_4x: str):
    """
    image: RGB numpy from Gradio
    """
    if image is None:
        return None

    # OpenCV expects RGB okay; we keep in RGB space for UI
    image = image.astype(np.uint8)

    if method == "FSRCNN (Y channel)":
        weights_map = {2: weights_2x, 3: weights_3x, 4: weights_4x}
        weights = weights_map.get(scale_factor) or None
        try:
            out = fsrcnn_upscale_rgb(image, scale_factor, weights)
        except Exception as e:
            # Fallback to bicubic if weights wrong/missing
            print(f"[FSRCNN] Error: {e}. Falling back to Bicubic.")
            out = bicubic_upscale_rgb(image, scale_factor)
    else:
        out = bicubic_upscale_rgb(image, scale_factor)

    return out


with gr.Blocks(title="FSRCNN Super-Resolution") as demo:
    gr.Markdown("# 🧠 FSRCNN Image Upscaling (x2/x3/x4)\n"
                "Upload an image, choose a scale, and select FSRCNN or Bicubic. "
                "For best quality, load pretrained weights for each scale.")

    with gr.Row():
        with gr.Column():
            inp_img = gr.Image(type="numpy", label="Input Image")
            scale = gr.Radio(choices=SCALE_OPTIONS, value=2, label="Upscale factor")
            method = gr.Radio(choices=["FSRCNN (Y channel)", "Bicubic"],
                              value="FSRCNN (Y channel)", label="Method")

            gr.Markdown("**Optional: load FSRCNN weights (.pth) per scale**")
            weights_2x = gr.Textbox(label="Weights path for x2 (optional)", placeholder="models/fsrcnn_x2.pth")
            weights_3x = gr.Textbox(label="Weights path for x3 (optional)", placeholder="models/fsrcnn_x3.pth")
            weights_4x = gr.Textbox(label="Weights path for x4 (optional)", placeholder="models/fsrcnn_x4.pth")

            run_btn = gr.Button("Upscale")
        with gr.Column():
            out_img = gr.Image(type="numpy", label="Upscaled Output")

    run_btn.click(
        fn=upscale_ui,
        inputs=[inp_img, scale, method, weights_2x, weights_3x, weights_4x],
        outputs=[out_img]
    )

if __name__ == "__main__":
    # share=True if you want a public link
    demo.launch()
