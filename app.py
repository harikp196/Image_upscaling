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
import os

import cv2
import numpy as np
import torch
from torch import nn
import gradio as gr
import math



class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5 // 2),
            nn.PReLU(d)
        )
        self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
        for _ in range(m):
            self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3 // 2), nn.PReLU(s)])
        self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(
            d, num_channels, kernel_size=9,
            stride=scale_factor, padding=9 // 2,
            output_padding=scale_factor - 1
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0,
                                std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x




# ----------------------------
# FSRCNN model (1-channel Y)
# ----------------------------
# class FSRCNN(nn.Module):
#     """
#     FSRCNN for single-channel (Y) super-resolution.
#     Args:
#         upscale_factor (int): 2, 3, or 4
#     """

#     def __init__(self, upscale_factor: int) -> None:
#         super(FSRCNN, self).__init__()
#         # Feature extraction
#         self.feature_extraction = nn.Sequential(
#             nn.Conv2d(1, 56, (5, 5), (1, 1), (2, 2)),
#             nn.PReLU(56)
#         )
#         # Shrink
#         self.shrink = nn.Sequential(
#             nn.Conv2d(56, 12, (1, 1), (1, 1), (0, 0)),
#             nn.PReLU(12)
#         )
#         # Mapping
#         self.map = nn.Sequential(
#             nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
#             nn.PReLU(12),
#             nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
#             nn.PReLU(12),
#             nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
#             nn.PReLU(12),
#             nn.Conv2d(12, 12, (3, 3), (1, 1), (1, 1)),
#             nn.PReLU(12)
#         )
#         # Expand
#         self.expand = nn.Sequential(
#             nn.Conv2d(12, 56, (1, 1), (1, 1), (0, 0)),
#             nn.PReLU(56)
#         )
#         # Deconvolution (learned upsampling)
#         self.deconv = nn.ConvTranspose2d(
#             56, 1, (9, 9),
#             (upscale_factor, upscale_factor),
#             (4, 4),
#             (upscale_factor - 1, upscale_factor - 1)
#         )

#         self._initialize_weights()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         out = self.feature_extraction(x)
#         out = self.shrink(out)
#         out = self.map(out)
#         out = self.expand(out)
#         out = self.deconv(out)
#         return out

#     def _initialize_weights(self) -> None:
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight.data, mean=0.0,
#                                 std=sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
#                 nn.init.zeros_(m.bias.data)
#         nn.init.normal_(self.deconv.weight.data, mean=0.0, std=0.001)
#         nn.init.zeros_(self.deconv.bias.data)


# ----------------------------
# Helpers
# ----------------------------
Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cache: scale -> (model, has_valid_weights)
MODEL_CACHE: dict[int, tuple[FSRCNN, bool]] = {}


def try_load_weights(model, weights_path):
    """Load weights that use first_part, mid_part, last_part structure."""
    if not weights_path or not os.path.isfile(weights_path):
        print(f"[FSRCNN] No valid weights at {weights_path}. Falling back to Bicubic.")
        return False
    try:
        state_dict = model.state_dict()
        checkpoint = torch.load(weights_path, map_location=Device)
        for n, p in checkpoint.items():
            if n in state_dict.keys():
                state_dict[n].copy_(p)
            else:
                print(f"[FSRCNN] Unexpected key in weights: {n}")
        model.load_state_dict(state_dict, strict=True)
        print(f"[FSRCNN] Loaded weights from {weights_path}")
        return True
    except Exception as e:
        print(f"[FSRCNN] Failed to load weights: {e}")
        return False
        

# def try_load_weights(model: nn.Module, weights_path: tp.Optional[str]) -> bool:
#     """Return True if weights successfully loaded, else False."""
#     if not weights_path:
#         print("[FSRCNN] No weights path provided, using Bicubic fallback.")
#         return False
#     if not os.path.isfile(weights_path):
#         print(f"[FSRCNN] Weights not found: {weights_path}. Using Bicubic fallback.")
#         return False
#     try:
#         state = torch.load(weights_path, map_location=Device)
#         if isinstance(state, dict) and "state_dict" in state:
#             state = {k.replace("module.", ""): v for k, v in state["state_dict"].items()}
#         else:
#             # raw state dict; also strip potential 'module.' prefixes
#             state = {k.replace("module.", ""): v for k, v in state.items()}
#         model.load_state_dict(state, strict=True)
#         print(f"[FSRCNN] Loaded weights: {weights_path}")
#         return True
#     except Exception as e:
#         print(f"[FSRCNN] Failed to load weights: {e}. Using Bicubic fallback.")
#         return False


def get_model(scale, weights_path=None):
    if scale not in MODEL_CACHE:
        model = FSRCNN(scale_factor=scale).to(Device).eval()
        has_weights = try_load_weights(model, weights_path)
        MODEL_CACHE[scale] = (model, has_weights)
    else:
        model, has_weights = MODEL_CACHE[scale]
    return MODEL_CACHE[scale]

# def get_model(scale: int, weights_path: tp.Optional[str] = None) -> tuple[FSRCNN, bool]:
#     if scale not in MODEL_CACHE:
#         model = FSRCNN(scale).to(Device).eval()
#         has_weights = try_load_weights(model, weights_path)
#         MODEL_CACHE[scale] = (model, has_weights)
#     else:
#         model, has_weights = MODEL_CACHE[scale]
#         # If the cache has a randomly-initialized model and user now supplied a path, try once:
#         if not has_weights and weights_path:
#             has_weights = try_load_weights(model, weights_path)
#             MODEL_CACHE[scale] = (model, has_weights)
#     return MODEL_CACHE[scale]


def rgb_to_ycbcr(img_rgb: np.ndarray) -> np.ndarray:
    # Expect uint8 RGB
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)  # (Y, Cr, Cb)


def ycbcr_to_rgb(img_ycrcb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)


def run_fsrcnn_on_y(y: np.ndarray, model: FSRCNN) -> np.ndarray:
    """
    y: HxW uint8
    returns: HxW uint8 (already upscaled by model's deconv stride)
    """
    y_f = y.astype(np.float32) / 255.0
    tens = torch.from_numpy(y_f).unsqueeze(0).unsqueeze(0).to(Device)  # 1x1xH xW
    with torch.inference_mode():
        out = model(tens)
    out_np = out.squeeze(0).squeeze(0).clamp(0.0, 1.0).cpu().numpy()
    out_u8 = (out_np * 255.0 + 0.5).astype(np.uint8)
    return out_u8


def fsrcnn_upscale_rgb(img_rgb: np.ndarray, scale: int,
                       weights: tp.Optional[str] = None) -> np.ndarray:
    """
    Process Y with FSRCNN (requires valid weights), upscale Cr/Cb by bicubic, merge back to RGB.
    If weights are missing/invalid, this function returns Bicubic instead (safe fallback).
    """
    h, w = img_rgb.shape[:2]
    model, has_weights = get_model(scale, weights)

    if not has_weights:
        return bicubic_upscale_rgb(img_rgb, scale)

    # Convert to YCrCb
    ycrcb = rgb_to_ycbcr(img_rgb)
    y = ycrcb[..., 0]
    cr = ycrcb[..., 1]
    cb = ycrcb[..., 2]

    # Super-resolve Y channel using FSRCNN
    y_sr = run_fsrcnn_on_y(y, model)

    # Bicubic upscale Cr/Cb to match
    new_w, new_h = w * scale, h * scale
    cr_up = cv2.resize(cr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    cb_up = cv2.resize(cb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    ycrcb_up = np.stack([y_sr, cr_up, cb_up], axis=-1)
    rgb_up = ycbcr_to_rgb(ycrcb_up)
    return rgb_up


def bicubic_upscale_rgb(img_rgb: np.ndarray, scale: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    if scale <= 1:
        return img_rgb
    return cv2.resize(img_rgb, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def maybe_downscale_for_memory(img_rgb: np.ndarray, max_pixels: int = 8_000_000) -> np.ndarray:
    """
    Keep memory in check on very large inputs by shrinking the preview before upscaling.
    max_pixels is the H*W threshold (default ~8 MP).
    """
    h, w = img_rgb.shape[:2]
    if h * w <= max_pixels:
        return img_rgb
    scale = (max_pixels / (h * w)) ** 0.5
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


# ----------------------------
# Gradio UI
# ----------------------------
SCALE_OPTIONS = [2, 3, 4]


def upscale_ui(image: np.ndarray, scale_factor: int, method: str,
               weights_2x: str, weights_3x: str, weights_4x: str):
    """
    image: RGB numpy from Gradio
    """
    if image is None:
        return None

    # Ensure uint8 RGB
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)  # grayscale → RGB
    elif image.shape[2] == 4:
        image = image[..., :3]  # drop alpha

    # Optional: guard against gigantic inputs to avoid OOM crashes in Spaces/CPU
    image = maybe_downscale_for_memory(image, max_pixels=8_000_000)

    if method == "FSRCNN (Y channel)":
        weights_map = {2: weights_2x.strip(), 3: weights_3x.strip(), 4: weights_4x.strip()}
        weights = weights_map.get(scale_factor) or None
        out = fsrcnn_upscale_rgb(image, scale_factor, weights)
    else:
        out = bicubic_upscale_rgb(image, scale_factor)

    return out


with gr.Blocks(title="FSRCNN Super-Resolution") as demo:
    gr.Markdown(
        "# 🧠 FSRCNN Image Upscaling (x2/x3/x4)\n"
        "- **Tip:** If no valid FSRCNN weights are provided, the app will automatically fall back to Bicubic.\n"
        "- Provide `.pth` files trained for each scale to see FSRCNN quality improvements."
    )

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
