# from math import sqrt
# import typing as tp
# import os

# import cv2
# import numpy as np
# import torch
# from torch import nn
# import gradio as gr
# import math



# class FSRCNN(nn.Module):
#     def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
#         super(FSRCNN, self).__init__()
#         self.first_part = nn.Sequential(
#             nn.Conv2d(num_channels, d, kernel_size=5, padding=5 // 2),
#             nn.PReLU(d)
#         )
#         self.mid_part = [nn.Conv2d(d, s, kernel_size=1), nn.PReLU(s)]
#         for _ in range(m):
#             self.mid_part.extend([nn.Conv2d(s, s, kernel_size=3, padding=3 // 2), nn.PReLU(s)])
#         self.mid_part.extend([nn.Conv2d(s, d, kernel_size=1), nn.PReLU(d)])
#         self.mid_part = nn.Sequential(*self.mid_part)
#         self.last_part = nn.ConvTranspose2d(
#             d, num_channels, kernel_size=9,
#             stride=scale_factor, padding=9 // 2,
#             output_padding=scale_factor - 1
#         )

#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.first_part:
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight.data, mean=0.0,
#                                 std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
#                 nn.init.zeros_(m.bias.data)
#         for m in self.mid_part:
#             if isinstance(m, nn.Conv2d):
#                 nn.init.normal_(m.weight.data, mean=0.0,
#                                 std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
#                 nn.init.zeros_(m.bias.data)
#         nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
#         nn.init.zeros_(self.last_part.bias.data)

#     def forward(self, x):
#         x = self.first_part(x)
#         x = self.mid_part(x)
#         x = self.last_part(x)
#         return x

# Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# MODEL_CACHE: dict[int, tuple[FSRCNN, bool]] = {}


# def try_load_weights(model, weights_path):
#     if not weights_path or not os.path.isfile(weights_path):
#         print(f"[FSRCNN] No valid weights at {weights_path}. Falling back to Bicubic.")
#         return False
#     try:
#         state_dict = model.state_dict()
#         checkpoint = torch.load(weights_path, map_location=Device, weights_only=False)
#         for n, p in checkpoint.items():
#             if n in state_dict.keys():
#                 state_dict[n].copy_(p)
#             else:
#                 print(f"[FSRCNN] Unexpected key in weights: {n}")
#         model.load_state_dict(state_dict, strict=True)
#         print(f"[FSRCNN] Loaded weights from {weights_path}")
#         return True
#     except Exception as e:
#         print(f"[FSRCNN] Failed to load weights: {e}")
#         return False
        


# def get_model(scale, weights_path=None):
#     if scale not in MODEL_CACHE:
#         print("Not in cache")
#         model = FSRCNN(scale_factor=scale).to(Device).eval()
#         has_weights = try_load_weights(model, weights_path)
#         MODEL_CACHE[scale] = (model, has_weights)
#     else:
#         model, has_weights = MODEL_CACHE[scale]
#     return MODEL_CACHE[scale]


# def rgb_to_ycbcr(img_rgb: np.ndarray) -> np.ndarray:
#     # Expect uint8 RGB
#     return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)  # (Y, Cr, Cb)


# def ycbcr_to_rgb(img_ycrcb: np.ndarray) -> np.ndarray:
#     return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)


# def run_fsrcnn_on_y(y: np.ndarray, model: FSRCNN) -> np.ndarray:
#     y_f = y.astype(np.float32) / 255.0
#     tens = torch.from_numpy(y_f).unsqueeze(0).unsqueeze(0).to(Device)  # 1x1xH xW
#     with torch.inference_mode():
#         out = model(tens)
#     out_np = out.squeeze(0).squeeze(0).clamp(0.0, 1.0).cpu().numpy()
#     out_u8 = (out_np * 255.0 + 0.5).astype(np.uint8)
#     return out_u8


# def fsrcnn_upscale_rgb(img_rgb: np.ndarray, scale: int,
#                        weights: tp.Optional[str] = None) -> np.ndarray:
#     h, w = img_rgb.shape[:2]
#     model, has_weights = get_model(scale, weights)

#     if not has_weights:
#         return bicubic_upscale_rgb(img_rgb, scale)

#     ycrcb = rgb_to_ycbcr(img_rgb)
#     y = ycrcb[..., 0]
#     cr = ycrcb[..., 1]
#     cb = ycrcb[..., 2]
#     y_sr = run_fsrcnn_on_y(y, model)
#     new_w, new_h = w * scale, h * scale
#     cr_up = cv2.resize(cr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
#     cb_up = cv2.resize(cb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

#     ycrcb_up = np.stack([y_sr, cr_up, cb_up], axis=-1)
#     rgb_up = ycbcr_to_rgb(ycrcb_up)
#     return rgb_up


# def bicubic_upscale_rgb(img_rgb: np.ndarray, scale: int) -> np.ndarray:
#     h, w = img_rgb.shape[:2]
#     if scale <= 1:
#         return img_rgb
#     return cv2.resize(img_rgb, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


# def maybe_downscale_for_memory(img_rgb: np.ndarray, max_pixels: int = 8_000_000) -> np.ndarray:
#     h, w = img_rgb.shape[:2]
#     if h * w <= max_pixels:
#         return img_rgb
#     scale = (max_pixels / (h * w)) ** 0.5
#     new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
#     return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

# SCALE_OPTIONS = [2, 3, 4]
# def upscale_ui(image: np.ndarray, scale_factor: int, method: str,
#                weights_2x: str, weights_3x: str, weights_4x: str):
#     if image is None:
#         return None
#     if image.dtype != np.uint8:
#         image = np.clip(image, 0, 255).astype(np.uint8)
#     if image.ndim == 2:
#         image = np.stack([image, image, image], axis=-1)
#     elif image.shape[2] == 4:
#         image = image[..., :3]

#     image = maybe_downscale_for_memory(image, max_pixels=8_000_000)

#     if method == "FSRCNN (Y channel)":
#         weights_map = {2: weights_2x.strip(), 3: weights_3x.strip(), 4: weights_4x.strip()}
#         print("Weights map : ",weights_map,"Scale : ",scale_factor)
#         weights = weights_map.get(scale_factor) or None
#         print("Weights : ",weights)
#         out = fsrcnn_upscale_rgb(image, scale_factor, weights)
#     else:
#         out = bicubic_upscale_rgb(image, scale_factor)

#     return out


# with gr.Blocks(title="FSRCNN Super-Resolution") as demo:
#     gr.Markdown(
#         "# 🧠 FSRCNN Image Upscaling (x2/x3/x4)\n"
#         "- **Tip:** If no valid FSRCNN weights are provided, the app will automatically fall back to Bicubic.\n"
#         "- Provide `.pth` files trained for each scale to see FSRCNN quality improvements."
#     )

#     with gr.Row():
#         with gr.Column():
#             inp_img = gr.Image(type="numpy", label="Input Image")
#             scale = gr.Radio(choices=SCALE_OPTIONS, value=2, label="Upscale factor")
#             method = gr.Radio(choices=["FSRCNN (Y channel)", "Bicubic"],
#                               value="FSRCNN (Y channel)", label="Method")

#             gr.Markdown("**Optional: load FSRCNN weights (.pth) per scale**")
#             weights_2x = gr.Textbox(label="Weights path for x2 (optional)", placeholder="models/fsrcnn_x2.pth")
#             weights_3x = gr.Textbox(label="Weights path for x3 (optional)", placeholder="models/fsrcnn_x3.pth")
#             weights_4x = gr.Textbox(label="Weights path for x4 (optional)", placeholder="models/fsrcnn_x4.pth")

#             run_btn = gr.Button("Upscale")
#         with gr.Column():
#             out_img = gr.Image(type="numpy", label="Upscaled Output")

#     run_btn.click(
#         fn=upscale_ui,
#         inputs=[inp_img, scale, method, weights_2x, weights_3x, weights_4x],
#         outputs=[out_img]
#     )

# if __name__ == "__main__": # pragma: no cover
#     demo.launch()





import os
import math
import typing as tp
import cv2
import numpy as np
import torch
from torch import nn
import gradio as gr


class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4):
        super(FSRCNN, self)._init_()
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


Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_CACHE: dict[int, tuple[FSRCNN, bool]] = {}

# Predefined paths for bundled weights
WEIGHTS_PATHS = {
    2: "models/fsrcnn_x2.pth",
    3: "models/fsrcnn_x3.pth",
    4: "models/fsrcnn_x4.pth"
}


def try_load_weights(model, weights_path):
    if not weights_path or not os.path.isfile(weights_path):
        print(f"[FSRCNN] No valid weights at {weights_path}. Falling back to Bicubic.")
        return False
    try:
        state_dict = model.state_dict()
        checkpoint = torch.load(weights_path, map_location=Device, weights_only=False)
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


def get_model(scale, weights_path=None):
    if scale not in MODEL_CACHE:
        print("Not in cache")
        model = FSRCNN(scale_factor=scale).to(Device).eval()
        has_weights = try_load_weights(model, weights_path)
        MODEL_CACHE[scale] = (model, has_weights)
    else:
        model, has_weights = MODEL_CACHE[scale]
    return MODEL_CACHE[scale]


def rgb_to_ycbcr(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)


def ycbcr_to_rgb(img_ycrcb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)


def run_fsrcnn_on_y(y: np.ndarray, model: FSRCNN) -> np.ndarray:
    y_f = y.astype(np.float32) / 255.0
    tens = torch.from_numpy(y_f).unsqueeze(0).unsqueeze(0).to(Device)  # 1x1xHxW
    with torch.inference_mode():
        out = model(tens)
    out_np = out.squeeze(0).squeeze(0).clamp(0.0, 1.0).cpu().numpy()
    out_u8 = (out_np * 255.0 + 0.5).astype(np.uint8)
    return out_u8


def fsrcnn_upscale_rgb(img_rgb: np.ndarray, scale: int, weights: tp.Optional[str] = None) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    model, has_weights = get_model(scale, weights)
    if not has_weights:
        return bicubic_upscale_rgb(img_rgb, scale)
    ycrcb = rgb_to_ycbcr(img_rgb)
    y = ycrcb[..., 0]
    cr = ycrcb[..., 1]
    cb = ycrcb[..., 2]
    y_sr = run_fsrcnn_on_y(y, model)
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
    h, w = img_rgb.shape[:2]
    if h * w <= max_pixels:
        return img_rgb
    scale = (max_pixels / (h * w)) ** 0.5
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


SCALE_OPTIONS = [2, 3, 4]


def upscale_ui(image: np.ndarray, scale_factor: int, method: str):
    if image is None:
        return None, "Please upload an image."
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    elif image.shape[2] == 4:
        image = image[..., :3]
    image = maybe_downscale_for_memory(image, max_pixels=8_000_000)

    weights_path = WEIGHTS_PATHS.get(scale_factor)
    if method == "FSRCNN (Y channel)":
        out = fsrcnn_upscale_rgb(image, scale_factor, weights_path)
        status = f"Used FSRCNN x{scale_factor} (bundled weights)."
    else:
        out = bicubic_upscale_rgb(image, scale_factor)
        status = f"Used Bicubic x{scale_factor}."
    return out, status


custom_theme = gr.themes.Default().set(
    button_primary_background_fill="#1769aa", # deep blue
    button_primary_text_color="#ffffff", # dark blue
    button_secondary_background_fill="#e0e0e0", # light gray
    button_secondary_text_color="#222222", # very dark gray
)


with gr.Blocks(title="FSRCNN Super-Resolution", theme=custom_theme) as demo:
    gr.Markdown(
        "# 🧠 FSRCNN Image Upscaling (x2/x3/x4)\n"
        "- Use deep learning (FSRCNN) or traditional bicubic interpolation for high-quality image upscaling.\n"
        "- Tip: Pretrained FSRCNN weights for each scale are bundled within the app."
    )

    with gr.Row():
        with gr.Column():
            inp_img = gr.Image(type="numpy", label="Input Image", interactive=True)
            scale = gr.Dropdown(choices=SCALE_OPTIONS, value=2, label="Upscale factor")
            method = gr.Radio(
                choices=["FSRCNN (Y channel)", "Bicubic"],
                value="FSRCNN (Y channel)", label="Upscale Method"
            )
            with gr.Row():
                run_btn = gr.Button("Upscale", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")
            status_box = gr.Textbox(value="", label="Status", interactive=False)
        with gr.Column():
            out_img = gr.Image(type="numpy", label="Upscaled Output")

    run_btn.click(
        fn=upscale_ui,
        inputs=[inp_img, scale, method],
        outputs=[out_img, status_box]
    )

    def clear_all():
        return None, 2, "FSRCNN (Y channel)", None, ""
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[inp_img, scale, method, out_img, status_box]
    )


if __name__ == "__main__": # pragma: no cover
    demo.launch()
