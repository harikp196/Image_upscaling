import os, time, io
import gradio as gr
from huggingface_hub import InferenceClient
from PIL import Image

HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "stabilityai/stable-diffusion-x4-upscaler")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
client = InferenceClient(token=HF_TOKEN)

def call_inference_image(fileobj, scale: int = 2):
    start = time.time()
    if hasattr(fileobj, "name"):
        img_bytes = open(fileobj.name, "rb").read()
    else:
        img_bytes = fileobj.read()
    params = {"scale": int(scale)} if scale else {}
    res = client.invoke(HF_MODEL_ID, inputs=img_bytes, params=params)
    latency = time.time() - start
    if isinstance(res, (bytes, bytearray)):
        return Image.open(io.BytesIO(res)), f"{latency:.3f}s"
    return str(res), f"{latency:.3f}s"

with gr.Blocks() as demo:
    gr.Markdown("# API-based Upscaler (calls HF Inference API)")
    with gr.Row():
        inp = gr.Image(type="file", label="Upload image")
        scale = gr.Slider(2, 8, value=2, step=1, label="Scale (requested)")
    out_img = gr.Image(label="Result")
    latency = gr.Textbox(label="Latency")
    btn = gr.Button("Upscale via API")
    btn.click(call_inference_image, inputs=[inp, scale], outputs=[out_img, latency])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
