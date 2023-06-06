import base64
from io import BytesIO
from typing import Any, Dict, List, Union
import torch
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline  # noqa
from diffusers.models import AutoencoderKL  # noqa


HF_MODEL_IDS = [
    "SG161222/Realistic_Vision_V2.0",
    "stabilityai/sd-vae-ft-mse",
]


def model_fn(model_dir):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = StableDiffusionPipeline.from_pretrained(
        HF_MODEL_IDS[0], torch_dtype=torch.float16
    ).to(device)

    if len(HF_MODEL_IDS) > 1:
        model.vae = AutoencoderKL.from_pretrained(
            HF_MODEL_IDS[1], torch_dtype=torch.float16
        ).to(device)

    model.scheduler = EulerDiscreteScheduler.from_config(
        model.scheduler.config, use_karras_sigmas=True
    )

    model.unet.load_attn_procs(model_dir)

    return model


def predict_fn(
    data: Dict[str, Union[int, float, str]], model: Any
) -> Dict[str, List[str]]:
    prompt = data.pop("prompt", data)
    height = data.pop("height", 512)
    width = data.pop("width", 512)
    num_inference_steps = data.pop("num_inference_steps", 50)
    guidance_scale = data.pop("guidance_scale", 7.5)
    negative_prompt = data.pop("negative_prompt", None)
    num_images_per_prompt = data.pop("num_images_per_prompt", 4)
    seed = data.pop("seed", 42)
    cross_attention_scale = data.pop("cross_attention_scale", 0.5)

    negative_prompt = None if len(negative_prompt) == 0 else negative_prompt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device).manual_seed(seed)
    generated_images = model(
        prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
        cross_attention_kwargs={"scale": cross_attention_scale},
    )["images"]

    encoded_images = []
    for image in generated_images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_images.append(base64.b64encode(buffered.getvalue()).decode())

    return {"generated_images": encoded_images, "prompt": prompt}
