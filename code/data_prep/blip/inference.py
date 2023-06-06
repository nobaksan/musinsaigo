import io
import json
from typing import Any, Dict
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,  # noqa
)

MAX_NEW_TOKENS = 50


def model_fn(model_dir: str) -> Dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_dir)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_dir,
        torch_dtype=torch.float16,
    )
    _ = model.to(device)

    return {
        "processor": processor,
        "model": model,
    }


def transform_fn(
    model: Dict[str, Any],
    input_data: bytes,
    content_type: str,
    accept: str,
    verbose: bool = True,
) -> str:
    text = None
    data = {
        "images": Image.open(io.BytesIO(input_data)),
        "text": text,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = model["processor"]
    model = model["model"]

    inputs = processor(
        images=data["images"], text=data["text"], return_tensors="pt"
    ).to(device, torch.float16)
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()

    if verbose:
        print(f"generated text: {generated_text}")

    return json.dumps({"text": generated_text})
