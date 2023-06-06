import os
from shutil import copyfile
import boto3
import torch
import yaml
from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline
from diffusers.models import AutoencoderKL
from huggingface_hub import create_repo, upload_folder
from utils.logger import logger
from utils.misc import create_bucket_if_not_exists, decompress_file
from utils.torch_utils import bin_to_safetensors, convert_lora_safetensor_to_diffusers

HF_MODEL_IDS = [
    "SG161222/Realistic_Vision_V2.0",
    "stabilityai/sd-vae-ft-mse",
]
CROSS_ATTENTION_SCALE = 0.5


if __name__ == "__main__":
    with open(os.path.join("..", "config", "config.yaml"), encoding="utf-8") as file_path:
        config = yaml.safe_load(file_path)

    profile_name = config["environment"]["iam_profile_name"]
    region_name = config["environment"]["region_name"]
    models_dir = config["environment"]["ebs_models_dir"]
    bucket = config["environment"]["s3_bucket"]
    base_prefix = config["environment"]["s3_base_prefix"]
    hf_token = config["environment"]["hf_token"]

    model_data = config["model"]["model_data"]
    hf_model_id = config["model"]["hf_model_id"]

    # Downloading the fine-tuned model from S3

    boto_session = boto3.Session(profile_name=profile_name, region_name=region_name)
    s3_client = boto_session.client("s3")
    bucket = (
        create_bucket_if_not_exists(boto_session, region_name, logger=logger)
        if bucket is None
        else bucket
    )

    src_model_dir = os.path.join("..", models_dir, "src")
    tgt_model_dir = os.path.join("..", models_dir, "tgt")

    os.makedirs(src_model_dir, exist_ok=True)

    zip_file_path = os.path.join(src_model_dir, "model.tar.gz")

    s3_client.download_file(
        bucket, f"{base_prefix}/models/{model_data}/output/model.tar.gz", zip_file_path
    )
    decompress_file(zip_file_path, src_model_dir, compression="tar")

    # Merging the original model and LoRA weights

    bin_path = os.path.join(src_model_dir, "pytorch_lora_weights.bin")
    safetensors_path = os.path.join(src_model_dir, "pytorch_lora_weights.safetensors")

    bin_to_safetensors(bin_path, safetensors_path)

    model = StableDiffusionPipeline.from_pretrained(
        HF_MODEL_IDS[0], torch_dtype=torch.float32
    )
    model.vae = AutoencoderKL.from_pretrained(
        HF_MODEL_IDS[1], torch_dtype=torch.float32
    )
    model.scheduler = model.scheduler = EulerDiscreteScheduler.from_config(
        model.scheduler.config, use_karras_sigmas=True
    )

    model = convert_lora_safetensor_to_diffusers(
        model, src_model_dir, cross_attention_scale=CROSS_ATTENTION_SCALE
    )

    model.save_pretrained(tgt_model_dir)

    doc_path = os.path.join("scripts", "README.md")
    if os.path.exists(doc_path):
        copyfile(doc_path, os.path.join(tgt_model_dir, "README.md"))

    # Uploading the model to the Huggingface hub

    _ = create_repo(
        repo_id=hf_model_id,
        token=hf_token,
        exist_ok=True,
    )

    _ = upload_folder(
        repo_id=hf_model_id,
        folder_path=tgt_model_dir,
        commit_message="End of training",
        token=hf_token,
        create_pr=True,
        ignore_patterns=["step_*", "epoch_*"],
    )

    logger.info(
        "Downloading the model from S3, merging the weights, and uploading it to the HuggingFace hub '%s' is complete.",
        hf_model_id,
    )
