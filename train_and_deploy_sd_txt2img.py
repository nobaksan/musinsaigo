# Here are some references.
# https://huggingface.co/docs/sagemaker/inference
# https://huggingface.co/docs/diffusers/training/lora
# https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image_lora.py
# https://stable-diffusion-art.com/common-problems-in-ai-images-and-how-to-fix-them/#Garbled_faces_and_eyes_problems
# https://stable-diffusion-art.com/know-these-important-parameters-for-stunning-ai-images/

import os
import random
import shutil
from pathlib import Path
import boto3
import sagemaker
import yaml
from huggingface_hub import snapshot_download
from sagemaker.huggingface import HuggingFace, HuggingFaceModel
from sagemaker.s3 import S3Uploader
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.utils import unique_name_from_base
from utils.logger import logger
from utils.misc import (
    compress_dir_to_model_tar_gz,
    create_bucket_if_not_exists,
    create_role_if_not_exists,
    make_s3_uri,
)

HF_MODEL_IDS = [
    "Salesforce/blip2-opt-2.7b",
    "SG161222/Realistic_Vision_V2.0",
]
TRANSFORMERS_VERSION = "4.26.0"
PYTORCH_VERSION = "1.13.1"
SKLEARN_VERSION = "1.0-1"
PY_VERSION = "py39"
MAX_PAYLOAD = 10
BASE_JOB_NAMES = ["image-prep", "data-prep", "model-train"]
SM_PROC_BASE_DIR = "/opt/ml/processing"

IS_MODEL_ALREADY_UPLOADED = True
SKIP_DATA_PREP = True


if __name__ == "__main__":
    with open(os.path.join("config", "config.yaml"), encoding="utf-8") as file_path:
        config = yaml.safe_load(file_path)

    profile_name = config["environment"]["iam_profile_name"]
    region_name = config["environment"]["region_name"]
    role = config["environment"]["iam_role"]
    bucket = config["environment"]["s3_bucket"]
    base_prefix = config["environment"]["s3_base_prefix"]
    dataset_prefix = config["environment"]["s3_dataset_prefix"]
    hf_token = config["environment"]["hf_token"]
    wandb_api_key = config["environment"]["wandb_api_key"]

    prompt_prefix = config["data"]["prompt_prefix"]
    prompt_suffix = config["data"]["prompt_suffix"]

    model_data = config["model"]["model_data"]
    image_prep_instance_type = config["model"]["image_prep_instance_type"]
    data_prep_instance_type = config["model"]["data_prep_instance_type"]
    infer_instance_type = config["model"]["infer_instance_type"]
    train_instance_type = config["model"]["train_instance_type"]
    num_train_epochs = config["model"]["num_train_epochs"]
    resolution = config["model"]["resolution"]
    center_crop = config["model"]["center_crop"]
    random_flip = config["model"]["random_flip"]
    batch_size = config["model"]["batch_size"]
    max_train_steps = config["model"]["max_train_steps"]
    learning_rate = config["model"]["learning_rate"]
    lr_scheduler = config["model"]["lr_scheduler"]
    push_to_hub = config["model"]["push_to_hub"]
    hf_model_id = config["model"]["hf_model_id"]
    reduce_memory_usage = config["model"]["reduce_memory_usage"]
    endpoint_name = config["model"]["sm_endpoint_name"]

    boto_session = boto3.Session(profile_name=profile_name, region_name=region_name)
    sm_session = sagemaker.session.Session(boto_session=boto_session)
    role = (
        create_role_if_not_exists(boto_session, region_name, logger=logger)
        if role is None
        else role
    )
    bucket = (
        create_bucket_if_not_exists(boto_session, region_name, logger=logger)
        if bucket is None
        else bucket
    )

    train_model_uri = (
        f"s3://{bucket}/{base_prefix}/{HF_MODEL_IDS[0].rsplit('/', maxsplit=1)[-1]}"
    )
    dataset_uri = make_s3_uri(bucket, f"{base_prefix}/{dataset_prefix}")
    artifacts_uri = make_s3_uri(bucket, f"{base_prefix}/artifacts")
    models_uri = make_s3_uri(bucket, f"{base_prefix}/models")

    # Compress and upload the BLIP2 model downloaded from the HuggingFace hub to S3

    if IS_MODEL_ALREADY_UPLOADED:
        train_model_uri += "/model.tar.gz"

    else:
        snapshot_dir = snapshot_download(
            repo_id=HF_MODEL_IDS[0], use_auth_token=hf_token
        )

        model_dir = Path(f"model-{random.getrandbits(16)}")
        model_dir.mkdir(exist_ok=True)

        shutil.copytree(snapshot_dir, str(model_dir), dirs_exist_ok=True)
        shutil.copytree(
            os.path.join("code", "data_prep", "blip") + os.path.sep,
            str(model_dir.joinpath("code")),
        )

        compress_dir_to_model_tar_gz(str(model_dir), logger=logger)
        shutil.rmtree(str(model_dir))

        train_model_uri = S3Uploader.upload(
            local_path="model.tar.gz",
            desired_s3_uri=train_model_uri,
        )

        logger.info("model uploaded to: %s", train_model_uri)

    if model_data is None:
        # Preparing a dataset for image captioning with the BLIP2 model

        if not SKIP_DATA_PREP:
            model = HuggingFaceModel(
                role=role,
                model_data=train_model_uri,
                transformers_version=TRANSFORMERS_VERSION,
                pytorch_version=PYTORCH_VERSION,
                py_version=PY_VERSION,
                sagemaker_session=sm_session,
            )

            transformer = model.transformer(
                instance_count=1,
                instance_type=image_prep_instance_type,
                strategy="SingleRecord",
                output_path=artifacts_uri,
                max_payload=MAX_PAYLOAD,
            )

            job_name = unique_name_from_base(BASE_JOB_NAMES[0]).replace("_", "-")
            transformer.transform(
                data=f"{dataset_uri}/images",
                data_type="S3Prefix",
                content_type="image/x-image",
                job_name=job_name,
                logs=True,
            )

            arguments = ["--base-dir", SM_PROC_BASE_DIR]
            if prompt_prefix is not None:
                arguments.extend(["--prompt-prefix", prompt_prefix])
            if prompt_suffix is not None:
                arguments.extend(["--prompt-suffix", prompt_suffix])

            processor = SKLearnProcessor(
                framework_version=SKLEARN_VERSION,
                role=role,
                instance_count=1,
                instance_type=data_prep_instance_type,
                base_job_name=BASE_JOB_NAMES[1],
                sagemaker_session=sm_session,
            )

            processor.run(
                inputs=[
                    ProcessingInput(
                        source=artifacts_uri,
                        destination=f"{SM_PROC_BASE_DIR}/inputs",
                    )
                ],
                outputs=[
                    ProcessingOutput(
                        source=f"{SM_PROC_BASE_DIR}/output", destination=dataset_uri
                    ),
                ],
                code=os.path.join("code", "data_prep", "data_prep", "data_prep.py"),
                arguments=arguments,
            )

        # Fine-Tuning the Stable Diffusion model with the LoRA technique

        params = {
            "pretrained_model_name_or_path": HF_MODEL_IDS[1],
            "dataloader_num_workers": 8,
            "resolution": 512 if resolution is None else resolution,
            "train_batch_size": 1 if batch_size is None else batch_size,
            "learning_rate": 1e-04 if learning_rate is None else learning_rate,
            "max_grad_norm": 1,
            "lr_scheduler": "cosine" if lr_scheduler is None else lr_scheduler,
            "lr_warmup_steps": 0,
            "checkpointing_steps": 500,
            "seed": 42,
        }

        if num_train_epochs is None and max_train_steps is None:
            params["num_train_epochs"] = 100
        elif num_train_epochs is None:
            params["max_train_steps"] = max_train_steps
        else:
            params["num_train_epochs"] = num_train_epochs

        if center_crop:
            params["center_crop"] = ""
        if random_flip:
            params["random_flip"] = ""

        if push_to_hub and hf_model_id is not None:
            params["push_to_hub"] = ""
            params["hub_token"] = hf_token
            params["hub_model_id"] = hf_model_id

        if reduce_memory_usage:
            # arams["mixed_precision"] = "fp16"
            params["gradient_accumulation_steps"] = 4
            params["enable_xformers_memory_efficient_attention"] = ""

        if wandb_api_key is not None:
            prompt_prefix = "" if prompt_prefix is None else f"{prompt_prefix} "
            prompt_suffix = "" if prompt_suffix is None else f" {prompt_suffix}"
            validation_prompt = f"'{prompt_prefix}a korean woman in a purple shirt and shorts{prompt_suffix}'"

            params["report_to"] = "wandb"
            params["validation_prompt"] = validation_prompt.strip()
            params["wandb_api_key"] = wandb_api_key

        job_name = (
            unique_name_from_base(BASE_JOB_NAMES[2]).replace("_", "-").replace("/", "-")
        )

        estimator = HuggingFace(
            py_version=PY_VERSION,
            entry_point="train.py",
            transformers_version=TRANSFORMERS_VERSION,
            pytorch_version=PYTORCH_VERSION,
            source_dir=os.path.join("code", "train", "sd_txt2img"),
            hyperparameters=params,
            role=role,
            instance_count=1,
            instance_type=train_instance_type,
            output_path=models_uri,
            base_job_name=job_name,
            sagemaker_session=sm_session,
        )

        _ = estimator.fit({"training": dataset_uri + os.path.sep}, logs=True)

        model = estimator.create_model(
            role=role,
            entry_point="inference.py",
            source_dir=os.path.join("code", "inference", "sd_txt2img"),
        )

    else:
        model_data = (
            f"s3://{bucket}/{base_prefix}/models/{model_data}/output/model.tar.gz"
        )

        model = HuggingFaceModel(
            model_data=model_data,
            role=role,
            entry_point="inference.py",
            transformers_version=TRANSFORMERS_VERSION,
            pytorch_version=PYTORCH_VERSION,
            py_version=PY_VERSION,
            source_dir=os.path.join("code", "inference", "sd_txt2img"),
            sagemaker_session=sm_session,
        )

    # Deploying the model to an endpoint

    _ = model.deploy(
        initial_instance_count=1,
        instance_type=infer_instance_type,
        endpoint_name=endpoint_name,
    )

    logger.info(
        "Training the stable diffusion model and deploying an endpoint to %s is complete.",
        endpoint_name,
    )
