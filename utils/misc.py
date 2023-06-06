import base64
import json
import logging
import os
import tarfile
import zipfile
from io import BytesIO
from typing import Any, List, Optional
import boto3
import matplotlib.pyplot as plt
from PIL import Image


def compress_dir_to_model_tar_gz(
    tar_dir: Optional[str] = None,
    output_file: str = "model.tar.gz",
    logger: Optional[logging.Logger] = None,
) -> None:
    parent_dir = os.getcwd()
    os.chdir(tar_dir)

    msg = "The following directories and files will be compressed."
    if logger is None:
        print(msg)
    else:
        logger.info(msg)

    with tarfile.open(os.path.join(parent_dir, output_file), "w:gz") as tar:
        for item in os.listdir("."):
            if logger is None:
                print(item)
            else:
                logger.info(item)
            tar.add(item, arcname=item)

    os.chdir(parent_dir)


def create_bucket_if_not_exists(
    boto_session: boto3.Session,
    region_name: str,
    logger: Optional[logging.Logger] = None,
):
    s3_client = boto_session.client("s3")
    sts_client = boto_session.client("sts")
    account_id = sts_client.get_caller_identity()["Account"]

    bucket_name = f"sagemaker-{region_name}-{account_id}"
    if (
        s3_client.head_bucket(Bucket=bucket_name)["ResponseMetadata"]["HTTPStatusCode"]
        == 404
    ):
        s3_client.create_bucket(Bucket=bucket_name)
        msg = f"The following S3 bucket was created: {bucket_name}"
        if logger is None:
            print(msg)
        else:
            logger.info(msg)
    else:
        msg = f"The following S3 bucket was found: {bucket_name}"
        if logger is None:
            print(msg)
        else:
            logger.info(msg)

    return bucket_name


def create_role_if_not_exists(
    boto_session: boto3.Session,
    region_name: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    iam_client = boto_session.client("iam")

    role_name = f"AmazonSageMaker-ExecutionRole-{region_name}"
    try:
        role = iam_client.get_role(RoleName=role_name)
        msg = f"The following IAM role was found: {role['Role']['Arn']}"
        if logger is None:
            print(msg)
        else:
            logger.info(msg)

    except iam_client.exceptions.NoSuchEntityException:
        assume_role_policy_document = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                }
            ],
        }
        role = iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(assume_role_policy_document),
            Description="SageMaker Execution Role",
        )
        policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
        iam_client.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)

        msg = f"The following IAM role was created: {role['Role']['Arn']}"
        if logger is None:
            print(msg)
        else:
            logger.info(msg)

    return role_name


def encode_base64_image(file_name: str) -> str:
    with open(file_name, "rb") as image:
        image_string = base64.b64encode(bytearray(image.read())).decode()
    return image_string


def decode_base64_image(image_string: str) -> Any:
    base64_image = base64.b64decode(image_string)
    buffer = BytesIO(base64_image)
    return Image.open(buffer)


def decompress_file(
    source_file_path: str, target_dir: str, compression: str = "zip"
) -> None:
    if compression == "zip":
        with zipfile.ZipFile(source_file_path) as file:
            file.extractall(target_dir)
    elif compression == "tar":
        with tarfile.open(source_file_path) as file:
            file.extractall(target_dir)
    else:
        msg = "The argument, 'compression' should be 'zip or 'tar'."
        ValueError(msg)


def delete_files_in_s3(
    boto_session: boto3.Session,
    bucket_name: str,
    prefix: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    s3_resource = boto_session.resource("s3")
    bucket = s3_resource.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=prefix):
        s3_resource.Object(bucket_name, obj.key).delete()
        msg = f"The 's3://{bucket_name}/{obj.key}' file has been deleted."
        if logger is None:
            print(msg)
        else:
            logger.info(msg)


def display_images(
    images: Optional[List[Any]] = None,
    columns: int = 3,
    width: int = 100,
    height: int = 100,
) -> None:
    plt.figure(figsize=(width, height))
    for i, image in enumerate(images):
        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.axis("off")
        plt.imshow(image)


def display_image_grid(imgs: Any, rows: int, cols: int) -> Any:
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def make_s3_uri(bucket: str, prefix: str, filename: Optional[str] = None) -> str:
    prefix = prefix if filename is None else prefix + "/" + filename
    return f"s3://{bucket}/{prefix}"


def upload_dir_to_s3(
    boto_session: boto3.Session,
    local_dir: str,
    bucket: str,
    prefix: str,
    file_ext_to_excl: Optional[List[str]] = None,
    public_readable: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    s3_client = boto_session.client("s3")
    file_ext_to_excl = [] if file_ext_to_excl is None else file_ext_to_excl
    extra_args = {"ACL": "public-read"} if public_readable else {}

    for root, _, files in os.walk(local_dir):
        for file in files:
            if file.split(".")[-1] not in file_ext_to_excl:
                s3_client.upload_file(
                    os.path.join(root, file),
                    bucket,
                    f"{prefix}/{file}",
                    ExtraArgs=extra_args,
                )
                msg = f"The '{file}' file has been uploaded to 's3://{bucket}/{prefix}/{file}'."
                if logger is None:
                    print(msg)
                else:
                    logger.info(msg)
