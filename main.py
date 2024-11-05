from PIL import Image
from dotenv import load_dotenv
from pathlib import Path
from roboflow import Roboflow
import ast
import boto3
import os
import pathlib
import shutil
import time
import torch
import torchvision.transforms as transforms
import yaml
import zipfile

# ---------------

load_dotenv()

sqs = boto3.client("sqs", region_name="us-east-1")
s3 = boto3.client("s3")

ROBOFLOW_KEY = os.getenv("ROBOFLOW_KEY")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# ---------------

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ],
)


def process_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transformed_image = transform(image)
    assert isinstance(image_path, pathlib.PosixPath)
    save_path = image_path.with_name(
        f"{image_path.stem}_transformed{image_path.suffix}"
    )
    transformed_image_pil = transforms.ToPILImage()(transformed_image)
    transformed_image_pil.save(save_path)
    return save_path


def download_dataset_from_roboflow(url):
    parts = url.split("/")
    ds_version = parts[-1]
    ds_project = parts[-3]
    ds_workspace = parts[-4]
    rf = Roboflow(api_key=ROBOFLOW_KEY)
    project = rf.workspace(ds_workspace).project(ds_project)
    version = project.version(ds_version)
    dataset = version.download("yolov11")
    return Path(dataset.location)


def upload_to_s3(local_path, s3_path, zip_name="upload.zip"):
    zip_path = os.path.join("/tmp", zip_name)  # Temporary path for the zip file
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, local_path))
    print(f"Zipped {local_path} to {zip_path}")

    s3_key = os.path.join(s3_path, zip_name)

    s3.upload_file(zip_path, S3_BUCKET_NAME, s3_key)
    print(f"Uploaded {zip_path} to s3://{S3_BUCKET_NAME}/{s3_key}")


def process_and_upload_dataset(url):
    dataset_dir = download_dataset_from_roboflow(url)

    # # Process each image in train, valid, test folders
    # for folder in ["train", "valid", "test"]:
    #     folder_path = dataset_dir / folder
    #     images_dir = folder_path / "images"
    #     labels_dir = folder_path / "labels"
    #
    #     transformed_folder_path = dataset_dir / f"{folder}_transformed"
    #     transformed_folder_path.mkdir(exist_ok=True)
    #
    #     transformed_images_dir = transformed_folder_path / "images"
    #     transformed_labels_dir = transformed_folder_path / "labels"
    #
    #     transformed_images_dir.mkdir(exist_ok=True)
    #     transformed_labels_dir.mkdir(exist_ok=True)
    #
    #     image_paths = [
    #         f
    #         for f in os.listdir(images_dir)
    #         if f.lower().endswith(
    #             (
    #                 ".jpg",
    #                 ".jpeg",
    #                 ".png",
    #                 ".ppm",
    #                 ".bmp",
    #                 ".pgm",
    #                 ".tif",
    #                 ".tiff",
    #                 ".webp",
    #             )
    #         )
    #     ]
    #
    #     for image_path in image_paths:
    #         transformed_image_path = process_image(images_dir / image_path)
    #         shutil.move(
    #             transformed_image_path,
    #             transformed_images_dir / Path(transformed_image_path).name,
    #         )
    #         shutil.move(
    #             labels_dir / f"{Path(image_path).stem}.txt",
    #             transformed_labels_dir / f"{Path(transformed_image_path).stem}.txt",
    #         )

    # Upload processed dataset to S3
    upload_to_s3(dataset_dir, "dataset")


def listen_to_sqs():
    while True:
        response = sqs.receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10,
        )

        if "Messages" in response:
            message = response["Messages"][0]
            receipt_handle = message["ReceiptHandle"]
            body = ast.literal_eval(message["Body"])

            try:
                # Process the dataset
                process_and_upload_dataset(body["roboflow_url"])

                # Delete message after successful processing
                sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
                print("Processed and deleted message from SQS.")

            except Exception as e:
                print(f"Error processing message: {e}")
        else:
            print("No messages in queue. Waiting...")
        time.sleep(5)  # Poll every 5 seconds


if __name__ == "__main__":
    listen_to_sqs()
