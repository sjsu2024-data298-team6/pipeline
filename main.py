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
DEPLOYMENT = os.getenv("DEPLOYMENT")

TYPE_ROBOFLOW = ("roboflow",)
TYPE_ZIPFILE = "zipfile"
input_types = {TYPE_ROBOFLOW, TYPE_ZIPFILE}

ROBOFLOW_YOLOV11 = "yolov11"
ROBOFLOW_YOLOV8 = "yolov8"
ROBOFLOW_DETECTRON = "coco"

ROBOFLOW_SUPPORTED_DATASETS = {ROBOFLOW_YOLOV11, ROBOFLOW_YOLOV8, ROBOFLOW_DETECTRON}

# ---------------


def download_dataset_from_roboflow(url, dl_format):
    parts = url.split("/")
    ds_version = parts[-1]
    ds_project = parts[-3]
    ds_workspace = parts[-4]
    rf = Roboflow(api_key=ROBOFLOW_KEY)
    project = rf.workspace(ds_workspace).project(ds_project)
    version = project.version(ds_version)
    dataset = version.download(dl_format, location=f"./{dl_format}", overwrite=True)
    return Path(dataset.location)


def upload_to_s3(local_path, s3_path, zip_name="upload.zip"):
    if DEPLOYMENT == "dev":
        print("Not uploading in dev env")
        return
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


def process_and_upload_dataset(url, dtype):
    if dtype not in input_types:
        print(f"{dtype} download type not supported")

    if dtype == TYPE_ROBOFLOW:
        for dl_format in ROBOFLOW_SUPPORTED_DATASETS:
            dataset_dir = download_dataset_from_roboflow(url, dl_format)
            upload_to_s3(dataset_dir, "dataset", zip_name=f"{dl_format}.zip")

    elif dtype == TYPE_ZIPFILE:
        print(f"{dtype} support in progress")
        return


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
                process_and_upload_dataset(body["roboflow_url"], dtype=TYPE_ROBOFLOW)

                # Delete message after successful processing
                sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
                print("Processed and deleted message from SQS.")

            except Exception as e:
                print(f"Error processing message: {e}")
        else:
            print("No messages in queue. Waiting...")
        time.sleep(5)  # Poll every 5 seconds


if __name__ == "__main__":
    if DEPLOYMENT == "dev":
        process_and_upload_dataset(
            "https://universe.roboflow.com/drone-obstacle-detection/drone-object-detection-yhpn6/dataset/73",
            dtype=TYPE_ROBOFLOW,
        )
    else:
        listen_to_sqs()
