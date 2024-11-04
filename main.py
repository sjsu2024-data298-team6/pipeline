import boto3
import time
import os
from roboflow import Roboflow
import torchvision.transforms as transforms
from PIL import Image
import torch
from pathlib import Path
import shutil
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set up SQS and S3 clients
sqs = boto3.client("sqs", region_name="us-east-1")
s3 = boto3.client("s3")

# Environment variables (set these before running the script or hard-code them here)
ROBOFLOW_KEY = os.getenv("ROBOFLOW_KEY")  # Your Roboflow API key
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")  # The URL of the SQS queue
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")  # The name of the S3 bucket

# Torchvision transformation pipeline
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ],
)


def process_image(image_path):
    """Applies torchvision transformations to an image and saves it."""
    image = Image.open(image_path).convert("RGB")
    transformed_image = transform(image)
    save_path = str(image_path).replace(".jpg", "_transformed.jpg")  # Adjust as needed
    transformed_image_pil = transforms.ToPILImage()(transformed_image)
    transformed_image_pil.save(save_path)
    return save_path


def download_dataset_from_roboflow():
    """Downloads the dataset from Roboflow and returns the path of the downloaded directory."""
    rf = Roboflow(api_key=ROBOFLOW_KEY)
    project = rf.workspace("drone-obstacle-detection").project(
        "drone-object-detection-yhpn6"
    )
    version = project.version(15)
    dataset = version.download("yolov11")
    return Path(dataset.location)


def upload_to_s3(local_path, s3_path):
    """Uploads files in a directory to S3."""
    for root, dirs, files in os.walk(local_path):
        for file in files:
            file_path = os.path.join(root, file)
            s3_key = os.path.join(s3_path, os.path.relpath(file_path, local_path))
            s3.upload_file(file_path, S3_BUCKET_NAME, s3_key)
            print(f"Uploaded {file_path} to s3://{S3_BUCKET_NAME}/{s3_key}")


def process_and_upload_dataset():
    """Main function that downloads, processes, and uploads dataset to S3."""
    # Download dataset
    dataset_dir = download_dataset_from_roboflow()
    print("Downloaded dataset from Roboflow.")

    # Process each image in train, valid, test folders
    for folder in ["train", "valid", "test"]:
        folder_path = dataset_dir / folder
        transformed_folder_path = dataset_dir / f"{folder}_transformed"
        transformed_folder_path.mkdir(exist_ok=True)

        for image_path in folder_path.glob(
            "*.jpg"
        ):  # Assuming .jpg files; adjust as needed
            transformed_image_path = process_image(image_path)
            shutil.move(
                transformed_image_path,
                transformed_folder_path / Path(transformed_image_path).name,
            )

    # Upload processed dataset to S3
    upload_to_s3(dataset_dir, "processed_dataset")


def listen_to_sqs():
    """Listens to the SQS queue for messages containing the Roboflow URL."""
    while True:
        response = sqs.receive_message(
            QueueUrl=SQS_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=10,  # Long polling
        )

        if "Messages" in response:
            message = response["Messages"][0]
            receipt_handle = message["ReceiptHandle"]

            try:
                # Process the dataset
                process_and_upload_dataset()

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
