from PIL import Image
from dotenv import load_dotenv
from pathlib import Path
from roboflow import Roboflow
import ast
import boto3
import json
import os
import pathlib
import random
import shutil
import time
import torch
import torchvision.transforms as transforms
import wget
import yaml
import zipfile
import time

# ---------------

load_dotenv()

sqs = boto3.client("sqs", region_name="us-east-1")
s3 = boto3.client("s3")
sns = boto3.client("sns", region_name="us-east-1")

ROBOFLOW_KEY = os.getenv("ROBOFLOW_KEY")
SQS_QUEUE_URL = os.getenv("SQS_QUEUE_URL")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
SNS_ARN = os.getenv("SNS_ARN")
DEPLOYMENT = os.getenv("DEPLOYMENT")

TYPE_ROBOFLOW = ("roboflow",)
TYPE_ZIPFILE = "zipfile"
TYPE_VISDRONE = "visdrone"
input_types = {TYPE_ROBOFLOW, TYPE_ZIPFILE, TYPE_VISDRONE}

ROBOFLOW_YOLOV11 = "yolov11"
ROBOFLOW_YOLOV8 = "yolov8"
ROBOFLOW_DETECTRON = "coco"

ROBOFLOW_SUPPORTED_DATASETS = {ROBOFLOW_YOLOV11, ROBOFLOW_YOLOV8, ROBOFLOW_DETECTRON}

# ---------------


def send_sns(subject, message):
    try:
        sns.publish(
            TargetArn=SNS_ARN,
            Message=message,
            Subject=subject,
        )

    except Exception as e:
        print("Failed to send message")
        pass


def print_timestamp(*args, **kwargs):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime()):^25}]", end=" ")
    print(*args, **kwargs)


def convert_box_to_yolo(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    return (
        (box[0] + box[2] / 2) * dw,
        (box[1] + box[3] / 2) * dh,
        box[2] * dw,
        box[3] * dh,
    )


def visdrone2yolo(dir: Path, names):
    (dir / "labels").mkdir(parents=True, exist_ok=True)  # make labels directory
    for f in (dir / "annotations").glob("*.txt"):
        img_size = Image.open((dir / "images" / f.name).with_suffix(".jpg")).size
        lines = []
        with open(f, "r") as file:  # read annotation.txt
            for row in [x.split(",") for x in file.read().strip().splitlines()]:
                if row[4] == "0":  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1
                box = convert_box_to_yolo(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(
                    str(f).replace(
                        f"{os.sep}annotations{os.sep}", f"{os.sep}labels{os.sep}"
                    ),
                    "w",
                ) as fl:
                    fl.writelines(lines)  # write label.txt

    splits = ["test", "train", "valid"]
    for split in splits:
        (dir / split / "images").mkdir(parents=True, exist_ok=True)
        (dir / split / "labels").mkdir(parents=True, exist_ok=True)

    dataset_len = len(list((dir / "labels").glob("*.txt")))
    train_len = int(0.8 * dataset_len)
    valtest_len = dataset_len - train_len
    val_len = int(valtest_len / 2)
    test_len = valtest_len - val_len

    counts = {"test": 0, "train": 0, "valid": 0}
    max_counts = {"test": test_len, "train": train_len, "valid": val_len}
    for f in (dir / "labels").glob("*.txt"):
        i = (dir / "images" / f.name).with_suffix(".jpg")
        split = random.choice(splits)
        shutil.move(f, dir / split / "labels")
        shutil.move(i, dir / split / "images")
        counts[split] += 1
        if counts[split] == max_counts[split]:
            splits.remove(split)

    yaml_file = {
        "train": "../train/images",
        "val": "../valid/images",
        "test": "../test/images",
        "nc": len(names),
        "names": names,
    }

    with open(dir / "data.yaml", "w") as file:
        yaml.dump(yaml_file, file)

    shutil.rmtree(dir / "annotations")
    shutil.rmtree(dir / "labels")
    shutil.rmtree(dir / "images")


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
    zip_path = os.path.join("/tmp", zip_name)  # Temporary path for the zip file
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(local_path):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, local_path))
    print_timestamp(f"Zipped {local_path} to {zip_path}")

    if DEPLOYMENT == "dev":
        print_timestamp("Not uploading in dev env")
        return
    s3_key = os.path.join(s3_path, zip_name)

    s3.upload_file(zip_path, S3_BUCKET_NAME, s3_key)
    print_timestamp(f"Uploaded {zip_path} to s3://{S3_BUCKET_NAME}/{s3_key}")


def process_and_upload_dataset(url, dtype, names=None):
    if dtype not in input_types:
        print_timestamp(f"{dtype} download type not supported")

    if dtype == TYPE_ROBOFLOW:
        print_timestamp(f"{dtype} support in progress")
        return
        # for dl_format in ROBOFLOW_SUPPORTED_DATASETS:
        #     dataset_dir = download_dataset_from_roboflow(url, dl_format)
        #     upload_to_s3(dataset_dir, "dataset", zip_name=f"{dl_format}.zip")

    elif dtype == TYPE_ZIPFILE:
        print_timestamp(f"{dtype} support in progress")
        return

    elif dtype == TYPE_VISDRONE:
        if names is None:
            print_timestamp("Names are required for visdrone")
            return
        print_timestamp("Downloading original dataset")
        wget.download(url=url, out="visdrone.zip", bar=None)
        with zipfile.ZipFile("visdrone.zip", "r") as zipf:
            dir_name = Path("./" + zipf.namelist()[0])
            print_timestamp()
            print_timestamp("Unzipped to ", dir_name)
            zipf.extractall()
        print_timestamp("Converting to YOLO format")
        visdrone2yolo(dir_name, names)
        upload_to_s3(dir_name, "dataset", zip_name="yolo.zip")
        splits = ["test", "train", "valid"]
        print_timestamp("Converting to COCO format")
        for split in splits:
            yolo_to_coco(
                dir_name / split / "images",
                dir_name / split / "labels",
                dir_name / split / "images/_annotations.coco.json",
                names,
            )
            shutil.rmtree(dir_name / split / "labels")
            for file_path in (dir_name / split / "images").glob("*"):
                shutil.move(str(file_path), str(dir_name / split))
            os.rmdir(dir_name / split / "images")
        os.remove(dir_name / "data.yaml")
        upload_to_s3(dir_name, "dataset", zip_name="coco.zip")
        os.remove("visdrone.zip")
        shutil.rmtree(dir_name)
        print_timestamp("Done")
        send_sns(
            "Converted dataset",
            f"""Converted dataset from {url}
timestamp: {time.time()}
datasets location: {S3_BUCKET_NAME}/datasets/""",
        )


def trigger_training(model):
    ec2 = boto3.client("ec2", region_name="us-east-1")

    # Define User Data script
    user_data_script = f"""#!/bin/bash
sudo apt update -y
sudo apt upgrade -y
sudo apt install python3-full python3-pip git -y
git clone https://github.com/sjsu2024-data298-team6/trainer /home/ubuntu/trainer
cd /home/ubuntu/trainer
echo "DEPLOYMENT=prod\nS3_BUCKET_NAME={S3_BUCKET_NAME}\nSNS_ARN={SNS_ARN}\nMODEL_TO_TRAIN={model}" >> .env
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 main.py
sudo shutdown -h now
    """

    # Launch EC2 instance
    response = ec2.run_instances(
        ImageId="ami-0e2c8caa4b6378d8c",
        InstanceType="g5.2xlarge",
        InstanceInitiatedShutdownBehavior="terminate",
        KeyName="sjsu-fall24-data298-team6-key-pair",
        MinCount=1,
        MaxCount=1,
        UserData=user_data_script,
        IamInstanceProfile={
            "Arn": os.getenv("EC2_INSTANCE_IAM_ARM"),
        },
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "Encrypted": False,
                    "DeleteOnTermination": True,
                    "Iops": 3000,
                    "SnapshotId": "snap-0ea137085731e5c98",
                    "VolumeSize": 30,
                    "VolumeType": "gp3",
                    "Throughput": 125,
                },
            }
        ],
        NetworkInterfaces=[
            {
                "AssociatePublicIpAddress": True,
                "DeviceIndex": 0,
                "Groups": [
                    "sg-0ae6a08ce3772678c",
                ],
            },
        ],
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [
                    {
                        "Key": "Name",
                        "Value": f"sfdt-trainer-{model}",
                    },
                ],
            },
        ],
    )

    instance_id = response["Instances"][0]["InstanceId"]
    print_timestamp(f"Trainer EC2 instance launched: {instance_id}")


def yolo_to_coco(image_dir, label_dir, output_path, categories):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": name} for i, name in enumerate(categories)],
    }

    # Initialize annotation id
    ann_id = 0

    # Loop through all images
    for img_id, img_name in enumerate(os.listdir(image_dir)):
        if not img_name.endswith((".jpg", ".jpeg", ".png")):
            continue

        # Get image path
        img_path = os.path.join(image_dir, img_name)

        # Open image to get dimensions
        img = Image.open(img_path)
        width, height = img.size

        # Add image info to COCO format
        coco_format["images"].append(
            {"id": img_id, "file_name": img_name, "width": width, "height": height}
        )

        # Get corresponding label file
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(label_dir, label_name)

        if not os.path.exists(label_path):
            continue

        # Read YOLO annotations
        with open(label_path, "r") as f:
            label_lines = f.readlines()

        # Convert YOLO annotations to COCO format
        for line in label_lines:
            class_id, x_center, y_center, bbox_width, bbox_height = map(
                float, line.strip().split()
            )

            # Convert YOLO coordinates to COCO coordinates
            x = (x_center - bbox_width / 2) * width
            y = (y_center - bbox_height / 2) * height
            w = bbox_width * width
            h = bbox_height * height

            # Add annotation to COCO format
            coco_format["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(class_id),
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    # Save to JSON file
    with open(output_path, "w") as f:
        json.dump(coco_format, f, indent=2)


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
            url = body["url"]
            dtype = body["dtype"]
            model = body["model"]
            names = ast.literal_eval(body["names"]) if body["names"] != "none" else None

            try:
                # Process the dataset
                process_and_upload_dataset(url=url, dtype=dtype, names=names)
                trigger_training(model)

                # Delete message after successful processing
                sqs.delete_message(QueueUrl=SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
                print_timestamp("Processed and deleted message from SQS.")

            except Exception as e:
                print_timestamp(f"Error processing message: {e}")
        else:
            print_timestamp("No messages in queue. Waiting...")
        time.sleep(5)  # Poll every 5 seconds


if __name__ == "__main__":
    if DEPLOYMENT == "dev":
        process_and_upload_dataset(
            "file:///mnt/d/datasets/VisDroneSmall.zip",
            dtype=TYPE_VISDRONE,
            names=[
                "pedestrian",
                "people",
                "bicycle",
                "car",
                "van",
                "truck",
                "tricycle",
                "awning-tricycle",
                "bus",
                "motor",
            ],
        )
        trigger_training("yolo")
    else:
        listen_to_sqs()
