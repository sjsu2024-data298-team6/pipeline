from dotenv import load_dotenv
from pathlib import Path
from roboflow import Roboflow
from tqdm import tqdm
import ast
import boto3
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
TYPE_VISDRONE = "visdrone"
input_types = {TYPE_ROBOFLOW, TYPE_ZIPFILE, TYPE_VISDRONE}

ROBOFLOW_YOLOV11 = "yolov11"
ROBOFLOW_YOLOV8 = "yolov8"
ROBOFLOW_DETECTRON = "coco"

ROBOFLOW_SUPPORTED_DATASETS = {ROBOFLOW_YOLOV11, ROBOFLOW_YOLOV8, ROBOFLOW_DETECTRON}

# ---------------


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
    pbar = tqdm((dir / "annotations").glob("*.txt"), desc=f"Converting {dir}")
    for f in pbar:
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
    pbar = tqdm((dir / "labels").glob("*.txt"), desc=f"Performing data split")
    for f in pbar:
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
    print(f"Zipped {local_path} to {zip_path}")

    if DEPLOYMENT == "dev":
        print("Not uploading in dev env")
        return
    s3_key = os.path.join(s3_path, zip_name)

    s3.upload_file(zip_path, S3_BUCKET_NAME, s3_key)
    print(f"Uploaded {zip_path} to s3://{S3_BUCKET_NAME}/{s3_key}")


def process_and_upload_dataset(url, dtype, names=None):
    if dtype not in input_types:
        print(f"{dtype} download type not supported")

    if dtype == TYPE_ROBOFLOW:
        for dl_format in ROBOFLOW_SUPPORTED_DATASETS:
            dataset_dir = download_dataset_from_roboflow(url, dl_format)
            upload_to_s3(dataset_dir, "dataset", zip_name=f"{dl_format}.zip")

    elif dtype == TYPE_ZIPFILE:
        print(f"{dtype} support in progress")
        return

    elif dtype == TYPE_VISDRONE:
        if names is None:
            print("Names are required for visdrone")
            return
        wget.download(url=url, out="visdrone.zip")
        with zipfile.ZipFile("visdrone.zip", "r") as zipf:
            dir_name = Path("./" + zipf.namelist()[0])
            print(os.listdir("./"))
            print(dir_name)
            zipf.extractall()
        visdrone2yolo(dir_name, names)
        upload_to_s3(dir_name, "datasets", zip_name="yolo.zip")
        os.remove("visdrone.zip")
        shutil.rmtree(dir_name)
        print("Done")


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
            names = ast.literal_eval(body["names"]) if body["names"] != "none" else None

            try:
                # Process the dataset
                process_and_upload_dataset(url=url, dtype=dtype, names=names)

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
    else:
        listen_to_sqs()
