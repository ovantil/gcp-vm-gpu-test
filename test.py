import os
import cv2
import json
import time
import torch
import open_clip
import numpy as np

from PIL import Image
from rich import print as rprint
from sentence_transformers import util

rprint("[bold]Loading model...[/bold]")
load_start_time = time.time()
model_name = "ViT-H-14"
pretrained = "laion2b_s32b_b79k"
device = "cuda" if torch.cuda.is_available() else "mps"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained)
model.to(device)
rprint(f"-> Model loaded in {time.time() - load_start_time:.2f} seconds")

base_image_folder = "./data/base_images"
base_image_paths = [
    os.path.join(base_image_folder, img_name)
    for img_name in os.listdir(base_image_folder)
]

BATCH_SIZE = 256

def load_and_preprocess_images(image_paths):
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        imgs = [
            cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            for img_path in batch_paths
        ]
        imgs = [Image.fromarray(img).convert("RGB") for img in imgs]
        imgs = torch.stack([preprocess(img) for img in imgs]).to(device)
        yield imgs


def encode_images(image_paths):
    rprint(f"-> Encoding {len(image_paths)} images...")
    start_time = time.time()
    if not isinstance(image_paths, list):
        image_paths = [
            os.path.join(base_image_folder, img_name)
            for img_name in os.listdir(base_image_folder)
        ]

    encoded_images = []
    for batch_imgs in load_and_preprocess_images(image_paths):
        with torch.no_grad():
            batch_encoded_imgs = model.encode_image(batch_imgs)
            encoded_images.append(batch_encoded_imgs)
    
        rprint(f"-> Encoded {len(batch_imgs)} images in {time.time() - start_time:.2f} seconds")
    return torch.cat(encoded_images)


rprint("[bold]Encoding item images...[/bold]")
encode_start_time = time.time()
item_image_folders_path = "./data/item_images"
item_image_folders = os.listdir(item_image_folders_path)
image_paths = []
for folder in item_image_folders:
    folder_path = os.path.join(item_image_folders_path, folder)
    image_paths.extend(
        [os.path.join(folder_path, img) for img in os.listdir(folder_path)]
    )
encoded_item_images = encode_images(image_paths)
rprint(f"-> Encoded {len(image_paths)} images in {time.time() - encode_start_time:.2f} seconds")