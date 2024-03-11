import os
import cv2
import time
import torch
import open_clip
from PIL import Image
from rich import print as rprint

rprint("[bold]Loading model...[/bold]")
load_start_time = time.time()
model_name = "ViT-H-14"
pretrained = "laion2b_s32b_b79k"
model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])  # Use 4 GPUs
    model.to(device)
else:
    device = torch.device("mps")
    model.to(device)

rprint(f"-> Model loaded in {time.time() - load_start_time:.2f} seconds")

base_image_folder = "./data/base_images"
base_image_paths = [os.path.join(base_image_folder, img_name) for img_name in os.listdir(base_image_folder)]
BATCH_SIZE = 2000

def load_and_preprocess_images(image_paths):
    for i in range(0, len(image_paths), BATCH_SIZE):
        batch_paths = image_paths[i: i + BATCH_SIZE]
        imgs = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in batch_paths]
        imgs = [Image.fromarray(img).convert("RGB") for img in imgs]
        imgs = torch.stack([preprocess(img) for img in imgs]).to(device)
        yield imgs

def encode_images(image_paths):
    rprint(f"-> Encoding {len(image_paths)} images...")
    start_time = time.time()
    encoded_images = []
    for batch_imgs in load_and_preprocess_images(image_paths):
        with torch.no_grad():
            batch_encoded_imgs = model.module.encode_image(batch_imgs)
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
    if os.path.isdir(folder_path):
        image_paths.extend([os.path.join(folder_path, img) for img in os.listdir(folder_path)])
encoded_item_images = encode_images(image_paths)
rprint(f"-> Encoded {len(image_paths)} images in {time.time() - encode_start_time:.2f} seconds")
