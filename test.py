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

load_start_time = time.time()
model_name = "ViT-H-14"
# model_name="ViT-B-32"
pretrained = "laion2b_s32b_b79k"
# pretrained = "laion400m_e32"
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name, pretrained
)
model.to(device)
rprint(f"Model loaded in {time.time() - load_start_time:.2f} seconds")
rprint(f'Using device: {device}')