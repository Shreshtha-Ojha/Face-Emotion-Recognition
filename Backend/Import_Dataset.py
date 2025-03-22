import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
import kagglehub
import shutil

# Download and load the dataset
print("Downloading dataset...")
dataset_path = kagglehub.dataset_download("msambare/fer2013")


# Define folder structure
outer_names = ['test', 'train']
inner_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Create folders for data storage
print("Creating directories...")
os.makedirs('data', exist_ok=True)
for outer_name in outer_names:
    for inner_name in inner_names:
        os.makedirs(os.path.join('data', outer_name, inner_name), exist_ok=True)

# Initialize counters for each emotion
counts = { 'train': {name: 0 for name in inner_names}, 'test': {name: 0 for name in inner_names} }

# Prepare images
def copy_images(src_folder, dest_folder):
    for inner_name in inner_names:
        src_path = os.path.join(src_folder, inner_name)
        dest_path = os.path.join(dest_folder, inner_name)

        if not os.path.exists(src_path):
            print(f" Warning: {src_path} not found. Skipping...")
            continue

        # Copy images from source to destination
        for img_name in tqdm(os.listdir(src_path), desc=f"Processing {inner_name}"):
            src_img_path = os.path.join(src_path, img_name)
            dest_img_path = os.path.join(dest_path, img_name)
            shutil.copy(src_img_path, dest_img_path)

# Copy training images
print("\nCopying training images...")
train_path = os.path.join(dataset_path, "train")
copy_images(train_path, 'data/train')

# Copy testing images
print("\nCopying testing images...")
test_path = os.path.join(dataset_path, "test")
copy_images(test_path, 'data/test')

print("\n Dataset preparation complete! Images are saved in the 'data/' directory.")
