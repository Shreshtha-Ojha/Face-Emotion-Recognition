import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import os
import kagglehub

# Download and load the dataset
print("Downloading dataset...")
dataset_path = kagglehub.dataset_download("msambare/fer2013")

# Define CSV path
csv_path = os.path.join(dataset_path, "fer2013.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at {csv_path}")

# Load the dataset
print("Loading dataset...")
df = pd.read_csv(csv_path)

# Function to convert string to integer
def atoi(s):
    return int(s)

# Define folder structure
outer_names = ['test', 'train']
inner_names = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

# Create folders for data storage
print("Creating directories...")
os.makedirs('data', exist_ok=True)
for outer_name in outer_names:
    for inner_name in inner_names:
        os.makedirs(os.path.join('data', outer_name, inner_name), exist_ok=True)

# Initialize counters for each emotion
counts = { 'train': {name: 0 for name in inner_names}, 'test': {name: 0 for name in inner_names} }

# Prepare images
print("Processing and saving images...")
mat = np.zeros((48, 48), dtype=np.uint8)
for i in tqdm(range(len(df))):
    pixels = df['pixels'][i].split()
    emotion = df['emotion'][i]
    mode = 'train' if i < 28709 else 'test'
    
    # Convert pixels to a 48x48 image
    for j in range(2304):
        mat[j // 48][j % 48] = atoi(pixels[j])
    
    img = Image.fromarray(mat)
    emotion_label = inner_names[emotion]
    img_path = os.path.join('data', mode, emotion_label, f"im{counts[mode][emotion_label]}.png")
    img.save(img_path)
    counts[mode][emotion_label] += 1

print("Dataset preparation complete! Images are saved in the 'data/' directory.")