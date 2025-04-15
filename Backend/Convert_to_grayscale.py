from PIL import Image
import os
from tqdm import tqdm

def convert_to_grayscale(data_dir='data'):
    for split in ['train', 'test']:
        for emotion in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']:
            folder = os.path.join(data_dir, split, emotion)
            for img_name in tqdm(os.listdir(folder), desc=f"Converting {split}/{emotion}"):
                img_path = os.path.join(folder, img_name)
                try:
                    img = Image.open(img_path)
                    if img.mode != 'L':
                        gray_img = img.convert('L')
                        gray_img.save(img_path)
                except Exception as e:
                    print(f"Failed to convert {img_path}: {e}")

if __name__ == "__main__":
    convert_to_grayscale()
