from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from PIL import Image
import os, shutil, random
from tqdm import tqdm

def balance_dataset(data_dir='data', target_samples=8000, output_dir='balanced_data'):
    emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    splits = ['train', 'test']

    os.makedirs(output_dir, exist_ok=True)
    for split in splits:
        for emotion in emotions:
            os.makedirs(os.path.join(output_dir, split, emotion), exist_ok=True)

    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        brightness_range=[0.8, 1.2]
    )

    for split in splits:
        print(f"\nProcessing {split} set:")
        actual_target = target_samples if split == 'train' else int(target_samples * 0.25)

        for emotion in emotions:
            src_dir = os.path.join(data_dir, split, emotion)
            dst_dir = os.path.join(output_dir, split, emotion)

            orig_images = os.listdir(src_dir)
            num_orig = len(orig_images)
            print(f"  {emotion}: {num_orig} original, target: {actual_target}")

            if num_orig < actual_target:
                for img_name in orig_images:
                    shutil.copy(os.path.join(src_dir, img_name), os.path.join(dst_dir, img_name))

                to_generate = actual_target - num_orig
                aug_source_images = random.choices(orig_images, k=to_generate)

                for i, img_name in enumerate(tqdm(aug_source_images, desc=f"Augmenting {emotion}")):
                    img = Image.open(os.path.join(src_dir, img_name)).convert('L')
                    img_array = np.array(img)
                    img_array = np.expand_dims(img_array, axis=-1)
                    img_array = np.expand_dims(img_array, axis=0)
                    aug_img = next(datagen.flow(img_array, batch_size=1))[0].astype(np.uint8)
                    Image.fromarray(aug_img.squeeze(), mode='L').save(os.path.join(dst_dir, f"aug_{i}_{img_name}"))

            else:
                selected = random.sample(orig_images, actual_target)
                for img_name in tqdm(selected, desc=f"Copying {emotion}"):
                    shutil.copy(os.path.join(src_dir, img_name), os.path.join(dst_dir, img_name))

    print("\nBalanced dataset complete!\nStats:")
    for split in splits:
        print(f"{split.upper()} set:")
        for emotion in emotions:
            count = len(os.listdir(os.path.join(output_dir, split, emotion)))
            print(f"  {emotion}: {count} images")

if __name__ == "__main__":
    balance_dataset(target_samples=8000)
