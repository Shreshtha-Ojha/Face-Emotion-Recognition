from tqdm import tqdm
import os
import kagglehub
import shutil

# Download and load the dataset
print("Downloading dataset...")
dataset_path = kagglehub.dataset_download("msambare/fer2013")


raf_path =kagglehub.dataset_download("shuvoalok/raf-db-dataset")


# Define folder structure
outer_names = ['test', 'train']
inner_names1 = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
inner_names2 = ['6', '3', '2', '4', '5', '1', '7']
# Create folders for data storage
print("Creating directories...")
os.makedirs('data', exist_ok=True)
for outer_name in outer_names:
    for inner_name in inner_names1:
        os.makedirs(os.path.join('data', outer_name, inner_name), exist_ok=True)

# Initialize counters for each emotion
counts = { 'train': {name: 0 for name in inner_names1}, 'test': {name: 0 for name in inner_names1} }


# Prepare images
def copy_images(src_folder, dest_folder):
    for inner_name in inner_names1:
        src_path = os.path.join(src_folder, inner_name)
        dest_path = os.path.join(dest_folder, inner_name)

        if not os.path.exists(src_path):
            print(f"Warning: {src_path} not found. Skipping...")
            continue

        # Copy images from source to destination
        for img_name in tqdm(os.listdir(src_path), desc=f"Processing {inner_name}"):
            src_img_path = os.path.join(src_path, img_name)
            dest_img_path = os.path.join(dest_path, img_name)
            shutil.copy(src_img_path, dest_img_path)

def copy_images2(src_folder, dest_folder):
    for i in range(7):
        src_path = os.path.join(src_folder, inner_names2[i])
        dest_path = os.path.join(dest_folder, inner_names1[i])

        if not os.path.exists(src_path):
            print(f" Warning: {src_path} not found. Skipping...")
            continue

        # Copy images from source to destination
        for img_name in tqdm(os.listdir(src_path), desc=f"Processing {inner_names1[i]}"):
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

print("\nCopying RAF-DB images...")
raf_path=os.path.join(raf_path,"DATASET")
raf_train = os.path.join(raf_path, "train")
raf_test = os.path.join(raf_path, "test")

copy_images2(raf_train, 'data/train')
copy_images2(raf_test, 'data/test')

print("\n Dataset preparation complete! Here's the image count per category:")
for outer in outer_names:
    for emotion in inner_names1:
        folder = os.path.join("data", outer, emotion)
        count = len(os.listdir(folder))
        print(f"{outer}/{emotion}: {count} images")

