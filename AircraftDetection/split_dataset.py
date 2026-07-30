import os
import random
import shutil

random.seed(42)

BASE_DIR = "AircraftDetection"

IMAGES_DIR = os.path.join(BASE_DIR, "images")
LABELS_DIR = os.path.join(BASE_DIR, "labels")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")

# Create folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)

# Get all image names
images = [
    img for img in os.listdir(IMAGES_DIR)
    if img.endswith(".jpg")
]

random.shuffle(images)

total = len(images)

train_end = int(total * 0.7)
val_end = int(total * 0.9)

train = images[:train_end]
val = images[train_end:val_end]
test = images[val_end:]

print(f"Train : {len(train)}")
print(f"Validation : {len(val)}")
print(f"Test : {len(test)}")

def copy_files(image_list, split):

    for image in image_list:

        shutil.copy(
            os.path.join(IMAGES_DIR, image),
            os.path.join(DATASET_DIR, "images", split, image)
        )

        label = image.replace(".jpg", ".txt")

        shutil.copy(
            os.path.join(LABELS_DIR, label),
            os.path.join(DATASET_DIR, "labels", split, label)
        )

copy_files(train, "train")
copy_files(val, "val")
copy_files(test, "test")

print("\nDataset successfully split!")