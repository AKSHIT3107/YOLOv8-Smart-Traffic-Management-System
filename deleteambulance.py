import os
import random
import shutil
from glob import glob

# Set paths
image_dir = 'dataset/train/images'
label_dir = 'dataset/train/labels'
target_class_index = 0  # ambulance class index (from data.yaml)
target_keep_count = 1500

# Find all ambulance-labeled files
ambulance_files = []
for label_file in glob(f"{label_dir}/*.txt"):
    with open(label_file, 'r') as f:
        lines = f.readlines()
        if any(line.startswith(str(target_class_index)) for line in lines):
            base = os.path.basename(label_file).replace('.txt', '')
            ambulance_files.append(base)

# Shuffle and keep only a subset
random.shuffle(ambulance_files)
files_to_remove = ambulance_files[target_keep_count:]

# Delete images and labels
for base in files_to_remove:
    image_path = os.path.join(image_dir, base + '.jpg')  # or .png
    label_path = os.path.join(label_dir, base + '.txt')
    if os.path.exists(image_path):
        os.remove(image_path)
    if os.path.exists(label_path):
        os.remove(label_path)

print(f"Reduced ambulance-labeled samples to {target_keep_count}.")
