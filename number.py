import os
from collections import Counter
import yaml

# Load class names from YAML
with open('C:/Users/kharb/OneDrive/Desktop/Yolo/data.yaml', 'r') as f:
    data = yaml.safe_load(f)
class_names = data['names']

# Path to train/valid/test labels
label_dir = 'C:/Users/kharb/OneDrive/Desktop/Yolo/dataset/train/labels'

# Count classes
class_counts = Counter()
for label_file in os.listdir(label_dir):
    if label_file.endswith('.txt'):
        with open(os.path.join(label_dir, label_file), 'r') as f:
            for line in f:
                class_id = int(line.strip().split()[0])
                class_counts[class_id] += 1

# Print results
for class_id, count in class_counts.items():
    print(f"{class_names[class_id]}: {count} instances")
