import json
import os
import shutil
from sklearn.model_selection import train_test_split

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def train_test_split_data(dataset_root, dataset_json, dataset_dest):
    """
    Function to split the dataset into train and test JSON and move corresponding images
    """
    image_dir = os.path.join(dataset_root, "images", "train")
    with open(dataset_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = [entry for entry in data if entry['image'] in os.listdir(image_dir)]

    train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    # Save the split JSON files
    save_json(train_data, os.path.join(dataset_dest, 'train.json'))
    save_json(test_data, os.path.join(dataset_dest, 'test.json'))

    # Prepare output directories
    train_dir = os.path.join(dataset_dest, "images", "train_split")
    test_dir = os.path.join(dataset_dest, "images", "test_split")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Move image files
    for entry in train_data:
        src = os.path.join(image_dir, entry['image'])
        dst = os.path.join(train_dir, entry['image'])
        if os.path.exists(src):
            shutil.copy(src, dst)

    for entry in test_data:
        src = os.path.join(image_dir, entry['image'])
        dst = os.path.join(test_dir, entry['image'])
        if os.path.exists(src):
            shutil.copy(src, dst)


if __name__ == "__main__":
    dataset_root = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/data"
    dataset_json = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/llava/dataset/dataset.json"
    dataset_dest = "/Users/raghavg/Desktop/Datasets and Projects/Insurance/llava/dataset"
    train_test_split_data(dataset_root=dataset_root,
                     dataset_json=dataset_json,
                     dataset_dest=dataset_dest)