import os
import random

from src.utils.set_config import load_project_config
from src.utils.copy_image_label import copy_labels_images

def split_data__train_val_test(config: dict[str, dict[str, str]]):
    """Split the dataset into training, validation, and test sets based on the provided configuration.

    Args:
        config (dict): Configuration dictionary containing dataset paths and split ratios.
    """
    labels_dir = config["dataset_dirs"]["labels_output_dir"]
    images_dir = config["dataset_dirs"]["images_output_dir"]
    
    labels_train_dir = config["dataset_dirs"]["labels_train_dir"]
    labels_val_dir = config["dataset_dirs"]["labels_val_dir"]
    labels_test_dir = config["dataset_dirs"]["labels_test_dir"]
    
    images_train_dir = config["dataset_dirs"]["images_train_dir"]
    images_val_dir = config["dataset_dirs"]["images_val_dir"]
    images_test_dir = config["dataset_dirs"]["images_test_dir"]

    number_images = config["data_split"]["number_images"]
    train_ratio = config["data_split"]["train_ratio"]
    val_ratio = config["data_split"]["val_ratio"]
    test_ratio = config["data_split"]["test_ratio"]

    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)
    os.makedirs(labels_test_dir, exist_ok=True)
    
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(images_test_dir, exist_ok=True)

    all_label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    # Randomly sample `number_images`
    if number_images > len(all_label_files):
        number_images = len(all_label_files)
    selected_files = random.sample(all_label_files, number_images)

    random.shuffle(selected_files)
    
    # Compute split sizes
    n_train = int(number_images * train_ratio) 
    n_val = int(number_images * val_ratio) 
    
    train_files = selected_files[:n_train] 
    val_files = selected_files[n_train:n_train+n_val] 
    test_files = selected_files[n_train+n_val:] 
                
    # Copy files into corresponding dirs
    for file_name in selected_files:
        copy_labels_images(file_name, labels_dir, labels_train_dir, images_dir, images_train_dir)
        
    for file_name in val_files:
        copy_labels_images(file_name, labels_dir, labels_val_dir, images_dir, images_val_dir)
        
    for file_name in test_files:
        copy_labels_images(file_name, labels_dir, labels_test_dir, images_dir, images_test_dir)

    print(f"✅ Split complete: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")


if __name__ == "__main__":
    config = load_project_config()
    split_data__train_val_test(config)