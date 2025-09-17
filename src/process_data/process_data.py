import shutil
import json
import os
from collections import defaultdict

from src.utils.set_config import load_project_config

def add_image_info_to_annotations(images_data: list[dict], annotations_data: list[dict]):
    """
    Process annotations and add corresponding image dimensions
    
    Args:
        images_data (list[dict]): List of image dictionaries with 'id', 'height', 'width'
        annotations_data (list[dict]): List of annotation dictionaries with 'image_id', 'category_id', 'bbox'

    Returns:
        dict: Dictionary mapping image_id to list of annotation info with image dimensions
    """
    images = {image['id']: (image['file_name'], image['height'], image['width']) for image in images_data}

    # Group multiple annotations per image
    annotations = defaultdict(list)
    for ann in annotations_data:
        annotations[ann['image_id']].append((ann['category_id'] - 1, ann['bbox']))

    annotations_with_image_info = {}
    missing_images = []
    
    for image_id, anns in annotations.items():
        if image_id in images:
            file_name, height, width = images[image_id]
            annotations_with_image_info[image_id] = []
            for category_id, bbox in anns:
                annotations_with_image_info[image_id].append({
                    'category_id': category_id,
                    'bbox': bbox,
                    'image_height': height,
                    'image_width': width,
                    'file_name': file_name
                })
        else:
            missing_images.append(image_id)
    
    if missing_images:
        print(f"Warning: {len(missing_images)} image IDs not found in images data: {missing_images[:5]}{'...' if len(missing_images) > 5 else ''}")
    
    print(f"Processed {sum(len(v) for v in annotations_with_image_info.values())} annotations with image dimensions")
    
    return annotations_with_image_info

def create_annotations_files(
    annotations_images: dict[str, list[dict]],
    output_path_annotation: str,
    input_path_images: str,
    output_path_images: str,
    ):
    
    os.makedirs(output_path_annotation, exist_ok=True)
    os.makedirs(output_path_images, exist_ok=True)

    # Group annotations by image_file_name
    annotations_by_file = defaultdict(list)
    for anns in annotations_images.values():
        for ann in anns:
            annotations_by_file[ann['file_name']].append(ann)

    total_images = 0
    total_annotations = 0
    skipped_images = 0

    for image_file, annotations in annotations_by_file.items():
        image_path = os.path.join(input_path_images, image_file)
        output_image_path = os.path.join(output_path_images, image_file)

        if not os.path.isfile(image_path):
            skipped_images += 1
            continue

        label_path = os.path.join(output_path_annotation, os.path.splitext(image_file)[0] + '.txt')

        with open(label_path, 'w', encoding='utf-8') as f:
            for ann in annotations:
                x_min, y_min, width, height = ann['bbox']
                x_center = (x_min + width / 2) / ann['image_width']
                y_center = (y_min + height / 2) / ann['image_height']
                w_norm = width / ann['image_width']
                h_norm = height / ann['image_height']
                f.write(f"{ann['category_id']} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                total_annotations += 1

        shutil.copy2(image_path, output_image_path)
        total_images += 1

    print(f"Images processed: {total_images}")
    print(f"Annotations written: {total_annotations}")
    print(f"Images skipped: {skipped_images}")
   
def create_class_file(categories: list[dict], output_dir: str):
    """Create a class file for YOLO training.

    Args:
        categories (list[dict]): List of category dictionaries.
        output_dir (str): Directory to save the class file.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    class_file_path = os.path.join(output_dir, "classes.txt")
    with open(class_file_path, 'w', encoding='utf-8') as f:
        for category in categories:
            f.write(f"{category['id'] - 1} {category['name']}\n")

def create_dataset(config: dict[str, dict[str, str]]):
    """process the data.json and create a n image and labels folder"""

    labels_data_path = config["dataset_dirs"]["labels_input_dir"]
    output_path_labels = config["dataset_dirs"]["labels_output_dir"]
    
    input_path_images = config["dataset_dirs"]["images_input_dir"]
    output_path_images = config["dataset_dirs"]["images_output_dir"]
    
    classes_output_dir = config["project_dirs"]["data_dir"]

    data_path = os.path.join(labels_data_path, "train.json")

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
    except FileNotFoundError as e:
        print(f"Error reading data file: {e}")
        return

    # train_annotations_with_image_info = add_image_info_to_annotations(
    #     train_data['images'], 
    #     train_data['annotations']
    # )
    
    # create_annotations_files(train_annotations_with_image_info, output_path_labels, input_path_images, output_path_images)
    create_class_file(train_data['categories'], classes_output_dir)

if __name__ == "__main__":
    config = load_project_config()
    create_dataset(config)