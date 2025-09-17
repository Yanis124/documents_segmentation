import os
import shutil
from ultralytics import YOLO
import pandas as pd

from src.utils.set_config import load_project_config
from src.utils.experience import get_current_experiment_name


def get_model_layers(model: YOLO):
    """Get model layers and their parameters"""

    layers = []
    for name, param in model.named_parameters():
        layers.append((name, param))
    return layers

def train_model(config: dict[str, dict[str, str]], exp_num: int):
    """Train the YOLOv8 model"""
    
    experiments_dir = config["project_dirs"]["experiments_dir"]
    models_dir = config["project_dirs"]["models_dir"]
    
    exp_name = f"exp{exp_num}"

    model = YOLO("yolov8n.pt")

    # Extract config values
    yolo_yaml = config["yolo_config"]["yolo_yaml"]
    epochs = config["training"]["epochs"]
    batch = config["training"]["batch"]
    imgsz = config["training"]["imgsz"]
    patience = config["training"]["patience"]
    device = config["training"]["device"]

    for name, param in model.named_parameters():
        param.requires_grad = False
        if "cv3" in name or "dfl" in name: 
            param.requires_grad = True

    # Fine-tune on your dataset
    model.train(
        data=yolo_yaml,  # dataset config
        imgsz=imgsz,                # image size
        epochs=epochs,                # number of training epochs
        batch=batch,                  # batch size
        patience=5,               # early stopping patience (default 100 epochs)
        project=experiments_dir,   # output project name
        name=exp_name,    # output folder name
        device=0,              # GPU ID (0) or 'cpu'
        amp=True,               # automatic mixed precision
        exist_ok=True           # overwrite output directory if it exists
    )

    latest_exp_path = os.path.join(experiments_dir, exp_name)

    # Path to YOLO’s best model
    best_model_src = os.path.join(latest_exp_path, "weights", "best.pt")

    # Path in clean models folder at project root
    best_model_dst = os.path.join(models_dir, f"model{exp_num}.pt")

    # Copy best model only
    shutil.copy(best_model_src, best_model_dst)
 
if __name__ == "__main__":

    config = load_project_config()
    
    exp_num = get_current_experiment_name(config) + 1 # train a new model
    
    train_model(config, exp_num)




