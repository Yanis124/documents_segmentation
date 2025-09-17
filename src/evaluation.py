import os
import pandas as pd
from ultralytics import YOLO


from src.utils.set_config import load_project_config
from src.utils.experience import get_current_experiment_name

def evaluate_model(config: dict[str, dict[str, str]], exp_num: int):
    """Evaluate the YOLOv8 model.

    Args:
        config (dict[str, dict[str, str]]): Configuration dictionary containing project paths.
    """

    models_dir = config["project_dirs"]["models_dir"]
    experiments_dir = config["project_dirs"]["experiments_dir"]
    
    exp_name = f"exp{exp_num}"
    latest_exp_path = os.path.join(experiments_dir, exp_name)

    # Load the trained model
    model = YOLO(os.path.join(models_dir, f"model{exp_num}.pt"))

    # Run validation and force YOLO to save inside latest_exp_path
    results = model.val(
        project=latest_exp_path,  
        name="val",               
        exist_ok=True
    )

    # --- Per-class metrics ---
    per_class = results.summary()
    df = pd.DataFrame(per_class)
    df.to_csv(os.path.join(os.path.join(latest_exp_path, "val"), "val_metrics.csv"), index=False)
    
    overall = results.results_dict
    print(overall)
    

if __name__ == "__main__":
    
    config = load_project_config()
    
    exp_num = get_current_experiment_name(config)
    
    evaluate_model(config, exp_num)