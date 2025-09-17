from ultralytics import YOLO
import os
import argparse

from src.utils.set_config import load_project_config
from src.utils.experience import get_current_experiment_name


def inference(config: dict[str, dict[str, str]], exp_num: int, image_path: str, output_dir: str):
    """Run inference on an image using the trained YOLOv8 model
    
    Args:
        config (dict): Configuration dictionary containing project paths.
        exp_num (int): The number of the experiment folder to load the model from.
        image_path (str): Path to the input image for inference.
        output_dir (str): Directory where inference results should be saved.
    """

    models_dir = config["project_dirs"]["models_dir"]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    model = YOLO(os.path.join(models_dir, f"model{exp_num}.pt"))

    results = model.predict(
        source=image_path,
        save=True,                      # save annotated image(s)
        project=output_dir,             # where to save
        exist_ok=True                   # overwrite if exists
    )

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Inference Runner")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image for inference"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory where the inference output will be saved"
    )
    args = parser.parse_args()

    config = load_project_config()
    exp_num = get_current_experiment_name(config)

    inference(config, exp_num, args.image, args.output)
