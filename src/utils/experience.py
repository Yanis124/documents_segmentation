import os


def get_current_experiment_name(config: dict[str, dict[str, str]]):
    """Get the current experiment name based on existing experiment folders
    
    Args:
        config (dict): Configuration dictionary containing project paths.
        
    Returns:
        int: The number of the current experiment folder (e.g., "exp1", "exp2", etc.)"""
    
    experiments_dir = config["project_dirs"]["experiments_dir"]
    models_dir = config["project_dirs"]["models_dir"]

    # Ensure experiment folder exists
    os.makedirs(experiments_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    exp_folders = [
        f for f in os.listdir(experiments_dir)
        if os.path.isdir(os.path.join(experiments_dir, f)) and f.startswith("exp")
    ]
    if exp_folders:
        last_num = max([int(f.replace("exp", "")) for f in exp_folders if f.replace("exp", "").isdigit()])
        next_exp_num = last_num
    else:
        next_exp_num = 0
    
    return next_exp_num