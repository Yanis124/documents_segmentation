import yaml

def load_project_config(config_path="configs/project.yaml"): 
    """Load global project config (paths, dataset yaml)""" 
    with open(config_path, "r") as f: 
        return yaml.safe_load(f)