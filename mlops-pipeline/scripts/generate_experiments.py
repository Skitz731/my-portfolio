"""
Script to run multiple experiments with varying hyperparameters.
"""
import yaml
import shutil
import os
import subprocess
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"

def run_experiment(params, run_name):
    """Update config, run training, restore config."""
    # Load original config
    with open(CONFIG_PATH, 'r') as f:
        original_config = yaml.safe_load(f) or {}
    
    # Update model params
    if 'model' not in original_config:
        original_config['model'] = {}
    original_config['model'].update(params)
    
    # Use a proper temporary file to avoid collisions and hardcoded paths
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        yaml.dump(original_config, tmp)
        temp_config_path = tmp.name
    
    try:
        print(f"Running experiment: {run_name} with params: {params}")
        # Run training from the project root to ensure internal paths resolve
        cmd = ["python", str(PROJECT_ROOT / "src" / "model_training.py"), "--config", temp_config_path, "--run-name", run_name]
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        
        if result.returncode != 0:
            print(f"Experiment {run_name} failed.")
        else:
            print(f"Experiment {run_name} completed successfully.")
    finally:
        # Ensure cleanup of the temporary configuration file
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

if __name__ == "__main__":
    experiments = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 15},
        {"n_estimators": 100, "max_depth": 5, "min_samples_split": 10},
        {"n_estimators": 50, "max_depth": 20}
    ]
    
    for i, params in enumerate(experiments, 1):
        run_name = f"Exp{i}-{params['n_estimators']}est-{params['max_depth']}depth"
        run_experiment(params, run_name)
    
    print("All experiments completed!")