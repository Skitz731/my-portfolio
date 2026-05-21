"""
Model Training Script with MLflow Integration
Trains a Random Forest classifier and logs experiments to MLflow.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import yaml
import os
import sys
import argparse
from pathlib import Path

# Add the script directory and possible package roots to path for imports
script_path = Path(__file__).resolve()
script_dir = script_path.parent
project_root = script_dir.parent

for path in [project_root, script_dir]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from data_preprocessing import load_data, preprocess_data, split_data, load_config
except ImportError as exc:
    # Only wrap the error if it's specifically the data_preprocessing module missing
    if getattr(exc, 'name', None) == 'data_preprocessing':
        raise ImportError(
            f"Could not find 'data_preprocessing.py' in {script_dir} or {project_root}. "
            "Please ensure the file exists and is named correctly."
        ) from exc
    # If it's a dependency of data_preprocessing that's missing, let the original error through
    raise

def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate evaluation metrics.
    Handles both binary and multiclass classification gracefully.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # ROC-AUC only makes sense for binary or multiclass with probabilities
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr')
        except ValueError:
            # Fallback if labels are not suitable for ROC-AUC
            metrics['roc_auc'] = 0.0
            
    return metrics

def train_model(config_path: str = "configs/training_config.yaml", run_name: str = None):
    """
    Main training function with MLflow integration.
    
    Args:
        config_path: Path to the YAML configuration file
        run_name: Optional custom name for the MLflow run
        
    Returns:
        Tuple of (run_id, metrics_dict)
    """
    
    # 1. Load Configuration
    print(f"Loading configuration from {config_path}...")
    config = load_config(config_path)
    if not config:
        print(f"ERROR: Configuration file not found or empty at {config_path}")
        sys.exit(1)
    
    # 2. Set MLflow Experiment
    experiment_name = "Employee Attrition Prediction"
    mlflow.set_experiment(experiment_name)
    
    # 3. Load and Preprocess Data
    print("Loading and preprocessing data...")
    try:
        df = load_data(config['data']['raw_path'])
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
        
    X, y, preprocessor = preprocess_data(df, config['data']['target_column'], config)
    
    # 4. Split Data
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    
    print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"  Target distribution (Train): {y_train.value_counts().to_dict()}")
    
    # 5. Extract Model Parameters from Config
    model_params = config['model'].copy()
    
    # 6. Start MLflow Run
    run_name = run_name or f"RF-{model_params['n_estimators']}-{model_params['max_depth']}"
    
    with mlflow.start_run(run_name=run_name) as run:
        print(f"Starting MLflow run: {run.info.run_id} ({run_name})")
        
        # --- Log Parameters ---
        # Log model hyperparameters
        mlflow.log_params(model_params)
        
        # Log data info
        mlflow.log_param("data_source", config['data']['raw_path'])
        mlflow.log_param("data_version", "v1") # In a real pipeline, this would be the DVC hash
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("preprocessor_type", "ColumnTransformer")
        
        # --- Train Model ---
        print("Training Random Forest model...")
        model = RandomForestClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # --- Evaluate Model ---
        print("Evaluating model...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        
        metrics = calculate_metrics(y_test, y_pred, y_proba)
        
        # --- Log Metrics ---
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, float(metric_value))
            print(f"  {metric_name}: {metric_value:.4f}")
        
        # --- Log Model Artifact ---
        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(model, "model")
        
        # --- Check Performance Thresholds ---
        thresholds = config.get('thresholds', {})
        primary_metric = config['metrics']['primary']
        min_threshold = thresholds.get(f"min_{primary_metric}", 0.0)
        
        current_performance = metrics.get(primary_metric, 0.0)
        
        print(f"\nPerformance Check: {primary_metric}={current_performance:.4f} (Threshold: {min_threshold})")
        
        if current_performance < min_threshold:
            print(f"FAILURE: Model performance ({current_performance:.4f}) is below threshold ({min_threshold}).")
            print("Exiting with error code 1 to fail CI/CD pipeline.")
            sys.exit(1)
        else:
            print(f"SUCCESS: Model meets performance threshold.")
            
        print(f"\nTraining completed successfully. Run ID: {run.info.run_id}")
        print(f"Model URI: mlflow-artifacts:/{run.info.run_id}/artifacts/model")
        
        return run.info.run_id, metrics

def main():
    parser = argparse.ArgumentParser(description="Train MLOps model with MLflow")
    parser.add_argument("--config", type=str, default="configs/training_config.yaml", help="Path to config file")
    parser.add_argument("--run-name", type=str, default=None, help="Custom run name")
    
    args = parser.parse_args()
    
    try:
        run_id, metrics = train_model(args.config, args.run_name)
        print(f"Final Run ID: {run_id}")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()