"""
Model training module with MLflow integration.
Supports multiple model types and hyperparameter tuning.
"""

import os
import yaml
import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from preprocess import prepare_data, save_vectorizer


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_imdb_data(data_path: str = None) -> pd.DataFrame:
    """
    Load IMDB dataset. Can be local file or downloaded.
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        DataFrame with 'text' and 'label' columns
    """
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        # Try to download from Kaggle or use sample
        print("Downloading IMDB dataset...")
        # In production, implement proper download logic
        # For now, expect data to be provided
        raise FileNotFoundError("Please provide IMDB dataset path")
    
    # Ensure required columns exist
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must have 'text' and 'label' columns")
    
    return df


def create_model(model_type: str, config: Dict[str, Any]):
    """
    Create model instance based on configuration.
    
    Args:
        model_type: Type of model to create
        config: Model configuration
        
    Returns:
        Initialized model
    """
    model_configs = config.get(model_type, {})
    
    if model_type == 'logistic_regression':
        return LogisticRegression(**model_configs)
    elif model_type == 'random_forest':
        return RandomForestClassifier(**model_configs)
    elif model_type == 'gradient_boosting':
        return GradientBoostingClassifier(**model_configs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_model(model, X_test, y_test, metrics: list) -> Dict[str, float]:
    """
    Evaluate model on test set.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        metrics: List of metric names to compute
        
    Returns:
        Dictionary of metric names to values
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    results = {}
    
    if 'accuracy' in metrics:
        results['accuracy'] = accuracy_score(y_test, y_pred)
    if 'precision' in metrics:
        results['precision'] = precision_score(y_test, y_pred)
    if 'recall' in metrics:
        results['recall'] = recall_score(y_test, y_pred)
    if 'f1' in metrics:
        results['f1'] = f1_score(y_test, y_pred)
    if 'roc_auc' in metrics and y_proba is not None:
        results['roc_auc'] = roc_auc_score(y_test, y_proba)
    
    return results


def train_model(config_path: str = 'configs/config.yaml', 
                data_path: str = None,
                num_runs: int = 5) -> str:
    """
    Train model with MLflow tracking.
    
    Args:
        config_path: Path to config YAML
        data_path: Path to IMDB dataset
        num_runs: Number of different configurations to try
        
    Returns:
        Path to best model
    """
    # Load configuration
    config = load_config(config_path)
    
    # Setup MLflow
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    # Load data
    df = load_imdb_data(data_path)
    
    # Define different configurations to try
    model_configs_to_try = []
    model_type = config['model']['type']
    
    if model_type == 'logistic_regression':
        model_configs_to_try = [
            {'C': 0.1, 'max_iter': 500},
            {'C': 1.0, 'max_iter': 1000},
            {'C': 10.0, 'max_iter': 1000},
            {'C': 0.5, 'max_iter': 800},
            {'C': 2.0, 'max_iter': 1200}
        ]
    elif model_type == 'random_forest':
        model_configs_to_try = [
            {'n_estimators': 50, 'max_depth': 5},
            {'n_estimators': 100, 'max_depth': 10},
            {'n_estimators': 150, 'max_depth': 15},
            {'n_estimators': 100, 'max_depth': 5},
            {'n_estimators': 200, 'max_depth': 10}
        ]
    elif model_type == 'gradient_boosting':
        model_configs_to_try = [
            {'n_estimators': 50, 'learning_rate': 0.1},
            {'n_estimators': 100, 'learning_rate': 0.1},
            {'n_estimators': 100, 'learning_rate': 0.05},
            {'n_estimators': 150, 'learning_rate': 0.1},
            {'n_estimators': 100, 'learning_rate': 0.2}
        ]
    
    # Prepare data once
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(df, config)
    
    # Resolve paths relative to this script
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Save vectorizer
    vectorizer_path = os.environ.get('VECTORIZER_PATH', os.path.join(BASE_DIR, "..", "..", "..", "models", "vectorizer.pkl"))
    save_vectorizer(vectorizer, vectorizer_path)
    
    best_run_id = None
    best_accuracy = 0
    
    # Track multiple runs
    for i, override_config in enumerate(model_configs_to_try[:num_runs]):
        with mlflow.start_run(run_name=f"run_{i+1}") as run:
            # Log parameters
            mlflow.log_param("model_type", model_type)
            mlflow.log_params(override_config)
            mlflow.log_param("data_version", "imdb_v1")
            
            # Create and train model
            model = create_model(model_type, {model_type: {**config.get(model_type, {}), **override_config}})
            model.fit(X_train, y_train)
            
            # Evaluate
            metrics = evaluate_model(model, X_test, y_test, config['evaluation']['metrics'])
            
            # Log metrics
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Log model
            model_path = f"models/{model_type}_{i}"
            mlflow.sklearn.log_model(model, model_path)
            
            # Track best model
            if metrics.get('accuracy', 0) > best_accuracy:
                best_accuracy = metrics['accuracy']
                best_run_id = run.info.run_id
            
            print(f"Run {i+1}: Accuracy = {metrics.get('accuracy', 0):.4f}")
    
    # Find and save best model
    best_run = mlflow.search_runs().iloc[0]  # Get best run
    best_model_uri = f"runs:/{best_run_id}/models/{model_type}_0"
    
    # Load and save best model locally
    best_model = mlflow.sklearn.load_model(best_model_uri)
    model_path = os.environ.get('MODEL_PATH', os.path.join(BASE_DIR, "..", "..", "..", "models", "best_model.pkl"))
    joblib.dump(best_model, model_path)
    
    print(f"\nBest model saved to {model_path}")
    print(f"Best accuracy: {best_accuracy:.4f}")
    
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IMDB sentiment model")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    parser.add_argument("--data", default=None, help="Dataset path")
    parser.add_argument("--runs", type=int, default=5, help="Number of experiment runs")
    
    args = parser.parse_args()
    train_model(args.config, args.data, args.runs)