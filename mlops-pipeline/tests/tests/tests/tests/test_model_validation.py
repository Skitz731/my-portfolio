"""
Model Validation Tests
Trains a small model on a sample and verifies output types, shapes, and performance.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent.parent

sys.path.insert(0, str(project_root))

from src.data_preprocessing import load_data, preprocess_data, split_data, load_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

@pytest.fixture
def prepared_data():
    """Prepare a small sample of data for model testing."""
    config_path = project_root / "config" / "config.yaml"
    config = load_config(str(config_path))
    df = load_data(config['data']['raw_path'])
    
    # Take a small sample for speed
    df_sample = df.sample(n=min(100, len(df)), random_state=42)
    
    X, y, preprocessor = preprocess_data(df_sample, config['data']['target_column'], config)
    X_train, X_test, y_train, y_test = split_data(X, y, config)
    
    return X_train, X_test, y_train, y_test

def test_model_predicts_correct_shape(prepared_data):
    """Test that model predictions have the correct shape."""
    X_train, X_test, y_train, y_test = prepared_data
    
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    assert predictions.shape[0] == X_test.shape[0], "Prediction count mismatch"
    assert len(predictions.shape) == 1, "Predictions should be 1D array"

def test_model_predicts_correct_type(prepared_data):
    """Test that model predictions are of the correct type (integers for classification)."""
    X_train, X_test, y_train, y_test = prepared_data
    
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    # Check if predictions are integers (or can be cast to int)
    assert np.issubdtype(predictions.dtype, np.integer) or np.issubdtype(predictions.dtype, np.floating), \
        f"Predictions should be numeric, got {predictions.dtype}"

def test_model_meets_minimum_performance(prepared_data):
    """Test that the model achieves a minimum performance threshold on the sample."""
    X_train, X_test, y_train, y_test = prepared_data
    
    # Train a slightly better model for the test
    model = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Minimum threshold for a random forest on this dataset should be > 0.6 on a small sample
    # Adjust this threshold based on your specific dataset difficulty
    min_threshold = 0.60 
    
    assert accuracy >= min_threshold, \
        f"Model accuracy ({accuracy:.4f}) is below minimum threshold ({min_threshold})"

def test_model_produces_probabilities(prepared_data):
    """Test that the model can produce probability predictions."""
    X_train, X_test, y_train, y_test = prepared_data
    
    model = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    # Check if predict_proba exists and works
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_test)
        assert probas.shape[0] == X_test.shape[0], "Probability shape mismatch"
        assert probas.shape[1] == len(y_train.unique()), "Probability columns mismatch"
        # Probabilities should sum to 1
        np.testing.assert_almost_equal(probas.sum(axis=1), 1.0, decimal=5)
    else:
        pytest.skip("Model does not support predict_proba")