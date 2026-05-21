"""
Data Validation Tests
Verifies properties of the loaded dataset (columns, ranges, target values).
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

current_path = Path(__file__).resolve()
repo_root = current_path
while repo_root != repo_root.parent:
    if (repo_root / 'src').exists():
        break
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))

from src.data_preprocessing import load_data, load_config

@pytest.fixture
def dataset():
    """Load the actual dataset for validation tests."""
    config = load_config(str(repo_root / 'config' / 'config.yaml'))
    path = config['data']['raw_path']
    if not os.path.exists(path):
        pytest.skip("Dataset file not found.")
    return load_data(path)

def test_expected_columns_present(dataset):
    """Test that expected columns are present."""
    config = load_config(str(repo_root / 'config' / 'config.yaml'))
    target_col = config['data']['target_column']
    
    # Check target column
    assert target_col in dataset.columns, f"Target column '{target_col}' missing."
    
    # Check for a few known feature columns from the Employee Attrition dataset
    expected_features = ['Age', 'Department', 'JobRole', 'MonthlyIncome']
    for col in expected_features:
        assert col in dataset.columns, f"Expected column '{col}' missing."

def test_target_variable_values(dataset):
    """Test that target variable contains only expected values."""
    config = load_config(str(repo_root / 'config' / 'config.yaml'))
    target_col = config['data']['target_column']
    
    unique_values = dataset[target_col].unique()
    
    # For Employee Attrition, we expect 'Yes' and 'No' (or 1/0 if preprocessed, but raw is Yes/No)
    # We allow both string and numeric representations
    valid_values = {'Yes', 'No', 'yes', 'no', 1, 0}
    
    for val in unique_values:
        assert val in valid_values, f"Unexpected target value: {val}"

def test_numeric_features_within_range(dataset):
    """Test that numeric features are within reasonable ranges."""
    # Age should be between 18 and 70
    if 'Age' in dataset.columns:
        assert dataset['Age'].min() >= 18, "Age below 18"
        assert dataset['Age'].max() <= 70, "Age above 70"
    
    # MonthlyIncome should be positive
    if 'MonthlyIncome' in dataset.columns:
        assert dataset['MonthlyIncome'].min() > 0, "Negative income"

def test_dataset_has_minimum_rows(dataset):
    """Test that dataset meets the minimum row requirement (1000)."""
    assert len(dataset) >= 1000, f"Dataset has only {len(dataset)} rows, need at least 1000."

def test_dataset_has_minimum_features(dataset):
    """Test that dataset has at least 8 features (excluding target)."""
    config = load_config(str(repo_root / 'config' / 'config.yaml'))
    target_col = config['data']['target_column']
    
    feature_count = len(dataset.columns) - 1
    assert feature_count >= 8, f"Dataset has only {feature_count} features, need at least 8."