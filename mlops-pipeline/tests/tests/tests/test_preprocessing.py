"""
Unit Tests for Preprocessing Functions
Tests specific functions for correctness, edge cases, and error handling.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.data_preprocessing import (
    handle_missing_values,
    encode_target,
    load_data,
    load_config
)

# --- Fixtures ---

@pytest.fixture
def sample_dataframe_with_missing():
    """Create a sample dataframe with missing values for testing."""
    data = {
        'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0],
        'numeric_col2': [10, 20, 30, np.nan, 50],
        'categorical_col': ['A', 'B', 'A', np.nan, 'C'],
        'target': ['Yes', 'No', 'Yes', 'No', 'Yes']
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_dataframe_no_missing():
    """Create a clean dataframe."""
    data = {
        'numeric_col': [1.0, 2.0, 3.0, 4.0, 5.0],
        'categorical_col': ['A', 'B', 'A', 'B', 'C'],
        'target': ['Yes', 'No', 'Yes', 'No', 'Yes']
    }
    return pd.DataFrame(data)

# --- Tests: handle_missing_values ---

def test_handle_missing_values_numeric_median(sample_dataframe_with_missing):
    """Test that numeric missing values are filled with median."""
    df_clean = handle_missing_values(sample_dataframe_with_missing, strategy='median')
    
    # Check no missing values remain
    assert df_clean.isnull().sum().sum() == 0
    
    # Check specific values (Median of [1, 2, 4, 5] is 3.0)
    assert df_clean.loc[2, 'numeric_col'] == 3.0
    # Median of [10, 20, 30, 50] is 25.0
    assert df_clean.loc[3, 'numeric_col2'] == 25.0

def test_handle_missing_values_categorical_mode(sample_dataframe_with_missing):
    """Test that categorical missing values are filled with mode."""
    df_clean = handle_missing_values(sample_dataframe_with_missing, strategy='median')
    
    # Mode of ['A', 'B', 'A', 'C'] is 'A'
    assert df_clean.loc[3, 'categorical_col'] == 'A'

def test_handle_missing_values_does_not_modify_original(sample_dataframe_with_missing):
    """Test that the original dataframe is not modified."""
    original_df = sample_dataframe_with_missing.copy()
    df_clean = handle_missing_values(sample_dataframe_with_missing)
    
    # Check original still has NaN
    assert pd.isna(original_df.loc[2, 'numeric_col'])
    assert pd.isna(original_df.loc[3, 'categorical_col'])
    
    # Check clean does not have NaN
    assert not pd.isna(df_clean.loc[2, 'numeric_col'])

def test_handle_missing_values_invalid_strategy(sample_dataframe_with_missing):
    """Test that invalid strategy raises an error or defaults safely."""
    # Our implementation currently ignores invalid strategies for numeric (keeps median logic if not mean)
    # But let's test that it doesn't crash and fills values
    df_clean = handle_missing_values(sample_dataframe_with_missing, strategy='invalid')
    assert df_clean.isnull().sum().sum() == 0

# --- Tests: encode_target ---

def test_encode_target_yes_no_to_1_0():
    """Test conversion of 'Yes'/'No' to 1/0."""
    y = pd.Series(['Yes', 'No', 'Yes', 'No'])
    y_encoded = encode_target(y)
    
    expected = pd.Series([1, 0, 1, 0])
    pd.testing.assert_series_equal(y_encoded, expected)

def test_encode_target_already_numeric():
    """Test that numeric targets are returned unchanged."""
    y = pd.Series([1, 0, 1, 0])
    y_encoded = encode_target(y)
    
    expected = pd.Series([1, 0, 1, 0])
    pd.testing.assert_series_equal(y_encoded, expected)

def test_encode_target_mixed_case():
    """Test conversion of mixed case 'yes'/'no'."""
    y = pd.Series(['yes', 'NO', 'Yes', 'no'])
    y_encoded = encode_target(y)
    
    # Our mapping handles 'yes' and 'no' lower case too
    expected = pd.Series([1, 0, 1, 0])
    pd.testing.assert_series_equal(y_encoded, expected)

# --- Tests: load_data ---

def test_load_data_file_not_found():
    """Test that load_data raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")

def test_load_data_valid_file():
    """Test loading a valid file (using the actual dataset if available)."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    config_path_candidates = [
        os.path.join(root_dir, 'config.yaml'),
        os.path.join(root_dir, 'config', 'config.yaml'),
        os.path.join(root_dir, 'params.yaml'),
        os.path.join(root_dir, 'config.yml')
    ]

    config_path = None
    for candidate in config_path_candidates:
        if os.path.exists(candidate):
            config_path = candidate
            break

    if config_path is None:
        pytest.skip("Config file not found, skipping valid file test.")

    config = load_config(config_path=config_path)
    path = config['data']['raw_path']
    
    if os.path.exists(path):
        df = load_data(path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
    else:
        pytest.skip("Dataset file not found, skipping valid file test.")