import pandas as pd
import numpy as np
import pytest
import sys
import os

# Calculate the path to the 'src' directory relative to this test file
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, src_path)

from preprocessing import fill_missing_with_median

def test_fill_missing_replaces_nulls():
    """Median fill should replace NaN values with the column median."""
    df = pd.DataFrame({
        "age": [20.0, 30.0, np.nan, 40.0, 50.0]
    })

    result = fill_missing_with_median(df, ["age"])

    assert result["age"].isna().sum() == 0, "There should be no missing values after filling"
    assert result["age"].iloc[2] == 35.0, "Missing value should be filled with median (35.0)"

def test_fill_missing_does_not_modify_original():
    """The original dataframe should not be changed."""
    df = pd.DataFrame({
        "age": [20.0, np.nan, 40.0]
    })
    original_null_count = df["age"].isna().sum()

    fill_missing_with_median(df, ["age"])

    assert df["age"].isna().sum() == original_null_count, \
        "Original dataframe should not be modified"

def test_fill_missing_handles_no_nulls():
    """If there are no missing values, the data should be unchanged."""
    df = pd.DataFrame({
        "age": [20.0, 30.0, 40.0]
    })

    result = fill_missing_with_median(df, ["age"])

    pd.testing.assert_frame_equal(result, df)

def test_fill_missing_multiple_columns():
    """Should handle filling multiple columns at once."""
    df = pd.DataFrame({
        "age": [20.0, np.nan, 40.0],
        "income": [50000.0, 60000.0, np.nan]
    })

    result = fill_missing_with_median(df, ["age", "income"])

    assert result["age"].isna().sum() == 0
    assert result["income"].isna().sum() == 0

def test_fill_missing_raises_on_bad_column():
    """Should raise ValueError if column does not exist."""
    df = pd.DataFrame({
        "age": [20.0, 30.0]
    })

    with pytest.raises(ValueError, match="not found"):
        fill_missing_with_median(df, ["nonexistent_column"])