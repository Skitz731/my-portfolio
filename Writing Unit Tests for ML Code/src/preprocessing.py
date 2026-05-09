import pandas as pd
import numpy as np

def fill_missing_with_median(df, columns):
    """Fill missing values in specified columns with the column median."""
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in dataframe")
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    return df

def normalize_column(df, column, method="min-max"):
    """Normalize a column using min-max or z-score normalization."""
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    col_data = df[column]

    if method == "min-max":
        min_val = col_data.min()
        max_val = col_data.max()
        if min_val == max_val:
            df[column] = 0.0
        else:
            df[column] = (col_data - min_val) / (max_val - min_val)
    elif method == "z-score":
        mean_val = col_data.mean()
        std_val = col_data.std()
        if std_val == 0:
            df[column] = 0.0
        else:
            df[column] = (col_data - mean_val) / std_val
    else:
        raise ValueError(f"Unknown method: {method}. Use 'min-max' or 'z-score'")

    return df

def encode_binary_column(df, column, positive_value):
    """Convert a categorical column with two values into 0/1."""
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    unique_vals = df[column].dropna().unique()
    if len(unique_vals) > 2:
        raise ValueError(
            f"Column '{column}' has {len(unique_vals)} unique values, expected 2"
        )

    df[column] = (df[column] == positive_value).astype(int)
    return df

def create_age_bins(df, column="age", bins=None, labels=None):
    """Bin a continuous age column into categories."""
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    if bins is None:
        bins = [0, 18, 25, 35, 50, 65, 120]
    if labels is None:
        labels = ["under_18", "18-24", "25-34", "35-49", "50-64", "65+"]

    df[f"{column}_bin"] = pd.cut(df[column], bins=bins, labels=labels, right=False)
    return df

def remove_outliers(df, column, method="iqr", threshold=1.5):
    """Remove rows where a column's value is an outlier."""
    df = df.copy()
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in dataframe")

    if method == "iqr":
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        df = df[(df[column] >= lower) & (df[column] <= upper)]
    elif method == "zscore":
        mean = df[column].mean()
        std = df[column].std()
        df = df[abs(df[column] - mean) <= threshold * std]
    else:
        raise ValueError(f"Unknown method: {method}")

    return df