import pandas as pd
import numpy as np
import yaml
import os
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, Any
from scipy import sparse

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not os.path.exists(config_path):
        # Try looking relative to project root if not found
        root_config = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_path)
        if os.path.exists(root_config):
            config_path = root_config
        else:
            return {}
            
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    # 1. Try path as provided (relative to Current Working Directory)
    if os.path.exists(path):
        return pd.read_csv(path)
    
    # 2. Try path relative to the project root
    # __file__ is /home/deck/my-portfolio/mlops-pipeline/src/data_preprocessing.py
    # project_root should be /home/deck/my-portfolio/mlops-pipeline/
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    root_path = os.path.abspath(os.path.join(project_root, path))
    
    if os.path.exists(root_path):
        return pd.read_csv(root_path)
    
    # 3. If still not found, raise a detailed error
    raise FileNotFoundError(
        f"\n[Data Error]: Could not find file: {path}\n"
        f"Checked relative to CWD: {os.path.abspath(path)}\n"
        f"Checked relative to Project Root: {root_path}"
    )

def handle_missing_values(df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
    """Handle missing values by filling numeric columns with median/mean and categorical with mode."""
    df = df.copy()
    
    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        if strategy == 'mean':
            fill_value = df[col].mean()
        else:
            fill_value = df[col].median()
        df[col] = df[col].fillna(fill_value)
        
    # Handle categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        mode_series = df[col].mode()
        if not mode_series.empty:
            df[col] = df[col].fillna(mode_series[0])
            
    return df

def encode_target(y: pd.Series) -> pd.Series:
    """Encode Yes/No target strings to 1/0 integers."""
    if is_numeric_dtype(y):
        return y
    
    mapping = {'yes': 1, 'no': 0}
    return y.astype(str).str.lower().map(mapping).astype(int)

def preprocess_data(df: pd.DataFrame, target_column: str, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.Series, Any]:
    """
    Basic preprocessing: separates features/target and creates a ColumnTransformer.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create transformers
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Fit and transform the features
    X_processed = preprocessor.fit_transform(X)
    X_processed = np.asarray(X_processed)
    
    # Convert back to DataFrame to maintain MLflow compatibility and ease of use
    # (Getting feature names from OneHotEncoder)
    try:
        cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        feature_names = numeric_features + list(cat_names)
        X_processed = pd.DataFrame(X_processed, columns=feature_names)
    except:
        # Fallback if feature names extraction fails
        X_processed = pd.DataFrame(X_processed)
        
    return X_processed, y, preprocessor

def split_data(X: pd.DataFrame, y: pd.Series, config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and testing sets."""
    test_size = config['data'].get('test_size', 0.2)
    random_state = config['data'].get('random_state', 42)
    
    return tuple(train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    ))