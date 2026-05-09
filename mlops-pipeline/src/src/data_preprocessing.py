import pandas as pd
import yaml
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    # Try direct path
    if os.path.exists(config_path):
        actual_path = config_path
    else:
        # Try relative to project root (3 levels up from src/src/...)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        actual_path = os.path.join(project_root, config_path)
    
    if not os.path.exists(actual_path):
        return {}
            
    with open(actual_path, 'r') as f:
        return yaml.safe_load(f)

def load_data(path: str) -> pd.DataFrame:
    """Load dataset from CSV with robust path resolution."""
    # 1. Try path as provided (relative to Current Working Directory)
    if os.path.exists(path):
        return pd.read_csv(path)
    
    # 2. Try path relative to the project root
    # __file__ is /home/deck/my-portfolio/mlops-pipeline/src/src/data_preprocessing.py
    # project_root should be /home/deck/my-portfolio/mlops-pipeline/
    current_file_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
    root_path = os.path.abspath(os.path.join(project_root, path))
    
    if os.path.exists(root_path):
        return pd.read_csv(root_path)
    
    # 3. If still not found, raise a detailed error
    raise FileNotFoundError(
        f"\n[Data Error]: Could not find file: {path}\n"
        f"Checked relative to CWD: {os.path.abspath(path)}\n"
        f"Checked relative to Project Root: {root_path}\n"
        f"Current working directory: {os.getcwd()}\n"
        f"Script location: {current_file_path}"
    )

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

def split_data(X: pd.DataFrame, y: pd.Series, config: Dict[str, Any]) -> Tuple:
    """Split data into training and testing sets."""
    test_size = config['data'].get('test_size', 0.2)
    random_state = config['data'].get('random_state', 42)
    
    return train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
