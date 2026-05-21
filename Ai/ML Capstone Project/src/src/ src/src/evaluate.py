"""
Model evaluation utilities.
"""

import joblib
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def load_model(model_path: str):
    """Load trained model from disk."""
    return joblib.load(model_path)


def evaluate_on_test_set(model, X_test, y_test) -> Dict[str, float]:
    """
    Evaluate model on test set with all standard metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True labels
        
    Returns:
        Dictionary of evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    
    return metrics


def get_prediction_confidence(model, X_input) -> float:
    """
    Get prediction confidence score.
    
    Args:
        model: Trained model
        X_input: Input features
        
    Returns:
        Confidence score (probability of predicted class)
    """
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_input)[0]
        return max(proba)
    return 0.5


def interpret_prediction(prediction: int, confidence: float) -> str:
    """
    Interpret model prediction in human-readable form.
    
    Args:
        prediction: Model prediction (0 or 1)
        confidence: Confidence score
        
    Returns:
        Human-readable interpretation
    """
    sentiment = "positive" if prediction == 1 else "negative"
    confidence_level = "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low"
    
    return f"The model predicts this review is {sentiment} with {confidence_level} confidence ({confidence:.1%})"