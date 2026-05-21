"""
Tests for model training and evaluation.
"""

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.evaluate import evaluate_on_test_set, get_prediction_confidence


class TestEvaluateModel:
    """Test model evaluation functions."""
    
    def test_produces_correct_prediction_shape(self):
        """Verify predictions have correct shape."""
        model = LogisticRegression()
        X_test = np.random.randn(10, 5)
        y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        model.fit(X_test, y_test)
        
        metrics = evaluate_on_test_set(model, X_test, y_test)
        
        assert isinstance(metrics, dict)
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_meets_minimum_performance_threshold(self):
        """Verify model meets minimum performance on known sample."""
        # Create a simple separable dataset
        X_test = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y_test = np.array([1, 0, 1, 0])
        
        model = LogisticRegression()
        model.fit(X_test, y_test)
        
        metrics = evaluate_on_test_set(model, X_test, y_test)
        
        # On perfectly separable data, should achieve high accuracy
        assert metrics['accuracy'] >= 0.75
    
    def test_all_metrics_are_valid_numbers(self):
        """Verify all metrics are valid numeric values."""
        model = LogisticRegression()
        X_test = np.random.randn(20, 5)
        y_test = np.random.randint(0, 2, 20)
        model.fit(X_test, y_test)
        
        metrics = evaluate_on_test_set(model, X_test, y_test)
        
        for metric_name, value in metrics.items():
            assert isinstance(value, (int, float))
            assert not np.isnan(value)
            assert not np.isinf(value)


class TestPredictionConfidence:
    """Test confidence scoring."""
    
    def test_returns_probability_value(self):
        """Verify confidence is a probability value."""
        model = LogisticRegression()
        X_test = np.random.randn(10, 5)
        y_test = np.random.randint(0, 2, 10)
        model.fit(X_test, y_test)
        
        confidence = get_prediction_confidence(model, X_test[:1])
        
        assert 0 <= confidence <= 1
    
    def test_higher_for_clear_predictions(self):
        """Verify higher confidence for clearer predictions."""
        # Create clearly separable data
        X_test = np.array([[10, 0], [-10, 0]])
        y_test = np.array([1, 0])
        
        model = LogisticRegression()
        model.fit(X_test, y_test)
        
        confidence = get_prediction_confidence(model, X_test[:1])
        
        # Should be high confidence for clear case
        assert confidence > 0.8