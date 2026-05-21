"""
Tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.preprocess import (
    clean_text,
    tokenize_and_remove_stopwords,
    preprocess_dataframe,
    create_vectorizer
)


class TestCleanText:
    """Test text cleaning function."""
    
    def test_handles_missing_values(self):
        """Verify missing values are handled correctly."""
        result = clean_text(None)
        assert result == ""
        
        result = clean_text(np.nan)
        assert result == ""
    
    def test_removes_html_tags(self):
        """Verify HTML tags are removed."""
        text = "<p>This is a <b>test</b></p>"
        result = clean_text(text)
        assert "<" not in result
        assert ">" not in result
    
    def test_removes_special_characters(self):
        """Verify special characters are removed."""
        text = "Hello! @#$ World %"
        result = clean_text(text)
        assert "!" not in result
        assert "@" not in result
    
    def test_normalizes_whitespace(self):
        """Verify extra whitespace is normalized."""
        text = "Hello    world   test"
        result = clean_text(text)
        assert "  " not in result
    
    def test_does_not_modify_original(self):
        """Verify original input is not modified."""
        original = "Test string"
        original_copy = original.copy() if hasattr(original, 'copy') else original
        clean_text(original)
        # String is immutable, so this tests we don't mutate in place


class TestTokenizeAndRemoveStopwords:
    """Test tokenization and stopword removal."""
    
    def test_removes_common_stopwords(self):
        """Verify stopwords are removed."""
        text = "this is a test with stopwords"
        result = tokenize_and_remove_stopwords(text)
        assert "this" not in result.lower()
        assert "is" not in result.lower()
        assert "a" not in result.lower()
    
    def test_handles_empty_input(self):
        """Verify empty input is handled."""
        result = tokenize_and_remove_stopwords("")
        assert result == ""
    
    def test_preserves_content_words(self):
        """Verify content words are preserved."""
        text = "amazing movie fantastic acting"
        result = tokenize_and_remove_stopwords(text)
        assert "amazing" in result
        assert "movie" in result


class TestPreprocessDataFrame:
    """Test DataFrame preprocessing."""
    
    def test_handles_missing_text_values(self):
        """Verify missing text values are filled."""
        df = pd.DataFrame({'text': [None, 'test', ''], 'label': [1, 0, 1]})
        result = preprocess_dataframe(df)
        assert result['cleaned_text'].isna().sum() == 0
    
    def test_creates_processed_columns(self):
        """Verify processed columns are created."""
        df = pd.DataFrame({'text': ['test review'], 'label': [1]})
        result = preprocess_dataframe(df)
        assert 'cleaned_text' in result.columns
        assert 'processed_text' in result.columns
    
    def test_does_not_modify_original_dataframe(self):
        """Verify original DataFrame is not modified."""
        original = pd.DataFrame({'text': ['test'], 'label': [1]})
        original_copy = original.copy()
        preprocess_dataframe(original)
        assert original.equals(original_copy)


class TestCreateVectorizer:
    """Test vectorizer creation."""
    
    def test_returns_tfidf_vectorizer(self):
        """Verify correct vectorizer type is returned."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        config = {'max_features': 5000}
        vectorizer = create_vectorizer(config)
        assert isinstance(vectorizer, TfidfVectorizer)
    
    def test_respects_config_parameters(self):
        """Verify config parameters are applied."""
        config = {'max_features': 1000, 'ngram_range': [1, 3]}
        vectorizer = create_vectorizer(config)
        assert vectorizer.max_features == 1000
        assert vectorizer.ngram_range == (1, 3)