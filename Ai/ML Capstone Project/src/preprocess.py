"""
Text preprocessing module for IMDB sentiment analysis.
Handles tokenization, stopword removal, and vectorization.
"""

import pandas as pd
import numpy as np
import re
import pickle
import os
from typing import Tuple, List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure NLTK data is available
try:
    STOPWORDS = set(stopwords.words('english'))
except:
    STOPWORDS = set()


def clean_text(text: str) -> str:
    """
    Clean and normalize text data.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_and_remove_stopwords(text: str) -> str:
    """
    Tokenize text and remove stopwords.
    
    Args:
        text: Cleaned text string
        
    Returns:
        Tokenized text with stopwords removed
    """
    if not text:
        return ""
    
    try:
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word not in STOPWORDS]
        return ' '.join(filtered_tokens)
    except:
        return text


def preprocess_dataframe(df: pd.DataFrame, text_column: str = 'text', 
                         label_column: str = 'label') -> pd.DataFrame:
    """
    Preprocess entire DataFrame with cleaning and tokenization.
    
    Args:
        df: Input DataFrame
        text_column: Name of text column
        label_column: Name of label column
        
    Returns:
        Preprocessed DataFrame
    """
    df_clean = df.copy()
    
    # Handle missing values
    df_clean[text_column] = df_clean[text_column].fillna('')
    df_clean[label_column] = df_clean[label_column].fillna(-1)
    
    # Apply cleaning and tokenization
    df_clean['cleaned_text'] = df_clean[text_column].apply(clean_text)
    df_clean['processed_text'] = df_clean['cleaned_text'].apply(tokenize_and_remove_stopwords)
    
    return df_clean


def create_vectorizer(config: Dict[str, Any]) -> TfidfVectorizer:
    """
    Create TF-IDF vectorizer with configuration.
    
    Args:
        config: Preprocessing configuration dictionary
        
    Returns:
        Configured TfidfVectorizer
    """
    return TfidfVectorizer(
        max_features=config.get('max_features', 10000),
        ngram_range=tuple(config.get('ngram_range', [1, 2])),
        min_df=config.get('min_df', 2),
        max_df=config.get('max_df', 0.95),
        sublinear_tf=True
    )


def prepare_data(df: pd.DataFrame, config: Dict[str, Any], 
                 text_column: str = 'text', label_column: str = 'label') -> Tuple:
    """
    Prepare data for training: preprocess, vectorize, and split.
    
    Args:
        df: Input DataFrame
        config: Configuration dictionary
        text_column: Name of text column
        label_column: Name of label column
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, vectorizer)
    """
    # Preprocess
    df_processed = preprocess_dataframe(df, text_column, label_column)
    
    # Create and fit vectorizer
    vectorizer = create_vectorizer(config.get('preprocessing', {}))
    X = vectorizer.fit_transform(df_processed['processed_text'])
    y = df_processed[label_column].values
    
    # Split data
    train_size = 1 - config.get('data', {}).get('train_test_split', 0.2)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=1-train_size, random_state=config.get('model', {}).get('random_state', 42),
        stratify=y
    )
    
    return X_train, X_test, y_train, y_test, vectorizer


def save_vectorizer(vectorizer: TfidfVectorizer, path: str) -> None:
    """Save vectorizer to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(vectorizer, f)


def load_vectorizer(path: str) -> TfidfVectorizer:
    """Load vectorizer from disk."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def preprocess_single_review(text: str, vectorizer: TfidfVectorizer) -> np.ndarray:
    """
    Preprocess a single review for inference.
    
    Args:
        text: Raw review text
        vectorizer: Fitted TfidfVectorizer
        
    Returns:
        Transformed feature matrix
    """
    cleaned = clean_text(text)
    processed = tokenize_and_remove_stopwordscleaned)
    return vectorizer.transform([processed])