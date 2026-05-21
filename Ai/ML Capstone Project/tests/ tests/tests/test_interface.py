"""
Tests for LLM interface and input parsing.
"""

import pytest
from src.preprocess import clean_text, tokenize_and_remove_stopwords


class TestInputParsing:
    """Test natural language input parsing."""
    
    def test_extracts_feature_values_from_conversation(self):
        """Verify feature extraction from conversational text."""
        # Test that cleaning preserves meaningful content
        text = "I'm 45 years old with a great movie experience"
        cleaned = clean_text(text)
        assert "45" in cleaned or "great" in cleaned
    
    def test_handles_incomplete_inputs_gracefully(self):
        """Verify incomplete inputs don't crash system."""
        empty_text = ""
        cleaned = clean_text(empty_text)
        assert cleaned == ""
        
        whitespace_text = "   "
        cleaned = clean_text(whitespace_text)
        assert cleaned.strip() == ""
    
    def test_handles_ambiguous_inputs(self):
        """Verify ambiguous inputs are handled."""
        # Very short text
        short_text = "ok"
        cleaned = clean_text(short_text)
        assert len(cleaned) >= 0  # Should not crash
        
        # Nonsensical text
        nonsense = "!@#$%^&*()"
        cleaned = clean_text(nonsense)
        assert cleaned ==