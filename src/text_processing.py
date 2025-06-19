"""Advanced text processing with caching and optimization."""
import pandas as pd
import re
import string
import emoji
from functools import lru_cache
from typing import List, Dict, Any
import streamlit as st
from transformers import pipeline, AutoTokenizer
import torch

# Optimized model loading with user feedback
@st.cache_resource
def _load_sentiment_pipeline(_model_name: str):
    """Load the sentiment analysis pipeline with caching and user feedback."""
    device = 0 if torch.cuda.is_available() else -1
    
    # Show loading message to users
    with st.spinner(f"🤖 Loading AI model ({_model_name.split('/')[-1]})... First time only, please wait 30-60 seconds."):
        pipeline_instance = pipeline(
            "sentiment-analysis",
            model=_model_name,
            device=device,
            top_k=None,  # Fixed deprecation warning
            use_fast=True  # Faster tokenizer
        )
    
    st.success(f"✅ Model loaded successfully! ({_model_name.split('/')[-1]})")
    return pipeline_instance

class SentimentAnalyzer:
    """Handles sentiment analysis with caching and batch processing for Streamlit."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tokenizer = None
    
    @property
    def pipeline(self):
        """Cached model loading for Streamlit."""
        return _load_sentiment_pipeline(self.model_name)
    
    @property
    def tokenizer(self):
        """Cached tokenizer loading."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        return self._tokenizer
    
    def predict(self, text: str) -> Dict[str, Any]:
        """Predict sentiment with confidence scores."""
        # Use the standalone clean_text function
        cleaned_text = clean_text(text)
        
        # Truncate if too long
        max_length = 512
        if len(self.tokenizer.encode(cleaned_text)) > max_length:
            cleaned_text = self.tokenizer.decode(
                self.tokenizer.encode(cleaned_text)[:max_length-2]
            )
        
        # Get prediction using the cached pipeline
        results = self.pipeline(cleaned_text)[0]
        
        # Process results for better display
        processed_results = {}
        for result in results:
            label = result['label'].upper()
            if label in ['NEGATIVE', 'NEG']:
                processed_results['Negative'] = result['score']
            elif label in ['POSITIVE', 'POS']:
                processed_results['Positive'] = result['score']
            else:
                processed_results[label.title()] = result['score']
        
        return {
            'predictions': processed_results,
            'confidence': max(processed_results.values()),
            'predicted_class': max(processed_results, key=processed_results.get),
            'cleaned_text': cleaned_text
        }

# ================================
# STANDALONE FUNCTIONS (for scripts)
# ================================

@lru_cache(maxsize=1000)
def clean_text(text: str) -> str:
    """Standalone text cleaning function (used by both class and scripts)."""
    # Handle NaN/None values
    if pd.isna(text) or text is None:
        return ""
    
    # Convert to string if it's not already
    text = str(text).strip()
    
    # Skip if empty after conversion
    if not text:
        return ""
    
    try:
        # Convert emojis to text
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Handle contractions
        contractions = {
            "don't": "do not", "won't": "will not", "can't": "cannot",
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am"
        }
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()
        
    except Exception as e:
        # If any error occurs during cleaning, return empty string
        return ""

def apply_vader_sentiment(text: str) -> str:
    """Apply VADER sentiment analysis for basic labeling."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        
        # Return 'positive' if compound score > 0, else 'negative'
        return 'positive' if scores['compound'] > 0 else 'negative'
    except ImportError:
        # Fallback: simple keyword-based sentiment
        positive_words = ['good', 'great', 'love', 'amazing', 'awesome', 'best', '❤️', '😍']
        negative_words = ['bad', 'hate', 'worst', 'terrible', 'awful', '👎', '😤']
        
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        return 'positive' if pos_count > neg_count else 'negative'