"""
Streamlit application with LLM-powered natural language interface.
"""

import streamlit as st
import os
import joblib
import numpy as np
from typing import Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from preprocess import clean_text, tokenize_and_remove_stopwords, load_vectorizer
from evaluate import load_model, get_prediction_confidence, interpret_prediction

# Load environment variables
load_dotenv()

# Initialize OpenAI Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_llm_insights(review: str, sentiment: str) -> str:
    """Generate natural language insights using LLM."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that explains movie review sentiment analysis results."},
                {"role": "user", "content": f"The ML model classified this review as {sentiment}. Review: '{review}'. Explain in one or two sentences why this sentiment was likely chosen based on the tone."}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Could not generate LLM insights: {e}"

# Resolve paths relative to this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(BASE_DIR, "..", "models", "best_model.pkl")
DEFAULT_VEC = os.path.join(BASE_DIR, "..", "models", "vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="IMDB Sentiment Analyzer",
    layout="centered"
)

# Title and description
st.title("🎬 IMDB Movie Review Sentiment Analyzer")
st.markdown("""
Enter a movie review and our ML model will analyze its sentiment.
Powered by machine learning and enhanced with natural language understanding.
""")

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    model_path = os.environ.get('MODEL_PATH', DEFAULT_MODEL)
    vectorizer_path = os.environ.get('VECTORIZER_PATH', DEFAULT_VEC)
    
    st.info("This application uses a trained ML model to predict sentiment.")
    st.markdown("---")
    st.markdown("**About**")
    st.markdown("""
    - **Model**: Logistic Regression with TF-IDF features
    - **Dataset**: IMDB Movie Reviews
    - **Accuracy**: ~89% on test set
    """)

# Load model and vectorizer (cached for performance)
@st.cache_resource
def load_resources():
    """Load model and vectorizer."""
    try:
        model = load_model(model_path)
        vectorizer = load_vectorizer(vectorizer_path)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, vectorizer = load_resources()

if model is None or vectorizer is None:
    st.warning("Model not loaded. Please ensure the model files exist.")
    st.stop()

# Main input area
st.header("Analyze a Review")

# Text input for review
review_text = st.text_area(
    "Enter a movie review:",
    height=200,
    placeholder="Example: This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout..."
)

# Analyze button
if st.button("Analyze Sentiment", type="primary"):
    if not review_text.strip():
        st.warning("Please enter a review to analyze.")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Preprocess the review
                cleaned = clean_text(review_text)
                processed = tokenize_and_remove_stopwords(cleaned)
                
                # Transform to features
                X_input = vectorizer.transform([processed])
                
                # Get prediction
                prediction = model.predict(X_input)[0]
                confidence = get_prediction_confidence(model, X_input)
                
                # Display results
                st.success("Analysis Complete!")
                
                # Prediction result
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Sentiment", "Positive" if prediction == 1 else "Negative")
                with col2:
                    st.metric("Confidence", f"{confidence:.1%}")
                
                # LLM Insight Section
                sentiment_str = "Positive" if prediction == 1 else "Negative"
                st.subheader("🤖 AI Interpretation")
                st.info(generate_llm_insights(review_text, sentiment_str))

                # Detailed explanation
                with st.expander("View Detailed Analysis"):
                    st.write(interpret_prediction(prediction, confidence))
                    
                    # Feature importance (if available)
                    if hasattr(model, 'coef_'):
                        feature_names = vectorizer.get_feature_names_out()
                        coefficients = model.coef_[0]
                        
                        # Get top positive and negative features
                        top_positive_idx = np.argsort(coefficients)[-5:][::-1]
                        top_negative_idx = np.argsort(coefficients)[:5]
                        
                        st.subheader("Key Words Influencing Prediction")
                        st.write("**Positive indicators:**", ", ".join([feature_names[i] for i in top_positive_idx]))
                        st.write("**Negative indicators:**", ", ".join([feature_names[i] for i in top_negative_idx]))
                
                # Example queries
                st.markdown("---")
                st.markdown("**Try these example queries:**")
                examples = [
                    "This film was amazing! Best movie I've seen all year.",
                    "Terrible waste of time. Boring plot and bad acting.",
                    "It was okay, nothing special but not awful either."
                ]
                for example in examples:
                    st.code(example, language=None)
                    
            except Exception as e:
                st.error(f"Error during analysis: {e}")

# Footer
st.markdown("---")
st.markdown("""
<center>
Powered by Scikit-learn, MLflow, and Streamlit
</center>
""")