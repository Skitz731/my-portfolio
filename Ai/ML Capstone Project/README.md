# IMDB Sentiment Analysis Capstone

## Overview
An end-to-end ML application that classifies movie reviews as positive or negative, featuring a natural language interface powered by an LLM.

## Tech Stack
- **ML**: Scikit-learn, TF-IDF, Logistic Regression
- **Tracking**: MLflow
- **Interface**: Streamlit + OpenAI/Nebius API
- **Deployment**: Docker

## Quick Start
1. Clone the repo
2. Build the image: `docker build -t imdb-sentiment .`
3. Run: `docker run -p 8501:8501 --env-file .env imdb-sentiment`

## Results
- Best Model Accuracy: 89.5%
- Precision: 0.88
- Recall: 0.90
- F1-Score: 0.89    