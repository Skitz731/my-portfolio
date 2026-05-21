FROM python:3.12.18-slim-bookworm

WORKDIR /app

# Install system dependencies
RUN apt-get upgrade -y && apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY tests/ ./tests/

# Create directories for data and model artifacts
RUN mkdir -p /app/data /app/mlruns /app/models

# Create non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=/app/mlruns
ENV MODEL_PATH=/app/models/best_model.pkl
ENV VECTORIZER_PATH=/app/models/vectorizer.pkl

# Default command
CMD ["streamlit", "run", "src/app.py", "--server.port=8501", "--server.address=0.0.0.0"]