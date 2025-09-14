# Use official Python base image
FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system dependencies (needed for PyMuPDF, FAISS, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    pkg-config \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY ai-aggregator-fixed.py .

# Create necessary directories
RUN mkdir -p /app/cache /app/logs

# Expose Streamlit port
EXPOSE 8501

# Entrypoint
CMD ["streamlit", "run", "ai-aggregator-fixed.py", "--server.port=8501", "--server.address=0.0.0.0"]
