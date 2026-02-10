# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy all requirements files
COPY requirements.ml.txt ./

# Install Python dependencies in order (ML first, then extraction, then backend)
# This optimizes layer caching - ML packages rarely change
RUN pip install --no-cache-dir -r requirements.ml.txt

COPY requirements.extractions.txt ./
RUN pip install --no-cache-dir -r requirements.extractions.txt

COPY requirements.backend.txt ./
RUN pip install --no-cache-dir -r requirements.backend.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libgl1 \
    libfontconfig1 \
    libfreetype6 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# Copy application
COPY backend ./backend
COPY config ./config

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/api/health/', timeout=5)"

# Run application
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
