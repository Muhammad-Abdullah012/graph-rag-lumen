# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (libpq for psycopg)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
# Create virtual environment outside bind-mounted /app to avoid being masked at runtime
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.backend.txt ./
RUN uv pip install --no-cache-dir -r requirements.backend.txt

# Copy application
COPY backend ./backend
COPY config ./config

# Create upload directory
RUN mkdir -p /app/uploads

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD /opt/venv/bin/python -c "import requests; requests.get('http://localhost:8000/api/health/', timeout=5)"

# Run application with venv python
CMD ["/opt/venv/bin/python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
