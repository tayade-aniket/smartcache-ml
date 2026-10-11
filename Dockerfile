FROM python:3.11-slim

# Avoid writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency specifications
COPY requirements.txt pyproject.toml ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Install package in editable mode
RUN pip install --no-cache-dir -e .

ENV PORT=8000
EXPOSE 8000 8501

# Default launch command: FastAPI (binds to Render's dynamic $PORT or 8000)
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
