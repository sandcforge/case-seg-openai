# Use Python 3.11 slim image for smaller size
# Explicitly specify linux/amd64 platform for Cloud Run compatibility
FROM --platform=linux/amd64 python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run the application as a module
CMD ["python", "-m", "src.app_bigquery"]
