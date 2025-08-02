# Use Railway's recommended Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for pandas-ta
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose Railway port
EXPOSE $PORT

# Start with gunicorn
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120