FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install minimal system dependencies for JAX
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with memory optimization
RUN pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

# Copy application code
COPY . .

# Create non-root user (Railway compatible)
RUN useradd -m -u 1000 trader && chown -R trader:trader /app
USER trader

# Expose port
EXPOSE 5001

# Start application (Railway optimized)
CMD ["python", "app_turbo.py"]
