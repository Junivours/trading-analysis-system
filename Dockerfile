# 🚀 DOCKERFILE FOR RAILWAY AS ALTERNATIVE
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements-railway.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-railway.txt

# Copy application
COPY app_railway.py .

# Expose port (Railway will override this with dynamic PORT)
EXPOSE 5000

# Run application with dynamic port from environment
CMD gunicorn app_railway:app --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 300 --max-requests 1000 --preload
