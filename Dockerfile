# 🚀 DOCKERFILE FOR RAILWAY AS ALTERNATIVE
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements-railway.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-railway.txt

# Copy application
COPY app_railway.py .

# Expose port (Railway sets PORT environment variable)
ENV PORT=5000
EXPOSE $PORT

# Run application with dynamic port
CMD gunicorn app_railway:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300
