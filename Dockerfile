# 🚀 DOCKERFILE FOR RAILWAY AS ALTERNATIVE
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements-railway.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-railway.txt

# Copy application
COPY app_railway.py .

# Expose port
EXPOSE 5000

# Run application
CMD ["gunicorn", "app_railway:app", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "300"]
