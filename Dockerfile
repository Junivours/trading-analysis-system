# 🚀 DOCKERFILE FOR RAILWAY WITH DYNAMIC PORT
FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY requirements-railway.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements-railway.txt

# Copy application files
COPY . .

# Expose port (Railway will set PORT environment variable)
EXPOSE $PORT

# Run application with dynamic port
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 300 --max-requests 1000 --preload
