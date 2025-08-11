# 🚀 DOCKERFILE FOR RAILWAY - BOMBASTIC APP.PY v2.1
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the bombastic application
COPY app.py .

# Copy supporting modules
COPY analysis/ ./analysis/
COPY core/ ./core/
COPY templates/ ./templates/
COPY utils/ ./utils/

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Railway will set PORT at runtime, expose default port
EXPOSE 5000

# Run the bombastic trading app (Railway provides PORT env var)
CMD ["python", "app.py"]
