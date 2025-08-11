# 🚀 DOCKERFILE FOR RAILWAY - BOMBASTIC APP.PY v2.0
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
ENV PORT=5000
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose the port
EXPOSE $PORT

# Run the bombastic trading app
CMD ["python", "app.py"]
