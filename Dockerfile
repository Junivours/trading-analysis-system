# Lightweight production Dockerfile
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps including build dependencies for JAX
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -U pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Optional: copy version.txt early (keeps layer stable if unchanged)
COPY version.txt ./

COPY . .

RUN chmod +x /app/start.sh

# Expose default port (Railway uses $PORT anyway)
EXPOSE 8000

# Healthcheck hitting the /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["/app/start.sh"]
