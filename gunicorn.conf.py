# Optimized Gunicorn configuration for Railway rapid deployment
import os

# Server Socket
bind = f"0.0.0.0:{os.environ.get('PORT', '5000')}"
backlog = 2048

# Worker processes - optimized for faster startup
workers = 1  # Reduced from 2 for faster startup
worker_class = "sync"
worker_connections = 500  # Reduced for faster startup
timeout = 30  # Reduced timeout for faster health checks
keepalive = 2
max_requests = 500  # Reduced for faster restart
max_requests_jitter = 25
preload_app = True  # Important for health checks

# Security
limit_request_line = 4094
limit_request_fields = 100
limit_request_field_size = 8190

# Logging - minimal for startup speed
accesslog = "-"
errorlog = "-"
loglevel = "warning"  # Changed from info to warning for faster startup

# Process naming
proc_name = "trading-analysis-pro"

# Server mechanics
daemon = False
pidfile = None
user = None
group = None

# SSL (disabled for Railway)
keyfile = None
certfile = None

# Application
pythonpath = "/app"
