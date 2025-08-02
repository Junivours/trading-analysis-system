#!/bin/bash
# Railway startup script for Trading Analysis Pro

echo "🚀 Starting Trading Analysis Pro Enhanced v6.1"
echo "📊 Initializing production environment..."

# Set production environment
export FLASK_ENV=production
export PYTHONUNBUFFERED=1

# Health check before starting
echo "🔧 Running pre-flight checks..."

# Start the application
echo "✅ Starting Gunicorn server..."
exec gunicorn --config gunicorn.conf.py app:app
