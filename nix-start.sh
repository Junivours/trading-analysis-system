#!/bin/bash
# 🚀 Nix-Specific Railway Startup Script
# Alternative to Procfile for complex Nix environments

echo "🔧 Nix Railway Startup Script"
echo "📍 Working directory: $(pwd)"

# Environment setup
export PYTHONPATH="/app:$PYTHONPATH"
export FLASK_APP="app.py"
export FLASK_ENV="production"

# Find Python executable
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    # Search in Nix store
    for python_path in /nix/store/*/bin/python3 /nix/store/*/bin/python; do
        if [ -x "$python_path" ]; then
            PYTHON_CMD="$python_path"
            break
        fi
    done
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "❌ No Python executable found!"
    exit 1
fi

echo "🐍 Using Python: $PYTHON_CMD"

# Start the application
echo "🚀 Starting Flask application..."
exec $PYTHON_CMD start.py
