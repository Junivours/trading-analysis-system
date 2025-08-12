#!/usr/bin/env sh
set -eu

echo "[start.sh] Boot sequence..."
PYTHON=${PYTHON:-python}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-3}
TIMEOUT=${TIMEOUT:-120}
LOG_LEVEL=${LOG_LEVEL:-info}

# Generate version.txt if missing and git available
if [ ! -f version.txt ]; then
  if command -v git >/dev/null 2>&1; then
    git rev-parse --short HEAD 2>/dev/null > version.txt || echo unknown > version.txt
  else
    echo unknown > version.txt
  fi
  echo "[start.sh] version.txt created: $(cat version.txt)"
else
  echo "[start.sh] version.txt present: $(cat version.txt)"
fi

# Show environment summary
echo "[start.sh] PORT=$PORT WORKERS=$WORKERS TIMEOUT=$TIMEOUT LOG_LEVEL=$LOG_LEVEL"

# Simple pre-flight: python -c import check
$PYTHON - <<'EOF'
try:
 import flask, requests, numpy
 print("[preflight] Core libs OK")
except Exception as e:
 print("[preflight] Missing dependency:", e)
EOF

# Launch gunicorn
exec gunicorn app:app \
  --bind 0.0.0.0:${PORT} \
  --workers ${WORKERS} \
  --timeout ${TIMEOUT} \
  --log-level ${LOG_LEVEL}
