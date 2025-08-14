#!/usr/bin/env sh
set -eu

echo "[start.sh] Boot sequence..."
PYTHON=${PYTHON:-python}
PORT=${PORT:-8000}
WORKERS=${WORKERS:-3}
TIMEOUT=${TIMEOUT:-120}
LOG_LEVEL=${LOG_LEVEL:-info}

# Reduce TensorFlow verbosity and force CPU-only on servers without GPUs
export TF_CPP_MIN_LOG_LEVEL=${TF_CPP_MIN_LOG_LEVEL:-3}
export TF_ENABLE_ONEDNN_OPTS=${TF_ENABLE_ONEDNN_OPTS:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
export XLA_FLAGS=${XLA_FLAGS:---xla_gpu_cuda_data_dir=}

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
echo "[start.sh] TF_CPP_MIN_LOG_LEVEL=$TF_CPP_MIN_LOG_LEVEL TF_ENABLE_ONEDNN_OPTS=$TF_ENABLE_ONEDNN_OPTS"

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
