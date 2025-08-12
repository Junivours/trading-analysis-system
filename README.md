# 🚀 Trading Analysis System

Professional AI-powered trading analysis system with:
- Real-time price tracking
- Advanced technical indicators
- JAX neural networks
- Professional UI

## Quick Start (Local)
```bash
pip install -r requirements.txt
python app.py  # Dev server (Flask built-in)
```

Production (Gunicorn, like Railway):
```bash
gunicorn app:app --bind 0.0.0.0:8000 --workers 3
```

## Environment Variables (Optional)
| Variable | Purpose |
|----------|---------|
| GIT_REV | Override commit hash shown in /api/version |
| API_BASE_URL | (Future) External data proxy |
| LOG_LEVEL | logging level (INFO/DEBUG) |

## Build / Version Identification
The app exposes `/api/version` returning commit/build info. Commit detection order:
1. Env vars: `GIT_REV`, `RAILWAY_GIT_COMMIT_SHA`, `SOURCE_VERSION`, `SOURCE_COMMIT`, `COMMIT_HASH`, `RAILWAY_BUILD`
2. `version.txt` (first line)
3. `git rev-parse --short HEAD`
4. Fallback `unknown`

To force a specific version in Railway without git metadata available, set GIT_REV env var or create a build command that writes:
```bash
echo $RAILWAY_GIT_COMMIT_SHA > version.txt
```

## Deployment on Railway
1. Connect GitHub repository
2. Ensure Build Command (if you need version.txt): `echo $RAILWAY_GIT_COMMIT_SHA > version.txt || true`
3. Start Command auto from `Procfile` or set: `gunicorn app:app --bind 0.0.0.0:$PORT`
4. After deploy verify: `curl https://YOUR-APP/api/version`

### Health Check (optional)
You can use `/api/version` as a health endpoint. It returns JSON fast.

## Backtest & Analysis Endpoints
| Endpoint | Description |
|----------|-------------|
| /api/analyze/<symbol> | Full multi-timeframe + AI analysis |
| /api/analyze/<symbol>?refresh=1 | Forces cache bypass |
| /api/backtest/<symbol>?interval=1h&limit=500 | RSI mean reversion backtest |
| /api/backtest/<symbol>?...&refresh=1 | Force fresh data |
| /api/position/dca | POST DCA ladder |
| /api/ai/status | AI init & model status |
| /api/logs/recent | Recent in-memory logs |
| /api/version | Version & commit hash |

## Local Self-Test
```bash
python test_local.py
```
Ensures AI, analyze, backtest, logs endpoints work before pushing.

## Notes
- JAX optional: if not installed, AI returns neutral.
- All responses have `X-App-Version` header for deployment freshness.

Live Demo: (set after deployment)
