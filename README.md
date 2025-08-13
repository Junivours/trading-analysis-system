# 🚀 Trading Analysis System

Professional AI-powered trading analysis system with:
- Real-time price tracking
- Advanced technical indicators (RSI TV style, MACD curve strength, ATR, Fibonacci, Bollinger, Stoch, CCI, Pivots)
- Multi-timeframe consensus & pattern scanning
- Enterprise validation (contradictions: MACD/RSI/Patterns/AI/MTF, risk levels, action gating)
- Dynamic AI weighting & feature hashing (deterministic traceability)
- Phase timing metrics per analysis request
- JAX neural network (strict mode; if unavailable AI weight -> 0)
- Professional glassmorphism UI
- NEW: AI v2.1 feature engineering (trend, volatility, regime one-hot, pattern quality aggregates)
- NEW: Reliability score (probability margin + entropy) & adaptive temperature
- NEW: Refined pattern detection (distance-to-trigger, reliability, stricter breakout confirmation)
- NEW: Probability calibration layer (Platt scaling) with raw vs calibrated bullish probability

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
- Strict AI mode: if JAX environment not available the AI component is disabled and weight redistributed (transparently visible in final_score weights).
- Phase timing metrics returned in `phase_timings_ms` for performance monitoring.
- Feature hash (ai_feature_hash) allows caching/trace of identical feature vectors.
- All responses have `X-App-Version` header for deployment freshness.

Live Demo: (set after deployment)

## Architecture Overview
Core modules (all under `core/`):
| Module | Purpose |
|--------|---------|
| technical_analysis.py | Base indicators (RSI, MACD, MAs, support/resistance, volume, trend, momentum) |
| advanced_technical.py | Extended indicators (Bollinger, Stoch, Williams %R, CCI, ATR, Fibonacci, Ichimoku, Pivots) |
| patterns.py | Pattern detection & pattern-based trade generation (ChartPatternTrader) |
| ai.py | JAX neural network (probabilistic, MC dropout uncertainty) |
| position.py | PositionManager (risk/potential evaluation) |
| binance_client.py | Cached market data accessor |
| liquidation.py | Leverage liquidation level calculator |
| profiling.py | SymbolBehaviorProfiler (per-symbol volatility & bias) |

`app.py` now only provides the Flask routes & global initialization. The orchestration logic (multi-phase analysis, scoring, validation, adaptive risk, advanced trade setup generation) lives in `core/orchestration/master_analyzer.py` (migrated from the original monolith for maintainability and testability). AI upgraded to v2.1 (extended feature vector, online standardization, reliability scoring). Pattern engine adds distance-based reliability & stricter breakout rules.

## Trade Setups & Pattern-Based Ideas
Advanced setup generation is handled inside `MasterAnalyzer._generate_trade_setups` (now modular). It produces structured strategies:
- Pullback (Bullish / Bearish) with enterprise risk filters
- Breakout / Breakdown continuation
- Pattern Confirmation (ties ranked multi-timeframe patterns to entries)
- Momentum Continuation (MACD curve + RSI alignment)
- Mean Reversion (RSI extreme bands with adaptive thresholds)
- Support / Resistance Rejection scenarios
- Pattern Boost (injected when directional scarcity)
- Fallback generic setups (guarantee minimum coverage)

Each setup includes: normalized risk %, dynamic multi-R targets (1.5R .. 8R + swing extension), probability heuristic, confidence (contradiction & volatility aware), rationale and justification blocks.

`ChartPatternTrader.generate_pattern_trades` adds up to 5 supplemental pattern-centric trades (entry / stop / target / RR) which are merged & ranked with core strategies. Pattern objects include: `quality_grade`, `reliability_score`, `distance_to_trigger_pct` to help filter premature signals.

## Testing
Initial pytest smoke tests added (`tests/test_master_analyzer_basic.py`) verifying structure of analysis & backtest. Planned expansions:
- Indicator correctness (edge cases, insufficient data)
- Pattern detection sample fixtures & reliability scoring thresholds
- AI feature vector integrity & determinism (hash / length / schema)
- Position risk calculations
- Backtest engine sanity vs synthetic data

## Calibration & Reliability
The AI outputs both raw and calibrated bullish probabilities:
- bull_probability_raw: direct sum of BUY + STRONG_BUY class probabilities
- bull_probability_calibrated: Platt-scaled version using logistic parameters A,B fit on rolling (max 500) outcome samples

Calibration automatically updates every 30s when >=40 labeled samples exist. Until at least 20 samples, raw probability is used. The final scoring layer prefers the AI calibrated probability over the legacy score-derived heuristic when available (field: probability_bullish_source = 'ai_calibrated'). Reliability (probability margin + entropy) feeds into validation warnings.

## Roadmap
- (In Progress) Broaden unit test coverage
- Optional CI workflow (GitHub Actions) once token workflow permissions available
- Expand pattern library & dynamic regime weighting
- WebSocket real-time streaming prices (Binance) with incremental indicator updates
- Persist feature standardization stats for restart consistency
- (Done) Extract MasterAnalyzer into `core/orchestration/master_analyzer.py`
- (Done) AI v2.1 reliability + pattern precision upgrade
- (Done) Probability calibration (Platt) integrated into final scoring
