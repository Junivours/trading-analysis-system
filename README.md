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

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/decision/<symbol>?tf=1h | GET | Minimal decision: LONG/SHORT/NEUTRAL with 2–4 short reasons (no trading) |
| /api/analyze/<symbol> | GET | Full multi-timeframe + AI + patterns + setups analysis |
| /api/analyze/<symbol>?refresh=1 | GET | Force cache bypass for live recomputation |
| /api/analyze/<symbol>?validate=1 | GET | Include enterprise validation block in response |
| /api/backtest/<symbol>?interval=1h&limit=500 | GET | RSI mean reversion backtest (interval: 1m/5m/15m/1h/4h/1d) |
| /api/backtest/<symbol>?...&refresh=1 | GET | Backtest with data refresh bypassing cache |
| /api/search/<query> | GET | Lightweight symbol search (Binance) |
| /api/quick-price/<symbol> | GET | Fast current price lookup (cached) |
| /api/liquidation/<symbol>/<entry_price>/<position_type> | GET | Approx. liquidation price calculator (isolated assumptions) |
| /api/position/dca | POST | Build DCA ladder (Body: symbol,total_capital,entries,spacing_pct,max_risk_pct) |
| /api/ai/status | GET | AI initialization & calibration status |
| /api/outcome/ai | POST | Provide a labeled outcome for a prior AI probability (Body: raw_prob, success) |
| /api/outcome/pattern | POST | Report pattern trade success for rolling calibration (Body: pattern_type, success) |
| /admin/save-state | POST | Force persistence save to disk (manual checkpoint) |
| /api/logs/recent?limit=100&level=INFO | GET | Recent in-memory logs (filterable) |
| /api/validate/<symbol>?refresh=1 | GET | Return a concise enterprise validation report only |
| /api/bot/run | POST | Run trading bot once (Body: symbol, interval, exchange, equity, risk_pct, paper) |
| /api/version | GET | Version & commit hash (also sets X-App-Version header) |
| /health | GET | Simple uptime/health probe |

### Outcome / Calibration Payloads
Pattern outcome example:
```json
POST /api/outcome/pattern
{ "pattern_type": "ascending_triangle", "success": true }
```

AI outcome example:
```json
POST /api/outcome/ai
{ "raw_prob": 0.67, "success": false }
```
`raw_prob` is the bullish probability originally returned (`bull_probability_raw`). `success` is `true` if a bullish continuation materialized within your evaluation window; otherwise `false`.

### Persistence
On startup the system loads persisted calibration & pattern statistics if present:
```
ai_calibration_state.json
pattern_stats_state.json
```
They are updated automatically after successful outcome submissions and can be checkpointed manually via `/admin/save-state`.

### Confidence Attribution
Analyses include `confidence_attribution` breaking down how pattern reliability, AI calibrated probability, technical consensus, and contradictions contribute to final confidence. This aids explainability and auditability.

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

### AI backend selection (optional)
- Default backend is JAX and requires `jax[cpu]` (already in `requirements.txt`).
- You can switch backend via environment variable before starting the app:
	- `AI_BACKEND=jax` (default)
	- `AI_BACKEND=torch` (requires `torch` installed) – lightweight adapter
	- `AI_BACKEND=tf` (requires `tensorflow` installed) – lightweight adapter
	- `AI_BACKEND=ensemble` – averages across available backends (JAX + any optional ones)

If Torch/TF are not installed, the adapters fall back to the JAX implementation and annotate `framework` in the AI response for transparency.

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

## Automated Trading Bot (Paper by default)

- Location: `core/trading/` with `bot.py`, `exchange_adapter.py`, `mexc_adapter.py`, and `storage.py`.
- **Supports both Binance and MEXC exchanges**
- By default runs in dry-run (paper) mode unless API keys are set and you pass `{"paper": false}`.
- Simple REST to execute once:

**Binance Trading:**
```bash
POST /api/bot/run
Body: {"symbol":"BTCUSDT","interval":"1h","exchange":"binance"}
```

**MEXC Trading:**
```bash
POST /api/bot/run  
Body: {"symbol":"BTCUSDT","interval":"1h","exchange":"mexc"}
```

Response includes selected setups and any executed paper orders. Position sizing uses risk % and ATR-informed stops from the analyzer.

### Exchange Configuration

| Exchange | API Keys Required | Environment Variables |
|----------|-------------------|----------------------|
| Binance | BINANCE_API_KEY, BINANCE_API_SECRET | For live trading |
| MEXC | MEXC_API_KEY, MEXC_API_SECRET | For live trading |

**Setup Example:**
```bash
# For MEXC
export MEXC_API_KEY="your_mexc_key"
export MEXC_API_SECRET="your_mexc_secret"

# For Binance  
export BINANCE_API_KEY="your_binance_key"
export BINANCE_API_SECRET="your_binance_secret"
```

See `MEXC_SETUP.md` for detailed MEXC configuration guide.


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
- `bull_probability_raw`: direct sum of BUY + STRONG_BUY class probabilities
- `bull_probability_calibrated`: Platt-scaled version using logistic parameters A,B fit on rolling (max 500) outcome samples

Calibration automatically updates every 30s (or on sufficient new samples) when >= 40 labeled samples exist. Prior to 20 samples, raw probability is used. The scoring layer prefers the calibrated probability (field: `probability_bullish_source = 'ai_calibrated'`). Reliability (probability margin + entropy) feeds into validation warnings, and pattern reliability contributes via dynamic weighting.

## Roadmap
- (In Progress) Broaden unit test coverage
- Optional CI workflow (GitHub Actions) once token workflow permissions available
- Expand pattern library & dynamic regime weighting
- WebSocket real-time streaming prices (Binance) with incremental indicator updates
- Persist feature standardization stats for restart consistency
- (Done) Extract MasterAnalyzer into `core/orchestration/master_analyzer.py`
- (Done) AI v2.1 reliability + pattern precision upgrade
- (Done) Probability calibration (Platt) integrated into final scoring

## Trading Bot (Local Only Safety)

The trading endpoint `/api/bot/run` is disabled by default in hosted environments (e.g., Railway). It responds with HTTP 403 and message "Trading disabled on this deployment (analysis-only mode).".

To enable locally, set in your `.env`:

```
ALLOW_TRADING=true
```

On Railway (any `RAILWAY_*` env present), trading remains disabled regardless of this flag to protect your account.

## Minimal UI on Railway (Analysis-only)
On Railway the root page `/` automatically shows a compact interface powered by `/api/decision/<symbol>` that returns:
- Decision: LONG / SHORT / NEUTRAL
- 2–4 concise reasons (e.g., Trend, RSI, MACD, MTF, AI, Pattern)

Preview minimal UI locally by setting:
```bash
ANALYSIS_MODE=minimal python app.py
```

## Local vs Railway behavior
- Local (developer machine): Full dashboard + all analysis endpoints. Trading can be enabled locally with `ALLOW_TRADING=true`.
- Railway (hosted): Minimal dashboard + minimal decision API only; trading endpoints return 403 by design.
