# ========================================================================================
# ðŸš€ ULTIMATE TRADING SYSTEM V5 - BEAUTIFUL & INTELLIGENT EDITION  
# ========================================================================================
# Professional Trading Dashboard mit intelligenter Position Management
# Basierend auf deinem schÃ¶nen Backup + erweiterte Features

from flask import Flask, jsonify, render_template_string, request
import os
import time
import requests
import numpy as np
from datetime import datetime

# Optional JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax.nn import logsumexp
    from jax import random
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False

from core.technical_analysis import TechnicalAnalysis
from core.advanced_technical import AdvancedTechnicalAnalysis
from core.patterns import AdvancedPatternDetector, ChartPatternTrader
from core.position import PositionManager
from core.ai import AdvancedJAXAI
from core.binance_client import BinanceClient
from core.liquidation import LiquidationCalculator
from core.profiling import SymbolBehaviorProfiler
from core.orchestration.master_analyzer import MasterAnalyzer
from collections import deque
import json, hashlib, logging, uuid

# Orchestration (scoring/validation/trade setups) fully migrated to core.orchestration.master_analyzer

# Initialize Flask app (was previously removed during refactor)
app = Flask(__name__)

# Initialize the master analyzer (single global instance used by routes)
master_analyzer = MasterAnalyzer()
CALIBRATION_STATE_PATH = os.getenv('CALIBRATION_STATE_PATH','data/calibration_state.json')
PATTERN_STATE_PATH = os.getenv('PATTERN_STATE_PATH','data/pattern_stats.json')

def _load_persistent_state():
    # AI calibration
    try:
        master_analyzer.ai_system.load_calibration_state(CALIBRATION_STATE_PATH)
    except Exception:
        pass
    # Pattern stats
    try:
        import json
        if os.path.exists(PATTERN_STATE_PATH):
            with open(PATTERN_STATE_PATH,'r',encoding='utf-8') as f:
                data=json.load(f)
            if isinstance(data,dict):
                from core.patterns import AdvancedPatternDetector
                AdvancedPatternDetector._pattern_stats.update(data)
    except Exception:
        pass

def _save_persistent_state():
    # AI calibration
    try:
        master_analyzer.ai_system.save_calibration_state(CALIBRATION_STATE_PATH)
    except Exception:
        pass
    # Pattern stats
    try:
        from core.patterns import AdvancedPatternDetector
        import json
        os.makedirs(os.path.dirname(PATTERN_STATE_PATH), exist_ok=True)
        with open(PATTERN_STATE_PATH,'w',encoding='utf-8') as f:
            json.dump(AdvancedPatternDetector._pattern_stats, f)
    except Exception:
        pass

_load_persistent_state()

# ========================================================================================
# ðŸ§¾ STRUCTURED LOGGING (in-memory ring buffer + stdout)
# ========================================================================================
logger = logging.getLogger("trading_app")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_h)

RECENT_LOGS = deque(maxlen=300)

def log_event(level: str, message: str, **context):
    """Log event to stdout and store a compact version in RECENT_LOGS.
    Returns a short log_id for correlation."""
    log_id = str(uuid.uuid4())[:8]
    context['log_id'] = log_id
    try:
        RECENT_LOGS.append({
            'ts': time.time(),
            'level': level.upper(),
            'message': message,
            'context': context
        })
        log_line = f"[{log_id}] {message} | {context}"
        if level.lower() == 'error':
            logger.error(log_line)
        elif level.lower() == 'warning':
            logger.warning(log_line)
        else:
            logger.info(log_line)
    except Exception as e:
        print(f"Logging failure: {e}")
    return log_id

# ========================================================================================
# ðŸŒ API ROUTES
# ========================================================================================

# ========================================================================================
# ðŸŒ API ROUTES
# ========================================================================================

@app.route('/')
def dashboard():
    """Render the beautiful main dashboard"""
    return render_template_string(DASHBOARD_HTML)

@app.route('/api/search/<query>')
def search_symbols(query):
    """Search for trading symbols"""
    try:
        symbols = master_analyzer.binance_client.search_symbols(query)
        return jsonify({
            'success': True,
            'symbols': symbols,
            'count': len(symbols)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/analyze/<symbol>')
def analyze_symbol(symbol):
    """Complete analysis of a trading symbol"""
    try:
        # Optional cache bypass: /api/analyze/BTCUSDT?refresh=1
        if request.args.get('refresh') == '1':
            master_analyzer.binance_client.clear_symbol_cache(symbol.upper())
            log_event('info', 'Cache cleared for analyze', symbol=symbol.upper())
        log_id = log_event('info', 'Analyze request start', symbol=symbol.upper(), refresh=request.args.get('refresh')=='1')
        analysis = master_analyzer.analyze_symbol(symbol.upper())
        
        if 'error' in analysis:
            err_id = log_event('error', 'Analyze error', symbol=symbol.upper(), error=analysis['error'], parent=log_id)
            return jsonify({
                'success': False,
                'error': analysis['error'],
                'log_id': err_id,
                'symbol': symbol.upper()
            }), 400
        log_event('info', 'Analyze success', symbol=symbol.upper(), parent=log_id)
        
        return jsonify({
            'success': True,
            'data': analysis,
            'log_id': log_id
        })
    except Exception as e:
        err_id = log_event('error', 'Analyze exception', symbol=symbol.upper(), error=str(e))
        return jsonify({
            'success': False,
            'error': str(e),
            'log_id': err_id,
            'symbol': symbol.upper()
        }), 500

@app.route('/api/liquidation/<symbol>/<float:entry_price>/<position_type>')
def calculate_liquidation(symbol, entry_price, position_type):
    """Calculate liquidation levels"""
    try:
        long_liq = master_analyzer.liquidation_calc.calculate_liquidation_levels(entry_price, 'long')
        short_liq = master_analyzer.liquidation_calc.calculate_liquidation_levels(entry_price, 'short')
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'entry_price': entry_price,
            'liquidation_long': long_liq,
            'liquidation_short': short_liq
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/quick-price/<symbol>')
def quick_price(symbol):
    """Get quick price for a symbol"""
    try:
        price = master_analyzer.binance_client.get_current_price(symbol.upper())
        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'price': price
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/favicon.ico')
def favicon():
    # Serve a tiny inline favicon (16x16) to prevent browser 404/502 spam.
    # If behind a proxy sometimes an empty 204 can show as 502 when worker restarts.
    import base64
    ico_b64 = (
        b'AAABAAEAEBAAAAEAIABoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAQAQAAAAAAAAAAAAAAAAAAAAA\n'
        b'AAAAAACZmZkAZmZmAGZmZgBmZmYAZmZmAGZmZgBmZmYAZmZmAGZmZgBmZmYAZmZmAGZmZgBmZmYAZmZm\n'
        b'AGZmZgBmZmYA///////////8AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n'
        b'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA////////////////////////\n'
        b'////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n'
        b'AAAAAAAAAAAAAAAAAAAAAAAAAAAA////////////////////AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n'
        b'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n'
    )
    data = base64.b64decode(ico_b64)
    return app.response_class(data, mimetype='image/x-icon')

@app.route('/api/ai/status')
def ai_status():
    try:
        status_data = {}
        try:
            if hasattr(master_analyzer, 'ai_system') and master_analyzer.ai_system:
                status_data = master_analyzer.ai_system.get_status()
            else:
                status_data = {'initialized': False, 'model_version': 'unavailable'}
        except Exception as inner:
            status_data = {'initialized': False, 'model_version': 'error', 'error': str(inner)}
        return jsonify({'success': True, 'data': status_data})
    except Exception as e:
        # Absolute fallback â€“ never raise 500 for status endpoint
        return jsonify({'success': True, 'data': {'initialized': False, 'model_version': 'unavailable', 'error': str(e)}}), 200

@app.route('/api/backtest/<symbol>')
def backtest(symbol):
    """Run a lightweight backtest on-demand (RSI mean reversion)."""
    interval = request.args.get('interval', '1h')
    limit = int(request.args.get('limit', '500'))
    try:
        if request.args.get('refresh') == '1':
            master_analyzer.binance_client.clear_symbol_cache(symbol.upper())
            log_event('info', 'Cache cleared for backtest', symbol=symbol.upper())
        log_id = log_event('info', 'Backtest start', symbol=symbol.upper(), interval=interval, limit=limit, refresh=request.args.get('refresh')=='1')
        data = master_analyzer.run_backtest(symbol, interval=interval, limit=limit)
        if 'error' in data:
            # Attach meta for client-side diagnosis
            err_id = log_event('warning', 'Backtest insufficient / error', symbol=symbol.upper(), interval=interval, limit=limit, error=data.get('error'), have=data.get('have'), need=data.get('need'))
            return jsonify({'success': False, 'error': data['error'], 'meta': {
                'symbol': symbol.upper(), 'interval': interval, 'limit': limit
            }, 'log_id': err_id}), 400
        log_event('info', 'Backtest success', symbol=symbol.upper(), interval=interval, limit=limit, parent=log_id, trades=data.get('metrics',{}).get('total_trades'))
        return jsonify({'success': True, 'data': data, 'meta': {'symbol': symbol.upper(), 'interval': interval, 'limit': limit}, 'log_id': log_id})
    except Exception as e:
        err_id = log_event('error', 'Backtest exception', symbol=symbol.upper(), interval=interval, limit=limit, error=str(e))
        return jsonify({'success': False, 'error': str(e), 'log_id': err_id}), 500

@app.route('/api/version')
def api_version():
    commit = 'unknown'
    try:
        for env_var in ['GIT_REV','RAILWAY_GIT_COMMIT_SHA','SOURCE_VERSION','SOURCE_COMMIT','COMMIT_HASH','RAILWAY_BUILD']:
            val = os.getenv(env_var)
            if val:
                commit = val[:12]
                break
        if commit == 'unknown' and os.path.exists('version.txt'):
            with open('version.txt','r',encoding='utf-8') as f:
                line = f.readline().strip()
                if line:
                    commit = line[:12]
    except Exception:
        pass
    return jsonify({'success':True,'version':commit,'ts': int(time.time()*1000)})

@app.route('/health')
def health():
    return jsonify({'ok':True,'ts': int(time.time()*1000)})

@app.route('/api/outcome/pattern', methods=['POST'])
def pattern_outcome():
    try:
        from core.patterns import AdvancedPatternDetector
        payload = request.get_json(force=True) or {}
        ptype = payload.get('pattern_type')
        success = bool(payload.get('success'))
        if not ptype:
            return jsonify({'success':False,'error':'pattern_type required'}),400
        AdvancedPatternDetector.record_pattern_outcome(ptype, success)
        _save_persistent_state()
        stats = AdvancedPatternDetector._pattern_stats.get(ptype, {})
        return jsonify({'success':True,'stats':stats,'pattern_type':ptype})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}),500

@app.route('/api/outcome/ai', methods=['POST'])
def ai_outcome():
    try:
        payload = request.get_json(force=True) or {}
        raw_prob = payload.get('raw_prob')
        success = payload.get('success')
        if raw_prob is None or success is None:
            return jsonify({'success':False,'error':'raw_prob & success required'}),400
        ok = master_analyzer.ai_system.add_calibration_observation(float(raw_prob), bool(success))
        if ok:
            _save_persistent_state()
        return jsonify({'success':ok,'calibration': master_analyzer.ai_system.get_calibration_status()})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}),500

@app.route('/admin/save-state', methods=['POST'])
def admin_save_state():
    _save_persistent_state()
    return jsonify({'success':True,'saved':True})

@app.route('/api/logs/recent')
def recent_logs():
    """Return recent in-memory logs (sanitized). Optional filter by level & limit.
    /api/logs/recent?limit=50&level=ERROR"""
    try:
        level_filter = request.args.get('level')
        limit = int(request.args.get('limit', '100'))
        limit = max(1, min(limit, 200))
        out = []
        for item in list(RECENT_LOGS)[-limit:][::-1]:  # newest first
            if level_filter and item['level'] != level_filter.upper():
                continue
            out.append(item)
        return jsonify({'success': True, 'logs': out, 'count': len(out)})
    except Exception as e:
        err_id = log_event('error', 'Logs endpoint failure', error=str(e))
        return jsonify({'success': False, 'error': str(e), 'log_id': err_id}), 500

@app.route('/api/position/dca', methods=['POST'])
def dca_position():
    """Calculate a DCA ladder given total capital, number of entries and risk params.
    Body: {symbol, total_capital, entries, base_price(optional), spacing_pct(optional), max_risk_pct(optional)}"""
    try:
        payload = request.get_json(force=True) or {}
        symbol = (payload.get('symbol') or 'BTCUSDT').upper()
        entries = max(1, min(int(payload.get('entries', 5)), 12))
        total_capital = float(payload.get('total_capital', 1000))
        spacing_pct = float(payload.get('spacing_pct', 1.25))  # percent between ladder steps
        max_risk_pct = float(payload.get('max_risk_pct', 2.0))  # overall account risk
        current_price = master_analyzer.binance_client.get_current_price(symbol) or payload.get('base_price') or 0
        if not current_price:
            return jsonify({'success': False, 'error': 'Preis nicht verfÃ¼gbar'}), 400
        # Build descending ladder for LONG (basic); could extend with direction later
        per_step_capital = total_capital / ((entries*(entries+1))/2)  # weighted heavier lower fills
        ladder = []
        cumulative_size = 0
        avg_price_num = 0
        for i in range(entries):
            # deeper steps lower percentage
            level_price = current_price * (1 - (spacing_pct/100.0)*i)
            size = per_step_capital * (i+1)
            cumulative_size += size
            avg_price_num += size * level_price
            ladder.append({
                'step': i+1,
                'price': round(level_price,2),
                'allocation': round(size,2),
                'weight_pct': round(size/total_capital*100,2)
            })
        avg_entry = avg_price_num / cumulative_size if cumulative_size else current_price
        # Risk calc: stop below last ladder price by one spacing
        stop_price = ladder[-1]['price'] * (1 - spacing_pct/100.0)
        risk_per_unit = avg_entry - stop_price
        position_value = cumulative_size  # assume 1:1 notional (spot)
        risk_pct = (risk_per_unit / avg_entry) * 100 if avg_entry else 0
        ok = risk_pct <= max_risk_pct
        return jsonify({
            'success': True,
            'symbol': symbol,
            'current_price': current_price,
            'avg_entry': round(avg_entry,2),
            'stop_price': round(stop_price,2),
            'ladder': ladder,
            'risk_pct_move_to_stop': round(risk_pct,2),
            'within_risk_limit': ok
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ========================================================================================
# ðŸŽ¨ BEAUTIFUL GLASSMORPHISM FRONTEND
# ========================================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ðŸš€ Ultimate Trading System V5</title>
    <style>
        :root {
            --bg: #0b0f17;
            --bg-alt: #111a24;
            --bg-soft: #1a2532;
            --card-bg: linear-gradient(160deg, rgba(26,37,50,0.85) 0%, rgba(15,21,29,0.92) 100%);
            --card-border: rgba(255,255,255,0.08);
            --card-border-strong: rgba(255,255,255,0.18);
            --text-primary: #ffffff;
            --text-secondary: rgba(255,255,255,0.75);
            --text-dim: rgba(255,255,255,0.55);
            --accent: #0d6efd;
            --accent-glow: 0 0 0 3px rgba(13,110,253,0.15);
            --success: #26c281;
            --danger: #ff4d4f;
            --warning: #f5b041;
            --info: #36c2ff;
            --purple: #8b5cf6;
            --radius-sm: 8px;
            --radius-md: 14px;
            --radius-lg: 24px;
            --shadow-soft: 0 4px 18px -4px rgba(0,0,0,0.55);
            --shadow-hover: 0 8px 28px -6px rgba(0,0,0,0.6);
            --gradient-accent: linear-gradient(90deg,#0d6efd,#8b5cf6 60%,#36c2ff);
        }

        body.dim {
            --bg: #06090f;
            --bg-alt: #0d141c;
            --bg-soft: #16202b;
            --card-bg: linear-gradient(150deg, rgba(20,29,40,0.88) 0%, rgba(10,15,21,0.95) 100%);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: radial-gradient(circle at 70% 15%, #13202e 0%, var(--bg) 55%);
            min-height: 100vh;
            overflow-x: hidden;
            color: var(--text-primary);
            line-height: 1.42;
            -webkit-font-smoothing: antialiased;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(18px) saturate(160%);
            border-radius: var(--radius-lg);
            border: 1px solid var(--card-border);
            padding: 26px 28px 30px;
            margin-bottom: 24px;
            box-shadow: var(--shadow-soft), 0 0 0 1px rgba(255,255,255,0.03) inset;
            position: relative;
            overflow: hidden;
            transition: border-color .25s ease, transform .25s ease, box-shadow .25s ease;
        }
        .glass-card:before {
            content: ""; position:absolute; inset:0; pointer-events:none;
            background: linear-gradient(120deg, rgba(255,255,255,0.06), rgba(255,255,255,0) 40%);
            mix-blend-mode: overlay; opacity:.4;
        }
        .glass-card:hover { border-color: var(--card-border-strong); box-shadow: var(--shadow-hover); }
        .glass-card h3 { letter-spacing:.5px; font-weight:600; }

        /* Header */
        .header {
            text-align: center;
            margin-bottom: 30px;
        }

    .header-inner { display:flex; flex-direction:column; gap:10px; align-items:center; }
    .header h1 { font-size: 2.55rem; font-weight:700; background: var(--gradient-accent); -webkit-background-clip:text; color:transparent; letter-spacing:1px; }
    .header p { font-size:1.05rem; color: var(--text-secondary); letter-spacing:.4px; }
    .toolbar { display:flex; gap:12px; margin-top:4px; }
    .btn-ghost { background:rgba(255,255,255,0.08); border:1px solid rgba(255,255,255,0.12); padding:8px 16px; border-radius: var(--radius-sm); font-size:.8rem; letter-spacing:.6px; cursor:pointer; color:var(--text-secondary); display:inline-flex; gap:6px; align-items:center; transition:all .25s; }
    .btn-ghost:hover { color:#fff; border-color:var(--text-secondary); }
    .btn-ghost:active { transform:translateY(1px); }

        /* Search Section */
        .search-section {
            margin-bottom: 30px;
        }

        .search-container {
            position: relative;
            max-width: 600px;
            margin: 0 auto;
        }

    .search-input { width:100%; padding:16px 22px; font-size:1rem; border:1px solid rgba(255,255,255,0.12); border-radius:18px; background:rgba(255,255,255,0.07); color:#fff; outline:none; transition:border-color .25s, background .25s; }
    .search-input:focus { border-color: var(--accent); background:rgba(255,255,255,0.12); box-shadow: var(--accent-glow); }

    .search-btn { position:absolute; right:6px; top:50%; transform:translateY(-50%); background:var(--gradient-accent); color:#fff; border:none; padding:12px 22px; border-radius:16px; cursor:pointer; font-weight:600; font-size:.85rem; letter-spacing:.5px; box-shadow:0 4px 14px -4px rgba(13,110,253,0.55); transition:filter .25s, transform .25s; }
    .search-btn:hover { filter:brightness(1.15); }

        /* Grid Layout */
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .grid-full {
            grid-column: 1 / -1;
        }

        /* Loading Animation */
        .loading {
            display: none;
            text-align: center;
            color: white;
            font-size: 1.2rem;
            margin: 20px 0;
        }

        .loading.active {
            display: block;
        }

        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top: 4px solid white;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Analysis Results */
        .analysis-results {
            display: none;
        }

        .analysis-results.active {
            display: block;
        }

        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 25px;
        }

    .metric-card { background: linear-gradient(145deg, rgba(255,255,255,0.08), rgba(255,255,255,0.02)); border-radius:18px; padding:18px 16px 20px; text-align:center; border:1px solid rgba(255,255,255,0.07); position:relative; overflow:hidden; }
    .metric-card:after { content:""; position:absolute; inset:0; background:linear-gradient(120deg,transparent,rgba(255,255,255,0.08) 60%,transparent); opacity:.15; }

    .metric-value { font-size:1.5rem; font-weight:600; color: #fff; margin-bottom:4px; letter-spacing:.5px; }

        .metric-label {
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Signal Display */
    .signal-display { text-align:center; padding:34px 20px 30px; border-radius:30px; margin-bottom:10px; background: radial-gradient(circle at 50% 0%, rgba(13,110,253,0.35), rgba(13,110,253,0) 65%); position:relative; }
    .signal-display:before { content:""; position:absolute; inset:0; background:linear-gradient(140deg,rgba(255,255,255,0.1),rgba(255,255,255,0)); mix-blend-mode:overlay; opacity:.35; }

    .signal-value { font-size:3.1rem; font-weight:800; letter-spacing:1px; margin-bottom:10px; }

        .signal-score {
            font-size: 1.5rem;
            opacity: 0.9;
            margin-bottom: 15px;
        }

        .signal-weights {
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
        }

    .weight-item { background:rgba(255,255,255,0.08); padding:6px 14px; border-radius:14px; font-size:0.7rem; letter-spacing:1px; text-transform:uppercase; color:var(--text-secondary); }

        /* Position Management */
        .position-management {
            margin-top: 25px;
        }

    .recommendation { background:rgba(255,255,255,0.06); border-radius:18px; padding:18px 20px 20px; margin-bottom:18px; border:1px solid rgba(255,255,255,0.08); position:relative; }
    .recommendation h4 { font-size:1.05rem; letter-spacing:.5px; }
    .recommendation:before { content:""; position:absolute; inset:0; border-radius:inherit; background:linear-gradient(120deg,rgba(255,255,255,0.12),rgba(255,255,255,0) 60%); opacity:.1; pointer-events:none; }

        .recommendation h4 {
            color: white;
            margin-bottom: 10px;
            font-size: 1.2rem;
        }

        .recommendation p {
            color: rgba(255, 255, 255, 0.9);
            margin-bottom: 8px;
        }

        .confidence-bar {
            background: rgba(255, 255, 255, 0.2);
            height: 8px;
            border-radius: 4px;
            overflow: hidden;
            margin-top: 10px;
        }

        .confidence-fill {
            height: 100%;
            background: #000000;
            transition: width 0.5s ease;
        }

        /* Extended Technical Analysis Styles */
    .indicator-section { margin:22px 0; padding:18px 18px 20px; background:rgba(255,255,255,0.04); border-radius:18px; border:1px solid rgba(255,255,255,0.08); }

        .indicator-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }

    .indicator-item { display:flex; justify-content:space-between; gap:10px; align-items:center; padding:8px 14px; background:rgba(255,255,255,0.05); border-radius:12px; border:1px solid rgba(255,255,255,0.07); }

        .indicator-name {
            color: rgba(255, 255, 255, 0.8);
            font-weight: 500;
        }

        .indicator-value {
            color: white;
            font-weight: 700;
            font-size: 1.1em;
        }

        .indicator-signal {
            color: rgba(255, 255, 255, 0.6);
            font-size: 0.9em;
            font-style: italic;
        }

        .risk-indicators, .levels-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 10px;
        }

        .risk-item, .level-item {
            padding: 12px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .risk-label, .level-name {
            display: block;
            color: rgba(255, 255, 255, 0.7);
            font-size: 0.9em;
            margin-bottom: 5px;
        }

        .risk-value, .level-value {
            color: white;
            font-weight: 700;
            font-size: 1.2em;
        }

        .risk-level, .level-distance {
            color: rgba(255, 255, 255, 0.6);
            font-size: 0.85em;
        }

        .fib-levels {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }

        .fib-item {
            text-align: center;
            padding: 10px;
            background: rgba(111, 66, 193, 0.1);
            border-radius: 8px;
            border: 1px solid rgba(111, 66, 193, 0.3);
        }

        .fib-label {
            display: block;
            color: #6f42c1;
            font-weight: 600;
            font-size: 0.9em;
        }

        .fib-value {
            color: white;
            font-weight: 700;
            font-size: 1.1em;
        }

        .extreme-signal {
            color: #6f42c1 !important;
            font-weight: 700;
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.6; }
            100% { opacity: 1; }
        }

        /* Enhanced Metric Cards */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }

        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            border-color: rgba(255,255,255,0.4);
        }

        .metric-value {
            font-size: 1.4em;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .metric-label {
            color: rgba(255,255,255,0.7);
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* Liquidation Tables */
        .liquidation-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        .liquidation-table th,
        .liquidation-table td {
            padding: 12px;
            text-align: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .liquidation-table th {
            background: rgba(255, 255, 255, 0.1);
            color: white;
            font-weight: 600;
        }

        .liquidation-table td {
            color: rgba(255, 255, 255, 0.9);
        }

        /* Pattern Display */
    .pattern-item { background:rgba(255,255,255,0.06); border-radius:18px; padding:16px 18px 18px; margin-bottom:16px; border:1px solid rgba(255,255,255,0.08); position:relative; overflow:hidden; }
    .pattern-item:before { content:""; position:absolute; inset:0; background:linear-gradient(130deg,rgba(255,255,255,0.15),rgba(255,255,255,0)); opacity:.07; }

        .pattern-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 8px;
        }

        .pattern-type {
            color: white;
            font-weight: 600;
        }

    .pattern-confidence { background:rgba(255,255,255,0.12); padding:4px 10px; border-radius:14px; font-size:.65rem; letter-spacing:.5px; text-transform:uppercase; color:var(--text-secondary); }

    /* Unified Section Titles */
    .section-title { display:flex; align-items:center; gap:10px; margin:-4px 0 20px; padding:0 0 10px; font-size:1rem; font-weight:600; letter-spacing:.6px; color:var(--text-primary); position:relative; }
    .section-title:after { content:""; position:absolute; left:0; bottom:0; height:2px; width:60px; background:var(--gradient-accent); border-radius:2px; }
    .section-title .tag { font-size:.55rem; background:rgba(255,255,255,0.12); padding:4px 8px; border-radius:8px; letter-spacing:1px; color:var(--text-secondary); }
    .section-title .icon { font-size:1.15rem; filter:drop-shadow(0 2px 4px rgba(0,0,0,0.4)); }
    .sub-note { font-size:.6rem; color:var(--text-dim); letter-spacing:.5px; margin-top:-12px; margin-bottom:10px; }

        /* Responsive Design */
        @media (max-width: 768px) {
            .container {
                padding: 15px;
            }

            .header h1 {
                font-size: 2rem;
            }

            .metrics-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .signal-value {
                font-size: 2rem;
            }

            .signal-weights {
                flex-direction: column;
                align-items: center;
            }
        }

        /* Color Classes */
        .text-success { color: #28a745 !important; }
        .text-danger { color: #dc3545 !important; }
        .text-warning { color: #ffc107 !important; }
        .text-info { color: #17a2b8 !important; }
        .text-primary { color: #007bff !important; }

        .bg-success { background-color: #28a745 !important; }
        .bg-danger { background-color: #dc3545 !important; }
        .bg-warning { background-color: #ffc107 !important; }
        .bg-info { background-color: #17a2b8 !important; }
        .bg-primary { background-color: #007bff !important; }

        /* Animation Classes */
        .fade-in {
            animation: fadeIn 0.5s ease-in;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .pulse {
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(255, 255, 255, 0.4); }
            70% { box-shadow: 0 0 0 20px rgba(255, 255, 255, 0); }
            100% { box-shadow: 0 0 0 0 rgba(255, 255, 255, 0); }
        }
    /* Trade Setups */
    .setup-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(250px,1fr)); gap:18px; }
    .setup-card { position:relative; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); padding:16px 18px 18px; border-radius:16px; }
    .setup-title { font-size:1.05rem; font-weight:600; margin:0 0 4px; display:flex; align-items:center; gap:8px; }
    .setup-badge { font-size:0.6rem; background:#0d6efd; color:#fff; padding:3px 6px; border-radius:6px; letter-spacing:0.5px; }
    .setup-badge.short { background:#dc3545; }
    .setup-badge.long { background:#198754; }
    
    /* Pattern Trading Cards */
    .pattern-card {
        background: linear-gradient(135deg, rgba(255, 215, 0, 0.08), rgba(255, 165, 0, 0.04));
        border: 1px solid rgba(255, 215, 0, 0.2);
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.1);
    }
    .pattern-card:hover {
        border-color: rgba(255, 215, 0, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 215, 0, 0.15);
    }
    .pattern-badge {
        background: linear-gradient(45deg, #FFD700, #FFA500) !important;
        color: #000 !important;
        font-weight: 700;
        text-shadow: none;
    }
    .target-pill.pattern-target {
        background: linear-gradient(45deg, rgba(255, 215, 0, 0.2), rgba(255, 165, 0, 0.1));
        border: 1px solid rgba(255, 215, 0, 0.3);
        color: #FFD700;
    }
    .setup-line { display:flex; justify-content:space-between; font-size:0.72rem; padding:1px 0; color:rgba(255,255,255,0.82); }
    .setup-sep { margin:6px 0 8px; height:1px; background:linear-gradient(90deg,rgba(255,255,255,0),rgba(255,255,255,0.18),rgba(255,255,255,0)); }
    .confidence-chip { position:absolute; top:10px; right:10px; background:#198754; color:#fff; padding:4px 10px; font-size:0.65rem; font-weight:600; border-radius:14px; }
    .confidence-chip.mid { background:#fd7e14; }
    .confidence-chip.low { background:#dc3545; }
    .targets { display:flex; gap:6px; flex-wrap:wrap; margin-top:6px; }
    .target-pill { background:rgba(255,255,255,0.1); padding:4px 8px; border-radius:12px; font-size:0.65rem; }
    .conditions { list-style:none; margin:8px 0 0; padding:0; font-size:0.6rem; line-height:1.1rem; color:rgba(255,255,255,0.55); }
    .conditions li { display:flex; gap:4px; align-items:center; }
    .c-ok { color:#20c997; }
    .c-warn { color:#ffc107; }
    .c-bad { color:#dc3545; }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <div class="header-inner">
                <h1>ðŸš€ Ultimate Trading System V5</h1>
                <p>Professional Analysis â€¢ Intelligent Position Management â€¢ JAX Neural Networks</p>
                <div class="toolbar">
                    <button id="themeToggle" class="btn-ghost" title="Theme umschalten">ðŸŒ— Theme</button>
                    <button id="refreshBtn" class="btn-ghost" onclick="searchSymbol()" title="Neu analysieren">ðŸ”„ Refresh</button>
                </div>
            </div>
        </div>

        <!-- Search Section -->
        <div class="glass-card search-section">
            <div class="search-container">
                <input type="text" id="searchInput" class="search-input" 
                       placeholder="Enter symbol (e.g., BTC, ETH, DOGE...)" 
                       onkeypress="if(event.key==='Enter') searchSymbol()">
                <button class="search-btn" onclick="searchSymbol()">ðŸ” Analyze</button>
            </div>
        </div>

        <!-- Loading Animation -->
        <div id="loadingSection" class="loading">
            <div class="spinner"></div>
            <p>ðŸ§  Analyzing with AI â€¢ ðŸ“Š Calculating Patterns â€¢ ðŸ’¡ Generating Insights...</p>
        </div>

        <!-- Analysis Results -->
        <div id="analysisResults" class="analysis-results">
            <!-- Main Signal Display -->
            <div class="glass-card">
                <div id="signalDisplay" class="signal-display">
                    <!-- Signal content will be inserted here -->
                </div>
            </div>

            <!-- Key Metrics -->
            <div class="glass-card">
                <div class="section-title"><span class="icon">ðŸ“Š</span> Key Metrics <span class="tag">LIVE</span></div>
                <div id="metricsGrid" class="metrics-grid">
                    <!-- Metrics will be inserted here -->
                </div>
            </div>

            <!-- Trade Setups -->
            <div class="glass-card" id="tradeSetupsCard">
                <h3 style="color: white; margin-bottom: 16px; display:flex; align-items:center; gap:10px;">ðŸ› ï¸ Trade Setups <span style="font-size:0.7rem; background:rgba(255,255,255,0.08); padding:4px 8px; border-radius:8px; letter-spacing:1px;">BETA</span></h3>
                <div id="tradeSetupsContent" class="setup-grid"></div>
                <div id="tradeSetupsStatus" style="font-size:0.75rem; color:rgba(255,255,255,0.6); margin-top:10px;"></div>
            </div>

            <!-- Position Size Calculator -->
            <div class="glass-card" id="positionSizerCard">
                <div class="section-title"><span class="icon">ðŸ“</span> Position Size Calculator <span class="tag">RISK</span></div>
                <div style="display:grid; gap:14px; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); margin-bottom:14px;">
                    <div>
                        <label style="font-size:.6rem; letter-spacing:1px; text-transform:uppercase; color:var(--text-dim);">Equity ($)</label>
                        <input id="psEquity" type="number" value="10000" class="search-input" style="padding:10px 14px; font-size:.8rem;">
                    </div>
                    <div>
                        <label style="font-size:.6rem; letter-spacing:1px; text-transform:uppercase; color:var(--text-dim);">Risk %</label>
                        <input id="psRiskPct" type="number" value="1" class="search-input" style="padding:10px 14px; font-size:.8rem;">
                    </div>
                    <div>
                        <label style="font-size:.6rem; letter-spacing:1px; text-transform:uppercase; color:var(--text-dim);">Entry</label>
                        <input id="psEntry" type="number" step="0.01" class="search-input" style="padding:10px 14px; font-size:.8rem;">
                    </div>
                    <div>
                        <label style="font-size:.6rem; letter-spacing:1px; text-transform:uppercase; color:var(--text-dim);">Stop</label>
                        <input id="psStop" type="number" step="0.01" class="search-input" style="padding:10px 14px; font-size:.8rem;">
                    </div>
                </div>
                <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:12px;">
                    <button class="btn-ghost" onclick="prefillFromFirstSetup()">â¤µï¸ Aus Setup Ã¼bernehmen</button>
                    <button class="btn-ghost" onclick="calcPositionSize()">ðŸ§® Berechnen</button>
                </div>
                <div id="psResult" style="font-size:.7rem; color:var(--text-secondary); line-height:1.1rem;"></div>
            </div>

            <!-- Two Column Layout -->
            <div class="grid">
                <!-- Position Management -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">ðŸŽ¯</span> Intelligent Position Management</div>
                    <div id="positionRecommendations">
                        <!-- Position recommendations will be inserted here -->
                    </div>
                </div>

                <!-- Technical Analysis -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">ðŸ“ˆ</span> Technical Analysis</div>
                    <div id="technicalAnalysis">
                        <!-- Technical analysis will be inserted here -->
                    </div>
                </div>

                <!-- Pattern Recognition -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">ðŸ”</span> Chart Patterns</div>
                    <div id="patternAnalysis">
                        <!-- Pattern analysis will be inserted here -->
                    </div>
                </div>

                <!-- Multi-Timeframe -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">ðŸ•’</span> Multi-Timeframe</div>
                    <div id="multiTimeframe">
                        <!-- MTF analysis -->
                    </div>
                </div>

                <!-- Market Regime -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">ðŸŽ¯</span> Market Regime <span class="tag">BETA</span></div>
                    <div id="regimeAnalysis">
                        <!-- Regime analysis -->
                    </div>
                </div>

                <!-- Adaptive Risk & Targets -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">ðŸŽ¯</span> Adaptive Risk Management <span class="tag">NEW</span></div>
                    <div id="adaptiveRiskTargets">
                        <!-- Adaptive risk and targets -->
                    </div>
                </div>

                <!-- Order Flow Analysis -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">ðŸ“Š</span> Order Flow <span class="tag">NEW</span></div>
                    <div id="orderFlowAnalysis">
                        <!-- Order flow analysis -->
                    </div>
                </div>

                <!-- AI Analysis -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">ðŸ¤–</span> JAX Neural Network</div>
                    <div id="aiAnalysis">
                        <!-- AI analysis will be inserted here -->
                    </div>
                    <div id="aiStatus" style="margin-top:14px; font-size:0.65rem; color:var(--text-dim); line-height:1rem;">
                        <!-- AI status -->
                    </div>
                </div>

                <!-- Feature Contributions -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">ðŸ”</span> AI Explainability <span class="tag">NEW</span></div>
                    <div id="featureContributions">
                        <!-- Feature contributions analysis -->
                    </div>
                </div>

                <!-- Backtest -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">ðŸ§ª</span> Backtest <span class="tag">BETA</span></div>
                    <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;">
                        <select id="btInterval" class="search-input" style="flex:0 0 110px; padding:8px 10px; font-size:0.65rem;">
                            <option value="1h">1h</option>
                            <option value="30m">30m</option>
                            <option value="15m">15m</option>
                            <option value="4h">4h</option>
                            <option value="1d">1d</option>
                        </select>
                        <input id="btLimit" type="number" value="500" min="100" max="1000" class="search-input" style="flex:0 0 110px; padding:8px 10px; font-size:0.65rem;" />
                        <button class="btn-ghost" onclick="runBacktest()" style="font-size:0.65rem;">â–¶ï¸ Run</button>
                    </div>
                    <div id="backtestStatus" style="font-size:0.65rem; color:var(--text-secondary); margin-bottom:8px;"></div>
                    <div id="backtestResults" style="font-size:0.65rem; line-height:1rem; color:var(--text-secondary);"></div>
                </div>
            </div>

            <!-- Liquidation Calculator -->
            <div class="glass-card grid-full">
                <div class="section-title"><span class="icon">ðŸ’°</span> Liquidation Calculator</div>
                <div class="grid">
                    <div>
                        <h4 style="color: #28a745; margin-bottom: 15px;">ðŸ“ˆ LONG Positions</h4>
                        <div style="overflow-x: auto;">
                            <table id="liquidationLongTable" class="liquidation-table">
                                <!-- Long liquidation data will be inserted here -->
                            </table>
                        </div>
                    </div>
                    <div>
                        <h4 style="color: #dc3545; margin-bottom: 15px;">ðŸ“‰ SHORT Positions</h4>
                        <div style="overflow-x: auto;">
                            <table id="liquidationShortTable" class="liquidation-table">
                                <!-- Short liquidation data will be inserted here -->
                            </table>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentSymbol = '';
        let analysisData = null;

        // Search and analyze symbol
        async function searchSymbol() {
            const query = document.getElementById('searchInput').value.trim();
            if (!query) {
                alert('Please enter a symbol');
                return;
            }

            showLoading(true);
            currentSymbol = query.toUpperCase();

            try {
                const response = await fetch(`/api/analyze/${currentSymbol}`);
                const result = await response.json();

                if (result.success) {
                    analysisData = result.data;
                    displayAnalysis(analysisData);
                } else {
                    alert(`Error: ${result.error}`);
                }
            } catch (error) {
                console.error('Analysis error:', error);
                alert('Failed to analyze symbol. Please try again.');
            } finally {
                showLoading(false);
            }
        }

        // Show/hide loading animation
        function showLoading(show) {
            const loading = document.getElementById('loadingSection');
            const results = document.getElementById('analysisResults');
            
            if (show) {
                loading.classList.add('active');
                results.classList.remove('active');
            } else {
                loading.classList.remove('active');
                results.classList.add('active');
            }
        }

        // Display complete analysis
        function displayAnalysis(data) {
            displayMainSignal(data);
            displayEnterpriseValidation(data); // NEW: Enterprise Validation
            displayMetrics(data);
            displayTradeSetups(data);
            window.__lastAnalysis = data; // store for position sizing
            displayPositionManagement(data);
            displayTechnicalAnalysis(data);
            displayPatternAnalysis(data);
            displayMultiTimeframe(data);
            displayRegimeAnalysis(data);
            displayAdaptiveRiskTargets(data);
            displayOrderFlowAnalysis(data);
            displayMarketBias(data);
            displayAIAnalysis(data);
            displayFeatureContributions(data);
            displayLiquidationTables(data);
        }

        // Display main trading signal
        function displayMainSignal(data) {
            const signal = data.final_score;
            const signalDisplay = document.getElementById('signalDisplay');
            
            signalDisplay.innerHTML = `
                <div class="signal-value" style="color: ${signal.signal_color}">
                    ${signal.signal}
                </div>
                <div class="signal-score" style="color: white">
                    Score: ${signal.score}/100
                </div>
                <div class="signal-weights">
                    <div class="weight-item">ðŸ“Š Technical: ${signal.technical_weight}</div>
                    <div class="weight-item">ðŸ” Patterns: ${signal.pattern_weight}</div>
                    <div class="weight-item">ðŸ¤– AI: ${signal.ai_weight}</div>
                </div>
            `;
        }

        // NEW: Enterprise Validation Display
        function displayEnterpriseValidation(data) {
            const validation = data.final_score.validation;
            const validationDiv = document.getElementById('enterpriseValidation') || createValidationDiv();
            
            let html = `
                <h3>ðŸ¢ ENTERPRISE VALIDATION</h3>
                <div class="validation-header">
                    <div class="trading-action" style="color: ${validation.trading_action === 'WAIT' ? '#dc3545' : '#28a745'}">
                        EMPFEHLUNG: ${validation.trading_action}
                    </div>
                    <div class="risk-level" style="color: ${getRiskColor(validation.risk_level)}">
                        RISIKO: ${validation.risk_level}
                    </div>
                    <div class="enterprise-ready" style="color: ${validation.enterprise_ready ? '#28a745' : '#dc3545'}">
                        ${validation.enterprise_ready ? 'âœ… ENTERPRISE READY' : 'âŒ NICHT BEREIT'}
                    </div>
                </div>
            `;

            // Contradictions (WidersprÃ¼che)
            if (validation.contradictions.length > 0) {
                html += `<div class="contradictions-section">
                    <h4 style="color: #dc3545">âš ï¸ WIDERSPRÃœCHE GEFUNDEN</h4>`;
                validation.contradictions.forEach(contradiction => {
                    html += `<div class="contradiction-item" style="border-left: 3px solid #dc3545; padding-left: 10px; margin: 5px 0;">
                        <div style="color: #dc3545; font-weight: bold">${contradiction.type}</div>
                        <div style="color: white">${contradiction.message}</div>
                        <div style="color: #ffc107; font-style: italic">${contradiction.recommendation}</div>
                    </div>`;
                });
                html += `</div>`;
            }

            // Warnings
            if (validation.warnings.length > 0) {
                html += `<div class="warnings-section">
                    <h4 style="color: #ffc107">âš ï¸ WARNUNGEN</h4>`;
                validation.warnings.forEach(warning => {
                    html += `<div class="warning-item" style="border-left: 3px solid #ffc107; padding-left: 10px; margin: 5px 0;">
                        <div style="color: #ffc107; font-weight: bold">${warning.type}</div>
                        <div style="color: white">${warning.message}</div>
                        <div style="color: #17a2b8; font-style: italic">${warning.recommendation}</div>
                    </div>`;
                });
                html += `</div>`;
            }

            // Confidence Factors
            html += `<div class="confidence-section">
                <h4 style="color: #17a2b8">âœ… CONFIDENCE FAKTOREN</h4>`;
            validation.confidence_factors.forEach(factor => {
                html += `<div class="confidence-item" style="color: white; margin: 2px 0;">${factor}</div>`;
            });
            html += `</div>`;

            validationDiv.innerHTML = html;
        }

        function createValidationDiv() {
            const div = document.createElement('div');
            div.id = 'enterpriseValidation';
            div.className = 'analysis-card';
            div.style.marginTop = '20px';
            
            // Insert after signal display
            const signalCard = document.querySelector('.signal-card');
            if (signalCard && signalCard.parentNode) {
                signalCard.parentNode.insertBefore(div, signalCard.nextSibling);
            }
            
            return div;
        }

        function getRiskColor(riskLevel) {
            switch(riskLevel) {
                case 'LOW': return '#28a745';
                case 'MEDIUM': return '#ffc107';
                case 'HIGH': return '#fd7e14';
                case 'VERY_HIGH': return '#dc3545';
                default: return '#6c757d';
            }
        }

        // Display key metrics
        function displayMetrics(data) {
            const metrics = [
                { label: 'Current Price', value: `$${data.current_price.toLocaleString()}`, color: 'white' },
                { label: '24h Change', value: `${parseFloat(data.market_data.priceChangePercent).toFixed(2)}%`, 
                  color: parseFloat(data.market_data.priceChangePercent) >= 0 ? '#28a745' : '#dc3545' },
                { label: 'RSI', value: `${data.technical_analysis.rsi.rsi.toFixed(1)}`, 
                  color: getRSIColor(data.technical_analysis.rsi.rsi) },
                { label: 'Volume 24h', value: formatVolume(data.market_data.volume), color: '#17a2b8' },
                { label: 'Support', value: `$${data.technical_analysis.support.toLocaleString()}`, color: '#28a745' },
                { label: 'Resistance', value: `$${data.technical_analysis.resistance.toLocaleString()}`, color: '#dc3545' }
            ];

            const metricsGrid = document.getElementById('metricsGrid');
            metricsGrid.innerHTML = metrics.map(metric => `
                <div class="metric-card fade-in">
                    <div class="metric-value" style="color: ${metric.color}">${metric.value}</div>
                    <div class="metric-label">${metric.label}</div>
                </div>
            `).join('');
        }

        // Trade Setups Renderer (array based) - Enhanced for Pattern Trades
        function displayTradeSetups(data) {
            const container = document.getElementById('tradeSetupsContent');
            const status = document.getElementById('tradeSetupsStatus');
            const setups = data.trade_setups || [];
            if (!Array.isArray(setups) || setups.length === 0) {
                container.innerHTML = '';
                status.textContent = 'Keine Setups generiert (Bedingungen nicht erfÃ¼llt).';
                return;
            }

            // Separate Pattern trades from regular trades
            const patternTrades = setups.filter(s => s.pattern_name || s.setup_type);
            const regularTrades = setups.filter(s => !s.pattern_name && !s.setup_type);

            let html = '';

            // Pattern Trades Section
            if (patternTrades.length > 0) {
                html += `<div class="trade-section">
                    <h4 style="color: #FFD700; margin-bottom: 12px; font-size: 0.85rem; display: flex; align-items: center;">
                        ðŸŽ¯ <span style="margin-left: 6px;">Chart Pattern Setups (${patternTrades.length})</span>
                    </h4>`;
                
                const patternBlocks = patternTrades.map(s => {
                    const confClass = s.confidence >= 70 ? '' : (s.confidence >= 55 ? 'mid' : 'low');
                    const targets = (s.targets || s.take_profits || []).map(t=>{
                        const price = t.price || t.level;
                        const label = t.label || t.level;
                        const percentage = t.percentage ? ` (${t.percentage}%)` : '';
                        const rr = t.rr ? ` ${t.rr}R` : '';
                        return `<span class="target-pill pattern-target">${label}: ${price}${percentage}${rr}</span>`;
                    }).join('');
                    
                    return `
                    <div class="setup-card pattern-card" style="border-left: 4px solid ${s.direction==='LONG'?'#28a745':'#dc3545'};">
                        <div class="confidence-chip ${confClass}">${s.confidence}%</div>
                        <div class="setup-title">
                            ${s.direction} 
                            <span class="setup-badge pattern-badge ${s.direction==='LONG'?'long':'short'}" style="background: linear-gradient(45deg, #FFD700, #FFA500); color: #000;">
                                ${s.pattern_name || s.strategy}
                            </span>
                        </div>
                        <div class="setup-line"><span>Entry</span><span>${s.entry_price || s.entry}</span></div>
                        <div class="setup-line"><span>Stop</span><span>${s.stop_loss}</span></div>
                        <div class="setup-line"><span>Risk%</span><span>${s.risk_percent || s.risk_reward_ratio}%</span></div>
                        ${s.risk_reward_ratio ? `<div class="setup-line"><span>R/R</span><span style="color: #28a745;">${s.risk_reward_ratio}</span></div>` : ''}
                        ${s.key_level ? `<div class="setup-line"><span>Key Level</span><span style="color: #FFD700;">${s.key_level}</span></div>` : ''}
                        ${s.regime_context ? `<div class="setup-line"><span>Regime</span><span style="color: #17a2b8;">${s.regime_context}</span></div>` : ''}
                        <div class="setup-sep"></div>
                        <div class="targets">${targets}</div>
                        ${s.rsi_caution && s.rsi_caution.caution_level !== 'none' ? `<div style="margin-top:6px; font-size:.5rem; color:#ffc107; line-height:0.75rem;"><strong>RSI:</strong> ${s.rsi_caution.narrative}</div>` : ''}
                        ${s.trade_plan ? `<div style="margin-top:8px; font-size:.55rem; color:#FFD700; line-height:0.75rem;"><strong>Plan:</strong> ${s.trade_plan}</div>` : ''}
                        ${s.market_structure ? `<div style="margin-top:4px; font-size:.55rem; color:rgba(255,255,255,0.7); line-height:0.75rem;"><strong>Structure:</strong> ${s.market_structure}</div>` : ''}
                        <div style="margin-top:6px; font-size:.55rem; color:rgba(255,255,255,0.55); line-height:0.75rem;">${s.rationale || s.trade_plan}</div>
                    </div>`;
                });
                html += patternBlocks.join('') + '</div>';
            }

            // Regular Technical Trades Section
            if (regularTrades.length > 0) {
                html += `<div class="trade-section">
                    <h4 style="color: #17a2b8; margin-bottom: 12px; font-size: 0.85rem; display: flex; align-items: center;">
                        ðŸ“Š <span style="margin-left: 6px;">Technical Analysis Setups (${regularTrades.length})</span>
                    </h4>`;
                
                const regularBlocks = regularTrades.map(s => {
                    const confClass = s.confidence >= 70 ? '' : (s.confidence >= 55 ? 'mid' : 'low');
                    const targets = (s.targets||[]).map(t=>`<span class="target-pill">${t.label}: ${t.price} (${t.rr}R)</span>`).join('');
                    const conds = (s.conditions||[]).map(c=>`<li class="${c.s==='ok'?'c-ok':(c.s==='bad'?'c-bad':'c-warn')}">${c.t}</li>`).join('');
                    
                    return `
                    <div class="setup-card">
                        <div class="confidence-chip ${confClass}">${s.confidence}%</div>
                        <div class="setup-title">
                            ${s.direction} 
                            <span class="setup-badge ${s.direction==='LONG'?'long':'short'}">${s.strategy}</span>
                            ${s.validation_score ? `<span class="validation-badge ${s.validation_score.toLowerCase()}">${s.validation_score}</span>` : ''}
                        </div>
                        <div class="setup-line"><span>Entry</span><span>${s.entry}</span></div>
                        <div class="setup-line"><span>Stop</span><span>${s.stop_loss}</span></div>
                        <div class="setup-line"><span>Risk%</span><span>${s.risk_percent}%</span></div>
                        ${s.primary_rr ? `<div class="setup-line"><span>R/R</span><span style="color: #28a745;">${s.primary_rr}R</span></div>` : ''}
                        <div class="setup-sep"></div>
                        <div class="targets">${targets}</div>
                        <ul class="conditions">${conds}</ul>
                        <div style="margin-top:6px; font-size:.55rem; color:rgba(255,255,255,0.55); line-height:0.75rem;">${s.rationale}</div>
                    </div>`;
                });
                html += regularBlocks.join('') + '</div>';
            }

            container.innerHTML = html;
            status.textContent = `${setups.length} Trading-Setups generiert (${patternTrades.length} Pattern + ${regularTrades.length} Technical)`;
        }

        // Display position management recommendations
        function displayPositionManagement(data) {
            const positions = data.position_analysis;
            const recommendations = positions.recommendations || [];
            
            let html = `
                <div class="metrics-grid" style="margin-bottom: 20px;">
                    <div class="metric-card">
                        <div class="metric-value text-success">${positions.resistance_potential.toFixed(1)}%</div>
                        <div class="metric-label">Resistance Potential</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value text-danger">${positions.support_risk.toFixed(1)}%</div>
                        <div class="metric-label">Support Risk</div>
                    </div>
                </div>
            `;

            html += recommendations.map(rec => `
                <div class="recommendation fade-in" style="border-left-color: ${rec.color}">
                    <h4>${rec.type}: ${rec.action}</h4>
                    <p><strong>Reason:</strong> ${rec.reason}</p>
                    <p>${rec.details}</p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${rec.confidence}%"></div>
                    </div>
                    <small style="color: rgba(255,255,255,0.7)">Confidence: ${rec.confidence}%</small>
                </div>
            `).join('');

            document.getElementById('positionRecommendations').innerHTML = html;
        }

        // Display technical analysis
    function displayTechnicalAnalysis(data) {
            const tech = data.technical_analysis;
            const extended = data.extended_analysis || {}; // Safety check
            
            // Check if extended analysis is available
            if (!extended || Object.keys(extended).length === 0) {
                console.log("Extended analysis not available, falling back to basic display");
                displayBasicTechnicalAnalysis(tech);
                return;
            }
            
            // Horizontal compact layout (scrollable row of groups)
            const html = `
              <div style="display:flex; gap:18px; overflow-x:auto; padding-bottom:6px; scrollbar-width:thin;" class="ta-horizontal">
                 <!-- Snapshot Cards -->
                 <div style="flex:0 0 260px; display:flex; flex-direction:column; gap:12px;">
                     <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:12px;">
                         <div class="metric-card" style="min-height:92px;">
                             <div class="metric-value ${getTrendColor(tech.trend.trend)}" style="font-size:1.1rem;">${tech.trend.trend.toUpperCase()}</div>
                             <div class="metric-label" style="font-size:.5rem;">TREND</div>
                         </div>
                         <div class="metric-card" style="min-height:92px;">
                             <div class="metric-value" style="font-size:1.05rem;">${tech.macd.curve_direction.replace('_',' ').toUpperCase()}</div>
                             <div class="metric-label" style="font-size:.5rem;">MACD-SIGNAL</div>
                         </div>
                         <div class="metric-card" style="min-height:92px;">
                             <div class="metric-value ${getIndicatorColor(extended?.stochastic?.signal)}" style="font-size:1.05rem;">${(extended?.stochastic?.signal||'neutral').toUpperCase()}</div>
                             <div class="metric-label" style="font-size:.5rem;">STOCHASTISCH</div>
                         </div>
                         <div class="metric-card" style="min-height:92px;">
                             <div class="metric-value ${getVolatilityColor(extended.atr.volatility)}" style="font-size:1.05rem;">${extended.atr.volatility.toUpperCase()}</div>
                             <div class="metric-label" style="font-size:.5rem;">VOLATILITÃ„T (ATR)</div>
                         </div>
                     </div>
                 </div>
                 <!-- Core Indicators -->
                 <div style="flex:0 0 230px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:14px 16px; backdrop-filter:blur(4px);">
                     <div style="font-size:.6rem; letter-spacing:.6px; font-weight:600; color:#17a2b8; margin-bottom:10px; display:flex; align-items:center; gap:4px;">ðŸ“Š KERNINDIKATOREN</div>
                     <div style="display:flex; flex-direction:column; gap:6px;">
                         <div style="display:flex; justify-content:space-between; font-size:.6rem;">
                             <span style="color:var(--text-secondary);">RSI:</span>
                             <span style="font-weight:600;" class="${getRsiColor(tech.rsi.rsi)}">${tech.rsi.rsi.toFixed(1)}</span>
                             <span style="opacity:.55;">(${tech.rsi.trend})</span>
                         </div>
                         <div style="display:flex; justify-content:space-between; font-size:.6rem;">
                             <span style="color:var(--text-secondary);">MACD:</span>
                             <span style="font-weight:600;">${tech.macd.macd.toFixed(4)}</span>
                             <span style="opacity:.55;">(${tech.macd.curve_direction})</span>
                         </div>
                         <div style="display:flex; justify-content:space-between; font-size:.6rem;">
                             <span style="color:var(--text-secondary);">Volumen:</span>
                             <span style="font-weight:600;">${tech.volume_analysis.ratio.toFixed(2)}x</span>
                             <span style="opacity:.55;">(${tech.volume_analysis.trend})</span>
                         </div>
                         <div style="display:flex; justify-content:space-between; font-size:.6rem;">
                             <span style="color:var(--text-secondary);">Schwung:</span>
                             <span style="font-weight:600;" class="${getMomentumColor(tech.momentum.value)}">${tech.momentum.value.toFixed(2)}%</span>
                             <span style="opacity:.55;">(${tech.momentum.trend})</span>
                         </div>
                     </div>
                 </div>
                 <!-- Advanced Indicators -->
                 <div style="flex:0 0 250px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:14px 16px;">
                     <div style="font-size:.6rem; letter-spacing:.6px; font-weight:600; color:#ffc107; margin-bottom:10px;">ðŸ”¬ ERWEITERTE</div>
                     <div style="display:flex; flex-direction:column; gap:6px; font-size:.6rem;">
                         <div style="display:flex; justify-content:space-between;">
                             <span style="color:var(--text-secondary);">Bollinger:</span>
                             <span style="font-weight:600;">${extended.bollinger_bands.signal.toUpperCase()}</span>
                             <span style="opacity:.55;">(${(extended.bollinger_bands.position*100).toFixed(0)}%)</span>
                         </div>
                         <div style="display:flex; justify-content:space-between;">
                             <span style="color:var(--text-secondary);">Stoch %K:</span>
                             <span style="font-weight:600;" class="${getStochasticColor(extended.stochastic.k)}">${extended.stochastic.k.toFixed(1)}</span>
                             <span style="opacity:.55;">%D ${extended.stochastic.d.toFixed(1)}</span>
                         </div>
                         <div style="display:flex; justify-content:space-between;">
                             <span style="color:var(--text-secondary);">Williams %R:</span>
                             <span style="font-weight:600;" class="${getWilliamsColor(extended.williams_r.value)}">${extended.williams_r.value.toFixed(1)}</span>
                             <span style="opacity:.55;">(${extended.williams_r.signal})</span>
                         </div>
                         <div style="display:flex; justify-content:space-between;">
                             <span style="color:var(--text-secondary);">CCI:</span>
                             <span style="font-weight:600;" class="${getCciColor(extended.cci.value)}">${extended.cci.value.toFixed(1)}</span>
                             <span style="opacity:.55;" class="${extended.cci.extreme ? 'extreme-signal' : ''}">${extended.cci.signal}</span>
                         </div>
                     </div>
                 </div>
                 <!-- Volatility & Risk -->
                 <div style="flex:0 0 230px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:14px 16px;">
                     <div style="font-size:.6rem; letter-spacing:.6px; font-weight:600; color:#dc3545; margin-bottom:10px;">âš ï¸ VOLA & RISK</div>
                     <div style="display:flex; flex-direction:column; gap:6px; font-size:.6rem;">
                         <div style="display:flex; justify-content:space-between;">
                             <span style="color:var(--text-secondary);">ATR %:</span>
                             <span style="font-weight:600;" class="${getVolatilityColor(extended.atr.volatility)}">${extended.atr.percentage.toFixed(2)}%</span>
                             <span style="opacity:.55;">(${extended.atr.risk_level})</span>
                         </div>
                         <div style="display:flex; justify-content:space-between;">
                             <span style="color:var(--text-secondary);">Trend Strength:</span>
                             <span style="font-weight:600;" class="${getTrendStrengthColor(extended.trend_strength.strength)}">${extended.trend_strength.strength.toUpperCase()}</span>
                             <span style="opacity:.55;">(${extended.trend_strength.direction})</span>
                         </div>
                     </div>
                 </div>
                 <!-- Levels & Fib -->
                 <div style="flex:0 0 310px; display:flex; flex-direction:column; gap:12px;">
                     <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:14px 16px;">
                         <div style="font-size:.6rem; letter-spacing:.6px; font-weight:600; color:#28a745; margin-bottom:10px;">ðŸ“ˆ LEVELS</div>
                         <div style="display:flex; flex-direction:column; gap:6px; font-size:.6rem;">
                             <div style="display:flex; justify-content:space-between;">
                                <span style="color:var(--text-secondary);">Resistance:</span>
                                <span style="font-weight:600;">${tech.resistance.toFixed(4)}</span>
                                <span style="opacity:.55; color:#dc3545;">+${(((tech.resistance - tech.current_price)/tech.current_price)*100).toFixed(2)}%</span>
                             </div>
                             <div style="display:flex; justify-content:space-between;">
                                <span style="color:var(--text-secondary);">Support:</span>
                                <span style="font-weight:600;">${tech.support.toFixed(4)}</span>
                                <span style="opacity:.55; color:#26c281;">${(((tech.support - tech.current_price)/tech.current_price)*100).toFixed(2)}%</span>
                             </div>
                             <div style="display:flex; justify-content:space-between;">
                                <span style="color:var(--text-secondary);">Pivot:</span>
                                <span style="font-weight:600;">${extended.pivot_points.pivot.toFixed(4)}</span>
                                <span style="opacity:.55;">R1 ${extended.pivot_points.r1.toFixed(4)}</span>
                             </div>
                         </div>
                     </div>
                     <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:14px 16px;">
                         <div style="font-size:.6rem; letter-spacing:.6px; font-weight:600; color:#6f42c1; margin-bottom:10px;">ðŸŒ€ FIBONACCI</div>
                         <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:6px; font-size:.6rem;">
                             <div style="display:flex; justify-content:space-between;"><span style="color:var(--text-secondary);">23.6%:</span><span style="font-weight:600;">${extended.fibonacci.fib_236.toFixed(4)}</span></div>
                             <div style="display:flex; justify-content:space-between;"><span style="color:var(--text-secondary);">38.2%:</span><span style="font-weight:600;">${extended.fibonacci.fib_382.toFixed(4)}</span></div>
                             <div style="display:flex; justify-content:space-between;"><span style="color:var(--text-secondary);">50%:</span><span style="font-weight:600;">${extended.fibonacci.fib_500.toFixed(4)}</span></div>
                             <div style="display:flex; justify-content:space-between;"><span style="color:var(--text-secondary);">61.8%:</span><span style="font-weight:600;">${extended.fibonacci.fib_618.toFixed(4)}</span></div>
                         </div>
                     </div>
                 </div>
              </div>`;

            document.getElementById('technicalAnalysis').innerHTML = html;
        }

        // Fallback function for basic technical analysis
        function displayBasicTechnicalAnalysis(tech) {
            const html = `
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value ${getTrendColor(tech.trend.trend)}">${tech.trend.trend.toUpperCase()}</div>
                        <div class="metric-label">Trend</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${tech.macd.curve_direction.replace('_', ' ').toUpperCase()}</div>
                        <div class="metric-label">MACD Signal</div>
                    </div>
                </div>
                
                <div class="indicator-section">
                    <h4 style="color: #17a2b8; margin: 15px 0 10px 0;">ðŸ“Š BASIC INDICATORS</h4>
                    <div class="indicator-grid">
                        <div class="indicator-item">
                            <span class="indicator-name">RSI:</span>
                            <span class="indicator-value ${getRsiColor(tech.rsi.rsi)}">${tech.rsi.rsi.toFixed(1)}</span>
                            <span class="indicator-signal">(${tech.rsi.trend})</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-name">MACD:</span>
                            <span class="indicator-value">${tech.macd.macd.toFixed(4)}</span>
                            <span class="indicator-signal">(${tech.macd.curve_direction})</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-name">Volume:</span>
                            <span class="indicator-value">${tech.volume_analysis.ratio.toFixed(2)}x</span>
                            <span class="indicator-signal">(${tech.volume_analysis.trend})</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-name">Momentum:</span>
                            <span class="indicator-value ${getMomentumColor(tech.momentum.value)}">${tech.momentum.value.toFixed(2)}%</span>
                            <span class="indicator-signal">(${tech.momentum.trend})</span>
                        </div>
                    </div>
                </div>

                <div class="indicator-section">
                    <h4 style="color: #28a745; margin: 15px 0 10px 0;">ðŸ“ˆ LEVELS</h4>
                    <div class="levels-grid">
                        <div class="level-item">
                            <span class="level-name">Resistance:</span>
                            <span class="level-value">${tech.resistance.toFixed(4)}</span>
                            <span class="level-distance">+${(((tech.resistance - tech.current_price) / tech.current_price) * 100).toFixed(2)}%</span>
                        </div>
                        <div class="level-item">
                            <span class="level-name">Support:</span>
                            <span class="level-value">${tech.support.toFixed(4)}</span>
                            <span class="level-distance">${(((tech.support - tech.current_price) / tech.current_price) * 100).toFixed(2)}%</span>
                        </div>
                    </div>
                </div>
            `;

            document.getElementById('technicalAnalysis').innerHTML = html;
        }

        // Display pattern analysis
        function displayPatternAnalysis(data) {
            const patterns = data.pattern_analysis || {};
            const list = patterns.patterns || [];
            let html = `
                <div class=\"metric-card\" style=\"margin-bottom: 12px;\">
                    <div class=\"metric-value ${getSignalColor(patterns.overall_signal || 'neutral')}\">${patterns.overall_signal || 'NEUTRAL'}</div>
                    <div class=\"metric-label\">Overall Pattern Signal</div>
                </div>`;
            if (patterns.pattern_summary) {
                html += `<p style=\"color: var(--text-secondary); font-size:0.6rem; line-height:0.9rem; margin-bottom:10px;\">${patterns.pattern_summary}</p>`;
            }
            if (list.length === 0) {
                html += '<p style="color: rgba(255,255,255,0.5); font-size:0.65rem;">Keine Muster erkannt</p>';
            } else {
                html += '<div style="display:flex; flex-direction:column; gap:10px;">';
                list.forEach(p => {
                    html += `
                    <div class=\"pattern-item fade-in\" style=\"border-left:4px solid ${getSignalColor(p.signal)}\">
                        <div class=\"pattern-header\">
                            <span class=\"pattern-type\">${p.type || p.name}<span style=\"margin-left:6px; font-size:0.5rem; background:rgba(255,255,255,0.12); padding:3px 6px; border-radius:6px; letter-spacing:.5px;\">${p.timeframe||'1h'}</span></span>
                            <span class=\"pattern-confidence\">${p.confidence}% | Q:${p.quality_grade || '-'} ${p.quality_score ? '('+p.quality_score+')' : ''}</span>
                        </div>
                        <div style=\"font-size:0.55rem; color:var(--text-secondary); margin-bottom:4px;\">Signal: <span style=\"color:${p.signal==='bullish'?'#28a745':p.signal==='bearish'?'#dc3545':'#ffc107'}\">${p.signal}</span></div>
                        ${p.description?`<div style=\"font-size:0.55rem; color:rgba(255,255,255,0.55); line-height:0.85rem;\">${p.description}</div>`:''}
                    </div>`;
                });
                html += '</div>';
            }
            document.getElementById('patternAnalysis').innerHTML = html;
        }

        // Display AI analysis
        function displayAIAnalysis(data) {
            const ai = data.ai_analysis;
            
            const html = `
                <div class="metric-card" style="margin-bottom: 15px;">
                    <div class="metric-value ${getSignalColor(ai.signal)}">${ai.signal}</div>
                    <div class="metric-label">AI Signal</div>
                </div>
                
                <div class="metric-card" style="margin-bottom: 15px;">
                    <div class="metric-value">${ai.confidence.toFixed(1)}%</div>
                    <div class="metric-label">AI Confidence</div>
                </div>
                
                <p style="color: rgba(255,255,255,0.9); margin-bottom: 10px;">
                    <strong>AI Recommendation:</strong><br>
                    ${ai.ai_recommendation}
                </p>
                
                <small style="color: rgba(255,255,255,0.7);">
                    Model: ${ai.model_version || 'JAX-v2.0'}
                </small>
            `;

            document.getElementById('aiAnalysis').innerHTML = html;
        }

        function displayMultiTimeframe(data) {
            const mt = data.multi_timeframe || {};
            const el = document.getElementById('multiTimeframe');
            if (!mt.timeframes || !mt.timeframes.length) { el.innerHTML = '<small style="color:var(--text-dim)">No data</small>'; return; }

            // Legend / explanation tooltip
            const legend = `<div style=\"font-size:0.48rem; letter-spacing:.4px; color:var(--text-dim); margin:-2px 0 6px; line-height:.7rem;\">
                <span style=\"color:#28a745;\">Bull/Bear Scores</span> = gewichtete Summe der Signale Ã¼ber Zeitrahmen. Verteilung zeigt prozentuale HÃ¤ufigkeit von bull / neutral / bear Kategorien.
            </div>`;

            let dist = '';
            if (mt.distribution_pct) {
                dist = '<div style="display:flex; gap:6px; flex-wrap:wrap; margin:2px 0 10px;">' +
                    Object.entries(mt.distribution_pct).map(([k,v])=>`<div style=\"font-size:0.5rem; background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08); backdrop-filter:blur(3px); padding:4px 6px; border-radius:6px;\">${k}: ${v}%</div>`).join('') + '</div>';
            }
            const cons = mt.consensus || {};
            const consColor = cons.primary==='BULLISH'? '#26c281': cons.primary==='BEARISH'? '#ff4d4f':'#f5b041';
            let rows = mt.timeframes.map(t => {
                if (t.error) return `<div style='font-size:0.55rem; color:#ff4d4f; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); padding:6px 8px; border-radius:10px;'>${t.tf}: ${t.error}</div>`;
                const sigColor = t.signal?.includes('bull')?'#26c281': t.signal?.includes('bear')?'#ff4d4f':'#f5b041';
                return `<div style=\"display:grid; grid-template-columns:50px 82px 52px 1fr; gap:4px; align-items:center; font-size:0.55rem; background:linear-gradient(145deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015)); border:1px solid rgba(255,255,255,0.06); padding:6px 8px; border-radius:12px; box-shadow:0 2px 4px -1px rgba(0,0,0,0.5);\">
                        <div style=\"font-weight:600; letter-spacing:.5px; color:var(--text-secondary);\">${t.tf}</div>
                        <div style=\"color:${sigColor}; font-weight:600;\">${t.signal}</div>
                        <div style=\"opacity:.85;\">RSI ${t.rsi ?? '-'} </div>
                        <div style=\"opacity:.55; font-style:italic;\">${t.trend || ''}</div>
                    </div>`;
            }).join('');
            el.innerHTML = `
                <div style=\"font-size:0.6rem; margin-bottom:6px; font-weight:600; letter-spacing:.5px;\">Consensus: <span style=\"color:${consColor}; font-weight:700;\">${cons.primary||'-'}</span>
                    <span style=\"font-size:0.5rem; font-weight:400; color:var(--text-dim);\">(Bull ${cons.bull_score||0} / Bear ${cons.bear_score||0})</span>
                </div>
                ${legend}
                ${dist}
                <div style=\"display:flex; flex-direction:column; gap:8px;\">${rows}</div>`;
        }

                function displayMarketBias(data){
                        const bias = data.market_bias; if(!bias) return;
                        const container = document.getElementById('signalDisplay');
                        const longPct = bias.long_strength_pct||0; const shortPct = bias.short_strength_pct||0; const neutralPct = Math.max(0, 100 - longPct - shortPct);
                        const existing = document.getElementById('marketBiasBar');
                        const html = `
                                <div id=\"marketBiasBar\" style=\"margin-top:18px;\">
                                    <div style=\"font-size:0.55rem; letter-spacing:.5px; color:var(--text-dim); margin-bottom:4px;\">MARKET BIAS</div>
                                    <div style=\"height:14px; width:100%; background:rgba(255,255,255,0.1); border-radius:8px; overflow:hidden; display:flex;\">
                                        <div title=\"Long\" style=\"flex:0 0 ${longPct}%; background:#198754;\"></div>
                                        <div title=\"Neutral\" style=\"flex:0 0 ${neutralPct}%; background:linear-gradient(90deg,#6c757d,#495057);\"></div>
                                        <div title=\"Short\" style=\"flex:0 0 ${shortPct}%; background:#dc3545;\"></div>
                                    </div>
                                    <div style=\"display:flex; justify-content:space-between; font-size:0.5rem; margin-top:4px; color:var(--text-secondary);\">
                                        <span>Long ${longPct}%</span><span>Neutral ${neutralPct}%</span><span>Short ${shortPct}%</span>
                                    </div>
                                </div>`;
                        if(existing){ existing.outerHTML = html; } else { container.insertAdjacentHTML('beforeend', html); }
                }

                function displayRegimeAnalysis(data) {
                    const regime = data.regime_analysis;
                    const el = document.getElementById('regimeAnalysis');
                    if (!regime || regime.regime === 'error') {
                        el.innerHTML = `<small style="color:#dc3545">${regime?.rationale || 'Regime analysis failed'}</small>`;
                        return;
                    }
                    
                    const regimeColors = {
                        'trending': '#0d6efd',
                        'ranging': '#6f42c1', 
                        'expansion': '#fd7e14',
                        'volatility_crush': '#20c997'
                    };
                    
                    const regimeIcons = {
                        'trending': 'ðŸ“ˆ',
                        'ranging': 'â†”ï¸',
                        'expansion': 'ðŸ’¥',
                        'volatility_crush': 'ðŸ¤'
                    };
                    
                    const color = regimeColors[regime.regime] || '#6c757d';
                    const icon = regimeIcons[regime.regime] || 'â“';
                    
                    let html = `
                        <div class="metric-card" style="margin-bottom: 15px; border-left: 4px solid ${color};">
                            <div class="metric-value" style="color: ${color}">
                                ${icon} ${regime.regime.toUpperCase()}
                            </div>
                            <div class="metric-label">Market Regime (${regime.confidence}%)</div>
                        </div>
                        
                        <div style="font-size: 0.65rem; color: var(--text-secondary); margin-bottom: 10px;">
                            ${regime.rationale}
                        </div>
                    `;
                    
                    if (regime.secondary_regime) {
                        const secColor = regimeColors[regime.secondary_regime] || '#6c757d';
                        const secIcon = regimeIcons[regime.secondary_regime] || 'â“';
                        html += `
                            <div style="font-size: 0.6rem; color: ${secColor}; margin: 5px 0;">
                                Secondary: ${secIcon} ${regime.secondary_regime}
                            </div>
                        `;
                    }
                    
                    html += `
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin-top: 10px;">
                            <div style="background: rgba(255,255,255,0.05); padding: 6px 8px; border-radius: 8px; font-size: 0.55rem;">
                                <div style="color: var(--text-dim);">ATR</div>
                                <div style="color: white; font-weight: 600;">${regime.atr_percentage?.toFixed(1) || 'N/A'}%</div>
                            </div>
                            <div style="background: rgba(255,255,255,0.05); padding: 6px 8px; border-radius: 8px; font-size: 0.55rem;">
                                <div style="color: var(--text-dim);">Volatility</div>
                                <div style="color: white; font-weight: 600;">${regime.volatility_level || 'N/A'}</div>
                            </div>
                        </div>
                    `;
                    
                    if (regime.regime_scores) {
                        html += `
                            <div style="margin-top: 10px; font-size: 0.5rem;">
                                <div style="color: var(--text-dim); margin-bottom: 4px;">Regime Scores:</div>
                                <div style="display: flex; gap: 4px; flex-wrap: wrap;">
                        `;
                        Object.entries(regime.regime_scores).forEach(([key, score]) => {
                            const color = regimeColors[key] || '#6c757d';
                            html += `<span style="background: ${color}20; color: ${color}; padding: 2px 5px; border-radius: 4px;">${key}: ${score}</span>`;
                        });
                        html += `</div></div>`;
                    }
                    
                    el.innerHTML = html;
                }
                
                function displayOrderFlowAnalysis(data) {
                    const orderFlowContainer = document.getElementById('orderFlowAnalysis');
                    if (!orderFlowContainer || !data.order_flow_analysis) return;
                    
                    const flow = data.order_flow_analysis;
                    if (flow.error) {
                        orderFlowContainer.innerHTML = `<div class="alert alert-warning">âš ï¸ ${flow.error}</div>`;
                        return;
                    }
                    
                    const sentimentColors = {
                        'buy_pressure': '#28a745',
                        'sell_pressure': '#dc3545', 
                        'neutral': '#6c757d',
                        'low_liquidity': '#ffc107',
                        'unknown': '#6c757d'
                    };
                    
                    const sentimentEmojis = {
                        'buy_pressure': 'ðŸŸ¢',
                        'sell_pressure': 'ðŸ”´',
                        'neutral': 'âšª',
                        'low_liquidity': 'ðŸŸ¡',
                        'unknown': 'â“'
                    };
                    
                    const imbalancePercent = (flow.order_book_imbalance * 100).toFixed(1);
                    const deltaPercent = (flow.delta_momentum * 100).toFixed(1);
                    
                    orderFlowContainer.innerHTML = `
                        <div class="order-flow-display" style="border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:18px 18px 16px; margin:10px 0; background:linear-gradient(155deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); backdrop-filter:blur(6px); box-shadow:0 4px 18px -6px rgba(0,0,0,0.55);">
                            <h5 style="margin:0 0 14px; font-size:0.8rem; letter-spacing:.5px; font-weight:600; color:var(--text-primary);">ðŸ“Š Order Flow Analysis</h5>
                            
                            <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; margin:0 0 12px;">
                                <div class="flow-metric">
                                    <strong style="color:var(--text-secondary);">Flow Sentiment:</strong>
                                    <span style="color:${sentimentColors[flow.flow_sentiment]}; font-weight:600;">
                                        ${sentimentEmojis[flow.flow_sentiment]} ${flow.flow_sentiment.replace('_', ' ').toUpperCase()}
                                    </span>
                                </div>
                                
                                <div class="flow-metric">
                                    <strong style="color:var(--text-secondary);">Flow Strength:</strong>
                                    <span style="color:${flow.flow_strength === 'strong' ? '#28a745' : flow.flow_strength === 'moderate' ? '#ffc107' : '#6c757d'}; font-weight:600;">
                                        ${flow.flow_strength.toUpperCase()}
                                    </span>
                                </div>
                                
                                <div class="flow-metric">
                                    <strong style="color:var(--text-secondary);">Spread:</strong>
                                    <span>${flow.spread_bps || 0} bps</span>
                                </div>
                                
                                <div class="flow-metric">
                                    <strong style="color:var(--text-secondary);">Order Imbalance:</strong>
                                    <span style="color: ${flow.order_book_imbalance > 0 ? '#28a745' : flow.order_book_imbalance < 0 ? '#dc3545' : '#6c757d'};">
                                        ${imbalancePercent}%
                                    </span>
                                </div>
                                
                                <div class="flow-metric">
                                    <strong style="color:var(--text-secondary);">Delta Momentum:</strong>
                                    <span style="color: ${flow.delta_momentum > 0 ? '#28a745' : flow.delta_momentum < 0 ? '#dc3545' : '#6c757d'};">
                                        ${deltaPercent}%
                                    </span>
                                </div>
                                
                                <div class="flow-metric">
                                    <strong style="color:var(--text-secondary);">Volume POC:</strong>
                                    <span>${flow.volume_profile_poc || 'N/A'}</span>
                                </div>
                            </div>
                            
                            ${flow.liquidity_zones && flow.liquidity_zones.length > 0 ? `
                                <div class="liquidity-zones" style="margin-top:6px;">
                                    <strong style="color:var(--text-secondary); font-size:0.65rem;">Liquidity Zones:</strong>
                                    <div style="margin-top:6px; display:flex; flex-wrap:wrap; gap:6px;">
                                        ${flow.liquidity_zones.map(zone => `
                                            <span style="display:inline-flex; align-items:center; gap:4px; padding:4px 8px; border-radius:8px; font-size:0.55rem; letter-spacing:.3px; 
                                                  background:${zone.type === 'support' ? 'rgba(38,194,129,0.15)' : 'rgba(255,77,79,0.15)'}; 
                                                  border:1px solid ${zone.type === 'support' ? 'rgba(38,194,129,0.35)' : 'rgba(255,77,79,0.35)'}; 
                                                  color:${zone.type === 'support' ? '#26c281' : '#ff4d4f'}; font-weight:500;">
                                                ${zone.type.toUpperCase()} @ ${zone.level} <span style='opacity:.75;'>(${zone.strength})</span>
                                            </span>
                                        `).join('')}
                                    </div>
                                </div>
                            ` : ''}
                            
                            ${flow.analysis_note ? `
                                <div style="margin-top:10px; font-size:0.55rem; color:var(--text-secondary); font-style:italic; padding:8px 10px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:10px;">
                                    ðŸ’¡ ${flow.analysis_note}
                                </div>
                            ` : ''}
                        </div>
                    `;
                }
                
                function displayFeatureContributions(data) {
                    const featureContainer = document.getElementById('featureContributions');
                    if (!featureContainer || !data.ai_analysis?.feature_contributions) return;
                    
                    const features = data.ai_analysis.feature_contributions;
                    if (features.error) {
                        featureContainer.innerHTML = `<div class="alert alert-warning">âš ï¸ ${features.error}</div>`;
                        return;
                    }
                    
                    featureContainer.innerHTML = `
                        <div class="feature-contributions-display" style="border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:18px 18px 16px; margin:10px 0; background:linear-gradient(150deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); backdrop-filter:blur(6px); box-shadow:0 4px 18px -6px rgba(0,0,0,0.55);">
                            <h5 style="margin:0 0 14px; font-size:0.8rem; letter-spacing:.5px; font-weight:600; color:var(--text-primary);">ðŸ” AI Feature Contributions</h5>
                            
                            <div style="margin:0 0 12px;">
                                <strong style="color:var(--text-secondary);">Signal Confidence:</strong>
                                <span style="color:${features.ai_signal_confidence > 70 ? '#28a745' : features.ai_signal_confidence > 50 ? '#ffc107' : '#dc3545'}; font-weight:700;">
                                    ${features.ai_signal_confidence?.toFixed(1) || 0}%
                                </span>
                                <span style="margin-left:10px; color:var(--text-dim); font-size:0.55rem;">
                                    (${features.total_features_analyzed || 0} features analyzed)
                                </span>
                            </div>
                            
                            ${features.top_features && features.top_features.length > 0 ? `
                                <div class="top-features" style="margin:0 0 12px;">
                                    <strong style="color:var(--text-secondary);">Top Contributing Features:</strong>
                                    <div style="margin-top:6px; display:flex; flex-direction:column; gap:6px;">
                                        ${features.top_features.map(feature => `
                                            <div style="display:grid; grid-template-columns:1fr 70px 74px; align-items:center; gap:8px; padding:6px 8px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:10px; font-size:0.6rem;">
                                                <span style="font-weight:500; color:var(--text-primary);">${feature.feature}</span>
                                                <span style="text-align:center; color:${feature.impact === 'positive' ? '#26c281' : '#ff4d4f'}; font-weight:600;">${feature.impact === 'positive' ? '+' : '-'}${feature.importance}%</span>
                                                <span style="font-size:0.55rem; color:var(--text-dim);">val: ${feature.value}</span>
                                            </div>
                                        `).join('')}
                                    </div>
                                </div>
                            ` : ''}
                            
                            ${features.contextual_interpretations && features.contextual_interpretations.length > 0 ? `
                                <div class="contextual-interpretations" style="margin-top:10px;">
                                    <strong style="color:var(--text-secondary);">Key Interpretations:</strong>
                                    <ul style="margin:6px 0 0; padding-left:16px;">
                                        ${features.contextual_interpretations.map(interp => `
                                            <li style="font-size:0.55rem; color:var(--text-secondary); margin:2px 0;">${interp}</li>
                                        `).join('')}
                                    </ul>
                                </div>
                            ` : ''}
                            
                            ${features.note ? `
                                <div style="margin-top:10px; font-size:0.55rem; color:var(--text-secondary); font-style:italic; padding:8px 10px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:10px;">
                                    ðŸ’¡ ${features.note}
                                </div>
                            ` : ''}
                        </div>
                    `;
                }
                
                function displayAdaptiveRiskTargets(data) {
                    const adaptiveContainer = document.getElementById('adaptiveRiskTargets');
                    if (!adaptiveContainer || !data.adaptive_risk_targets) return;
                    
                    const risk = data.adaptive_risk_targets;
                    if (risk.error) {
                        adaptiveContainer.innerHTML = `<div class="alert alert-warning">âš ï¸ ${risk.error}</div>`;
                        return;
                    }
                    
                    const riskColors = {
                        'low': '#28a745',
                        'medium': '#ffc107',
                        'high': '#dc3545'
                    };
                    
                    const targets = risk.targets || {};
                    
                    adaptiveContainer.innerHTML = `
                        <div class="adaptive-risk-display" style="border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:18px 18px 16px; margin:10px 0; background:linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); backdrop-filter:blur(6px); box-shadow:0 4px 18px -6px rgba(0,0,0,0.55);">
                            <h5 style="margin:0 0 14px; font-size:0.8rem; letter-spacing:.5px; font-weight:600; color:var(--text-primary);">ðŸŽ¯ Adaptive Risk Management</h5>
                            
                            <!-- Risk Overview -->
                            <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:0 0 14px;">
                                <div class="risk-metric">
                                    <strong style="color:var(--text-secondary);">Risk %:</strong>
                                    <span style="color:${riskColors[risk.risk_category]}; font-weight:700;">
                                        ${risk.adaptive_risk_pct}%
                                    </span>
                                </div>
                                
                                <div class="risk-metric">
                                    <strong style="color:var(--text-secondary);">Reward Ratio:</strong>
                                    <span style="color:#0d6efd; font-weight:700;">
                                        1:${risk.adaptive_reward_ratio}
                                    </span>
                                </div>
                                
                                <div class="risk-metric">
                                    <strong style="color:var(--text-secondary);">Position Size:</strong>
                                    <span>${risk.position_size}</span>
                                </div>
                                
                                <div class="risk-metric">
                                    <strong style="color:var(--text-secondary);">Risk Amount:</strong>
                                    <span style="color:${riskColors[risk.risk_category]}; font-weight:600;">
                                        $${risk.risk_amount_usd}
                                    </span>
                                </div>
                            </div>
                            
                            <!-- Stop Loss & Targets -->
                            <div class="stop-targets" style="margin:0 0 12px;">
                                <h6 style="margin:0 0 10px; font-size:0.65rem; letter-spacing:.5px; font-weight:600; color:var(--text-secondary);">ðŸ“ Stop Loss & Targets</h6>
                                <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px;">
                                    <div style="padding:10px 8px; background:rgba(255,77,79,0.08); border:1px solid rgba(255,77,79,0.35); border-radius:12px; text-align:center;">
                                        <div style="font-size:0.55rem; letter-spacing:.5px; color:#ff4d4f; font-weight:600;">Stop Loss</div>
                                        <div style="font-weight:700; color:#ff4d4f; font-size:0.7rem;">${risk.stop_loss}</div>
                                    </div>
                                    <div style="padding:10px 8px; background:rgba(13,110,253,0.08); border:1px solid rgba(13,110,253,0.35); border-radius:12px; text-align:center;">
                                        <div style="font-size:0.55rem; letter-spacing:.5px; color:#0d6efd; font-weight:600;">Target 1</div>
                                        <div style="font-weight:700; color:#26c281; font-size:0.7rem;">${targets.target_1}</div>
                                    </div>
                                    <div style="padding:10px 8px; background:rgba(38,194,129,0.10); border:1px solid rgba(38,194,129,0.35); border-radius:12px; text-align:center;">
                                        <div style="font-size:0.55rem; letter-spacing:.5px; color:#26c281; font-weight:600;">Target 2</div>
                                        <div style="font-weight:700; color:#26c281; font-size:0.7rem;">${targets.target_2}</div>
                                    </div>
                                    <div style="padding:10px 8px; background:linear-gradient(135deg, rgba(38,194,129,0.18), rgba(13,110,253,0.12)); border:1px solid rgba(38,194,129,0.40); border-radius:12px; text-align:center;">
                                        <div style="font-size:0.55rem; letter-spacing:.5px; color:#26c281; font-weight:600;">Target 3</div>
                                        <div style="font-weight:700; color:#26c281; font-size:0.7rem;">${targets.target_3}</div>
                                    </div>
                                </div>
                            </div>
                            
                            ${risk.reasoning ? `
                                <div style="margin-top:8px; font-size:0.55rem; color:var(--text-secondary); font-style:italic; padding:10px 12px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:12px;">
                                    ðŸ’¡ ${risk.reasoning}
                                </div>
                            ` : ''}
                        </div>
                    `;
                }

        // Fetch AI status
        async function fetchAIStatus() {
            try {
                const res = await fetch('/api/ai/status');
                const j = await res.json();
                if (j.success) {
                    const d = j.data;
                    const last = d.last_train ? `<br><span style='color:var(--text-secondary)'>Last Train: ${(d.last_train.updated || '').replace('T',' ').split('.')[0]}</span>` : '';
                    document.getElementById('aiStatus').innerHTML = `Model: <span style='color:#0d6efd'>${d.model_version}</span><br>Samples: <span style='color:#8b5cf6'>${d.samples_collected}</span>${last}`;
                }
            } catch(e) {
                // silent
            }
        }
        setInterval(fetchAIStatus, 15000);
        fetchAIStatus();

        // Run backtest
        async function runBacktest() {
            if (!currentSymbol) { alert('Erst Symbol analysieren.'); return; }
            const interval = document.getElementById('btInterval').value;
            const limit = document.getElementById('btLimit').value;
            const statusEl = document.getElementById('backtestStatus');
            const resultEl = document.getElementById('backtestResults');
            statusEl.textContent = 'Running backtest...';
            resultEl.textContent = '';
            try {
                const res = await fetch(`/api/backtest/${currentSymbol}?interval=${interval}&limit=${limit}`);
                const j = await res.json();
                if (!j.success) { statusEl.textContent = 'Error: '+ j.error; return; }
                const m = j.data.metrics;
                statusEl.textContent = `${j.data.strategy} â€¢ ${j.data.candles} candles`;
                let html = `<strong>Performance</strong><br>` +
                    `Trades: ${m.total_trades} | WinRate: ${m.win_rate_pct}% | PF: ${m.profit_factor}<br>` +
                    `Avg: ${m.avg_return_pct}% | Total: ${m.total_return_pct}% | MDD: ${m.max_drawdown_pct}%<br>` +
                    `Expectancy: ${m.expectancy_pct}% | Sharpeâ‰ˆ ${m.sharpe_approx}`;
                if (j.data.trades && j.data.trades.length) {
                    const last = j.data.trades.slice(-5).map(t=>`${new Date(t.exit_time).toLocaleDateString()} ${t.return_pct}%`).join(' â€¢ ');
                    html += `<br><strong>Last Trades:</strong> ${last}`;
                }
                resultEl.innerHTML = html;
            } catch(e) {
                statusEl.textContent = 'Backtest error';
            }
        }

        // Display liquidation tables
        function displayLiquidationTables(data) {
            displayLiquidationTable('liquidationLongTable', data.liquidation_long, 'LONG');
            displayLiquidationTable('liquidationShortTable', data.liquidation_short, 'SHORT');
        }

        function displayLiquidationTable(tableId, liquidationData, type) {
            const table = document.getElementById(tableId);
            
            let html = `
                <thead>
                    <tr>
                        <th>Leverage</th>
                        <th>Liquidation Price</th>
                        <th>Distance</th>
                        <th>Risk Level</th>
                    </tr>
                </thead>
                <tbody>
            `;

            liquidationData.forEach(item => {
                html += `
                    <tr>
                        <td><strong>${item.leverage}</strong></td>
                        <td>$${item.liquidation_price.toLocaleString()}</td>
                        <td>${item.distance_percent.toFixed(2)}%</td>
                        <td style="color: ${item.risk_color}"><strong>${item.risk_level}</strong></td>
                    </tr>
                `;
            });

            html += '</tbody>';
            table.innerHTML = html;
        }

        // Helper functions
        function getRSIColor(rsi) {
            if (rsi > 70) return '#dc3545';
            if (rsi < 30) return '#28a745';
            return '#ffc107';
        }

        function getTrendColor(trend) {
            if (trend.includes('bullish')) return 'text-success';
            if (trend.includes('bearish')) return 'text-danger';
            return 'text-warning';
        }

        function getSignalColor(signal) {
            const signalStr = signal.toLowerCase();
            if (signalStr.includes('buy') || signalStr.includes('bullish')) return '#28a745';
            if (signalStr.includes('sell') || signalStr.includes('bearish')) return '#dc3545';
            return '#6c757d';
        }

        // NEW: Extended Indicator Color Functions
        function getRsiColor(rsi) {
            if (rsi > 80) return '#dc3545'; // Overbought - Red
            if (rsi < 20) return '#28a745'; // Oversold - Green
            if (rsi > 70) return '#fd7e14'; // Warning - Orange
            if (rsi < 30) return '#20c997'; // Opportunity - Teal
            return '#17a2b8'; // Neutral - Blue
        }

        function getStochasticColor(k) {
            if (k > 80) return '#dc3545'; // Overbought
            if (k < 20) return '#28a745'; // Oversold
            return '#17a2b8'; // Neutral
        }

        function getWilliamsColor(wr) {
            if (wr > -20) return '#dc3545'; // Overbought
            if (wr < -80) return '#28a745'; // Oversold
            return '#17a2b8'; // Neutral
        }

        function getCciColor(cci) {
            if (cci > 100) return '#dc3545'; // Overbought
            if (cci < -100) return '#28a745'; // Oversold
            if (Math.abs(cci) > 200) return '#6f42c1'; // Extreme
            return '#17a2b8'; // Neutral
        }

        function getVolatilityColor(volatility) {
            switch(volatility) {
                case 'very_high': return '#dc3545';
                case 'high': return '#fd7e14';
                case 'medium': return '#ffc107';
                case 'low': return '#28a745';
                default: return '#6c757d';
            }
        }

        function getTrendStrengthColor(strength) {
            switch(strength) {
                case 'very_strong': return '#dc3545';
                case 'strong': return '#fd7e14';
                case 'moderate': return '#ffc107';
                case 'weak': return '#6c757d';
                default: return '#17a2b8';
            }
        }

        function getMomentumColor(momentum) {
            if (momentum > 5) return '#28a745';
            if (momentum > 2) return '#20c997';
            if (momentum < -5) return '#dc3545';
            if (momentum < -2) return '#fd7e14';
            return '#6c757d';
        }

        function getIndicatorColor(signal) {
            if(!signal || typeof signal !== 'string') return '#6c757d';
            switch(signal.toLowerCase()) {
                case 'bullish': return '#28a745';
                case 'bearish': return '#dc3545';
                case 'overbought': return '#fd7e14';
                case 'oversold': return '#20c997';
                default: return '#6c757d';
            }
        }

        function formatVolume(volume) {
            const num = parseFloat(volume);
            if (num >= 1e9) return (num / 1e9).toFixed(1) + 'B';
            if (num >= 1e6) return (num / 1e6).toFixed(1) + 'M';
            if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
            return num.toFixed(0);
        }

        // Position Size Calculator Logic
        function prefillFromFirstSetup() {
            const data = window.__lastAnalysis;
            if(!data || !Array.isArray(data.trade_setups) || data.trade_setups.length===0) {
                document.getElementById('psResult').textContent = 'âš ï¸ Keine Setups vorhanden zum Ãœbernehmen.';
                return;
            }
            const s = data.trade_setups[0];
            document.getElementById('psEntry').value = s.entry;
            document.getElementById('psStop').value = s.stop_loss;
            calcPositionSize();
        }
        function calcPositionSize() {
            const equity = parseFloat(psEquity.value)||0;
            const riskPct = parseFloat(psRiskPct.value)||0;
            const entry = parseFloat(psEntry.value)||0;
            const stop = parseFloat(psStop.value)||0;
            const res = document.getElementById('psResult');
            if(equity<=0||riskPct<=0||entry<=0||stop<=0) { res.textContent='Bitte Werte eingeben.'; return; }
            const riskAmount = equity * (riskPct/100);
            const diff = Math.abs(entry - stop);
            if(diff <= 0) { res.textContent='Entry und Stop dÃ¼rfen nicht identisch sein.'; return; }
            const qty = riskAmount / diff;
            // Suggest capital usage (notional)
            const notional = qty * entry;
            const rr2 = entry + (diff*2);
            const rr3 = entry + (diff*3);
            res.innerHTML = `Risiko: $${riskAmount.toFixed(2)} | GrÃ¶ÃŸe: <b>${qty.toFixed(4)}</b> | Notional ca: $${notional.toFixed(2)}<br>`+
                `TP 2R: ${rr2.toFixed(2)} | TP 3R: ${rr3.toFixed(2)} | Abstand (R): 1R = ${(diff).toFixed(2)}`;
        }

        // Initialize with a default symbol & interactions
        document.addEventListener('DOMContentLoaded', function() {
            const input = document.getElementById('searchInput');
            input.value = 'BTCUSDT';
            searchSymbol();

            // Theme toggle
            const toggle = document.getElementById('themeToggle');
            if (toggle) {
                toggle.addEventListener('click', () => {
                    document.body.classList.toggle('dim');
                });
            }

            // Smooth scroll to results after first analysis
            const observer = new MutationObserver(() => {
                const results = document.getElementById('analysisResults');
                if (results && results.classList.contains('active')) {
                    results.scrollIntoView({behavior:'smooth', block:'start'});
                    observer.disconnect();
                }
            });
            observer.observe(document.body, {subtree:true, attributes:true, attributeFilter:['class']});
        });

        // Add Enter key support
        document.getElementById('searchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchSymbol();
            }
        });
    </script>
</body>
</html>
"""

print("ðŸš€ ULTIMATE TRADING SYSTEM")
print("ðŸ“Š Professional Trading Analysis")
print("âš¡ Server starting on port: 5000")
print("ðŸŒ Environment: Development")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)