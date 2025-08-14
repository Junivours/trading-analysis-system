# ========================================================================================
# ULTIMATE TRADING SYSTEM V5 - BEAUTIFUL & INTELLIGENT EDITION  
# ========================================================================================
# Professional Trading Dashboard mit intelligenter Position Management
# Basierend auf deinem schönen Backup + erweiterte Features

from flask import Flask, jsonify, render_template_string, request, send_from_directory
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
from core.binance_client import BinanceClient
from core.liquidation import LiquidationCalculator
from core.profiling import SymbolBehaviorProfiler
from core.orchestration.master_analyzer import MasterAnalyzer
from core.diagnostics import run_symbol_diagnostics
from collections import deque
import json, hashlib, logging, uuid
import math

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
# 🧾 STRUCTURED LOGGING (in-memory ring buffer + stdout)
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
# 🌐 API ROUTES
# ========================================================================================

# ========================================================================================
# 🌐 API ROUTES
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
        base_tf = (request.args.get('tf') or '1h').lower()
        # Limit to supported set
        if base_tf not in ('15m','1h','4h','1d'):
            base_tf = '1h'
        log_id = log_event('info', 'Analyze request start', symbol=symbol.upper(), refresh=request.args.get('refresh')=='1', tf=base_tf)
        analysis = master_analyzer.analyze_symbol(symbol.upper(), base_interval=base_tf)
        if request.args.get('diag') == '1':
            try:
                analysis['diagnostics'] = run_symbol_diagnostics(analysis)
            except Exception as _d:
                analysis['diagnostics'] = {'status':'error','error': str(_d)}

        # --- Fallback / Repair for position_analysis to avoid dict-float subtraction errors ---
        try:
            pa = analysis.get('position_analysis') if isinstance(analysis, dict) else None
            tech = analysis.get('technical_analysis', {}) if isinstance(analysis, dict) else {}
            support = tech.get('support')
            resistance = tech.get('resistance')
            current_price = tech.get('current_price') or analysis.get('current_price')

            def _to_num(v):
                if isinstance(v,(int,float)) and not isinstance(v,bool):
                    return float(v)
                if isinstance(v, dict):  # try common keys
                    for k in ('value','price','level'): 
                        if isinstance(v.get(k),(int,float)):
                            return float(v[k])
                return None

            support_n = _to_num(support)
            resistance_n = _to_num(resistance)
            price_n = _to_num(current_price)

            invalid = False
            if not pa or not isinstance(pa, dict):
                invalid = True
            else:
                # if subtraction would break due to dict types or missing numeric fields
                if not isinstance(pa.get('resistance_potential'), (int,float)) or not isinstance(pa.get('support_risk'), (int,float)):
                    invalid = True

            if (support is not None and support_n is None) or (resistance is not None and resistance_n is None):
                invalid = True

            if invalid:
                # Recompute simple potentials safely
                if price_n and isinstance(price_n,(int,float)):
                    if resistance_n and isinstance(resistance_n,(int,float)):
                        resistance_potential = ((resistance_n - price_n) / price_n) * 100 if resistance_n > 0 else 0
                    else:
                        resistance_potential = 0
                    if support_n and isinstance(support_n,(int,float)):
                        support_risk = ((price_n - support_n) / price_n) * 100 if support_n > 0 else 0
                    else:
                        support_risk = 0
                else:
                    resistance_potential = 0
                    support_risk = 0

                recommendations = []
                # Simple heuristic recommendations
                if resistance_potential > 5 and resistance_potential >= support_risk:
                    recommendations.append({
                        'type': 'LONG', 'action': 'WATCH / BUILD', 'reason': f'{resistance_potential:.1f}% Up Potential', 'details': 'Basic heuristic (fallback)', 'confidence': 55, 'color': '#28a745'
                    })
                if support_risk > 5 and support_risk > resistance_potential:
                    recommendations.append({
                        'type': 'SHORT', 'action': 'WATCH / BUILD', 'reason': f'{support_risk:.1f}% Down Potential', 'details': 'Basic heuristic (fallback)', 'confidence': 55, 'color': '#dc3545'
                    })
                if not recommendations:
                    recommendations.append({
                        'type': 'NEUTRAL', 'action': 'NO ACTION', 'reason': 'Kein klares Setup (Fallback)', 'details': 'Low differential', 'confidence': 40, 'color': '#6c757d'
                    })

                analysis['position_analysis'] = {
                    'resistance_potential': round(resistance_potential,2),
                    'support_risk': round(support_risk,2),
                    'recommendations': recommendations,
                    'fallback': True,
                    'error': pa.get('error') if isinstance(pa, dict) and pa.get('error') else None
                }
        except Exception as pe:
            # Last resort minimal stub
            analysis.setdefault('position_analysis', {
                'resistance_potential': 0.0,
                'support_risk': 0.0,
                'recommendations': [{
                    'type': 'NEUTRAL','action':'ERROR','reason':'PositionManager Fehler','details':str(pe),'confidence':0,'color':'#dc3545'
                }],
                'fallback': True,
                'error': str(pe)
            })

        # Optional: inject enterprise validation when requested or missing
        try:
            want_validation = request.args.get('validate') == '1'
            fs = analysis.get('final_score', {}) if isinstance(analysis, dict) else {}
            if want_validation or not isinstance(fs.get('validation'), dict):
                validation = _compute_enterprise_validation(analysis)
                analysis.setdefault('final_score', {})['validation'] = validation
        except Exception as _ve:
            # Never break analyze on validation errors; expose as soft warning in validation field
            try:
                analysis.setdefault('final_score', {})['validation'] = {
                    'trading_action': 'WAIT', 'risk_level': 'UNKNOWN', 'enterprise_ready': False,
                    'contradictions': [], 'warnings': [{'type':'validation_error','message': str(_ve), 'recommendation': 'Review data fields & ranges.'}],
                    'confidence_factors': []
                }
            except Exception:
                pass
        
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

def _safe_num(x):
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            return float(x)
        return None
    except Exception:
        return None

def _compute_enterprise_validation(analysis: dict) -> dict:
    """Sanity-check numeric ranges and detect contradictions for enterprise readiness.
    Returns a compact validation dict consumed by UI.
    """
    contradictions = []
    warnings = []
    confidence_factors = []

    # Extract key sections safely
    fs = analysis.get('final_score') or {}
    tech = analysis.get('technical_analysis') or {}
    ext = analysis.get('extended_analysis') or {}
    ai = analysis.get('ai_analysis') or {}
    mt = (analysis.get('multi_timeframe') or {}).get('consensus') or {}
    setups = analysis.get('trade_setups') or []

    # Score and weights sanity
    score = _safe_num((fs or {}).get('score'))
    tw = _safe_num((fs or {}).get('technical_weight')) or 0.0
    pw = _safe_num((fs or {}).get('pattern_weight')) or 0.0
    aw = _safe_num((fs or {}).get('ai_weight')) or 0.0
    weights_sum = tw + pw + aw
    if score is None or not (0 <= score <= 100):
        warnings.append({'type':'score_range','message':'Final score out of [0,100]','recommendation':'Clamp or review scoring pipeline.'})
    if not (0 <= weights_sum <= 100.5):  # allow tiny drift
        warnings.append({'type':'weights_sum','message':f'Weights sum {weights_sum:.1f} not in [0,100]','recommendation':'Normalize weights to 100.'})
    else:
        confidence_factors.append(f'Weights sum ok: {weights_sum:.1f}')

    # Technicals
    rsi = _safe_num(((tech.get('rsi') or {}).get('rsi')))
    if rsi is None or not (0 <= rsi <= 100):
        warnings.append({'type':'rsi_range','message':f'RSI out of range: {rsi}','recommendation':'Ensure RSI calculation returns [0,100].'})
    else:
        confidence_factors.append(f'RSI in range: {rsi:.1f}')
    support = _safe_num(tech.get('support'))
    resistance = _safe_num(tech.get('resistance'))
    price = _safe_num(analysis.get('current_price') or (tech.get('current_price') if isinstance(tech.get('current_price'), (int,float)) else None))
    if price is None or price <= 0:
        contradictions.append({'type':'price_invalid','message':'Current price missing/invalid','recommendation':'Refresh data source; enforce positive price.'})
    if support is not None and resistance is not None and support > resistance:
        warnings.append({'type':'levels_inverted','message':'Support above resistance','recommendation':'Recompute levels or check timeframe.'})

    # AI section
    ai_signal = str(ai.get('signal') or '').upper()
    ai_conf = _safe_num(ai.get('confidence'))
    if ai_conf is not None and not (0 <= ai_conf <= 100):
        warnings.append({'type':'ai_conf_range','message':f'AI confidence out of [0,100]: {ai_conf}','recommendation':'Calibrate/clip confidence.'})
    # Ensemble alignment if present
    ens = ai.get('ensemble') or {}
    if ens and isinstance(ens, dict) and ens.get('alignment') == 'conflict':
        contradictions.append({'type':'ensemble_conflict','message':'Rule vs AI conflict','recommendation':'Prefer higher reliability path or WAIT.'})

    # Directional conflicts
    fs_sig = str((fs.get('signal') or '')).upper()
    mt_primary = str(mt.get('primary') or '').upper()
    if ai_signal and fs_sig and ai_signal != fs_sig and (ai_conf or 0) >= 55:
        contradictions.append({'type':'ai_vs_final','message':f'AI {ai_signal} vs Final {fs_sig}','recommendation':'Defer to consensus or WAIT; check inputs.'})
    if mt_primary and fs_sig and ((mt_primary == 'BULLISH' and 'SELL' in fs_sig) or (mt_primary == 'BEARISH' and 'BUY' in fs_sig)):
        warnings.append({'type':'mtf_vs_final','message':f'MTF {mt_primary} vs Final {fs_sig}','recommendation':'Reduce size or require stronger confluence.'})

    # Setups sanity
    long_hi = max((s.get('confidence') or 0) for s in setups if s.get('direction') == 'LONG') if setups else 0
    short_hi = max((s.get('confidence') or 0) for s in setups if s.get('direction') == 'SHORT') if setups else 0
    if long_hi >= 60 and short_hi >= 60:
        contradictions.append({'type':'setup_conflict','message':'High-confidence LONG and SHORT present','recommendation':'Favor MTF/AI direction; limit to 1 side.'})

    # Basic SL/TP monotonicity for first few setups
    def _check_setup(s):
        direction = s.get('direction')
        entry = _safe_num(s.get('entry') or s.get('entry_price'))
        stop = _safe_num(s.get('stop_loss'))
        tps = s.get('targets') or s.get('take_profits') or []
        tp_prices = []
        for t in tps:
            tp_prices.append(_safe_num(t.get('price') or t.get('level')))
        if entry is None or stop is None:
            return 'missing_fields'
        if direction == 'LONG':
            if stop >= entry:
                return 'long_stop_not_below_entry'
            if any((tp is not None and tp <= entry) for tp in tp_prices):
                return 'long_tp_not_above_entry'
        elif direction == 'SHORT':
            if stop <= entry:
                return 'short_stop_not_above_entry'
            if any((tp is not None and tp >= entry) for tp in tp_prices):
                return 'short_tp_not_below_entry'
        return None

    for s in (setups or [])[:5]:
        err = _check_setup(s)
        if err:
            warnings.append({'type':'setup_validation', 'message': f"Setup invalid: {err}", 'recommendation':'Fix stop/targets monotonicity.'})

    # Risk assessment from ATR
    vola = ((ext.get('atr') or {}).get('volatility')) or 'unknown'
    risk_level = 'MEDIUM'
    if isinstance(vola, str):
        v = vola.lower()
        if v in ('very_high','high'):
            risk_level = 'HIGH'
        elif v in ('low',):
            risk_level = 'LOW'

    # Trading action suggestion
    action = 'TRADE'
    if contradictions:
        action = 'WAIT'
    elif risk_level == 'HIGH' and (ai_conf or 0) < 55 and (score or 0) < 60:
        action = 'WAIT'

    enterprise_ready = (action != 'WAIT') and not contradictions
    # Confidence factors (compact)
    if ai_conf is not None:
        confidence_factors.append(f'AI conf: {ai_conf:.1f}%')
    if fs_sig:
        confidence_factors.append(f'Final: {fs_sig}')
    if mt_primary:
        confidence_factors.append(f'MTF: {mt_primary}')

    return {
        'trading_action': action,
        'risk_level': risk_level,
        'enterprise_ready': bool(enterprise_ready),
        'contradictions': contradictions,
        'warnings': warnings,
        'confidence_factors': confidence_factors
    }

@app.route('/api/validate/<symbol>')
def validate_symbol(symbol):
    """Return a concise enterprise validation report for a symbol.
    Optional: ?refresh=1 to bypass cached data.
    """
    try:
        if request.args.get('refresh') == '1':
            master_analyzer.binance_client.clear_symbol_cache(symbol.upper())
            log_event('info', 'Cache cleared for validate', symbol=symbol.upper())
        log_id = log_event('info', 'Validate request start', symbol=symbol.upper(), refresh=request.args.get('refresh')=='1')
        analysis = master_analyzer.analyze_symbol(symbol.upper())
        validation = _compute_enterprise_validation(analysis)
        log_event('info', 'Validate success', symbol=symbol.upper(), parent=log_id)
        return jsonify({'success': True, 'symbol': symbol.upper(), 'validation': validation, 'log_id': log_id})
    except Exception as e:
        err_id = log_event('error', 'Validate exception', symbol=symbol.upper(), error=str(e))
        return jsonify({'success': False, 'error': str(e), 'symbol': symbol.upper(), 'log_id': err_id}), 500

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

@app.route('/static/<path:filename>')
def static_files(filename):
    base_path = os.path.join(os.path.dirname(__file__), 'static')
    return send_from_directory(base_path, filename)

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
    # Absolute fallback – never raise 500 for status endpoint
        return jsonify({'success': True, 'data': {'initialized': False, 'model_version': 'unavailable', 'error': str(e)}}), 200

@app.route('/api/ai/compare/<symbol>')
def ai_compare(symbol):
    """Run both AI backends (torch and tensorflow) against the same features to validate side-by-side.
    Optional: ?tf=1h (15m|1h|4h|1d)
    """
    try:
        tf = (request.args.get('tf') or '1h').lower()
        if tf not in ('15m','1h','4h','1d'):
            tf = '1h'
        # Reuse full pipeline to get consistent technicals/patterns/extended
        analysis = master_analyzer.analyze_symbol(symbol.upper(), base_interval=tf)
        tech = analysis.get('technical_analysis') or {}
        patterns = analysis.get('pattern_analysis') or {}
        extended = analysis.get('extended_analysis') or {}
        position = analysis.get('position_analysis') or {}
        # Prefer market_data from analysis, else fetch ticker directly
        ticker = analysis.get('market_data') or {}
        if not ticker:
            try:
                ticker = master_analyzer.binance_client.get_ticker_data(symbol.upper())
            except Exception:
                ticker = {}

        # Build features once using the neutral feature engine
        from core.ai_backends import FeatureEngineNeutral, TorchAIAdapter, TensorFlowAIAdapter, EnsembleAI
        engine = FeatureEngineNeutral()
        features = engine.prepare_advanced_features(tech, patterns, ticker, position, extended)

        # Run adapters (will neutral-fallback if framework not installed)
        torch_pred = None
        tf_pred = None
        try:
            torch_pred = TorchAIAdapter(feature_engine=engine).predict_with_uncertainty(features)
        except Exception as e:
            torch_pred = {'initialized': False, 'framework': 'torch', 'error': str(e)}
        try:
            tf_pred = TensorFlowAIAdapter(feature_engine=engine).predict_with_uncertainty(features)
        except Exception as e:
            tf_pred = {'initialized': False, 'framework': 'tensorflow', 'error': str(e)}

        # Ensemble across available members
        try:
            ens = EnsembleAI([
                TorchAIAdapter(feature_engine=engine),
                TensorFlowAIAdapter(feature_engine=engine)
            ], feature_engine=engine)
            ens_pred = ens.predict_with_uncertainty(features)
            ens_status = ens.get_status()
        except Exception as e:
            ens_pred = {'initialized': False, 'framework': 'ensemble', 'error': str(e)}
            ens_status = {'initialized': False, 'mode': 'ensemble', 'error': str(e)}

        return jsonify({
            'success': True,
            'symbol': symbol.upper(),
            'timeframe': tf,
            'data_latency': analysis.get('latency_ms') if isinstance(analysis, dict) else None,
            'torch': torch_pred,
            'tensorflow': tf_pred,
            'ensemble': ens_pred,
            'ensemble_status': ens_status,
            'master_ai': analysis.get('ai_analysis'),
        })
    except Exception as e:
        err_id = log_event('error', 'AI compare exception', symbol=symbol.upper(), error=str(e))
        return jsonify({'success': False, 'error': str(e), 'symbol': symbol.upper(), 'log_id': err_id}), 500

@app.route('/api/backtest/<symbol>')
def backtest(symbol):
    """Run a lightweight backtest on-demand (RSI mean reversion)."""
    interval = request.args.get('interval', '1h')
    limit = int(request.args.get('limit', '500'))
    fee_bps = request.args.get('fee_bps')
    slip_bps = request.args.get('slip_bps')
    try:
        if request.args.get('refresh') == '1':
            master_analyzer.binance_client.clear_symbol_cache(symbol.upper())
            log_event('info', 'Cache cleared for backtest', symbol=symbol.upper())
        log_id = log_event('info', 'Backtest start', symbol=symbol.upper(), interval=interval, limit=limit, refresh=request.args.get('refresh')=='1')
        kwargs = {}
        if fee_bps is not None: 
            try: kwargs['fee_bps'] = float(fee_bps)
            except: pass
        if slip_bps is not None:
            try: kwargs['slip_bps'] = float(slip_bps)
            except: pass
        data = master_analyzer.run_backtest(symbol, interval=interval, limit=limit, **kwargs)
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

@app.route('/api/backtest/vector/<symbol>')
def backtest_vector_scapling(symbol):
    """Run vector-candle scalp backtest on low timeframe (default 5m)."""
    interval = request.args.get('interval', '5m')
    limit = int(request.args.get('limit', '720'))
    fee_bps = request.args.get('fee_bps')
    slip_bps = request.args.get('slip_bps')
    try:
        if request.args.get('refresh') == '1':
            master_analyzer.binance_client.clear_symbol_cache(symbol.upper())
            log_event('info', 'Cache cleared for vector backtest', symbol=symbol.upper())
        log_id = log_event('info', 'Vector backtest start', symbol=symbol.upper(), interval=interval, limit=limit)
        kwargs = {}
        if fee_bps is not None:
            try: kwargs['fee_bps'] = float(fee_bps)
            except: pass
        if slip_bps is not None:
            try: kwargs['slip_bps'] = float(slip_bps)
            except: pass
        data = master_analyzer.run_vector_scalp_backtest(symbol, interval=interval, limit=limit, **kwargs)
        if 'error' in data:
            err_id = log_event('warning', 'Vector backtest insufficient / error', symbol=symbol.upper(), interval=interval, limit=limit, error=data.get('error'), have=data.get('have'), need=data.get('need'))
            return jsonify({'success': False, 'error': data['error'], 'meta': {'symbol': symbol.upper(), 'interval': interval, 'limit': limit}, 'log_id': err_id}), 400
        log_event('info', 'Vector backtest success', symbol=symbol.upper(), interval=interval, limit=limit, parent=log_id, trades=data.get('metrics',{}).get('total_trades'))
        return jsonify({'success': True, 'data': data, 'meta': {'symbol': symbol.upper(), 'interval': interval, 'limit': limit}, 'log_id': log_id})
    except Exception as e:
        err_id = log_event('error', 'Vector backtest exception', symbol=symbol.upper(), interval=interval, limit=limit, error=str(e))
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
            return jsonify({'success': False, 'error': 'Preis nicht verfügbar'}), 400
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
# BEAUTIFUL GLASSMORPHISM FRONTEND
# ========================================================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ultimate Trading System V5</title>
    <link rel="stylesheet" href="/static/css/main.css">
    <script src="/static/js/helpers.js"></script>
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

    .signal-display.long { background: radial-gradient(circle at 50% 0%, rgba(40,167,69,0.35), rgba(40,167,69,0) 65%); }
    .signal-display.short { background: radial-gradient(circle at 50% 0%, rgba(220,53,69,0.35), rgba(220,53,69,0) 65%); }

    .signal-summary {
        text-align: center;
        margin-top: 20px;
        font-size: 1rem;
        color: rgba(255, 255, 255, 0.9);
    }

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
    /* Duplicate metric styles moved to static/css/main.css */

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
                <h1>Ultimate Trading System V5</h1>
                <p>Professional Analysis • Intelligent Position Management • Pluggable AI</p>
                <div class="toolbar">
                    <button id="themeToggle" class="btn-ghost" title="Theme umschalten">🌓 Theme</button>
                    <button id="refreshBtn" class="btn-ghost" onclick="searchSymbol()" title="Neu analysieren">🔄 Refresh</button>
                    <select id="baseTfSelect" class="btn-ghost" title="Basis-Zeiteinheit" style="padding:8px 10px;">
                        <option value="15m">15m</option>
                        <option value="1h" selected>1h</option>
                        <option value="4h">4h</option>
                        <option value="1d">1d</option>
                    </select>
                </div>
            </div>
        </div>

        <!-- Search Section -->
        <div class="glass-card search-section">
            <div class="search-container">
                <input type="text" id="searchInput" class="search-input" 
                       placeholder="Enter symbol (e.g., BTC, ETH, DOGE...)" 
                       onkeypress="if(event.key==='Enter') searchSymbol()">
                <button class="search-btn" onclick="searchSymbol()">🔍 Analyze</button>
            </div>
        </div>

        <!-- Loading Animation -->
        <div id="loadingSection" class="loading">
            <div class="spinner"></div>
            <p>Analyzing with AI • Calculating Patterns • Generating Insights...</p>
        </div>

        <!-- Analysis Results -->
        <div id="analysisResults" class="analysis-results">
            <!-- Main Signal Display -->
            <div class="glass-card">
                <div id="signalDisplay" class="signal-display">
                    <!-- Signal content will be inserted here -->
                </div>
            </div>

            <!-- Diagnostics Panel -->
            <div class="glass-card" id="diagnosticsCard">
                <div class="section-title">Diagnostics <span class="tag">BETA</span></div>
                <div id="diagnosticsPanel" style="display:flex; flex-direction:column; gap:8px;"></div>
            </div>

            <!-- Key Metrics -->
            <div class="glass-card">
                <div class="section-title">Key Metrics <span class="tag">LIVE</span></div>
                <div id="metricsGrid" class="metrics-grid">
                    <!-- Metrics will be inserted here -->
                </div>
            </div>

            <!-- Trade Setups -->
            <div class="glass-card" id="tradeSetupsCard">
                <h3 style="color: white; margin-bottom: 10px; display:flex; align-items:center; gap:10px;">🛠 Trade Setups <span style="font-size:0.7rem; background:rgba(255,255,255,0.08); padding:4px 8px; border-radius:8px; letter-spacing:1px;">BETA</span></h3>
                <div id="tradeSetupFilters" style="display:flex; gap:6px; flex-wrap:wrap; margin-bottom:10px; font-size:0.6rem;">
                    <button class="btn-ghost" onclick="setTradeSetupFilter('ALL')" id="filterAll" style="padding:4px 10px;">All</button>
                    <button class="btn-ghost" onclick="setTradeSetupFilter('LONG')" id="filterLong" style="padding:4px 10px;">Long</button>
                    <button class="btn-ghost" onclick="setTradeSetupFilter('SHORT')" id="filterShort" style="padding:4px 10px;">Short</button>
                    <button class="btn-ghost" onclick="toggleMaxSetups()" id="maxSetupToggle" style="padding:4px 10px;">Max: 2</button>
                </div>
                <div id="tradeSetupsContent" class="setup-grid"></div>
                <div id="tradeSetupsStatus" style="font-size:0.75rem; color:rgba(255,255,255,0.6); margin-top:10px;"></div>
            </div>

            <!-- Position Size Calculator -->
            <div class="glass-card" id="positionSizerCard">
                <div class="section-title"><span class="icon">📐</span> Position Size Calculator <span class="tag">RISK</span></div>
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
                    <button class="btn-ghost" onclick="prefillFromFirstSetup()">⬇️ Aus Setup übernehmen</button>
                    <button class="btn-ghost" onclick="calcPositionSize()">🧮 Berechnen</button>
                </div>
                <div id="psResult" style="font-size:.7rem; color:var(--text-secondary); line-height:1.1rem;"></div>
            </div>

            <!-- Two Column Layout -->
            <div class="grid">
                <!-- Position Management -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">🎯</span> Intelligent Position Management</div>
                    <div id="positionRecommendations">
                        <!-- Position recommendations will be inserted here -->
                    </div>
                </div>

                <!-- Technical Analysis -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">📈</span> Technical Analysis</div>
                    <div id="technicalAnalysis">
                        <!-- Technical analysis will be inserted here -->
                    </div>
                </div>

                <!-- Pattern Recognition -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">🔍</span> Chart Patterns</div>
                    <div id="patternAnalysis">
                        <!-- Pattern analysis will be inserted here -->
                    </div>
                </div>

                <!-- Multi-Timeframe -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">🕒</span> Multi-Timeframe</div>
                    <div id="multiTimeframe">
                        <!-- MTF analysis -->
                    </div>
                </div>

                <!-- Market Regime -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">📉</span> Market Regime <span class="tag">BETA</span></div>
                    <div id="regimeAnalysis">
                        <!-- Regime analysis -->
                    </div>
                </div>

                <!-- Adaptive Risk & Targets -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">🎯</span> Adaptive Risk Management <span class="tag">NEW</span></div>
                    <div id="adaptiveRiskTargets">
                        <!-- Adaptive risk and targets -->
                    </div>
                </div>

                <!-- Order Flow Analysis -->
                <div class="glass-card">
                    <div class="section-title">Order Flow <span class="tag">NEW</span></div>
                    <div id="orderFlowAnalysis">
                        <!-- Order flow analysis -->
                    </div>
                </div>

                <!-- AI Analysis -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">🤖</span> AI Engine</div>
                    <div id="aiAnalysis">
                        <!-- AI analysis will be inserted here -->
                    </div>
                    <div id="aiStatus" style="margin-top:14px; font-size:0.65rem; color:var(--text-dim); line-height:1rem;">
                        <!-- AI status -->
                    </div>
                </div>

                <!-- Feature Contributions -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">🧪</span> AI Explainability <span class="tag">NEW</span></div>
                    <div id="featureContributions">
                        <!-- Feature contributions analysis -->
                    </div>
                </div>

                <!-- Backtest -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">🧪</span> Backtest <span class="tag">BETA</span></div>
                    <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;">
                        <select id="btInterval" class="search-input" style="flex:0 0 110px; padding:8px 10px; font-size:0.65rem;">
                            <option value="1h">1h</option>
                            <option value="30m">30m</option>
                            <option value="15m">15m</option>
                            <option value="4h">4h</option>
                            <option value="1d">1d</option>
                        </select>
                        <input id="btLimit" type="number" value="500" min="100" max="1000" class="search-input" style="flex:0 0 110px; padding:8px 10px; font-size:0.65rem;" />
                        <button class="btn-ghost" onclick="runBacktest()" style="font-size:0.65rem;">▶️ Run</button>
                    </div>
                    <div id="backtestStatus" style="font-size:0.65rem; color:var(--text-secondary); margin-bottom:8px;"></div>
                    <div id="backtestResults" style="font-size:0.65rem; line-height:1rem; color:var(--text-secondary);"></div>
                </div>

                <!-- Vector Scalp Backtest -->
                <div class="glass-card">
                    <div class="section-title"><span class="icon">⚡</span> Vector Scalp Backtest <span class="tag">NEW</span></div>
                    <div style="display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px;">
                        <select id="vbtInterval" class="search-input" style="flex:0 0 110px; padding:8px 10px; font-size:0.65rem;">
                            <option value="5m">5m</option>
                            <option value="3m">3m</option>
                            <option value="1m">1m</option>
                        </select>
                        <input id="vbtLimit" type="number" value="720" min="200" max="2000" class="search-input" style="flex:0 0 110px; padding:8px 10px; font-size:0.65rem;" />
                        <button class="btn-ghost" onclick="runVectorBacktest()" style="font-size:0.65rem;">▶️ Run</button>
                    </div>
                    <div id="vectorBacktestStatus" style="font-size:0.65rem; color:var(--text-secondary); margin-bottom:8px;"></div>
                    <div id="vectorBacktestResults" style="font-size:0.65rem; line-height:1rem; color:var(--text-secondary);"></div>
                </div>
            </div>

            <!-- Liquidation Calculator -->
            <div class="glass-card grid-full">
                <div class="section-title"><span class="icon">💰</span> Liquidation Calculator</div>
                <div class="grid">
                    <div>
                        <h4 style="color: #28a745; margin-bottom: 15px;">📈 LONG Positions</h4>
                        <div style="overflow-x: auto;">
                            <table id="liquidationLongTable" class="liquidation-table">
                                <!-- Long liquidation data will be inserted here -->
                            </table>
                        </div>
                    </div>
                    <div>
                        <h4 style="color: #dc3545; margin-bottom: 15px;">📉 SHORT Positions</h4>
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
        let tradeSetupFilter = 'ALL'; // ALL | LONG | SHORT
        let tradeSetupMax = 2; // default user preference: only see top 2 setups

        function setTradeSetupFilter(f) {
            tradeSetupFilter = f;
            updateTradeSetupFilterButtons();
            if (analysisData) displayTradeSetups(analysisData);
        }
        function toggleMaxSetups() {
            tradeSetupMax = tradeSetupMax === 2 ? 999 : 2;
            const btn = document.getElementById('maxSetupToggle');
            if (btn) btn.textContent = tradeSetupMax === 2 ? 'Max: 2' : 'Max: ALL';
            if (analysisData) displayTradeSetups(analysisData);
        }
        function updateTradeSetupFilterButtons(){
            ['All','Long','Short'].forEach(n=>{
                const el = document.getElementById('filter'+n);
                if(el){
                    const key = n.toUpperCase();
                    const active = (tradeSetupFilter === 'ALL' && n==='All') || tradeSetupFilter===key;
                    el.style.background = active ? 'var(--accent)' : 'rgba(255,255,255,0.05)';
                    el.style.color = active ? '#000' : 'var(--text-primary)';
                }
            });
        }

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
                const tfSel = document.getElementById('baseTfSelect');
                const tf = tfSel && tfSel.value ? tfSel.value : '1h';
                const response = await fetch(`/api/analyze/${currentSymbol}?diag=1&validate=1&tf=${tf}`);
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
            displayDiagnostics(data);
            displayLiquidationTables(data);
        }

        function displayDiagnostics(data){
            const diag = data.diagnostics;
            const root = document.getElementById('diagnosticsPanel');
            if(!root){return;}
            if(!diag){ root.innerHTML = '<div style="font-size:.55rem; color:var(--text-dim);">Keine Diagnostics Daten.</div>'; return; }
            if(diag.status==='error'){ root.innerHTML = `<div class='alert alert-danger' style='font-size:.55rem;'>Diagnostics Fehler: ${diag.error||'unknown'}</div>`; return; }
            const badgeColor = diag.readiness==='GOOD' ? '#26c281' : diag.readiness==='ATTENTION' ? '#ffc107' : '#ff4d4f';
            const findingsHtml = (diag.findings||[]).slice(0,12).map(f=>`<div style='display:flex; justify-content:space-between; gap:6px; padding:6px 8px; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.07); border-left:3px solid ${f.severity==='error'?'#ff4d4f':f.severity==='warn'?'#ffc107':'#0d6efd'}; border-radius:10px;'>
                <span style='font-size:.52rem; color:var(--text-secondary); line-height:1.1;'>${f.message}</span>
                <span style='font-size:.45rem; opacity:.55;'>${f.type}</span>
            </div>`).join('');
            const ideasHtml = (diag.ideas||[]).slice(0,6).map(i=>`<li style='font-size:.5rem; line-height:.9rem;'>${i}</li>`).join('');
            root.innerHTML = `
                <div style='display:flex; align-items:center; justify-content:space-between; margin:0 0 10px;'>
                    <h4 style='margin:0; font-size:.7rem; letter-spacing:.5px; display:flex; align-items:center; gap:6px;'>🩺 Diagnostics <span style="background:${badgeColor};color:#000;font-size:.55rem;padding:2px 8px;border-radius:12px;">${diag.readiness}</span></h4>
                    <span style='font-size:.45rem; color:var(--text-dim);'>${diag.latency_ms}ms</span>
                </div>
                <div style='display:flex; flex-direction:column; gap:6px; margin-bottom:10px;'>${findingsHtml || '<div style="font-size:.55rem; color:var(--text-dim);">Keine Findings</div>'}</div>
                <div style='margin-top:4px;'>
                    <div style='font-size:.55rem; font-weight:600; color:#0d6efd; letter-spacing:.5px; margin:0 0 4px;'>IDEEN</div>
                    <ul style='margin:0; padding-left:16px; display:flex; flex-direction:column; gap:2px;'>${ideasHtml || '<li style="font-size:.5rem; color:var(--text-dim);">Keine Ideen</li>'}</ul>
                </div>`;
        }

        // Display main trading signal
        function displayMainSignal(data) {
            const signal = data.final_score;
            const signalDisplay = document.getElementById('signalDisplay');
            const mt = data.multi_timeframe || {}; const tWanted=['15m','1h','4h','1d'];
            let rsiChips='';
            try {
                const map={}; (mt.timeframes||[]).forEach(t=>map[t.tf]=t);
                rsiChips = '<div style="display:flex; gap:6px; flex-wrap:wrap; margin-top:10px;">' + tWanted.map(tf=>{
                    const r=map[tf]?.rsi; let col='#6c757d';
                    if(typeof r==='number'){ if(r>70) col='#dc3545'; else if(r<30) col='#0d6efd'; else col='#198754'; }
                    return `<span style=\"font-size:0.5rem; background:rgba(255,255,255,0.07); padding:4px 6px; border-radius:8px; letter-spacing:.5px; color:${col}; font-weight:600;\">${tf} RSI ${r??'-'}</span>`;}).join('') + '</div>';
            } catch(e){ rsiChips=''; }
            const emotionBadges = `
                <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap;">
                    <div style="background:rgba(255,255,255,0.08); padding:4px 8px; border-radius:12px; font-size:0.5rem; letter-spacing:.5px;">
                        📊 Emotion: ${data.emotion_analysis?.overall?.emotion || 'neutral'}
                    </div>
                    <div style="background:rgba(255,255,255,0.08); padding:4px 8px; border-radius:12px; font-size:0.5rem; letter-spacing:.5px;">
                        Sentiment: ${data.sentiment_analysis?.overall?.sentiment || 'neutral'}
                    </div>
                </div>
            `;
            const baseTfBadge = `<div style="font-size:.55rem; color:var(--text-dim); margin-top:8px;">Basis-Zeiteinheit: <span style="color:#0d6efd; font-weight:700;">${data.base_interval || '1h'}</span></div>`;
            signalDisplay.innerHTML = `
                <div class="signal-value" style="color: ${signal.signal_color}">
                    ${signal.signal}
                </div>
                <div class="signal-score" style="color: white">
                    Score: ${signal.score}/100
                </div>
                <div class="signal-weights">
                    <div class="weight-item">Technical: ${signal.technical_weight}</div>
                    <div class="weight-item">🔍 Patterns: ${signal.pattern_weight}</div>
                    <div class="weight-item">🤖 AI: ${signal.ai_weight}</div>
                </div>
                ${baseTfBadge}
                ${rsiChips}
                ${emotionBadges}`;
            try { displaySignalReasons(data); } catch(e) { /* non-fatal */ }
        }

        // NEW: Decision Helper – konkrete Gründe und Wann LONG/SHORT Leitplanken
        function displaySignalReasons(data){
            const el = document.getElementById('signalDisplay');
            if(!el) return;
            const fs = data?.final_score || {};
            const mt = data?.multi_timeframe?.consensus || {};
            const tech = data?.technical_analysis || {};
            const ext = data?.extended_analysis || {};
            const ai = data?.ai_analysis || {};
            const pat = data?.pattern_analysis || {};
            const emo = data?.emotion_analysis?.overall?.emotion || 'neutral';
            const flow = data?.order_flow_analysis || {};

            const trendBull = (tech?.trend?.trend || '').toLowerCase().includes('bull');
            const trendBear = (tech?.trend?.trend || '').toLowerCase().includes('bear');
            // MACD curve_direction uses 'bullish'/'bearish' in backend
            const macdUp = (tech?.macd?.curve_direction || '').toLowerCase().includes('bull');
            const macdDown = (tech?.macd?.curve_direction || '').toLowerCase().includes('bear');
            const mtBull = (mt?.primary || '').toUpperCase()==='BULLISH';
            const mtBear = (mt?.primary || '').toUpperCase()==='BEARISH';
            const aiBuy = (ai?.signal || '').toUpperCase().includes('BUY');
            const aiSell = (ai?.signal || '').toUpperCase().includes('SELL');
            const aiConf = (typeof ai?.confidence==='number')? ai.confidence : null;
            const flowBuy = flow?.flow_sentiment==='buy_pressure';
            const flowSell = flow?.flow_sentiment==='sell_pressure';
            const vola = ext?.atr?.volatility || 'unknown';
            const rsi = tech?.rsi?.rsi;

            // Build highlight reasons
            const highlights = [];
            if(mt?.primary){ highlights.push(`MTF: ${mt.primary} (Bull ${mt.bull_score||0} / Bear ${mt.bear_score||0})`); }
            if(typeof rsi==='number'){ highlights.push(`RSI(1h): ${rsi.toFixed(1)}`); }
            if(tech?.trend?.trend){ highlights.push(`Trend: ${String(tech.trend.trend).toUpperCase()}`); }
            if(tech?.macd?.curve_direction){ highlights.push(`MACD: ${tech.macd.curve_direction}`); }
            if(pat?.overall_signal){ highlights.push(`Pattern: ${String(pat.overall_signal).toUpperCase()}`); }
            if(ai?.signal){ highlights.push(`AI: ${ai.signal} (${aiConf!=null?aiConf.toFixed(1)+'%':'-'})`); }
            if(flow?.flow_sentiment){ highlights.push(`OrderFlow: ${flow.flow_sentiment.replace('_',' ')}`); }
            if(emo){ highlights.push(`Emotion: ${emo}`); }
            if(vola){ highlights.push(`Volatility: ${vola}`); }

            // Condition checks for guidance
            const longChecks = [
                {label:'MTF bullish', ok: mtBull},
                {label:'Trend bullish', ok: trendBull},
                {label:'MACD up', ok: macdUp},
                {label:'AI BUY (≥55%)', ok: aiBuy && (aiConf==null || aiConf>=55)},
                {label:'Order Flow buy pressure', ok: !!flowBuy},
                {label:'Emotion nicht euphorisch', ok: emo!=='euphoria'}
            ];
            const shortChecks = [
                {label:'MTF bearish', ok: mtBear},
                {label:'Trend bearish', ok: trendBear},
                {label:'MACD down', ok: macdDown},
                {label:'AI SELL (≥55%)', ok: aiSell && (aiConf==null || aiConf>=55)},
                {label:'Order Flow sell pressure', ok: !!flowSell},
                {label:'Emotion nicht kapitulatorisch', ok: emo!=='capitulation'}
            ];

            const longScore = longChecks.filter(c=>c.ok).length;
            const shortScore = shortChecks.filter(c=>c.ok).length;
            const fsSig = (fs?.signal||'').toUpperCase();
            let verdict = 'Neutral';
            if(longScore>shortScore && (fsSig.includes('BUY') || mtBull)) verdict='Bevorzugt LONG';
            else if(shortScore>longScore && (fsSig.includes('SELL') || mtBear)) verdict='Bevorzugt SHORT';

            // Contradictions from validation if present
            let contradictions = [];
            try{ const cons = fs?.validation?.contradictions||[]; contradictions = cons.slice(0,3).map(c=>`${c.type}: ${c.message}`); }catch(e){}

            // Top setup per direction (for concise entry/invalidations)
            let topLong = null, topShort = null;
            try{
                const setups = Array.isArray(data?.trade_setups)? data.trade_setups : [];
                for(const s of setups){
                    if(s?.direction==='LONG' && (!topLong || (s.confidence||0)>(topLong.confidence||0))) topLong = s;
                    if(s?.direction==='SHORT' && (!topShort || (s.confidence||0)>(topShort.confidence||0))) topShort = s;
                }
            }catch(e){}

            const fmtSetup = (s)=>{
                if(!s) return '';
                const entry = s.entry ?? s.entry_price;
                const stop = s.stop_loss;
                const tf = s.timeframe || s.pattern_timeframe || '1h';
                const t1 = (s.targets && s.targets[0]?.price) || (s.take_profits && s.take_profits[0]?.level);
                return `<div style='font-size:.52rem;color:var(--text-secondary);line-height:.75rem;margin-top:6px;'>`+
                       `<div><strong>Entry:</strong> ${entry ?? '-'}  <span style='opacity:.7'>(TF ${tf})</span></div>`+
                       `<div><strong>Invalidation:</strong> Stop ${stop ?? '-'}</div>`+
                       `${t1?`<div><strong>Erstes Ziel:</strong> ${t1}</div>`:''}`+
                       `${s.rationale?`<div style='opacity:.8;'>${s.rationale}</div>`:''}`+
                       `</div>`;
            };

            const chip = (txt,color)=>`<span style="background:${color}20;border:1px solid ${color}55;color:${color};padding:3px 8px;border-radius:10px;font-size:.55rem;">${txt}</span>`;
            const checkRow = (c)=>`<div style='display:flex;justify-content:space-between;gap:8px;padding:6px 8px;border:1px solid rgba(255,255,255,0.08);border-radius:10px;'>
                <span style='font-size:.55rem;color:var(--text-secondary);'>${c.label}</span>
                <span style='font-weight:700;color:${c.ok?'#26c281':'#dc3545'}'>${c.ok?'✓':'✗'}</span>
            </div>`;

            const html = `
                <div class='decision-helper' style='margin-top:14px;padding:12px;border:1px solid rgba(255,255,255,0.1);border-radius:14px;background:linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));'>
                    <div style='display:flex;align-items:center;justify-content:space-between;gap:10px;margin-bottom:8px;'>
                        <div style='font-size:.65rem;font-weight:700;letter-spacing:.5px;color:#0d6efd;'>🧭 Decision Helper</div>
                        <div>${chip(verdict, verdict.includes('LONG')?'#26c281': verdict.includes('SHORT')?'#dc3545':'#ffc107')}</div>
                    </div>
                    <div style='display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;'>
                        ${highlights.map(h=>chip(h,'#8b5cf6')).join(' ')}
                    </div>
                    <div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;'>
                        <div style='border:1px solid rgba(38,194,129,0.25);border-radius:12px;padding:10px;background:rgba(38,194,129,0.05);'>
                            <div style='font-size:.6rem;font-weight:700;color:#26c281;margin-bottom:6px;'>Wann LONG günstig?</div>
                            <div style='display:flex;flex-direction:column;gap:6px;'>${longChecks.map(checkRow).join('')}</div>
                            ${fmtSetup(topLong)}
                        </div>
                        <div style='border:1px solid rgba(255,77,79,0.25);border-radius:12px;padding:10px;background:rgba(255,77,79,0.05);'>
                            <div style='font-size:.6rem;font-weight:700;color:#ff4d4f;margin-bottom:6px;'>Wann SHORT günstig?</div>
                            <div style='display:flex;flex-direction:column;gap:6px;'>${shortChecks.map(checkRow).join('')}</div>
                            ${fmtSetup(topShort)}
                        </div>
                    </div>
                    <div style='margin-top:8px;display:flex;flex-wrap:wrap;gap:8px;'>
                        ${chip('Schwellen: RSI LONG >55, SHORT <45','#17a2b8')}
                        ${chip('AI ≥55% stärkt Richtung','#17a2b8')}
                        ${chip(`Vola: ${String(vola).toUpperCase()} – Stops anpassen`,'#17a2b8')}
                    </div>
                    ${contradictions.length?`<div style='margin-top:8px;'>
                        <div style='font-size:.55rem;color:#ffc107;font-weight:600;margin-bottom:4px;'>Widersprüche</div>
                        <ul style='margin:0;padding-left:16px;'>${contradictions.map(c=>`<li style="font-size:.52rem;color:var(--text-secondary);">${c}</li>`).join('')}</ul>
                    </div>`:''}
                    <div style='margin-top:8px;font-size:.48rem;color:var(--text-dim);'>Hinweis: Checkmarks sind unterstützende Signale – keine Garantie. Achte auf Volatilität und Risiken.</div>
                </div>`;

            el.insertAdjacentHTML('beforeend', html);
        }

        // NEW: Enterprise Validation Display
        function displayEnterpriseValidation(data) {
            const validation = data.final_score.validation;
            const validationDiv = document.getElementById('enterpriseValidation') || createValidationDiv();
            
            let html = `
                <h3>🏢 ENTERPRISE VALIDATION</h3>
                <div class="validation-header">
                    <div class="trading-action" style="color: ${validation.trading_action === 'WAIT' ? '#dc3545' : '#28a745'}">
                        EMPFEHLUNG: ${validation.trading_action}
                    </div>
                    <div class="risk-level" style="color: ${validation.risk_level === 'LOW' ? '#28a745' : validation.risk_level === 'HIGH' ? '#dc3545' : '#ffc107'}">
                        RISIKO: ${validation.risk_level}
                    </div>
                    <div class="enterprise-ready" style="color: ${validation.enterprise_ready ? '#28a745' : '#dc3545'}">
                        ${validation.enterprise_ready ? '✅ ENTERPRISE READY' : '❌ NICHT BEREIT'}
                    </div>
                </div>
            `;

            // Contradictions (Widersprüche)
            if (validation.contradictions.length > 0) {
                html += `<div class="contradictions-section">
                    <h4 style="color: #dc3545">⚠️ WIDERSPRÜCHE GEFUNDEN</h4>`;
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
                    <h4 style="color: #ffc107">⚠️ WARNUNGEN</h4>`;
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
                <h4 style="color: #17a2b8">✅ CONFIDENCE FAKTOREN</h4>`;
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

        // Trade Setups Renderer with filter & max display
        function displayTradeSetups(data) {
            const container = document.getElementById('tradeSetupsContent');
            const status = document.getElementById('tradeSetupsStatus');
            const all = Array.isArray(data.trade_setups) ? data.trade_setups : [];
            if (all.length === 0) {
                container.innerHTML = '';
                status.textContent = 'Keine Setups generiert (Bedingungen nicht erfüllt).';
                return;
            }
            // Filter by direction
            let filtered = all.filter(s => tradeSetupFilter === 'ALL' || s.direction === tradeSetupFilter);
            // Sort: confidence desc then (risk_reward_ratio || primary_rr) desc
            filtered.sort((a,b)=> (b.confidence||0) - (a.confidence||0) || ((b.risk_reward_ratio||b.primary_rr||0) - (a.risk_reward_ratio||a.primary_rr||0)) );
            const limited = filtered.slice(0, tradeSetupMax);
            // Partition limited into pattern vs regular
            const patternTrades = limited.filter(s => s.pattern_name || s.setup_type);
            const regularTrades = limited.filter(s => !s.pattern_name && !s.setup_type);
            let html='';
            if (patternTrades.length){
                html += `<div class="trade-section">
                    <h4 style="color: #FFD700; margin-bottom: 12px; font-size: 0.75rem; display:flex; align-items:center; gap:6px;">
                        🎯 <span style="margin-left:2px;">Pattern (${patternTrades.length})</span>
                    </h4>`;
                html += patternTrades.map(s=>{
                    const confClass = s.confidence >= 70 ? '' : (s.confidence >= 55 ? 'mid' : 'low');
                    const targets = (s.targets || s.take_profits || []).map(t=>{
                        const price = t.price || t.level;
                        const label = t.label || t.level;
                        const percentage = t.percentage ? ` (${t.percentage}%)` : '';
                        const rr = t.rr ? ` ${t.rr}R` : '';
                        return `<span class="target-pill pattern-target">${label}: ${price}${percentage}${rr}</span>`;
                    }).join('');
                    return `<div class="setup-card pattern-card" style="border-left:4px solid ${s.direction==='LONG'?'#28a745':'#dc3545'};">
                        <div class="confidence-chip ${confClass}">${s.confidence}%</div>
                        <div class="setup-title">${s.direction} <span class="setup-badge pattern-badge ${s.direction==='LONG'?'long':'short'}" style="background:linear-gradient(45deg,#FFD700,#FFA500); color:#000;">${s.pattern_name || s.strategy}</span>
                            <span style="margin-left:6px; font-size:.5rem; opacity:.75; background:rgba(255,255,255,0.08); padding:2px 6px; border-radius:8px;">${s.timeframe || s.pattern_timeframe || '1h'}</span>
                        </div>
                        <div class="setup-line"><span>Entry</span><span>${s.entry_price || s.entry}</span></div>
                        <div class="setup-line"><span>Stop</span><span>${s.stop_loss}</span></div>
                        ${s.risk_percent || s.risk_reward_ratio ? `<div class="setup-line"><span>Risk%</span><span>${s.risk_percent || s.risk_reward_ratio}%</span></div>` : ''}
                        ${s.risk_reward_ratio ? `<div class="setup-line"><span>R/R</span><span style="color:#28a745;">${s.risk_reward_ratio}</span></div>`:''}
                        <div class="setup-sep"></div>
                        <div class="targets">${targets}</div>
                        ${s.rationale ? `<div style="margin-top:6px; font-size:.55rem; color:rgba(255,255,255,0.55); line-height:.75rem;">${s.rationale}</div>` : ''}
                    </div>`;
                }).join('') + '</div>';
            }
            if (regularTrades.length){
                html += `<div class="trade-section">
                    <h4 style="color: #17a2b8; margin-bottom: 12px; font-size: 0.75rem; display:flex; align-items:center; gap:6px;">
                        📈 <span style="margin-left:2px;">Technical (${regularTrades.length})</span>
                    </h4>`;
                html += regularTrades.map(s=>{
                    const confClass = s.confidence >= 70 ? '' : (s.confidence >= 55 ? 'mid' : 'low');
                    const targets = (s.targets||[]).map(t=>`<span class="target-pill">${t.label}: ${t.price}${t.rr?` (${t.rr}R)`:''}</span>`).join('');
                    return `<div class="setup-card" style="border-left:4px solid ${s.direction==='LONG'?'#28a745':'#dc3545'};">
                        <div class="confidence-chip ${confClass}">${s.confidence}%</div>
                        <div class="setup-title">${s.direction} <span class="setup-badge ${s.direction==='LONG'?'long':'short'}">${s.strategy}</span>
                            <span style="margin-left:6px; font-size:.5rem; opacity:.75; background:rgba(255,255,255,0.08); padding:2px 6px; border-radius:8px;">${s.timeframe || '1h'}</span>
                        </div>
                        <div class="setup-line"><span>Entry</span><span>${s.entry}</span></div>
                        <div class="setup-line"><span>Stop</span><span>${s.stop_loss}</span></div>
                        ${s.risk_percent ? `<div class="setup-line"><span>Risk%</span><span>${s.risk_percent}%</span></div>`:''}
                        ${s.primary_rr ? `<div class="setup-line"><span>R/R</span><span style=\"color:#28a745;\">${s.primary_rr}R</span></div>`:''}
                        <div class="setup-sep"></div>
                        <div class="targets">${targets}</div>
                        ${s.rationale ? `<div style=\"margin-top:6px; font-size:.55rem; color:rgba(255,255,255,0.55); line-height:.75rem;\">${s.rationale}</div>`:''}
                    </div>`;
                }).join('') + '</div>';
            }
            container.innerHTML = html || '<div style="font-size:.65rem; color:var(--text-dim);">Keine passenden Setups nach Filter.</div>';
            status.textContent = `Zeige ${limited.length} von ${filtered.length} (${all.length} gesamt) | Filter: ${tradeSetupFilter} | Limit: ${tradeSetupMax===2?'2':'ALLE'}`;
        }

        // Display position management recommendations
        function displayPositionManagement(data) {
            const positions = data.position_analysis || {};
            const safeNum = v => (typeof v === 'number' && isFinite(v)) ? v : 0;
            const resPot = safeNum(positions.resistance_potential).toFixed(1);
            const supRisk = safeNum(positions.support_risk).toFixed(1);
            const recommendations = Array.isArray(positions.recommendations) ? positions.recommendations : [];
            let html = '';
            if (positions.error) {
                html += `<div style="background:rgba(220,53,69,0.15); border:1px solid #dc3545; padding:10px; border-radius:8px; font-size:.65rem; margin-bottom:14px;">⚠️ PositionManager-Fehler: ${positions.error}</div>`;
            } else if (positions.fallback) {
                html += `<div style="background:rgba(255,193,7,0.12); border:1px solid #ffc107; padding:10px; border-radius:8px; font-size:.6rem; margin-bottom:14px;">Fallback Position Analysis aktiv (vereinfachte Berechnung)</div>`;
            }
            html += `
                <div class="metrics-grid" style="margin-bottom: 16px;">
                    <div class="metric-card">
                        <div class="metric-value text-success">${resPot}%</div>
                        <div class="metric-label">Resistance Potential</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value text-danger">${supRisk}%</div>
                        <div class="metric-label">Support Risk</div>
                    </div>
                </div>`;
            if (!recommendations.length) {
                html += '<div style="font-size:.6rem; color:var(--text-dim);">Keine Empfehlungen verfügbar.</div>';
            } else {
                html += recommendations.map(rec => `
                    <div class="recommendation fade-in" style="border-left-color:${rec.color || '#6c757d'}">
                        <h4 style="margin:0 0 6px; font-size:.7rem;">${rec.type}: ${rec.action}</h4>
                        <p style="margin:0 0 4px; font-size:.55rem;"><strong>Grund:</strong> ${rec.reason}</p>
                        <p style="margin:0 0 6px; font-size:.55rem; line-height:.7rem;">${rec.details || ''}</p>
                        <div class="confidence-bar" style="height:4px; margin:4px 0 6px;">
                            <div class="confidence-fill" style="width:${safeNum(rec.confidence)}%"></div>
                        </div>
                        <small style="color: rgba(255,255,255,0.6); font-size:.5rem;">Confidence: ${safeNum(rec.confidence)}%</small>
                    </div>`).join('');
            }
            document.getElementById('positionRecommendations').innerHTML = html;
        }

        // Display technical analysis (hardened with null safety)
    function displayTechnicalAnalysis(data) {
            const tech = data?.technical_analysis || {};
            const extended = data?.extended_analysis || {}; // Safety check

            // Helper utilities for safe rendering
            const safeUpper = v => {
                if (v === null || v === undefined) return '';
                try { return String(v).toUpperCase(); } catch { return ''; }
            };
            const safeFixed = (v, d=2) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '-');
            const pct = (num, den) => (typeof num === 'number' && typeof den === 'number' && den !== 0 ? (((num-den)/den)*100).toFixed(2) : '-');

            // If extended analysis missing fall back
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
                             <div class="metric-value ${getTrendColor(tech?.trend?.trend)}" style="font-size:1.1rem;">${safeUpper(tech?.trend?.trend)}</div>
                             <div class="metric-label" style="font-size:.5rem;">TREND</div>
                         </div>
                         <div class="metric-card" style="min-height:92px;">
                             <div class="metric-value" style="font-size:1.05rem;">${safeUpper((tech?.macd?.curve_direction || '').replace('_',' '))}</div>
                             <div class="metric-label" style="font-size:.5rem;">MACD-SIGNAL</div>
                         </div>
                         <div class="metric-card" style="min-height:92px;">
                             <div class="metric-value ${getIndicatorColor(extended?.stochastic?.signal)}" style="font-size:1.05rem;">${safeUpper(extended?.stochastic?.signal || 'neutral')}</div>
                             <div class="metric-label" style="font-size:.5rem;">STOCHASTISCH</div>
                         </div>
                         <div class="metric-card" style="min-height:92px;">
                             <div class="metric-value ${getVolatilityColor(extended?.atr?.volatility)}" style="font-size:1.05rem;">${safeUpper(extended?.atr?.volatility)}</div>
                             <div class="metric-label" style="font-size:.5rem;">VOLATILITÄT (ATR)</div>
                         </div>
                     </div>
                 </div>
                 <!-- Core Indicators -->
                 <div style="flex:0 0 230px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:14px 16px; backdrop-filter:blur(4px);">
                     <div style="font-size:.6rem; letter-spacing:.6px; font-weight:600; color:#17a2b8; margin-bottom:10px; display:flex; align-items:center; gap:4px;">KERNINDIKATOREN</div>
                     <div style="display:flex; flex-direction:column; gap:6px;">
                         <div style="display:flex; justify-content:space-between; font-size:.6rem;">
                             <span style="color:var(--text-secondary);">RSI:</span>
                             <span style="font-weight:600;" class="${getRsiColor(tech?.rsi?.rsi)}">${safeFixed(tech?.rsi?.rsi,1)}</span>
                             <span style="opacity:.55;">(${tech?.rsi?.trend || '-'})</span>
                         </div>
                         <div style="display:flex; justify-content:space-between; font-size:.6rem;">
                             <span style="color:var(--text-secondary);">MACD:</span>
                             <span style="font-weight:600;">${safeFixed(tech?.macd?.macd,4)}</span>
                             <span style="opacity:.55;">(${tech?.macd?.curve_direction || '-'})</span>
                         </div>
                         <div style="display:flex; justify-content:space-between; font-size:.6rem;">
                             <span style="color:var(--text-secondary);">Volumen:</span>
                             <span style="font-weight:600;">${safeFixed(tech?.volume_analysis?.ratio,2)}x</span>
                             <span style="opacity:.55;">(${tech?.volume_analysis?.trend || '-'})</span>
                         </div>
                         <div style="display:flex; justify-content:space-between; font-size:.6rem;">
                             <span style="color:var(--text-secondary);">Schwung:</span>
                             <span style="font-weight:600;" class="${getMomentumColor(tech?.momentum?.value)}">${safeFixed(tech?.momentum?.value,2)}%</span>
                             <span style="opacity:.55;">(${tech?.momentum?.trend || '-'})</span>
                         </div>
                     </div>
                 </div>
                 <!-- Advanced Indicators -->
                 <div style="flex:0 0 250px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:14px 16px;">
                     <div style="font-size:.6rem; letter-spacing:.6px; font-weight:600; color:#ffc107; margin-bottom:10px;">🔬 ERWEITERTE</div>
                     <div style="display:flex; flex-direction:column; gap:6px; font-size:.6rem;">
                         <div style="display:flex; justify-content:space-between;">
                             <span style="color:var(--text-secondary);">Bollinger:</span>
                             <span style="font-weight:600;">${safeUpper(extended?.bollinger_bands?.signal)}</span>
                             <span style="opacity:.55;">${(typeof extended?.bollinger_bands?.position === 'number' ? '('+ (extended.bollinger_bands.position*100).toFixed(0)+'%)' : '')}</span>
                         </div>
                         <div style="display:flex; justify-content:space-between;">
                             <span style="color:var(--text-secondary);">Stoch %K:</span>
                             <span style="font-weight:600;" class="${getStochasticColor(extended?.stochastic?.k)}">${safeFixed(extended?.stochastic?.k,1)}</span>
                             <span style="opacity:.55;">%D ${safeFixed(extended?.stochastic?.d,1)}</span>
                         </div>
                         <div style="display:flex; justify-content:space-between;">
                             <span style="color:var(--text-secondary);">Williams %R:</span>
                             <span style="font-weight:600;" class="${getWilliamsColor(extended?.williams_r?.value)}">${safeFixed(extended?.williams_r?.value,1)}</span>
                             <span style="opacity:.55;" class="${extended?.williams_r?.extreme ? 'extreme-signal' : ''}">${extended?.williams_r?.signal || '-'}</span>
                         </div>
                         <div style="display:flex; justify-content:space-between;">
                             <span style="color:var(--text-secondary);">CCI:</span>
                             <span style="font-weight:600;" class="${getCciColor(extended?.cci?.value)}">${safeFixed(extended?.cci?.value,1)}</span>
                             <span style="opacity:.55;" class="${extended?.cci?.extreme ? 'extreme-signal' : ''}">${extended?.cci?.signal || '-'}</span>
                         </div>
                     </div>
                 </div>
                 <!-- Volatility & Risk -->
                 <div style="flex:0 0 230px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:14px 16px;">
                     <div style="font-size:.6rem; letter-spacing:.6px; font-weight:600; color:#dc3545; margin-bottom:10px;">⚠️ VOLA & RISK</div>
                     <div style="display:flex; flex-direction:column; gap:6px; font-size:.6rem;">
                         <div style="display:flex; justify-content:space-between;">
                             <span style="color:var(--text-secondary);">ATR %:</span>
                             <span style="font-weight:600;" class="${getVolatilityColor(extended?.atr?.volatility)}">${safeFixed(extended?.atr?.percentage,2)}%</span>
                             <span style="opacity:.55; color:#dc3545;">(${extended?.atr?.risk_level || '-'})</span>
                         </div>
                         <div style="display:flex; justify-content:space-between;">
                             <span style="color:var(--text-secondary);">Trend Strength:</span>
                             <span style="font-weight:600;" class="${getTrendStrengthColor(extended?.trend_strength?.strength)}">${safeUpper(extended?.trend_strength?.strength)}</span>
                             <span style="opacity:.55; color:var(--text-dim);">(${extended?.trend_strength?.direction || '-'})</span>
                         </div>
                     </div>
                 </div>
                 <!-- Levels & Fib -->
                 <div style="flex:0 0 310px; display:flex; flex-direction:column; gap:12px;">
                     <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:14px 16px;">
                         <div style="font-size:.6rem; letter-spacing:.6px; font-weight:600; color:#28a745; margin-bottom:10px;">📈 LEVELS</div>
                         <div style="display:flex; flex-direction:column; gap:6px; font-size:.6rem;">
                             <div style="display:flex; justify-content:space-between;">
                               <span style="color:var(--text-secondary);">Resistance:</span>
                               <span style="font-weight:600;">${safeFixed(tech?.resistance,4)}</span>
                               <span style="opacity:.55; color:#dc3545;">+${(typeof tech?.resistance==='number' && typeof tech?.current_price==='number'? (((tech.resistance - tech.current_price)/tech.current_price)*100).toFixed(2):'-')}%</span>
                             </div>
                             <div style="display:flex; justify-content:space-between;">
                               <span style="color:var(--text-secondary);">Support:</span>
                               <span style="font-weight:600;">${safeFixed(tech?.support,4)}</span>
                               <span style="opacity:.55; color:#26c281;">${(typeof tech?.support==='number' && typeof tech?.current_price==='number'? (((tech.support - tech.current_price)/tech.current_price)*100).toFixed(2):'-')}%</span>
                             </div>
                             <div style="display:flex; justify-content:space-between;">
                               <span style="color:var(--text-secondary);">Pivot:</span>
                               <span style="font-weight:600;">${safeFixed(extended?.pivot_points?.pivot,4)}</span>
                               <span style="opacity:.55;">${(typeof extended?.pivot_points?.r1==='number' ? 'R1 '+extended.pivot_points.r1.toFixed(4):'')}</span>
                             </div>
                         </div>
                     </div>
                     <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:14px 16px;">
                         <div style="font-size:.6rem; letter-spacing:.6px; font-weight:600; color:#6f42c1; margin-bottom:10px;">🌐 FIBONACCI</div>
                         <div style="display:grid; grid-template-columns:repeat(2,1fr); gap:6px; font-size:.6rem;">
                             <div style="display:flex; justify-content:space-between;"><span style="color:var(--text-secondary);">23.6%:</span><span style="font-weight:600;">${safeFixed(extended?.fibonacci?.fib_236,4)}</span></div>
                             <div style="display:flex; justify-content:space-between;"><span style="color:var(--text-secondary);">38.2%:</span><span style="font-weight:600;">${safeFixed(extended?.fibonacci?.fib_382,4)}</span></div>
                             <div style="display:flex; justify-content:space-between;"><span style="color:var(--text-secondary);">50%:</span><span style="font-weight:600;">${safeFixed(extended?.fibonacci?.fib_500,4)}</span></div>
                             <div style="display:flex; justify-content:space-between;"><span style="color:var(--text-secondary);">61.8%:</span><span style="font-weight:600;">${safeFixed(extended?.fibonacci?.fib_618,4)}</span></div>
                         </div>
                     </div>
                 </div>
              </div>`;

            document.getElementById('technicalAnalysis').innerHTML = html;
        }

        // Fallback function for basic technical analysis
        function displayBasicTechnicalAnalysis(tech) {
            const safeUpper = v => { if (v===null||v===undefined) return ''; try { return String(v).toUpperCase(); } catch { return ''; } };
            const safeFixed = (v,d=2)=> (typeof v==='number' && isFinite(v)? v.toFixed(d):'-');
            const html = `
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value ${getTrendColor(tech?.trend?.trend)}">${safeUpper(tech?.trend?.trend)}</div>
                        <div class="metric-label">Trend</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${safeUpper((tech?.macd?.curve_direction || '').replace('_',' '))}</div>
                        <div class="metric-label">MACD Signal</div>
                    </div>
                </div>
                
                <div class="indicator-section">
                    <h4 style="color: #17a2b8; margin: 15px 0 10px 0;">BASIC INDICATORS</h4>
                    <div class="indicator-grid">
                        <div class="indicator-item">
                            <span class="indicator-name">RSI:</span>
                            <span class="indicator-value ${getRsiColor(tech?.rsi?.rsi)}">${safeFixed(tech?.rsi?.rsi,1)}</span>
                            <span class="indicator-signal">(${tech?.rsi?.trend || '-'})</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-name">MACD:</span>
                            <span class="indicator-value">${safeFixed(tech?.macd?.macd,4)}</span>
                            <span class="indicator-signal">(${tech?.macd?.curve_direction || '-'})</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-name">Volume:</span>
                            <span class="indicator-value">${safeFixed(tech?.volume_analysis?.ratio,2)}x</span>
                            <span class="indicator-signal">(${tech?.volume_analysis?.trend || '-'})</span>
                        </div>
                        <div class="indicator-item">
                            <span class="indicator-name">Momentum:</span>
                            <span class="indicator-value ${getMomentumColor(tech?.momentum?.value)}">${safeFixed(tech?.momentum?.value,2)}%</span>
                            <span class="indicator-signal">(${tech?.momentum?.trend || '-'})</span>
                        </div>
                    </div>
                </div>

                <div class="indicator-section">
                    <h4 style="color: #28a745; margin: 15px 0 10px 0;">📈 LEVELS</h4>
                    <div class="levels-grid">
                        <div class="level-item">
                            <span class="level-name">Resistance:</span>
                            <span class="level-value">${safeFixed(tech?.resistance,4)}</span>
                            <span class="level-distance">${(typeof tech?.resistance==='number' && typeof tech?.current_price==='number'? '+'+(((tech.resistance - tech.current_price) / tech.current_price) * 100).toFixed(2)+'%':'-')}</span>
                        </div>
                        <div class="level-item">
                            <span class="level-name">Support:</span>
                            <span class="level-value">${safeFixed(tech?.support,4)}</span>
                            <span class="level-distance">${(typeof tech?.support==='number' && typeof tech?.current_price==='number'? (((tech.support - tech.current_price) / tech.current_price) * 100).toFixed(2)+'%':'-')}</span>
                        </div>
                        <div class="level-item">
                            <span class="level-name">Pivot:</span>
                            <span class="level-value">${safeFixed(extended?.pivot_points?.pivot,4)}</span>
                            <span class="level-distance">${(typeof extended?.pivot_points?.r1==='number' ? 'R1 '+extended.pivot_points.r1.toFixed(4):'')}</span>
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
            const ens = ai && ai.ensemble ? ai.ensemble : null;
            let ensBadge = '';
            if(ens){
                const alignColor = ens.alignment==='aligned' ? '#28a745' : (ens.alignment==='conflict' ? '#dc3545' : '#ffc107');
                ensBadge = `<div style="display:flex; flex-wrap:wrap; gap:6px; margin:0 0 10px;">
                    <span style="background:rgba(255,255,255,0.08); padding:4px 8px; border-radius:12px; font-size:0.5rem; letter-spacing:.5px;">Ensemble <strong>${ens.ensemble_signal}</strong></span>
                    <span style="background:rgba(255,255,255,0.08); padding:4px 8px; border-radius:12px; font-size:0.5rem; letter-spacing:.5px;">Bull ${ens.ensemble_bullish_pct ?? '-'}%</span>
                    <span style="background:rgba(255,255,255,0.08); padding:4px 8px; border-radius:12px; font-size:0.5rem; letter-spacing:.5px;">Rule ${ens.rule_prob_bullish_pct ?? '-'}%</span>
                    <span style="background:rgba(255,255,255,0.08); padding:4px 8px; border-radius:12px; font-size:0.5rem; letter-spacing:.5px;">AI ${ens.ai_prob_bullish_pct ?? '-'}%</span>
                    <span style="background:${alignColor}; color:#000; padding:4px 8px; border-radius:12px; font-size:0.5rem; font-weight:600;">${ens.alignment}</span>
                </div>`;
            }
            const html = `
                <div class="metric-card" style="margin-bottom: 15px;">
                    <div class="metric-value ${getSignalColor(ai.signal)}">${ai.signal}</div>
                    <div class="metric-label">AI Signal</div>
                </div>
                <div class="metric-card" style="margin-bottom: 15px;">
                    <div class="metric-value">${ai.confidence.toFixed(1)}%</div>
                    <div class="metric-label">AI Confidence</div>
                </div>
                ${ensBadge}
                <p style="color: rgba(255,255,255,0.9); margin-bottom: 10px;">
                    <strong>AI Recommendation:</strong><br>
                    ${ai.ai_recommendation}
                </p>
                <small style="color: rgba(255,255,255,0.7);">
                    Model: ${ai.model_version || 'JAX-v2.0'}
                </small>`;
            document.getElementById('aiAnalysis').innerHTML = html;
        }

        function displayMultiTimeframe(data) {
            const mt = data.multi_timeframe || {};
            const el = document.getElementById('multiTimeframe');
            if (!mt.timeframes || !mt.timeframes.length) { el.innerHTML = '<small style="color:var(--text-dim)">No data</small>'; return; }

            // Legend / explanation tooltip
            const legend = `<div style=\"font-size:0.48rem; letter-spacing:.4px; color:var(--text-dim); margin:-2px 0 6px; line-height:.7rem;\">
                <span style=\"color:#28a745;\">Bull/Bear Scores</span> = gewichtete Summe der Signale über Zeitrahmen. Verteilung zeigt prozentuale Häufigkeit von bull / neutral / bear Kategorien.
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
                                    <div style=\"font-size:0.55rem; letter-spacing:.5px; color:var(--text-dim); margin-bottom:4px; line-height:.7rem;\">
                                        MARKET BIAS
                                    </div>
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
                        'trending': '',
                        'ranging': '',
                        'expansion': '',
                        'volatility_crush': ''
                    };
                    
                    const color = regimeColors[regime.regime] || '#6c757d';
                    const icon = regimeIcons[regime.regime] || '❓';
                    
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
                        const secIcon = regimeIcons[regime.secondary_regime] || '❓';
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
                                <div style="color: white; font-weight: 600;">${regime.atr_percentage?.toFixed?regime.atr_percentage.toFixed(1):regime.atr_percentage||0}%</div>
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
                        orderFlowContainer.innerHTML = `<div class="alert alert-warning">⚠️ ${flow.error}</div>`;
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
                        'buy_pressure': '🟢',
                        'sell_pressure': '🔴',
                        'neutral': '⚖️',
                        'low_liquidity': '🟡',
                        'unknown': '❓'
                    };
                    
                    const imbalancePercent = (flow.order_book_imbalance * 100).toFixed(1);
                    const deltaPercent = (flow.delta_momentum * 100).toFixed(1);
                    
                    orderFlowContainer.innerHTML = `
                        <div class="order-flow-display" style="border:1px solid rgba(255,255,255,0.08); border-radius:16px; padding:18px 18px 16px; margin:10px 0; background:linear-gradient(155deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)); backdrop-filter:blur(6px); box-shadow:0 4px 18px -6px rgba(0,0,0,0.55);">
                            <h5 style="margin:0 0 14px; font-size:0.8rem; letter-spacing:.5px; font-weight:600; color:var(--text-primary);">Order Flow Analysis</h5>
                            
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
                                <div style="margin-top:10px; font-size:0.55rem; color:var(--text-secondary); font-style:italic; padding:10px 12px; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:10px;">
                                    💡 ${flow.analysis_note}
                                </div>
                            ` : ''}
                        </div>
                    `;
                }
                
                function displayFeatureContributions(data) {
                    const featureContainer = document.getElementById('featureContributions');
                    if (!featureContainer || !data.ai_analysis?.feature_contributions) return;
                    const features = data.ai_analysis.feature_contributions;
                    if (features.error) { featureContainer.innerHTML = `<div class='alert alert-warning'>⚠️ ${features.error}</div>`; return; }
                    const explainMeta = data.ai_explainability_meta;
                    let serverExplanation = '';
                    if (explainMeta && !explainMeta.error) {
                        const grp = (title, arr, color) => (Array.isArray(arr)&&arr.length)?`<div style='margin-top:8px;'><div style='font-size:0.58rem;font-weight:600;color:${color};margin:0 0 4px;'>${title}</div><ul style='margin:0;padding-left:16px;display:flex;flex-direction:column;gap:3px;'>${arr.map(r=>`<li style="font-size:0.53rem;line-height:1.05rem;color:var(--text-secondary);">${r}</li>`).join('')}</ul></div>`:'';
                        let debugBlock='';
                        if (explainMeta.debug_factors && Object.keys(explainMeta.debug_factors).length){
                            const rows = Object.entries(explainMeta.debug_factors).map(([k,v])=>`<div style='display:flex;justify-content:space-between;gap:8px;'><span style='color:var(--text-dim);'>${k}</span><span style='color:#8b5cf6;'>${v==null?'-':v}</span></div>`).join('');
                            debugBlock = `<div id='debugFactorsBlock' style='display:none;margin-top:10px;padding:8px 10px;border:1px dashed rgba(255,255,255,0.15);border-radius:10px;background:rgba(255,255,255,0.03);'><div style='font-size:0.55rem;font-weight:600;margin:0 0 6px;color:#8b5cf6;'>Debug Faktoren</div><div style='display:flex;flex-direction:column;gap:4px;font-size:0.5rem;'>${rows}</div></div>`;
                        }
                        serverExplanation = `<div id='serverExplainBox' style='margin-top:12px;padding:10px 12px;border:1px solid rgba(255,255,255,0.08);border-radius:12px;background:linear-gradient(135deg, rgba(13,110,253,0.10), rgba(255,255,255,0.02));'>
                            <div style='display:flex;align-items:center;gap:6px;margin:0 0 4px;font-size:0.65rem;font-weight:600;color:#0d6efd;'>🤖 KI Erklärbarkeit (Server)</div>
                            <div style='font-size:0.5rem;color:var(--text-dim);'>Signal: <span style='color:#fff;'>${explainMeta.signal}</span> • Conf ${explainMeta.confidence?.toFixed?explainMeta.confidence.toFixed(1):explainMeta.confidence}% • Rel ${explainMeta.reliability?.toFixed?explainMeta.reliability.toFixed(1):explainMeta.reliability}%</div>
                            ${grp('Widersprechende Faktoren', explainMeta.reasons_negative,'#ff4d4f')}
                            ${grp('Unterstützende Faktoren', explainMeta.reasons_positive,'#26c281')}
                            ${grp('Neutrale Kontextfaktoren', explainMeta.reasons_neutral,'#ffc107')}
                            ${debugBlock}
                            <div style='margin-top:6px;font-size:0.45rem;color:var(--text-dim);'>Serverseitige Meta erklärt das Modell-Rationale.</div>
                        </div>`;
                    }
                    let fallbackHeuristic='';
                    if(!serverExplanation){
                        try{ const aiSignal=data.ai_analysis?.signal; const rsiVal=(data.technical_analysis?.rsi?.rsi)||50; if(['SELL','STRONG_SELL'].includes(aiSignal)&& rsiVal<40){ fallbackHeuristic = `<div style='margin-top:10px;font-size:0.5rem;color:var(--text-dim);'>Heuristik: KI SELL trotz niedriger RSI – strukturelle Schwäche.</div>`;} }catch(e){}
                    }
                    const featRows = (features.top_features||[]).map(f=>{ const barColor = f.impact==='positive'? 'linear-gradient(90deg, rgba(38,194,129,0.55), rgba(38,194,129,0.10))':'linear-gradient(90deg, rgba(255,77,79,0.55), rgba(255,77,79,0.10))'; const imp=f.importance||0; return `<div class='feature-bar' title='${f.feature}'> <div style="position:absolute;left:0;top:0;bottom:0;width:${Math.min(100,imp)}%;background:${barColor};opacity:.55;"></div><span style='position:relative;font-weight:500;'>${f.feature}</span><span style='text-align:center;color:${f.impact==='positive'?'#26c281':'#ff4d4f'};font-weight:600;position:relative;'>${f.impact==='positive'?'+':'-'}${imp}%</span><span style='font-size:0.55rem;color:var(--text-dim);position:relative;'>val: ${f.value}</span></div>`; }).join('');
                    featureContainer.innerHTML = `<div class='feature-contributions-display' style='border:1px solid rgba(255,255,255,0.08);border-radius:16px;padding:18px 18px 16px;margin:10px 0;background:linear-gradient(150deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));'>
                        <h5 style='margin:0 0 14px;font-size:0.8rem;letter-spacing:.5px;font-weight:600;'>🔍 AI Feature Contributions</h5>
                        <div style='margin:0 0 12px;'><strong style='color:var(--text-secondary);'>Signal Confidence:</strong> <span style='color:${features.ai_signal_confidence>70?'#28a745':features.ai_signal_confidence>50?'#ffc107':'#dc3545'};font-weight:700;'>${features.ai_signal_confidence?.toFixed?features.ai_signal_confidence.toFixed(1):features.ai_signal_confidence||0}%</span> <span style='margin-left:10px;color:var(--text-dim);font-size:0.55rem;'>(${features.total_features_analyzed||0} features)</span></div>
                        ${featRows?`<div><div style='margin-bottom:6px;display:flex;gap:8px;flex-wrap:wrap;'><button id='exportFeatJsonBtn' class='btn-ghost' style='font-size:0.55rem;padding:5px 10px;'>Export JSON</button><button id='toggleExplainBtn' class='btn-ghost' style='font-size:0.55rem;padding:5px 10px;'>Meta ein/aus</button>${(explainMeta && explainMeta.debug_factors)?"<button id='toggleDebugBtn' class='btn-ghost' style='font-size:0.55rem;padding:5px 10px;'>Debug</button>":''}</div><div style='display:flex;flex-direction:column;gap:6px;'>${featRows}</div></div>`:''}
                        ${(features.contextual_interpretations||[]).length?`<div style='margin-top:10px;'><strong style='color:var(--text-secondary);'>Key Interpretations:</strong><ul style='margin:6px 0 0;padding-left:16px;'>${features.contextual_interpretations.map(i=>`<li style="font-size:0.55rem;color:var(--text-secondary);margin:2px 0;">${i}</li>`).join('')}</ul></div>`:''}
                        ${features.note?`<div style='margin-top:10px;font-size:0.55rem;color:var(--text-secondary);font-style:italic;'>💡 ${features.note}</div>`:''}
                        ${serverExplanation || fallbackHeuristic}
                    </div>`;
                    // Events
                    try{ const t=document.getElementById('toggleExplainBtn'); if(t){ t.addEventListener('click',()=>{ const box=document.getElementById('serverExplainBox'); if(box) box.style.display = box.style.display==='none'? '' : 'none'; }); }
                        const dbg=document.getElementById('toggleDebugBtn'); if(dbg){ dbg.addEventListener('click',()=>{ const d=document.getElementById('debugFactorsBlock'); if(d) d.style.display = d.style.display==='none'? 'block':'none'; }); }
                        const exp=document.getElementById('exportFeatJsonBtn'); if(exp){ exp.addEventListener('click',()=>{ try { const payload={ timestamp:new Date().toISOString(), symbol:data.symbol, ai_signal:data.ai_analysis?.signal, ai_confidence:data.ai_analysis?.confidence, reliability:data.ai_analysis?.reliability_score, top_features:features.top_features, interpretations:features.contextual_interpretations, explainability_meta: explainMeta||null }; const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download=`ai_feature_attribution_${data.symbol||'symbol'}_${Date.now()}.json`; document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url);}catch(e){ console.warn('Export failed',e);} }); }
                    }catch(e){}
                }
                
                function displayAdaptiveRiskTargets(data) {
                    const adaptiveContainer = document.getElementById('adaptiveRiskTargets');
                    if (!adaptiveContainer || !data.adaptive_risk_targets) return;
                    
                    const risk = data.adaptive_risk_targets;
                    if (risk.error) {
                        adaptiveContainer.innerHTML = `<div class="alert alert-warning">⚠️ ${risk.error}</div>`;
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
                            <h5 style="margin:0 0 14px; font-size:0.8rem; letter-spacing:.5px; font-weight:600; color:var(--text-primary);">🎯 Adaptive Risk Management</h5>
                            
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
                                <h6 style="margin:0 0 10px; font-size:0.65rem; letter-spacing:.5px; font-weight:600; color:var(--text-secondary);">📍 Stop Loss & Targets</h6>
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
                                    💡 ${risk.reasoning}
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
                statusEl.textContent = `${j.data.strategy} • ${j.data.candles} candles`;
                let html = `<strong>Performance</strong><br>` +
                    `Trades: ${m.total_trades} | WinRate: ${m.win_rate_pct}% | PF: ${m.profit_factor}<br>` +
                    `Avg: ${m.avg_return_pct}% | Total: ${m.total_return_pct}% | MDD: ${m.max_drawdown_pct}%<br>` +
                    `Expectancy: ${m.expectancy_pct}% | Sharpe≈ ${m.sharpe_approx}`;
                if (m.fee_bps !== undefined) {
                    html += `<br><span style='color:var(--text-dim)'>Fees: ${m.fee_bps} bps • Slippage: ${m.slip_bps} bps</span>`;
                }
                if (j.data.trades && j.data.trades.length) {
                    const last = j.data.trades.slice(-5).map(t=>`${new Date(t.exit_time).toLocaleDateString()} ${t.return_pct}%`).join(' • ');
                    html += `<br><strong>Last Trades:</strong> ${last}`;
                }
                resultEl.innerHTML = html;
            } catch(e) {
                statusEl.textContent = 'Backtest error';
            }
        }

        // Run vector scalp backtest
        async function runVectorBacktest() {
            if (!currentSymbol) { alert('Erst Symbol analysieren.'); return; }
            const interval = document.getElementById('vbtInterval').value;
            const limit = document.getElementById('vbtLimit').value;
            const statusEl = document.getElementById('vectorBacktestStatus');
            const resultEl = document.getElementById('vectorBacktestResults');
            statusEl.textContent = 'Running vector backtest...';
            resultEl.textContent = '';
            try {
                const res = await fetch(`/api/backtest/vector/${currentSymbol}?interval=${interval}&limit=${limit}`);
                const j = await res.json();
                if (!j.success) { statusEl.textContent = 'Error: '+ j.error; return; }
                const m = j.data.metrics || {};
                statusEl.textContent = `${j.data.strategy} • ${j.meta.interval} • candles: ${j.meta.limit}`;
                let html = `Trades: ${m.total_trades||0} | WinRate: ${m.win_rate_pct||0}% | PF(R): ${m.profit_factor_r}` +
                           `<br>Avg RR (gross): ${m.avg_rr} | ΣR (gross): ${m.equity_sum_r} | Max DD (R): ${m.max_drawdown_r}` +
                           `<br>Avg RR (net): ${m.avg_rr_net} | ΣR (net): ${m.equity_sum_r_net} | Max DD (R) net: ${m.max_drawdown_r_net}`;
                if (m.fee_bps !== undefined) {
                    html += `<br><span style='color:var(--text-dim)'>Fees: ${m.fee_bps} bps • Slippage: ${m.slip_bps} bps</span>`;
                }
                if (j.data.trades && j.data.trades.length) {
                    const last = j.data.trades.slice(-5).map(t=>`${new Date(t.exit_time).toLocaleString()} • ${t.direction} • ${t.rr}R`).join('<br>');
                    html += `<br><strong>Last Trades</strong><br>${last}`;
                } else if (j.data.note) {
                    html += `<br>${j.data.note}`;
                }
                resultEl.innerHTML = html;
            } catch(e) {
                statusEl.textContent = 'Vector backtest error';
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
                document.getElementById('psResult').textContent = '⚠️ Keine Setups vorhanden zum Übernehmen.';
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
            if(diff <= 0) { res.textContent='Entry und Stop dürfen nicht identisch sein.'; return; }
            const qty = riskAmount / diff;
            // Suggest capital usage (notional)
            const notional = qty * entry;
            const rr2 = entry + (diff*2);
            const rr3 = entry + (diff*3);
            res.innerHTML = `Risiko: $${riskAmount.toFixed(2)} | Größe: <b>${qty.toFixed(4)}</b> | Notional ca: $${notional.toFixed(2)}<br>`+
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

print("ULTIMATE TRADING SYSTEM")
print("Professional Trading Analysis")
print("⚡ Server starting on port: 5000")
print("🌍 Environment: Development")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)