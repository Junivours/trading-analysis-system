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
from collections import deque
import json, hashlib, logging, uuid

app = Flask(__name__)

## Duplicate AdvancedPatternDetector removed (using core.patterns.AdvancedPatternDetector)

# Initialize globals
position_manager = PositionManager()
advanced_ai = AdvancedJAXAI()
pattern_detector = AdvancedPatternDetector()

# ========================================================================================
# ðŸ“ˆ ENHANCED TECHNICAL ANALYSIS WITH CURVE DETECTION
# ========================================================================================

## Duplicate TechnicalAnalysis removed (now imported from core.technical_analysis)

# ========================================================================================
# ðŸ“Š ERWEITERTE TECHNISCHE ANALYSE - ENTERPRISE LEVEL
# ========================================================================================

## (Advanced technical analysis methods moved to core.advanced_technical.AdvancedTechnicalAnalysis)

# ========================================================================================
# ðŸ”— ENHANCED BINANCE CLIENT WITH SYMBOL SEARCH
# ========================================================================================


# ========================================================================================
# ðŸ’° ENHANCED LIQUIDATION CALCULATOR
# ========================================================================================


# ========================================================================================
# ðŸŽ¯ MASTER ANALYZER - ORCHESTRATING ALL SYSTEMS
# ========================================================================================

    # (Removed mis-indented duplicate SymbolBehaviorProfiler definition)

# ========================================================================================
# 🧬 SYMBOL BEHAVIOR PROFILER (per-Symbol Muster & Volatilitätsprofil)
# ========================================================================================

class MasterAnalyzer:
    def __init__(self):
        self.technical_analysis = TechnicalAnalysis()
        self.pattern_detector = AdvancedPatternDetector()
        self.position_manager = PositionManager()
        self.liquidation_calc = LiquidationCalculator()
        self.binance_client = BinanceClient()
        self.ai_system = AdvancedJAXAI()
        # NEW: symbol behavior profiler
        self.symbol_profiler = SymbolBehaviorProfiler()
        
        # Weighting configuration (70% Technical, 20% Patterns, 10% AI)
        self.weights = {
            'technical': 0.70,
            'patterns': 0.20,
            'ai': 0.10
        }

    def run_backtest(self, symbol, interval='1h', limit=500):
        """Lightweight RSI mean-reversion backtest.
        Strategy: Enter long when RSI < 30, exit when RSI > 55 or stop at entry - 2*ATR.
        Capital model: 100% capital per trade (simplified). Educational only.
        Returns metrics, trade list, equity curve (trimmed) and strategy meta.
        """
        try:
            # Normalize params
            interval = (interval or '1h').lower()
            try:
                limit = int(limit)
            except Exception:
                limit = 500
            limit = max(100, min(limit, 1000))  # ensure enough data but cap for performance

            # Determine minimum required candles (adaptive: at least 120, aim for 240 for longer intervals)
            min_required = 120
            if interval in ('4h','1d'):
                min_required = 150
            if limit < min_required:
                # Auto-raise limit silently to improve user experience
                limit = min_required

            # Use existing TA candle fetcher (pass interval & limit) else fallback to direct klines
            klines = self.technical_analysis.get_candle_data(symbol, limit=limit, interval=interval)
            if not klines or len(klines) < min_required:
                print("âš ï¸ Fallback to direct klines fetch for backtest (insufficient or empty from TA layer)")
                url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
                r = requests.get(url, timeout=10)
                klines = r.json()

            if not isinstance(klines, list) or len(klines) < min_required:
                have = len(klines) if isinstance(klines, list) else 0
                return {'error': f'Not enough historical data: have {have}, need >= {min_required}', 'have': have, 'need': min_required, 'interval': interval, 'suggestion': 'Increase limit or choose higher timeframe'}

            # Normalize structure to list of dicts with open, high, low, close, time
            if isinstance(klines[0], list):
                candles = [{
                    'time': k[0], 'open': float(k[1]), 'high': float(k[2]),
                    'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])
                } for k in klines]
            else:
                candles = klines

            closes = np.array([c['close'] for c in candles], dtype=float)
            highs = np.array([c['high'] for c in candles], dtype=float)
            lows = np.array([c['low'] for c in candles], dtype=float)
            times = [c['time'] for c in candles]

            period = 14
            if len(closes) <= period + 5:
                return {'error': 'Series too short'}

            # RSI
            delta = np.diff(closes)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            def rma(x, n):
                a = np.zeros_like(x)
                a[n-1] = np.mean(x[:n])
                for i in range(n, len(x)):
                    a[i] = (a[i-1]*(n-1) + x[i]) / n
                return a
            avg_gain = rma(gain, period)
            avg_loss = rma(loss, period)
            rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
            rsi_vals = 100 - (100 / (1 + rs))
            rsi_vals = np.concatenate([[50]*(len(closes)-1-len(rsi_vals)), rsi_vals])  # align length-1 diff
            rsi_vals = np.concatenate([[50], rsi_vals])  # pad to closes length

            # ATR
            tr = np.maximum(highs[1:], closes[:-1]) - np.minimum(lows[1:], closes[:-1])
            atr_vals = rma(tr, period)
            atr_vals = np.concatenate([[atr_vals[period-1]]*(len(closes)-1-len(atr_vals)), atr_vals])
            atr_vals = np.concatenate([[atr_vals[0]], atr_vals])

            equity = 1000.0
            base_equity = equity
            position = None
            trades = []
            equity_curve = []
            max_equity = equity
            max_dd = 0

            for i in range(period+5, len(closes)):
                price = closes[i]
                equity_curve.append({'t': times[i], 'equity': equity})
                if position is None:
                    if rsi_vals[i] < 30:
                        position = {
                            'entry_price': price,
                            'entry_time': times[i],
                            'stop': price - 2 * atr_vals[i] if atr_vals[i] > 0 else price * 0.98
                        }
                else:
                    stop_hit = price <= position['stop']
                    take = rsi_vals[i] > 55
                    if stop_hit or take:
                        ret_pct = (price - position['entry_price']) / position['entry_price'] * 100
                        equity *= (1 + ret_pct/100)
                        trades.append({
                            'entry_time': position['entry_time'],
                            'exit_time': times[i],
                            'entry': round(position['entry_price'],6),
                            'exit': round(price,6),
                            'return_pct': round(ret_pct,2),
                            'outcome': 'win' if ret_pct > 0 else 'loss'
                        })
                        position = None
                if equity > max_equity:
                    max_equity = equity
                dd = (max_equity - equity) / max_equity
                if dd > max_dd:
                    max_dd = dd

            wins = sum(1 for t in trades if t['outcome']=='win')
            losses = sum(1 for t in trades if t['outcome']=='loss')
            total_trades = len(trades)
            win_rate = (wins/total_trades*100) if total_trades else 0
            avg_ret = np.mean([t['return_pct'] for t in trades]) if trades else 0
            total_ret = (equity/base_equity - 1)*100
            returns = np.array([t['return_pct']/100 for t in trades]) if trades else np.array([])
            sharpe = 0
            if len(returns) > 1:
                sharpe = np.mean(returns)/(np.std(returns)+1e-9)
            profits = [t['return_pct'] for t in trades if t['return_pct']>0]
            losses_list = [-t['return_pct'] for t in trades if t['return_pct']<0]
            profit_factor = (sum(profits)/sum(losses_list)) if losses_list else float('inf') if profits else 0
            expectancy = (win_rate/100)*(np.mean(profits) if profits else 0) - (1-win_rate/100)*(np.mean(losses_list) if losses_list else 0)

            # --- Additional advanced metrics ---
            buy_hold_return = (closes[-1]/closes[0]-1)*100 if len(closes) > 1 else 0
            relative_outperformance = total_ret - buy_hold_return
            # Exposure: proportion of bars where a position was open
            exposure_bars = sum(1 for t in equity_curve if True)  # placeholder full length
            # We didn't store per-bar position state; approximate exposure via average holding duration * trades
            # Track holding durations by reconstructing from trades (bars_held captured below if we enhance loop)
            max_consec_wins = 0
            max_consec_losses = 0
            cur_wins = 0
            cur_losses = 0
            for t in trades:
                if t['outcome'] == 'win':
                    cur_wins += 1
                    cur_losses = 0
                else:
                    cur_losses += 1
                    cur_wins = 0
                max_consec_wins = max(max_consec_wins, cur_wins)
                max_consec_losses = max(max_consec_losses, cur_losses)
            avg_win = np.mean(profits) if profits else 0
            avg_loss = np.mean(losses_list) if losses_list else 0
            win_loss_ratio = (avg_win/avg_loss) if avg_loss else float('inf') if avg_win>0 else 0
            risk_adjusted_return = round(total_ret/(max_dd*100+1e-9),2) if max_dd>0 else 'INF'

            return {
                'symbol': symbol.upper(),
                'interval': interval,
                'candles': len(closes),
                'strategy': 'RSI Mean Reversion V1',
                'metrics': {
                    'total_trades': total_trades,
                    'wins': wins,
                    'losses': losses,
                    'win_rate_pct': round(win_rate,2),
                    'avg_return_pct': round(avg_ret,2),
                    'total_return_pct': round(total_ret,2),
                    'max_drawdown_pct': round(max_dd*100,2),
                    'profit_factor': round(profit_factor,2) if profit_factor != float('inf') else 'INF',
                    'expectancy_pct': round(expectancy,2),
                    'sharpe_approx': round(sharpe,2),
                    'buy_hold_return_pct': round(buy_hold_return,2),
                    'alpha_vs_buy_hold_pct': round(relative_outperformance,2),
                    'avg_win_pct': round(avg_win,2),
                    'avg_loss_pct': round(avg_loss,2),
                    'win_loss_ratio': round(win_loss_ratio,2) if win_loss_ratio != float('inf') else 'INF',
                    'max_consecutive_wins': max_consec_wins,
                    'max_consecutive_losses': max_consec_losses,
                    'risk_adjusted_return_ratio': risk_adjusted_return
                },
                'trades': trades[-120:],
                'equity_curve': equity_curve[-250:],
                'disclaimer': 'Backtest ist vereinfacht. Vergangene Performance garantiert keine Zukunft.'
            }
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_symbol(self, symbol):
        """Complete analysis of a trading symbol"""
        try:
            print(f"ðŸ” Starting analysis for {symbol}")
            phase_t0 = time.time()
            timings = {}
            
            # Get market data
            t_phase = time.time()
            ticker_data = self.binance_client.get_ticker_data(symbol)
            current_price = float(ticker_data.get('lastPrice', 0))
            timings['market_data_ms'] = round((time.time()-t_phase)*1000,2)
            print(f"âœ… Got price: {current_price}")
            
            if current_price == 0:
                return {'error': 'Symbol not found or no price data available'}
            
            # Get candlestick data
            t_phase = time.time()
            candles = self.technical_analysis.get_candle_data(symbol, interval='1h')
            timings['candles_fetch_ms'] = round((time.time()-t_phase)*1000,2)
            if not candles:
                return {'error': 'Unable to fetch candlestick data'}
            print(f"âœ… Got {len(candles)} candles")
            
            # Technical Analysis (70% weight) - BASIC ONLY FOR NOW
            print("ðŸ” Starting technical analysis...")
            t_phase = time.time()
            tech_analysis = self.technical_analysis.calculate_advanced_indicators(candles)
            timings['technical_ms'] = round((time.time()-t_phase)*1000,2)
            print("âœ… Technical analysis complete")
            
            # Extended Technical Analysis (Enterprise Level) - Temporarily with error handling
            try:
                t_phase = time.time()
                extended_analysis = AdvancedTechnicalAnalysis.calculate_extended_indicators(candles)
                timings['extended_ms'] = round((time.time()-t_phase)*1000,2)
                print("âœ… Extended analysis successful")
            except Exception as e:
                print(f"âŒ Extended analysis error: {e}")
                extended_analysis = {
                    'bollinger_bands': {'signal': 'neutral', 'position': 0.5},
                    'stochastic': {'k': 50, 'd': 50, 'signal': 'neutral'},
                    'williams_r': {'value': -50, 'signal': 'neutral'},
                    'cci': {'value': 0, 'signal': 'neutral', 'extreme': False},
                    'atr': {'percentage': 2, 'volatility': 'medium', 'risk_level': 'medium'},
                    'fibonacci': {'fib_236': 0, 'fib_382': 0, 'fib_500': 0, 'fib_618': 0},
                    'pivot_points': {'pivot': tech_analysis.get('current_price', 0), 'r1': 0},
                    'trend_strength': {'strength': 'medium', 'direction': 'neutral'}
                }
                timings['extended_ms'] = round((time.time()-t_phase)*1000,2)
            
            # Pattern Recognition (20% weight)
            t_phase = time.time()
            pattern_analysis = self.pattern_detector.detect_advanced_patterns(candles)
            # Ensure timeframe tagging for primary pattern detection timeframe
            try:
                for p in pattern_analysis.get('patterns', []):
                    p.setdefault('timeframe', '1h')
            except Exception:
                pass
            timings['patterns_ms'] = round((time.time()-t_phase)*1000,2)

            # Multi-timeframe pattern scan (added enterprise)
            mt_pattern_frames = ['15m','4h','1d']
            multi_tf_patterns = []
            for ptf in mt_pattern_frames:
                try:
                    ptf_candles = self.technical_analysis.get_candle_data(symbol, interval=ptf, limit=120 if ptf!='1d' else 100)
                    if not ptf_candles or len(ptf_candles) < 40:
                        continue
                    pa = self.pattern_detector.detect_advanced_patterns(ptf_candles)
                    for pat in pa.get('patterns', []):
                        pat = dict(pat)
                        pat['timeframe'] = ptf
                        multi_tf_patterns.append(pat)
                except Exception as _e:
                    continue
            if multi_tf_patterns:
                pattern_analysis['multi_timeframe_patterns'] = multi_tf_patterns

            # Multi-Timeframe Analysis (NEW)
            mt_timeframes = ['15m', '1h', '4h', '1d']
            multi_timeframe = {'timeframes': [], 'consensus': {}}
            mt_signals = []
            for tf in mt_timeframes:
                try:
                    tf_candles = self.technical_analysis.get_candle_data(symbol, interval=tf, limit=150 if tf!='1d' else 120)
                    if len(tf_candles) < 50:
                        multi_timeframe['timeframes'].append({'tf': tf, 'error': 'no_data'})
                        continue
                    tf_analysis = self.technical_analysis.calculate_advanced_indicators(tf_candles)
                    rsi_v = tf_analysis.get('rsi', {}).get('rsi', 50)
                    sma_fast = tf_analysis.get('sma_9', tf_analysis.get('current_price'))
                    sma_slow = tf_analysis.get('sma_20', tf_analysis.get('current_price'))
                    trend_dir = 'UP' if sma_fast > sma_slow else 'DOWN' if sma_fast < sma_slow else 'SIDE'
                    # Signal logic
                    if rsi_v > 60 and trend_dir == 'UP':
                        tf_signal = 'strong_bull'
                    elif rsi_v > 55 and trend_dir in ('UP','SIDE'):
                        tf_signal = 'bull'
                    elif rsi_v < 40 and trend_dir == 'DOWN':
                        tf_signal = 'strong_bear'
                    elif rsi_v < 45 and trend_dir in ('DOWN','SIDE'):
                        tf_signal = 'bear'
                    else:
                        tf_signal = 'neutral'
                    mt_signals.append(tf_signal)
                    # timeframe contribution percentages will be added after loop
                    multi_timeframe['timeframes'].append({
                        'tf': tf,
                        'rsi': round(rsi_v,2),
                        'trend': trend_dir,
                        'signal': tf_signal,
                        'price': tf_analysis.get('current_price'),
                        'support': tf_analysis.get('support'),
                        'resistance': tf_analysis.get('resistance')
                    })
                except Exception as e:
                    multi_timeframe['timeframes'].append({'tf': tf, 'error': str(e)})
            # Consensus
            if mt_signals:
                counts = {k: mt_signals.count(k) for k in set(mt_signals)}
                def group_score(g):
                    return counts.get(g,0)
                bull_score = group_score('bull') + group_score('strong_bull')*1.5
                bear_score = group_score('bear') + group_score('strong_bear')*1.5
                if bull_score > bear_score and bull_score >= len(mt_signals)*0.5:
                    primary = 'BULLISH'
                elif bear_score > bull_score and bear_score >= len(mt_signals)*0.5:
                    primary = 'BEARISH'
                else:
                    primary = 'NEUTRAL'
                multi_timeframe['consensus'] = {
                    'bull_score': bull_score,
                    'bear_score': bear_score,
                    'total': len(mt_signals),
                    'primary': primary
                }
                # Add percentage distribution
                total_counts = sum(counts.values()) or 1
                multi_timeframe['distribution_pct'] = {k: round(v/total_counts*100,2) for k,v in counts.items()}
            else:
                multi_timeframe['consensus'] = {'primary': 'UNKNOWN'}

            # Market side strength (long vs short)
            side_score = {'long':0.0,'short':0.0,'neutral':0.0}
            side_basis = {}
            rsi_v_main = tech_analysis.get('rsi', {}).get('rsi',50)
            if rsi_v_main > 55: side_score['long']+=1; side_basis['rsi']='long'
            elif rsi_v_main < 45: side_score['short']+=1; side_basis['rsi']='short'
            else: side_basis['rsi']='neutral'
            macd_hist = tech_analysis.get('macd', {}).get('histogram',0)
            if macd_hist > 0: side_score['long']+=1; side_basis['macd']='long'
            elif macd_hist < 0: side_score['short']+=1; side_basis['macd']='short'
            else: side_basis['macd']='neutral'
            curve_dir = tech_analysis.get('macd', {}).get('curve_direction','neutral')
            if 'bullish' in curve_dir: side_score['long']+=0.5; side_basis['macd_curve']=curve_dir
            elif 'bearish' in curve_dir: side_score['short']+=0.5; side_basis['macd_curve']=curve_dir
            trend_struct = tech_analysis.get('trend', {}) if isinstance(tech_analysis.get('trend'), dict) else {}
            if trend_struct.get('trend') in ('bullish','strong_bullish'): side_score['long']+=1; side_basis['trend']='long'
            elif trend_struct.get('trend') in ('bearish','strong_bearish'): side_score['short']+=1; side_basis['trend']='short'
            else: side_basis['trend']=trend_struct.get('trend','neutral')
            bull_patterns = sum(1 for p in pattern_analysis.get('patterns', []) if p.get('signal')=='bullish')
            bear_patterns = sum(1 for p in pattern_analysis.get('patterns', []) if p.get('signal')=='bearish')
            if bull_patterns > bear_patterns: side_score['long']+=1; side_basis['patterns']='long'
            elif bear_patterns > bull_patterns: side_score['short']+=1; side_basis['patterns']='short'
            else: side_basis['patterns']='neutral'
            total_side = side_score['long']+side_score['short']+side_score['neutral'] or 1
            market_bias = {
                'long_strength_pct': round(side_score['long']/total_side*100,2),
                'short_strength_pct': round(side_score['short']/total_side*100,2),
                'basis': side_basis
            }
            
            # Position Management Analysis
            position_analysis = self.position_manager.analyze_position_potential(
                current_price,
                tech_analysis.get('support'),
                tech_analysis.get('resistance'),
                tech_analysis.get('trend', {}),
                pattern_analysis
            )
            
            # ðŸ“Š Order Flow Analysis (Enhanced Market Context)
            order_flow_data = self._analyze_order_flow(
                symbol, 
                current_price, 
                tech_analysis.get('volume_analysis', {}),
                multi_timeframe
            )
            
            # AI Analysis (10% weight)
            t_phase = time.time()
            ai_features = self.ai_system.prepare_advanced_features(
                tech_analysis, pattern_analysis, ticker_data, position_analysis, extended_analysis, regime_data=None
            )
            # Feature integrity hash & stats
            try:
                feat_payload = json.dumps(ai_features, sort_keys=True, default=str).encode()
                feature_hash = hashlib.sha256(feat_payload).hexdigest()[:16]
            except Exception:
                feature_hash = 'hash_error'
            # Monte Carlo uncertainty (passes env configurable)
            try:
                mc_passes = int(os.getenv('AI_MC_PASSES','15'))
            except Exception:
                mc_passes = 15
            ai_analysis = self.ai_system.predict_with_uncertainty(ai_features, passes=mc_passes)
            ai_analysis['feature_hash'] = feature_hash
            ai_analysis['feature_count'] = len(ai_features) if isinstance(ai_features, dict) else 0
            
            # ðŸ” Feature Contribution Analysis (AI Explainability)
            feature_contributions = self._analyze_feature_contributions(
                ai_features, 
                ai_analysis, 
                tech_analysis, 
                pattern_analysis
            )
            ai_analysis['feature_contributions'] = feature_contributions
            
            timings['ai_ms'] = round((time.time()-t_phase)*1000,2)
            
            # Calculate weighted final score
            t_phase = time.time()
            final_score = self._calculate_weighted_score(tech_analysis, pattern_analysis, ai_analysis)
            timings['scoring_ms'] = round((time.time()-t_phase)*1000,2)
            
            # Liquidation Analysis
            liquidation_long = self.liquidation_calc.calculate_liquidation_levels(current_price, 'long')
            liquidation_short = self.liquidation_calc.calculate_liquidation_levels(current_price, 'short')

            # ðŸŽ¯ Regime Detection (market classification)
            regime_data = self._detect_market_regime(
                candles,
                tech_analysis,
                extended_analysis,
                pattern_analysis,
                multi_timeframe
            )

            # Trade Setups (basic R/R framework)
            t_phase = time.time()
            trade_setups = self._generate_trade_setups(
                current_price,
                tech_analysis,
                extended_analysis,
                pattern_analysis,
                final_score,
                multi_timeframe,
                regime_data  # Pass regime context to setups
            )
            # Enrich trade setup rationales with multi-timeframe consensus & pattern counts
            try:
                bull_patterns = sum(1 for p in pattern_analysis.get('patterns', []) if p.get('signal')=='bullish')
                bear_patterns = sum(1 for p in pattern_analysis.get('patterns', []) if p.get('signal')=='bearish')
                mt_primary = multi_timeframe.get('consensus', {}).get('primary')
                data_span_days = None
                try:
                    first_ts = candles[0]['time'] if isinstance(candles[0], dict) else candles[0][0]
                    last_ts = candles[-1]['time'] if isinstance(candles[-1], dict) else candles[-1][0]
                    data_span_days = round((last_ts - first_ts)/(1000*60*60*24),2)
                except Exception:
                    pass
                # attach global data span for response
                if data_span_days is not None:
                    tech_analysis['data_span_days'] = data_span_days
                for s in trade_setups:
                    addon = f" | MTF: {mt_primary} | Patterns B:{bull_patterns}/S:{bear_patterns}"
                    if 'rationale' in s:
                        if addon.strip() not in s['rationale']:
                            s['rationale'] += addon
                    else:
                        s['rationale'] = addon.strip()
                    if data_span_days is not None:
                        s['data_span_days'] = data_span_days
                    s['market_bias'] = market_bias
                    # remove internal relaxation/fallback flags from public output if present
                    if 'relaxation_meta' in s:
                        rm = s['relaxation_meta']
                        rm.pop('fallback_generated', None)
            except Exception:
                pass
            timings['setups_ms'] = round((time.time()-t_phase)*1000,2)

            # Re-run validation including multi-timeframe consensus contradictions enhancement
            try:
                if isinstance(final_score, dict):
                    final_score['validation'] = self._validate_signals(tech_analysis, pattern_analysis, ai_analysis, final_score.get('signal'), multi_timeframe)
            except Exception as _e:
                pass
            
            # SAFE RETURN - Convert all numpy types to native Python
            print("ðŸ” Preparing return data...")
            
            def make_json_safe(obj):
                """Convert numpy types to JSON-safe types"""
                if isinstance(obj, dict):
                    return {k: make_json_safe(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_safe(item) for item in obj]
                elif hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif hasattr(obj, 'tolist'):  # numpy array
                    return obj.tolist()
                else:
                    return obj
            
            # Ensure validation structure always present even if scoring failed
            safe_final_score = final_score if isinstance(final_score, dict) else {
                'score': 50,
                'signal': 'HOLD',
                'signal_color': '#6c757d',
                'technical_weight': f"{self.weights['technical']*100}%",
                'pattern_weight': f"{self.weights['patterns']*100}%",
                'ai_weight': f"{self.weights['ai']*100}%",
                'component_scores': {'technical': 50, 'patterns': 50, 'ai': 50},
                'validation': {
                    'trading_action': 'WAIT',
                    'risk_level': 'MEDIUM',
                    'contradictions': [],
                    'warnings': [],
                    'confidence_factors': ['Default safety score used'],
                    'enterprise_ready': False
                }
            }

            # Build full response object expected by frontend
            timings['total_ms'] = round((time.time()-phase_t0)*1000,2)
            validation_summary = safe_final_score.get('validation', {})
            try:
                log_event('info', 'Enterprise validation summary', symbol=symbol, risk=validation_summary.get('risk_level'), contradictions=len(validation_summary.get('contradictions', [])), warnings=len(validation_summary.get('warnings', [])))
            except Exception:
                pass

            # Data Freshness & Timing augmentation
            try:
                price_ts = None
                # Binance ticker has 'closeTime' or 'closeTime' like field? Use eventTime if available, else derive from last candle
                for k in ('closeTime','close_time','eventTime','E','close'):  # attempt common keys
                    if k in ticker_data and isinstance(ticker_data.get(k), (int,float)) and ticker_data.get(k) > 0:
                        price_ts = int(ticker_data.get(k))
                        break
                if price_ts is None and candles:
                    price_ts = candles[-1].get('time') or candles[-1].get('timestamp')
                now_ms = int(time.time()*1000)
                freshness_ms = int(now_ms - price_ts) if price_ts else None
            except Exception:
                price_ts = None
                freshness_ms = None

            # ðŸ“Š Adaptive Risk & Target Sizing
            adaptive_risk = self._calculate_adaptive_risk_targets(symbol, current_price, tech_analysis, regime_data, ai_analysis)

            result = make_json_safe({
                'symbol': symbol,
                'current_price': float(current_price),
                'market_data': ticker_data,  # original ticker payload for 24h stats
                'technical_analysis': tech_analysis,
                'extended_analysis': extended_analysis,
                'pattern_analysis': pattern_analysis,
                'multi_timeframe': multi_timeframe,
                'position_analysis': position_analysis,
                'ai_analysis': ai_analysis,
                'ai_feature_hash': ai_analysis.get('feature_hash'),
                'order_flow_analysis': order_flow_data,
                'adaptive_risk_targets': adaptive_risk,
                'market_bias': market_bias,
                'regime_analysis': regime_data,
                'liquidation_long': liquidation_long,
                'liquidation_short': liquidation_short,
                'trade_setups': trade_setups,
                'weights': self.weights,
                'final_score': safe_final_score,
                'phase_timings_ms': timings,
                'timestamp': datetime.now().isoformat(),
                'server_time_ms': int(time.time()*1000),
                'price_timestamp_ms': price_ts,
                'data_freshness_ms': freshness_ms
            })
            
            print("âœ… Return data prepared successfully")
            return result
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Analysis error: {e}")
            print(f"Full traceback: {error_trace}")
            return {'error': f'Analysis failed: {str(e)}', 'traceback': error_trace}
    
    def _detect_market_regime(self, candles, tech_analysis, extended_analysis, pattern_analysis, multi_timeframe):
        """ðŸŽ¯ Market Regime Classification: Trend, Range, Expansion, Volatility Crush"""
        try:
            if len(candles) < 50:
                return {'regime': 'unknown', 'confidence': 0, 'rationale': 'Insufficient data'}

            closes = np.array([c['close'] for c in candles])
            highs = np.array([c['high'] for c in candles])
            lows = np.array([c['low'] for c in candles])
            volumes = np.array([c['volume'] for c in candles])
            
            # Get ATR for volatility assessment
            atr_data = extended_analysis.get('atr', {})
            volatility_level = atr_data.get('volatility', 'medium')
            atr_pct = atr_data.get('percentage', 2.0)
            
            # Trend Assessment
            trend_data = tech_analysis.get('trend', {})
            trend_strength = trend_data.get('strength', 'weak')
            trend_direction = trend_data.get('trend', 'neutral')
            
            # Range Detection (price bounded within support/resistance)
            support = tech_analysis.get('support', 0)
            resistance = tech_analysis.get('resistance', 0)
            current_price = closes[-1]
            range_pct = ((resistance - support) / current_price * 100) if resistance > support else 0
            
            # Multi-timeframe consensus
            mt_consensus = multi_timeframe.get('consensus', {}).get('primary', 'NEUTRAL')
            
            # Volume Analysis
            vol_trend = tech_analysis.get('volume_analysis', {}).get('trend', 'normal')
            
            # Pattern Momentum
            patterns = pattern_analysis.get('patterns', [])
            breakout_patterns = [p for p in patterns if 'breakout' in p.get('type', '').lower()]
            consolidation_patterns = [p for p in patterns if any(x in p.get('type', '').lower() for x in ['triangle', 'flag', 'pennant'])]
            
            # Decision Logic
            regime_scores = {
                'trending': 0,
                'ranging': 0,
                'expansion': 0,
                'volatility_crush': 0
            }
            
            # Trending signals
            if trend_strength in ['strong', 'very_strong'] and trend_direction != 'neutral':
                regime_scores['trending'] += 30
            if mt_consensus in ['BULLISH', 'BEARISH']:
                regime_scores['trending'] += 20
            if atr_pct > 2.5 and vol_trend in ['high', 'very_high']:
                regime_scores['trending'] += 15
            
            # Ranging signals
            if range_pct < 8 and current_price > support * 1.02 and current_price < resistance * 0.98:
                regime_scores['ranging'] += 35
            if trend_direction in ['neutral', 'sideways']:
                regime_scores['ranging'] += 25
            if len(consolidation_patterns) > 0:
                regime_scores['ranging'] += 15
                
            # Expansion signals (high volatility + breakouts)
            if atr_pct > 4.0:
                regime_scores['expansion'] += 25
            if len(breakout_patterns) > 0:
                regime_scores['expansion'] += 30
            if vol_trend == 'very_high':
                regime_scores['expansion'] += 20
                
            # Volatility Crush signals (low volatility + compression)
            if atr_pct < 1.5:
                regime_scores['volatility_crush'] += 30
            if volatility_level == 'low':
                regime_scores['volatility_crush'] += 25
            if vol_trend in ['low', 'below_average']:
                regime_scores['volatility_crush'] += 20
            
            # Determine regime
            primary_regime = max(regime_scores, key=regime_scores.get)
            confidence = regime_scores[primary_regime]
            
            # Rationale construction
            if primary_regime == 'trending':
                rationale = f"Strong directional movement detected. Trend: {trend_direction} ({trend_strength}), MTF: {mt_consensus}, ATR: {atr_pct:.1f}%"
            elif primary_regime == 'ranging':
                rationale = f"Price bounded in range. S/R spread: {range_pct:.1f}%, Trend: {trend_direction}, Consolidation patterns: {len(consolidation_patterns)}"
            elif primary_regime == 'expansion':
                rationale = f"High volatility expansion phase. ATR: {atr_pct:.1f}%, Breakouts: {len(breakout_patterns)}, Volume: {vol_trend}"
            else:  # volatility_crush
                rationale = f"Low volatility compression. ATR: {atr_pct:.1f}%, Vol trend: {vol_trend}, Squeeze conditions present"
            
            # Secondary regime (if close scores)
            sorted_scores = sorted(regime_scores.items(), key=lambda x: x[1], reverse=True)
            secondary = None
            if len(sorted_scores) > 1 and sorted_scores[1][1] > confidence * 0.7:
                secondary = sorted_scores[1][0]
            
            return {
                'regime': primary_regime,
                'secondary_regime': secondary,
                'confidence': min(100, confidence),
                'rationale': rationale,
                'regime_scores': regime_scores,
                'volatility_level': volatility_level,
                'atr_percentage': atr_pct,
                'range_percentage': range_pct,
                'trend_classification': f"{trend_direction} ({trend_strength})"
            }
            
        except Exception as e:
            return {
                'regime': 'error', 
                'confidence': 0, 
                'rationale': f'Regime detection failed: {str(e)}'
            }
    
    def _generate_rsi_caution_narrative(self, rsi, trend):
        """ðŸš¨ Generate RSI-based caution narrative and confidence penalties"""
        caution_level = 'none'
        narrative = ''
        confidence_penalty = 0
        signal_quality = 'ok'
        
        if rsi >= 80:
            caution_level = 'extreme'
            narrative = 'âš ï¸ EXTREME ÃœBERKAUFT: RSI sehr hoch - Pullback-Risiko erhÃ¶ht, reduzierte Position empfohlen'
            confidence_penalty = 25
            signal_quality = 'bad'
        elif rsi >= 70:
            caution_level = 'high'
            narrative = 'âš ï¸ ÃœBERKAUFT-WARNUNG: RSI Ã¼ber 70 - Vorsicht bei LONG-Einstiegen, enge Stops verwenden'
            confidence_penalty = 15
            signal_quality = 'warn'
        elif rsi <= 20:
            caution_level = 'extreme_oversold'
            narrative = 'ðŸ’¡ EXTREME ÃœBERVERKAUFT: RSI sehr niedrig - Bounce-Potential hoch, aber weitere SchwÃ¤che mÃ¶glich'
            confidence_penalty = 0  # Oversold can be opportunity
            signal_quality = 'ok'
        elif rsi <= 30:
            caution_level = 'oversold_opportunity'
            narrative = 'ðŸ’¡ ÃœBERVERKAUFT: RSI unter 30 - Potentielle Einstiegschance fÃ¼r LONG bei BestÃ¤tigung'
            confidence_penalty = -5  # Small bonus for oversold
            signal_quality = 'ok'
        elif rsi >= 60 and 'bearish' in trend:
            caution_level = 'trend_conflict'
            narrative = 'âš ï¸ TREND-KONFLIKT: RSI erhÃ¶ht in bearischem Trend - kurzfristige Rallye kÃ¶nnte enden'
            confidence_penalty = 10
            signal_quality = 'warn'
        elif rsi <= 40 and 'bullish' in trend:
            caution_level = 'healthy_pullback'
            narrative = 'âœ… GESUNDER PULLBACK: RSI moderat in bullischem Trend - Einstiegschance bei Support'
            confidence_penalty = -3  # Small bonus
            signal_quality = 'ok'
            
        return {
            'caution_level': caution_level,
            'narrative': narrative,
            'confidence_penalty': confidence_penalty,
            'signal_quality': signal_quality,
            'rsi_value': rsi,
            'recommendation': self._get_rsi_recommendation(rsi, trend)
        }
    
    def _get_rsi_recommendation(self, rsi, trend):
        """Get specific RSI-based trading recommendation"""
        if rsi >= 80:
            return 'Avoid new LONG positions, consider profit-taking'
        elif rsi >= 70:
            return 'Use tight stops on LONG positions, monitor for reversal signals'
        elif rsi <= 20:
            return 'Monitor for reversal confirmation before entering LONG'
        elif rsi <= 30:
            return 'Good LONG entry zone if trend supports'
        elif 40 <= rsi <= 60:
            return 'Neutral RSI - rely on other indicators'
        else:
            return 'RSI in normal range - standard trade management'
    
    def _analyze_order_flow(self, symbol, current_price, volume_analysis, multi_timeframe):
        """ðŸ“Š Order Flow & Market Microstructure Analysis"""
        try:
            # Simulated Order Flow (placeholder for real orderbook integration)
            # In production, this would fetch actual orderbook data
            order_flow = {
                'bid_ask_spread': 0,
                'order_book_imbalance': 0,
                'delta_momentum': 0,
                'volume_profile_poc': current_price,
                'liquidity_zones': [],
                'flow_sentiment': 'neutral'
            }
            
            # Volume-based flow estimation
            vol_ratio = volume_analysis.get('ratio', 1.0)
            vol_trend = volume_analysis.get('trend', 'normal')
            
            # Estimate bid/ask spread based on volatility
            if vol_ratio > 2.0:
                spread_estimate = current_price * 0.001  # 0.1% in high vol
            elif vol_ratio > 1.5:
                spread_estimate = current_price * 0.0005  # 0.05%
            else:
                spread_estimate = current_price * 0.0002  # 0.02% normal
            
            order_flow['bid_ask_spread'] = round(spread_estimate, 6)
            order_flow['spread_bps'] = round((spread_estimate / current_price) * 10000, 2)
            
            # Estimate order imbalance from volume pattern
            if vol_trend == 'very_high' and vol_ratio > 1.8:
                # High volume suggests imbalance
                imbalance = min(0.7, (vol_ratio - 1) * 0.3)  # Cap at 70%
                order_flow['order_book_imbalance'] = round(imbalance, 3)
                order_flow['flow_sentiment'] = 'buy_pressure' if imbalance > 0.3 else 'sell_pressure'
            elif vol_trend == 'below_average':
                order_flow['order_book_imbalance'] = round(-(1 - vol_ratio) * 0.2, 3)
                order_flow['flow_sentiment'] = 'low_liquidity'
            else:
                order_flow['order_book_imbalance'] = round((vol_ratio - 1) * 0.1, 3)
            
            # Multi-timeframe delta estimation
            mt_consensus = multi_timeframe.get('consensus', {}).get('primary', 'NEUTRAL')
            if mt_consensus == 'BULLISH':
                order_flow['delta_momentum'] = round(0.3 + vol_ratio * 0.2, 3)
            elif mt_consensus == 'BEARISH':
                order_flow['delta_momentum'] = round(-0.3 - vol_ratio * 0.2, 3)
            else:
                order_flow['delta_momentum'] = round((vol_ratio - 1) * 0.1, 3)
            
            # Volume Profile Point of Control estimation
            order_flow['volume_profile_poc'] = round(current_price * (1 + order_flow['delta_momentum'] * 0.01), 4)
            
            # Liquidity zones (approximate based on volume)
            if vol_ratio > 1.5:
                zones = [
                    {'level': round(current_price * 0.995, 4), 'type': 'support', 'strength': 'medium'},
                    {'level': round(current_price * 1.005, 4), 'type': 'resistance', 'strength': 'medium'}
                ]
                if vol_ratio > 2.0:
                    zones.extend([
                        {'level': round(current_price * 0.99, 4), 'type': 'support', 'strength': 'strong'},
                        {'level': round(current_price * 1.01, 4), 'type': 'resistance', 'strength': 'strong'}
                    ])
                order_flow['liquidity_zones'] = zones
            
            # Flow strength assessment
            imb_abs = abs(order_flow['order_book_imbalance'])
            delta_abs = abs(order_flow['delta_momentum'])
            
            if imb_abs > 0.4 or delta_abs > 0.5:
                flow_strength = 'strong'
            elif imb_abs > 0.2 or delta_abs > 0.3:
                flow_strength = 'moderate'
            else:
                flow_strength = 'weak'
            
            order_flow['flow_strength'] = flow_strength
            order_flow['analysis_note'] = f"Estimated flow from volume patterns (spread: {order_flow['spread_bps']}bps, imbalance: {order_flow['order_book_imbalance']:.1%})"
            
            return order_flow
            
        except Exception as e:
            return {
                'error': f'Order flow analysis failed: {str(e)}',
                'flow_sentiment': 'unknown',
                'flow_strength': 'unknown'
            }
    
    def _analyze_feature_contributions(self, features, ai_analysis, tech_analysis, pattern_analysis):
        """ðŸ” AI Feature Contribution Analysis (Explainability)"""
        try:
            if not isinstance(features, (list, np.ndarray)) or len(features) == 0:
                return {'error': 'No features available for contribution analysis'}
            
            # Convert to numpy array if needed
            if isinstance(features, list):
                features = np.array(features)
            
            # Feature importance heuristic (simplified attribution)
            # In production, this would use actual gradient-based attribution
            
            feature_names = [
                'RSI', 'RSI_Overbought', 'RSI_Oversold', 'MACD', 'MACD_Hist', 
                'MACD_Bull_Curve', 'MACD_Bear_Curve', 'MACD_Bull_Rev', 'MACD_Bear_Rev',
                'SMA_9', 'SMA_20', 'Support_Strength', 'Resistance_Strength'
            ]
            
            # Extend feature names for all 128 features
            for i in range(len(feature_names), 50):
                feature_names.append(f'Tech_{i}')
            for i in range(50, 80):
                feature_names.append(f'Pattern_{i-50}')
            for i in range(80, 100):
                feature_names.append(f'Market_{i-80}')
            for i in range(100, 120):
                feature_names.append(f'Position_{i-100}')
            for i in range(120, 128):
                feature_names.append(f'Time_{i-120}')
            
            # Ensure we have enough names
            while len(feature_names) < len(features):
                feature_names.append(f'Feature_{len(feature_names)}')
            
            # Calculate pseudo-importance (magnitude * activation)
            feature_magnitudes = np.abs(features)
            feature_activations = np.where(features > 0, features, -features * 0.5)  # Positive bias
            importance_scores = feature_magnitudes * feature_activations
            
            # Normalize to percentages
            total_importance = np.sum(importance_scores)
            if total_importance > 0:
                normalized_importance = (importance_scores / total_importance) * 100
            else:
                normalized_importance = np.zeros_like(importance_scores)
            
            # Get top contributing features
            top_indices = np.argsort(normalized_importance)[-10:][::-1]  # Top 10
            
            contributions = []
            for idx in top_indices:
                if idx < len(feature_names) and normalized_importance[idx] > 0.5:  # Only meaningful contributions
                    contributions.append({
                        'feature': feature_names[idx],
                        'importance': round(float(normalized_importance[idx]), 2),
                        'value': round(float(features[idx]), 4),
                        'impact': 'positive' if features[idx] > 0 else 'negative'
                    })
            
            # Add contextual interpretations
            interpretations = []
            rsi_val = tech_analysis.get('rsi', {}).get('rsi', 50)
            if rsi_val > 70:
                interpretations.append('RSI overbought condition reducing BUY confidence')
            elif rsi_val < 30:
                interpretations.append('RSI oversold condition increasing BUY potential')
            
            pattern_count = len(pattern_analysis.get('patterns', []))
            if pattern_count > 0:
                bull_patterns = sum(1 for p in pattern_analysis.get('patterns', []) if p.get('signal') == 'bullish')
                if bull_patterns > 0:
                    interpretations.append(f'{bull_patterns} bullish patterns supporting upside')
            
            ai_signal = ai_analysis.get('signal', 'HOLD')
            ai_conf = ai_analysis.get('confidence', 0)
            if ai_conf > 70:
                interpretations.append(f'High AI confidence ({ai_conf:.1f}%) in {ai_signal} signal')
            
            return {
                'top_features': contributions[:5],  # Top 5 for display
                'total_features_analyzed': len(features),
                'ai_signal_confidence': ai_conf,
                'contextual_interpretations': interpretations,
                'analysis_method': 'magnitude_activation_heuristic',
                'note': 'Simplified feature attribution - production systems use gradient-based methods'
            }
            
        except Exception as e:
            return {
                'error': f'Feature contribution analysis failed: {str(e)}',
                'top_features': [],
                'analysis_method': 'error'
            }
    
    def _calculate_adaptive_risk_targets(self, symbol, current_price, tech_analysis, regime_data, ai_analysis):
        """ðŸ“Š Adaptive Risk & Target Sizing based on Market Conditions"""
        try:
            # Base parameters
            base_risk_pct = 2.0  # 2% base risk
            base_reward_ratio = 2.0  # 1:2 risk/reward base
            
            # Volatility adjustment from ATR
            atr = tech_analysis.get('atr', {}).get('atr', current_price * 0.02)
            atr_pct = (atr / current_price) * 100
            
            # Volatility risk scaling
            if atr_pct > 5.0:  # Very high volatility
                vol_multiplier = 0.6  # Reduce risk
                reward_multiplier = 3.0  # Increase reward ratio
            elif atr_pct > 3.0:  # High volatility
                vol_multiplier = 0.8
                reward_multiplier = 2.5
            elif atr_pct > 1.5:  # Normal volatility
                vol_multiplier = 1.0
                reward_multiplier = 2.0
            elif atr_pct > 0.8:  # Low volatility
                vol_multiplier = 1.2  # Increase risk slightly
                reward_multiplier = 1.8
            else:  # Very low volatility
                vol_multiplier = 1.4
                reward_multiplier = 1.5
            
            # Regime-based adjustments
            regime_multiplier = 1.0
            regime_reward_adj = 1.0
            
            if regime_data and 'regime_type' in regime_data:
                regime_type = regime_data['regime_type']
                regime_confidence = regime_data.get('confidence', 50)
                
                if regime_type == 'trending':
                    # Higher confidence in trends = more aggressive
                    if regime_confidence > 70:
                        regime_multiplier = 1.3
                        regime_reward_adj = 2.5
                    else:
                        regime_multiplier = 1.1
                        regime_reward_adj = 2.2
                        
                elif regime_type == 'ranging':
                    # Ranging markets = smaller targets, tighter stops
                    regime_multiplier = 0.8
                    regime_reward_adj = 1.5
                    
                elif regime_type == 'expansion':
                    # Expansion = potential for larger moves
                    regime_multiplier = 1.2
                    regime_reward_adj = 3.0
                    
                elif regime_type == 'volatility_crush':
                    # Low vol environment
                    regime_multiplier = 0.7
                    regime_reward_adj = 1.3
            
            # AI confidence adjustment
            ai_confidence = ai_analysis.get('confidence', 50)
            ai_signal = ai_analysis.get('signal', 'HOLD')
            
            if ai_confidence > 80 and ai_signal != 'HOLD':
                confidence_multiplier = 1.4  # High confidence = more aggressive
            elif ai_confidence > 60:
                confidence_multiplier = 1.2
            elif ai_confidence > 40:
                confidence_multiplier = 1.0
            else:
                confidence_multiplier = 0.7  # Low confidence = conservative

            # --- NEW: AI Uncertainty Adjustment (entropy & std) ---
            uncertainty = ai_analysis.get('uncertainty') or {}
            entropy = uncertainty.get('entropy')
            avg_std = uncertainty.get('avg_std')
            # Maximum entropy for 4 classes = ln(4)
            import math
            max_entropy = math.log(4)
            if entropy is not None and entropy >= 0:
                norm_entropy = min(1.0, entropy / max_entropy)
                # Reduce risk up to -40% at maximum entropy (very uncertain)
                uncertainty_multiplier = 1.0 - 0.4 * norm_entropy
            else:
                uncertainty_multiplier = 1.0
            # Additional penalty if model probability std is high (>0.12 typical moderate)
            if avg_std is not None:
                if avg_std > 0.18:
                    uncertainty_multiplier *= 0.8  # strong penalty
                elif avg_std > 0.12:
                    uncertainty_multiplier *= 0.9  # mild penalty
            # Floor to avoid zeroing risk completely
            uncertainty_multiplier = max(0.5, min(1.0, uncertainty_multiplier))
            
            # Calculate final risk percentage
            adaptive_risk_pct = base_risk_pct * vol_multiplier * regime_multiplier * confidence_multiplier * uncertainty_multiplier
            adaptive_risk_pct = max(0.5, min(5.0, adaptive_risk_pct))  # Cap between 0.5% and 5%
            
            # Calculate reward ratio
            adaptive_reward_ratio = base_reward_ratio * reward_multiplier * regime_reward_adj
            adaptive_reward_ratio = max(1.2, min(4.0, adaptive_reward_ratio))  # Cap between 1.2:1 and 4:1
            
            # Position sizing (assuming $10,000 account)
            account_size = 10000
            risk_amount = account_size * (adaptive_risk_pct / 100)
            
            # Stop loss distance
            stop_distance_pct = atr_pct * 0.8  # Use 80% of ATR for stop
            stop_distance = current_price * (stop_distance_pct / 100)
            
            # Position size calculation
            position_size = risk_amount / stop_distance if stop_distance > 0 else 0
            
            # Target levels
            target_1_distance = stop_distance * (adaptive_reward_ratio * 0.6)  # 60% of full target
            target_2_distance = stop_distance * adaptive_reward_ratio  # Full target
            target_3_distance = stop_distance * (adaptive_reward_ratio * 1.5)  # Extended target
            
            if ai_signal == 'BUY':
                stop_loss = current_price - stop_distance
                target_1 = current_price + target_1_distance
                target_2 = current_price + target_2_distance
                target_3 = current_price + target_3_distance
            else:  # SELL
                stop_loss = current_price + stop_distance
                target_1 = current_price - target_1_distance
                target_2 = current_price - target_2_distance
                target_3 = current_price - target_3_distance
            
            # Risk assessment
            risk_category = 'low'
            if adaptive_risk_pct > 3.5:
                risk_category = 'high'
            elif adaptive_risk_pct > 2.5:
                risk_category = 'medium'
            
            return {
                'adaptive_risk_pct': round(adaptive_risk_pct, 2),
                'adaptive_reward_ratio': round(adaptive_reward_ratio, 1),
                'position_size': round(position_size, 4),
                'stop_loss': round(stop_loss, 4),
                'targets': {
                    'target_1': round(target_1, 4),
                    'target_2': round(target_2, 4), 
                    'target_3': round(target_3, 4)
                },
                'risk_amount_usd': round(risk_amount, 2),
                'stop_distance_pct': round(stop_distance_pct, 3),
                'atr_pct': round(atr_pct, 3),
                'risk_category': risk_category,
                'adjustments': {
                    'volatility_multiplier': round(vol_multiplier, 2),
                    'regime_multiplier': round(regime_multiplier, 2),
                    'confidence_multiplier': round(confidence_multiplier, 2),
                    'uncertainty_multiplier': round(uncertainty_multiplier, 2),
                    'reward_multiplier': round(reward_multiplier, 1)
                },
                'reasoning': f"Risk adjusted for {regime_data.get('regime_type', 'normal')} regime, {atr_pct:.1f}% volatility, {ai_confidence:.0f}% AI confidence, entropy={entropy if entropy is not None else 'n/a'}"
            }
            
        except Exception as e:
            return {
                'error': f'Adaptive risk calculation failed: {str(e)}',
                'adaptive_risk_pct': 2.0,
                'adaptive_reward_ratio': 2.0
            }
    
    def _calculate_weighted_score(self, tech_analysis, pattern_analysis, ai_analysis):
        """Calculate weighted final trading score"""
        # Technical score (70%)
        tech_score = 50  # Neutral base
        
        # Fix RSI access - it's directly available, not nested
        rsi = tech_analysis.get('rsi', {}).get('rsi', 50)
        if isinstance(rsi, (int, float)):
            if rsi > 70:
                tech_score -= (rsi - 70) * 0.5  # Penalize overbought
            elif rsi < 30:
                tech_score += (30 - rsi) * 0.5  # Reward oversold
        
        # Fix trend access
        trend_data = tech_analysis.get('trend', {})
        if isinstance(trend_data, dict):
            trend = trend_data.get('trend', 'neutral')
        else:
            trend = 'neutral'
            
        if trend in ['strong_bullish', 'bullish']:
            tech_score += 25
        elif trend in ['strong_bearish', 'bearish']:
            tech_score -= 25
        
        # Fix MACD access
        macd_data = tech_analysis.get('macd', {})
        if isinstance(macd_data, dict):
            macd_signal = macd_data.get('curve_direction', 'neutral')
            if 'bullish' in macd_signal:
                tech_score += 15
            elif 'bearish' in macd_signal:
                tech_score -= 15
        
        # Pattern score (20%)
        pattern_score = 50  # Neutral base
        patterns = pattern_analysis.get('patterns', [])
        
        for pattern in patterns:
            if pattern['signal'] == 'bullish':
                pattern_score += pattern['confidence'] * 0.3
            elif pattern['signal'] == 'bearish':
                pattern_score -= pattern['confidence'] * 0.3
        
        # AI score (10%)
        ai_signal = ai_analysis.get('signal', 'HOLD')
        ai_confidence = ai_analysis.get('confidence', 50)
        
        if ai_signal == 'STRONG_BUY':
            ai_score = 75 + (ai_confidence - 50) * 0.5
        elif ai_signal == 'BUY':
            ai_score = 60 + (ai_confidence - 50) * 0.3
        elif ai_signal == 'STRONG_SELL':
            ai_score = 25 - (ai_confidence - 50) * 0.5
        elif ai_signal == 'SELL':
            ai_score = 40 - (ai_confidence - 50) * 0.3
        else:
            ai_score = 50
        
        # Dynamic weight adaptation (reduce AI weight if offline or low confidence)
        dyn_weights = dict(self.weights)
        ai_conf = ai_analysis.get('confidence', 50)
        ai_status = ai_analysis.get('status') or ( 'offline' if not ai_analysis.get('initialized', True) else 'online')
        if ai_status == 'offline' or ai_conf < 40:
            # remove AI contribution, re-normalize others
            removed = dyn_weights.get('ai', 0)
            dyn_weights['ai'] = 0.0
            rem = dyn_weights['technical'] + dyn_weights['patterns']
            if rem <= 0:
                dyn_weights['technical'] = 0.7
                dyn_weights['patterns'] = 0.3
            else:
                dyn_weights['technical'] = dyn_weights['technical']/rem
                dyn_weights['patterns'] = dyn_weights['patterns']/rem
        # Weighted final score (raw aggregate)
        final_score = (
            tech_score * dyn_weights['technical'] +
            pattern_score * dyn_weights['patterns'] +
            ai_score * dyn_weights.get('ai',0)
        )

        # --- NEW: Uncertainty Damping (reduces extremes when entropy high) ---
        unc = ai_analysis.get('uncertainty') or {}
        entropy = unc.get('entropy')
        avg_std = unc.get('avg_std')
        if entropy is not None:
            import math
            max_ent = math.log(4)
            norm_ent = min(1.0, entropy / max_ent)
            # Pull score 50% towards 50 baseline proportional to uncertainty (up to 25% shrink)
            damping = 0.25 * norm_ent
            final_score = 50 + (final_score - 50) * (1 - damping)
        if avg_std is not None and avg_std > 0.12:
            # Additional mild damping for high predictive variance
            extra = 0.1 if avg_std > 0.18 else 0.05
            final_score = 50 + (final_score - 50) * (1 - extra)
        
        # Clamp to 0-100 range
        final_score = max(0, min(100, final_score))
        
        # Determine final signal
        if final_score >= 75:
            signal = 'STRONG_BUY'
            signal_color = '#28a745'
        elif final_score >= 60:
            signal = 'BUY'
            signal_color = '#6f42c1'
        elif final_score <= 25:
            signal = 'STRONG_SELL'
            signal_color = '#dc3545'
        elif final_score <= 40:
            signal = 'SELL'
            signal_color = '#fd7e14'
        else:
            signal = 'HOLD'
            signal_color = '#6c757d'
        
        # Pseudo-Probability Calibration: map 0-100 score into 0-1 via logistic (center 50, slope 0.09)
        import math
        calibrated_prob = 1/(1+math.exp(-(final_score-50)*0.09))
        # Side-specific probability (prob of bullishness). If signal is SELL side invert.
        bullish_prob = calibrated_prob
        if signal in ['SELL','STRONG_SELL']:
            bullish_prob = 1 - bullish_prob
        # Attach reason if AI disabled
        ai_reason = None
        if dyn_weights.get('ai',0) == 0 and self.weights.get('ai',0) > 0:
            # Determine reason heuristically
            if ai_analysis.get('mode') == 'offline' or not ai_analysis.get('signal'): ai_reason = 'AI offline/initialization failed'
            elif ai_analysis.get('confidence',0) < 40: ai_reason = f'AI low confidence {ai_analysis.get("confidence",0)} < 40'
            else: ai_reason = 'AI weight dynamically suppressed'
        return {
            'score': round(final_score, 1),
            'probability_bullish': round(bullish_prob*100,2),
            'calibrated_probability': round(calibrated_prob*100,2),
            'probability_note': 'Heuristische Kalibrierung (logistische Kurve) â€“ keine echte statistische Eintrittswahrscheinlichkeit',
            'signal': signal,
            'signal_color': signal_color,
            'technical_weight': f"{dyn_weights['technical']*100:.1f}%",
            'pattern_weight': f"{dyn_weights['patterns']*100:.1f}%",
            'ai_weight': f"{dyn_weights.get('ai',0)*100:.1f}%",
            'ai_disable_reason': ai_reason,
            'component_scores': {
                'technical': round(tech_score, 1),
                'patterns': round(pattern_score, 1),
                'ai': round(ai_score, 1)
            },
            'validation': self._validate_signals(tech_analysis, pattern_analysis, ai_analysis, signal)
        }

    # (Deprecated earlier _generate_trade_setups removed in favor of advanced version below)
    
    def _validate_signals(self, tech_analysis, pattern_analysis, ai_analysis, final_signal, multi_timeframe=None):
        """Enterprise-Level Signal Validation - Eliminiert WidersprÃ¼che"""
        warnings = []
        contradictions = []
        confidence_factors = []
        
        # 1. MACD vs Final Signal Validation
        macd_signal = tech_analysis.get('macd', {}).get('curve_direction', 'neutral')
        
        if 'bearish' in macd_signal and final_signal in ['BUY', 'STRONG_BUY']:
            contradictions.append({
                'type': 'MACD_CONTRADICTION',
                'message': f'âš ï¸ MACD zeigt {macd_signal.upper()}, aber Signal ist {final_signal}',
                'severity': 'HIGH',
                'recommendation': 'WARTE auf besseren Einstieg - MACD Bogen ist bearish!'
            })
        
        if 'bullish' in macd_signal and final_signal in ['SELL', 'STRONG_SELL']:
            contradictions.append({
                'type': 'MACD_CONTRADICTION', 
                'message': f'âš ï¸ MACD zeigt {macd_signal.upper()}, aber Signal ist {final_signal}',
                'severity': 'HIGH',
                'recommendation': 'WARTE auf besseren Einstieg - MACD Bogen ist bullish!'
            })
        
        # 2. RSI Extreme Levels Validation
        rsi_data = tech_analysis.get('rsi', {})
        if isinstance(rsi_data, dict):
            rsi = rsi_data.get('rsi', 50)
        else:
            rsi = rsi_data if isinstance(rsi_data, (int, float)) else 50
        
        if rsi > 80 and final_signal in ['BUY', 'STRONG_BUY']:
            warnings.append({
                'type': 'RSI_OVERBOUGHT',
                'message': f'âš ï¸ RSI Ã¼berkauft ({rsi:.1f}) - Vorsicht bei LONG!',
                'recommendation': 'Warte auf RSI RÃ¼ckgang unter 70'
            })
        
        if rsi < 20 and final_signal in ['SELL', 'STRONG_SELL']:
            warnings.append({
                'type': 'RSI_OVERSOLD',
                'message': f'âš ï¸ RSI Ã¼berverkauft ({rsi:.1f}) - Vorsicht bei SHORT!',
                'recommendation': 'Warte auf RSI Anstieg Ã¼ber 30'
            })
        
        # 3. Support/Resistance Validation
        support = tech_analysis.get('support', 0)
        resistance = tech_analysis.get('resistance', 0)
        current_price = tech_analysis.get('current_price', 0)
        
        if current_price > 0:
            distance_to_resistance = ((resistance - current_price) / current_price) * 100
            distance_to_support = ((current_price - support) / current_price) * 100
            
            if distance_to_resistance < 2 and final_signal in ['BUY', 'STRONG_BUY']:
                warnings.append({
                    'type': 'NEAR_RESISTANCE',
                    'message': f'âš ï¸ Preis nur {distance_to_resistance:.1f}% unter Resistance',
                    'recommendation': 'Sehr riskanter LONG Einstieg - Resistance sehr nah!'
                })
            
            if distance_to_support < 2 and final_signal in ['SELL', 'STRONG_SELL']:
                warnings.append({
                    'type': 'NEAR_SUPPORT',
                    'message': f'âš ï¸ Preis nur {distance_to_support:.1f}% Ã¼ber Support',
                    'recommendation': 'Sehr riskanter SHORT Einstieg - Support sehr nah!'
                })
        
        # 4. Pattern Consistency Check
        patterns = pattern_analysis.get('patterns', [])
        pattern_signals = [p['signal'] for p in patterns]
        
        bearish_patterns = sum(1 for s in pattern_signals if s == 'bearish')
        bullish_patterns = sum(1 for s in pattern_signals if s == 'bullish')
        
        if bearish_patterns > bullish_patterns and final_signal in ['BUY', 'STRONG_BUY']:
            contradictions.append({
                'type': 'PATTERN_CONTRADICTION',
                'message': f'âš ï¸ {bearish_patterns} bearish vs {bullish_patterns} bullish patterns',
                'severity': 'MEDIUM',
                'recommendation': 'Chart Muster sprechen gegen LONG Position!'
            })
        
        # 5. AI Confidence & Consistency Validation
        ai_confidence = ai_analysis.get('confidence', 50)
        ai_signal = ai_analysis.get('signal', 'HOLD')
        if ai_confidence < 60:
            warnings.append({
                'type': 'LOW_AI_CONFIDENCE',
                'message': f'âš ï¸ KI Confidence nur {ai_confidence}%',
                'recommendation': 'KI ist unsicher - warte auf klarere Signale!'
            })
        # AI vs final signal disagreement (directionally opposite)
        opposite_map = {
            'STRONG_BUY': ['SELL','STRONG_SELL'],
            'BUY': ['STRONG_SELL','SELL'],
            'STRONG_SELL': ['BUY','STRONG_BUY'],
            'SELL': ['STRONG_BUY','BUY']
        }
        if ai_signal in opposite_map.get(final_signal, []) and ai_confidence >= 55:
            contradictions.append({
                'type': 'AI_FINAL_CONTRADICTION',
                'message': f'âš ï¸ KI signal {ai_signal} widerspricht {final_signal}',
                'severity': 'MEDIUM',
                'recommendation': 'Weitere BestÃ¤tigung abwarten'
            })
        # Multi-timeframe consensus contradictions
        if multi_timeframe and isinstance(multi_timeframe, dict):
            mt_primary = multi_timeframe.get('consensus', {}).get('primary')
            if mt_primary == 'BULLISH' and final_signal in ['SELL','STRONG_SELL']:
                contradictions.append({
                    'type': 'MTF_CONTRADICTION',
                    'message': 'MTF Konsens BULLISH aber finales Signal bearish',
                    'severity': 'HIGH',
                    'recommendation': 'Auf Alignment warten'
                })
            if mt_primary == 'BEARISH' and final_signal in ['BUY','STRONG_BUY']:
                contradictions.append({
                    'type': 'MTF_CONTRADICTION',
                    'message': 'MTF Konsens BEARISH aber finales Signal bullish',
                    'severity': 'HIGH',
                    'recommendation': 'Auf Alignment warten'
                })
            # AI vs MTF
            if mt_primary == 'BULLISH' and ai_signal in ['SELL','STRONG_SELL'] and ai_confidence >= 55:
                warnings.append({
                    'type': 'AI_MTF_MISMATCH',
                    'message': 'KI bearish vs MTF bullish',
                    'recommendation': 'SignalqualitÃ¤t prÃ¼fen'
                })
            if mt_primary == 'BEARISH' and ai_signal in ['BUY','STRONG_BUY'] and ai_confidence >= 55:
                warnings.append({
                    'type': 'AI_MTF_MISMATCH',
                    'message': 'KI bullish vs MTF bearish',
                    'recommendation': 'SignalqualitÃ¤t prÃ¼fen'
                })
        # Multi-timeframe pattern majority vs final
        try:
            mtp = pattern_analysis.get('multi_timeframe_patterns', [])
            if mtp:
                b_mt = sum(1 for p in mtp if p.get('signal')=='bullish')
                br_mt = sum(1 for p in mtp if p.get('signal')=='bearish')
                if b_mt > br_mt and final_signal in ['SELL','STRONG_SELL']:
                    contradictions.append({
                        'type': 'MTPATTERN_CONTRADICTION',
                        'message': f'Mehrheit {b_mt} bullish MTF Patterns aber finales Signal bearish',
                        'severity': 'MEDIUM',
                        'recommendation': 'Auf BestÃ¤tigung warten'
                    })
                if br_mt > b_mt and final_signal in ['BUY','STRONG_BUY']:
                    contradictions.append({
                        'type': 'MTPATTERN_CONTRADICTION',
                        'message': f'Mehrheit {br_mt} bearish MTF Patterns aber finales Signal bullish',
                        'severity': 'MEDIUM',
                        'recommendation': 'Auf BestÃ¤tigung warten'
                    })
        except Exception:
            pass
        
        # 6. Overall Risk Assessment
        risk_level = 'LOW'
        if len(contradictions) > 0:
            risk_level = 'VERY_HIGH'
        elif len(warnings) > 2:
            risk_level = 'HIGH'
        elif len(warnings) > 0:
            risk_level = 'MEDIUM'
        
        # 7. Final Trading Recommendation
        trading_action = final_signal
        if len(contradictions) > 0:
            trading_action = 'WAIT'
            confidence_factors.append('âŒ SIGNALE WIDERSPRECHEN SICH - WARTE!')
        elif risk_level in ['HIGH', 'VERY_HIGH']:
            trading_action = 'WAIT'
            confidence_factors.append('âš ï¸ HOHES RISIKO - besseren Einstieg abwarten!')
        else:
            confidence_factors.append('âœ… Signale sind konsistent')
        
        return {
            'trading_action': trading_action,
            'risk_level': risk_level,
            'contradictions': contradictions,
            'warnings': warnings,
            'confidence_factors': confidence_factors,
            'enterprise_ready': len(contradictions) == 0 and risk_level in ['LOW', 'MEDIUM']
        }

    def _generate_trade_setups(self, current_price, tech_analysis, extended_analysis, pattern_analysis, final_score, multi_timeframe=None, regime_data=None):
        """Generate structured long & short trade setups based on technicals, volatility and validation.
        Returns list of up to 8 setups sorted by confidence."""
        setups = []
        try:
            validation = final_score.get('validation', {}) if isinstance(final_score, dict) else {}
            support = tech_analysis.get('support') or current_price * 0.985
            resistance = tech_analysis.get('resistance') or current_price * 1.015
            rsi = tech_analysis.get('rsi', {}).get('rsi', 50)
            trend = tech_analysis.get('trend', {}).get('trend', 'neutral')
            atr_val = extended_analysis.get('atr', {}).get('value') or (current_price * 0.004)
            atr_perc = extended_analysis.get('atr', {}).get('percentage') or (atr_val / current_price * 100)
            fib = extended_analysis.get('fibonacci', {})
            enterprise_ready = validation.get('enterprise_ready', False)
            risk_level = validation.get('risk_level', 'MEDIUM')
            contradictions = validation.get('contradictions', [])
            contradiction_count = len(contradictions)
            patterns = pattern_analysis.get('patterns', [])
            bullish_pattern_present = any(p.get('signal')=='bullish' for p in patterns)
            bearish_pattern_present = any(p.get('signal')=='bearish' for p in patterns)
            # Relaxation metadata container
            relaxation = {
                'trend_original': trend,
                'rsi_original': rsi,
                'relaxed_trend_logic': False,
                'relaxed_rsi_bounds': False,
                'fallback_generated': False,
                'pattern_injected': False
            }

            # Ensure a minimum ATR baseline so targets are not "zu knapp"
            min_atr = max(atr_val, current_price * 0.0025)
            # Helper to widen targets dynamically: structural + ATR extension
            def _structural_targets(direction, entry):
                ext_mult = 8.0  # Erweitert von 4.0 auf 8.0 fÃ¼r echte Swing-Targets
                swing_target = entry + min_atr * ext_mult if direction=='LONG' else entry - min_atr * ext_mult
                return swing_target

            def _confidence(base, adds):
                score = base + sum(adds)
                # ðŸ” STRENGE VALIDIERUNG wie echte Trader
                if contradiction_count: score -= 35  # ErhÃ¶ht von 25 auf 35
                if risk_level in ['HIGH', 'VERY_HIGH']: score -= 25  # ErhÃ¶ht von 15 auf 25
                if atr_perc and atr_perc > 1.4: score -= 15  # ErhÃ¶ht von 8 auf 15
                # ZusÃ¤tzliche Validierung
                if atr_perc and atr_perc > 2.0: score -= 25  # Extreme VolatilitÃ¤t
                if not enterprise_ready: score -= 20  # Keine Enterprise-Validierung
                return max(10, min(95, round(score)))  # Min erhÃ¶ht von 5 auf 10

            def _targets(entry, stop, direction, extra=None):
                risk = (entry - stop) if direction=='LONG' else (stop - entry)  # absolute price risk
                # ðŸ”¥ REALISTISCHE TRADER STOPS - breiter fÃ¼r echte Marktbedingungen
                if risk < min_atr * 1.2:  # Erweitert von 0.8 auf 1.2 fÃ¼r realistischere Stops
                    # Wesentlich breitere Stops wie echte Trader
                    if direction=='LONG':
                        stop = entry - min_atr * 1.2
                    else:
                        stop = entry + min_atr * 1.2
                    risk = (entry - stop) if direction=='LONG' else (stop - entry)
                risk = max(risk, min_atr*1.0)  # Minimum risk erhÃ¶ht

                base = []
                # ðŸŽ¯ REALISTISCHE TP TARGETS - wie echte Trader nutzen
                # Erste Gewinnmitnahme bei 1.5R, dann grÃ¶ÃŸere Swings
                for m in [1.5, 2.5, 4, 6, 8]:  # Entfernt 1R - zu enge, hinzugefÃ¼gt 6R, 8R
                    tp = entry + risk*m if direction=='LONG' else entry - risk*m
                    base.append({'label': f'{m}R', 'price': round(tp,2), 'rr': float(m)})
                # Add structural / swing extension
                swing_ext = _structural_targets(direction, entry)
                base.append({'label':'Swing', 'price': round(swing_ext,2), 'rr': round(abs((swing_ext-entry)/risk),2)})
                if extra:
                    for lbl, lvl in extra:
                        if lvl:
                            rr = (lvl - entry)/risk if direction=='LONG' else (entry - lvl)/risk
                            base.append({'label': lbl, 'price': round(lvl,2), 'rr': round(rr,2)})
                base.sort(key=lambda x: x['rr'])
                # ðŸŽ¯ PROFESSIONAL TARGET FILTERING - breiter gefiltert
                # Keep top distinct targets (remove those closer than 0.8R apart fÃ¼r realistischere AbstÃ¤nde)
                filtered = []
                last_rr = -999
                for t in base:
                    if t['rr'] - last_rr >= 0.8:  # ErhÃ¶ht von 0.4 auf 0.8
                        filtered.append(t)
                        last_rr = t['rr']
                    if len(filtered) >= 5:  # Reduziert von 7 auf 5 Targets
                        break
                return filtered

            # Aggregate multi-timeframe patterns for ranking
            timeframe_weight = {'15m':0.6,'1h':1.0,'4h':1.4,'1d':1.8}
            all_patterns = []
            try:
                base_p = pattern_analysis.get('patterns', [])
                mt_p = pattern_analysis.get('multi_timeframe_patterns', [])
                for p in base_p + mt_p:
                    tf = p.get('timeframe','1h')
                    conf = p.get('confidence',50)
                    w = timeframe_weight.get(tf,1.0)
                    p['_rank_score'] = conf * w
                    all_patterns.append(p)
            except Exception:
                pass
            bull_ranked = sorted([p for p in all_patterns if p.get('signal')=='bullish'], key=lambda x: x.get('_rank_score',0), reverse=True)
            bear_ranked = sorted([p for p in all_patterns if p.get('signal')=='bearish'], key=lambda x: x.get('_rank_score',0), reverse=True)

            # ðŸ” ENHANCED TRADER VALIDIERUNG
            # Mehrstufige Validierung wie professionelle Trader
            setup_quality_filters = {
                'minimum_confidence': 35,  # ErhÃ¶ht von Standard
                'maximum_risk_percent': 3.0,  # Max 3% Risk pro Trade
                'require_multiple_confirmations': True,
                'avoid_high_volatility_entries': atr_perc > 2.5,
                'trend_alignment_required': True
            }
            
            # ðŸš¨ RSI CAUTION NARRATIVE INJECTION
            rsi_caution = self._generate_rsi_caution_narrative(rsi, trend)
            
            # Relaxed trend rule: allow LONG setups if not strongly bearish
            # ðŸš¨ ABER mit zusÃ¤tzlicher Validierung + RSI CAUTION
            trend_validation_passed = False
            if 'bullish' in trend or trend in ['neutral','weak','moderate']:
                if 'bullish' not in trend:
                    relaxation['relaxed_trend_logic'] = True
                    # ZusÃ¤tzliche BestÃ¤tigung erforderlich bei schwachem Trend
                    if rsi > 45 and contradiction_count == 0:
                        trend_validation_passed = True
                else:
                    trend_validation_passed = True
                if trend_validation_passed:  # Nur wenn Validierung bestanden
                    entry_pb = support * 1.003
                    stop_pb = support - atr_val*0.9  # Erweitert von 0.6 auf 0.9
                    risk_pct = round((entry_pb-stop_pb)/entry_pb*100,2)
                    
                    # ðŸ” ZUSÃ„TZLICHE RISK VALIDIERUNG
                    if risk_pct <= setup_quality_filters['maximum_risk_percent']:
                        # Apply RSI caution to rationale
                        base_rationale = 'Multi-validated Einstieg nahe Support mit Professional Risk Management'
                        enhanced_rationale = f"{base_rationale}. {rsi_caution['narrative']}" if rsi_caution['caution_level'] != 'none' else base_rationale
                        
                        setups.append({
                            'id':'L-PB', 'direction':'LONG', 'strategy':'Professional Bullish Pullback',
                            'entry': round(entry_pb,2), 'stop_loss': round(stop_pb,2),
                            'risk_percent': risk_pct,
                            'targets': _targets(entry_pb, stop_pb,'LONG', [
                                ('Resistance', resistance), ('Fib 0.382', fib.get('fib_382')), ('Fib 0.618', fib.get('fib_618'))
                            ]),
                            'confidence': _confidence(55,[15 if enterprise_ready else 5, 8 if rsi<65 else 0, 10 if trend_validation_passed else 0]) - rsi_caution['confidence_penalty'],
                            'conditions': [
                                {'t':'Trend validation','s':'ok' if trend_validation_passed else 'warn'},
                                {'t':f'RSI {rsi:.1f}','s':rsi_caution['signal_quality']},
                                {'t':f'Risk {risk_pct:.1f}%','s':'ok' if risk_pct<=2.0 else 'warn'},
                                {'t':'Enterprise Ready','s':'ok' if enterprise_ready else 'warn'},
                                {'t':'Low Contradictions','s':'ok' if contradiction_count<=1 else 'bad'}
                            ],
                            'validation_score': 'PROFESSIONAL' if all([
                                trend_validation_passed, enterprise_ready, contradiction_count==0, risk_pct<=2.0
                            ]) else 'STANDARD',
                            'rationale': enhanced_rationale,
                            'rsi_caution': rsi_caution,
                            'regime_context': regime_data.get('regime', 'unknown') if regime_data else 'unknown',
                            'justification': {
                                'core_thesis': 'Pullback in intaktem AufwÃ¤rtstrend zurÃ¼ck in Nachfragezone (Support Re-Test).',
                                'confluence': [
                                    'Trend Align (bullish / nicht bearish)',
                                    f'RSI moderat ({rsi:.1f}) -> noch kein Extrem',
                                    'Support strukturell bestÃ¤tigt',
                                    'Risk <= 2% akzeptabel',
                                    'Keine starken WidersprÃ¼che'
                                ],
                                'risk_model': 'Stop unter strukturellem Support + ATR-Puffer (~1.2 ATR).',
                                'invalidations': [
                                    'Tiefer Schlusskurs 1.5% unter Support',
                                    'RSI Divergenz bearish + MACD Curve kippt',
                                    'Volumen Distribution Shift gegen Trend'
                                ],
                                'execution_plan': 'Limit/Stop-Order leicht Ã¼ber Re-Test Candle High, Teilgewinn bei 2R, Rest trailen.'
                            }
                        })

                entry_bo = resistance * 1.0015
                stop_bo = resistance - atr_val
                setups.append({
                    'id':'L-BO', 'direction':'LONG', 'strategy':'Resistance Breakout',
                    'entry': round(entry_bo,2), 'stop_loss': round(stop_bo,2),
                    'risk_percent': round((entry_bo-stop_bo)/entry_bo*100,2),
                    'targets': _targets(entry_bo, stop_bo,'LONG', [
                        ('Fib 0.618', fib.get('fib_618')), ('Fib 0.786', fib.get('fib_786'))
                    ]),
                    'confidence': _confidence(48,[15 if enterprise_ready else 5, 6 if rsi<70 else -4]),
                    'conditions': [
                        {'t':'Break Ã¼ber Resistance','s':'ok'},
                        {'t':'Momentum intakt','s':'ok'},
                        {'t':'Kein starker Widerspruch','s':'ok' if contradiction_count==0 else 'bad'}
                    ],
                    'rationale':'Ausbruch nutzt Momentum Beschleunigung',
                    'justification': {
                        'core_thesis': 'Preis akzeptiert oberhalb vorherigen Angebotslevels -> mÃ¶gliche Preisentfaltung / Imbalance Fill.',
                        'confluence': [
                            'Break + Close Ã¼ber Resistance',
                            'Momentum bestÃ¤tigt (MACD Curve / Volumen Spike mÃ¶glich)',
                            f'RSI noch unter Extremzone ({rsi:.1f})',
                            'Keine akuten Widerspruchs-Signale'
                        ],
                        'risk_model': 'Stop unter ehemaligem Widerstand (jetzt potentielle UnterstÃ¼tzung) + ATR-Schutz.',
                        'invalidations': [
                            'RÃ¼ckfall & Close zurÃ¼ck unter Level',
                            'Low-Volume Fakeout (Volumen unter Durchschnitt)',
                            'Bearish Engulfing direkt nach Break'
                        ],
                        'execution_plan': 'Stop-Order geringfÃ¼gig Ã¼ber Break-Level, Confirm Candle abwarten, dann Staffeln der TPs.'
                    }
                })

                # Pattern Confirmation LONG
                if bull_ranked:
                    top_b = bull_ranked[0]
                    tfb = top_b.get('timeframe','1h')
                    entry_pc = current_price * 1.001 if current_price < resistance else resistance*1.001
                    stop_pc = entry_pc - atr_val*0.8
                    setups.append({
                        'id':'L-PC', 'direction':'LONG', 'strategy':'Pattern Confirmation',
                        'entry': round(entry_pc,2), 'stop_loss': round(stop_pc,2),
                        'risk_percent': round((entry_pc-stop_pc)/entry_pc*100,2),
                        'targets': _targets(entry_pc, stop_pc,'LONG', [('Resistance', resistance)]),
                        'confidence': _confidence(52,[min(18,int(top_b.get('confidence',50)/3)), 5 if 'bullish' in trend else 0]),
                        'conditions': [
                            {'t':f'Pattern {top_b.get("name","?")}@{tfb}','s':'ok'},
                            {'t': f"MACD Curve {tech_analysis.get('macd', {}).get('curve_direction')}", 's':'ok'},
                            {'t':f'RSI {rsi:.1f}','s':'ok' if rsi<70 else 'warn'}
                        ],
                        'pattern_timeframe': tfb,
                        'pattern_refs':[f"{top_b.get('name','?')}@{tfb}"],
                        'source_signals':['pattern','macd_curve','rsi'],
                        'rationale':'Bullisches Muster bestÃ¤tigt Fortsetzung',
                        'justification': {
                            'core_thesis': f'BestÃ¤tigtes bullisches {top_b.get("name","Pattern")} auf {tfb} mit Momentum-UnterstÃ¼tzung.',
                            'confluence': [
                                'Muster + Trend nicht dagegen',
                                'MACD Curve positiv',
                                f'RSI gesund ({rsi:.1f})',
                                'Kein unmittelbarer Widerstand direkt vor Entry'
                            ],
                            'risk_model': 'Stop unter Pattern-Struktur + ATR-Puffer.',
                            'invalidations': [
                                'Bruch Pattern Low',
                                'Volumen-Absorption am Entry',
                                'Bearish Divergenz entsteht'
                            ],
                            'execution_plan': 'Einstieg nach BestÃ¤tigungs-Close, TeilverkÃ¤ufe an 2R / Strukturzonen.'
                        }
                    })

                # Momentum Continuation LONG
                macd_curve = tech_analysis.get('macd',{}).get('curve_direction','neutral')
                if 'bullish' in macd_curve and rsi > 55:
                    entry_momo = current_price * 1.0005
                    stop_momo = entry_momo - atr_val
                    setups.append({
                        'id':'L-MOMO', 'direction':'LONG', 'strategy':'Momentum Continuation',
                        'entry': round(entry_momo,2), 'stop_loss': round(stop_momo,2),
                        'risk_percent': round((entry_momo-stop_momo)/entry_momo*100,2),
                        'targets': _targets(entry_momo, stop_momo,'LONG', [('Resistance', resistance)]),
                        'confidence': _confidence(50,[10 if rsi>60 else 5, 6]),
                        'conditions':[{'t':'MACD Curve bullish','s':'ok'},{'t':f'RSI {rsi:.1f}','s':'ok'},{'t':'Trend nicht bearish','s':'ok' if 'bearish' not in trend else 'warn'}],
                        'source_signals':['macd_curve','rsi','trend'],
                        'rationale':'Momentum Fortsetzung basierend auf MACD Bogen + RSI',
                        'justification': {
                            'core_thesis': 'Fortsetzung nach impulsiver Expansionsphase ohne ErschÃ¶pfungssignale.',
                            'confluence': [ 'Bull MACD Curve', f'RSI > 55 ({rsi:.1f})', 'Keine bearishe Divergenz', 'Trend nicht kontra' ],
                            'risk_model': 'Stop unter kurzfristigem Momentum Pivot (letzte Mini-Konsolidierung).',
                            'invalidations': [ 'Momentum Collapse (starker Gegen-Volumen Spike)', 'RSI fÃ¤llt unter 48', 'MACD Curve dreht sofort bearisch' ],
                            'execution_plan': 'Market/Limit Entry in leichte Pullback-Candle, aggressives Trail nach 2R.'
                        }
                    })

                # Support Rejection LONG
                if support and (current_price - support)/current_price*100 < 1.2 and bull_ranked:
                    top_b2 = bull_ranked[0]
                    tfb2 = top_b2.get('timeframe','1h')
                    entry_rej = support * 1.004
                    stop_rej = support - atr_val*0.7
                    setups.append({
                        'id':'L-REJ', 'direction':'LONG', 'strategy':'Support Rejection',
                        'entry': round(entry_rej,2), 'stop_loss': round(stop_rej,2),
                        'risk_percent': round((entry_rej-stop_rej)/entry_rej*100,2),
                        'targets': _targets(entry_rej, stop_rej,'LONG', [('Resistance', resistance)]),
                        'confidence': _confidence(48,[8, min(14,int(top_b2.get('confidence',50)/4))]),
                        'conditions':[{'t':'Nahe Support','s':'ok'},{'t':'Bull Pattern','s':'ok'},{'t':'VolatilitÃ¤t ok','s':'ok' if atr_perc<1.5 else 'warn'}],
                        'pattern_timeframe': tfb2,
                        'pattern_refs':[f"{top_b2.get('name','?')}@{tfb2}"],
                        'source_signals':['support','pattern'],
                        'rationale':'Rejection nahe Support + bullisches Muster',
                        'justification': {
                            'core_thesis': 'Agressive KÃ¤ufer verteidigen Key-Support -> frische Nachfrage bestÃ¤tigt.',
                            'confluence': [ 'Wick Rejection / schnelle ZurÃ¼ckweisung', 'Bull Pattern aktiv', 'VolatilitÃ¤t moderat', 'Support mehrfach getestet' ],
                            'risk_model': 'Stop unter Rejection-Trigger + Struktur Low.',
                            'invalidations': [ 'Close unter Support', 'Volumen trocknet aus', 'Pattern verliert Struktur' ],
                            'execution_plan': 'Entry nach Rejection Candle High Break, konservatives Teilziel bei 2R.'
                        }
                    })

            # Relax RSI: previously 32 -> now 35
            if rsi < 32:
                relaxation['relaxed_rsi_bounds'] = True  # because condition used strict threshold but we mark band widening below
            if rsi < 35:
                entry_mr = current_price*0.998
                stop_mr = entry_mr - atr_val*0.9
                setups.append({
                    'id':'L-MR', 'direction':'LONG', 'strategy':'RSI Mean Reversion',
                    'entry': round(entry_mr,2), 'stop_loss': round(stop_mr,2),
                    'risk_percent': round((entry_mr-stop_mr)/entry_mr*100,2),
                    'targets': _targets(entry_mr, stop_mr,'LONG', [('Resistance', resistance)]),
                    'confidence': _confidence(42,[10 if rsi<28 else 4]),
                    'conditions': [ {'t':f'RSI {rsi:.1f}','s':'ok'}, {'t':'Trend nicht stark bearish','s':'ok' if 'bearish' not in trend else 'warn'} ],
                    'rationale':'Ãœberverkaufte Bedingung -> Rebound Szenario',
                    'justification': {
                        'core_thesis': 'Kurzfristige Ãœbertreibung (Oversold) mit mean reversion Potenzial.',
                        'confluence': [ f'RSI < 35 ({rsi:.1f})', 'Trend nicht stark bearish', 'Keine massiven Distribution-Spikes' ],
                        'risk_model': 'Stop unter lokaler Exhaustion / Spike Low.',
                        'invalidations': [ 'Weitere starke Long Liquidations', 'RSI fÃ¤llt unter 20 ohne Reaktionsvolumen' ],
                        'execution_plan': 'Scaling Entry in 2 Tranchen, enges Management, frÃ¼hes Secure bei 1.5-2R.'
                    }
                })

            # SHORT strategies (relax: allow if not strongly bullish)
            # ðŸš¨ ABER mit professioneller Validierung
            short_trend_validation_passed = False
            if 'bearish' in trend or trend in ['neutral','weak','moderate']:
                if 'bearish' not in trend:
                    relaxation['relaxed_trend_logic'] = True
                    # ZusÃ¤tzliche BestÃ¤tigung erforderlich bei schwachem Trend
                    if rsi < 55 and contradiction_count == 0:
                        short_trend_validation_passed = True
                else:
                    short_trend_validation_passed = True
                    
                if short_trend_validation_passed:  # Nur wenn Validierung bestanden
                    entry_pbs = resistance*0.997
                    stop_pbs = resistance + atr_val*0.9  # Erweitert von 0.6 auf 0.9
                    risk_pct_short = round((stop_pbs-entry_pbs)/entry_pbs*100,2)
                    
                    # ðŸ” ZUSÃ„TZLICHE RISK VALIDIERUNG fÃ¼r SHORT
                    if risk_pct_short <= setup_quality_filters['maximum_risk_percent']:
                        setups.append({
                            'id':'S-PB', 'direction':'SHORT', 'strategy':'Professional Bearish Pullback',
                            'entry': round(entry_pbs,2), 'stop_loss': round(stop_pbs,2),
                            'risk_percent': risk_pct_short,
                            'targets': _targets(entry_pbs, stop_pbs,'SHORT', [('Support', support), ('Fib 0.382', fib.get('fib_382'))]),
                            'confidence': _confidence(55,[15 if enterprise_ready else 5, 8 if rsi>35 else 0, 10 if short_trend_validation_passed else 0]),
                            'conditions': [ 
                                {'t':'Short Trend validation','s':'ok' if short_trend_validation_passed else 'warn'}, 
                                {'t':f'RSI {rsi:.1f}','s':'ok' if rsi>35 else 'warn'},
                                {'t':f'Risk {risk_pct_short:.1f}%','s':'ok' if risk_pct_short<=2.0 else 'warn'},
                                {'t':'Enterprise Ready','s':'ok' if enterprise_ready else 'warn'}
                            ],
                            'validation_score': 'PROFESSIONAL' if all([
                                short_trend_validation_passed, enterprise_ready, contradiction_count==0, risk_pct_short<=2.0
                            ]) else 'STANDARD',
                            'rationale':'Multi-validated Pullback an Widerstand mit Professional Risk Management',
                            'justification': {
                                'core_thesis': 'Pullback in aktiven AbwÃ¤rtstrend zurÃ¼ck in Angebotszone (Lower High Opportunity).',
                                'confluence': [ 'Trend Align bearish / nicht bullisch', f'RSI neutral ({rsi:.1f}) -> Raum fÃ¼r AbwÃ¤rtsbewegung', 'Widerstand bestÃ¤tigt', 'Risk <= 2%' ],
                                'risk_model': 'Stop Ã¼ber strukturellem Swing High + ATR-Puffer.',
                                'invalidations': [ 'Starker Close Ã¼ber Widerstand', 'Bullische Volume Absorption', 'Momentum Shift (MACD Curve bullisch)' ],
                                'execution_plan': 'Entry mittels Limit nahe Pullback Spitze, Teilgewinn bei 2R.'
                            }
                        })

                entry_bd = support*0.9985
                stop_bd = support + atr_val
                setups.append({
                    'id':'S-BD', 'direction':'SHORT', 'strategy':'Support Breakdown',
                    'entry': round(entry_bd,2), 'stop_loss': round(stop_bd,2),
                    'risk_percent': round((stop_bd-entry_bd)/entry_bd*100,2),
                    'targets': _targets(entry_bd, stop_bd,'SHORT', [('Fib 0.236', fib.get('fib_236'))]),
                    'confidence': _confidence(48,[14 if enterprise_ready else 4, 5]),
                    'conditions': [ {'t':'Bruch unter Support','s':'ok'}, {'t':'Keine bull. Divergenz','s':'ok'} ],
                    'rationale':'Beschleunigter Momentum-Handel beim Support-Bruch',
                    'justification': {
                        'core_thesis': 'Akzeptanz unter Key-Support -> mÃ¶glicher Preis-Entleerungsbereich (Liquidity Vacuum).',
                        'confluence': [ 'Bruch + Close unter Support', 'Keine bullische Divergenz', 'Volumen nicht kollabierend', 'Trend nicht bullisch' ],
                        'risk_model': 'Stop Ã¼ber Breakdown Level + ATR.',
                        'invalidations': [ 'Starker Reclaim Support', 'Volumen Divergenz (fallender Preis, fallendes Volumen)', 'Bull Pattern formt sich sofort' ],
                        'execution_plan': 'Entry Ã¼ber Stop-Order unter BestÃ¤tigungscandle, schnelles Tightening nach initialem Flush.'
                    }
                })

                # Pattern Confirmation SHORT
                if bear_ranked:
                    top_s = bear_ranked[0]
                    tfs = top_s.get('timeframe','1h')
                    entry_ps = current_price * 0.999 if current_price > support else support*0.999
                    stop_ps = entry_ps + atr_val*0.8
                    setups.append({
                        'id':'S-PC', 'direction':'SHORT', 'strategy':'Pattern Confirmation',
                        'entry': round(entry_ps,2), 'stop_loss': round(stop_ps,2),
                        'risk_percent': round((stop_ps-entry_ps)/entry_ps*100,2),
                        'targets': _targets(entry_ps, stop_ps,'SHORT', [('Support', support)]),
                        'confidence': _confidence(52,[min(18,int(top_s.get('confidence',50)/3)), 5 if 'bearish' in trend else 0]),
                        'conditions': [
                            {'t':f'Pattern {top_s.get("name","?")}@{tfs}','s':'ok'},
                            {'t': f"MACD Curve {tech_analysis.get('macd', {}).get('curve_direction')}", 's':'ok'},
                            {'t':f'RSI {rsi:.1f}','s':'ok' if rsi>30 else 'warn'}
                        ],
                        'pattern_timeframe': tfs,
                        'pattern_refs':[f"{top_s.get('name','?')}@{tfs}"],
                        'source_signals':['pattern','macd_curve','rsi'],
                        'rationale':'Bearishes Muster bestÃ¤tigt Fortsetzung',
                        'justification': {
                            'core_thesis': f'BestÃ¤tigtes bearisches {top_s.get("name","Pattern")} auf {tfs} mit Momentum-UnterstÃ¼tzung.',
                            'confluence': [ 'Muster + Trend nicht bullish', 'MACD Curve negativ', f'RSI intakt ({rsi:.1f})', 'Kein unmittelbarer Support darunter' ],
                            'risk_model': 'Stop Ã¼ber Pattern-Struktur + ATR-Puffer.',
                            'invalidations': [ 'Close zurÃ¼ck in Pattern', 'Volumen Absorption durch KÃ¤ufer', 'Bull Divergenz bildet sich' ],
                            'execution_plan': 'Entry nach BestÃ¤tigungs-Close, Teilziel 2R, Rest laufen bis Strukturbruch.'
                        }
                    })

                # Momentum Continuation SHORT
                macd_curve_s = tech_analysis.get('macd',{}).get('curve_direction','neutral')
                if 'bearish' in macd_curve_s and rsi < 45:
                    entry_momo_s = current_price * 0.9995
                    stop_momo_s = entry_momo_s + atr_val
                    setups.append({
                        'id':'S-MOMO', 'direction':'SHORT', 'strategy':'Momentum Continuation',
                        'entry': round(entry_momo_s,2), 'stop_loss': round(stop_momo_s,2),
                        'risk_percent': round((stop_momo_s-entry_momo_s)/entry_momo_s*100,2),
                        'targets': _targets(entry_momo_s, stop_momo_s,'SHORT', [('Support', support)]),
                        'confidence': _confidence(50,[10 if rsi<40 else 5, 6]),
                        'conditions':[{'t':'MACD Curve bearish','s':'ok'},{'t':f'RSI {rsi:.1f}','s':'ok'},{'t':'Trend nicht bullish','s':'ok' if 'bullish' not in trend else 'warn'}],
                        'source_signals':['macd_curve','rsi','trend'],
                        'rationale':'Momentum Fortsetzung basierend auf MACD Bogen + RSI',
                        'justification': {
                            'core_thesis': 'Fortlaufende AbwÃ¤rts-Beschleunigung ohne deutliche Gegenreaktion (Momentum Squeeze).',
                            'confluence': [ 'Bear MACD Curve', f'RSI < 45 ({rsi:.1f})', 'Kein aggressives Buying', 'Trend nicht kontra' ],
                            'risk_model': 'Stop Ã¼ber kurzfristigem Momentum Pivot / Mini Range High.',
                            'invalidations': [ 'Sharp Reversal Candle mit Volumen', 'RSI Erholung > 52', 'Pattern Flip bullisch' ],
                            'execution_plan': 'Entry in Micro-Pullback, aggressives Trailing nach 2R.'
                        }
                    })

                # Resistance Rejection SHORT
                if resistance and (resistance - current_price)/current_price*100 < 1.2 and bear_ranked:
                    top_s2 = bear_ranked[0]
                    tfs2 = top_s2.get('timeframe','1h')
                    entry_rej_s = resistance * 0.996
                    stop_rej_s = resistance + atr_val*0.7
                    setups.append({
                        'id':'S-REJ', 'direction':'SHORT', 'strategy':'Resistance Rejection',
                        'entry': round(entry_rej_s,2), 'stop_loss': round(stop_rej_s,2),
                        'risk_percent': round((stop_rej_s-entry_rej_s)/entry_rej_s*100,2),
                        'targets': _targets(entry_rej_s, stop_rej_s,'SHORT', [('Support', support)]),
                        'confidence': _confidence(48,[8, min(14,int(top_s2.get('confidence',50)/4))]),
                        'pattern_timeframe': tfs2,
                        'pattern_refs':[f"{top_s2.get('name','?')}@{tfs2}"],
                        'source_signals':['resistance','pattern'],
                        'rationale':'Rejection nahe Resistance + bearisches Muster',
                        'justification': {
                            'core_thesis': 'Verkaufsabsorptionszone wird verteidigt -> Angebot dominiert weiter.',
                            'confluence': [ 'Mehrfaches Rejection Verhalten', 'Bear Pattern aktiv', 'VolatilitÃ¤t moderat', 'Kein Momentum Reversal' ],
                            'risk_model': 'Stop Ã¼ber Rejection Wick + ATR.',
                            'invalidations': [ 'Close Ã¼ber Level', 'Volumen Shift pro KÃ¤ufer', 'Pattern Struktur bricht' ],
                            'execution_plan': 'Entry nach BestÃ¤tigung Reversal Candle, konservatives Ziel 2R, Rest strukturbasiert.'
                        }
                    })

            # Relax RSI upper band: previously 68 -> now 65
            if rsi > 68:
                relaxation['relaxed_rsi_bounds'] = True
            if rsi > 65:
                entry_mrs = current_price*1.002
                stop_mrs = entry_mrs + atr_val*0.9
                setups.append({
                    'id':'S-MR', 'direction':'SHORT', 'strategy':'RSI Mean Reversion',
                    'entry': round(entry_mrs,2), 'stop_loss': round(stop_mrs,2),
                    'risk_percent': round((stop_mrs-entry_mrs)/entry_mrs*100,2),
                    'targets': _targets(entry_mrs, stop_mrs,'SHORT', [('Support', support)]),
                    'confidence': _confidence(42,[10 if rsi>72 else 4]),
                    'conditions': [ {'t':f'RSI {rsi:.1f}','s':'ok'}, {'t':'Trend nicht stark bullish','s':'ok' if 'bullish' not in trend else 'warn'} ],
                    'rationale':'Ãœberkaufte Bedingung -> RÃ¼cksetzer / Mean Reversion',
                    'justification': {
                        'core_thesis': 'Kurzfristige Ãœberdehnung (Overbought) lÃ¤dt Mean-Reversion Bewegung ein.',
                        'confluence': [ f'RSI > 65 ({rsi:.1f})', 'Keine frische Breakout Momentum Candle', 'Trend nicht stark bullish' ],
                        'risk_model': 'Stop Ã¼ber Exhaustion Hoch.',
                        'invalidations': [ 'Fortgesetzter Momentum Squeeze', 'RSI > 78 mit Volumen Expansion' ],
                        'execution_plan': 'Entry gestaffelt, schneller Stop Tightening, conservative Targets.'
                    }
                })

            # Pattern injected setups if patterns present & not enough direction from trend
            if bullish_pattern_present and len([s for s in setups if s['direction']=='LONG']) < 2:
                entry_pat = current_price*1.001
                stop_pat = current_price - atr_val
                setups.append({
                    'id':'L-PAT', 'direction':'LONG', 'strategy':'Pattern Boost Long',
                    'entry': round(entry_pat,2), 'stop_loss': round(stop_pat,2),
                    'risk_percent': round((entry_pat-stop_pat)/entry_pat*100,2),
                    'targets': _targets(entry_pat, stop_pat,'LONG', [('Resistance', resistance)]),
                    'confidence': 55,
                    'conditions': [{'t':'Bullish Pattern','s':'ok'}],
                    'rationale':'Bullish Chart Pattern aktiviert (relaxed)'
                })
                relaxation['pattern_injected'] = True
            if bearish_pattern_present and len([s for s in setups if s['direction']=='SHORT']) < 2:
                entry_pats = current_price*0.999
                stop_pats = current_price + atr_val
                setups.append({
                    'id':'S-PAT', 'direction':'SHORT', 'strategy':'Pattern Boost Short',
                    'entry': round(entry_pats,2), 'stop_loss': round(stop_pats,2),
                    'risk_percent': round((stop_pats-entry_pats)/entry_pats*100,2),
                    'targets': _targets(entry_pats, stop_pats,'SHORT', [('Support', support)]),
                    'confidence': 55,
                    'conditions': [{'t':'Bearish Pattern','s':'ok'}],
                    'rationale':'Bearish Chart Pattern aktiviert (relaxed)'
                })
                relaxation['pattern_injected'] = True

            # Fallback generic setups if still too few (ensure at least 2 directions)
            if len(setups) < 2:
                relaxation['fallback_generated'] = True
                generic_risk = max(atr_val, current_price*0.003)
                # Generic LONG
                entry_gl = current_price
                stop_gl = entry_gl - generic_risk
                setups.append({
                    'id':'L-FB', 'direction':'LONG', 'strategy':'Generic Long',
                    'entry': round(entry_gl,2), 'stop_loss': round(stop_gl,2),
                    'risk_percent': round((entry_gl-stop_gl)/entry_gl*100,2),
                    'targets': _targets(entry_gl, stop_gl,'LONG', [('Resistance', resistance)]),
                    'confidence': 45,
                    'conditions': [{'t':'Fallback','s':'info'}],
                    'rationale':'Fallback Long Setup (relaxed)'
                })
                # Generic SHORT
                entry_gs = current_price
                stop_gs = entry_gs + generic_risk
                setups.append({
                    'id':'S-FB', 'direction':'SHORT', 'strategy':'Generic Short',
                    'entry': round(entry_gs,2), 'stop_loss': round(stop_gs,2),
                    'risk_percent': round((stop_gs-entry_gs)/entry_gs*100,2),
                    'targets': _targets(entry_gs, stop_gs,'SHORT', [('Support', support)]),
                    'confidence': 45,
                    'conditions': [{'t':'Fallback','s':'info'}],
                    'rationale':'Fallback Short Setup (relaxed)'
                })

            # Add probability estimates (heuristic) per setup using calibrated final score
            try:
                base_prob = 0.5
                if isinstance(final_score, dict):
                    cp = final_score.get('calibrated_probability')
                    if isinstance(cp, (int,float)):
                        base_prob = max(0.01, min(0.99, cp/100.0))
                for s in setups:
                    conf = s.get('confidence',50)/100.0
                    if s.get('direction') == 'LONG':
                        p = base_prob + (conf-0.5)*0.35
                    else:
                        p = (1-base_prob) + (conf-0.5)*0.35
                    p = max(0.02, min(0.98, p))
                    s['probability_estimate_pct'] = round(p*100,2)
                    s['probability_note'] = 'Heuristisch (Score + Confidence). Nicht kalibriert.'
            except Exception:
                pass

            for s in setups:
                if s.get('targets'):
                    s['primary_rr'] = s['targets'][0]['rr']
            
            # ðŸŽ¯ INTEGRATE CHART PATTERN TRADES
            pattern_trades = []
            if pattern_analysis and pattern_analysis.get('patterns'):
                pattern_trades = ChartPatternTrader.generate_pattern_trades(
                    pattern_analysis['patterns'], 
                    current_price, 
                    atr_val,
                    support, 
                    resistance
                )
                print(f"ðŸ“Š Generated {len(pattern_trades)} pattern-based trades")
            
            # Combine traditional setups with pattern trades
            all_setups = setups + pattern_trades
            all_setups.sort(key=lambda x: x.get('confidence', 50), reverse=True)
            trimmed = all_setups[:12]  # Erweitert von 8 auf 12 fÃ¼r Pattern Trades
            
            # Attach relaxation meta to first element for transparency
            if trimmed:
                trimmed[0]['relaxation_meta'] = relaxation
            return trimmed
        except Exception as e:
            print(f"Trade setup generation error: {e}")
            return []

# Initialize the master analyzer
master_analyzer = MasterAnalyzer()

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
                             <div class="metric-value ${getIndicatorColor(extended.stochastic.signal)}" style="font-size:1.05rem;">${extended.stochastic.signal.toUpperCase()}</div>
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