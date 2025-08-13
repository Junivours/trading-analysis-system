import os, time, json, hashlib, math, logging
import numpy as np
import requests
from datetime import datetime
from core.technical_analysis import TechnicalAnalysis
from core.advanced_technical import AdvancedTechnicalAnalysis
from core.patterns import AdvancedPatternDetector, ChartPatternTrader
from core.position import PositionManager
from core.ai import AdvancedJAXAI
from core.ai_backends import get_ai_system
from core.binance_client import BinanceClient
from core.liquidation import LiquidationCalculator
from core.profiling import SymbolBehaviorProfiler

class MasterAnalyzer:
    def __init__(self):
        self.technical_analysis = TechnicalAnalysis()
        self.pattern_detector = AdvancedPatternDetector()
        self.position_manager = PositionManager()
        self.liquidation_calc = LiquidationCalculator()
        self.binance_client = BinanceClient()
        # Pluggable AI backend: 'jax' (default), 'torch', 'tf', 'ensemble' via AI_BACKEND env
        try:
            backend = os.getenv('AI_BACKEND', 'jax')
            self.ai_system = get_ai_system(backend)
        except Exception:
            self.ai_system = AdvancedJAXAI()
        self.symbol_profiler = SymbolBehaviorProfiler()
        self.weights = {'technical':0.70,'patterns':0.20,'ai':0.10}
        self.logger = logging.getLogger("master_analyzer")
        # Default execution model
        self.default_fee_bps = float(os.getenv('DEFAULT_FEE_BPS', '6'))  # 0.06% per side
        self.default_slip_bps = float(os.getenv('DEFAULT_SLIP_BPS', '2'))  # 0.02% slippage per side

    # ======== Exchange precision helpers ========
    def _get_symbol_filters(self, symbol: str):
        try:
            f = self.binance_client.get_symbol_filters(symbol)
            return f or {}
        except Exception:
            return {}
    def _round_to_step(self, value: float, step: float) -> float:
        try:
            if not step or step <= 0:
                return float(value)
            return float(round(value / step) * step)
        except Exception:
            return float(value)
    def _round_price(self, symbol: str, price: float) -> float:
        f = self._get_symbol_filters(symbol)
        tick = float(f.get('tickSize') or 0.0)
        if tick and tick > 0:
            return round(self._round_to_step(price, tick), 8)
        # Fallback: round to 2 decimals for USD quotes
        try:
            return round(float(price), 2)
        except Exception:
            return float(price)

    # ============================= PUBLIC API METHODS ============================= #
    def run_backtest(self, symbol, interval='1h', limit=500, fee_bps=None, slip_bps=None):
        """Lightweight RSI mean-reversion backtest (educational)."""
        try:
            if fee_bps is None:
                fee_bps = self.default_fee_bps
            if slip_bps is None:
                slip_bps = self.default_slip_bps
            fee_pct = max(0.0, float(fee_bps) / 10000.0)
            slip_pct = max(0.0, float(slip_bps) / 10000.0)
            interval = (interval or '1h').lower()
            try:
                limit = int(limit)
            except Exception:
                limit = 500
            limit = max(100, min(limit, 1000))
            min_required = 120 if interval not in ('4h','1d') else 150
            if limit < min_required:
                limit = min_required
            klines = self.technical_analysis.get_candle_data(symbol, limit=limit, interval=interval)
            if not klines or len(klines) < min_required:
                self.logger.info("Fallback direct klines fetch for backtest")
                url = f"https://api.binance.com/api/v3/klines?symbol={symbol.upper()}&interval={interval}&limit={limit}"
                r = requests.get(url, timeout=10)
                klines = r.json()
            if not isinstance(klines, list) or len(klines) < min_required:
                have = len(klines) if isinstance(klines, list) else 0
                return {'error': f'Not enough historical data: have {have}, need >= {min_required}', 'have': have, 'need': min_required, 'interval': interval}
            if isinstance(klines[0], list):
                candles = [{'time': k[0], 'open': float(k[1]), 'high': float(k[2]), 'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])} for k in klines]
            else:
                candles = klines
            closes = np.array([c['close'] for c in candles], dtype=float)
            highs = np.array([c['high'] for c in candles], dtype=float)
            lows = np.array([c['low'] for c in candles], dtype=float)
            times = [c['time'] for c in candles]
            period = 14
            if len(closes) <= period + 5:
                return {'error': 'Series too short'}
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
            rsi_vals = np.concatenate([[50]*(len(closes)-1-len(rsi_vals)), rsi_vals])
            rsi_vals = np.concatenate([[50], rsi_vals])
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
                        # Enter LONG at worse price due to slippage
                        entry_raw = price * (1 + slip_pct)
                        entry = self._round_price(symbol, entry_raw)
                        stop_raw = (price - 2 * atr_vals[i]) if atr_vals[i] > 0 else price * 0.98
                        stop = self._round_price(symbol, stop_raw)
                        position = {'entry_price': entry, 'entry_time': times[i], 'stop': stop}
                else:
                    stop_hit = price <= position['stop']
                    take = rsi_vals[i] > 55
                    if stop_hit or take:
                        # Determine exit price with slippage adverse on exit
                        if stop_hit:
                            exit_price = self._round_price(symbol, position['stop'] * (1 - slip_pct))
                        else:
                            # Take profit at current price; selling -> adverse slippage
                            exit_price = self._round_price(symbol, price * (1 - slip_pct))
                        entry_price = float(position['entry_price'])
                        gross_ret = (exit_price - entry_price) / entry_price
                        # Subtract taker fees both sides
                        net_ret = gross_ret - (2 * fee_pct)
                        equity *= (1 + net_ret)
                        ret_pct = net_ret * 100.0
                        trades.append({'entry_time': position['entry_time'], 'exit_time': times[i], 'entry': round(entry_price,6), 'exit': round(exit_price,6), 'return_pct': round(ret_pct,2), 'outcome': 'win' if ret_pct > 0 else 'loss'})
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
            buy_hold_return = (closes[-1]/closes[0]-1)*100 if len(closes) > 1 else 0
            relative_outperformance = total_ret - buy_hold_return
            max_consec_wins = 0; max_consec_losses = 0; cur_wins=0; cur_losses=0
            for t in trades:
                if t['outcome']=='win': cur_wins+=1; cur_losses=0
                else: cur_losses+=1; cur_wins=0
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
                    'risk_adjusted_return_ratio': risk_adjusted_return,
                    'fee_bps': float(fee_bps),
                    'slip_bps': float(slip_bps)
                },
                'trades': trades[-120:],
                'equity_curve': equity_curve[-250:],
                'disclaimer': 'Backtest berücksichtigt Gebühren/Slippage vereinfacht. Vergangene Performance garantiert keine Zukunft.'
            }
        except Exception as e:
            return {'error': str(e)}

    def run_vector_scalp_backtest(self, symbol, interval='5m', limit=720, fee_bps=None, slip_bps=None):
        """Backtest the Vector Candle Scalp strategy on a lower timeframe.
        Detection: body_ratio>=0.7, range>1.2x ATR14, volume>1.5x SMA20.
        Entry: 62% retrace toward origin (bounded by mid-body). SL: beyond opposite wick + 0.6*ATR.
        TPs: 1.5R, 2.5R, 3.5R. Conservative intra-candle resolution: SL checked before TP.
        """
        try:
            if fee_bps is None:
                fee_bps = self.default_fee_bps
            if slip_bps is None:
                slip_bps = self.default_slip_bps
            fee_pct = max(0.0, float(fee_bps) / 10000.0)
            slip_pct = max(0.0, float(slip_bps) / 10000.0)
            interval = (interval or '5m').lower()
            try:
                limit = int(limit)
            except Exception:
                limit = 720
            limit = max(200, min(limit, 2000))
            candles = self.technical_analysis.get_candle_data(symbol, interval=interval, limit=limit)
            if not candles or len(candles) < 120:
                return {'error': 'Not enough data for vector scalp backtest', 'have': len(candles) if candles else 0, 'need': 120}
            opens = np.array([c['open'] for c in candles], dtype=float)
            highs = np.array([c['high'] for c in candles], dtype=float)
            lows = np.array([c['low'] for c in candles], dtype=float)
            closes = np.array([c['close'] for c in candles], dtype=float)
            volumes = np.array([c.get('volume', 0.0) for c in candles], dtype=float)
            times = [c.get('time') for c in candles]
            # ATR14 via RMA
            tr = np.maximum(highs[1:], closes[:-1]) - np.minimum(lows[1:], closes[:-1])
            atr = np.zeros_like(highs)
            if len(tr) >= 14:
                rma = np.zeros_like(tr)
                rma[13] = tr[:14].mean()
                for i in range(14, len(tr)):
                    rma[i] = (rma[i-1]*13 + tr[i]) / 14
                atr[1+13:] = rma[13:]
                atr[:1+13] = rma[13]
            else:
                atr[:] = (highs - lows).mean()
            # Volume SMA20
            vol_sma = np.zeros_like(volumes)
            if len(volumes) >= 20:
                for i in range(19, len(volumes)):
                    vol_sma[i] = volumes[i-19:i+1].mean()
                vol_sma[:19] = volumes[:19].mean() if len(volumes)>=1 else 0
            else:
                vol_sma[:] = volumes.mean() if len(volumes) else 0
            # Detect all vector candles
            candidates = []
            for idx in range(20, len(candles)):
                o = opens[idx]; h = highs[idx]; l = lows[idx]; c = closes[idx]
                rng = max(1e-9, h - l); body = abs(c - o)
                body_ratio = body / rng if rng else 0
                atr_k = atr[idx] if atr[idx] > 0 else rng
                v_sma = vol_sma[idx] if vol_sma[idx] > 0 else (volumes.mean() if volumes.size>0 else 0)
                is_spike = (volumes[idx] > 1.5 * v_sma) if v_sma else False
                is_range_big = (rng > 1.2 * atr_k)
                if body_ratio >= 0.7 and is_range_big and is_spike:
                    direction = 'bull' if c > o else 'bear'
                    candidates.append({'idx': idx, 'o': float(o), 'h': float(h), 'l': float(l), 'c': float(c), 'rng': float(rng), 'atr': float(atr_k), 'dir': direction})
            if not candidates:
                return {'symbol': symbol.upper(), 'interval': interval, 'limit': limit, 'strategy': 'Vector Candle Scalp', 'metrics': {'total_trades': 0}, 'trades': [], 'note': 'No vector candles detected'}
            # Simulation
            open_pos = None
            trades = []
            equity_r = 0.0
            equity_r_net = 0.0
            max_dd_r = 0.0
            max_dd_r_net = 0.0
            peak_r = 0.0
            peak_r_net = 0.0
            def touch(level, hi, lo, side):
                if side == 'LONG':
                    return lo <= level <= hi
                else:
                    return lo <= level <= hi
            i = 0
            kset = 0
            for k in range(len(candidates)):
                idx = candidates[k]['idx']
                o = candidates[k]['o']; h = candidates[k]['h']; l = candidates[k]['l']; c = candidates[k]['c']
                rng = candidates[k]['rng']; atr_k = candidates[k]['atr']; 
                if candidates[k]['dir'] == 'bull':
                    entry = l + rng * 0.62
                    entry = max(entry, (o + c) / 2)
                    sl = l - atr_k * 0.6
                    side = 'LONG'
                else:
                    entry = h - rng * 0.62
                    entry = min(entry, (o + c) / 2)
                    sl = h + atr_k * 0.6
                    side = 'SHORT'
                risk = abs(entry - sl)
                if risk <= 0:
                    continue
                tps = [entry + risk*1.5, entry + risk*2.5, entry + risk*3.5] if side=='LONG' else [entry - risk*1.5, entry - risk*2.5, entry - risk*3.5]
                # Round to tick size for realism
                entry = self._round_price(symbol, entry)
                sl = self._round_price(symbol, sl)
                tps = [self._round_price(symbol, x) for x in tps]
                # Wait for entry after the vector candle
                entered = False
                entry_time = None
                entry_candle = None
                for j in range(idx+1, len(candles)):
                    hi = highs[j]; lo = lows[j]
                    # Entry touch?
                    if lo <= entry <= hi:
                        entered = True
                        entry_time = times[j]
                        entry_candle = j
                        break
                if not entered:
                    continue
                # After entry, simulate SL first then TP hits
                outcome = None
                exit_price = None
                exit_time = None
                best_rr = 0.0
                for j in range(entry_candle, len(candles)):
                    hi = highs[j]; lo = lows[j]
                    if side == 'LONG':
                        # Conservative: SL first
                        if lo <= sl:
                            outcome = 'loss'
                            exit_price = sl
                            exit_time = times[j]
                            best_rr = min(best_rr, -1.0)
                            break
                        # TP hits (take first achieved)
                        if hi >= tps[0]:
                            outcome = 'win'
                            exit_price = tps[0]
                            exit_time = times[j]
                            best_rr = max(best_rr, 1.5 if hi < tps[1] else (2.5 if hi < tps[2] else 3.5))
                            break
                    else:
                        if hi >= sl:
                            outcome = 'loss'
                            exit_price = sl
                            exit_time = times[j]
                            best_rr = min(best_rr, -1.0)
                            break
                        if lo <= tps[0]:
                            outcome = 'win'
                            exit_price = tps[0]
                            exit_time = times[j]
                            best_rr = max(best_rr, 1.5 if lo > tps[1] else (2.5 if lo > tps[2] else 3.5))
                            break
                if outcome is None:
                    # trade expired without SL/TP, close at last close
                    outcome = 'expired'
                    exit_price = closes[-1]
                    exit_time = times[-1]
                    rr = ((exit_price - entry)/risk) if side=='LONG' else ((entry - exit_price)/risk)
                else:
                    rr = ((exit_price - entry)/risk) if side=='LONG' else ((entry - exit_price)/risk)
                # Compute net RR after fees and slippage
                # Adverse slippage both at entry and exit
                # Convert absolute cost to R using risk size
                cost_abs = entry * (2*fee_pct + 2*slip_pct)
                cost_r = cost_abs / max(risk, 1e-9)
                rr_net = rr - cost_r
                trades.append({
                    'time': times[idx], 'entry_time': entry_time, 'exit_time': exit_time,
                    'direction': side, 'entry': round(entry,6), 'stop': round(sl,6),
                    'tp1': round(tps[0],6), 'tp2': round(tps[1],6), 'tp3': round(tps[2],6),
                    'exit': round(exit_price,6), 'outcome': outcome, 'rr': round(float(rr),2), 'rr_net': round(float(rr_net),2)
                })
                equity_r += rr
                equity_r_net += rr_net
                peak_r = max(peak_r, equity_r)
                peak_r_net = max(peak_r_net, equity_r_net)
                dd = peak_r - equity_r
                if dd > max_dd_r:
                    max_dd_r = dd
                ddn = peak_r_net - equity_r_net
                if ddn > max_dd_r_net:
                    max_dd_r_net = ddn
            total = len(trades)
            wins = sum(1 for t in trades if t['outcome']=='win')
            losses = sum(1 for t in trades if t['outcome']=='loss')
            expired = sum(1 for t in trades if t['outcome']=='expired')
            avg_rr = float(np.mean([t['rr'] for t in trades])) if trades else 0.0
            avg_rr_net = float(np.mean([t.get('rr_net', t['rr']) for t in trades])) if trades else 0.0
            profits = [t['rr'] for t in trades if t['rr']>0]
            loss_vals = [-t['rr'] for t in trades if t['rr']<0]
            profit_factor = (sum(profits)/sum(loss_vals)) if loss_vals else float('inf') if profits else 0
            profits_net = [t.get('rr_net', t['rr']) for t in trades if t.get('rr_net', t['rr'])>0]
            loss_vals_net = [-t.get('rr_net', t['rr']) for t in trades if t.get('rr_net', t['rr'])<0]
            profit_factor_net = (sum(profits_net)/sum(loss_vals_net)) if loss_vals_net else float('inf') if profits_net else 0
            return {
                'symbol': symbol.upper(), 'interval': interval, 'limit': limit,
                'strategy': 'Vector Candle Scalp',
                'metrics': {
                    'total_trades': total,
                    'wins': wins, 'losses': losses, 'expired': expired,
                    'win_rate_pct': round((wins/total*100) if total else 0,2),
                    'avg_rr': round(avg_rr,2), 'equity_sum_r': round(equity_r,2),
                    'avg_rr_net': round(avg_rr_net,2), 'equity_sum_r_net': round(equity_r_net,2),
                    'max_drawdown_r': round(max_dd_r,2), 'max_drawdown_r_net': round(max_dd_r_net,2),
                    'profit_factor_r': round(profit_factor,2) if profit_factor!=float('inf') else 'INF',
                    'profit_factor_r_net': round(profit_factor_net,2) if profit_factor_net!=float('inf') else 'INF',
                    'fee_bps': float(fee_bps), 'slip_bps': float(slip_bps)
                },
                'trades': trades[-200:],
                'disclaimer': 'Simplified execution model; SL checked before TP within candle for conservatism.'
            }
        except Exception as e:
            return {'error': str(e)}

    def analyze_symbol(self, symbol):
        """Complete analysis pipeline for a symbol."""
        try:
            phase_t0 = time.time(); timings = {}
            ticker_data = self.binance_client.get_ticker_data(symbol)
            current_price = float(ticker_data.get('lastPrice', 0))
            timings['market_data_ms'] = round((time.time()-phase_t0)*1000,2)
            if current_price == 0:
                return {'error': 'Symbol not found or no price data available'}
            t_phase = time.time()
            candles = self.technical_analysis.get_candle_data(symbol, interval='1h')
            timings['candles_fetch_ms'] = round((time.time()-t_phase)*1000,2)
            if not candles:
                return {'error': 'Unable to fetch candlestick data'}
            t_phase = time.time()
            tech_analysis = self.technical_analysis.calculate_advanced_indicators(candles)
            timings['technical_ms'] = round((time.time()-t_phase)*1000,2)
            try:
                t_phase = time.time()
                extended_analysis = AdvancedTechnicalAnalysis.calculate_extended_indicators(candles)
                timings['extended_ms'] = round((time.time()-t_phase)*1000,2)
            except Exception:
                extended_analysis = {}
                timings['extended_ms'] = round((time.time()-t_phase)*1000,2)
            t_phase = time.time()
            pattern_analysis = self.pattern_detector.detect_advanced_patterns(candles)
            try:
                for p in pattern_analysis.get('patterns', []):
                    p.setdefault('timeframe','1h')
            except Exception:
                pass
            timings['patterns_ms'] = round((time.time()-t_phase)*1000,2)
            # Multi-timeframe patterns
            multi_tf_patterns = []
            for ptf in ['15m','4h','1d']:
                try:
                    ptf_candles = self.technical_analysis.get_candle_data(symbol, interval=ptf, limit=120 if ptf!='1d' else 100)
                    if not ptf_candles or len(ptf_candles)<40: continue
                    pa = self.pattern_detector.detect_advanced_patterns(ptf_candles)
                    for pat in pa.get('patterns', []):
                        pat = dict(pat); pat['timeframe']=ptf; multi_tf_patterns.append(pat)
                except Exception:
                    continue
            if multi_tf_patterns:
                pattern_analysis['multi_timeframe_patterns'] = multi_tf_patterns
            # Multi-timeframe technical consensus
            multi_timeframe = {'timeframes': [], 'consensus': {}}
            mt_signals = []
            for tf in ['15m','1h','4h','1d']:
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
                    if rsi_v > 60 and trend_dir == 'UP': tf_signal='strong_bull'
                    elif rsi_v > 55 and trend_dir in ('UP','SIDE'): tf_signal='bull'
                    elif rsi_v < 40 and trend_dir == 'DOWN': tf_signal='strong_bear'
                    elif rsi_v < 45 and trend_dir in ('DOWN','SIDE'): tf_signal='bear'
                    else: tf_signal='neutral'
                    mt_signals.append(tf_signal)
                    multi_timeframe['timeframes'].append({'tf': tf,'rsi': round(rsi_v,2),'trend': trend_dir,'signal': tf_signal,'price': tf_analysis.get('current_price'),'support': tf_analysis.get('support'),'resistance': tf_analysis.get('resistance')})
                except Exception as e:
                    multi_timeframe['timeframes'].append({'tf': tf,'error': str(e)})
            if mt_signals:
                counts = {k: mt_signals.count(k) for k in set(mt_signals)}
                bull_score = counts.get('bull',0) + counts.get('strong_bull',0)*1.5
                bear_score = counts.get('bear',0) + counts.get('strong_bear',0)*1.5
                if bull_score > bear_score and bull_score >= len(mt_signals)*0.5: primary='BULLISH'
                elif bear_score > bull_score and bear_score >= len(mt_signals)*0.5: primary='BEARISH'
                else: primary='NEUTRAL'
                multi_timeframe['consensus']={'bull_score':bull_score,'bear_score':bear_score,'total':len(mt_signals),'primary':primary}
                total_counts = sum(counts.values()) or 1
                multi_timeframe['distribution_pct'] = {k: round(v/total_counts*100,2) for k,v in counts.items()}
            else:
                multi_timeframe['consensus']={'primary':'UNKNOWN'}
            # Market side strength
            side_score={'long':0.0,'short':0.0,'neutral':0.0}; side_basis={}
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
            market_bias = {'long_strength_pct': round(side_score['long']/total_side*100,2),'short_strength_pct': round(side_score['short']/total_side*100,2),'basis': side_basis}
            # Position potential
            # Korrekte Parameter-Reihenfolge: (symbol, current_price, support, resistance, patterns)
            try:
                pattern_list_for_pos = pattern_analysis.get('patterns', []) if isinstance(pattern_analysis, dict) else []
            except Exception:
                pattern_list_for_pos = []
            position_analysis = self.position_manager.analyze_position_potential(symbol, current_price, tech_analysis.get('support'), tech_analysis.get('resistance'), pattern_list_for_pos)
            # Order flow
            order_flow_data = self._analyze_order_flow(symbol, current_price, tech_analysis.get('volume_analysis', {}), multi_timeframe)
            # Inject lightweight derived patterns (OrderFlow, POC bounce, Correlation)
            try:
                derived = self._derive_additional_patterns(symbol, tech_analysis, extended_analysis, multi_timeframe, order_flow_data)
                if derived:
                    pattern_analysis.setdefault('patterns', []).extend(derived)
                    # Keep a compact summary note
                    summ = pattern_analysis.get('pattern_summary') or ''
                    add_note = ', '.join(sorted({d.get('type') for d in derived if d.get('type')}))
                    pattern_analysis['pattern_summary'] = f"{summ} | + {add_note}".strip(' |')
            except Exception:
                pass
            # Vector Candle detection on 5m for scalping
            vector_analysis = {}
            try:
                vector_analysis = self._detect_vector_candles_lowtf(symbol, base_interval='5m')
            except Exception as _e_vec:
                vector_analysis = {'error': str(_e_vec)}
            # AI features & prediction
            t_phase = time.time()
            ai_features = self.ai_system.prepare_advanced_features(tech_analysis, pattern_analysis, ticker_data, position_analysis, extended_analysis, regime_data=None)
            try:
                feat_payload = json.dumps(ai_features, sort_keys=True, default=str).encode(); feature_hash = hashlib.sha256(feat_payload).hexdigest()[:16]
            except Exception:
                feature_hash = 'hash_error'
            try:
                mc_passes = int(os.getenv('AI_MC_PASSES','15'))
            except Exception:
                mc_passes = 15
            ai_analysis = self.ai_system.predict_with_uncertainty(ai_features, passes=mc_passes)
            ai_analysis['feature_hash'] = feature_hash
            ai_analysis['feature_count'] = len(ai_features) if isinstance(ai_features, dict) else 0
            feature_contributions = self._analyze_feature_contributions(ai_features, ai_analysis, tech_analysis, pattern_analysis)
            ai_analysis['feature_contributions'] = feature_contributions
            # AI precision enhancement / lightweight ensemble blending
            try:
                ai_analysis = self._enhance_ai_precision(ai_analysis, tech_analysis, pattern_analysis, multi_timeframe)
            except Exception as _e_prec:
                ai_analysis.setdefault('precision_meta_error', str(_e_prec))
            timings['ai_ms'] = round((time.time()-t_phase)*1000,2)
            # Backend Explainability Meta
            explain_meta = self._build_ai_explainability_meta(ai_analysis, tech_analysis, pattern_analysis, position_analysis, extended_analysis)
            t_phase = time.time()
            # Market Emotion (Euphoria/Fear) index to improve clarity of signals
            emotion = self._compute_market_emotion(tech_analysis, extended_analysis, pattern_analysis, multi_timeframe)
            final_score = self._calculate_weighted_score(tech_analysis, pattern_analysis, ai_analysis, emotion)
            timings['scoring_ms'] = round((time.time()-t_phase)*1000,2)
            liquidation_long = self.liquidation_calc.calculate_liquidation_levels(current_price, 'long')
            liquidation_short = self.liquidation_calc.calculate_liquidation_levels(current_price, 'short')
            regime_data = self._detect_market_regime(candles, tech_analysis, extended_analysis, pattern_analysis, multi_timeframe)
            t_phase = time.time()
            # No-Trade-Zone (Extrembereiche -> lieber warten, verringert Overtrading in Live Geldumgebung)
            ntz_meta = {}
            try:
                sup = tech_analysis.get('support') or 0
                res = tech_analysis.get('resistance') or 0
                ntz_upper = (res and current_price > res * 1.005)
                ntz_lower = (sup and current_price < sup * 0.995)
                if ntz_upper or ntz_lower:
                    ntz_meta = {
                        'active': True,
                        'reason': 'price_extended_above_resistance' if ntz_upper else 'price_breaking_below_support',
                        'note': 'No-Trade-Zone aktiviert: Preis in Extrembereich (>0.5% über R oder <0.5% unter S)'
                    }
                else:
                    # Zusätzlich: sehr enger Korridor -> Vermeide False Break Scalps
                    if sup and res and (res - sup)/current_price*100 < 0.9:
                        ntz_meta = {
                            'active': True,
                            'reason': 'ultra_tight_range',
                            'note': 'No-Trade-Zone: extrem enge Range (<0.9%) – geringes CRV'
                        }
                    else:
                        ntz_meta = {'active': False}
            except Exception:
                ntz_meta = {'active': False, 'error': 'ntz_calc_failed'}

            trade_setups = []
            base_setups = self._generate_trade_setups(symbol, current_price, tech_analysis, extended_analysis, pattern_analysis, final_score, multi_timeframe, regime_data)
            # Ensure timeframe on base setups (defaulting to main analysis tf 1h or pattern_timeframe)
            try:
                for s in (base_setups or []):
                    if 'timeframe' not in s:
                        s['timeframe'] = s.get('pattern_timeframe', '1h')
            except Exception:
                pass
            vector_setups = self._generate_vector_scalp_setups(symbol, current_price, vector_analysis, tech_analysis, extended_analysis, multi_timeframe)
            trade_setups = self._merge_and_prune_setups(base_setups, vector_setups)
            # Hard directional consistency with AI signal: if AI says SELL, hide LONG setups and vice versa
            try:
                ai_sig = None
                ens = (ai_analysis or {}).get('ensemble') if isinstance(ai_analysis, dict) else None
                if isinstance(ens, dict):
                    ai_sig = ens.get('ensemble_signal')
                if not ai_sig:
                    ai_sig = (ai_analysis or {}).get('signal') if isinstance(ai_analysis, dict) else None
                if isinstance(ai_sig, str) and trade_setups:
                    up = ai_sig.upper()
                    if 'SELL' in up:
                        trade_setups = [s for s in trade_setups if s.get('direction') == 'SHORT']
                    elif 'BUY' in up:
                        trade_setups = [s for s in trade_setups if s.get('direction') == 'LONG']
            except Exception:
                pass
            # If No-Trade-Zone triggered, do not block completely; lower confidence and flag warning
            try:
                if ntz_meta.get('active') and isinstance(trade_setups, list):
                    for s in trade_setups:
                        s['confidence'] = max(10, int(s.get('confidence', 50) - 15))
                        conds = s.setdefault('conditions', [])
                        conds.append({'t': 'No-Trade-Zone', 's': 'warn'})
            except Exception:
                pass
            timings['setups_ms'] = round((time.time()-t_phase)*1000,2)
            try:
                if isinstance(final_score, dict):
                    final_score['validation'] = self._validate_signals(tech_analysis, pattern_analysis, ai_analysis, final_score.get('signal'), multi_timeframe)
            except Exception:
                pass
            def make_json_safe(obj):
                if isinstance(obj, dict): return {k: make_json_safe(v) for k,v in obj.items()}
                elif isinstance(obj, list): return [make_json_safe(x) for x in obj]
                elif hasattr(obj,'item'): return obj.item()
                elif hasattr(obj,'tolist'): return obj.tolist()
                else: return obj
            safe_final_score = final_score if isinstance(final_score, dict) else {'score':50,'signal':'HOLD','signal_color':'#6c757d','technical_weight': f"{self.weights['technical']*100}%",'pattern_weight': f"{self.weights['patterns']*100}%",'ai_weight': f"{self.weights['ai']*100}%",'component_scores':{'technical':50,'patterns':50,'ai':50},'validation':{'trading_action':'WAIT','risk_level':'MEDIUM','contradictions':[],'warnings':[],'confidence_factors':['Default safety score used'],'enterprise_ready':False}}
            timings['total_ms'] = round((time.time()-phase_t0)*1000,2)
            # Adaptive risk
            adaptive_risk = self._calculate_adaptive_risk_targets(symbol, current_price, tech_analysis, regime_data, ai_analysis)
            # Price timestamp / freshness
            price_ts = None; freshness_ms=None
            try:
                for k in ('closeTime','close_time','eventTime','E','close'):
                    if k in ticker_data and isinstance(ticker_data.get(k),(int,float)) and ticker_data.get(k)>0:
                        price_ts=int(ticker_data.get(k)); break
                if price_ts is None and candles:
                    price_ts = candles[-1].get('time') or candles[-1].get('timestamp')
                now_ms = int(time.time()*1000)
                if price_ts: freshness_ms = int(now_ms - price_ts)
            except Exception:
                pass
            result = make_json_safe({
                'symbol': symbol,
                'current_price': float(current_price),
                'market_data': ticker_data,
                'technical_analysis': tech_analysis,
                'extended_analysis': extended_analysis,
                'pattern_analysis': pattern_analysis,
                'multi_timeframe': multi_timeframe,
                'position_analysis': position_analysis,
                'ai_analysis': ai_analysis,
                'ai_explainability_meta': explain_meta,
                'ai_feature_hash': ai_analysis.get('feature_hash'),
                'order_flow_analysis': order_flow_data,
                'vector_candles': vector_analysis,
                'emotion_analysis': emotion,
                'adaptive_risk_targets': adaptive_risk,
                'market_bias': market_bias,
                'regime_analysis': regime_data,
                'liquidation_long': liquidation_long,
                'liquidation_short': liquidation_short,
                'trade_setups': trade_setups,
                'trade_filter_meta': ntz_meta,
                'weights': self.weights,
                'final_score': safe_final_score,
                'phase_timings_ms': timings,
                'timestamp': datetime.now().isoformat(),
                'server_time_ms': int(time.time()*1000),
                'price_timestamp_ms': price_ts,
                'data_freshness_ms': freshness_ms
            })
            return result
        except Exception as e:
            import traceback
            return {'error': f'Analysis failed: {str(e)}', 'traceback': traceback.format_exc()}

    # ============================= INTERNAL HELPERS ============================= #
    def _detect_market_regime(self, candles, tech_analysis, extended_analysis, pattern_analysis, multi_timeframe):
        try:
            if len(candles) < 50:
                return {'regime': 'unknown', 'confidence': 0, 'rationale': 'Insufficient data'}
            closes = np.array([c['close'] for c in candles])
            atr_data = extended_analysis.get('atr', {})
            volatility_level = atr_data.get('volatility', 'medium')
            atr_pct = atr_data.get('percentage', 2.0)
            trend_data = tech_analysis.get('trend', {})
            trend_strength = trend_data.get('strength', 'weak')
            trend_direction = trend_data.get('trend', 'neutral')
            support = tech_analysis.get('support', 0)
            resistance = tech_analysis.get('resistance', 0)
            current_price = closes[-1]
            range_pct = ((resistance - support) / current_price * 100) if resistance > support else 0
            mt_consensus = multi_timeframe.get('consensus', {}).get('primary', 'NEUTRAL')
            vol_trend = tech_analysis.get('volume_analysis', {}).get('trend', 'normal')
            patterns = pattern_analysis.get('patterns', [])
            breakout_patterns = [p for p in patterns if 'breakout' in p.get('type', '').lower()]
            consolidation_patterns = [p for p in patterns if any(x in p.get('type', '').lower() for x in ['triangle', 'flag', 'pennant'])]
            regime_scores = {'trending':0,'ranging':0,'expansion':0,'volatility_crush':0}
            if trend_strength in ['strong','very_strong'] and trend_direction != 'neutral': regime_scores['trending'] += 30
            if mt_consensus in ['BULLISH','BEARISH']: regime_scores['trending'] += 20
            if atr_pct > 2.5 and vol_trend in ['high','very_high']: regime_scores['trending'] += 15
            if range_pct < 8 and current_price > support * 1.02 and current_price < resistance * 0.98: regime_scores['ranging'] += 35
            if trend_direction in ['neutral','sideways']: regime_scores['ranging'] += 25
            if len(consolidation_patterns) > 0: regime_scores['ranging'] += 15
            if atr_pct > 4.0: regime_scores['expansion'] += 25
            if len(breakout_patterns) > 0: regime_scores['expansion'] += 30
            if vol_trend == 'very_high': regime_scores['expansion'] += 20
            if atr_pct < 1.5: regime_scores['volatility_crush'] += 30
            if volatility_level == 'low': regime_scores['volatility_crush'] += 25
            if vol_trend in ['low','below_average']: regime_scores['volatility_crush'] += 20
            primary_regime = max(regime_scores, key=regime_scores.get)
            confidence = regime_scores[primary_regime]
            if primary_regime == 'trending': rationale = f"Strong directional movement detected. Trend: {trend_direction} ({trend_strength}), MTF: {mt_consensus}, ATR: {atr_pct:.1f}%"
            elif primary_regime == 'ranging': rationale = f"Price bounded in range. S/R spread: {range_pct:.1f}%, Trend: {trend_direction}, Consolidation patterns: {len(consolidation_patterns)}"
            elif primary_regime == 'expansion': rationale = f"High volatility expansion phase. ATR: {atr_pct:.1f}%, Breakouts: {len(breakout_patterns)}, Volume: {vol_trend}"
            else: rationale = f"Low volatility compression. ATR: {atr_pct:.1f}%, Vol trend: {vol_trend}, Squeeze conditions present"
            sorted_scores = sorted(regime_scores.items(), key=lambda x: x[1], reverse=True)
            secondary = None
            if len(sorted_scores) > 1 and sorted_scores[1][1] > confidence * 0.7:
                secondary = sorted_scores[1][0]
            return {'regime': primary_regime,'secondary_regime': secondary,'confidence': min(100, confidence),'rationale': rationale,'regime_scores': regime_scores,'volatility_level': volatility_level,'atr_percentage': atr_pct,'range_percentage': range_pct,'trend_classification': f"{trend_direction} ({trend_strength})"}
        except Exception as e:
            return {'regime':'error','confidence':0,'rationale': f'Regime detection failed: {str(e)}'}

    def _generate_rsi_caution_narrative(self, rsi, trend):
        caution_level='none'; narrative=''; confidence_penalty=0; signal_quality='ok'
        if rsi >= 80:
            caution_level='extreme'; narrative='⚠️ EXTREME ÜBERKAUFT: RSI sehr hoch - Pullback-Risiko erhöht'; confidence_penalty=25; signal_quality='bad'
        elif rsi >= 70:
            caution_level='high'; narrative='⚠️ ÜBERKAUFT-WARNUNG: RSI über 70 - Vorsicht bei LONG-Einstiegen'; confidence_penalty=15; signal_quality='warn'
        elif rsi <= 20:
            caution_level='extreme_oversold'; narrative='💡 EXTREME ÜBERVERKAUFT: Bounce-Potential hoch'; confidence_penalty=0; signal_quality='ok'
        elif rsi <= 30:
            caution_level='oversold_opportunity'; narrative='💡 ÜBERVERKAUFT: Potentielle LONG Chance'; confidence_penalty=-5; signal_quality='ok'
        elif rsi >= 60 and 'bearish' in trend:
            caution_level='trend_conflict'; narrative='⚠️ TREND-KONFLIKT: RSI erhöht in bearischem Trend'; confidence_penalty=10; signal_quality='warn'
        elif rsi <= 40 and 'bullish' in trend:
            caution_level='healthy_pullback'; narrative='✅ Gesunder Pullback in bullischem Trend'; confidence_penalty=-3; signal_quality='ok'
        return {'caution_level':caution_level,'narrative':narrative,'confidence_penalty':confidence_penalty,'signal_quality':signal_quality,'rsi_value':rsi,'recommendation': self._get_rsi_recommendation(rsi, trend)}

    def _get_rsi_recommendation(self, rsi, trend):
        if rsi >= 80: return 'Avoid new LONG positions, consider profit-taking'
        if rsi >= 70: return 'Use tight stops on LONG positions, monitor for reversal'
        if rsi <= 20: return 'Monitor for reversal confirmation before entering LONG'
        if rsi <= 30: return 'Good LONG entry zone if trend supports'
        if 40 <= rsi <= 60: return 'Neutral RSI - rely on other indicators'
        return 'RSI in normal range - standard trade management'

    def _analyze_order_flow(self, symbol, current_price, volume_analysis, multi_timeframe):
        try:
            order_flow={'bid_ask_spread':0,'order_book_imbalance':0,'delta_momentum':0,'volume_profile_poc':current_price,'liquidity_zones':[],'flow_sentiment':'neutral'}
            vol_ratio = volume_analysis.get('ratio',1.0)
            vol_trend = volume_analysis.get('trend','normal')
            if vol_ratio > 2.0: spread_estimate = current_price*0.001
            elif vol_ratio > 1.5: spread_estimate = current_price*0.0005
            else: spread_estimate = current_price*0.0002
            order_flow['bid_ask_spread']=round(spread_estimate,6)
            order_flow['spread_bps']=round((spread_estimate/current_price)*10000,2)
            if vol_trend == 'very_high' and vol_ratio > 1.8:
                imbalance = min(0.7,(vol_ratio-1)*0.3)
                order_flow['order_book_imbalance']=round(imbalance,3)
                order_flow['flow_sentiment'] = 'buy_pressure' if imbalance>0.3 else 'sell_pressure'
            elif vol_trend == 'below_average':
                order_flow['order_book_imbalance']=round(-(1-vol_ratio)*0.2,3)
                order_flow['flow_sentiment']='low_liquidity'
            else:
                order_flow['order_book_imbalance']=round((vol_ratio-1)*0.1,3)
            mt_consensus = multi_timeframe.get('consensus', {}).get('primary','NEUTRAL')
            if mt_consensus=='BULLISH': order_flow['delta_momentum']=round(0.3+vol_ratio*0.2,3)
            elif mt_consensus=='BEARISH': order_flow['delta_momentum']=round(-0.3 - vol_ratio*0.2,3)
            else: order_flow['delta_momentum']=round((vol_ratio-1)*0.1,3)
            order_flow['volume_profile_poc']=round(current_price*(1+order_flow['delta_momentum']*0.01),4)
            if vol_ratio > 1.5:
                zones=[{'level': round(current_price*0.995,4),'type':'support','strength':'medium'},{'level': round(current_price*1.005,4),'type':'resistance','strength':'medium'}]
                if vol_ratio>2.0:
                    zones.extend([{'level': round(current_price*0.99,4),'type':'support','strength':'strong'},{'level': round(current_price*1.01,4),'type':'resistance','strength':'strong'}])
                order_flow['liquidity_zones']=zones
            imb_abs=abs(order_flow['order_book_imbalance']); delta_abs=abs(order_flow['delta_momentum'])
            if imb_abs>0.4 or delta_abs>0.5: flow_strength='strong'
            elif imb_abs>0.2 or delta_abs>0.3: flow_strength='moderate'
            else: flow_strength='weak'
            order_flow['flow_strength']=flow_strength
            order_flow['analysis_note']=f"Estimated flow (spread: {order_flow['spread_bps']}bps, imbalance: {order_flow['order_book_imbalance']:.1%})"
            return order_flow
        except Exception as e:
            return {'error': f'Order flow analysis failed: {str(e)}','flow_sentiment':'unknown','flow_strength':'unknown'}

    def _derive_additional_patterns(self, symbol, tech, extended, multi_timeframe, order_flow):
        """Leitet zusätzliche leichte Muster aus vorhandenen Analysen ab (keine Extra-API-Calls).
        - OrderFlow Imbalance (Proxy via Volumenverhältnis/Flow-Sentiment)
        - POC Bounce (Nähe zu volume_profile_poc)
        - Correlation Break (RSI-Dispersion über TFs)
        Gibt eine Liste kompatibler Pattern-Dicts zurück.
        """
        out = []
        try:
            price = tech.get('current_price')
            vol_ratio = tech.get('volume_analysis', {}).get('ratio')
            flow = (order_flow or {}).get('flow_sentiment')
            poc = (order_flow or {}).get('volume_profile_poc')
            tf_list = multi_timeframe.get('timeframes', []) if isinstance(multi_timeframe, dict) else []
            rsis = [t.get('rsi') for t in tf_list if isinstance(t, dict) and isinstance(t.get('rsi'), (int, float))]
            # 1) OrderFlow Imbalance
            if isinstance(vol_ratio,(int,float)) and vol_ratio>=1.4 and flow in ('buy_pressure','sell_pressure'):
                out.append({
                    'type': 'OrderFlow Imbalance',
                    'signal': 'bullish' if flow=='buy_pressure' else 'bearish',
                    'timeframe': '1h',
                    'strength': 'MEDIUM',
                    'confidence': 60,
                    'quality_grade': 'C',
                    'description': f"Flow: {flow}, Vol {vol_ratio:.2f}x",
                    'reliability_score': 55
                })
            # 2) POC Bounce
            if isinstance(price,(int,float)) and isinstance(poc,(int,float)) and price>0 and poc>0:
                d_pct = abs(price-poc)/price*100
                if d_pct < 0.35:
                    out.append({
                        'type': 'POC Bounce',
                        'signal': 'bullish' if price>=poc else 'bearish',
                        'timeframe': '1h',
                        'strength': 'MEDIUM',
                        'confidence': 58,
                        'quality_grade': 'C',
                        'description': f"Preis ~{d_pct:.2f}% vom POC",
                        'reliability_score': 52
                    })
            # 3) Correlation Break (RSI-Dispersion)
            if len(rsis)>=3:
                mu = sum(rsis)/len(rsis)
                disp = sum(abs(x-mu) for x in rsis)/len(rsis)
                if disp >= 6.0:
                    out.append({
                        'type': 'Correlation Break (RSI)',
                        'signal': 'neutral',
                        'timeframe': 'MULTI',
                        'strength': 'LOW',
                        'confidence': 54,
                        'quality_grade': 'D',
                        'description': f"RSI-Dispersion ~{disp:.1f} über TFs",
                        'reliability_score': 45
                    })
        except Exception:
            return []
        return out

    def _analyze_feature_contributions(self, features, ai_analysis, tech_analysis, pattern_analysis):
        try:
            # Unified Eingang: dict (neues System) oder list/array (ältere Backtests)
            feature_names = []
            values = []
            if isinstance(features, dict):
                # Bevorzugt Schema des AI-Systems für stabile Reihenfolge
                try:
                    schema = self.ai_system.get_feature_schema() if hasattr(self, 'ai_system') else None
                except Exception:
                    schema = None
                if schema:
                    for name in schema:
                        try:
                            v = features.get(name, 0)
                            if isinstance(v,(int,float)):
                                feature_names.append(name); values.append(float(v))
                            else:
                                feature_names.append(name); values.append(0.0)
                        except Exception:
                            feature_names.append(name); values.append(0.0)
                else:
                    # Fallback: alphabetisch sortierte Keys
                    for name in sorted(features.keys()):
                        v = features.get(name,0)
                        if isinstance(v,(int,float)):
                            feature_names.append(name); values.append(float(v))
                if not values:
                    return {'error':'Keine numerischen Feature-Werte gefunden','top_features':[],'analysis_method':'z_score'}
                values_arr = np.array(values, dtype=float)
            elif isinstance(features,(list,tuple,np.ndarray)) and len(features)>0:
                values_arr = np.array(features, dtype=float)
                feature_names = [f'feat_{i}' for i in range(len(values_arr))]
            else:
                return {'error':'Keine Features für Analyse verfügbar','top_features':[],'analysis_method':'z_score'}

            # Z-Scores (Standardisierung) -> Importance = |z|; robust gegen Skalierung
            mean = float(values_arr.mean())
            std = float(values_arr.std())
            if std < 1e-9:
                # Alle Werte praktisch identisch -> keine Aussagekraft
                return {'top_features':[], 'total_features_analyzed': int(len(values_arr)), 'ai_signal_confidence': ai_analysis.get('confidence',0), 'contextual_interpretations':['Alle Features nahezu identisch – keine dominante Einflussquelle'], 'analysis_method':'z_score', 'note':'Keine Varianz'}
            z = (values_arr - mean) / (std + 1e-9)
            importance = np.abs(z)
            # Normalisieren auf 100%
            total_imp = importance.sum()
            if total_imp <= 0:
                return {'top_features':[], 'total_features_analyzed': int(len(values_arr)), 'ai_signal_confidence': ai_analysis.get('confidence',0), 'contextual_interpretations':['Keine signifikanten Abweichungen'], 'analysis_method':'z_score'}
            norm_imp = importance / total_imp * 100.0
            # Top N auswählen
            top_n = 6
            idx_sorted = np.argsort(norm_imp)[::-1][:top_n]
            contributions = []
            ai_signal = ai_analysis.get('signal','HOLD')
            ai_conf = ai_analysis.get('confidence',0)
            for idx in idx_sorted:
                if norm_imp[idx] < 1.0:  # Skip sehr kleine Beiträge
                    continue
                raw_val = values_arr[idx]
                impact = 'positiv' if raw_val >= mean else 'negativ'
                contributions.append({
                    'feature': feature_names[idx],
                    'importance_pct': round(float(norm_imp[idx]),2),
                    'z_score': round(float(z[idx]),3),
                    'raw_value': round(float(raw_val),4),
                    'impact_direction': impact
                })

            # Kontextuelle Interpretationen
            interpretations = []
            try:
                rsi_val = tech_analysis.get('rsi', {}).get('rsi',50)
                if isinstance(rsi_val,(int,float)):
                    if rsi_val > 70: interpretations.append('RSI überkauft – limitiert bullische Qualität')
                    elif rsi_val < 30: interpretations.append('RSI überverkauft – mögliches Rebound-Potenzial')
                pattern_list = pattern_analysis.get('patterns', [])
                if pattern_list:
                    bull_patterns = sum(1 for p in pattern_list if p.get('signal')=='bullish')
                    bear_patterns = sum(1 for p in pattern_list if p.get('signal')=='bearish')
                    if bull_patterns and bull_patterns > bear_patterns:
                        interpretations.append(f'{bull_patterns} bullishe Pattern unterstützen Aufwärtsrichtung')
                    if bear_patterns and bear_patterns > bull_patterns:
                        interpretations.append(f'{bear_patterns} bearishe Pattern dämpfen Kaufsignale')
                if ai_conf > 70:
                    interpretations.append(f'Hohe KI-Konfidenz ({ai_conf:.1f}%) für {ai_signal}')
            except Exception:
                pass

            # Kompatibilitätsfelder für Frontend: importance, value, impact
            legacy_formatted = []
            for c in contributions:
                legacy_formatted.append({
                    'feature': c['feature'],
                    'importance': c['importance_pct'],  # Prozentwert
                    'value': c['raw_value'],
                    'impact': 'positive' if c['impact_direction'] == 'positiv' else 'negative'
                })
            return {
                'top_features': legacy_formatted,
                'total_features_analyzed': int(len(values_arr)),
                'ai_signal_confidence': ai_conf,
                'contextual_interpretations': interpretations,
                'analysis_method': 'z_score',
                'note': 'Heuristische Attribution basierend auf |z|-Gewichtung'
            }
        except Exception as e:
            return {'error': f'Feature contribution analysis failed: {str(e)}','top_features': [],'analysis_method':'error'}

    def _build_ai_explainability_meta(self, ai_analysis, tech, patterns, position, extended):
        """Erzeugt strukturierte Erklärgründe (serverseitig) für das KI-Signal.
        Liefert Gründe pro Kategorie: widersprechend, unterstützend, neutral.
        """
        meta = {
            'signal': ai_analysis.get('signal'),
            'confidence': ai_analysis.get('confidence'),
            'reliability': ai_analysis.get('reliability_score'),
            'entropy': ai_analysis.get('entropy'),
            'prob_margin': ai_analysis.get('prob_margin'),
            'reasons_negative': [],
            'reasons_positive': [],
            'reasons_neutral': [],
            'debug_factors': {}
        }
        try:
            rsi = tech.get('rsi', {}).get('rsi', 50)
            trend_obj = tech.get('trend', {}) if isinstance(tech.get('trend'), dict) else {}
            macd_curve = tech.get('macd', {}).get('curve_direction', 'neutral')
            support = tech.get('support'); resistance = tech.get('resistance'); price = tech.get('current_price')
            pat_list = patterns.get('patterns', []) if isinstance(patterns, dict) else []
            bull_p = sum(1 for p in pat_list if p.get('signal')=='bullish')
            bear_p = sum(1 for p in pat_list if p.get('signal')=='bearish')
            bull_dom = bull_p > bear_p
            bear_dom = bear_p > bull_p
            dist_res = ((resistance - price)/price*100) if (resistance and price) else None
            dist_sup = ((price - support)/price*100) if (support and price) else None
            atr_pct = extended.get('atr', {}).get('percentage') if isinstance(extended, dict) else None
            regime = extended.get('regime_type') or extended.get('regime')
            # store debug
            meta['debug_factors'] = {
                'rsi': rsi,
                'trend': trend_obj.get('trend'),
                'trend_strength': trend_obj.get('strength'),
                'macd_curve': macd_curve,
                'bull_patterns': bull_p,
                'bear_patterns': bear_p,
                'dist_resistance_pct': dist_res,
                'dist_support_pct': dist_sup,
                'atr_pct': atr_pct,
                'regime': regime
            }
            signal = meta['signal']
            # Positive / supportive factors
            if bull_dom:
                meta['reasons_positive'].append(f"{bull_p} bullishe Pattern unterstützen Aufwärtsstruktur")
            if bear_dom:
                meta['reasons_positive'].append(f"{bear_p} bearishe Pattern bestätigen Abwärtsstruktur")
            if isinstance(rsi, (int,float)) and 45 <= rsi <= 60:
                meta['reasons_positive'].append(f"RSI neutral/stabil ({rsi:.1f}) – Spielraum für Trend-Fortsetzung")
            if 'bullish' in str(macd_curve) and signal in ['BUY','STRONG_BUY']:
                meta['reasons_positive'].append('MACD Kurvenrichtung bullisch unterstützt Kaufsignal')
            if 'bearish' in str(macd_curve) and signal in ['SELL','STRONG_SELL']:
                meta['reasons_positive'].append('MACD Kurvenrichtung bearisch stützt Verkaufssignal')
            # Negative / contradictive
            if signal in ['BUY','STRONG_BUY'] and bear_dom:
                meta['reasons_negative'].append(f"{bear_p} bearishe Pattern widersprechen LONG")
            if signal in ['SELL','STRONG_SELL'] and bull_dom:
                meta['reasons_negative'].append(f"{bull_p} bullishe Pattern dämpfen SELL")
            if signal in ['BUY','STRONG_BUY'] and dist_res is not None and dist_res < 1.2:
                meta['reasons_negative'].append(f"Widerstand sehr nah ({dist_res:.2f}%) – begrenztes Upside")
            if signal in ['SELL','STRONG_SELL'] and dist_sup is not None and dist_sup < 1.2:
                meta['reasons_negative'].append(f"Support sehr nah ({dist_sup:.2f}%) – begrenztes Downside")
            if signal in ['BUY','STRONG_BUY'] and 'bearish' in str(trend_obj.get('trend')):
                meta['reasons_negative'].append('Übergeordneter Trend bearish – Trendkonflikt')
            if signal in ['SELL','STRONG_SELL'] and 'bullish' in str(trend_obj.get('trend')):
                meta['reasons_negative'].append('Übergeordneter Trend bullisch – Trendkonflikt')
            if isinstance(rsi,(int,float)) and rsi > 70 and signal in ['BUY','STRONG_BUY']:
                meta['reasons_negative'].append(f"RSI überkauft ({rsi:.1f}) – Einstiegsrisiko")
            if isinstance(rsi,(int,float)) and rsi < 30 and signal in ['SELL','STRONG_SELL']:
                meta['reasons_negative'].append(f"RSI überverkauft ({rsi:.1f}) – Short-Risiko")
            if atr_pct and atr_pct > 2.5:
                meta['reasons_negative'].append(f"Hohe Volatilität (ATR {atr_pct:.2f}%) – erhöhtes Whipsaw-Risiko")
            # Neutral context
            if dist_res is not None and dist_sup is not None:
                range_w = dist_res + dist_sup
                if range_w < 2.0:
                    meta['reasons_neutral'].append(f"Sehr schmale Range ({range_w:.2f}%) – Breakout-Wahrscheinlichkeit erhöht")
            if meta['reliability'] is not None and meta['reliability'] < 45:
                meta['reasons_negative'].append(f"Niedrige KI Reliability ({meta['reliability']:.1f}%) – Signal vorsichtig interpretieren")
            if not meta['reasons_positive'] and signal not in ['HOLD', None]:
                meta['reasons_positive'].append('Keine starken gegenläufigen Faktoren gefunden')
        except Exception as e:
            meta['error'] = f'explainability_build_failed: {e}'
        return meta
    def _calculate_adaptive_risk_targets(self, symbol, current_price, tech_analysis, regime_data, ai_analysis):
        try:
            base_risk_pct=2.0; base_reward_ratio=2.0
            atr = tech_analysis.get('atr', {}).get('atr', current_price*0.02)
            atr_pct = (atr/current_price)*100 if current_price else 2.0
            if atr_pct > 5.0: vol_multiplier=0.6; reward_multiplier=3.0
            elif atr_pct > 3.0: vol_multiplier=0.8; reward_multiplier=2.5
            elif atr_pct > 1.5: vol_multiplier=1.0; reward_multiplier=2.0
            elif atr_pct > 0.8: vol_multiplier=1.2; reward_multiplier=1.8
            else: vol_multiplier=1.4; reward_multiplier=1.5
            regime_multiplier=1.0; regime_reward_adj=1.0
            if regime_data and 'regime_type' in regime_data:
                regime_type = regime_data['regime_type']; rc = regime_data.get('confidence',50)
                if regime_type=='trending':
                    if rc>70: regime_multiplier=1.3; regime_reward_adj=2.5
                    else: regime_multiplier=1.1; regime_reward_adj=2.2
                elif regime_type=='ranging': regime_multiplier=0.8; regime_reward_adj=1.5
                elif regime_type=='expansion': regime_multiplier=1.2; regime_reward_adj=3.0
                elif regime_type=='volatility_crush': regime_multiplier=0.7; regime_reward_adj=1.3
            ai_confidence = ai_analysis.get('confidence',50); ai_signal = ai_analysis.get('signal','HOLD')
            if ai_confidence > 80 and ai_signal != 'HOLD': confidence_multiplier=1.4
            elif ai_confidence > 60: confidence_multiplier=1.2
            elif ai_confidence > 40: confidence_multiplier=1.0
            else: confidence_multiplier=0.7
            uncertainty = ai_analysis.get('uncertainty') or {}
            entropy = uncertainty.get('entropy'); avg_std = uncertainty.get('avg_std')
            max_entropy = math.log(4)
            if entropy is not None and entropy >= 0:
                norm_entropy = min(1.0, entropy/max_entropy)
                uncertainty_multiplier = 1.0 - 0.4*norm_entropy
            else: uncertainty_multiplier = 1.0
            if avg_std is not None:
                if avg_std > 0.18: uncertainty_multiplier *= 0.8
                elif avg_std > 0.12: uncertainty_multiplier *= 0.9
            uncertainty_multiplier = max(0.5, min(1.0, uncertainty_multiplier))
            adaptive_risk_pct = base_risk_pct * vol_multiplier * regime_multiplier * confidence_multiplier * uncertainty_multiplier
            adaptive_risk_pct = max(0.5, min(5.0, adaptive_risk_pct))
            adaptive_reward_ratio = base_reward_ratio * reward_multiplier * regime_reward_adj
            adaptive_reward_ratio = max(1.2, min(4.0, adaptive_reward_ratio))
            account_size = 10000; risk_amount = account_size * (adaptive_risk_pct/100)
            stop_distance_pct = atr_pct * 0.8; stop_distance = current_price * (stop_distance_pct/100)
            position_size = risk_amount/stop_distance if stop_distance>0 else 0
            target_1_distance = stop_distance * (adaptive_reward_ratio*0.6)
            target_2_distance = stop_distance * adaptive_reward_ratio
            target_3_distance = stop_distance * (adaptive_reward_ratio*1.5)
            if ai_signal == 'BUY':
                stop_loss = current_price - stop_distance
                target_1 = current_price + target_1_distance; target_2 = current_price + target_2_distance; target_3 = current_price + target_3_distance
            else:
                stop_loss = current_price + stop_distance
                target_1 = current_price - target_1_distance; target_2 = current_price - target_2_distance; target_3 = current_price - target_3_distance
            risk_category='low'
            if adaptive_risk_pct > 3.5: risk_category='high'
            elif adaptive_risk_pct > 2.5: risk_category='medium'
            return {'adaptive_risk_pct': round(adaptive_risk_pct,2),'adaptive_reward_ratio': round(adaptive_reward_ratio,1),'position_size': round(position_size,4),'stop_loss': round(stop_loss,4),'targets':{'target_1': round(target_1,4),'target_2': round(target_2,4),'target_3': round(target_3,4)},'risk_amount_usd': round(risk_amount,2),'stop_distance_pct': round(stop_distance_pct,3),'atr_pct': round(atr_pct,3),'risk_category': risk_category,'adjustments':{'volatility_multiplier': round(vol_multiplier,2),'regime_multiplier': round(regime_multiplier,2),'confidence_multiplier': round(confidence_multiplier,2),'uncertainty_multiplier': round(uncertainty_multiplier,2),'reward_multiplier': round(reward_multiplier,1)},'reasoning': f"Risk adjusted for {regime_data.get('regime_type','normal')} regime, {atr_pct:.1f}% vol, {ai_confidence:.0f}% AI conf, entropy={entropy if entropy is not None else 'n/a'}"}
        except Exception as e:
            return {'error': f'Adaptive risk calculation failed: {str(e)}','adaptive_risk_pct':2.0,'adaptive_reward_ratio':2.0}

    def _calculate_weighted_score_base(self, tech_analysis, pattern_analysis, ai_analysis):
        """Combine technical, pattern, and AI into a calibrated score and discrete signal.
        Optionally modulate extremes with market emotion (euphoria/fear) when available.
        """
        # Backward-compatible signature shim
        emotion = None
        if isinstance(pattern_analysis, dict) and pattern_analysis.get('__EMOTION__'):
            # not used; reserved
            pass
        # Ensure ai_reason is defined for later annotations
        ai_reason = None
        tech_score=50
        rsi = tech_analysis.get('rsi', {}).get('rsi',50)
        if isinstance(rsi,(int,float)):
            if rsi>70: tech_score -= (rsi-70)*0.5
            elif rsi<30: tech_score += (30-rsi)*0.5
        trend_data = tech_analysis.get('trend', {})
        trend = trend_data.get('trend','neutral') if isinstance(trend_data,dict) else 'neutral'
        if trend in ['strong_bullish','bullish']: tech_score += 25
        elif trend in ['strong_bearish','bearish']: tech_score -= 25
        macd_data = tech_analysis.get('macd', {})
        if isinstance(macd_data, dict):
            macd_signal = macd_data.get('curve_direction','neutral')
            if 'bullish' in macd_signal: tech_score += 15
            elif 'bearish' in macd_signal: tech_score -= 15
        pattern_score=50
        patterns = pattern_analysis.get('patterns', [])
        for pattern in patterns:
            if pattern.get('signal')=='bullish': pattern_score += pattern.get('confidence',0)*0.3
            elif pattern.get('signal')=='bearish': pattern_score -= pattern.get('confidence',0)*0.3
        ai_signal = ai_analysis.get('signal','HOLD'); ai_confidence = ai_analysis.get('confidence',50)
        if ai_signal=='STRONG_BUY': ai_score = 75 + (ai_confidence-50)*0.5
        elif ai_signal=='BUY': ai_score = 60 + (ai_confidence-50)*0.3
        elif ai_signal=='STRONG_SELL': ai_score = 25 - (ai_confidence-50)*0.5
        elif ai_signal=='SELL': ai_score = 40 - (ai_confidence-50)*0.3
        else: ai_score=50
        dyn_weights=dict(self.weights)
        ai_conf = ai_analysis.get('confidence',50)
        ai_status = ai_analysis.get('status') or ('offline' if not ai_analysis.get('initialized',True) else 'online')
        if ai_status=='offline' or ai_conf < 40:
            # Allow a minimal AI weight floor when near-threshold confidence but aligned context
            floor = 0.0
            try:
                ens = ai_analysis.get('ensemble') or {}
                aligned = (ens.get('alignment') == 'aligned')
                if 30 <= ai_conf < 40 and aligned:
                    floor = min(self.weights.get('ai', 0.1), 0.05)
            except Exception:
                floor = 0.0
            removed = dyn_weights.get('ai',0)
            dyn_weights['ai'] = floor
            rem = dyn_weights['technical'] + dyn_weights['patterns']
            if rem<=0: dyn_weights['technical']=0.7; dyn_weights['patterns']=0.3
            else:
                # Re-normalize the remaining weight mass (1 - floor)
                scale = (1.0 - floor) / rem if rem>0 else 1.0
                dyn_weights['technical']=dyn_weights['technical']*scale
                dyn_weights['patterns']=dyn_weights['patterns']*scale
        final_score=(tech_score*dyn_weights['technical'] + pattern_score*dyn_weights['patterns'] + ai_score*dyn_weights.get('ai',0))
        unc = ai_analysis.get('uncertainty') or {}
        entropy = unc.get('entropy'); avg_std = unc.get('avg_std')
        if entropy is not None:
            max_ent = math.log(4); norm_ent = min(1.0, entropy/max_ent)
            damping = 0.25*norm_ent
            final_score = 50 + (final_score-50)*(1-damping)
        if avg_std is not None and avg_std > 0.12:
            extra = 0.1 if avg_std>0.18 else 0.05
            final_score = 50 + (final_score-50)*(1-extra)
        final_score = max(0,min(100,final_score))
        # Emotion-aware modulation (avoid FOMO at extremes, clarify caution)
        try:
            # Emotion may be injected via new optional param (when available)
            # In older calls, there's no emotion param. We try to detect via a hidden attr.
            _emotion = None
            if isinstance(ai_analysis, dict):
                _emotion = ai_analysis.get('__emotion_ctx__')  # unused path
            # Prefer direct variable if provided by caller through closure; else look up from tech_analysis (none by default)
        except Exception:
            _emotion = None
        # No-op: caller patched signature will pass emotion separately
        # Stricter gating: require minimum ai confidence & reliability alignment
        ai_conf_gate = ai_analysis.get('confidence',50)
        ai_rel_gate = ai_analysis.get('reliability_score',50)
        if final_score>=78 and ai_conf_gate>=55 and ai_rel_gate>=45:
            signal='STRONG_BUY'; signal_color='#28a745'
        elif final_score>=63 and ai_conf_gate>=50:
            signal='BUY'; signal_color='#6f42c1'
        elif final_score<=22 and ai_conf_gate>=55 and ai_rel_gate>=45:
            signal='STRONG_SELL'; signal_color='#dc3545'
        elif final_score<=37 and ai_conf_gate>=50:
            signal='SELL'; signal_color='#fd7e14'
        else:
            signal='HOLD'; signal_color='#6c757d'
        # Graceful fallback: when AI is offline/low-confidence, allow tech+pattern to drive a softer signal
        if signal=='HOLD':
            ai_weight_zero = dyn_weights.get('ai',0) == 0
            if ai_weight_zero or ai_status=='offline' or ai_conf_gate < 45:
                if final_score >= 61:
                    signal = 'BUY'; signal_color = '#6f42c1'
                elif final_score <= 39:
                    signal = 'SELL'; signal_color = '#fd7e14'
                # annotate reason
                if ai_reason is None:
                    ai_reason = 'fallback:tech_pattern'
        # Legacy heuristic probability from final score
        calibrated_prob = 1/(1+math.exp(-(final_score-50)*0.09))
        bullish_prob = calibrated_prob
        if signal in ['SELL','STRONG_SELL']:
            bullish_prob = 1-bullish_prob
        # Prefer AI calibrated probability if provided by AI module
        prob_source = 'score_logistic'
        ai_bull_cal = ai_analysis.get('bull_probability_calibrated')
        if isinstance(ai_bull_cal,(int,float)):
            # ai_bull_cal already percent (0-100)
            ai_bull_cal_val = ai_bull_cal/100.0 if ai_bull_cal > 1 else ai_bull_cal
            if 0 <= ai_bull_cal_val <= 1:
                bullish_prob = ai_bull_cal_val
                prob_source = 'ai_calibrated'
        # If AI weight was suppressed, annotate a disable reason (do not overwrite an existing fallback reason)
        if ai_reason is None and dyn_weights.get('ai',0)==0 and self.weights.get('ai',0)>0:
            if ai_analysis.get('mode')=='offline' or not ai_analysis.get('signal'):
                ai_reason='AI offline/initialization failed'
            elif ai_analysis.get('confidence',0)<40:
                ai_reason=f"AI low confidence {ai_analysis.get('confidence',0)} < 40"
            else:
                ai_reason='AI weight dynamically suppressed'
        return {
            'score': round(final_score,1),
            'probability_bullish': round(bullish_prob*100,2),
            'probability_bullish_source': prob_source,
            'ai_bull_probability_calibrated': ai_analysis.get('bull_probability_calibrated'),
            'ai_bull_probability_raw': ai_analysis.get('bull_probability_raw'),
            'calibrated_probability': round(calibrated_prob*100,2),
            'probability_note': 'Probability source: ' + prob_source + (' (AI Platt scaled)' if prob_source=='ai_calibrated' else ' (score heuristic)'),
            'signal': signal,
            'signal_color': signal_color,
            'technical_weight': f"{dyn_weights['technical']*100:.1f}%",
            'pattern_weight': f"{dyn_weights['patterns']*100:.1f}%",
            'ai_weight': f"{dyn_weights.get('ai',0)*100:.1f}%",
            'ai_disable_reason': ai_reason,
            'component_scores': {'technical': round(tech_score,1),'patterns': round(pattern_score,1),'ai': round(ai_score,1)},
            'validation': self._validate_signals(tech_analysis, pattern_analysis, ai_analysis, signal),
            'confidence_attribution': {
                'technical_contribution': round((tech_score*dyn_weights['technical'])/final_score*100,2) if final_score else 0,
                'pattern_contribution': round((pattern_score*dyn_weights['patterns'])/final_score*100,2) if final_score else 0,
                'ai_contribution': round((ai_score*dyn_weights.get('ai',0))/final_score*100,2) if final_score else 0,
                'uncertainty_damping_applied': entropy is not None or avg_std is not None,
                'prob_source': prob_source
            }
        }

    def _calculate_weighted_score(self, tech_analysis, pattern_analysis, ai_analysis, emotion=None):
        # Build base score using original logic, then apply emotion adjustments
        base = self._calculate_weighted_score_base(tech_analysis, pattern_analysis, ai_analysis)
        # Apply emotion-aware damping/amplification to clarify extremes
        try:
            if emotion and isinstance(emotion, dict):
                state = emotion.get('state')
                score = base.get('score', 50)
                # Damp chasing in bull euphoria and in bear capitulation
                if state in ('euphoria_bull','excited_bull') and score > 50:
                    score = 50 + (score - 50) * 0.86
                if state in ('capitulation_bear','panic_bear','anxious_bear') and score < 50:
                    score = 50 - (50 - score) * 0.86
                base['score'] = round(max(0, min(100, score)), 1)
                # Add context
                base['emotion_context'] = {
                    'state': state,
                    'euphoria_score': emotion.get('euphoria_score'),
                    'motion_index': emotion.get('motion_index')
                }
        except Exception:
            pass
        return base

    def _compute_market_emotion(self, tech_analysis, extended_analysis, pattern_analysis, multi_timeframe):
        """Compute a lightweight Market Emotion index to expose euphoric/fear states and motion.
        No external deps; uses existing indicators to derive a 0-100 score and state label.
        """
        try:
            rsi = tech_analysis.get('rsi', {}).get('rsi', 50)
            macd_curve = tech_analysis.get('macd', {}).get('curve_direction', 'neutral') or 'neutral'
            trend_obj = tech_analysis.get('trend', {}) if isinstance(tech_analysis.get('trend'), dict) else {}
            trend = str(trend_obj.get('trend') or 'neutral')
            support = tech_analysis.get('support') or 0
            resistance = tech_analysis.get('resistance') or 0
            price = tech_analysis.get('current_price') or 0
            atr_pct = None
            vol_trend = 'normal'
            try:
                atr_pct = extended_analysis.get('atr', {}).get('percentage')
                vol_trend = tech_analysis.get('volume_analysis', {}).get('trend', 'normal')
            except Exception:
                pass
            # Pattern differential
            patterns = pattern_analysis.get('patterns', []) if isinstance(pattern_analysis, dict) else []
            bull_p = sum(1 for p in patterns if p.get('signal') == 'bullish')
            bear_p = sum(1 for p in patterns if p.get('signal') == 'bearish')
            mt_primary = multi_timeframe.get('consensus', {}).get('primary', 'NEUTRAL') if isinstance(multi_timeframe, dict) else 'NEUTRAL'
            # Bull/Bear pressure components (0..1 each)
            bull = 0.0; bear = 0.0; drivers = []
            if isinstance(rsi, (int, float)):
                if rsi >= 60:
                    inc = min(1.0, (rsi - 60) / 20.0) * 0.35
                    bull += inc; drivers.append(f"RSI high {rsi:.1f} -> bull +{inc:.2f}")
                elif rsi <= 40:
                    inc = min(1.0, (40 - rsi) / 20.0) * 0.35
                    bear += inc; drivers.append(f"RSI low {rsi:.1f} -> bear +{inc:.2f}")
            if 'bullish' in macd_curve:
                bull += 0.12; drivers.append('MACD curve bullish +0.12')
            if 'bearish' in macd_curve:
                bear += 0.12; drivers.append('MACD curve bearish +0.12')
            if 'strong_bull' in trend: bull += 0.18; drivers.append('Trend strong bull +0.18')
            elif 'bull' in trend: bull += 0.1; drivers.append('Trend bull +0.10')
            if 'strong_bear' in trend: bear += 0.18; drivers.append('Trend strong bear +0.18')
            elif 'bear' in trend: bear += 0.1; drivers.append('Trend bear +0.10')
            # Pattern dominance
            if bull_p or bear_p:
                diff = (bull_p - bear_p)
                if diff > 0:
                    inc = min(0.18, 0.04 * diff); bull += inc; drivers.append(f"Bull patterns {bull_p}>{bear_p} +{inc:.2f}")
                elif diff < 0:
                    inc = min(0.18, 0.04 * (-diff)); bear += inc; drivers.append(f"Bear patterns {bear_p}>{bull_p} +{inc:.2f}")
            # MTF consensus
            if mt_primary == 'BULLISH': bull += 0.12; drivers.append('MTF BULLISH +0.12')
            elif mt_primary == 'BEARISH': bear += 0.12; drivers.append('MTF BEARISH +0.12')
            # Proximity to S/R -> overextension
            if price and resistance and resistance > 0:
                d_res = (resistance - price) / price * 100.0
                if d_res < 0.4: bull += 0.12; drivers.append('Near/above resistance -> overextended bull +0.12')
            if price and support and support > 0:
                d_sup = (price - support) / price * 100.0
                if d_sup < 0.4: bear += 0.12; drivers.append('Near/below support -> overextended bear +0.12')
            # Volatility/Volume amplify emotion
            if isinstance(atr_pct, (int, float)):
                amp = 0.0
                if atr_pct > 3.0: amp = 0.10
                elif atr_pct > 1.8: amp = 0.06
                if amp:
                    if bull >= bear: bull += amp; drivers.append(f'ATR {atr_pct:.1f}% amplifies bull +{amp:.2f}')
                    else: bear += amp; drivers.append(f'ATR {atr_pct:.1f}% amplifies bear +{amp:.2f}')
            if vol_trend in ('high','very_high'):
                if bull >= bear: bull += 0.05; drivers.append('Volume high amplifies bull +0.05')
                else: bear += 0.05; drivers.append('Volume high amplifies bear +0.05')
            # Clamp and scale to 0..100
            bull = max(0.0, min(1.6, bull))
            bear = max(0.0, min(1.6, bear))
            bull_score = round(min(100, bull / 1.2 * 100), 1)
            bear_score = round(min(100, bear / 1.2 * 100), 1)
            euphoria_score = max(bull_score, bear_score)
            state = 'neutral'
            if bull_score >= 75: state = 'euphoria_bull'
            elif bull_score >= 60: state = 'excited_bull'
            elif bear_score >= 75: state = 'capitulation_bear'
            elif bear_score >= 60: state = 'anxious_bear'
            # Motion index ~ ATR% and MTF dispersion of RSI
            tf_list = multi_timeframe.get('timeframes', []) if isinstance(multi_timeframe, dict) else []
            rsis = [t.get('rsi') for t in tf_list if isinstance(t, dict) and isinstance(t.get('rsi'), (int, float))]
            rsi_disp = 0.0
            if len(rsis) >= 2:
                mu = sum(rsis) / len(rsis)
                rsi_disp = sum(abs(x - mu) for x in rsis) / len(rsis)
            motion = 0.0
            if isinstance(atr_pct, (int, float)):
                motion += min(1.0, atr_pct / 4.0) * 0.7
            motion += min(1.0, rsi_disp / 20.0) * 0.3
            motion_index = round(min(100, motion * 100), 1)
            return {
                'euphoria_score': euphoria_score,
                'bull_pressure': bull_score,
                'bear_pressure': bear_score,
                'state': state,
                'motion_index': motion_index,
                'drivers': drivers[:10]
            }
        except Exception as e:
            return {'euphoria_score': 50, 'state': 'neutral', 'motion_index': 0, 'error': str(e)}

    def _validate_signals(self, tech_analysis, pattern_analysis, ai_analysis, final_signal, multi_timeframe=None):
        warnings=[]; contradictions=[]; confidence_factors=[]
        macd_signal = tech_analysis.get('macd', {}).get('curve_direction','neutral')
        if 'bearish' in macd_signal and final_signal in ['BUY','STRONG_BUY']:
            contradictions.append({'type':'MACD_CONTRADICTION','message': f'MACD {macd_signal} vs {final_signal}','severity':'HIGH','recommendation':'Warte auf besseren Einstieg'})
        ai_reliability = ai_analysis.get('reliability_score') if isinstance(ai_analysis, dict) else None
        if isinstance(ai_reliability,(int,float)) and ai_reliability < 40:
            warnings.append({'type':'LOW_AI_RELIABILITY','message': f'KI Zuverlässigkeit niedrig ({ai_reliability:.1f})','recommendation':'Zusätzliche Bestätigung abwarten'})
        if isinstance(ai_reliability,(int,float)) and ai_reliability < 25 and final_signal in ['STRONG_BUY','STRONG_SELL']:
            contradictions.append({'type':'AI_RELIABILITY_CONTRADICTION','message': f'Starkes Signal aber Reliability {ai_reliability:.1f}','severity':'MEDIUM','recommendation':'Signal abschwächen / Bestätigung suchen'})
            contradictions.append({'type':'MACD_CONTRADICTION','message': f'MACD {macd_signal} vs {final_signal}','severity':'HIGH','recommendation':'Warte auf besseren Einstieg'})
        rsi_data = tech_analysis.get('rsi', {}); rsi = rsi_data.get('rsi',50) if isinstance(rsi_data,dict) else (rsi_data if isinstance(rsi_data,(int,float)) else 50)
        if rsi>80 and final_signal in ['BUY','STRONG_BUY']:
            warnings.append({'type':'RSI_OVERBOUGHT','message': f'RSI überkauft ({rsi:.1f})','recommendation':'Warte auf Rückgang unter 70'})
        if rsi<20 and final_signal in ['SELL','STRONG_SELL']:
            warnings.append({'type':'RSI_OVERSOLD','message': f'RSI überverkauft ({rsi:.1f})','recommendation':'Warte auf Anstieg über 30'})
        support=tech_analysis.get('support',0); resistance=tech_analysis.get('resistance',0); current_price=tech_analysis.get('current_price',0)
        if current_price>0:
            d_res = ((resistance-current_price)/current_price)*100
            d_sup = ((current_price-support)/current_price)*100
            if d_res < 2 and final_signal in ['BUY','STRONG_BUY']:
                warnings.append({'type':'NEAR_RESISTANCE','message': f'Preis {d_res:.1f}% unter Resistance','recommendation':'Riskanter LONG'})
            if d_sup < 2 and final_signal in ['SELL','STRONG_SELL']:
                warnings.append({'type':'NEAR_SUPPORT','message': f'Preis {d_sup:.1f}% über Support','recommendation':'Riskanter SHORT'})
        patterns = pattern_analysis.get('patterns', [])
        # Pattern quality heuristics (nicht blockierend)
        try:
            high_grade = sum(1 for p in patterns if p.get('quality_grade') in ('A','B'))
            low_grade = sum(1 for p in patterns if p.get('quality_grade') in ('D',))
            avg_dist = pattern_analysis.get('avg_distance_to_trigger_pct')
            if isinstance(avg_dist,(int,float)) and avg_dist > 1.2:
                warnings.append({'type':'PATTERN_DISTANCE','message': f'Pattern weit vom Trigger (~{avg_dist:.2f}%)','recommendation':'Bestätigung abwarten'})
            if low_grade and high_grade==0:
                warnings.append({'type':'LOW_QUALITY_PATTERNS','message': 'Nur niedrige Pattern-Qualität gefunden','recommendation':'Signal konservativ interpretieren'})
            if high_grade>=2:
                confidence_factors.append(f'{high_grade} hochwertige Pattern stützen Signal')
        except Exception:
            pass
        bearish_patterns = sum(1 for p in patterns if p.get('signal')=='bearish')
        bullish_patterns = sum(1 for p in patterns if p.get('signal')=='bullish')
        if bearish_patterns > bullish_patterns and final_signal in ['BUY','STRONG_BUY']:
            contradictions.append({'type':'PATTERN_CONTRADICTION','message': f'{bearish_patterns} bearish vs {bullish_patterns} bullish','severity':'MEDIUM','recommendation':'Muster widersprechen LONG'})
        elif bullish_patterns > bearish_patterns and final_signal in ['SELL','STRONG_SELL']:
            contradictions.append({'type':'PATTERN_CONTRADICTION','message': f'{bullish_patterns} bullish vs {bearish_patterns} bearish','severity':'MEDIUM','recommendation':'Muster widersprechen SHORT'})
        ai_confidence = ai_analysis.get('confidence',50); ai_signal = ai_analysis.get('signal','HOLD')
        if ai_confidence < 60:
            warnings.append({'type':'LOW_AI_CONFIDENCE','message': f'KI Confidence {ai_confidence}%','recommendation':'Auf klarere Signale warten'})
        opposite_map={'STRONG_BUY':['SELL','STRONG_SELL'],'BUY':['STRONG_SELL','SELL'],'STRONG_SELL':['BUY','STRONG_BUY'],'SELL':['STRONG_BUY','BUY']}
        if ai_signal in opposite_map.get(final_signal, []) and ai_confidence >=55:
            contradictions.append({'type':'AI_FINAL_CONTRADICTION','message': f'KI {ai_signal} widerspricht {final_signal}','severity':'MEDIUM','recommendation':'Bestätigung abwarten'})
        if multi_timeframe and isinstance(multi_timeframe,dict):
            mt_primary = multi_timeframe.get('consensus', {}).get('primary')
            if mt_primary=='BULLISH' and final_signal in ['SELL','STRONG_SELL']:
                contradictions.append({'type':'MTF_CONTRADICTION','message':'MTF BULLISH vs bearish Signal','severity':'HIGH','recommendation':'Auf Alignment warten'})
            if mt_primary=='BEARISH' and final_signal in ['BUY','STRONG_BUY']:
                contradictions.append({'type':'MTF_CONTRADICTION','message':'MTF BEARISH vs bullish Signal','severity':'HIGH','recommendation':'Auf Alignment warten'})
        risk_level='LOW'
        if len(contradictions)>0: risk_level='VERY_HIGH'
        elif len(warnings)>2: risk_level='HIGH'
        elif len(warnings)>0: risk_level='MEDIUM'
        trading_action=final_signal
        if len(contradictions)>0 or risk_level in ['HIGH','VERY_HIGH']:
            trading_action='WAIT'; confidence_factors.append('Signale widersprüchlich oder Risiko hoch')
        else:
            confidence_factors.append('Signale konsistent')
        return {'trading_action': trading_action,'risk_level': risk_level,'contradictions': contradictions,'warnings': warnings,'confidence_factors': confidence_factors,'enterprise_ready': len(contradictions)==0 and risk_level in ['LOW','MEDIUM']}

    def _generate_trade_setups(self, symbol, current_price, tech_analysis, extended_analysis, pattern_analysis, final_score, multi_timeframe=None, regime_data=None):
        """Advanced multi-strategy trade setup generation (migrated from legacy app.py).
        Returns up to 12 setups (core strategies + pattern trades) with confidence, targets & rationale."""
        setups = []
        try:
            # Configurable thresholds (soften to ensure at least some setups)
            RISK_HARD_CAP = 3.2  # previously 3.0
            RISK_CONF_THRESHOLD = 2.5  # previously 2.2
            validation = final_score.get('validation', {}) if isinstance(final_score, dict) else {}
            support = tech_analysis.get('support') or current_price * 0.985
            resistance = tech_analysis.get('resistance') or current_price * 1.015
            rsi = tech_analysis.get('rsi', {}).get('rsi', 50)
            trend = tech_analysis.get('trend', {}).get('trend', 'neutral')
            atr_val = (
                extended_analysis.get('atr', {}).get('value') or
                extended_analysis.get('atr', {}).get('atr') or
                (current_price * 0.004)
            )
            atr_perc = extended_analysis.get('atr', {}).get('percentage') or (atr_val / current_price * 100 if atr_val else 0)
            fib = extended_analysis.get('fibonacci', {})
            enterprise_ready = validation.get('enterprise_ready', False)
            risk_level = validation.get('risk_level', 'MEDIUM')
            contradictions = validation.get('contradictions', [])
            contradiction_count = len(contradictions)
            patterns = pattern_analysis.get('patterns', []) if isinstance(pattern_analysis, dict) else []
            bullish_pattern_present = any(p.get('signal') == 'bullish' for p in patterns)
            bearish_pattern_present = any(p.get('signal') == 'bearish' for p in patterns)

            relaxation = {
                'trend_original': trend,
                'rsi_original': rsi,
                'relaxed_trend_logic': False,
                'relaxed_rsi_bounds': False,
                'fallback_generated': False,
                'pattern_injected': False
            }

            # Baseline ATR floor
            min_atr = max(atr_val, current_price * 0.0025)

            def _structural_targets(direction, entry):
                ext_mult = 8.0  # Wide swing extension for realistic R multiples
                swing_target = entry + min_atr * ext_mult if direction == 'LONG' else entry - min_atr * ext_mult
                return swing_target

            def _confidence(base, adds):
                score = base + sum(adds)
                if contradiction_count:
                    score -= 35
                if risk_level in ['HIGH', 'VERY_HIGH']:
                    score -= 25
                if atr_perc and atr_perc > 1.4:
                    score -= 15
                if atr_perc and atr_perc > 2.0:
                    score -= 25
                if not enterprise_ready:
                    score -= 20
                return max(10, min(95, round(score)))

            # Momentum/Pump Detection (verhindert zu enge SL/TP bei starken Moves)
            macd_curve = tech_analysis.get('macd', {}).get('curve_direction', 'neutral')
            momentum_flag = (
                (rsi and rsi > 68) or
                (atr_perc and atr_perc > 1.6) or
                (isinstance(macd_curve, str) and 'bullish' in macd_curve) or
                (regime_data and regime_data.get('regime') in ['trending','expansion'])
            )

            def _smart_stop(entry, raw_stop, direction, structural_level=None):
                """Adaptive Stop-Ermittlung mit Momentum-/Regime-Schutz.
                Ziel: Keine zu engen SL bei hoher Volatilität oder Momentum-Beschleunigung.
                structural_level: support (LONG) bzw. resistance (SHORT) für strukturelle Distanz."""
                raw_dist = (entry - raw_stop) if direction == 'LONG' else (raw_stop - entry)
                # Strukturpuffer
                if structural_level:
                    struct_dist = (entry - structural_level) if direction == 'LONG' else (structural_level - entry)
                    if struct_dist and struct_dist > 0:
                        struct_dist += min_atr * (1.15 if momentum_flag else 0.7)
                    else:
                        struct_dist = None
                else:
                    struct_dist = None
                # Dynamische Mindestmultiplikatoren
                regime = (regime_data or {}).get('regime') if isinstance(regime_data, dict) else None
                base_mult = 1.2
                if momentum_flag:
                    base_mult = 1.7
                if regime in ['trending','expansion']:
                    base_mult += 0.25
                if rsi and ((direction=='LONG' and rsi>70) or (direction=='SHORT' and rsi<30)):
                    # Extreme RSI -> etwas weiter
                    base_mult += 0.15
                # Zusätzliche Ausweitung bei sehr hoher kurzfristiger ATR-%
                if atr_perc and atr_perc > 2.2:
                    base_mult += 0.3
                vol_min = min_atr * base_mult
                # Mindest absoluter Stop Abstand (Preisprozentsatz) verhindert Mikro-SL
                abs_min_dist = current_price * 0.0035  # ~0.35%
                desired = max(vol_min, struct_dist or 0, raw_dist or 0, abs_min_dist)
                if direction == 'LONG':
                    return round(entry - desired, 4)
                return round(entry + desired, 4)

            def _targets(entry, stop, direction, extra=None):
                structural_ref = support if direction == 'LONG' else resistance
                adj_stop = _smart_stop(entry, stop, direction, structural_ref)
                risk_abs = (entry - adj_stop) if direction == 'LONG' else (adj_stop - entry)
                # Dynamische RR-Leiter: bei Expansion/Trending und Momentum weitere Staffeln
                regime = (regime_data or {}).get('regime') if isinstance(regime_data, dict) else None
                if momentum_flag and regime in ['trending','expansion']:
                    rr_ladder = [2, 3, 5, 8, 12, 16]
                elif momentum_flag:
                    rr_ladder = [2, 3, 5, 8, 11]
                elif regime == 'ranging':
                    rr_ladder = [1.2, 2, 3.2, 4.5, 6]
                else:
                    rr_ladder = [1.5, 2.5, 4, 6, 8]
                base_targets = []
                for m in rr_ladder:
                    tp = entry + risk_abs * m if direction == 'LONG' else entry - risk_abs * m
                    base_targets.append({'label': f'{m}R', 'price': round(tp, 2), 'rr': float(m)})
                swing_ext = _structural_targets(direction, entry)
                swing_rr = abs((swing_ext - entry) / risk_abs)
                if swing_rr > (rr_ladder[-1] * 0.65):
                    base_targets.append({'label': 'Swing', 'price': round(swing_ext, 2), 'rr': round(swing_rr, 2)})
                if extra:
                    for lbl, lvl in extra:
                        if lvl:
                            rr = (lvl - entry) / risk_abs if direction == 'LONG' else (entry - lvl) / risk_abs
                            if rr > 1.05:
                                base_targets.append({'label': lbl, 'price': round(lvl, 2), 'rr': round(rr, 2)})
                # Dedup
                seen = set(); dedup=[]
                for t in sorted(base_targets, key=lambda x: x['rr']):
                    pr = round(t['price'],2)
                    if pr in seen: continue
                    dedup.append(t); seen.add(pr)
                # Progression erzwingen
                filtered=[]; last_rr=0
                for t in dedup:
                    if t['rr'] - last_rr >= 0.65:
                        filtered.append(t); last_rr = t['rr']
                    if len(filtered) >= 7: break
                # Mindestabstand absolut
                min_tp_distance = max(min_atr * (1.15 if momentum_flag else 0.95), current_price * 0.0045)
                filtered_far = [t for t in filtered if abs(t['price'] - entry) >= min_tp_distance]
                if filtered_far:
                    filtered = filtered_far
                # Sicherstellen, dass erster TP >= 1.3R (sonst entfernen und nachrücken)
                filtered = [t for t in filtered if t['rr'] >= 1.3] or filtered
                return filtered, adj_stop

            timeframe_weight = {'15m': 0.6, '1h': 1.0, '4h': 1.4, '1d': 1.8}
            all_patterns_rank = []
            try:
                base_p = patterns
                mt_p = pattern_analysis.get('multi_timeframe_patterns', []) if isinstance(pattern_analysis, dict) else []
                for p in base_p + mt_p:
                    tf = p.get('timeframe', '1h')
                    conf = p.get('confidence', 50)
                    w = timeframe_weight.get(tf, 1.0)
                    p['_rank_score'] = conf * w
                    all_patterns_rank.append(p)
            except Exception:
                pass
            bull_ranked = sorted([p for p in all_patterns_rank if p.get('signal') == 'bullish'], key=lambda x: x.get('_rank_score', 0), reverse=True)
            bear_ranked = sorted([p for p in all_patterns_rank if p.get('signal') == 'bearish'], key=lambda x: x.get('_rank_score', 0), reverse=True)

            rsi_caution = self._generate_rsi_caution_narrative(rsi, trend)

            # LONG logic (relaxed if not strongly bearish)
            trend_validation_passed = False
            if 'bullish' in trend or trend in ['neutral', 'weak', 'moderate']:
                if 'bullish' not in trend:
                    relaxation['relaxed_trend_logic'] = True
                    if rsi > 45 and contradiction_count == 0:
                        trend_validation_passed = True
                else:
                    trend_validation_passed = True
                if trend_validation_passed:
                    entry_pb = support * 1.003
                    raw_stop_pb = support - atr_val * 0.9
                    targets_pb, smart_stop_pb = _targets(entry_pb, raw_stop_pb, 'LONG', [
                        ('Resistance', resistance), ('Fib 0.382', fib.get('fib_382')), ('Fib 0.618', fib.get('fib_618'))
                    ])
                    stop_pb = smart_stop_pb
                    risk_pct = round((entry_pb - stop_pb)/entry_pb*100,2)
                    if risk_pct <= 3.0:
                        base_rationale = 'Multi-validated Einstieg nahe Support mit Professional Risk Management'
                        enhanced_rationale = f"{base_rationale}. {rsi_caution['narrative']}" if rsi_caution['caution_level'] != 'none' else base_rationale
                        setups.append({
                            'id': 'L-PB', 'direction': 'LONG', 'strategy': 'Professional Bullish Pullback',
                            'entry': round(entry_pb, 2), 'stop_loss': round(stop_pb, 2),
                            'risk_percent': risk_pct,
                            'targets': targets_pb,
                            'confidence': _confidence(55, [15 if enterprise_ready else 5, 8 if rsi < 65 else 0, 10 if trend_validation_passed else 0]) - rsi_caution['confidence_penalty'],
                            'conditions': [
                                {'t': 'Trend validation', 's': 'ok' if trend_validation_passed else 'warn'},
                                {'t': f'RSI {rsi:.1f}', 's': rsi_caution['signal_quality']},
                                {'t': f'Risk {risk_pct:.1f}%', 's': 'ok' if risk_pct <= 2.0 else 'warn'},
                                {'t': 'Enterprise Ready', 's': 'ok' if enterprise_ready else 'warn'},
                                {'t': 'Low Contradictions', 's': 'ok' if contradiction_count <= 1 else 'bad'}
                            ],
                            'validation_score': 'PROFESSIONAL' if all([
                                trend_validation_passed, enterprise_ready, contradiction_count == 0, risk_pct <= 2.0
                            ]) else 'STANDARD',
                            'rationale': enhanced_rationale,
                            'rsi_caution': rsi_caution,
                            'regime_context': (regime_data.get('regime') if regime_data else 'unknown'),
                            'justification': {
                                'core_thesis': 'Pullback in intaktem Aufwärtstrend zurück in Nachfragezone (Support Re-Test).',
                                'confluence': [
                                    'Trend Align (bullish / nicht bearish)',
                                    f'RSI moderat ({rsi:.1f})',
                                    'Support strukturell bestätigt',
                                    'Risk <= 2% akzeptabel',
                                    'Keine starken Widersprüche'
                                ],
                                'risk_model': 'Stop unter strukturellem Support + ATR-Puffer (~1.2 ATR).',
                                'invalidations': [
                                    'Tiefer Schlusskurs 1.5% unter Support',
                                    'RSI Divergenz bearish + MACD Curve kippt',
                                    'Volumen Distribution Shift gegen Trend'
                                ],
                                'execution_plan': 'Limit/Stop-Order leicht über Re-Test Candle High, Teilgewinn bei 2R, Rest trailen.'
                            }
                        })
                # Breakout LONG
                entry_bo = resistance * 1.0015
                raw_stop_bo = resistance - atr_val
                targets_bo, smart_stop_bo = _targets(entry_bo, raw_stop_bo, 'LONG', [
                    ('Fib 0.618', fib.get('fib_618')), ('Fib 0.786', fib.get('fib_786'))
                ])
                stop_bo = smart_stop_bo
                setups.append({
                    'id': 'L-BO', 'direction': 'LONG', 'strategy': 'Resistance Breakout',
                    'entry': round(entry_bo, 2), 'stop_loss': round(stop_bo, 2),
                    'risk_percent': round((entry_bo - stop_bo) / entry_bo * 100, 2),
                    'targets': targets_bo,
                    'confidence': _confidence(48, [15 if enterprise_ready else 5, 6 if rsi < 70 else -4]),
                    'conditions': [
                        {'t': 'Break über Resistance', 's': 'ok'},
                        {'t': 'Momentum intakt', 's': 'ok'},
                        {'t': 'Kein starker Widerspruch', 's': 'ok' if contradiction_count == 0 else 'bad'}
                    ],
                    'rationale': 'Ausbruch nutzt Momentum Beschleunigung'
                })
                if bull_ranked:
                    top_b = bull_ranked[0]
                    tfb = top_b.get('timeframe', '1h')
                    entry_pc = current_price * 1.001 if current_price < resistance else resistance * 1.001
                    raw_stop_pc = entry_pc - atr_val * 0.8
                    targets_pc, smart_stop_pc = _targets(entry_pc, raw_stop_pc, 'LONG', [('Resistance', resistance)])
                    stop_pc = smart_stop_pc
                    setups.append({
                        'id': 'L-PC', 'direction': 'LONG', 'strategy': 'Pattern Confirmation',
                        'entry': round(entry_pc, 2), 'stop_loss': round(stop_pc, 2),
                        'risk_percent': round((entry_pc - stop_pc) / entry_pc * 100, 2),
                        'targets': targets_pc,
                        'confidence': _confidence(52, [min(18, int(top_b.get('confidence', 50) / 3)), 5 if 'bullish' in trend else 0]),
                        'conditions': [
                            {'t': f'Pattern {top_b.get("name", "?")}@{tfb}', 's': 'ok'},
                            {'t': f"MACD Curve {tech_analysis.get('macd', {}).get('curve_direction')}", 's': 'ok'},
                            {'t': f'RSI {rsi:.1f}', 's': 'ok' if rsi < 70 else 'warn'}
                        ],
                        'pattern_timeframe': tfb,
                        'pattern_refs': [f"{top_b.get('name','?')}@{tfb}"],
                        'source_signals': ['pattern', 'macd_curve', 'rsi'],
                        'rationale': 'Bullisches Muster bestätigt Fortsetzung'
                    })
                macd_curve = tech_analysis.get('macd', {}).get('curve_direction', 'neutral')
                if 'bullish' in macd_curve and rsi > 55:
                    entry_momo = current_price * 1.0005
                    raw_stop_momo = entry_momo - atr_val
                    targets_momo, smart_stop_momo = _targets(entry_momo, raw_stop_momo, 'LONG', [('Resistance', resistance)])
                    stop_momo = smart_stop_momo
                    setups.append({
                        'id': 'L-MOMO', 'direction': 'LONG', 'strategy': 'Momentum Continuation',
                        'entry': round(entry_momo, 2), 'stop_loss': round(stop_momo, 2),
                        'risk_percent': round((entry_momo - stop_momo) / entry_momo * 100, 2),
                        'targets': targets_momo,
                        'confidence': _confidence(50, [10 if rsi > 60 else 5, 6]),
                        'conditions': [
                            {'t': 'MACD Curve bullish', 's': 'ok'},
                            {'t': f'RSI {rsi:.1f}', 's': 'ok'},
                            {'t': 'Trend nicht bearish', 's': 'ok' if 'bearish' not in trend else 'warn'}
                        ],
                        'source_signals': ['macd_curve', 'rsi', 'trend'],
                        'rationale': 'Momentum Fortsetzung basierend auf MACD Bogen + RSI'
                    })
                if support and (current_price - support) / current_price * 100 < 1.2 and bull_ranked:
                    top_b2 = bull_ranked[0]
                    tfb2 = top_b2.get('timeframe', '1h')
                    entry_rej = support * 1.004
                    raw_stop_rej = support - atr_val * 0.7
                    targets_rej, smart_stop_rej = _targets(entry_rej, raw_stop_rej, 'LONG', [('Resistance', resistance)])
                    stop_rej = smart_stop_rej
                    setups.append({
                        'id': 'L-REJ', 'direction': 'LONG', 'strategy': 'Support Rejection',
                        'entry': round(entry_rej, 2), 'stop_loss': round(stop_rej, 2),
                        'risk_percent': round((entry_rej - stop_rej) / entry_rej * 100, 2),
                        'targets': targets_rej,
                        'confidence': _confidence(48, [8, min(14, int(top_b2.get('confidence', 50) / 4))]),
                        'conditions': [
                            {'t': 'Nahe Support', 's': 'ok'},
                            {'t': 'Bull Pattern', 's': 'ok'},
                            {'t': 'Volatilität ok', 's': 'ok' if atr_perc < 1.5 else 'warn'}
                        ],
                        'pattern_timeframe': tfb2,
                        'pattern_refs': [f"{top_b2.get('name','?')}@{tfb2}"],
                        'source_signals': ['support', 'pattern'],
                        'rationale': 'Rejection nahe Support + bullisches Muster'
                    })
            # LONG mean reversion
            if rsi < 32:
                relaxation['relaxed_rsi_bounds'] = True
            if rsi < 35:
                entry_mr = current_price * 0.998
                raw_stop_mr = entry_mr - atr_val * 0.9
                targets_mr, smart_stop_mr = _targets(entry_mr, raw_stop_mr, 'LONG', [('Resistance', resistance)])
                stop_mr = smart_stop_mr
                setups.append({
                    'id': 'L-MR', 'direction': 'LONG', 'strategy': 'RSI Mean Reversion',
                    'entry': round(entry_mr, 2), 'stop_loss': round(stop_mr, 2),
                    'risk_percent': round((entry_mr - stop_mr) / entry_mr * 100, 2),
                    'targets': targets_mr,
                    'confidence': _confidence(42, [10 if rsi < 28 else 4]),
                    'conditions': [
                        {'t': f'RSI {rsi:.1f}', 's': 'ok'},
                        {'t': 'Trend nicht stark bearish', 's': 'ok' if 'bearish' not in trend else 'warn'}
                    ],
                    'rationale': 'Überverkaufte Bedingung -> Rebound Szenario'
                })

            # SHORT logic
            short_trend_validation_passed = False
            if 'bearish' in trend or trend in ['neutral', 'weak', 'moderate']:
                if 'bearish' not in trend:
                    relaxation['relaxed_trend_logic'] = True
                    if rsi < 55 and contradiction_count == 0:
                        short_trend_validation_passed = True
                else:
                    short_trend_validation_passed = True
                if short_trend_validation_passed:
                    entry_pbs = resistance * 0.997
                    stop_pbs = resistance + atr_val * 0.9
                    risk_pct_short = round((stop_pbs - entry_pbs) / entry_pbs * 100, 2)
                    if risk_pct_short <= 3.0:
                        targets_pbs, smart_stop_pbs = _targets(entry_pbs, stop_pbs, 'SHORT', [('Support', support), ('Fib 0.382', fib.get('fib_382'))])
                        stop_pbs = smart_stop_pbs
                        setups.append({
                            'id': 'S-PB', 'direction': 'SHORT', 'strategy': 'Professional Bearish Pullback',
                            'entry': round(entry_pbs, 2), 'stop_loss': round(stop_pbs, 2),
                            'risk_percent': risk_pct_short,
                            'targets': targets_pbs,
                            'confidence': _confidence(55, [15 if enterprise_ready else 5, 8 if rsi > 35 else 0, 10 if short_trend_validation_passed else 0]),
                            'conditions': [
                                {'t': 'Short Trend validation', 's': 'ok' if short_trend_validation_passed else 'warn'},
                                {'t': f'RSI {rsi:.1f}', 's': 'ok' if rsi > 35 else 'warn'},
                                {'t': f'Risk {risk_pct_short:.1f}%', 's': 'ok' if risk_pct_short <= 2.0 else 'warn'},
                                {'t': 'Enterprise Ready', 's': 'ok' if enterprise_ready else 'warn'}
                            ],
                            'validation_score': 'PROFESSIONAL' if all([
                                short_trend_validation_passed, enterprise_ready, contradiction_count == 0, risk_pct_short <= 2.0
                            ]) else 'STANDARD',
                            'rationale': 'Multi-validated Pullback an Widerstand mit Professional Risk Management'
                        })
                entry_bd = support * 0.9985
                stop_bd = support + atr_val
                targets_bd, smart_stop_bd = _targets(entry_bd, stop_bd, 'SHORT', [('Fib 0.236', fib.get('fib_236'))])
                stop_bd = smart_stop_bd
                setups.append({
                    'id': 'S-BD', 'direction': 'SHORT', 'strategy': 'Support Breakdown',
                    'entry': round(entry_bd, 2), 'stop_loss': round(stop_bd, 2),
                    'risk_percent': round((stop_bd - entry_bd) / entry_bd * 100, 2),
                    'targets': targets_bd,
                    'confidence': _confidence(48, [14 if enterprise_ready else 4, 5]),
                    'conditions': [
                        {'t': 'Bruch unter Support', 's': 'ok'},
                        {'t': 'Keine bull. Divergenz', 's': 'ok'}
                    ],
                    'rationale': 'Beschleunigter Momentum-Handel beim Support-Bruch'
                })
                if bear_ranked:
                    top_s = bear_ranked[0]
                    tfs = top_s.get('timeframe', '1h')
                    entry_ps = current_price * 0.999 if current_price > support else support * 0.999
                    stop_ps = entry_ps + atr_val * 0.8
                    targets_ps, smart_stop_ps = _targets(entry_ps, stop_ps, 'SHORT', [('Support', support)])
                    stop_ps = smart_stop_ps
                    setups.append({
                        'id': 'S-PC', 'direction': 'SHORT', 'strategy': 'Pattern Confirmation',
                        'entry': round(entry_ps, 2), 'stop_loss': round(stop_ps, 2),
                        'risk_percent': round((stop_ps - entry_ps) / entry_ps * 100, 2),
                        'targets': targets_ps,
                        'confidence': _confidence(52, [min(18, int(top_s.get('confidence', 50) / 3)), 5 if 'bearish' in trend else 0]),
                        'conditions': [
                            {'t': f'Pattern {top_s.get("name", "?")}@{tfs}', 's': 'ok'},
                            {'t': f"MACD Curve {tech_analysis.get('macd', {}).get('curve_direction')}", 's': 'ok'},
                            {'t': f'RSI {rsi:.1f}', 's': 'ok' if rsi > 30 else 'warn'}
                        ],
                        'pattern_timeframe': tfs,
                        'pattern_refs': [f"{top_s.get('name','?')}@{tfs}"],
                        'source_signals': ['pattern', 'macd_curve', 'rsi'],
                        'rationale': 'Bearishes Muster bestätigt Fortsetzung'
                    })
                macd_curve_s = tech_analysis.get('macd', {}).get('curve_direction', 'neutral')
                if 'bearish' in macd_curve_s and rsi < 45:
                    entry_momo_s = current_price * 0.9995
                    stop_momo_s = entry_momo_s + atr_val
                    targets_momo_s, smart_stop_momo_s = _targets(entry_momo_s, stop_momo_s, 'SHORT', [('Support', support)])
                    stop_momo_s = smart_stop_momo_s
                    setups.append({
                        'id': 'S-MOMO', 'direction': 'SHORT', 'strategy': 'Momentum Continuation',
                        'entry': round(entry_momo_s, 2), 'stop_loss': round(stop_momo_s, 2),
                        'risk_percent': round((stop_momo_s - entry_momo_s) / entry_momo_s * 100, 2),
                        'targets': targets_momo_s,
                        'confidence': _confidence(50, [10 if rsi < 40 else 5, 6]),
                        'conditions': [
                            {'t': 'MACD Curve bearish', 's': 'ok'},
                            {'t': f'RSI {rsi:.1f}', 's': 'ok'},
                            {'t': 'Trend nicht bullish', 's': 'ok' if 'bullish' not in trend else 'warn'}
                        ],
                        'source_signals': ['macd_curve', 'rsi', 'trend'],
                        'rationale': 'Momentum Fortsetzung basierend auf MACD Bogen + RSI'
                    })
                if resistance and (resistance - current_price) / current_price * 100 < 1.2 and bear_ranked:
                    top_s2 = bear_ranked[0]
                    tfs2 = top_s2.get('timeframe', '1h')
                    entry_rej_s = resistance * 0.996
                    stop_rej_s = resistance + atr_val * 0.7
                    targets_rej_s, smart_stop_rej_s = _targets(entry_rej_s, stop_rej_s, 'SHORT', [('Support', support)])
                    stop_rej_s = smart_stop_rej_s
                    setups.append({
                        'id': 'S-REJ', 'direction': 'SHORT', 'strategy': 'Resistance Rejection',
                        'entry': round(entry_rej_s, 2), 'stop_loss': round(stop_rej_s, 2),
                        'risk_percent': round((stop_rej_s - entry_rej_s) / entry_rej_s * 100, 2),
                        'targets': targets_rej_s,
                        'confidence': _confidence(48, [8, min(14, int(top_s2.get('confidence', 50) / 4))]),
                        'pattern_timeframe': tfs2,
                        'pattern_refs': [f"{top_s2.get('name','?')}@{tfs2}"],
                        'source_signals': ['resistance', 'pattern'],
                        'rationale': 'Rejection nahe Resistance + bearisches Muster'
                    })

            # SHORT mean reversion
            if rsi > 68:
                relaxation['relaxed_rsi_bounds'] = True
            if rsi > 65:
                entry_mrs = current_price * 1.002
                raw_stop_mrs = entry_mrs + atr_val * 0.9
                targets_mrs, smart_stop_mrs = _targets(entry_mrs, raw_stop_mrs, 'SHORT', [('Support', support)])
                stop_mrs = smart_stop_mrs
                setups.append({
                    'id': 'S-MR', 'direction': 'SHORT', 'strategy': 'RSI Mean Reversion',
                    'entry': round(entry_mrs, 2), 'stop_loss': round(stop_mrs, 2),
                    'risk_percent': round((stop_mrs - entry_mrs) / entry_mrs * 100, 2),
                    'targets': targets_mrs,
                    'confidence': _confidence(42, [10 if rsi > 72 else 4]),
                    'conditions': [
                        {'t': f'RSI {rsi:.1f}', 's': 'ok'},
                        {'t': 'Trend nicht stark bullish', 's': 'ok' if 'bullish' not in trend else 'warn'}
                    ],
                    'rationale': 'Überkaufte Bedingung -> Rücksetzer / Mean Reversion'
                })

            # Pattern injected setups if directional scarcity
            if bullish_pattern_present and len([s for s in setups if s['direction'] == 'LONG']) < 2:
                entry_pat = current_price * 1.001
                raw_stop_pat = current_price - atr_val
                targets_pat, smart_stop_pat = _targets(entry_pat, raw_stop_pat, 'LONG', [('Resistance', resistance)])
                stop_pat = smart_stop_pat
                setups.append({
                    'id': 'L-PAT', 'direction': 'LONG', 'strategy': 'Pattern Boost Long',
                    'entry': round(entry_pat, 2), 'stop_loss': round(stop_pat, 2),
                    'risk_percent': round((entry_pat - stop_pat) / entry_pat * 100, 2),
                    'targets': targets_pat,
                    'confidence': 55,
                    'conditions': [{'t': 'Bullish Pattern', 's': 'ok'}],
                    'rationale': 'Bullish Chart Pattern aktiviert (relaxed)'
                })
                relaxation['pattern_injected'] = True
            if bearish_pattern_present and len([s for s in setups if s['direction'] == 'SHORT']) < 2:
                entry_pats = current_price * 0.999
                raw_stop_pats = current_price + atr_val
                targets_pats, smart_stop_pats = _targets(entry_pats, raw_stop_pats, 'SHORT', [('Support', support)])
                stop_pats = smart_stop_pats
                setups.append({
                    'id': 'S-PAT', 'direction': 'SHORT', 'strategy': 'Pattern Boost Short',
                    'entry': round(entry_pats, 2), 'stop_loss': round(stop_pats, 2),
                    'risk_percent': round((stop_pats - entry_pats) / entry_pats * 100, 2),
                    'targets': targets_pats,
                    'confidence': 55,
                    'conditions': [{'t': 'Bearish Pattern', 's': 'ok'}],
                    'rationale': 'Bearish Chart Pattern aktiviert (relaxed)'
                })
                relaxation['pattern_injected'] = True

            # Fallback generic setups
            if len(setups) < 2:
                relaxation['fallback_generated'] = True
                generic_risk = max(atr_val, current_price * 0.003)
                entry_gl = current_price
                raw_stop_gl = entry_gl - generic_risk
                targets_gl, smart_stop_gl = _targets(entry_gl, raw_stop_gl, 'LONG', [('Resistance', resistance)])
                stop_gl = smart_stop_gl
                setups.append({
                    'id': 'L-FB', 'direction': 'LONG', 'strategy': 'Generic Long',
                    'entry': round(entry_gl, 2), 'stop_loss': round(stop_gl, 2),
                    'risk_percent': round((entry_gl - stop_gl) / entry_gl * 100, 2),
                    'targets': targets_gl,
                    'confidence': 45,
                    'conditions': [{'t': 'Fallback', 's': 'info'}],
                    'rationale': 'Fallback Long Setup (relaxed)'
                })
                entry_gs = current_price
                raw_stop_gs = entry_gs + generic_risk
                targets_gs, smart_stop_gs = _targets(entry_gs, raw_stop_gs, 'SHORT', [('Support', support)])
                stop_gs = smart_stop_gs
                setups.append({
                    'id': 'S-FB', 'direction': 'SHORT', 'strategy': 'Generic Short',
                    'entry': round(entry_gs, 2), 'stop_loss': round(stop_gs, 2),
                    'risk_percent': round((stop_gs - entry_gs) / entry_gs * 100, 2),
                    'targets': targets_gs,
                    'confidence': 45,
                    'conditions': [{'t': 'Fallback', 's': 'info'}],
                    'rationale': 'Fallback Short Setup (relaxed)'
                })

            # Probability heuristics
            try:
                base_prob = 0.5
                if isinstance(final_score, dict):
                    cp = final_score.get('calibrated_probability')
                    if isinstance(cp, (int, float)):
                        base_prob = max(0.01, min(0.99, cp / 100.0))
                for s in setups:
                    conf_v = s.get('confidence', 50) / 100.0
                    if s.get('direction') == 'LONG':
                        p = base_prob + (conf_v - 0.5) * 0.35
                    else:
                        p = (1 - base_prob) + (conf_v - 0.5) * 0.35
                    p = max(0.02, min(0.98, p))
                    # Leichte zusätzliche Kalibrierung durch Setup-spezifische Qualität (nicht signal-killend)
                    quality_adj = 1.0
                    if 'Pattern' in s.get('strategy',''):
                        quality_adj += 0.03
                    if any(c.get('s')=='bad' for c in s.get('conditions', [])):
                        quality_adj -= 0.04
                    p_adj = max(0.01, min(0.99, p * quality_adj))
                    s['probability_estimate_pct'] = round(p_adj * 100, 2)
                    s['probability_note'] = 'Heuristisch kalibriert (Score+Confidence+Qualität).'
            except Exception:
                pass

            for s in setups:
                if s.get('targets'):
                    s['primary_rr'] = s['targets'][0]['rr']

            # Integrate pattern trades (current ChartPatternTrader API)
            pattern_trades = []
            try:
                pattern_trades = ChartPatternTrader.generate_pattern_trades(symbol, pattern_analysis, tech_analysis, extended_analysis, current_price)
            except Exception:
                pattern_trades = []
            all_setups = setups + pattern_trades
            all_setups.sort(key=lambda x: x.get('confidence', 50), reverse=True)
            trimmed = all_setups[:12]
            if trimmed:
                trimmed[0]['relaxation_meta'] = relaxation

            # Zusätzliche Risiko / Reliability Nachfilter
            ai_rel = 50
            try:
                # final_score evtl. enthält validation / ai signal info, fallback aus component scores
                ai_rel = final_score.get('validation', {}).get('ai_reliability') if isinstance(final_score, dict) else 50
            except Exception:
                pass
            # Falls AI-Analyse Objekt verfügbar, entnehme reliability_score
            try:
                # Not available hier lokal -> ignorieren; könnte später injiziert werden
                pass
            except Exception:
                pass

            filtered_post = []
            for s in trimmed:
                risk_pct = s.get('risk_percent', 0)
                # Relaxed: allow slightly higher risk, and if removed keep best fallback later
                if risk_pct > RISK_HARD_CAP:
                    s['filtered_reason'] = 'risk_pct_hard_cap'
                    continue
                if risk_pct > RISK_CONF_THRESHOLD and s.get('confidence',0) < 55:
                    s['filtered_reason'] = 'risk_pct_conf_low'
                    continue
                filtered_post.append(s)
            if not filtered_post:
                # Fallback: keep best (lowest risk * highest confidence rank)
                if trimmed:
                    fallback_sorted = sorted(trimmed, key=lambda x: (x.get('risk_percent',10), -x.get('confidence',0)))
                    fb = fallback_sorted[0]
                    fb['filter_bypass'] = True
                    filtered_post = [fb]

            # === Backend Pruning auf max 1 Long & 1 Short ===
            def _rank(s):
                rr = s.get('primary_rr') or (s['targets'][0]['rr'] if s.get('targets') else 1)
                risk_pct = s.get('risk_percent', 2.0)
                return (s.get('confidence',50)) * rr * (1 - min(risk_pct,5)/100.0)
            best_long = None; best_short = None
            for s in filtered_post:
                if s.get('direction') == 'LONG':
                    if best_long is None or _rank(s) > _rank(best_long):
                        best_long = s
                elif s.get('direction') == 'SHORT':
                    if best_short is None or _rank(s) > _rank(best_short):
                        best_short = s
            pruned = []
            if best_long: pruned.append(best_long)
            if best_short: pruned.append(best_short)
            if not pruned and filtered_post:
                pruned = [filtered_post[0]]
            for p in pruned:
                p['selection_method'] = 'directional_top'
            return pruned
        except Exception as e:
            self.logger.error(f"Trade setup generation error: {e}")
            return []

    def _enhance_ai_precision(self, ai_analysis, tech_analysis, pattern_analysis, multi_timeframe):
        """Lightweight ensemble-style refinement to increase precision without duplicating core model code.
        Blends base AI signal with rule-based probability derived from technical + pattern context.
        Adjusts confidence & reliability when both align or contradict.
        Returns updated ai_analysis dict (in-place modifications)."""
        if not isinstance(ai_analysis, dict):
            return ai_analysis
        base_signal = ai_analysis.get('signal')
        base_conf = ai_analysis.get('confidence', 50)
        base_rel = ai_analysis.get('reliability_score', 50)
        try:
            # === Heuristic directional score (bullishness 0-100) ===
            rsi_v = tech_analysis.get('rsi', {}).get('rsi', 50)
            trend_obj = tech_analysis.get('trend', {}) if isinstance(tech_analysis.get('trend'), dict) else {}
            trend_label = (trend_obj.get('trend') or 'neutral')
            macd_curve = tech_analysis.get('macd', {}).get('curve_direction', 'neutral') or 'neutral'
            patterns = pattern_analysis.get('patterns', []) if isinstance(pattern_analysis, dict) else []
            bull_p = sum(1 for p in patterns if p.get('signal') == 'bullish')
            bear_p = sum(1 for p in patterns if p.get('signal') == 'bearish')
            mt_primary = multi_timeframe.get('consensus', {}).get('primary') if isinstance(multi_timeframe, dict) else 'NEUTRAL'
            score = 50
            if rsi_v > 60: score += 5
            if rsi_v < 40: score -= 5
            if 'strong_bull' in trend_label: score += 15
            elif 'strong_bear' in trend_label: score -= 15
            elif 'bull' in trend_label: score += 8
            elif 'bear' in trend_label: score -= 8
            if 'bullish' in macd_curve: score += 7
            elif 'bearish' in macd_curve: score -= 7
            score += (bull_p - bear_p) * 3  # pattern differential
            if mt_primary == 'BULLISH': score += 6
            elif mt_primary == 'BEARISH': score -= 6
            # Context: volatility / regime (soft shrink toward neutral to reduce false extremes)
            atr_pct = None
            try:
                atr_pct = tech_analysis.get('atr', {}).get('percentage') or tech_analysis.get('atr_pct')
            except Exception:
                atr_pct = None
            regime_ctx = ai_analysis.get('regime_analysis', {}).get('regime') or tech_analysis.get('regime_context') if isinstance(tech_analysis, dict) else None
            if isinstance(atr_pct,(int,float)):
                if atr_pct > 3.5:
                    score = 50 + (score-50)*0.85
                elif atr_pct < 1.0:
                    score = 50 + (score-50)*0.9
            if isinstance(regime_ctx,str):
                if regime_ctx == 'expansion':
                    score = 50 + (score-50)*0.88
                elif regime_ctx == 'ranging' and ('strong_bull' in trend_label or 'strong_bear' in trend_label):
                    score = 50 + (score-50)*0.9
            rule_prob = max(0, min(100, score)) / 100.0
            # === Convert base AI signal to probability ===
            if base_signal in ['BUY','STRONG_BUY']:
                ai_prob = 0.65 if base_signal == 'BUY' else 0.80
            elif base_signal in ['SELL','STRONG_SELL']:
                ai_prob = 0.35 if base_signal == 'SELL' else 0.20
            else:
                ai_prob = 0.50
            # === Blend (confidence weighted) ===
            w_ai = min(1.0, base_conf/100.0)
            ensemble_prob = (ai_prob * (0.55 + 0.25*w_ai) + rule_prob * (0.45 - 0.25*(1-w_ai)))
            if isinstance(atr_pct,(int,float)) and atr_pct > 4.5:
                ensemble_prob = 0.5 + (ensemble_prob-0.5)*0.8
            if regime_ctx == 'ranging':
                ensemble_prob = 0.5 + (ensemble_prob-0.5)*0.85
            ensemble_prob = max(0.05, min(0.95, ensemble_prob))
            # === Ensemble signal thresholds ===
            if ensemble_prob >= 0.78: ens_signal = 'STRONG_BUY'
            elif ensemble_prob >= 0.60: ens_signal = 'BUY'
            elif ensemble_prob <= 0.22: ens_signal = 'STRONG_SELL'
            elif ensemble_prob <= 0.40: ens_signal = 'SELL'
            else: ens_signal = 'HOLD'
            # Alignment assessment
            alignment = 1 if ens_signal == base_signal else -1 if ((ens_signal.startswith('BUY') and str(base_signal or '').startswith('SELL')) or (ens_signal.startswith('SELL') and str(base_signal or '').startswith('BUY'))) else 0
            # Confidence & reliability tweaks
            conf_adj = 5 if alignment == 1 else -7 if alignment == -1 else 0
            rel_adj = 4 if alignment == 1 else -6 if alignment == -1 else 0
            ai_analysis['ensemble'] = {
                'rule_prob_bullish_pct': round(rule_prob*100,2),
                'ai_prob_bullish_pct': round(ai_prob*100,2),
                'ensemble_bullish_pct': round(ensemble_prob*100,2),
                'ensemble_signal': ens_signal,
                'alignment': 'aligned' if alignment==1 else 'conflict' if alignment==-1 else 'neutral',
                'volatility_pct': atr_pct,
                'regime_context': regime_ctx
            }
            ai_analysis['confidence'] = max(0, min(100, base_conf + conf_adj))
            if isinstance(base_rel,(int,float)):
                ai_analysis['reliability_score'] = max(0, min(100, base_rel + rel_adj))
            ai_analysis['precision_enhanced'] = True
        except Exception as e:
            ai_analysis.setdefault('ensemble_error', str(e))
        return ai_analysis

    def _detect_vector_candles_lowtf(self, symbol, base_interval='5m', limit=180):
        """Detect recent bullish/bearish 'vector candles' on a lower timeframe (default 5m).
        Vector candle heuristic: large body (>70% of range), range > 1.2x ATR(14), volume spike > 1.5x SMA20.
        Returns latest candidates and a compact context for scalping entries.
        """
        tf_candles = self.technical_analysis.get_candle_data(symbol, interval=base_interval, limit=limit)
        if not tf_candles or len(tf_candles) < 40:
            return {'timeframe': base_interval, 'candidates': []}
        # Build arrays
        opens = np.array([c['open'] for c in tf_candles], dtype=float)
        highs = np.array([c['high'] for c in tf_candles], dtype=float)
        lows = np.array([c['low'] for c in tf_candles], dtype=float)
        closes = np.array([c['close'] for c in tf_candles], dtype=float)
        volumes = np.array([c.get('volume', 0.0) for c in tf_candles], dtype=float)
        times = [c.get('time') for c in tf_candles]
        # ATR(14) approx on this tf
        tr = np.maximum(highs[1:], closes[:-1]) - np.minimum(lows[1:], closes[:-1])
        atr = np.zeros_like(highs)
        if len(tr) >= 14:
            rma = np.zeros_like(tr)
            rma[13] = tr[:14].mean()
            for i in range(14, len(tr)):
                rma[i] = (rma[i-1]*13 + tr[i]) / 14
            atr[1+13:] = rma[13:]
            atr[:1+13] = rma[13]
        else:
            atr[:] = (highs - lows).mean()
        # Volume SMA20
        vol_sma = np.zeros_like(volumes)
        if len(volumes) >= 20:
            for i in range(19, len(volumes)):
                vol_sma[i] = volumes[i-19:i+1].mean()
            vol_sma[:19] = volumes[:19].mean() if len(volumes)>=1 else 0
        else:
            vol_sma[:] = volumes.mean() if len(volumes) else 0
        # Evaluate last N
        candidates = []
        last_n = min(50, len(tf_candles))
        for idx in range(len(tf_candles)-last_n, len(tf_candles)):
            o = opens[idx]; h = highs[idx]; l = lows[idx]; c = closes[idx]
            rng = max(1e-9, h - l); body = abs(c - o)
            body_ratio = body / rng if rng else 0
            atr_k = atr[idx] if atr[idx] > 0 else (rng)
            vol_k = volumes[idx]; v_sma = vol_sma[idx] if vol_sma[idx] > 0 else (volumes.mean() if volumes.size>0 else 0)
            is_spike = (vol_k > 1.5 * v_sma) if v_sma else False
            is_range_big = (rng > 1.2 * atr_k)
            if body_ratio >= 0.7 and is_range_big and is_spike:
                direction = 'bull' if c > o else 'bear'
                upper_wick = h - max(c, o)
                lower_wick = min(c, o) - l
                drivers = {
                    'body_ratio': round(body_ratio,3), 'range_vs_atr': round(rng/max(1e-9, atr_k),2),
                    'vol_spike_x': round(vol_k/max(v_sma,1e-9),2), 'upper_wick_pct': round(upper_wick/rng*100,1), 'lower_wick_pct': round(lower_wick/rng*100,1)
                }
                candidates.append({
                    'time': times[idx], 'idx': idx, 'direction': direction,
                    'open': float(o), 'high': float(h), 'low': float(l), 'close': float(c),
                    'mid_body': float((o+c)/2), 'range': float(rng), 'atr': float(atr_k),
                    'volume': float(vol_k), 'vol_sma': float(v_sma), 'drivers': drivers
                })
        # Keep only last 3 by recency
        recent = candidates[-3:]
        return {'timeframe': base_interval, 'candidates': recent}

    def _generate_vector_scalp_setups(self, symbol, current_price, vector_analysis, tech, extended, multi_timeframe):
        """Create scalping setups based on latest vector candle(s).
        Entry: retest of 50-62% of vector candle in direction of impulse.
        SL: beyond opposite wick + ATR buffer; TPs: ladder at 1.5R/2.5R/3.5R.
        """
        try:
            cands = (vector_analysis or {}).get('candidates', [])
            if not cands:
                return []
            setups = []
            atr_pct = extended.get('atr', {}).get('percentage') if isinstance(extended, dict) else None
            for vc in cands[-2:]:
                direction = vc.get('direction')
                o = vc['open']; c = vc['close']; h = vc['high']; l = vc['low']
                rng = max(1e-9, vc['range'])
                half = (o + c) / 2
                # 62% retrace level (toward origin)
                if direction == 'bull':
                    entry = l + rng * 0.62
                    # prefer slightly above half to ensure fill bias
                    entry = max(entry, half)
                    raw_stop = l - vc['atr'] * 0.6
                    dir_lbl = 'LONG'; sid = 'SCALP-VEC-L'
                else:
                    entry = h - rng * 0.62
                    entry = min(entry, half)
                    raw_stop = h + vc['atr'] * 0.6
                    dir_lbl = 'SHORT'; sid = 'SCALP-VEC-S'
                # Risk and targets
                risk_abs = abs(entry - raw_stop)
                if risk_abs <= 0:
                    continue
                rr_levels = [1.5, 2.5, 3.5]
                targets = []
                for m in rr_levels:
                    tp = entry + (risk_abs * m) if dir_lbl=='LONG' else entry - (risk_abs * m)
                    targets.append({'label': f'{m}R', 'price': round(tp, 2), 'rr': m})
                # Confidence heuristic
                base_conf = 54
                # bonus if MTF consensus aligns
                mt = (multi_timeframe or {}).get('consensus', {}).get('primary')
                if dir_lbl=='LONG' and mt=='BULLISH': base_conf += 8
                if dir_lbl=='SHORT' and mt=='BEARISH': base_conf += 8
                # penalize very high ATR%
                if isinstance(atr_pct, (int, float)):
                    if atr_pct > 2.2: base_conf -= 8
                    elif atr_pct > 1.6: base_conf -= 4
                risk_pct = abs(entry - raw_stop) / max(1e-9, entry) * 100
                setups.append({
                    'id': sid,
                    'direction': dir_lbl,
                    'strategy': 'Vector Candle Scalp',
                    'timeframe': (vector_analysis or {}).get('timeframe', '5m'),
                    'entry': round(entry, 2),
                    'stop_loss': round(raw_stop, 2),
                    'risk_percent': round(risk_pct, 2),
                    'targets': targets,
                    'confidence': max(30, min(90, base_conf)),
                    'conditions': [
                        {'t': f"Vector {direction}", 's': 'ok'},
                        {'t': f"Body {vc['drivers'].get('body_ratio',0)*100:.0f}%", 's': 'ok'},
                        {'t': f"Vol x{vc['drivers'].get('vol_spike_x')}", 's': 'ok'}
                    ],
                    'rationale': 'Scalp auf Retracement der impulsiven Vector Candle',
                    'vector_ref': vc
                })
            return setups
        except Exception as e:
            self.logger.error(f"Vector scalp setup error: {e}")
            return []

    def _merge_and_prune_setups(self, base_setups, extra_setups):
        """Merge two setup lists and pick best 1 LONG and 1 SHORT by a simple rank.
        Rank = confidence * primary_rr * (1 - min(risk_pct,5)/100).
        """
        merged = (base_setups or []) + (extra_setups or [])
        if not merged:
            return []
        def _rank(s):
            rr = s.get('primary_rr') or (s['targets'][0]['rr'] if s.get('targets') else 1)
            risk_pct = s.get('risk_percent', 2.0)
            return (s.get('confidence',50)) * rr * (1 - min(risk_pct,5)/100.0)
        best_long = None; best_short = None
        for s in merged:
            if s.get('direction') == 'LONG':
                if best_long is None or _rank(s) > _rank(best_long):
                    best_long = s
            elif s.get('direction') == 'SHORT':
                if best_short is None or _rank(s) > _rank(best_short):
                    best_short = s
        pruned = []
        if best_long: pruned.append(best_long)
        if best_short: pruned.append(best_short)
        if not pruned:
            # fallback: pick highest confidence
            pruned = [sorted(merged, key=lambda x: x.get('confidence',0), reverse=True)[0]]
        # Mark origin
        for p in pruned:
            if p.get('strategy') == 'Vector Candle Scalp':
                p['selection_method'] = 'directional_top+vector'
        return pruned
