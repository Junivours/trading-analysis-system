import os, time, json, hashlib, math, logging
import numpy as np
import requests
from datetime import datetime
from core.technical_analysis import TechnicalAnalysis
from core.advanced_technical import AdvancedTechnicalAnalysis
from core.patterns import AdvancedPatternDetector, ChartPatternTrader
from core.position import PositionManager
from core.ai import AdvancedJAXAI
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
        self.ai_system = AdvancedJAXAI()
        self.symbol_profiler = SymbolBehaviorProfiler()
        self.weights = {'technical':0.70,'patterns':0.20,'ai':0.10}
        self.logger = logging.getLogger("master_analyzer")

    # ============================= PUBLIC API METHODS ============================= #
    def run_backtest(self, symbol, interval='1h', limit=500):
        """Lightweight RSI mean-reversion backtest (educational)."""
        try:
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
                        position = {'entry_price': price, 'entry_time': times[i], 'stop': price - 2 * atr_vals[i] if atr_vals[i] > 0 else price * 0.98}
                else:
                    stop_hit = price <= position['stop']
                    take = rsi_vals[i] > 55
                    if stop_hit or take:
                        ret_pct = (price - position['entry_price']) / position['entry_price'] * 100
                        equity *= (1 + ret_pct/100)
                        trades.append({'entry_time': position['entry_time'], 'exit_time': times[i], 'entry': round(position['entry_price'],6), 'exit': round(price,6), 'return_pct': round(ret_pct,2), 'outcome': 'win' if ret_pct > 0 else 'loss'})
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
                    'risk_adjusted_return_ratio': risk_adjusted_return
                },
                'trades': trades[-120:],
                'equity_curve': equity_curve[-250:],
                'disclaimer': 'Backtest ist vereinfacht. Vergangene Performance garantiert keine Zukunft.'
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
            position_analysis = self.position_manager.analyze_position_potential(current_price, tech_analysis.get('support'), tech_analysis.get('resistance'), tech_analysis.get('trend', {}), pattern_analysis)
            # Order flow
            order_flow_data = self._analyze_order_flow(symbol, current_price, tech_analysis.get('volume_analysis', {}), multi_timeframe)
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
            timings['ai_ms'] = round((time.time()-t_phase)*1000,2)
            t_phase = time.time()
            final_score = self._calculate_weighted_score(tech_analysis, pattern_analysis, ai_analysis)
            timings['scoring_ms'] = round((time.time()-t_phase)*1000,2)
            liquidation_long = self.liquidation_calc.calculate_liquidation_levels(current_price, 'long')
            liquidation_short = self.liquidation_calc.calculate_liquidation_levels(current_price, 'short')
            regime_data = self._detect_market_regime(candles, tech_analysis, extended_analysis, pattern_analysis, multi_timeframe)
            t_phase = time.time()
            trade_setups = self._generate_trade_setups(symbol, current_price, tech_analysis, extended_analysis, pattern_analysis, final_score, multi_timeframe, regime_data)
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

    def _analyze_feature_contributions(self, features, ai_analysis, tech_analysis, pattern_analysis):
        try:
            if not isinstance(features,(list,np.ndarray)) or len(features)==0:
                return {'error':'No features available for contribution analysis'}
            if isinstance(features,list): features=np.array(features)
            feature_names=['RSI','RSI_Overbought','RSI_Oversold','MACD','MACD_Hist','MACD_Bull_Curve','MACD_Bear_Curve','MACD_Bull_Rev','MACD_Bear_Rev','SMA_9','SMA_20','Support_Strength','Resistance_Strength']
            for i in range(len(feature_names),50): feature_names.append(f'Tech_{i}')
            for i in range(50,80): feature_names.append(f'Pattern_{i-50}')
            for i in range(80,100): feature_names.append(f'Market_{i-80}')
            for i in range(100,120): feature_names.append(f'Position_{i-100}')
            for i in range(120,128): feature_names.append(f'Time_{i-120}')
            while len(feature_names) < len(features): feature_names.append(f'Feature_{len(feature_names)}')
            feature_magnitudes = np.abs(features)
            feature_activations = np.where(features>0, features, -features*0.5)
            importance_scores = feature_magnitudes * feature_activations
            total_importance = np.sum(importance_scores)
            normalized_importance = (importance_scores/total_importance*100) if total_importance>0 else np.zeros_like(importance_scores)
            top_indices = np.argsort(normalized_importance)[-10:][::-1]
            contributions=[]
            for idx in top_indices:
                if idx < len(feature_names) and normalized_importance[idx] > 0.5:
                    contributions.append({'feature': feature_names[idx],'importance': round(float(normalized_importance[idx]),2),'value': round(float(features[idx]),4),'impact':'positive' if features[idx]>0 else 'negative'})
            interpretations=[]
            rsi_val = tech_analysis.get('rsi', {}).get('rsi',50)
            if rsi_val > 70: interpretations.append('RSI overbought condition reducing BUY confidence')
            elif rsi_val < 30: interpretations.append('RSI oversold condition increasing BUY potential')
            pattern_count = len(pattern_analysis.get('patterns', []))
            if pattern_count>0:
                bull_patterns = sum(1 for p in pattern_analysis.get('patterns', []) if p.get('signal')=='bullish')
                if bull_patterns>0: interpretations.append(f'{bull_patterns} bullish patterns supporting upside')
            ai_signal = ai_analysis.get('signal','HOLD'); ai_conf = ai_analysis.get('confidence',0)
            if ai_conf > 70: interpretations.append(f'High AI confidence ({ai_conf:.1f}%) in {ai_signal} signal')
            return {'top_features': contributions[:5],'total_features_analyzed': len(features),'ai_signal_confidence': ai_conf,'contextual_interpretations': interpretations,'analysis_method':'magnitude_activation_heuristic','note':'Simplified feature attribution'}
        except Exception as e:
            return {'error': f'Feature contribution analysis failed: {str(e)}','top_features': [],'analysis_method':'error'}

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

    def _calculate_weighted_score(self, tech_analysis, pattern_analysis, ai_analysis):
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
            removed=dyn_weights.get('ai',0); dyn_weights['ai']=0.0; rem=dyn_weights['technical']+dyn_weights['patterns']
            if rem<=0: dyn_weights['technical']=0.7; dyn_weights['patterns']=0.3
            else:
                dyn_weights['technical']=dyn_weights['technical']/rem
                dyn_weights['patterns']=dyn_weights['patterns']/rem
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
        if final_score>=75: signal='STRONG_BUY'; signal_color='#28a745'
        elif final_score>=60: signal='BUY'; signal_color='#6f42c1'
        elif final_score<=25: signal='STRONG_SELL'; signal_color='#dc3545'
        elif final_score<=40: signal='SELL'; signal_color='#fd7e14'
        else: signal='HOLD'; signal_color='#6c757d'
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
        ai_reason=None
        if dyn_weights.get('ai',0)==0 and self.weights.get('ai',0)>0:
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

            def _targets(entry, stop, direction, extra=None):
                risk_abs = (entry - stop) if direction == 'LONG' else (stop - entry)
                if risk_abs < min_atr * 1.2:  # ensure sufficiently wide stop
                    if direction == 'LONG':
                        stop = entry - min_atr * 1.2
                    else:
                        stop = entry + min_atr * 1.2
                    risk_abs = (entry - stop) if direction == 'LONG' else (stop - entry)
                risk_abs = max(risk_abs, min_atr * 1.0)
                base_targets = []
                for m in [1.5, 2.5, 4, 6, 8]:
                    tp = entry + risk_abs * m if direction == 'LONG' else entry - risk_abs * m
                    base_targets.append({'label': f'{m}R', 'price': round(tp, 2), 'rr': float(m)})
                swing_ext = _structural_targets(direction, entry)
                base_targets.append({'label': 'Swing', 'price': round(swing_ext, 2), 'rr': round(abs((swing_ext - entry) / risk_abs), 2)})
                if extra:
                    for lbl, lvl in extra:
                        if lvl:
                            rr = (lvl - entry) / risk_abs if direction == 'LONG' else (entry - lvl) / risk_abs
                            base_targets.append({'label': lbl, 'price': round(lvl, 2), 'rr': round(rr, 2)})
                base_targets.sort(key=lambda x: x['rr'])
                filtered = []
                last_rr = -999
                for t in base_targets:
                    if t['rr'] - last_rr >= 0.8:
                        filtered.append(t)
                        last_rr = t['rr']
                    if len(filtered) >= 5:
                        break
                return filtered

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
                    stop_pb = support - atr_val * 0.9
                    risk_pct = round((entry_pb - stop_pb) / entry_pb * 100, 2)
                    if risk_pct <= 3.0:
                        base_rationale = 'Multi-validated Einstieg nahe Support mit Professional Risk Management'
                        enhanced_rationale = f"{base_rationale}. {rsi_caution['narrative']}" if rsi_caution['caution_level'] != 'none' else base_rationale
                        setups.append({
                            'id': 'L-PB', 'direction': 'LONG', 'strategy': 'Professional Bullish Pullback',
                            'entry': round(entry_pb, 2), 'stop_loss': round(stop_pb, 2),
                            'risk_percent': risk_pct,
                            'targets': _targets(entry_pb, stop_pb, 'LONG', [
                                ('Resistance', resistance), ('Fib 0.382', fib.get('fib_382')), ('Fib 0.618', fib.get('fib_618'))
                            ]),
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
                stop_bo = resistance - atr_val
                setups.append({
                    'id': 'L-BO', 'direction': 'LONG', 'strategy': 'Resistance Breakout',
                    'entry': round(entry_bo, 2), 'stop_loss': round(stop_bo, 2),
                    'risk_percent': round((entry_bo - stop_bo) / entry_bo * 100, 2),
                    'targets': _targets(entry_bo, stop_bo, 'LONG', [
                        ('Fib 0.618', fib.get('fib_618')), ('Fib 0.786', fib.get('fib_786'))
                    ]),
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
                    stop_pc = entry_pc - atr_val * 0.8
                    setups.append({
                        'id': 'L-PC', 'direction': 'LONG', 'strategy': 'Pattern Confirmation',
                        'entry': round(entry_pc, 2), 'stop_loss': round(stop_pc, 2),
                        'risk_percent': round((entry_pc - stop_pc) / entry_pc * 100, 2),
                        'targets': _targets(entry_pc, stop_pc, 'LONG', [('Resistance', resistance)]),
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
                    stop_momo = entry_momo - atr_val
                    setups.append({
                        'id': 'L-MOMO', 'direction': 'LONG', 'strategy': 'Momentum Continuation',
                        'entry': round(entry_momo, 2), 'stop_loss': round(stop_momo, 2),
                        'risk_percent': round((entry_momo - stop_momo) / entry_momo * 100, 2),
                        'targets': _targets(entry_momo, stop_momo, 'LONG', [('Resistance', resistance)]),
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
                    stop_rej = support - atr_val * 0.7
                    setups.append({
                        'id': 'L-REJ', 'direction': 'LONG', 'strategy': 'Support Rejection',
                        'entry': round(entry_rej, 2), 'stop_loss': round(stop_rej, 2),
                        'risk_percent': round((entry_rej - stop_rej) / entry_rej * 100, 2),
                        'targets': _targets(entry_rej, stop_rej, 'LONG', [('Resistance', resistance)]),
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
                stop_mr = entry_mr - atr_val * 0.9
                setups.append({
                    'id': 'L-MR', 'direction': 'LONG', 'strategy': 'RSI Mean Reversion',
                    'entry': round(entry_mr, 2), 'stop_loss': round(stop_mr, 2),
                    'risk_percent': round((entry_mr - stop_mr) / entry_mr * 100, 2),
                    'targets': _targets(entry_mr, stop_mr, 'LONG', [('Resistance', resistance)]),
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
                        setups.append({
                            'id': 'S-PB', 'direction': 'SHORT', 'strategy': 'Professional Bearish Pullback',
                            'entry': round(entry_pbs, 2), 'stop_loss': round(stop_pbs, 2),
                            'risk_percent': risk_pct_short,
                            'targets': _targets(entry_pbs, stop_pbs, 'SHORT', [('Support', support), ('Fib 0.382', fib.get('fib_382'))]),
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
                setups.append({
                    'id': 'S-BD', 'direction': 'SHORT', 'strategy': 'Support Breakdown',
                    'entry': round(entry_bd, 2), 'stop_loss': round(stop_bd, 2),
                    'risk_percent': round((stop_bd - entry_bd) / entry_bd * 100, 2),
                    'targets': _targets(entry_bd, stop_bd, 'SHORT', [('Fib 0.236', fib.get('fib_236'))]),
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
                    setups.append({
                        'id': 'S-PC', 'direction': 'SHORT', 'strategy': 'Pattern Confirmation',
                        'entry': round(entry_ps, 2), 'stop_loss': round(stop_ps, 2),
                        'risk_percent': round((stop_ps - entry_ps) / entry_ps * 100, 2),
                        'targets': _targets(entry_ps, stop_ps, 'SHORT', [('Support', support)]),
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
                    setups.append({
                        'id': 'S-MOMO', 'direction': 'SHORT', 'strategy': 'Momentum Continuation',
                        'entry': round(entry_momo_s, 2), 'stop_loss': round(stop_momo_s, 2),
                        'risk_percent': round((stop_momo_s - entry_momo_s) / entry_momo_s * 100, 2),
                        'targets': _targets(entry_momo_s, stop_momo_s, 'SHORT', [('Support', support)]),
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
                    setups.append({
                        'id': 'S-REJ', 'direction': 'SHORT', 'strategy': 'Resistance Rejection',
                        'entry': round(entry_rej_s, 2), 'stop_loss': round(stop_rej_s, 2),
                        'risk_percent': round((stop_rej_s - entry_rej_s) / entry_rej_s * 100, 2),
                        'targets': _targets(entry_rej_s, stop_rej_s, 'SHORT', [('Support', support)]),
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
                stop_mrs = entry_mrs + atr_val * 0.9
                setups.append({
                    'id': 'S-MR', 'direction': 'SHORT', 'strategy': 'RSI Mean Reversion',
                    'entry': round(entry_mrs, 2), 'stop_loss': round(stop_mrs, 2),
                    'risk_percent': round((stop_mrs - entry_mrs) / entry_mrs * 100, 2),
                    'targets': _targets(entry_mrs, stop_mrs, 'SHORT', [('Support', support)]),
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
                stop_pat = current_price - atr_val
                setups.append({
                    'id': 'L-PAT', 'direction': 'LONG', 'strategy': 'Pattern Boost Long',
                    'entry': round(entry_pat, 2), 'stop_loss': round(stop_pat, 2),
                    'risk_percent': round((entry_pat - stop_pat) / entry_pat * 100, 2),
                    'targets': _targets(entry_pat, stop_pat, 'LONG', [('Resistance', resistance)]),
                    'confidence': 55,
                    'conditions': [{'t': 'Bullish Pattern', 's': 'ok'}],
                    'rationale': 'Bullish Chart Pattern aktiviert (relaxed)'
                })
                relaxation['pattern_injected'] = True
            if bearish_pattern_present and len([s for s in setups if s['direction'] == 'SHORT']) < 2:
                entry_pats = current_price * 0.999
                stop_pats = current_price + atr_val
                setups.append({
                    'id': 'S-PAT', 'direction': 'SHORT', 'strategy': 'Pattern Boost Short',
                    'entry': round(entry_pats, 2), 'stop_loss': round(stop_pats, 2),
                    'risk_percent': round((stop_pats - entry_pats) / entry_pats * 100, 2),
                    'targets': _targets(entry_pats, stop_pats, 'SHORT', [('Support', support)]),
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
                stop_gl = entry_gl - generic_risk
                setups.append({
                    'id': 'L-FB', 'direction': 'LONG', 'strategy': 'Generic Long',
                    'entry': round(entry_gl, 2), 'stop_loss': round(stop_gl, 2),
                    'risk_percent': round((entry_gl - stop_gl) / entry_gl * 100, 2),
                    'targets': _targets(entry_gl, stop_gl, 'LONG', [('Resistance', resistance)]),
                    'confidence': 45,
                    'conditions': [{'t': 'Fallback', 's': 'info'}],
                    'rationale': 'Fallback Long Setup (relaxed)'
                })
                entry_gs = current_price
                stop_gs = entry_gs + generic_risk
                setups.append({
                    'id': 'S-FB', 'direction': 'SHORT', 'strategy': 'Generic Short',
                    'entry': round(entry_gs, 2), 'stop_loss': round(stop_gs, 2),
                    'risk_percent': round((stop_gs - entry_gs) / entry_gs * 100, 2),
                    'targets': _targets(entry_gs, stop_gs, 'SHORT', [('Support', support)]),
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
            return trimmed
        except Exception as e:
            self.logger.error(f"Trade setup generation error: {e}")
            return []
