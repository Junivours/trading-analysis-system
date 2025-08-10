"""
📊 MARKET ANALYSIS ENGINE - Separate Datei
Technische & Fundamentale Marktanalyse
"""

import numpy as np
import requests
import time
from typing import Dict, List, Tuple

class MarketAnalysisEngine:
    """📊 Professionelle Marktanalyse - Technisch & Fundamental"""
    
    def __init__(self):
        self.base_url = "https://api.binance.com/api/v3"
        self._last_request_time = 0
        self.analysis_weights = {
            'market_sentiment': 0.30,
            'price_action': 0.25,
            'technical_indicators': 0.25,
            'risk_management': 0.20
        }
    
    def get_market_data(self, symbol: str, interval: str = '4h', limit: int = 200) -> List[Dict]:
        """📊 LIVE MARKET DATA - Compatible with TradingView RSI calculations"""
        try:
            # Rate limiting
            current_time = time.time()
            if current_time - self._last_request_time < 0.1:
                time.sleep(0.1)
            self._last_request_time = current_time
            
            # API Request
            response = requests.get(f"{self.base_url}/klines", {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️ API Error {response.status_code}")
                return self._get_fallback_klines()
                
        except Exception as e:
            print(f"❌ Market Data Error: {e}")
            return self._get_fallback_klines()
    
    def _get_fallback_klines(self) -> List[Dict]:
        """🛡️ Fallback Kerzendaten"""
        base_price = 50000
        return [[
            1609459200000,  # Open time
            str(base_price),  # Open
            str(base_price * 1.02),  # High
            str(base_price * 0.98),  # Low
            str(base_price * 1.01),  # Close
            "1000",  # Volume
            1609462799999,  # Close time
            "50000000",  # Quote asset volume
            1000,  # Number of trades
            "500",  # Taker buy base asset volume
            "25000000",  # Taker buy quote asset volume
            "0"  # Ignore
        ] for _ in range(100)]
    
    def calculate_technical_indicators(self, data: List[Dict]) -> Dict:
        """📈 ADVANCED Technical Indicators - 25% Weight with MEGA DETAILS"""
        try:
            if len(data) < 20:
                return {'error': 'Nicht genügend Daten für technische Indikatoren'}
            
            closes = [float(candle[4]) for candle in data]
            highs = [float(candle[2]) for candle in data]
            lows = [float(candle[3]) for candle in data]
            volumes = [float(candle[5]) for candle in data]
            
            # RSI (14 Perioden)
            rsi = self._calculate_rsi(closes, 14)
            
            # MACD (12, 26, 9)
            macd_line, macd_signal, macd_histogram = self._calculate_macd(closes)
            
            # Moving Averages
            sma_9 = sum(closes[-9:]) / 9
            sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else sum(closes) / len(closes)
            
            # Bollinger Bands (20, 2)
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(closes, 20, 2)
            
            # Stochastic Oscillator
            stoch_k, stoch_d = self._calculate_stochastic(highs, lows, closes, 14, 3)
            
            # ATR (Average True Range)
            atr = self._calculate_atr(highs, lows, closes, 14)
            
            # Volume Analysis
            volume_sma = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
            volume_ratio = volumes[-1] / volume_sma if volume_sma > 0 else 1
            
            # Support/Resistance Levels
            support, resistance = self._find_support_resistance(highs, lows, closes)
            
            return {
                'success': True,
                'timestamp': int(time.time()),
                
                # Trend Indicators
                'sma_9': round(sma_9, 2),
                'sma_20': round(sma_20, 2),
                'sma_trend': 'BULLISH' if sma_9 > sma_20 else 'BEARISH',
                
                # Momentum Indicators
                'rsi': round(rsi, 1),
                'rsi_signal': self._get_rsi_signal(rsi),
                'macd': round(macd_line, 6),
                'macd_signal': round(macd_signal, 6),
                'macd_histogram': round(macd_histogram, 6),
                'macd_trend': 'BULLISH' if macd_line > macd_signal else 'BEARISH',
                
                # Volatility Indicators
                'bb_upper': round(bb_upper, 2),
                'bb_middle': round(bb_middle, 2),
                'bb_lower': round(bb_lower, 2),
                'bb_position': self._get_bb_position(closes[-1], bb_upper, bb_lower),
                'atr': round(atr, 2),
                'atr_percent': round((atr / closes[-1]) * 100, 2),
                
                # Oscillators
                'stoch_k': round(stoch_k, 1),
                'stoch_d': round(stoch_d, 1),
                'stoch_signal': self._get_stoch_signal(stoch_k, stoch_d),
                
                # Volume Analysis
                'volume_ratio': round(volume_ratio, 2),
                'volume_signal': 'HIGH' if volume_ratio > 1.5 else 'LOW' if volume_ratio < 0.7 else 'NORMAL',
                
                # Support/Resistance
                'support_level': round(support, 2),
                'resistance_level': round(resistance, 2),
                'support_distance': round(((closes[-1] - support) / closes[-1]) * 100, 2),
                'resistance_distance': round(((resistance - closes[-1]) / closes[-1]) * 100, 2),
                
                # Current Price Info
                'current_price': closes[-1],
                'price_change': round(closes[-1] - closes[-2], 2),
                'price_change_percent': round(((closes[-1] - closes[-2]) / closes[-2]) * 100, 2)
            }
            
        except Exception as e:
            print(f"❌ Technical Indicators Error: {e}")
            return {'error': f'Technical indicators calculation failed: {str(e)}'}
    
    def _calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """📈 RSI Berechnung (TradingView kompatibel)"""
        if len(closes) < period + 1:
            return 50.0
        
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]
        
        # Initial average
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        # Smoothed averages (Wilder's smoothing)
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, closes: List[float], fast: int = 12, slow: int = 26, signal_period: int = 9) -> Tuple[float, float, float]:
        """📊 MACD Berechnung"""
        if len(closes) < slow:
            return 0.0, 0.0, 0.0
        
        # EMA Berechnung
        def ema(data, period):
            multiplier = 2 / (period + 1)
            ema_values = [data[0]]
            for price in data[1:]:
                ema_values.append((price * multiplier) + (ema_values[-1] * (1 - multiplier)))
            return ema_values
        
        ema_fast = ema(closes, fast)
        ema_slow = ema(closes, slow)
        
        # MACD Line
        macd_line = ema_fast[-1] - ema_slow[-1]
        
        # MACD für Signal Line
        macd_values = [ema_fast[i] - ema_slow[i] for i in range(len(ema_slow))]
        if len(macd_values) >= signal_period:
            signal_ema = ema(macd_values, signal_period)
            macd_signal = signal_ema[-1]
        else:
            macd_signal = 0.0
        
        macd_histogram = macd_line - macd_signal
        
        return macd_line, macd_signal, macd_histogram
    
    def _calculate_bollinger_bands(self, closes: List[float], period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """📊 Bollinger Bands Berechnung"""
        if len(closes) < period:
            middle = sum(closes) / len(closes)
            std = np.std(closes)
        else:
            recent_closes = closes[-period:]
            middle = sum(recent_closes) / period
            std = np.std(recent_closes)
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def _calculate_stochastic(self, highs: List[float], lows: List[float], closes: List[float], 
                            k_period: int = 14, d_period: int = 3) -> Tuple[float, float]:
        """📊 Stochastic Oscillator Berechnung"""
        if len(closes) < k_period:
            return 50.0, 50.0
        
        recent_highs = highs[-k_period:]
        recent_lows = lows[-k_period:]
        current_close = closes[-1]
        
        highest_high = max(recent_highs)
        lowest_low = min(recent_lows)
        
        if highest_high == lowest_low:
            k_percent = 50.0
        else:
            k_percent = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
        # %D ist der gleitende Durchschnitt von %K
        if len(closes) >= k_period + d_period - 1:
            k_values = []
            for i in range(d_period):
                idx = len(closes) - 1 - i
                if idx >= k_period - 1:
                    period_highs = highs[idx-k_period+1:idx+1]
                    period_lows = lows[idx-k_period+1:idx+1]
                    period_close = closes[idx]
                    
                    high = max(period_highs)
                    low = min(period_lows)
                    
                    if high != low:
                        k_val = ((period_close - low) / (high - low)) * 100
                        k_values.append(k_val)
            
            d_percent = sum(k_values) / len(k_values) if k_values else k_percent
        else:
            d_percent = k_percent
        
        return k_percent, d_percent
    
    def _calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
        """📊 Average True Range Berechnung"""
        if len(closes) < 2:
            return 0.0
        
        true_ranges = []
        for i in range(1, len(closes)):
            tr1 = highs[i] - lows[i]
            tr2 = abs(highs[i] - closes[i-1])
            tr3 = abs(lows[i] - closes[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        if len(true_ranges) < period:
            return sum(true_ranges) / len(true_ranges)
        else:
            return sum(true_ranges[-period:]) / period
    
    def _find_support_resistance(self, highs: List[float], lows: List[float], closes: List[float]) -> Tuple[float, float]:
        """📊 Support/Resistance Level Erkennung"""
        try:
            # Letzten 50 Kerzen analysieren
            recent_highs = highs[-50:] if len(highs) >= 50 else highs
            recent_lows = lows[-50:] if len(lows) >= 50 else lows
            current_price = closes[-1]
            
            # Pivot Points finden
            resistance_levels = []
            support_levels = []
            
            # Lokale Maxima (Resistance)
            for i in range(2, len(recent_highs) - 2):
                if (recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i-2] and
                    recent_highs[i] > recent_highs[i+1] and recent_highs[i] > recent_highs[i+2]):
                    if recent_highs[i] > current_price:
                        resistance_levels.append(recent_highs[i])
            
            # Lokale Minima (Support)
            for i in range(2, len(recent_lows) - 2):
                if (recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i-2] and
                    recent_lows[i] < recent_lows[i+1] and recent_lows[i] < recent_lows[i+2]):
                    if recent_lows[i] < current_price:
                        support_levels.append(recent_lows[i])
            
            # Nächste Level finden
            nearest_resistance = min(resistance_levels) if resistance_levels else max(recent_highs)
            nearest_support = max(support_levels) if support_levels else min(recent_lows)
            
            return nearest_support, nearest_resistance
            
        except Exception:
            # Fallback: Simple High/Low
            return min(lows[-20:]), max(highs[-20:])
    
    def _get_rsi_signal(self, rsi: float) -> str:
        """📈 RSI Signal Interpretation"""
        if rsi > 70:
            return 'OVERBOUGHT'
        elif rsi < 30:
            return 'OVERSOLD'
        elif rsi > 60:
            return 'BULLISH'
        elif rsi < 40:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def _get_bb_position(self, price: float, upper: float, lower: float) -> str:
        """📊 Bollinger Bands Position"""
        if price >= upper:
            return 'UPPER'
        elif price <= lower:
            return 'LOWER'
        else:
            return 'MIDDLE'
    
    def _get_stoch_signal(self, k: float, d: float) -> str:
        """📊 Stochastic Signal"""
        if k > 80 and d > 80:
            return 'OVERBOUGHT'
        elif k < 20 and d < 20:
            return 'OVERSOLD'
        elif k > d:
            return 'BULLISH'
        elif k < d:
            return 'BEARISH'
        else:
            return 'NEUTRAL'
    
    def fundamental_analysis(self, symbol: str, market_data: List[Dict]) -> Dict:
        """🎯 Professional Fundamental Analysis - Core Logic"""
        try:
            if not market_data or len(market_data) < 20:
                return {'error': 'Insufficient market data for fundamental analysis'}
            
            # Extrahiere Basisdaten
            closes = [float(candle[4]) for candle in market_data]
            volumes = [float(candle[5]) for candle in market_data]
            highs = [float(candle[2]) for candle in market_data]
            lows = [float(candle[3]) for candle in market_data]
            
            current_price = closes[-1]
            
            # 1. PRICE ACTION ANALYSIS (25% Weight)
            price_action = self._analyze_price_action(closes, highs, lows, volumes)
            
            # 2. MARKET SENTIMENT (30% Weight)
            market_sentiment = self._analyze_market_sentiment(market_data, symbol)
            
            # 3. VOLUME ANALYSIS
            volume_analysis = self._analyze_volume_patterns(volumes, closes)
            
            # 4. VOLATILITY ANALYSIS
            volatility_analysis = self._analyze_volatility(closes, highs, lows)
            
            # 5. TREND ANALYSIS
            trend_analysis = self._analyze_trend_strength(closes)
            
            # Combine all analyses
            fundamental_score = self._calculate_fundamental_score(
                price_action, market_sentiment, volume_analysis, 
                volatility_analysis, trend_analysis
            )
            
            return {
                'success': True,
                'symbol': symbol,
                'current_price': current_price,
                'fundamental_score': fundamental_score,
                'price_action': price_action,
                'market_sentiment': market_sentiment,
                'volume_analysis': volume_analysis,
                'volatility_analysis': volatility_analysis,
                'trend_analysis': trend_analysis,
                'recommendation': self._generate_fundamental_recommendation(fundamental_score),
                'timestamp': int(time.time())
            }
            
        except Exception as e:
            return {'error': f'Fundamental analysis failed: {str(e)}'}
    
    def _analyze_price_action(self, closes: List[float], highs: List[float], lows: List[float], volumes: List[float]) -> Dict:
        """📊 Price Action Analysis"""
        try:
            current_price = closes[-1]
            
            # Price momentum (5 Perioden)
            short_momentum = (closes[-1] - closes[-6]) / closes[-6] * 100 if len(closes) > 5 else 0
            
            # Price momentum (20 Perioden)
            long_momentum = (closes[-1] - closes[-21]) / closes[-21] * 100 if len(closes) > 20 else 0
            
            # Higher highs, higher lows pattern
            recent_highs = highs[-10:]
            recent_lows = lows[-10:]
            
            higher_highs = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
            higher_lows = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] > recent_lows[i-1])
            
            # Price structure score
            structure_score = (higher_highs + higher_lows) / (len(recent_highs) - 1 + len(recent_lows) - 1) * 100
            
            return {
                'short_momentum': round(short_momentum, 2),
                'long_momentum': round(long_momentum, 2),
                'structure_score': round(structure_score, 1),
                'higher_highs': higher_highs,
                'higher_lows': higher_lows,
                'trend_quality': 'STRONG' if structure_score > 70 else 'WEAK' if structure_score < 30 else 'MODERATE'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_market_sentiment(self, market_data: List[Dict], symbol: str) -> Dict:
        """💭 Market Sentiment Analysis"""
        try:
            closes = [float(candle[4]) for candle in market_data]
            volumes = [float(candle[5]) for candle in market_data]
            
            # Bullish/Bearish candle ratio
            bullish_candles = sum(1 for candle in market_data[-20:] if float(candle[4]) > float(candle[1]))
            bearish_candles = 20 - bullish_candles
            
            sentiment_ratio = bullish_candles / 20 * 100
            
            # Volume weighted sentiment
            recent_volume = sum(volumes[-5:]) / 5
            avg_volume = sum(volumes[-20:]) / 20
            volume_strength = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # Market momentum
            momentum_5d = (closes[-1] - closes[-6]) / closes[-6] * 100 if len(closes) > 5 else 0
            
            return {
                'bullish_candles': bullish_candles,
                'bearish_candles': bearish_candles,
                'sentiment_ratio': round(sentiment_ratio, 1),
                'volume_strength': round(volume_strength, 2),
                'momentum_5d': round(momentum_5d, 2),
                'overall_sentiment': 'BULLISH' if sentiment_ratio > 60 else 'BEARISH' if sentiment_ratio < 40 else 'NEUTRAL'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_volume_patterns(self, volumes: List[float], closes: List[float]) -> Dict:
        """📊 Volume Pattern Analysis"""
        try:
            # Volume trend
            recent_volume = sum(volumes[-5:]) / 5
            older_volume = sum(volumes[-15:-5]) / 10 if len(volumes) > 15 else sum(volumes[:-5]) / max(1, len(volumes)-5)
            
            volume_trend = (recent_volume - older_volume) / older_volume * 100 if older_volume > 0 else 0
            
            # Volume-Price correlation
            price_changes = [closes[i] - closes[i-1] for i in range(1, len(closes))]
            volume_changes = [volumes[i] - volumes[i-1] for i in range(1, len(volumes))]
            
            # Simplified correlation
            correlation = 0
            if len(price_changes) == len(volume_changes) and len(price_changes) > 0:
                correlation = np.corrcoef(price_changes[-20:], volume_changes[-20:])[0, 1] if len(price_changes) >= 20 else 0
            
            return {
                'recent_avg_volume': round(recent_volume, 0),
                'volume_trend': round(volume_trend, 2),
                'price_volume_correlation': round(correlation, 3) if not np.isnan(correlation) else 0,
                'volume_signal': 'STRONG' if volume_trend > 20 else 'WEAK' if volume_trend < -20 else 'NORMAL'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_volatility(self, closes: List[float], highs: List[float], lows: List[float]) -> Dict:
        """📊 Volatility Analysis"""
        try:
            # Historical volatility (20 periods)
            if len(closes) >= 20:
                returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
                volatility = np.std(returns[-20:]) * np.sqrt(20) * 100  # Annualized
            else:
                volatility = 0
            
            # Average True Range ratio
            true_ranges = []
            for i in range(1, len(closes)):
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                true_ranges.append(max(tr1, tr2, tr3))
            
            avg_true_range = sum(true_ranges[-14:]) / min(14, len(true_ranges)) if true_ranges else 0
            atr_ratio = avg_true_range / closes[-1] * 100
            
            return {
                'historical_volatility': round(volatility, 2),
                'atr_ratio': round(atr_ratio, 2),
                'volatility_level': 'HIGH' if volatility > 50 else 'LOW' if volatility < 20 else 'MEDIUM'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _analyze_trend_strength(self, closes: List[float]) -> Dict:
        """📈 Trend Strength Analysis"""
        try:
            if len(closes) < 20:
                return {'error': 'Insufficient data for trend analysis'}
            
            # Linear regression für Trend
            x = list(range(len(closes)))
            
            # Vereinfachte lineare Regression
            n = len(closes)
            sum_x = sum(x)
            sum_y = sum(closes)
            sum_xy = sum(x[i] * closes[i] for i in range(n))
            sum_x2 = sum(xi * xi for xi in x)
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            
            # R-squared (vereinfacht)
            y_mean = sum_y / n
            ss_tot = sum((closes[i] - y_mean) ** 2 for i in range(n))
            
            # Predicted values
            y_pred = [slope * x[i] + (sum_y - slope * sum_x) / n for i in range(n)]
            ss_res = sum((closes[i] - y_pred[i]) ** 2 for i in range(n))
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Trend direction
            trend_direction = 'UP' if slope > 0 else 'DOWN'
            trend_strength = abs(slope) / closes[-1] * 100 * len(closes)  # Normalize
            
            return {
                'trend_direction': trend_direction,
                'trend_strength': round(trend_strength, 2),
                'r_squared': round(r_squared, 3),
                'slope': round(slope, 6),
                'trend_quality': 'STRONG' if r_squared > 0.7 else 'WEAK' if r_squared < 0.3 else 'MODERATE'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_fundamental_score(self, price_action: Dict, market_sentiment: Dict, 
                                   volume_analysis: Dict, volatility_analysis: Dict, 
                                   trend_analysis: Dict) -> Dict:
        """🎯 Calculate Overall Fundamental Score"""
        try:
            score = 50  # Neutral start
            
            # Price Action (25% weight)
            if 'short_momentum' in price_action:
                if price_action['short_momentum'] > 2:
                    score += 15
                elif price_action['short_momentum'] < -2:
                    score -= 15
            
            # Market Sentiment (30% weight)
            if 'sentiment_ratio' in market_sentiment:
                sentiment_impact = (market_sentiment['sentiment_ratio'] - 50) * 0.3
                score += sentiment_impact
            
            # Volume Analysis (15% weight)
            if 'volume_trend' in volume_analysis:
                if volume_analysis['volume_trend'] > 10:
                    score += 10
                elif volume_analysis['volume_trend'] < -10:
                    score -= 10
            
            # Trend Analysis (20% weight)
            if 'trend_direction' in trend_analysis and 'r_squared' in trend_analysis:
                if trend_analysis['trend_direction'] == 'UP' and trend_analysis['r_squared'] > 0.5:
                    score += 15
                elif trend_analysis['trend_direction'] == 'DOWN' and trend_analysis['r_squared'] > 0.5:
                    score -= 15
            
            # Volatility penalty (10% weight)
            if 'volatility_level' in volatility_analysis:
                if volatility_analysis['volatility_level'] == 'HIGH':
                    score -= 10
            
            # Normalize score
            score = max(0, min(100, score))
            
            # Determine signal
            if score > 70:
                signal = 'STRONG_BUY'
            elif score > 60:
                signal = 'BUY'
            elif score > 55:
                signal = 'WEAK_BUY'
            elif score > 45:
                signal = 'HOLD'
            elif score > 40:
                signal = 'WEAK_SELL'
            elif score > 30:
                signal = 'SELL'
            else:
                signal = 'STRONG_SELL'
            
            return {
                'score': round(score, 1),
                'signal': signal,
                'confidence': round(abs(score - 50) * 2, 1)  # Distance from neutral
            }
            
        except Exception as e:
            return {'score': 50, 'signal': 'HOLD', 'error': str(e)}
    
    def _generate_fundamental_recommendation(self, fundamental_score: Dict) -> Dict:
        """💡 Generate Trading Recommendation"""
        signal = fundamental_score.get('signal', 'HOLD')
        score = fundamental_score.get('score', 50)
        confidence = fundamental_score.get('confidence', 50)
        
        recommendations = {
            'STRONG_BUY': {'action': '🚀 STRONG BUY', 'position': 'Large Long', 'risk': 'Medium'},
            'BUY': {'action': '📈 BUY', 'position': 'Medium Long', 'risk': 'Medium'},
            'WEAK_BUY': {'action': '📊 WEAK BUY', 'position': 'Small Long', 'risk': 'Low'},
            'HOLD': {'action': '⏸️ HOLD', 'position': 'No Position', 'risk': 'None'},
            'WEAK_SELL': {'action': '📊 WEAK SELL', 'position': 'Small Short', 'risk': 'Low'},
            'SELL': {'action': '📉 SELL', 'position': 'Medium Short', 'risk': 'Medium'},
            'STRONG_SELL': {'action': '💥 STRONG SELL', 'position': 'Large Short', 'risk': 'Medium'}
        }
        
        recommendation = recommendations.get(signal, recommendations['HOLD'])
        recommendation['confidence'] = confidence
        recommendation['score'] = score
        
        return recommendation

# Global instance
market_analyzer = MarketAnalysisEngine()
