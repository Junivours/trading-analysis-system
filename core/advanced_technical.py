import numpy as np
from core.technical_analysis import TechnicalAnalysis

class AdvancedTechnicalAnalysis:
    """Extended enterprise-level technical indicators separated from API layer."""
    @staticmethod
    def calculate_extended_indicators(candles):
        if len(candles) < 50:
            return {}
        closes = np.array([c['close'] for c in candles])
        highs = np.array([c['high'] for c in candles])
        lows = np.array([c['low'] for c in candles])
        volumes = np.array([c['volume'] for c in candles])
        bb_data = AdvancedTechnicalAnalysis._calculate_bollinger_bands(closes)
        stoch_data = AdvancedTechnicalAnalysis._calculate_stochastic(highs, lows, closes)
        williams_r = AdvancedTechnicalAnalysis._calculate_williams_r(highs, lows, closes)
        cci = AdvancedTechnicalAnalysis._calculate_cci(highs, lows, closes)
        atr = AdvancedTechnicalAnalysis._calculate_atr(highs, lows, closes)
        fib_levels = AdvancedTechnicalAnalysis._calculate_fibonacci_levels(highs, lows)
        ichimoku = AdvancedTechnicalAnalysis._calculate_ichimoku(highs, lows, closes)
        pivot_points = AdvancedTechnicalAnalysis._calculate_pivot_points(highs[-1], lows[-1], closes[-1])
        volume_indicators = AdvancedTechnicalAnalysis._calculate_volume_indicators(volumes, closes)
        trend_strength = AdvancedTechnicalAnalysis._calculate_trend_strength(closes)
        return {
            'bollinger_bands': bb_data,
            'stochastic': stoch_data,
            'williams_r': williams_r,
            'cci': cci,
            'atr': atr,
            'fibonacci': fib_levels,
            'ichimoku': ichimoku,
            'pivot_points': pivot_points,
            'volume_indicators': volume_indicators,
            'trend_strength': trend_strength
        }
    @staticmethod
    def _calculate_bollinger_bands(closes, period=20, std_multiplier=2):
        if len(closes) < period:
            return {}
        sma = TechnicalAnalysis._sma(closes, period)
        if len(sma) == 0:
            return {}
        stds = []
        for i in range(period, len(closes)+1):
            window = closes[i-period:i]
            stds.append(np.std(window))
        stds = np.array(stds)
        upper = sma + std_multiplier * stds
        lower = sma - std_multiplier * stds
        if len(sma) == 0:
            return {}
        price = closes[-1]
        if price > upper[-1]: pos = 'above_upper'
        elif price < lower[-1]: pos = 'below_lower'
        elif price > sma[-1]: pos = 'between_mid_upper'
        else: pos = 'between_mid_lower'
        is_squeeze = (upper[-1]-lower[-1])/sma[-1] < 0.08 if sma[-1] else False
        return {
            'middle': float(sma[-1]),
            'upper': float(upper[-1]),
            'lower': float(lower[-1]),
            'bandwidth_pct': float((upper[-1]-lower[-1])/sma[-1]*100) if sma[-1] else 0,
            'price_position': pos,
            'squeeze': is_squeeze
        }
    @staticmethod
    def _calculate_stochastic(highs, lows, closes, k_period=14, d_period=3):
        if len(closes) < k_period:
            return {}
        k_values = []
        for i in range(k_period, len(closes)+1):
            h = np.max(highs[i-k_period:i])
            l = np.min(lows[i-k_period:i])
            c = closes[i-1]
            k = ((c-l)/(h-l)*100) if h!=l else 50
            k_values.append(k)
        k_arr = np.array(k_values)
        d_values = []
        for i in range(d_period, len(k_arr)+1):
            d_values.append(np.mean(k_arr[i-d_period:i]))
        d_arr = np.array(d_values)
        if len(k_arr)==0 or len(d_arr)==0:
            return {}
        state = 'neutral'
        if k_arr[-1] > 80 and d_arr[-1] > 80: state = 'overbought'
        elif k_arr[-1] < 20 and d_arr[-1] < 20: state = 'oversold'
        elif k_arr[-1] > d_arr[-1]: state = 'bullish_cross'
        elif k_arr[-1] < d_arr[-1]: state = 'bearish_cross'
        return {'k': float(k_arr[-1]), 'd': float(d_arr[-1]), 'state': state}
    @staticmethod
    def _calculate_williams_r(highs, lows, closes, period=14):
        if len(closes) < period:
            return None
        highest_high = np.max(highs[-period:])
        lowest_low = np.min(lows[-period:])
        if highest_high == lowest_low:
            return -50
        wr = (highest_high - closes[-1]) / (highest_high - lowest_low) * -100
        return float(wr)
    @staticmethod
    def _calculate_cci(highs, lows, closes, period=20):
        if len(closes) < period:
            return None
        typical_prices = (highs + lows + closes) / 3
        sma = np.mean(typical_prices[-period:])
        mad = np.mean(np.abs(typical_prices[-period:] - sma))
        if mad == 0:
            return 0
        cci = (typical_prices[-1] - sma) / (0.015 * mad)
        return float(cci)
    @staticmethod
    def _calculate_atr(highs, lows, closes, period=14):
        if len(closes) < period + 1:
            return {'value': None, 'percentage': None}
        trs = []
        for i in range(1, len(closes)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
            trs.append(tr)
        atr = np.mean(trs[-period:])
        pct = (atr / closes[-1]) * 100 if closes[-1] else 0
        return {'value': float(atr), 'percentage': float(pct)}
    @staticmethod
    def _calculate_fibonacci_levels(highs, lows):
        if len(highs) == 0 or len(lows) == 0:
            return {}
        high = np.max(highs)
        low = np.min(lows)
        diff = high - low if high != low else 1
        levels = {
            '0.0': float(high),
            '0.236': float(high - diff * 0.236),
            '0.382': float(high - diff * 0.382),
            '0.5': float(high - diff * 0.5),
            '0.618': float(high - diff * 0.618),
            '0.786': float(high - diff * 0.786),
            '1.0': float(low)
        }
        return levels
    @staticmethod
    def _calculate_ichimoku(highs, lows, closes):
        if len(closes) < 52:
            return {}
        def conv_line():
            return (np.max(highs[-9:]) + np.min(lows[-9:])) / 2
        def base_line():
            return (np.max(highs[-26:]) + np.min(lows[-26:])) / 2
        def span_a():
            return (conv_line() + base_line()) / 2
        def span_b():
            return (np.max(highs[-52:]) + np.min(lows[-52:])) / 2
        conversion = conv_line(); base = base_line(); a = span_a(); b = span_b(); price = closes[-1]
        cloud_color = 'bullish' if a > b else 'bearish' if a < b else 'neutral'
        price_vs_cloud = 'above' if price > max(a,b) else 'below' if price < min(a,b) else 'inside'
        return {
            'conversion_line': float(conversion),
            'base_line': float(base),
            'span_a': float(a),
            'span_b': float(b),
            'cloud_color': cloud_color,
            'price_vs_cloud': price_vs_cloud
        }
    @staticmethod
    def _calculate_pivot_points(high, low, close):
        pivot = (high + low + close)/3
        r1 = 2*pivot - low; s1 = 2*pivot - high
        r2 = pivot + (high - low); s2 = pivot - (high - low)
        return {'pivot': float(pivot), 'r1': float(r1), 's1': float(s1), 'r2': float(r2), 's2': float(s2)}
    @staticmethod
    def _calculate_volume_indicators(volumes, closes, period=20):
        if len(volumes) < period:
            return {}
        vol_ma = np.mean(volumes[-period:])
        price_change = (closes[-1] - closes[-period]) / closes[-period] if closes[-period] else 0
        volume_trend = 'expansion' if volumes[-1] > vol_ma * 1.3 else 'contraction' if volumes[-1] < vol_ma * 0.7 else 'normal'
        return {
            'volume_ma': float(vol_ma),
            'current_volume': float(volumes[-1]),
            'volume_trend': volume_trend,
            'price_change_pct': float(price_change * 100)
        }
    @staticmethod
    def _calculate_trend_strength(closes, period=20):
        if len(closes) < period:
            return {}
        sma = TechnicalAnalysis._sma(closes, period)
        if len(sma) == 0:
            return {}
        recent = closes[-period:]
        slope = np.polyfit(range(len(recent)), recent, 1)[0]
        strength = abs(slope)/np.mean(recent) if np.mean(recent) else 0
        if strength > 0.01: label = 'strong_trend'
        elif strength > 0.005: label = 'moderate_trend'
        else: label = 'weak_trend'
        return {'slope': float(slope), 'strength': float(strength), 'label': label}
