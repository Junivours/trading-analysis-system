import numpy as np
import requests

class TechnicalAnalysis:
    """Core technische Analyse: Kernelemente (RSI, MACD, MA, Support/Resistance, Volumen, Trend, Momentum).
    Aus app.py extrahiert – keine Logikänderung."""
    @staticmethod
    def get_candle_data(symbol, limit=100, interval='1h'):
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {'symbol': symbol.upper(), 'interval': interval, 'limit': limit}
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            if not isinstance(data, list):
                return []
            candles = []
            for item in data:
                candles.append({
                    'timestamp': int(item[0]),
                    'time': int(item[0]),
                    'open': float(item[1]),
                    'high': float(item[2]),
                    'low': float(item[3]),
                    'close': float(item[4]),
                    'volume': float(item[5])
                })
            return candles
        except Exception as e:
            print(f"Error getting candle data ({interval}): {e}")
            return []

    @staticmethod
    def calculate_advanced_indicators(candles):
        if len(candles) < 50:
            return {}
        closes = np.array([c['close'] for c in candles])
        highs = np.array([c['high'] for c in candles])
        lows = np.array([c['low'] for c in candles])
        volumes = np.array([c['volume'] for c in candles])
        rsi_data = TechnicalAnalysis._calculate_advanced_rsi(closes)
        macd_data = TechnicalAnalysis._calculate_advanced_macd(closes)
        sma_9 = TechnicalAnalysis._sma(closes, 9)
        sma_20 = TechnicalAnalysis._sma(closes, 20)
        ema_12 = TechnicalAnalysis._ema(closes, 12)
        ema_26 = TechnicalAnalysis._ema(closes, 26)
        # Enhanced Support/Resistance with zones and strength
        support, resistance, sr_meta = TechnicalAnalysis._calculate_support_resistance_zones(highs, lows, closes)
        volume_analysis = TechnicalAnalysis._analyze_volume(volumes, closes)
        trend_analysis = TechnicalAnalysis._analyze_trend(closes, sma_9, sma_20)
        momentum = TechnicalAnalysis._calculate_momentum(closes)
        return {
            'rsi': rsi_data,
            'macd': macd_data,
            'sma_9': sma_9[-1] if len(sma_9) > 0 else closes[-1],
            'sma_20': sma_20[-1] if len(sma_20) > 0 else closes[-1],
            'ema_12': ema_12[-1] if len(ema_12) > 0 else closes[-1],
            'ema_26': ema_26[-1] if len(ema_26) > 0 else closes[-1],
            'support': support,
            'resistance': resistance,
            'support_strength': sr_meta.get('support_strength', TechnicalAnalysis._calculate_level_strength(lows, support)),
            'resistance_strength': sr_meta.get('resistance_strength', TechnicalAnalysis._calculate_level_strength(highs, resistance)),
            # New: richer zone metadata (non-breaking addition)
            'support_zones': sr_meta.get('support_zones', []),
            'resistance_zones': sr_meta.get('resistance_zones', []),
            'support_distance_pct': sr_meta.get('support_distance_pct'),
            'resistance_distance_pct': sr_meta.get('resistance_distance_pct'),
            'volume_analysis': volume_analysis,
            'trend': trend_analysis,
            'momentum': momentum,
            'current_price': closes[-1],
            'price_position': TechnicalAnalysis._calculate_price_position(closes[-1], support, resistance)
        }

    @staticmethod
    def _calculate_advanced_rsi(closes, period=14):
        n = int(period)
        if len(closes) < n + 1:
            return {'error': 'insufficient_data', 'needed': n+1, 'have': len(closes)}
        closes_arr = np.asarray(closes, dtype=float)
        deltas = np.diff(closes_arr)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains[:n].mean()
        avg_loss = losses[:n].mean()
        rsi_series = np.full(len(closes_arr), np.nan, dtype=float)
        rs = avg_gain / avg_loss if avg_loss > 1e-12 else np.inf
        rsi_series[n] = 100 - (100 / (1 + rs)) if np.isfinite(rs) else 100.0
        for i in range(n+1, len(closes_arr)):
            gain = gains[i-1]
            loss = losses[i-1]
            avg_gain = (avg_gain * (n - 1) + gain) / n
            avg_loss = (avg_loss * (n - 1) + loss) / n
            rs = avg_gain / avg_loss if avg_loss > 1e-12 else np.inf
            rsi_series[i] = 100 - (100 / (1 + rs)) if np.isfinite(rs) else 100.0
        current_rsi = float(rsi_series[-1])
        tv_rsi = current_rsi
        rsi_diff = round(abs(current_rsi - tv_rsi), 4)
        recent = rsi_series[~np.isnan(rsi_series)][-5:]
        slope = np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) >= 2 else 0
        if current_rsi >= 80:
            trend = 'overbought'; strength = 'very_strong' if slope > 0 else 'strong'
        elif current_rsi >= 70:
            trend = 'overbought_risk' if slope > 0 else 'weakening_overbought'; strength = 'strong'
        elif current_rsi <= 20:
            trend = 'oversold'; strength = 'very_strong' if slope < 0 else 'strong'
        elif current_rsi <= 30:
            trend = 'oversold_risk' if slope < 0 else 'weakening_oversold'; strength = 'strong'
        elif 40 <= current_rsi <= 60:
            trend = 'neutral'; strength = 'medium'
        else:
            trend = 'bullish_bias' if slope > 0 else 'bearish_bias' if slope < 0 else 'neutral'; strength = 'medium'
        divergence = TechnicalAnalysis._check_rsi_divergence(closes_arr[-10:], rsi_series[~np.isnan(rsi_series)][-10:] if len(rsi_series[~np.isnan(rsi_series)]) >= 10 else [])
        return {
            'rsi': round(current_rsi, 2),
            'tv_rsi': round(tv_rsi, 2),
            'rsi_diff': rsi_diff,
            'trend': trend,
            'strength': strength,
            'divergence': divergence,
            'period': n,
            'series_tail': [round(x,2) for x in rsi_series[-30:].tolist()]
        }

    @staticmethod
    def _calculate_advanced_macd(closes, fast=12, slow=26, signal=9):
        if len(closes) < slow + signal + 10:
            return {'macd': 0, 'signal': 0, 'histogram': 0, 'curve_direction': 'neutral'}
        ema_fast = TechnicalAnalysis._ema(closes, fast)
        ema_slow = TechnicalAnalysis._ema(closes, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalAnalysis._ema(macd_line, signal)
        histogram = macd_line - signal_line
        if len(histogram) >= 5:
            recent_hist = histogram[-5:]
            curve_trend = np.polyfit(range(len(recent_hist)), recent_hist, 1)[0]
            if len(histogram) >= 10:
                prev_hist = histogram[-10:-5]
                prev_trend = np.polyfit(range(len(prev_hist)), prev_hist, 1)[0]
                if prev_trend < -0.5 and curve_trend > 0.5:
                    curve_direction = 'bullish_reversal'
                elif prev_trend > 0.5 and curve_trend < -0.5:
                    curve_direction = 'bearish_reversal'
                elif curve_trend > 1.0:
                    curve_direction = 'bullish_curve'
                elif curve_trend < -1.0:
                    curve_direction = 'bearish_curve'
                else:
                    curve_direction = 'neutral'
            else:
                curve_direction = 'bullish_curve' if curve_trend > 0 else 'bearish_curve'
        else:
            curve_direction = 'neutral'
        curve_strength = 0.0
        try:
            if len(histogram) >= 6:
                recent = np.array(histogram[-6:], dtype=float)
                d1 = np.diff(recent)
                if len(d1) >= 2:
                    d2 = np.diff(d1)
                    curvature = float(np.mean(d2))
                    scale = float(np.mean(np.abs(recent)) + 1e-6)
                    norm_curv = curvature / (scale * 3)
                    curve_strength = max(-1.0, min(1.0, norm_curv))
        except Exception:
            pass
        return {
            'macd': macd_line[-1],
            'signal': signal_line[-1],
            'histogram': histogram[-1],
            'curve_direction': curve_direction,
            'trend_strength': abs(histogram[-1]) / max(abs(min(histogram)), abs(max(histogram))) if len(histogram) > 0 else 0,
            'curve_strength': curve_strength,
            'curve_strength_pct': round(curve_strength*100,2)
        }

    @staticmethod
    def _calculate_support_resistance_zones(highs, lows, closes):
        """Detect pivot-based support/resistance zones with ATR-aware clustering.
        Returns (support_price, resistance_price, meta_dict)
        meta_dict includes zones arrays and strengths."""
        try:
            H = np.asarray(highs, dtype=float)
            L = np.asarray(lows, dtype=float)
            C = np.asarray(closes, dtype=float)
            n = len(C)
            current_price = C[-1]
            # ATR% for dynamic tolerances
            atr_pct = TechnicalAnalysis._atr_percent(H, L, C, period=14)
            # Find swing pivots
            piv_hi = TechnicalAnalysis._find_pivots(H, mode='high', left=3, right=3)
            piv_lo = TechnicalAnalysis._find_pivots(L, mode='low', left=3, right=3)
            # Candidates: prices + indices
            highs_list = [(H[i], i) for i in piv_hi]
            lows_list = [(L[i], i) for i in piv_lo]
            # Cluster tolerance: tighter in calm, wider in high vol
            base_tol_pct = 0.003  # 0.30%
            if atr_pct is not None:
                # between 0.20% and 0.80% scaled by ATR
                base_tol_pct = float(min(0.008, max(0.002, (atr_pct/100.0) * 0.35)))
            res_clusters = TechnicalAnalysis._cluster_levels(highs_list, tolerance_pct=base_tol_pct)
            sup_clusters = TechnicalAnalysis._cluster_levels(lows_list, tolerance_pct=base_tol_pct)
            # Sort by distance to current price on correct side
            res_above = [z for z in res_clusters if z['center'] > current_price]
            sup_below = [z for z in sup_clusters if z['center'] < current_price]
            res_above.sort(key=lambda z: z['center'] - current_price)
            sup_below.sort(key=lambda z: current_price - z['center'])
            # Primary levels default fallbacks if missing
            resistance = res_above[0]['center'] if res_above else current_price * 1.05
            support = sup_below[0]['center'] if sup_below else current_price * 0.95
            # Strengths (1-10) from clusters
            res_strength = res_above[0]['strength'] if res_above else TechnicalAnalysis._calculate_level_strength(highs, resistance)
            sup_strength = sup_below[0]['strength'] if sup_below else TechnicalAnalysis._calculate_level_strength(lows, support)
            meta = {
                'resistance_zones': res_clusters,
                'support_zones': sup_clusters,
                'resistance_strength': int(res_strength),
                'support_strength': int(sup_strength),
                'resistance_distance_pct': float(((resistance - current_price) / current_price) * 100.0),
                'support_distance_pct': float(((current_price - support) / current_price) * 100.0),
            }
            return float(support), float(resistance), meta
        except Exception:
            # Fallback to simple heuristic
            recent_highs = highs[-20:] if len(highs) >= 20 else highs
            recent_lows = lows[-20:] if len(lows) >= 20 else lows
            current_price = closes[-1]
            resistance_candidates = [h for h in recent_highs if h > current_price * 1.001]
            resistance = min(resistance_candidates) if resistance_candidates else current_price * 1.05
            support_candidates = [l for l in recent_lows if l < current_price * 0.999]
            support = max(support_candidates) if support_candidates else current_price * 0.95
            return float(support), float(resistance), {
                'resistance_zones': [], 'support_zones': [],
                'resistance_strength': TechnicalAnalysis._calculate_level_strength(highs, resistance),
                'support_strength': TechnicalAnalysis._calculate_level_strength(lows, support),
                'resistance_distance_pct': ((resistance - current_price) / current_price) * 100.0,
                'support_distance_pct': ((current_price - support) / current_price) * 100.0,
            }

    @staticmethod
    def _find_pivots(series, mode='high', left=3, right=3):
        """Return indices of swing highs/lows using left/right window pivots."""
        idxs = []
        n = len(series)
        if n == 0:
            return idxs
        for i in range(left, n - right):
            window_left = series[i-left:i]
            window_right = series[i+1:i+1+right]
            if mode == 'high':
                if series[i] >= np.max(window_left) and series[i] > np.max(window_right):
                    idxs.append(i)
            else:
                if series[i] <= np.min(window_left) and series[i] < np.min(window_right):
                    idxs.append(i)
        # Keep last ~200 bars pivots for performance
        cutoff = max(0, n - 200)
        return [i for i in idxs if i >= cutoff]

    @staticmethod
    def _cluster_levels(points, tolerance_pct=0.003):
        """Cluster price levels with a proximity tolerance.
        points: list of (price, index). Returns list of zones with center/low/high/strength/count.
        Strength weights recent touches more."""
        if not points:
            return []
        pts = sorted(points, key=lambda x: x[0])
        clusters = []
        for price, idx in pts:
            if not clusters:
                clusters.append({'prices': [price], 'indices': [idx]})
                continue
            center = np.average(clusters[-1]['prices'], weights=TechnicalAnalysis._recency_weights(clusters[-1]['indices']))
            tol = center * tolerance_pct
            if abs(price - center) <= tol:
                clusters[-1]['prices'].append(price)
                clusters[-1]['indices'].append(idx)
            else:
                clusters.append({'prices': [price], 'indices': [idx]})
        # Build zones
        zones = []
        for c in clusters:
            prices = np.array(c['prices'], dtype=float)
            indices = np.array(c['indices'], dtype=int)
            w = TechnicalAnalysis._recency_weights(indices)
            center = float(np.average(prices, weights=w)) if np.sum(w) > 0 else float(np.mean(prices))
            low = float(np.min(prices))
            high = float(np.max(prices))
            count = int(len(prices))
            # Strength: touches weighted by recency, scaled 1..10
            raw_strength = float(np.sum(w))
            strength = TechnicalAnalysis._scale_strength(raw_strength, count)
            zones.append({'center': round(center, 6), 'low': round(low, 6), 'high': round(high, 6), 'count': count, 'strength': strength})
        # Sort by center price
        zones.sort(key=lambda z: z['center'])
        return zones

    @staticmethod
    def _recency_weights(indices):
        if len(indices) == 0:
            return np.array([1.0])
        # Normalize to 0..1 then map to 1..2 weight
        i = np.asarray(indices, dtype=float)
        if i.size == 0:
            return np.array([1.0])
        m = np.max(i) if np.max(i) > 0 else 1.0
        norm = i / m
        return 1.0 + norm  # recent touches ~2x weight

    @staticmethod
    def _scale_strength(raw, count):
        # Convert to a 1..10 bucket with mild emphasis on touch count
        score = raw * 0.7 + count * 0.6
        # Normalize by a reasonable cap
        cap = 12.0
        val = max(1.0, min(10.0, (score / cap) * 10.0))
        return int(round(val))

    @staticmethod
    def _atr_percent(highs, lows, closes, period=14):
        try:
            H = np.asarray(highs, dtype=float)
            L = np.asarray(lows, dtype=float)
            C = np.asarray(closes, dtype=float)
            if len(C) < period + 2:
                return None
            prev_close = C[:-1]
            tr1 = H[1:] - L[1:]
            tr2 = np.abs(H[1:] - prev_close)
            tr3 = np.abs(L[1:] - prev_close)
            tr = np.maximum.reduce([tr1, tr2, tr3])
            # RMA
            rma = np.zeros_like(tr)
            rma[period-1] = np.mean(tr[:period])
            for i in range(period, len(tr)):
                rma[i] = (rma[i-1]*(period-1) + tr[i]) / period
            atr = rma[-1]
            price = C[-1]
            if price <= 0:
                return None
            return float((atr / price) * 100.0)
        except Exception:
            return None

    @staticmethod
    def _calculate_level_strength(prices, level):
        touches = sum(1 for p in prices if abs(p - level) / level < 0.005)
        return min(touches, 10)

    @staticmethod
    def _analyze_volume(volumes, closes):
        if len(volumes) < 20:
            return {'trend': 'unknown', 'strength': 'weak'}
        avg_volume = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume
        price_change = (closes[-1] - closes[-2]) / closes[-2]
        if volume_ratio > 1.5:
            if price_change > 0:
                volume_trend = 'bullish_volume_surge'; strength = 'very_strong'
            else:
                volume_trend = 'bearish_volume_surge'; strength = 'very_strong'
        elif volume_ratio > 1.2:
            volume_trend = 'above_average'; strength = 'strong'
        elif volume_ratio < 0.7:
            volume_trend = 'below_average'; strength = 'weak'
        else:
            volume_trend = 'normal'; strength = 'medium'
        return {
            'trend': volume_trend,
            'strength': strength,
            'ratio': volume_ratio,
            'current': current_volume,
            'average': avg_volume
        }

    @staticmethod
    def _analyze_trend(closes, sma_9, sma_20):
        if len(closes) < 20 or len(sma_9) == 0 or len(sma_20) == 0:
            return {'trend': 'neutral', 'strength': 'weak'}
        current_price = closes[-1]
        ma_bullish = sma_9[-1] > sma_20[-1]
        price_above_ma = current_price > sma_9[-1] and current_price > sma_20[-1]
        short_term_change = (closes[-1] - closes[-5]) / closes[-5] if len(closes) >= 5 else 0
        medium_term_change = (closes[-1] - closes[-10]) / closes[-10] if len(closes) >= 10 else 0
        if ma_bullish and price_above_ma and short_term_change > 0.02:
            trend = 'strong_bullish'; strength = 'very_strong'
        elif ma_bullish and price_above_ma:
            trend = 'bullish'; strength = 'strong'
        elif not ma_bullish and not price_above_ma and short_term_change < -0.02:
            trend = 'strong_bearish'; strength = 'very_strong'
        elif not ma_bullish and not price_above_ma:
            trend = 'bearish'; strength = 'strong'
        elif abs(short_term_change) < 0.005:
            trend = 'sideways'; strength = 'weak'
        else:
            trend = 'neutral'; strength = 'medium'
        return {
            'trend': trend,
            'strength': strength,
            'short_term_momentum': short_term_change,
            'medium_term_momentum': medium_term_change,
            'ma_alignment': 'bullish' if ma_bullish else 'bearish'
        }

    @staticmethod
    def _calculate_momentum(closes, period=10):
        if len(closes) < period:
            return {'value': 0, 'trend': 'neutral'}
        momentum = (closes[-1] - closes[-period]) / closes[-period] * 100
        if momentum > 5:
            trend = 'very_bullish'
        elif momentum > 2:
            trend = 'bullish'
        elif momentum < -5:
            trend = 'very_bearish'
        elif momentum < -2:
            trend = 'bearish'
        else:
            trend = 'neutral'
        return {'value': momentum, 'trend': trend}

    @staticmethod
    def _calculate_price_position(current_price, support, resistance):
        if resistance <= support:
            return 0.5
        position = (current_price - support) / (resistance - support)
        return max(0, min(1, position))

    @staticmethod
    def _check_rsi_divergence(prices, rsi_values):
        if len(prices) < 5 or len(rsi_values) < 5:
            return 'none'
        price_trend = np.polyfit(range(len(prices)), prices, 1)[0]
        rsi_trend = np.polyfit(range(len(rsi_values)), rsi_values, 1)[0]
        if price_trend > 0 and rsi_trend < 0:
            return 'bearish_divergence'
        elif price_trend < 0 and rsi_trend > 0:
            return 'bullish_divergence'
        else:
            return 'none'

    @staticmethod
    def _sma(data, window):
        if len(data) < window:
            return np.array([])
        return np.array([np.mean(data[i-window:i]) for i in range(window, len(data) + 1)])

    @staticmethod
    def _ema(data, window):
        if len(data) < window:
            return np.zeros(len(data))
        alpha = 2 / (window + 1)
        ema = np.zeros(len(data))
        for i in range(window):
            ema[i] = np.mean(data[:i+1])
        for i in range(window, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i-1]
        return ema
