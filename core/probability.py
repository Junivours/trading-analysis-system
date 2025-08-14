import math
from typing import Dict, Any, List, Tuple

import numpy as np

from core.technical_analysis import TechnicalAnalysis


class DropOutcomeModel:
    """
    Empirical, technical-only estimator of bounce vs continuation probability after a drop.

    Method: On each bar t, detect if a drop of at least `drop_pct` occurred within the last `window_bars` bars
    using close-to-close returns: close[t] / close[t-k] - 1 <= -drop_pct.
    For each detected event at t, compute the future return over `horizon_bars`: close[t+h] / close[t] - 1.
    - If future return > 0 -> bounce
    - If future return < 0 -> continuation
    - Else -> neutral

    Returns empirical probabilities and magnitudes, plus a quick RSI-conditional breakdown.
    """

    @staticmethod
    def estimate_drop_outcomes(
        symbol: str,
        interval: str = "1h",
        limit: int = 1500,
        drop_pct: float = 10.0,
        window_bars: int = 24,
        horizon_bars: int = 6,
        min_gap_between_events: int = 3,
    ) -> Dict[str, Any]:
        drop_frac = float(drop_pct) / 100.0
        window = int(max(1, window_bars))
        horizon = int(max(1, horizon_bars))
        gap = int(max(0, min_gap_between_events))

        candles = TechnicalAnalysis.get_candle_data(symbol, limit=limit, interval=interval)
        if not candles or len(candles) < (window + horizon + 10):
            return {
                'success': False,
                'error': 'insufficient_data',
                'have': len(candles) if candles else 0,
                'need': (window + horizon + 10),
            }

        closes = np.array([c['close'] for c in candles], dtype=float)
        rsi_data = TechnicalAnalysis._calculate_advanced_rsi(closes)
        # The advanced RSI returns only the last ~30 values; compute a full-length RSI to align
        try:
            rsi_full = DropOutcomeModel._rsi_full(closes, period=int(rsi_data.get('period', 14)))
        except Exception:
            rsi_full = np.full_like(closes, np.nan)

        n = len(closes)
        events: List[Tuple[int, int]] = []  # (t_index, k_used)
        last_event_idx = -10_000
        for t in range(window, n - horizon):
            # Skip if within gap of previous event
            if t - last_event_idx <= gap:
                continue
            # Find any k in [1, window] with drop condition
            base_price = closes[t]
            found_k = 0
            for k in range(1, window + 1):
                prev = closes[t - k]
                if prev > 0 and (base_price / prev - 1.0) <= -drop_frac:
                    found_k = k
                    break
            if found_k > 0:
                events.append((t, found_k))
                last_event_idx = t

        outcomes = []  # (future_ret, rsi_at_t)
        for (t, k) in events:
            ft = t + horizon
            if ft >= n:
                continue
            future_ret = (closes[ft] / closes[t]) - 1.0
            rsi_t = float(rsi_full[t]) if t < len(rsi_full) else np.nan
            outcomes.append((float(future_ret), rsi_t))

        total = len(outcomes)
        if total == 0:
            return {
                'success': True,
                'symbol': symbol.upper(),
                'interval': interval,
                'drop_pct': drop_pct,
                'window_bars': window,
                'horizon_bars': horizon,
                'events': 0,
                'message': 'No historical drops matching the condition.',
            }

        future_returns = np.array([o[0] for o in outcomes], dtype=float)
        rsi_at_t = np.array([o[1] for o in outcomes], dtype=float)

        bounce_mask = future_returns > 0
        cont_mask = future_returns < 0
        neutral_mask = ~bounce_mask & ~cont_mask

        def pct(x: int) -> float:
            return round(100.0 * (float(x) / float(total)), 2)

        bounce_prob = pct(int(np.sum(bounce_mask)))
        continue_prob = pct(int(np.sum(cont_mask)))
        neutral_prob = max(0.0, 100.0 - bounce_prob - continue_prob)

        stats = {
            'success': True,
            'symbol': symbol.upper(),
            'interval': interval,
            'drop_pct': drop_pct,
            'window_bars': window,
            'horizon_bars': horizon,
            'events': total,
            'probabilities': {
                'bounce_up_pct': bounce_prob,
                'continue_down_pct': continue_prob,
                'neutral_pct': neutral_prob,
            },
            'magnitudes': {
                'avg_future_return_pct': round(float(np.mean(future_returns)) * 100.0, 3),
                'median_future_return_pct': round(float(np.median(future_returns)) * 100.0, 3),
                'avg_bounce_return_pct': round(float(np.mean(future_returns[bounce_mask])) * 100.0, 3) if np.any(bounce_mask) else None,
                'avg_continue_return_pct': round(float(np.mean(future_returns[cont_mask])) * 100.0, 3) if np.any(cont_mask) else None,
            },
            'conditioning': DropOutcomeModel._conditional_by_rsi(rsi_at_t, future_returns),
        }

        # Detect if current bar is in a drop condition
        t_now = n - 1
        current_drop_k = 0
        for k in range(1, window + 1):
            prev = closes[t_now - k]
            if prev > 0 and (closes[t_now] / prev - 1.0) <= -drop_frac:
                current_drop_k = k
                break
        if current_drop_k:
            stats['current_event'] = {
                'detected': True,
                'k_bars': current_drop_k,
                'drop_from_pct': round(((closes[t_now] / closes[t_now - current_drop_k]) - 1.0) * 100.0, 2),
                'rsi_now': float(rsi_full[t_now]) if not math.isnan(rsi_full[t_now]) else None,
            }
        else:
            stats['current_event'] = {'detected': False}

        return stats

    @staticmethod
    def _rsi_full(closes: np.ndarray, period: int = 14) -> np.ndarray:
        """Vectorized RSI for full series to align features."""
        if closes.size < period + 1:
            return np.full_like(closes, np.nan, dtype=float)
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        rsi = np.full(closes.shape, np.nan, dtype=float)
        # Wilder's smoothing (RMA)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        rs = (avg_gain / avg_loss) if avg_loss > 1e-12 else np.inf
        rsi[period] = 100.0 - (100.0 / (1.0 + rs)) if np.isfinite(rs) else 100.0
        for i in range(period + 1, closes.size):
            gain = gains[i - 1]
            loss = losses[i - 1]
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period
            rs = (avg_gain / avg_loss) if avg_loss > 1e-12 else np.inf
            rsi[i] = 100.0 - (100.0 / (1.0 + rs)) if np.isfinite(rs) else 100.0
        return rsi

    @staticmethod
    def _conditional_by_rsi(rsi_at_t: np.ndarray, future_returns: np.ndarray) -> Dict[str, Any]:
        try:
            buckets = {
                'RSI<=25': (rsi_at_t <= 25),
                '25<RSI<=35': (rsi_at_t > 25) & (rsi_at_t <= 35),
                '35<RSI<=45': (rsi_at_t > 35) & (rsi_at_t <= 45),
                '45<RSI<=55': (rsi_at_t > 45) & (rsi_at_t <= 55),
                '55<RSI<=65': (rsi_at_t > 55) & (rsi_at_t <= 65),
                'RSI>65': (rsi_at_t > 65),
            }
            out: Dict[str, Any] = {}
            total = future_returns.size
            for name, mask in buckets.items():
                sel = future_returns[mask]
                n = int(sel.size)
                if n == 0:
                    out[name] = {'events': 0}
                    continue
                b = int(np.sum(sel > 0))
                c = int(np.sum(sel < 0))
                out[name] = {
                    'events': n,
                    'bounce_up_pct': round(100.0 * b / n, 2),
                    'continue_down_pct': round(100.0 * c / n, 2),
                    'avg_future_return_pct': round(float(np.mean(sel)) * 100.0, 3),
                }
            return out
        except Exception:
            return {}
