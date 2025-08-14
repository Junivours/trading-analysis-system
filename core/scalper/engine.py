import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import numpy as np

from core.technical_analysis import TechnicalAnalysis

@dataclass
class ScalpConfig:
    tf: str = '1m'                  # 1m or 3m
    limit: int = 180                # candles to fetch (<=240)
    max_signals: int = 3            # return top N latest signals
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0


def _ema(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) < n:
        return np.array([])
    k = 2.0/(n+1)
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for i in range(1,len(arr)):
        out[i] = arr[i]*k + out[i-1]*(1-k)
    return out


def scan_scalp_signals(symbol: str, cfg: Optional[ScalpConfig] = None) -> Dict[str, Any]:
    cfg = cfg or ScalpConfig()
    t0 = time.time()
    tf = cfg.tf if cfg.tf in ('1m','3m') else '1m'
    limit = max(60, min(int(cfg.limit), 240))

    candles = TechnicalAnalysis.get_candle_data(symbol.upper(), limit=limit, interval=tf)
    if not candles or len(candles) < 60:
        return {'success': False, 'error': 'insufficient_data', 'have': len(candles) if candles else 0}

    closes = np.array([c['close'] for c in candles], dtype=float)
    highs = np.array([c['high'] for c in candles], dtype=float)
    lows  = np.array([c['low'] for c in candles], dtype=float)

    ema9 = _ema(closes, 9)
    ema20 = _ema(closes, 20)

    # RSI (fast)
    rsi_obj = TechnicalAnalysis._calculate_advanced_rsi(closes)
    rsi = float(rsi_obj.get('rsi')) if isinstance(rsi_obj, dict) else 50.0

    i = len(closes) - 1
    prev_hi = float(highs[i-1]) if i-1>=0 else closes[-1]
    prev_lo = float(lows[i-1]) if i-1>=0 else closes[-1]
    last = float(closes[i])
    e9 = float(ema9[i]) if i < len(ema9) and len(ema9) > 0 else last
    e20 = float(ema20[i]) if i < len(ema20) and len(ema20) > 0 else last

    signals: List[Dict[str, Any]] = []
    # BUY scalp: above EMAs and breaking previous high, RSI not overbought
    if last > e9 > e20 and last > prev_hi and rsi < cfg.rsi_overbought:
        signals.append({'type':'SCALP_BUY','reason':'Close>EMA9>EMA20 & Break prev high & RSI<OB','price': last})
    # SELL scalp: below EMAs and breaking previous low, RSI not oversold
    if last < e9 < e20 and last < prev_lo and rsi > cfg.rsi_oversold:
        signals.append({'type':'SCALP_SELL','reason':'Close<EMA9<EMA20 & Break prev low & RSI>OS','price': last})

    # Simple forming wedge detector (reuse app logic)
    def find_forming_wedge(H, L, lookback=60):
        try:
            ll = max(20, min(lookback, len(H)))
            Hs = H[-ll:]
            Ls = L[-ll:]
            idx = np.arange(len(Hs))
            uh_mask = Hs >= (np.quantile(Hs, 0.8))
            lh_mask = Ls <= (np.quantile(Ls, 0.2))
            if np.sum(uh_mask) >= 3 and np.sum(lh_mask) >= 3:
                x_u = idx[uh_mask]; y_u = Hs[uh_mask]
                x_l = idx[lh_mask]; y_l = Ls[lh_mask]
                su, iu = np.polyfit(x_u, y_u, 1)
                sl, il = np.polyfit(x_l, y_l, 1)
                band_now = float(np.max(Hs) - np.min(Ls))
                band_past = float((iu + su*0) - (il + sl*0)) if (iu is not None and il is not None) else band_now
                if su < 0 and sl > 0 and band_now < band_past*0.85:
                    return {'type':'WEDGE_FORMING','details': {'su': su, 'sl': sl, 'band_now': band_now, 'band_past': band_past}}
        except Exception:
            return None
        return None

    forming = []
    fw = find_forming_wedge(highs, lows, lookback=60)
    if fw:
        forming.append(fw)

    dt_ms = int((time.time()-t0)*1000)
    return {
        'success': True,
        'symbol': symbol.upper(),
        'interval': tf,
        'signals': signals[:cfg.max_signals],
        'forming_patterns': forming,
        'latency_ms': dt_ms,
        'rsi': rsi,
        'ema': {'ema9': e9, 'ema20': e20},
        'last': last,
    }
