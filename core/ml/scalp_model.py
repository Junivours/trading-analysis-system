import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

import numpy as np

from core.technical_analysis import TechnicalAnalysis


MODEL_PATH = os.path.join('data', 'models', 'scalp_logreg.json')


@dataclass
class ScalpTrainConfig:
    symbol: str
    interval: str = '1m'
    limit: int = 3000
    horizon_bars: int = 8
    target_up_pct: float = 0.35  # +0.35% within horizon
    stop_down_pct: float = 0.35  # -0.35% within horizon


def _ema(arr: np.ndarray, n: int) -> np.ndarray:
    if len(arr) < n:
        return np.array([])
    k = 2.0/(n+1)
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = arr[i]*k + out[i-1]*(1-k)
    return out


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> np.ndarray:
    if len(close) < n + 1:
        return np.array([])
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    atr = np.zeros_like(close)
    atr[:n] = np.nan
    # Wilder's smoothing
    atr_val = np.mean(tr[:n])
    atr[n] = atr_val
    for i in range(n+1, len(close)):
        atr_val = (atr_val*(n-1) + tr[i-1]) / n
        atr[i] = atr_val
    return atr


def _rsi(closes: np.ndarray, n: int = 14) -> np.ndarray:
    if len(closes) < n + 1:
        return np.array([])
    deltas = np.diff(closes)
    ups = np.where(deltas > 0, deltas, 0.0)
    downs = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.zeros_like(closes)
    avg_loss = np.zeros_like(closes)
    avg_gain[:n] = np.nan
    avg_loss[:n] = np.nan
    gain = np.mean(ups[:n])
    loss = np.mean(downs[:n])
    avg_gain[n] = gain
    avg_loss[n] = loss
    for i in range(n+1, len(closes)):
        gain = (gain*(n-1) + ups[i-1]) / n
        loss = (loss*(n-1) + downs[i-1]) / n
        avg_gain[i] = gain
        avg_loss[i] = loss
    rs = np.divide(avg_gain, np.where(avg_loss == 0, np.nan, avg_loss))
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _mk_features(candles: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    closes = np.array([c['close'] for c in candles], dtype=float)
    highs = np.array([c['high'] for c in candles], dtype=float)
    lows = np.array([c['low'] for c in candles], dtype=float)
    vols = np.array([c.get('volume', 0.0) or 0.0 for c in candles], dtype=float)
    ema9 = _ema(closes, 9)
    ema20 = _ema(closes, 20)
    rsi = _rsi(closes, 14)
    atr = _atr(highs, lows, closes, 14)
    ret1 = np.zeros_like(closes)
    ret1[1:] = (closes[1:] / closes[:-1] - 1.0) * 100.0
    vol_ma = np.convolve(vols, np.ones(20)/20.0, mode='same')
    vol_ratio = np.divide(vols, np.where(vol_ma == 0, 1.0, vol_ma))
    # previous H/L
    prev_high = np.roll(highs, 1)
    prev_low = np.roll(lows, 1)
    prev_high[0] = highs[0]
    prev_low[0] = lows[0]
    broke_high = (closes > prev_high).astype(float)
    broke_low = (closes < prev_low).astype(float)
    dist_ph = (closes / np.where(prev_high == 0, closes, prev_high) - 1.0) * 100.0
    dist_pl = (closes / np.where(prev_low == 0, closes, prev_low) - 1.0) * 100.0
    ema9_slope = np.zeros_like(closes)
    ema20_slope = np.zeros_like(closes)
    ema9_slope[1:] = (ema9[1:] - ema9[:-1]) / np.where(closes[1:] == 0, 1.0, closes[1:])
    ema20_slope[1:] = (ema20[1:] - ema20[:-1]) / np.where(closes[1:] == 0, 1.0, closes[1:])
    above_e9 = (closes >= ema9).astype(float)
    above_e20 = (closes >= ema20).astype(float)

    feats = {
        'rsi': rsi,
        'atr_pct': (atr / np.where(closes == 0, 1.0, closes)) * 100.0,
        'ret1': ret1,
        'vol_ratio': vol_ratio,
        'dist_prev_high_pct': dist_ph,
        'dist_prev_low_pct': dist_pl,
        'broke_prev_high': broke_high,
        'broke_prev_low': broke_low,
        'ema9_slope': ema9_slope,
        'ema20_slope': ema20_slope,
        'above_e9': above_e9,
        'above_e20': above_e20,
    }
    return feats


def _stack_features(feats: Dict[str, np.ndarray], start: int, end: int) -> np.ndarray:
    keys = [
        'rsi','atr_pct','ret1','vol_ratio','dist_prev_high_pct','dist_prev_low_pct',
        'broke_prev_high','broke_prev_low','ema9_slope','ema20_slope','above_e9','above_e20'
    ]
    cols = [feats[k][start:end] for k in keys]
    X = np.vstack(cols).T
    # replace nans
    X = np.where(np.isfinite(X), X, 0.0)
    return X


def _make_labels(closes: np.ndarray, start: int, end: int, horizon: int, up_pct: float, dn_pct: float) -> np.ndarray:
    y = np.zeros(end - start, dtype=int)
    for i in range(start, end):
        base = closes[i]
        if base <= 0:
            y[i - start] = 0
            continue
        up_th = base * (1.0 + up_pct/100.0)
        dn_th = base * (1.0 - dn_pct/100.0)
        hi = np.max(closes[i+1:i+1+horizon]) if i+1+horizon <= len(closes) else np.max(closes[i+1:])
        lo = np.min(closes[i+1:i+1+horizon]) if i+1+horizon <= len(closes) else np.min(closes[i+1:])
        # success if hit up target before down stop in horizon; approximate with extremes
        # If both hit, decide by which extreme is closer to i in time: heuristic fallback
        up_hit = hi >= up_th
        dn_hit = lo <= dn_th
        if up_hit and not dn_hit:
            y[i - start] = 1
        elif not up_hit and dn_hit:
            y[i - start] = 0
        elif up_hit and dn_hit:
            # ambiguous: choose by larger excursion as proxy
            up_exc = (hi/base - 1.0)
            dn_exc = (1.0 - lo/base)
            y[i - start] = 1 if up_exc >= dn_exc else 0
        else:
            y[i - start] = 0
    return y


def prepare_training_data(symbol: str, interval: str, limit: int, horizon_bars: int, target_up_pct: float, stop_down_pct: float) -> Tuple[np.ndarray, np.ndarray, int]:
    candles = TechnicalAnalysis.get_candle_data(symbol.upper(), limit=limit, interval=interval)
    if not candles or len(candles) < (horizon_bars + 40):
        return np.zeros((0, 1)), np.zeros((0,), dtype=int), 0
    closes = np.array([c['close'] for c in candles], dtype=float)
    feats = _mk_features(candles)
    start = 25  # skip warmup for indicators
    end = len(closes) - horizon_bars - 1
    if end <= start:
        return np.zeros((0, 1)), np.zeros((0,), dtype=int), 0
    X = _stack_features(feats, start, end)
    y = _make_labels(closes, start, end, horizon_bars, target_up_pct, stop_down_pct)
    events = int(X.shape[0])
    return X, y, events


class ScalpLogReg:
    def __init__(self):
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.mu: np.ndarray | None = None
        self.sigma: np.ndarray | None = None

    def _scale(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        if fit:
            self.mu = X.mean(axis=0)
            self.sigma = X.std(axis=0)
            self.sigma = np.where(self.sigma == 0, 1.0, self.sigma)
        if self.mu is None or self.sigma is None:
            return X
        return (X - self.mu) / self.sigma

    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.1, epochs: int = 200) -> Dict[str, Any]:
        if X.size == 0:
            return {'trained': False, 'error': 'empty_data'}
        Xs = self._scale(X, fit=True)
        n, d = Xs.shape
        rng = np.random.default_rng(42)
        self.w = rng.normal(0, 0.01, size=(d,))
        self.b = 0.0
        for _ in range(epochs):
            z = Xs @ self.w + self.b
            p = 1.0 / (1.0 + np.exp(-z))
            grad_w = (Xs.T @ (p - y)) / n
            grad_b = float(np.mean(p - y))
            self.w -= lr * grad_w
            self.b -= lr * grad_b
        # simple accuracy
        preds = (1.0 / (1.0 + np.exp(-(Xs @ self.w + self.b))) >= 0.5).astype(int)
        acc = float(np.mean(preds == y)) if n > 0 else 0.0
        return {'trained': True, 'n': int(n), 'd': int(d), 'acc': acc}

    def predict_proba(self, x: np.ndarray) -> float:
        if self.w is None:
            return 50.0
        xs = self._scale(x.reshape(1, -1), fit=False)
        z = float(xs @ self.w + self.b)
        p = 1.0 / (1.0 + math.exp(-z))
        return round(p * 100.0, 2)

    def latest_features(self, symbol: str, interval: str = '1m') -> Tuple[np.ndarray, Dict[str, Any]]:
        candles = TechnicalAnalysis.get_candle_data(symbol.upper(), limit=200, interval=interval)
        if not candles or len(candles) < 40:
            return np.zeros((0,)), {'error': 'insufficient_data'}
        feats = _mk_features(candles)
        X = _stack_features(feats, len(candles)-1, len(candles))
        ctx = {
            'rsi': float(feats['rsi'][-1]) if np.isfinite(feats['rsi'][-1]) else None,
            'atr_pct': float(feats['atr_pct'][-1]) if np.isfinite(feats['atr_pct'][-1]) else None,
        }
        return X[0], ctx

    def save(self, path: str = MODEL_PATH) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            'w': self.w.tolist() if self.w is not None else None,
            'b': self.b,
            'mu': self.mu.tolist() if self.mu is not None else None,
            'sigma': self.sigma.tolist() if self.sigma is not None else None,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def load(self, path: str = MODEL_PATH) -> bool:
        if not os.path.exists(path):
            return False
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        self.w = np.array(obj.get('w')) if obj.get('w') is not None else None
        self.b = float(obj.get('b') or 0.0)
        self.mu = np.array(obj.get('mu')) if obj.get('mu') is not None else None
        self.sigma = np.array(obj.get('sigma')) if obj.get('sigma') is not None else None
        return self.w is not None
