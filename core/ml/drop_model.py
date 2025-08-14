import json
import os
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import numpy as np

from core.technical_analysis import TechnicalAnalysis

MODEL_DIR = os.getenv('MODEL_DIR', 'data/models')
MODEL_PATH = os.path.join(MODEL_DIR, 'drop_logreg.json')


def _ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


@dataclass
class TrainConfig:
    symbol: str
    interval: str = '1h'
    limit: int = 3000
    drop_pct: float = 10.0
    window_bars: int = 24
    horizon_bars: int = 6


class DropLogReg:
    """
    Sehr leichte, portierbare "KI": Logistische Regression per Newton-Schritt (2-3 Iterationen),
    nur mit Numpy implementiert (keine externen ML-Frameworks nötig),
    um aus Features (z. B. RSI, MACD, ATR%, Distanz zu S/R) die Bounce-Wahrscheinlichkeit zu schätzen.
    """

    def __init__(self):
        self.weights: np.ndarray | None = None
        self.feature_names: List[str] = []
        self.meta: Dict[str, Any] = {}

    # ---- Feature Engineering ----
    def _build_features(self, candles: List[Dict[str, Any]], cfg: TrainConfig) -> Tuple[np.ndarray, np.ndarray]:
        closes = np.array([c['close'] for c in candles], dtype=float)
        highs = np.array([c['high'] for c in candles], dtype=float)
        lows = np.array([c['low'] for c in candles], dtype=float)
        n = len(closes)
        # Compute tech pack
        rsi_obj = TechnicalAnalysis._calculate_advanced_rsi(closes)
        macd_obj = TechnicalAnalysis._calculate_advanced_macd(closes)
        atr_pct = TechnicalAnalysis._atr_percent(highs, lows, closes, period=14) or 0.0
        sup, res, meta = TechnicalAnalysis._calculate_support_resistance_zones(highs, lows, closes)
        price = float(closes[-1])
        dist_sup = (price - float(sup)) / price * 100.0 if price > 0 else 0.0
        dist_res = (float(res) - price) / price * 100.0 if price > 0 else 0.0
        rsi = float(rsi_obj.get('rsi')) if isinstance(rsi_obj, dict) else 50.0
        macd_curve = macd_obj.get('curve_strength', 0.0) if isinstance(macd_obj, dict) else 0.0
        trend = TechnicalAnalysis._analyze_trend(closes, TechnicalAnalysis._sma(closes,9), TechnicalAnalysis._sma(closes,20))
        trend_flag = 1.0 if (trend.get('trend') in ('bullish','strong_bullish')) else -1.0 if (trend.get('trend') in ('bearish','strong_bearish')) else 0.0
        # Feature vector at each t (use rolling alignment where possible)
        X_list = []
        y_list = []
        window = int(cfg.window_bars)
        horizon = int(cfg.horizon_bars)
        drop_frac = cfg.drop_pct / 100.0
        for t in range(window, n - horizon):
            # event
            base = closes[t]
            found = False
            for k in range(1, window + 1):
                prev = closes[t - k]
                if prev > 0 and (base / prev - 1.0) <= -drop_frac:
                    found = True
                    break
            if not found:
                continue
            ft = t + horizon
            future_ret = (closes[ft] / base) - 1.0
            y = 1.0 if future_ret > 0 else 0.0
            # Localized features (use values up to t)
            # Simple, stable pack to avoid overfitting
            vec = [
                1.0,  # bias
                min(max(rsi, 0.0), 100.0) / 100.0,
                float(macd_curve),
                float(atr_pct) / 100.0,
                float(dist_sup) / 10.0,
                float(dist_res) / 10.0,
                float(trend_flag),
            ]
            X_list.append(vec)
            y_list.append(y)
        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=float)
        self.feature_names = ['bias','rsi_norm','macd_curve','atr_pct_norm','dist_sup_norm','dist_res_norm','trend_flag']
        return X, y

    # ---- Training (Newton-Raphson steps) ----
    def fit(self, cfg: TrainConfig) -> Dict[str, Any]:
        candles = TechnicalAnalysis.get_candle_data(cfg.symbol, limit=cfg.limit, interval=cfg.interval)
        if not candles or len(candles) < (cfg.window_bars + cfg.horizon_bars + 30):
            raise RuntimeError('insufficient_data')
        X, y = self._build_features(candles, cfg)
        if X.size == 0:
            raise RuntimeError('no_events')
        w = np.zeros(X.shape[1], dtype=float)
        for _ in range(3):  # few steps
            z = X @ w
            p = 1.0 / (1.0 + np.exp(-z))
            # gradient and hessian
            g = X.T @ (p - y)
            W = p * (1 - p)
            H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
            try:
                step = np.linalg.solve(H, g)
            except Exception:
                step = np.linalg.pinv(H) @ g
            w = w - step
        self.weights = w
        self.meta = {'events': int(y.size), 'features': self.feature_names, 'symbol': cfg.symbol, 'interval': cfg.interval}
        return {'events': int(y.size), 'weights': w.tolist(), 'features': self.feature_names}

    def predict_proba(self, x: np.ndarray) -> float:
        if self.weights is None:
            raise RuntimeError('model_not_trained')
        z = float(np.dot(x, self.weights))
        return 1.0 / (1.0 + np.exp(-z))

    def latest_features(self, symbol: str, interval: str) -> np.ndarray:
        candles = TechnicalAnalysis.get_candle_data(symbol, limit=500, interval=interval)
        if not candles:
            raise RuntimeError('no_data')
        # reuse feature builder head for current snapshot
        closes = np.array([c['close'] for c in candles], dtype=float)
        highs = np.array([c['high'] for c in candles], dtype=float)
        lows = np.array([c['low'] for c in candles], dtype=float)
        rsi_obj = TechnicalAnalysis._calculate_advanced_rsi(closes)
        macd_obj = TechnicalAnalysis._calculate_advanced_macd(closes)
        atr_pct = TechnicalAnalysis._atr_percent(highs, lows, closes, period=14) or 0.0
        sup, res, meta = TechnicalAnalysis._calculate_support_resistance_zones(highs, lows, closes)
        price = float(closes[-1])
        dist_sup = (price - float(sup)) / price * 100.0 if price > 0 else 0.0
        dist_res = (float(res) - price) / price * 100.0 if price > 0 else 0.0
        rsi = float(rsi_obj.get('rsi')) if isinstance(rsi_obj, dict) else 50.0
        macd_curve = macd_obj.get('curve_strength', 0.0) if isinstance(macd_obj, dict) else 0.0
        trend = TechnicalAnalysis._analyze_trend(closes, TechnicalAnalysis._sma(closes,9), TechnicalAnalysis._sma(closes,20))
        trend_flag = 1.0 if (trend.get('trend') in ('bullish','strong_bullish')) else -1.0 if (trend.get('trend') in ('bearish','strong_bearish')) else 0.0
        x = np.array([
            1.0,
            min(max(rsi, 0.0), 100.0) / 100.0,
            float(macd_curve),
            float(atr_pct) / 100.0,
            float(dist_sup) / 10.0,
            float(dist_res) / 10.0,
            float(trend_flag),
        ], dtype=float)
        return x

    def save(self, path: str = MODEL_PATH):
        _ensure_dir(path)
        if self.weights is None:
            raise RuntimeError('no_model')
        data = {'weights': self.weights.tolist(), 'features': self.feature_names, 'meta': self.meta}
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f)

    def load(self, path: str = MODEL_PATH):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.weights = np.array(data['weights'], dtype=float)
        self.feature_names = data.get('features', [])
        self.meta = data.get('meta', {})
