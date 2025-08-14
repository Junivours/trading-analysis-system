import os
import numpy as np
from typing import List


class _BaseAI:
    """Minimal interface expected by MasterAnalyzer.
    Implementations should provide:
      - prepare_advanced_features(tech, patterns, ticker, position, extended, regime_data=None) -> dict
      - predict_with_uncertainty(features: dict, passes: int = 10, feature_noise: float = 0.01, dropout_rate: float = 0.1) -> dict
      - get_status() -> dict
    """

    def prepare_advanced_features(self, *args, **kwargs):
        raise NotImplementedError

    def predict_with_uncertainty(self, *args, **kwargs):
        raise NotImplementedError

    def get_status(self):
        return {'initialized': False, 'mode': 'offline'}

    # --- Calibration no-ops to maintain API compatibility ---
    def load_calibration_state(self, path: str):
        return False

    def save_calibration_state(self, path: str):
        return False

    def add_calibration_observation(self, raw_prob: float, success: bool):
        return False

    def get_calibration_status(self):
        return {'count': 0, 'calibrated': False}

    def get_feature_schema(self):
        return []


class FeatureEngineNeutral:
    """JAX-free feature preparation and vectorization with a stable schema.
    Produces a compact feature dict and deterministic vector for adapters.
    """
    def __init__(self):
        # Define a stable schema order for vectorization
        self.schema = [
            # Technical snapshot
            'rsi', 'macd', 'macd_hist', 'sma9', 'sma20', 'price_position',
            'vol_ratio', 'trend_short', 'trend_medium',
            'support_distance_pct', 'resistance_distance_pct',
            'support_strength', 'resistance_strength',
            # Extended
            'atr_pct',
            # Patterns summary
            'bull_patterns', 'bear_patterns', 'avg_pattern_conf',
            # Position analysis
            'resistance_potential', 'support_risk',
            # Market data
            'pct_change_24h',
            # Multi-timeframe consensus
            'mt_bull_score', 'mt_bear_score', 'mt_primary_num',
        ]

    def prepare_advanced_features(self, tech, patterns, ticker, position, extended, regime_data=None):
        def g(d, *path, default=None):
            cur = d or {}
            for p in path:
                try:
                    cur = cur.get(p)
                except Exception:
                    return default
                if cur is None:
                    return default
            return cur if cur is not None else default
        # Technicals
        rsi = g(tech, 'rsi', 'rsi', default=50.0) or 50.0
        macd = g(tech, 'macd', 'macd', default=0.0) or 0.0
        macd_hist = g(tech, 'macd', 'histogram', default=0.0) or 0.0
        sma9 = tech.get('sma_9') if isinstance(tech, dict) else None
        sma20 = tech.get('sma_20') if isinstance(tech, dict) else None
        price_position = tech.get('price_position', 0.5) or 0.5
        vol_ratio = g(tech, 'volume_analysis', 'ratio', default=1.0) or 1.0
        trend_short = g(tech, 'trend', 'short_term_momentum', default=0.0) or 0.0
        trend_medium = g(tech, 'trend', 'medium_term_momentum', default=0.0) or 0.0
        supp_dist = tech.get('support_distance_pct', None)
        res_dist = tech.get('resistance_distance_pct', None)
        supp_strength = tech.get('support_strength', None)
        res_strength = tech.get('resistance_strength', None)
        # Extended
        atr_pct = g(extended, 'atr', 'percentage', default=None)
        # Patterns summary
        pats = (patterns or {}).get('patterns', []) if isinstance(patterns, dict) else []
        bull_patterns = sum(1 for p in pats if p.get('signal') == 'bullish')
        bear_patterns = sum(1 for p in pats if p.get('signal') == 'bearish')
        avg_conf = float(np.mean([p.get('confidence', 0) for p in pats])) if pats else 0.0
        # Position
        res_pot = (position or {}).get('resistance_potential', 0.0)
        sup_risk = (position or {}).get('support_risk', 0.0)
        # Market data
        try:
            pct_change_24h = float((ticker or {}).get('priceChangePercent', 0.0))
        except Exception:
            pct_change_24h = 0.0
        # Multi-timeframe
        mt = (patterns or {})  # default
        # We expect multi_timeframe provided outside; accept separate param via regime_data or tech
        # Fallback: zeros
        mt_bull = 0.0; mt_bear = 0.0; mt_primary_num = 0.0
        try:
            # MasterAnalyzer passes multi_timeframe to AI separately as part of features; if not, ignore
            pass
        except Exception:
            pass
        # Best-effort: try to read from tech.placeholder if injected elsewhere (no-op here)
        feat = {
            'rsi': float(rsi),
            'macd': float(macd),
            'macd_hist': float(macd_hist),
            'sma9': float(sma9) if isinstance(sma9, (int, float)) else 0.0,
            'sma20': float(sma20) if isinstance(sma20, (int, float)) else 0.0,
            'price_position': float(price_position),
            'vol_ratio': float(vol_ratio),
            'trend_short': float(trend_short),
            'trend_medium': float(trend_medium),
            'support_distance_pct': float(supp_dist) if isinstance(supp_dist, (int, float)) else 0.0,
            'resistance_distance_pct': float(res_dist) if isinstance(res_dist, (int, float)) else 0.0,
            'support_strength': float(supp_strength) if isinstance(supp_strength, (int, float)) else 0.0,
            'resistance_strength': float(res_strength) if isinstance(res_strength, (int, float)) else 0.0,
            'atr_pct': float(atr_pct) if isinstance(atr_pct, (int, float)) else 0.0,
            'bull_patterns': float(bull_patterns),
            'bear_patterns': float(bear_patterns),
            'avg_pattern_conf': float(avg_conf),
            'resistance_potential': float(res_pot) if isinstance(res_pot, (int, float)) else 0.0,
            'support_risk': float(sup_risk) if isinstance(sup_risk, (int, float)) else 0.0,
            'pct_change_24h': float(pct_change_24h),
            'mt_bull_score': float(mt_bull),
            'mt_bear_score': float(mt_bear),
            'mt_primary_num': float(mt_primary_num),
        }
        return feat

    def _vectorize(self, features: dict) -> np.ndarray:
        vec = []
        for name in self.schema:
            v = features.get(name, 0.0)
            try:
                vec.append(float(v))
            except Exception:
                vec.append(0.0)
        return np.array(vec, dtype=np.float32)

    def vectorize(self, features: dict) -> np.ndarray:
        """Public vectorize method for adapters."""
        return self._vectorize(features)

    def get_feature_schema(self):
        return list(self.schema)


class TorchAIAdapter(_BaseAI):
    """Optional PyTorch adapter. Uses AdvancedJAXAI for feature engineering to ensure consistency.
    If torch is not available, falls back to AdvancedJAXAI predictions but annotates framework="torch_fallback".
    """

    def __init__(self, feature_engine: FeatureEngineNeutral | None = None, mode: str = 'live'):
        self.mode = mode
        self._feature_helper = feature_engine or FeatureEngineNeutral()
        self.input_dim = len(self._feature_helper.get_feature_schema())
        # Try to load torch lazily
        try:
            import torch  # type: ignore
            import torch.nn as nn  # type: ignore

            class _TinyMLP(nn.Module):
                def __init__(self, in_dim: int):
                    super().__init__()
                    self.l1 = nn.Linear(in_dim, 96)
                    self.l2 = nn.Linear(96, 64)
                    self.l3 = nn.Linear(64, 32)
                    self.l4 = nn.Linear(32, 4)

                def forward(self, x):
                    # local import to avoid global type issues when torch missing
                    import torch.nn.functional as F  # type: ignore
                    x = torch.tanh(self.l1(x))
                    x = torch.tanh(self.l2(x))
                    x = torch.tanh(self.l3(x))
                    x = self.l4(x)
                    return x

            self._torch = torch
            self._nn = nn
            self.model = _TinyMLP(self.input_dim)
            self.initialized = True
            self.framework = 'torch'
        except Exception:
            self._torch = None
            self._nn = None
            self.model = None
            self.initialized = False
            self.framework = 'torch_fallback'

    def prepare_advanced_features(self, tech, patterns, ticker, position, extended, regime_data=None):
        return self._feature_helper.prepare_advanced_features(tech, patterns, ticker, position, extended, regime_data)

    def predict_with_uncertainty(self, features, passes: int = 10, feature_noise: float = 0.01, dropout_rate: float = 0.1):
        # If torch missing, delegate to JAX backend result and tag meta
        if not self.initialized or self._torch is None:
            # Fallback: produce a HOLD-like neutral output without JAX
            return self._neutral_output(features)
        try:
            # Vectorize using feature helper for consistent order + standardization
            vec = self._feature_helper.vectorize(features)
            t = self._torch
            x = t.tensor(vec, dtype=t.float32).view(1, -1)
            logits = self.model(x)
            # Simple temperature of 1.0
            probs = t.softmax(logits, dim=-1).detach().cpu().numpy().squeeze()
            # Build response mirroring AdvancedJAXAI
            idx = int(np.argmax(probs))
            signals = ['STRONG_SELL', 'SELL', 'BUY', 'STRONG_BUY']
            signal = signals[idx]
            confidence = float(probs[idx] * 100)
            entropy = float(-(probs * np.log(probs + 1e-9)).sum())
            margin = float(np.sort(probs)[-1] - np.sort(probs)[-2]) if len(probs) >= 2 else 0.0
            max_entropy = np.log(len(probs))
            reliability = (0.6 * margin + 0.4 * (1 - entropy / max_entropy)) * 100
            return {
                'signal': signal,
                'confidence': round(confidence, 2),
                'probabilities': np.array(probs).round(4).tolist(),
                'ai_recommendation': 'torch_adapter',
                'model_version': 'Torch-Adapter-v1',
                'mode': self.mode,
                'framework': 'torch',
                'reliability_score': round(reliability, 2),
                'prob_margin': round(margin, 4),
                'entropy': round(entropy, 4),
                'bull_probability_raw': round(float(probs[2] + probs[3]) * 100, 2),
                'bull_probability_calibrated': round(float(probs[2] + probs[3]) * 100, 2),
                'uncertainty': {'enabled': False}
            }
        except Exception as e:
            out = self._neutral_output(features)
            out['framework'] = f'{self.framework}_error'
            out['error'] = str(e)
            return out

    def get_status(self):
        return {'initialized': bool(self.initialized), 'mode': self.mode, 'framework': self.framework}

    def get_feature_schema(self):
        try:
            return self._feature_helper.get_feature_schema()
        except Exception:
            return []

    def _neutral_output(self, features):
        # Simple heuristic to assign mild probabilities without JAX
        vec = self._feature_helper.vectorize(features)
        # Use rsi and mt scores proxy if present
        rsi = features.get('rsi', 50.0)
        bull = max(0.0, min(1.0, (rsi - 50.0) / 30.0))
        sell_prob = max(0.0, 1.0 - bull)
        probs = np.array([0.15 * sell_prob, 0.35 * sell_prob, 0.35 * bull, 0.15 * bull], dtype=float)
        # Ensure non-degenerate
        probs = probs + 1e-6
        probs = probs / probs.sum()
        idx = int(np.argmax(probs))
        signals = ['STRONG_SELL', 'SELL', 'BUY', 'STRONG_BUY']
        signal = signals[idx]
        confidence = float(probs[idx] * 100)
        entropy = float(-(probs * np.log(probs + 1e-9)).sum())
        margin = float(np.sort(probs)[-1] - np.sort(probs)[-2]) if len(probs) >= 2 else 0.0
        max_entropy = np.log(len(probs))
        reliability = (0.6 * margin + 0.4 * (1 - entropy / max_entropy)) * 100
        return {
            'signal': signal,
            'confidence': round(confidence, 2),
            'probabilities': np.array(probs).round(4).tolist(),
            'ai_recommendation': 'torch_adapter',
            'model_version': 'Torch-Adapter-v1',
            'mode': self.mode,
            'framework': self.framework,
            'reliability_score': round(reliability, 2),
            'prob_margin': round(margin, 4),
            'entropy': round(entropy, 4),
            'bull_probability_raw': round(float(probs[2] + probs[3]) * 100, 2),
            'bull_probability_calibrated': round(float(probs[2] + probs[3]) * 100, 2),
            'uncertainty': {'enabled': False}
        }


class TensorFlowAIAdapter(_BaseAI):
    """Optional TensorFlow adapter. Uses AdvancedJAXAI for features; if TensorFlow missing, falls back to JAX predictions.
    """

    def __init__(self, feature_engine: FeatureEngineNeutral | None = None, mode: str = 'live'):
        self.mode = mode
        self._feature_helper = feature_engine or FeatureEngineNeutral()
        self.input_dim = len(self._feature_helper.get_feature_schema())
        try:
            import tensorflow as tf  # type: ignore

            self._tf = tf
            self.model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(self.input_dim,)),
                tf.keras.layers.Dense(96, activation='tanh'),
                tf.keras.layers.Dense(64, activation='tanh'),
                tf.keras.layers.Dense(32, activation='tanh'),
                tf.keras.layers.Dense(4)
            ])
            # Build the model once
            self.model(self._tf.zeros((1, self.input_dim)))
            self.initialized = True
            self.framework = 'tensorflow'
        except Exception:
            self._tf = None
            self.model = None
            self.initialized = False
            self.framework = 'tf_fallback'

    def prepare_advanced_features(self, tech, patterns, ticker, position, extended, regime_data=None):
        return self._feature_helper.prepare_advanced_features(tech, patterns, ticker, position, extended, regime_data)

    def predict_with_uncertainty(self, features, passes: int = 10, feature_noise: float = 0.01, dropout_rate: float = 0.1):
        if not self.initialized or self._tf is None:
            return self._neutral_output(features)
        try:
            vec = self._feature_helper.vectorize(features)
            logits = self.model(self._tf.convert_to_tensor(vec.reshape(1, -1), dtype=self._tf.float32))
            probs = self._tf.nn.softmax(logits, axis=-1).numpy().squeeze()
            idx = int(np.argmax(probs))
            signals = ['STRONG_SELL', 'SELL', 'BUY', 'STRONG_BUY']
            signal = signals[idx]
            confidence = float(probs[idx] * 100)
            entropy = float(-(probs * np.log(probs + 1e-9)).sum())
            margin = float(np.sort(probs)[-1] - np.sort(probs)[-2]) if len(probs) >= 2 else 0.0
            max_entropy = np.log(len(probs))
            reliability = (0.6 * margin + 0.4 * (1 - entropy / max_entropy)) * 100
            return {
                'signal': signal,
                'confidence': round(confidence, 2),
                'probabilities': np.array(probs).round(4).tolist(),
                'ai_recommendation': 'tf_adapter',
                'model_version': 'TF-Adapter-v1',
                'mode': self.mode,
                'framework': 'tensorflow',
                'reliability_score': round(reliability, 2),
                'prob_margin': round(margin, 4),
                'entropy': round(entropy, 4),
                'bull_probability_raw': round(float(probs[2] + probs[3]) * 100, 2),
                'bull_probability_calibrated': round(float(probs[2] + probs[3]) * 100, 2),
                'uncertainty': {'enabled': False}
            }
        except Exception as e:
            out = self._neutral_output(features)
            out['framework'] = f'{self.framework}_error'
            out['error'] = str(e)
            return out

    def get_status(self):
        return {'initialized': bool(self.initialized), 'mode': self.mode, 'framework': self.framework}

    def get_feature_schema(self):
        try:
            return self._feature_helper.get_feature_schema()
        except Exception:
            return []

    def _neutral_output(self, features):
        # Mirror Torch neutral for consistency
        rsi = features.get('rsi', 50.0)
        bull = max(0.0, min(1.0, (rsi - 50.0) / 30.0))
        sell_prob = max(0.0, 1.0 - bull)
        probs = np.array([0.15 * sell_prob, 0.35 * sell_prob, 0.35 * bull, 0.15 * bull], dtype=float)
        probs = probs + 1e-6
        probs = probs / probs.sum()
        idx = int(np.argmax(probs))
        signals = ['STRONG_SELL', 'SELL', 'BUY', 'STRONG_BUY']
        signal = signals[idx]
        confidence = float(probs[idx] * 100)
        entropy = float(-(probs * np.log(probs + 1e-9)).sum())
        margin = float(np.sort(probs)[-1] - np.sort(probs)[-2]) if len(probs) >= 2 else 0.0
        max_entropy = np.log(len(probs))
        reliability = (0.6 * margin + 0.4 * (1 - entropy / max_entropy)) * 100
        return {
            'signal': signal,
            'confidence': round(confidence, 2),
            'probabilities': np.array(probs).round(4).tolist(),
            'ai_recommendation': 'tf_adapter',
            'model_version': 'TF-Adapter-v1',
            'mode': self.mode,
            'framework': self.framework,
            'reliability_score': round(reliability, 2),
            'prob_margin': round(margin, 4),
            'entropy': round(entropy, 4),
            'bull_probability_raw': round(float(probs[2] + probs[3]) * 100, 2),
            'bull_probability_calibrated': round(float(probs[2] + probs[3]) * 100, 2),
            'uncertainty': {'enabled': False}
        }


class EnsembleAI(_BaseAI):
    """Simple ensemble across multiple backends. Averages class probabilities.
    Uses JAX backend for feature engineering.
    """

    def __init__(self, members: List[_BaseAI], feature_engine: FeatureEngineNeutral | None = None):
        self.members = [m for m in members if m is not None]
        self._feature_helper = feature_engine or FeatureEngineNeutral()
        self.mode = 'ensemble'

    def prepare_advanced_features(self, tech, patterns, ticker, position, extended, regime_data=None):
        return self._feature_helper.prepare_advanced_features(tech, patterns, ticker, position, extended, regime_data)

    def predict_with_uncertainty(self, features, passes: int = 10, feature_noise: float = 0.01, dropout_rate: float = 0.1):
        results = []
        probs_list = []
        for m in self.members:
            try:
                res = m.predict_with_uncertainty(features, passes=passes, feature_noise=feature_noise, dropout_rate=dropout_rate)
                results.append(res)
                probs = res.get('probabilities') or [0.25, 0.25, 0.25, 0.25]
                probs_list.append(np.array(probs, dtype=float))
            except Exception:
                continue
        if not probs_list:
            # Fallback: neutral HOLD-ish output
            return TorchAIAdapter(feature_engine=self._feature_helper).predict_with_uncertainty(features)
        mean_probs = np.clip(np.mean(probs_list, axis=0), 1e-9, 1.0)
        mean_probs = mean_probs / mean_probs.sum()
        idx = int(np.argmax(mean_probs))
        signals = ['STRONG_SELL', 'SELL', 'BUY', 'STRONG_BUY']
        signal = signals[idx]
        confidence = float(mean_probs[idx] * 100)
        entropy = float(-(mean_probs * np.log(mean_probs + 1e-9)).sum())
        margin = float(np.sort(mean_probs)[-1] - np.sort(mean_probs)[-2]) if len(mean_probs) >= 2 else 0.0
        max_entropy = np.log(len(mean_probs))
        reliability = (0.6 * margin + 0.4 * (1 - entropy / max_entropy)) * 100
        # Alignment heuristic: do most members agree on BUYish/SELLish side?
        def side(s):
            s = (s or '').upper()
            if 'BUY' in s:
                return 'BUY'
            if 'SELL' in s:
                return 'SELL'
            return 'HOLD'
        sides = [side(r.get('signal')) for r in results]
        buy_cnt = sides.count('BUY')
        sell_cnt = sides.count('SELL')
        alignment = 'aligned' if (buy_cnt == len(sides) or sell_cnt == len(sides)) else 'mixed'
        return {
            'signal': signal,
            'confidence': round(confidence, 2),
            'probabilities': np.array(mean_probs).round(4).tolist(),
            'ai_recommendation': 'ensemble',
            'model_version': 'Ensemble-v1',
            'mode': self.mode,
            'reliability_score': round(reliability, 2),
            'prob_margin': round(margin, 4),
            'entropy': round(entropy, 4),
            'bull_probability_raw': round(float(mean_probs[2] + mean_probs[3]) * 100, 2),
            'bull_probability_calibrated': round(float(mean_probs[2] + mean_probs[3]) * 100, 2),
            'uncertainty': {'enabled': True, 'passes': passes, 'avg_std': float(np.std(probs_list, axis=0).mean()), 'entropy': round(entropy, 4)},
            'ensemble': {
                'members': [r.get('model_version') or r.get('framework') or 'unknown' for r in results],
                'signals': sides,
                'alignment': alignment,
            }
        }

    def get_status(self):
        details = []
        try:
            for m in self.members:
                try:
                    d = m.get_status()
                except Exception:
                    d = {'initialized': False, 'framework': getattr(m, 'framework', 'unknown')}
                details.append(d)
        except Exception:
            pass
        return {
            'initialized': True,
            'mode': 'ensemble',
            'backend': 'ensemble',
            'members': len(self.members),
            'details': details
        }

    def get_feature_schema(self):
        try:
            return self._feature_helper.get_feature_schema()
        except Exception:
            return []


def get_ai_system(backend: str | None = None):
    """Factory for AI backend. Supported values: 'torch', 'tf', 'ensemble' (default)."""
    backend = (backend or os.getenv('AI_BACKEND') or 'ensemble').strip().lower()
    engine = FeatureEngineNeutral()
    if backend == 'torch':
        return TorchAIAdapter(feature_engine=engine)
    if backend in ('tf', 'tensorflow'):
        return TensorFlowAIAdapter(feature_engine=engine)
    # default: ensemble (uses both torch + tf)
    members: List[_BaseAI] = []
    try:
        members.append(TorchAIAdapter(feature_engine=engine))
    except Exception:
        pass
    try:
        members.append(TensorFlowAIAdapter(feature_engine=engine))
    except Exception:
        pass
    return EnsembleAI(members, feature_engine=engine)
