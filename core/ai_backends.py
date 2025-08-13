import os
import numpy as np
from typing import List, TYPE_CHECKING

# Reuse the existing JAX-based implementation for features and default predictions
from core.ai import AdvancedJAXAI


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


class TorchAIAdapter(_BaseAI):
    """Optional PyTorch adapter. Uses AdvancedJAXAI for feature engineering to ensure consistency.
    If torch is not available, falls back to AdvancedJAXAI predictions but annotates framework="torch_fallback".
    """

    def __init__(self, input_dim: int = 160, mode: str = 'live'):
        self.mode = mode
        self.input_dim = input_dim
        self._feature_helper = AdvancedJAXAI(input_dim=input_dim, mode=mode)
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
            res = self._feature_helper.predict_with_uncertainty(features, passes=passes, feature_noise=feature_noise, dropout_rate=dropout_rate)
            res['framework'] = self.framework
            return res
        try:
            # Vectorize using JAX helper for consistent order + standardization
            vec = self._feature_helper._vectorize(features)
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
            # Fallback on any error
            res = self._feature_helper.predict_with_uncertainty(features, passes=passes, feature_noise=feature_noise, dropout_rate=dropout_rate)
            res['framework'] = f'{self.framework}_error'
            res['error'] = str(e)
            return res

    def get_status(self):
        return {'initialized': bool(self.initialized), 'mode': self.mode, 'framework': self.framework}


class TensorFlowAIAdapter(_BaseAI):
    """Optional TensorFlow adapter. Uses AdvancedJAXAI for features; if TensorFlow missing, falls back to JAX predictions.
    """

    def __init__(self, input_dim: int = 160, mode: str = 'live'):
        self.mode = mode
        self.input_dim = input_dim
        self._feature_helper = AdvancedJAXAI(input_dim=input_dim, mode=mode)
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
            res = self._feature_helper.predict_with_uncertainty(features, passes=passes, feature_noise=feature_noise, dropout_rate=dropout_rate)
            res['framework'] = self.framework
            return res
        try:
            vec = self._feature_helper._vectorize(features)
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
            res = self._feature_helper.predict_with_uncertainty(features, passes=passes, feature_noise=feature_noise, dropout_rate=dropout_rate)
            res['framework'] = f'{self.framework}_error'
            res['error'] = str(e)
            return res

    def get_status(self):
        return {'initialized': bool(self.initialized), 'mode': self.mode, 'framework': self.framework}


class EnsembleAI(_BaseAI):
    """Simple ensemble across multiple backends. Averages class probabilities.
    Uses JAX backend for feature engineering.
    """

    def __init__(self, members: List[_BaseAI]):
        self.members = [m for m in members if m is not None]
        self._feature_helper = None
        for m in self.members:
            # Prefer an AdvancedJAXAI instance for feature prep if present
            if isinstance(m, AdvancedJAXAI):
                self._feature_helper = m
                break
        if self._feature_helper is None:
            self._feature_helper = AdvancedJAXAI()
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
            # Fallback: use helper directly
            return self._feature_helper.predict_with_uncertainty(features, passes=passes, feature_noise=feature_noise, dropout_rate=dropout_rate)
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
        return {'initialized': True, 'mode': 'ensemble', 'members': len(self.members)}


def get_ai_system(backend: str | None = None):
    """Factory for AI backend. Supported values: 'jax' (default), 'torch', 'tf', 'ensemble'.
    Falls back safely to JAX if others unavailable.
    """
    backend = (backend or os.getenv('AI_BACKEND') or 'jax').strip().lower()
    if backend == 'torch':
        try:
            return TorchAIAdapter()
        except Exception:
            return AdvancedJAXAI()
    if backend in ('tf', 'tensorflow'):
        try:
            return TensorFlowAIAdapter()
        except Exception:
            return AdvancedJAXAI()
    if backend in ('ensemble', 'ens'):
        members: List[_BaseAI] = []
        # Always include JAX as a baseline
        members.append(AdvancedJAXAI())
        # Try to include optional backends if available
        try:
            m_t = TorchAIAdapter()
            members.append(m_t)
        except Exception:
            pass
        try:
            m_f = TensorFlowAIAdapter()
            members.append(m_f)
        except Exception:
            pass
        return EnsembleAI(members)
    # default
    return AdvancedJAXAI()
