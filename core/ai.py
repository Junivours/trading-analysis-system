import os, time, numpy as np, math
from datetime import datetime
try:
    import jax.numpy as jnp
    from jax.nn import logsumexp
    from jax import random
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False

class AdvancedJAXAI:
    """Lightweight in-process JAX model with improved feature engineering & reliability metrics.

    Upgrades (v2.1):
      - Extended engineered feature set (trend, volatility, pattern distribution, regime proxies)
      - Deterministic feature ordering + scaling (running mean/var) for more stable logits
      - Reliability score (probability margin + entropy inverse)
      - Simple adaptive temperature: lowers temperature when high consensus to sharpen probs
      - Feature list introspection via get_feature_schema()
    """
    def __init__(self, input_dim: int = 160, mode: str = 'live', temperature: float = 1.0):
        self.input_dim = input_dim
        self.mode = mode
        self.base_temperature = temperature
        self.temperature = temperature
        self.initialized = False
        self.training_data = []
        self.key = None
        self.model_params = {}
        # Running stats for naive online standardization
        self._feat_count = 0
        self._feat_mean = None
        self._feat_m2 = None  # sum of squares of differences
        self._feature_names_cache = None
        try:
            if JAX_AVAILABLE:
                self._init_model()
        except Exception as e:
            print(f"AI init failed: {e}")
            self.initialized = False
    def _init_model(self):
        self.key = random.PRNGKey(int(time.time()) & 0xFFFFFFFF)
        k1,k2,k3,k4 = random.split(self.key,4)
        self.model_params={
            'w1': random.normal(k1,(self.input_dim,96))*0.05,'b1': jnp.zeros(96),
            'w2': random.normal(k2,(96,64))*0.05,'b2': jnp.zeros(64),
            'w3': random.normal(k3,(64,32))*0.05,'b3': jnp.zeros(32),
            'w4': random.normal(k4,(32,4))*0.05,'b4': jnp.zeros(4)}
        self.initialized=True
        # Calibration (Platt scaling) parameters A,B for logistic: 1/(1+exp(A*x+B)) on bullish prob
        self._calib_samples = []  # list of (raw_bull_prob, label)
        self._calib_A = 0.0
        self._calib_B = 0.0
        self._calib_last_updated = None
    def prepare_advanced_features(self, tech, patterns, ticker, position, extended, regime_data=None):
        """Return raw (unscaled) engineered feature dict.
        Backwards compatible: retains earlier keys, adds new ones.
        """
        pattern_list = patterns.get('patterns', []) if isinstance(patterns, dict) else []
        bull = sum(1 for p in pattern_list if p.get('signal') == 'bullish')
        bear = sum(1 for p in pattern_list if p.get('signal') == 'bearish')
        macd = tech.get('macd', {}) if isinstance(tech, dict) else {}
        trend_struct = tech.get('trend', {}) if isinstance(tech, dict) else {}
        atr_pct = extended.get('atr', {}).get('percentage') if isinstance(extended, dict) else None
        volatility = atr_pct if isinstance(atr_pct, (int, float)) else 0.0
        regime = (regime_data or {}).get('regime') if isinstance(regime_data, dict) else None
        regime_one_hot = {
            'regime_trending': 1 if regime == 'trending' else 0,
            'regime_ranging': 1 if regime == 'ranging' else 0,
            'regime_expansion': 1 if regime == 'expansion' else 0,
            'regime_vol_crush': 1 if regime == 'volatility_crush' else 0
        }
        current_price = float(tech.get('current_price', 0)) if isinstance(tech, dict) else 0
        support = tech.get('support') if isinstance(tech, dict) else None
        resistance = tech.get('resistance') if isinstance(tech, dict) else None
        dist_support_pct = ((current_price - support) / current_price * 100) if support and current_price else 0
        dist_resist_pct = ((resistance - current_price) / current_price * 100) if resistance and current_price else 0
        feature_map = {
            # Core legacy
            'price': current_price,
            'rsi': float(tech.get('rsi', {}).get('rsi', 50)) if isinstance(tech.get('rsi'), dict) else 50,
            'macd_hist': float(macd.get('histogram', 0)) if isinstance(macd, dict) else 0,
            'pattern_count': len(pattern_list),
            'bullish_patterns': bull,
            'bearish_patterns': bear,
            'bull_bear_pattern_diff': bull - bear,
            'support_risk': float(position.get('support_risk', 0)) if isinstance(position, dict) else 0,
            'resistance_potential': float(position.get('resistance_potential', 0)) if isinstance(position, dict) else 0,
            # New technical context
            'trend_is_bull': 1 if 'bull' in str(trend_struct.get('trend', '')) else 0,
            'trend_is_bear': 1 if 'bear' in str(trend_struct.get('trend', '')) else 0,
            'trend_strength_score': {'weak': 0.3, 'moderate': 0.6, 'strong': 0.85, 'very_strong': 1.0}.get(trend_struct.get('strength', ''), 0.5),
            'volatility_atr_pct': volatility,
            'dist_support_pct': dist_support_pct,
            'dist_resistance_pct': dist_resist_pct,
            # Pattern quality aggregates
            'avg_pattern_conf': float(np.mean([p.get('confidence', 0) for p in pattern_list])) if pattern_list else 0,
            'avg_pattern_quality': float(np.mean([p.get('quality_score', 0) for p in pattern_list])) if pattern_list else 0,
            'high_reliability_patterns': sum(1 for p in pattern_list if p.get('quality_grade') in ('A', 'B')),
            # Regime one-hot
            **regime_one_hot
        }
        # Cache order once (sorted for deterministic vector)
        if self._feature_names_cache is None:
            self._feature_names_cache = sorted(feature_map.keys())
        return feature_map

    def get_feature_schema(self):
        return self._feature_names_cache or []

    # --------------------------- internal feature processing --------------------------- #
    def _vectorize(self, feature_map: dict):
        names = self._feature_names_cache or sorted(feature_map.keys())
        vec = np.zeros(self.input_dim, dtype=float)
        for i, name in enumerate(names):
            if i >= self.input_dim:
                break
            try:
                vec[i] = float(feature_map.get(name, 0))
            except Exception:
                vec[i] = 0.0
        # Online standardization (population variance) for first N dims actually used
        used_len = min(len(names), self.input_dim)
        slice_vec = vec[:used_len]
        if self._feat_mean is None:
            self._feat_mean = np.zeros(used_len)
            self._feat_m2 = np.zeros(used_len)
        # Expand stored stats if new features appear
        if used_len > len(self._feat_mean):
            pad = used_len - len(self._feat_mean)
            self._feat_mean = np.concatenate([self._feat_mean, np.zeros(pad)])
            self._feat_m2 = np.concatenate([self._feat_m2, np.zeros(pad)])
        self._feat_count += 1
        delta = slice_vec - self._feat_mean[:used_len]
        self._feat_mean[:used_len] += delta / self._feat_count
        delta2 = slice_vec - self._feat_mean[:used_len]
        self._feat_m2[:used_len] += delta * delta2
        variance = (self._feat_m2[:used_len] / max(1, self._feat_count - 1)).clip(min=1e-6)
        standardized = (slice_vec - self._feat_mean[:used_len]) / np.sqrt(variance)
        vec[:used_len] = standardized
        return vec
    def _postprocess(self, probs_arr, version_tag='JAX-v2.1'):
        probs_np = np.array(probs_arr)
        signals = ['STRONG_SELL', 'SELL', 'BUY', 'STRONG_BUY']
        idx = int(np.argmax(probs_np))
        signal = signals[idx]
        confidence = float(probs_np[idx] * 100)
        entropy = float(-(probs_np * np.log(probs_np + 1e-9)).sum())
        margin = float(np.sort(probs_np)[-1] - np.sort(probs_np)[-2]) if len(probs_np) >= 2 else 0
        max_entropy = np.log(len(probs_np))
        reliability = (0.6 * (margin) + 0.4 * (1 - entropy / max_entropy)) * 100  # 0..100
        raw_bull_prob = float(probs_np[2] + probs_np[3])  # BUY + STRONG_BUY mass
        calibrated_bull_prob = self._apply_calibration(raw_bull_prob)
        # Narrative
        if signal == 'STRONG_BUY' and confidence > 75:
            rec = '🚀 KI sehr bullish'
        elif signal == 'BUY' and confidence > 60:
            rec = '📈 Moderat bullish'
        elif signal == 'STRONG_SELL' and confidence > 75:
            rec = '📉 Stark bearish'
        elif signal == 'SELL' and confidence > 60:
            rec = '⚠️ Abwärtsrisiko'
        else:
            rec = '⚖️ Neutral'
        return {
            'signal': signal,
            'confidence': round(confidence, 2),
            'probabilities': probs_np.round(4).tolist(),
            'ai_recommendation': rec,
            'model_version': version_tag,
            'mode': self.mode,
            'reliability_score': round(reliability, 2),
            'prob_margin': round(margin, 4),
            'entropy': round(entropy, 4),
            'bull_probability_raw': round(raw_bull_prob*100,2),
            'bull_probability_calibrated': round(calibrated_bull_prob*100,2)
        }
    def predict_advanced(self, features):
        if not self.initialized or not JAX_AVAILABLE:
            return {'signal':'HOLD','confidence':50.0,'probabilities':[0.25]*4,'ai_recommendation':'offline','mode':self.mode}
        try:
            vec = self._vectorize(features)
            x=jnp.array(vec); h1=jnp.tanh(jnp.dot(x,self.model_params['w1'])+self.model_params['b1'])
            h2=jnp.tanh(jnp.dot(h1,self.model_params['w2'])+self.model_params['b2'])
            h3=jnp.tanh(jnp.dot(h2,self.model_params['w3'])+self.model_params['b3'])
            logits=jnp.dot(h3,self.model_params['w4'])+self.model_params['b4']
            # Adaptive temperature: sharper distribution when confident (low entropy)
            raw = jnp.array(logits)
            temp = max(0.25, min(4.0, self.base_temperature))
            # crude entropy proxy from logits variance
            log_var = float(jnp.var(raw))
            adapt = 1.0 / (1.0 + log_var * 0.25)
            eff_temp = temp * adapt
            scaled=logits/eff_temp; probs=jnp.exp(scaled-logsumexp(scaled))
            return self._postprocess(np.array(probs))
        except Exception as e:
            print(f"AI predict error: {e}")
            return {'signal':'HOLD','confidence':50.0,'probabilities':[0.25]*4,'ai_recommendation':f'error {e}','mode':self.mode}
    def predict_with_uncertainty(self, features, passes=10, feature_noise=0.01, dropout_rate=0.1):
        if not self.initialized or not JAX_AVAILABLE:
            base=self.predict_advanced(features); base['uncertainty']={'enabled':False}; return base
        try:
            feature_vec = self._vectorize(features)
            base_vec = feature_vec.copy()
            feats=jnp.array(base_vec); rng=self.key; probs_list=[]; keep_scale=1.0/(1.0-dropout_rate)
            for _ in range(passes):
                rng,k1,k2,k3,k4=random.split(rng,5)
                noisy=feats+random.normal(k1,feats.shape)*feature_noise
                h1=jnp.tanh(jnp.dot(noisy,self.model_params['w1'])+self.model_params['b1'])
                if dropout_rate>0:
                    h1=h1*(random.bernoulli(k2,1.0-dropout_rate,h1.shape)).astype(h1.dtype)*keep_scale
                h2=jnp.tanh(jnp.dot(h1,self.model_params['w2'])+self.model_params['b2'])
                if dropout_rate>0:
                    h2=h2*(random.bernoulli(k3,1.0-dropout_rate,h2.shape)).astype(h2.dtype)*keep_scale
                h3=jnp.tanh(jnp.dot(h2,self.model_params['w3'])+self.model_params['b3'])
                if dropout_rate>0:
                    h3=h3*(random.bernoulli(k4,1.0-dropout_rate,h3.shape)).astype(h3.dtype)*keep_scale
                logits=jnp.dot(h3,self.model_params['w4'])+self.model_params['b4']
                temp=max(0.25,min(4.0,self.temperature)); scaled=logits/temp; probs=jnp.exp(scaled-logsumexp(scaled))
                probs_list.append(np.array(probs))
            self.key=rng; arr=np.stack(probs_list,axis=0); mean=arr.mean(axis=0); std=arr.std(axis=0)
            base=self._postprocess(mean,'JAX-v2.1-MC'); entropy=float(-(mean*np.log(mean+1e-9)).sum())
            base['uncertainty']={'enabled':True,'passes':passes,'prob_std':std.round(4).tolist(),'avg_std':float(std.mean()),'entropy':round(entropy,4)}; return base
        except Exception as e:
            b=self.predict_advanced(features); b['uncertainty']={'enabled':False,'error':str(e)}; return b
    def add_training_data(self, features, actual_outcome):
        self.training_data.append({'features':features,'outcome':actual_outcome,'timestamp':datetime.now()});
        if len(self.training_data)>1000: self.training_data=self.training_data[-1000:]
        if len(self.training_data)%50==0 and len(self.training_data)>=100: self.auto_train()
        # If we captured a probability in features dict (optionally) add calibration sample
        # Expect features can include special key '__bull_prob_raw' injected by caller
        try:
            raw_p = features.get('__bull_prob_raw') if isinstance(features, dict) else None
            if isinstance(raw_p,(int,float)) and actual_outcome in ('profit','loss'):
                label = 1 if actual_outcome=='profit' else 0
                self._add_calibration_sample(raw_p, label)
        except Exception:
            pass
    def auto_train(self):
        if not JAX_AVAILABLE or len(self.training_data)<50: return
        try:
            X=np.array([d['features'] for d in self.training_data if isinstance(d.get('features'),(list,np.ndarray))])
            if X.ndim==1: X=X.reshape(-1,1)
            y=np.array([self._encode_outcome(d.get('outcome','neutral')) for d in self.training_data])
            if X.shape[0]==0: return
            lr=0.0005
            for _ in range(3):
                idx=np.random.permutation(len(X)); Xb=X[idx]; yb=y[idx]
                for i in range(len(Xb)):
                    feats=jnp.array(Xb[i]); h1=jnp.tanh(jnp.dot(feats,self.model_params['w1'])+self.model_params['b1'])
                    h2=jnp.tanh(jnp.dot(h1,self.model_params['w2'])+self.model_params['b2'])
                    h3=jnp.tanh(jnp.dot(h2,self.model_params['w3'])+self.model_params['b3'])
                    out=jnp.dot(h3,self.model_params['w4'])+self.model_params['b4']
                    err=yb[i]-out
                    self.model_params['w4']=self.model_params['w4']+lr*jnp.outer(h3,err)
                    self.model_params['b4']=self.model_params['b4']+lr*err
            self.last_train_info={'samples':len(self.training_data),'updated':datetime.now().isoformat()}
        except Exception as e:
            self.last_train_info={'error':str(e),'updated':datetime.now().isoformat()}
    def get_status(self):
        return {
            'initialized': self.initialized,
            'samples_collected': len(self.training_data),
            'last_train': getattr(self,'last_train_info',None),
            'model_version': 'JAX-v2.1' if self.initialized else 'unavailable',
            'mode': self.mode if self.initialized else 'offline',
            'feature_count': len(self._feature_names_cache or []),
            'calibration': self.get_calibration_status()
        }
    def _encode_outcome(self, outcome):
        if outcome=='profit': return [0,0,1,1]
        if outcome=='loss': return [1,1,0,0]
        return [0.25,0.25,0.25,0.25]

    # ---------------------- Calibration Helpers (Platt Scaling) ---------------------- #
    def _add_calibration_sample(self, raw_prob, label):
        raw_prob = float(min(0.999,max(0.001,raw_prob)))
        self._calib_samples.append((raw_prob,label))
        if len(self._calib_samples) > 500:  # rolling window
            self._calib_samples = self._calib_samples[-500:]
        if len(self._calib_samples) >= 40 and (self._calib_last_updated is None or (time.time()-self._calib_last_updated) > 30):
            self._update_calibration()

    def _apply_calibration(self, raw_prob: float) -> float:
        # Platt: p = 1/(1+exp(A*x + B)) ; if insufficient samples, return raw
        if len(self._calib_samples) < 20:
            return raw_prob
        A = self._calib_A; B = self._calib_B
        try:
            z = A*raw_prob + B
            return float(1.0/(1.0+math.exp(z)))
        except Exception:
            return raw_prob

    def _update_calibration(self):
        # Fit A,B by minimizing logistic loss on (raw_prob -> label)
        try:
            import math as _m
            xs = np.array([r for r,_ in self._calib_samples], dtype=float)
            ys = np.array([l for _,l in self._calib_samples], dtype=float)
            # Initialize
            A=0.0; B=0.0; lr=0.5
            for _ in range(60):
                z = A*xs + B
                # Avoid overflow
                z = np.clip(z, -20, 20)
                preds = 1/(1+np.exp(z))
                grad_A = np.mean((preds-ys)*xs)
                grad_B = np.mean(preds-ys)
                A -= lr*grad_A
                B -= lr*grad_B
                lr *= 0.98  # decay
            self._calib_A = float(A); self._calib_B = float(B); self._calib_last_updated = time.time()
        except Exception:
            pass

    def get_calibration_status(self):
        return {
            'samples': len(self._calib_samples),
            'A': round(self._calib_A,4),
            'B': round(self._calib_B,4),
            'last_update_age_s': None if self._calib_last_updated is None else round(time.time()-self._calib_last_updated,1)
        }

    # ---------------------- Persistence Helpers ---------------------- #
    def save_calibration_state(self, path: str):
        try:
            import json, os
            state = {
                'A': self._calib_A,
                'B': self._calib_B,
                'last_updated': self._calib_last_updated,
                'samples_tail': self._calib_samples[-120:]  # keep tail for warm start
            }
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state, f)
            return True
        except Exception:
            return False

    def load_calibration_state(self, path: str):
        try:
            import json, os
            if not os.path.exists(path):
                return False
            with open(path,'r',encoding='utf-8') as f:
                st = json.load(f)
            self._calib_A = float(st.get('A',0.0))
            self._calib_B = float(st.get('B',0.0))
            self._calib_last_updated = st.get('last_updated')
            tail = st.get('samples_tail') or []
            if isinstance(tail,list):
                self._calib_samples.extend([tuple(x) for x in tail if isinstance(x,(list,tuple)) and len(x)==2])
            return True
        except Exception:
            return False

    # Public shortcut for external outcome ingestion
    def add_calibration_observation(self, raw_prob: float, success: bool):
        try:
            label = 1 if success else 0
            self._add_calibration_sample(raw_prob, label)
            return True
        except Exception:
            return False
