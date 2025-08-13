import os, time, numpy as np
from datetime import datetime
try:
    import jax.numpy as jnp
    from jax.nn import logsumexp
    from jax import random
    JAX_AVAILABLE = True
except Exception:
    JAX_AVAILABLE = False

class AdvancedJAXAI:
    def __init__(self, input_dim: int = 128, mode: str = 'live', temperature: float = 1.0):
        self.input_dim = input_dim; self.mode = mode; self.temperature = temperature
        self.initialized = False; self.training_data = []; self.key=None; self.model_params={}
        try:
            if JAX_AVAILABLE: self._init_model()
        except Exception as e:
            print(f"AI init failed: {e}"); self.initialized=False
    def _init_model(self):
        self.key = random.PRNGKey(int(time.time()) & 0xFFFFFFFF)
        k1,k2,k3,k4 = random.split(self.key,4)
        self.model_params={
            'w1': random.normal(k1,(self.input_dim,96))*0.05,'b1': jnp.zeros(96),
            'w2': random.normal(k2,(96,64))*0.05,'b2': jnp.zeros(64),
            'w3': random.normal(k3,(64,32))*0.05,'b3': jnp.zeros(32),
            'w4': random.normal(k4,(32,4))*0.05,'b4': jnp.zeros(4)}; self.initialized=True
    def prepare_advanced_features(self, tech, patterns, ticker, position, extended, regime_data=None):
        return {
            'price': float(tech.get('current_price', 0)) if isinstance(tech, dict) else 0,
            'rsi': float(tech.get('rsi', {}).get('rsi', 50)) if isinstance(tech.get('rsi'), dict) else 50,
            'macd_hist': float(tech.get('macd', {}).get('histogram', 0)) if isinstance(tech.get('macd'), dict) else 0,
            'pattern_count': len(patterns.get('patterns', [])) if isinstance(patterns, dict) else 0,
            'bullish_patterns': sum(1 for p in patterns.get('patterns', []) if p.get('signal')=='bullish') if isinstance(patterns, dict) else 0,
            'bearish_patterns': sum(1 for p in patterns.get('patterns', []) if p.get('signal')=='bearish') if isinstance(patterns, dict) else 0,
            'support_risk': float(position.get('support_risk',0)) if isinstance(position, dict) else 0,
            'resistance_potential': float(position.get('resistance_potential',0)) if isinstance(position, dict) else 0
        }
    def _postprocess(self, probs_arr, version_tag='JAX-v2.0'):
        probs_np = np.array(probs_arr); signals=['STRONG_SELL','SELL','BUY','STRONG_BUY']
        idx=int(np.argmax(probs_np)); signal=signals[idx]; confidence=float(probs_np[idx]*100)
        if signal=='STRONG_BUY' and confidence>75: rec='🚀 KI sehr bullish'
        elif signal=='BUY' and confidence>60: rec='📈 Moderat bullish'
        elif signal=='STRONG_SELL' and confidence>75: rec='📉 Stark bearish'
        elif signal=='SELL' and confidence>60: rec='⚠️ Abwärtsrisiko'
        else: rec='⚖️ Neutral'
        return {'signal':signal,'confidence':round(confidence,2),'probabilities':probs_np.round(4).tolist(),'ai_recommendation':rec,'model_version':version_tag,'mode':self.mode}
    def predict_advanced(self, features):
        if not self.initialized or not JAX_AVAILABLE:
            return {'signal':'HOLD','confidence':50.0,'probabilities':[0.25]*4,'ai_recommendation':'offline','mode':self.mode}
        try:
            vec=np.zeros(self.input_dim)
            for i,(k,v) in enumerate(sorted(features.items())):
                if i>=self.input_dim: break
                try: vec[i]=float(v)
                except: vec[i]=0.0
            x=jnp.array(vec); h1=jnp.tanh(jnp.dot(x,self.model_params['w1'])+self.model_params['b1'])
            h2=jnp.tanh(jnp.dot(h1,self.model_params['w2'])+self.model_params['b2'])
            h3=jnp.tanh(jnp.dot(h2,self.model_params['w3'])+self.model_params['b3'])
            logits=jnp.dot(h3,self.model_params['w4'])+self.model_params['b4']
            temp=max(0.25,min(4.0,self.temperature)); scaled=logits/temp; probs=jnp.exp(scaled-logsumexp(scaled))
            return self._postprocess(np.array(probs))
        except Exception as e:
            print(f"AI predict error: {e}")
            return {'signal':'HOLD','confidence':50.0,'probabilities':[0.25]*4,'ai_recommendation':f'error {e}','mode':self.mode}
    def predict_with_uncertainty(self, features, passes=10, feature_noise=0.01, dropout_rate=0.1):
        if not self.initialized or not JAX_AVAILABLE:
            base=self.predict_advanced(features); base['uncertainty']={'enabled':False}; return base
        try:
            base_vec=np.zeros(self.input_dim)
            for i,(k,v) in enumerate(sorted(features.items())):
                if i>=self.input_dim: break
                try: base_vec[i]=float(v)
                except: pass
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
            base=self._postprocess(mean,'JAX-v2.0-MC'); entropy=float(-(mean*np.log(mean+1e-9)).sum())
            base['uncertainty']={'enabled':True,'passes':passes,'prob_std':std.round(4).tolist(),'avg_std':float(std.mean()),'entropy':round(entropy,4)}; return base
        except Exception as e:
            b=self.predict_advanced(features); b['uncertainty']={'enabled':False,'error':str(e)}; return b
    def add_training_data(self, features, actual_outcome):
        self.training_data.append({'features':features,'outcome':actual_outcome,'timestamp':datetime.now()});
        if len(self.training_data)>1000: self.training_data=self.training_data[-1000:]
        if len(self.training_data)%50==0 and len(self.training_data)>=100: self.auto_train()
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
        return {'initialized':self.initialized,'samples_collected':len(self.training_data),'last_train':getattr(self,'last_train_info',None),'model_version':'JAX-v2.0' if self.initialized else 'unavailable','mode':self.mode if self.initialized else 'offline'}
    def _encode_outcome(self, outcome):
        if outcome=='profit': return [0,0,1,1]
        if outcome=='loss': return [1,1,0,0]
        return [0.25,0.25,0.25,0.25]
