from collections import deque
from datetime import datetime

class SymbolBehaviorProfiler:
    def __init__(self):
        self.symbol_stats={}; self.last_profile=None
    def update(self, symbol, candles, pattern_analysis, tech_analysis):
        pats=pattern_analysis.get('patterns',[]) if isinstance(pattern_analysis,dict) else []
        total=len(pats); bull=sum(1 for p in pats if p.get('signal')=='bullish'); bear=sum(1 for p in pats if p.get('signal')=='bearish')
        directional=bull+bear; unique_types=len({p.get('type') for p in pats if p.get('type')}); diversity=(unique_types/total) if total>0 else 0.0
        bias_strength=(bull-bear)/directional if directional>0 else 0.0
        atr_pct=None
        try: atr_pct=tech_analysis.get('atr',{}).get('percentage')
        except Exception: atr_pct=None
        if atr_pct is None:
            try:
                if candles:
                    last=candles[-1]; cp=float(last.get('close',1.0)) or 1.0
                    rng=(float(last.get('high',cp))-float(last.get('low',cp)))/cp*100; atr_pct=rng
                else: atr_pct=2.0
            except Exception: atr_pct=2.0
        stat=self.symbol_stats.setdefault(symbol,{'atr_samples':deque(maxlen=60),'profiles':deque(maxlen=40)})
        try:
            stat['atr_samples'].append(float(atr_pct)); atr_avg=sum(stat['atr_samples'])/len(stat['atr_samples'])
        except Exception: atr_avg=float(atr_pct)
        vol_rank=0.5
        try:
            samples=list(stat['atr_samples'])
            if samples:
                lower=sum(1 for v in samples if v <= atr_pct); vol_rank=lower/len(samples)
        except Exception: pass
        recent_intensity=sum(1 for p in pats if p.get('confidence',0) >= 60)/total if total>0 else 0.0
        bull_rate=(bull/directional) if directional>0 else 0.0; bear_rate=(bear/directional) if directional>0 else 0.0
        profile={'symbol':symbol,'pattern_bias_strength':round(bias_strength,4),'pattern_diversity':round(diversity,4),'volatility_pct_rank':round(vol_rank,4),'avg_atr_pct':round(atr_avg,4),'bullish_pattern_rate':round(bull_rate,4),'bearish_pattern_rate':round(bear_rate,4),'recent_pattern_intensity':round(recent_intensity,4),'pattern_count':total,'timestamp':datetime.utcnow().isoformat()+"Z"}
        try: stat['profiles'].append(profile)
        except Exception: pass
        self.last_profile=profile; return profile
    def get_last_profile(self): return self.last_profile
