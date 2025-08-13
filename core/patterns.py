import numpy as np

class AdvancedPatternDetector:
    @staticmethod
    def detect_advanced_patterns(candles):
        if len(candles) < 30:
            return {'patterns': [], 'pattern_summary': 'Nicht genug Daten', 'visual_signals': [], 'overall_signal': 'NEUTRAL', 'confidence_score': 0, 'patterns_count': 0, 'average_quality_score': 0}
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        closes = [c['close'] for c in candles]
        volumes = [c['volume'] for c in candles]
        patterns = []
        visual_signals = []
        tri = AdvancedPatternDetector._detect_enhanced_triangle(highs, lows, volumes)
        if tri:
            patterns.append(tri); visual_signals.append(f"📐 {tri['type']} erkannt")
        hs = AdvancedPatternDetector._detect_head_shoulders_with_volume(highs, lows, volumes)
        if hs:
            patterns.append(hs); visual_signals.append(f"👤 {hs['type']} Pattern")
        dbl = AdvancedPatternDetector._detect_enhanced_double_patterns(highs, lows, volumes)
        if dbl:
            patterns.append(dbl); visual_signals.append(f"〰 {dbl['type']}")
        cup = AdvancedPatternDetector._detect_cup_and_handle(highs, lows, closes)
        if cup:
            patterns.append(cup); visual_signals.append("☕ Cup & Handle")
        brk = AdvancedPatternDetector._detect_breakout_patterns(highs, lows, closes, volumes)
        if brk:
            patterns.append(brk); visual_signals.append(f"🏃 {brk['direction']} Breakout")
        bullish = sum(1 for p in patterns if p.get('signal')=='bullish')
        bearish = sum(1 for p in patterns if p.get('signal')=='bearish')
        if bullish>bearish:
            overall='BULLISH'; summary=f"🚀 {bullish} bullische Patterns"
        elif bearish>bullish:
            overall='BEARISH'; summary=f"📉 {bearish} bearische Patterns"
        else:
            overall='NEUTRAL'; summary='Gemischte Pattern-Signale'
        for p in patterns:
            base = p.get('confidence',0)/100.0
            p['quality_score']=round(base*100,1)
            p['quality_grade'] = 'A' if base>0.8 else 'B' if base>0.65 else 'C' if base>0.5 else 'D'
        avg_quality = round(sum(p.get('quality_score',0) for p in patterns)/len(patterns),1) if patterns else 0
        return {'patterns':patterns,'pattern_summary':summary,'visual_signals':visual_signals,'overall_signal':overall,'confidence_score': np.mean([p.get('confidence',0) for p in patterns]) if patterns else 0,'patterns_count':len(patterns),'average_quality_score':avg_quality}

    @staticmethod
    def _detect_enhanced_triangle(highs, lows, volumes, lookback=20):
        if len(highs) < lookback:
            return None
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        recent_volumes = volumes[-lookback:]
        high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
        low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
        volume_trend = np.polyfit(range(len(recent_volumes)), recent_volumes, 1)[0]
        compression = (recent_highs[-1] - recent_lows[-1]) / recent_highs[0] if recent_highs[0] else 0
        if abs(high_trend) < 0.002 and low_trend > 0.015:
            return {'type':'Ascending Triangle','signal':'bullish','confidence':75,'description':'Höhere Tiefs, flache Hochs','breakout_level':recent_highs[-1]*1.01,'strength':'STRONG'}
        elif high_trend < -0.015 and abs(low_trend) < 0.002:
            return {'type':'Descending Triangle','signal':'bearish','confidence':75,'description':'Niedrigere Hochs, flache Tiefs','breakdown_level':recent_lows[-1]*0.99,'strength':'STRONG'}
        elif high_trend < -0.008 and low_trend > 0.008:
            vol_note = 'mit fallendem Volumen' if volume_trend < 0 else 'ohne klare Volumenbestätigung'
            return {'type':'Symmetrical Triangle','signal':'neutral','confidence':65,'description':f'Konvergenz {vol_note}','breakout_potential':'both','strength':'MEDIUM'}
        return None

    @staticmethod
    def _detect_head_shoulders_with_volume(highs, lows, volumes, lookback=25):
        if len(highs) < lookback:
            return None
        recent_highs = highs[-lookback:]; recent_volumes = volumes[-lookback:]
        max_idx = int(np.argmax(recent_highs))
        if max_idx < 3 or max_idx > lookback - 4:
            return None
        left = recent_highs[max_idx-3:max_idx]; right = recent_highs[max_idx+1:max_idx+4]
        left_avg = np.mean(left); right_avg = np.mean(right)
        head = recent_highs[max_idx]
        if head < left_avg*1.02 or head < right_avg*1.02:
            return None
        neckline = (lows[-lookback] + lows[-1]) / 2 if lookback>1 else lows[-1]
        head_ratio = head / max(left_avg, right_avg)
        vol_left = np.mean(recent_volumes[max_idx-3:max_idx]) if max_idx>=3 else 0
        vol_right = np.mean(recent_volumes[max_idx+1:max_idx+4]) if max_idx+4<=lookback else 0
        vol_pattern = 'bearish_volume_confirmation' if vol_right < vol_left*0.8 else 'weak_volume_confirmation'
        conf = 88 if vol_pattern=='bearish_volume_confirmation' else 75
        return {'type':'Head and Shoulders','signal':'bearish','confidence':conf,'description':f'KR: {head_ratio:.2f}, Vol: {vol_pattern}','neckline':neckline,'target':neckline - (head - neckline),'strength':'VERY_STRONG' if conf>80 else 'STRONG'}

    @staticmethod
    def _detect_enhanced_double_patterns(highs, lows, volumes, lookback=20):
        if len(highs) < lookback:
            return None
        recent_highs = highs[-lookback:]; recent_lows = lows[-lookback:]
        h_sorted = sorted(recent_highs, reverse=True)[:3]; l_sorted = sorted(recent_lows)[:3]
        if len(h_sorted) < 2 or len(l_sorted) < 2:
            return None
        high_diff = abs(h_sorted[0] - h_sorted[1]) / ((h_sorted[0] + h_sorted[1]) / 2)
        low_diff = abs(l_sorted[0] - l_sorted[1]) / ((l_sorted[0] + l_sorted[1]) / 2)
        if high_diff < 0.005:
            return {'type':'Double Top','signal':'bearish','confidence':70,'description':f'Doppeltes Top ~{high_diff:.2%}','target_level':min(recent_lows[-5:]),'strength':'MEDIUM'}
        if low_diff < 0.005:
            return {'type':'Double Bottom','signal':'bullish','confidence':70,'description':f'Doppelter Boden ~{low_diff:.2%}','target_level':max(recent_highs[-5:]),'strength':'MEDIUM'}
        return None

    @staticmethod
    def _detect_cup_and_handle(highs, lows, closes, lookback=30):
        if len(closes) < lookback:
            return None
        recent_closes = closes[-lookback:]
        max_price = max(recent_closes); min_price = min(recent_closes)
        start_price = recent_closes[0]; end_price = recent_closes[-1]
        depth = (max_price - min_price)/max_price if max_price else 0
        recovery = (end_price - min_price)/(max_price - min_price) if (max_price - min_price) else 0
        if 0.1 < depth < 0.5 and recovery > 0.7 and end_price < max_price*0.98:
            h_start = int(lookback * 0.7); hd = recent_closes[h_start:]
            if len(hd) > 5:
                h_high = max(hd); h_low = min(hd); h_depth = (h_high - h_low)/h_high if h_high else 0
                if 0.05 < h_depth < 0.18:
                    return {'type':'Cup and Handle','signal':'bullish','confidence':82,'description':f'Cup-Tiefe: {depth:.1%}, Handle-Korrektur: {h_depth:.1%}','breakout_level':h_high*1.02,'target':h_high*(1+depth),'strength':'VERY_STRONG'}
        return None

    @staticmethod
    def _detect_breakout_patterns(highs, lows, closes, volumes, lookback=15):
        if len(closes) < lookback:
            return None
        recent_highs = highs[-lookback:]; recent_lows = lows[-lookback:]; recent_closes = closes[-lookback:]; recent_volumes = volumes[-lookback:]
        current_price = recent_closes[-1]; avg_volume = np.mean(recent_volumes[:-5]) if len(recent_volumes) > 5 else np.mean(recent_volumes)
        current_volume = recent_volumes[-1]
        resistance = max(recent_highs[:-3]) if len(recent_highs)>3 else max(recent_highs)
        if current_price > resistance * 1.02 and current_volume > avg_volume * 1.5:
            return {'type':'Resistance Breakout','signal':'bullish','confidence':85,'description':f'Ausbruch über {resistance:.2f} mit {(current_volume/avg_volume):.1f}x Volumen','direction':'BULLISH','target':resistance*1.1,'strength':'VERY_STRONG'}
        support = min(recent_lows[:-3]) if len(recent_lows)>3 else min(recent_lows)
        if current_price < support * 0.98 and current_volume > avg_volume * 1.5:
            return {'type':'Support Breakdown','signal':'bearish','confidence':85,'description':f'Durchbruch unter {support:.2f} mit {(current_volume/avg_volume):.1f}x Volumen','direction':'BEARISH','target':support*0.9,'strength':'VERY_STRONG'}
        return None

class ChartPatternTrader:
    @staticmethod
    def generate_pattern_trades(symbol, pattern_analysis, tech_analysis, extended_analysis, current_price):
        trades = []
        if not pattern_analysis or not isinstance(pattern_analysis, dict):
            return trades
        patterns = pattern_analysis.get('patterns', [])
        support = tech_analysis.get('support') if isinstance(tech_analysis, dict) else None
        resistance = tech_analysis.get('resistance') if isinstance(tech_analysis, dict) else None
        atr = extended_analysis.get('atr') if isinstance(extended_analysis, dict) else None
        if isinstance(atr, dict):
            atr_value = atr.get('atr') or atr.get('value')
        else:
            atr_value = atr
        risk_unit = atr_value if atr_value and atr_value > 0 else (current_price * 0.01)
        for p in patterns:
            ptype = p.get('type','')
            signal = p.get('signal','neutral')
            conf = p.get('confidence',50)
            strength = p.get('strength','MEDIUM')
            if signal == 'bullish':
                entry = p.get('breakout_level') or p.get('target_level') or (resistance if resistance else current_price*1.01)
                stop = p.get('stop_level') or (support if support else current_price * 0.95)
                target = p.get('target') or (entry + risk_unit*2)
                r = (target-entry)/(entry-stop) if entry!=stop else 0
                trades.append({
                    'symbol': symbol,
                    'pattern': ptype,
                    'direction': 'LONG',
                    'entry': round(entry,6),
                    'stop': round(stop,6),
                    'target': round(target,6),
                    'rr': round(r,2),
                    'confidence': conf,
                    'strength': strength
                })
            elif signal == 'bearish':
                entry = p.get('breakdown_level') or p.get('target_level') or (support if support else current_price*0.99)
                stop = p.get('stop_level') or (resistance if resistance else current_price * 1.05)
                target = p.get('target') or (entry - risk_unit*2)
                r = (entry-target)/(stop-entry) if stop!=entry else 0
                trades.append({
                    'symbol': symbol,
                    'pattern': ptype,
                    'direction': 'SHORT',
                    'entry': round(entry,6),
                    'stop': round(stop,6),
                    'target': round(target,6),
                    'rr': round(r,2),
                    'confidence': conf,
                    'strength': strength
                })
        trades.sort(key=lambda x: (x['confidence'], x['rr']), reverse=True)
        return trades[:5]
