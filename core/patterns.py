import numpy as np

class AdvancedPatternDetector:
    # Primitive in-memory calibration stats (können später mit echten Outcomes gefüttert werden)
    _pattern_stats = {}
    _max_window = 500

    @classmethod
    def record_pattern_outcome(cls, pattern_type: str, success: bool):
        if not pattern_type:
            return
        st = cls._pattern_stats.setdefault(pattern_type, {'observed': 0, 'success': 0})
        st['observed'] += 1
        if success:
            st['success'] += 1
        # Fenster begrenzen – einfache proportionale Reduktion falls groß
        if st['observed'] > cls._max_window:
            st['observed'] = int(st['observed'] * 0.8)
            st['success'] = int(st['success'] * 0.8)

    @classmethod
    def get_pattern_adjustment(cls, pattern_type: str):
        st = cls._pattern_stats.get(pattern_type)
        if not st or st['observed'] < 15:
            return 1.0  # keine Anpassung bis genügend Stichprobe
        rate = st['success'] / max(1, st['observed'])
        # leichte Kalibrierung: 0.8 .. 1.15
        return max(0.8, min(1.15, 0.8 + rate * 0.35))
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
        # Enhanced quality & reliability scoring
        last_price = closes[-1]
        grade_counts = {'A':0,'B':0,'C':0,'D':0}
        distance_acc = []
        for p in patterns:
            base = p.get('confidence',0)/100.0
            # Distance factor: how far price is from breakout/neckline (closer => more reliable until triggered)
            ref_level = p.get('breakout_level') or p.get('breakdown_level') or p.get('neckline') or p.get('target_level') or last_price
            dist_pct = abs(last_price - ref_level)/last_price if last_price else 0
            dist_factor = 1.0 if dist_pct < 0.5/100 else 0.85 if dist_pct < 1.0/100 else 0.7 if dist_pct < 1.8/100 else 0.5
            volume_factor = 1.0
            if 'volume' in p.get('description',''):
                if 'x Volumen' in p['description']:
                    try:
                        ratio = float(p['description'].split('x Volumen')[0].split()[-1])
                        if ratio > 1.5:
                            volume_factor = 1.05
                    except Exception:
                        pass
            reliability = base * dist_factor * volume_factor
            # Kalibrierungsanpassung je Pattern-Typ (leicht, nicht signal-killend)
            try:
                adj = AdvancedPatternDetector.get_pattern_adjustment(p.get('type',''))
            except Exception:
                adj = 1.0
            reliability *= adj
            q_score = round(reliability*100,2)
            grade = 'A' if reliability>0.82 else 'B' if reliability>0.68 else 'C' if reliability>0.52 else 'D'
            p['quality_score'] = q_score
            p['reliability_score'] = q_score
            p['distance_to_trigger_pct'] = round(dist_pct*100,3)
            p['quality_grade'] = grade
            # Validierungsflags (rein informativ)
            p['validation_flags'] = {
                'close_to_trigger': dist_pct < 0.01,
                'very_close_to_trigger': dist_pct < 0.005,
                'volume_supportive': volume_factor > 1.0,
                'high_grade': grade in ('A','B'),
                'calibration_adjustment': round(adj,3)
            }
            grade_counts[grade] = grade_counts.get(grade,0)+1
            distance_acc.append(dist_pct*100)
        avg_quality = round(sum(p.get('quality_score',0) for p in patterns)/len(patterns),1) if patterns else 0
        distance_avg = round(np.mean(distance_acc),3) if distance_acc else 0
        return {
            'patterns':patterns,
            'pattern_summary':summary,
            'visual_signals':visual_signals,
            'overall_signal':overall,
            'confidence_score': np.mean([p.get('confidence',0) for p in patterns]) if patterns else 0,
            'patterns_count':len(patterns),
            'average_quality_score':avg_quality,
            'grade_distribution': grade_counts,
            'avg_distance_to_trigger_pct': distance_avg
        }

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
        # Stricter confirmation: candle close must exceed by 1%, volume >=1.6x, and body > half ATR proxy
        body_strength = abs(recent_closes[-1]-recent_closes[-2]) / recent_closes[-2] if len(recent_closes) > 1 else 0
        if current_price > resistance * 1.01 and current_volume > avg_volume * 1.6 and body_strength > 0.004:
            return {'type':'Resistance Breakout','signal':'bullish','confidence':88,'description':f'Ausbruch über {resistance:.2f} mit {(current_volume/avg_volume):.2f}x Volumen','direction':'BULLISH','target':resistance*1.1,'strength':'VERY_STRONG'}
        support = min(recent_lows[:-3]) if len(recent_lows)>3 else min(recent_lows)
        if current_price < support * 0.99 and current_volume > avg_volume * 1.6 and body_strength > 0.004:
            return {'type':'Support Breakdown','signal':'bearish','confidence':88,'description':f'Durchbruch unter {support:.2f} mit {(current_volume/avg_volume):.2f}x Volumen','direction':'BEARISH','target':support*0.9,'strength':'VERY_STRONG'}
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
