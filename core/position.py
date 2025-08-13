class PositionManager:
    def analyze_position_potential(self, symbol, current_price, support, resistance, patterns=None):
        try:
            patterns = patterns or []
            dist_support = ((current_price - support)/support*100) if support else 0
            dist_resistance = ((resistance - current_price)/current_price*100) if resistance else 0
            support_risk = dist_support; resistance_potential = dist_resistance
            position_status = 'NEUTRAL'
            if resistance_potential > 6 and support_risk < 2: position_status = 'BULLISH'
            elif support_risk > 6 and resistance_potential < 2: position_status = 'BEARISH'
            recommendations=[]; bullish_count=0; bearish_count=0
            for pattern in patterns:
                conf=pattern.get('confidence',0); sig=pattern.get('signal')
                if conf >= 60:
                    if sig=='bullish':
                        recommendations.append({'type':'PATTERN','action':'LONG SIGNAL','reason':f"📊 {pattern.get('type','Pattern')} ({conf}%)","details":pattern.get('description',''),'confidence':conf,'color':'#28a745'}); bullish_count+=1
                    elif sig=='bearish':
                        recommendations.append({'type':'PATTERN','action':'SHORT SIGNAL','reason':f"📊 {pattern.get('type','Pattern')} ({conf}%)","details":pattern.get('description',''),'confidence':conf,'color':'#dc3545'}); bearish_count+=1
            summary={'bullish_patterns':bullish_count,'bearish_patterns':bearish_count}
            return {'resistance_potential':resistance_potential,'support_risk':support_risk,'recommendations':recommendations,'status':position_status,'summary':summary}
        except Exception as e:
            return {'resistance_potential':0,'support_risk':0,'recommendations':[{'type':'ERROR','action':'N/A','reason':str(e),'details':'PositionManager failure','confidence':0,'color':'#ffc107'}],'status':'ERROR','summary':{'bullish_patterns':0,'bearish_patterns':0}}
