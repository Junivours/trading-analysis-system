import time


def analyze_trade_setups(setups):
    findings = []
    ideas = []
    if not setups:
        findings.append({'type': 'NO_SETUPS', 'severity': 'info', 'message': 'Keine Trade-Setups generiert (evtl. No-Trade-Zone aktiv oder widersprüchliche Signale).'})
        ideas.append('Wenn Markt in enger Range: Breakout-Watchlist vorbereiten statt forcierter Entries.')
        return findings, ideas
    if len(setups) > 2:
        findings.append({'type': 'SETUP_PRUNING', 'severity': 'warn', 'message': f'{len(setups)} Setups geliefert – Backend sollte auf max 2 prunen.'})
        ideas.append('Pruning-Regel prüfen – evtl. Ranking-Kriterien nachschärfen (RR * Confidence / Risk).')
    seen_ids = set()
    for s in setups:
        sid = s.get('id')
        if sid in seen_ids:
            findings.append({'type': 'DUPLICATE_ID', 'severity': 'warn', 'message': f'Doppeltes Setup-ID {sid} erkannt.'})
        else:
            seen_ids.add(sid)
        risk = s.get('risk_percent')
        if isinstance(risk,(int,float)) and risk > 3.0:
            findings.append({'type': 'HIGH_RISK', 'severity': 'warn', 'message': f'Setup {sid} Risk {risk}% > 3% Cap.'})
        targets = s.get('targets')
        if targets and isinstance(targets, list):
            # check monotonic RR
            rrs = [t.get('rr') for t in targets if isinstance(t, dict)]
            if any(rrs[i] >= rrs[i+1] for i in range(len(rrs)-1) if isinstance(rrs[i], (int,float)) and isinstance(rrs[i+1], (int,float))):
                findings.append({'type':'RR_ORDER','severity':'info','message': f'Setup {sid} RR nicht strikt steigend.'})
        else:
            findings.append({'type': 'NO_TARGETS', 'severity': 'error', 'message': f'Setup {sid} hat keine Targets.'})
    return findings, ideas


def analyze_ai(ai_analysis):
    findings = []
    ideas = []
    if not ai_analysis:
        findings.append({'type':'AI_MISSING','severity':'error','message':'AI Analyse fehlt.'})
        return findings, ideas
    conf = ai_analysis.get('confidence')
    rel = ai_analysis.get('reliability_score')
    if isinstance(conf,(int,float)) and conf < 45:
        findings.append({'type':'LOW_CONFIDENCE','severity':'warn','message': f'AI Confidence niedrig ({conf}%).'})
        ideas.append('Mehr Datendiversität oder Feature-Normalisierung prüfen um Confidence zu stabilisieren.')
    if isinstance(rel,(int,float)) and rel < 40:
        findings.append({'type':'LOW_RELIABILITY','severity':'warn','message': f'AI Reliability gering ({rel}%).'})
        ideas.append('Reliability unter 40%: ggf. Ensemble / Plausibilitätsfilter härter gewichten.')
    if ai_analysis.get('signal') == 'HOLD' and isinstance(conf,(int,float)) and conf > 70:
        findings.append({'type':'HOLD_HIGH_CONF','severity':'info','message':'HOLD mit hoher Confidence – Neutralität vermutlich strukturell.'})
        ideas.append('Neutral aber hohe Sicherheit: Range-Strategien (Mean Reversion) vorbereiten.')
    return findings, ideas


def analyze_explainability(meta):
    findings = []
    ideas = []
    if not meta:
        findings.append({'type':'EXPLAIN_META_MISSING','severity':'warn','message':'Explainability Meta fehlt.'})
        ideas.append('Serverseitige Explainability aktiv halten für Vertrauensaufbau.')
        return findings, ideas
    if meta.get('error'):
        findings.append({'type':'EXPLAIN_META_ERROR','severity':'error','message': meta.get('error')})
    neg = len(meta.get('reasons_negative') or [])
    pos = len(meta.get('reasons_positive') or [])
    if neg and pos == 0:
        ideas.append('Nur negative Gründe -> eventuell Modell-Bias Richtung Risiko reduzieren.')
    if pos and neg == 0:
        ideas.append('Nur unterstützende Gründe -> Check auf Overconfidence durchführen.')
    return findings, ideas


def analyze_feature_contributions(fc):
    findings = []
    ideas = []
    if not fc:
        findings.append({'type':'FEATURES_MISSING','severity':'warn','message':'Feature Contributions fehlen.'})
        return findings, ideas
    top = fc.get('top_features') or []
    if len(top) < 2:
        findings.append({'type':'LOW_FEATURE_DIVERSITY','severity':'info','message':'Wenige dominante Features.'})
        ideas.append('Feature Space erweitern (Regime / Volumetrische / Inter-MTF Relationen).')
    method = fc.get('analysis_method')
    if method != 'z_score':
        findings.append({'type':'METHOD_CHANGED','severity':'info','message': f'Analyse-Methode jetzt {method}.'})
    return findings, ideas


def analyze_consistency(final_score, multi_timeframe):
    findings = []
    ideas = []
    try:
        fs_sig = final_score.get('signal') if isinstance(final_score, dict) else None
        mt_primary = multi_timeframe.get('consensus', {}).get('primary') if isinstance(multi_timeframe, dict) else None
        if fs_sig and mt_primary and mt_primary != 'UNKNOWN':
            if (fs_sig in ['BUY','STRONG_BUY'] and mt_primary == 'BEARISH') or (fs_sig in ['SELL','STRONG_SELL'] and mt_primary == 'BULLISH'):
                findings.append({'type':'SIGNAL_CONSENSUS_CONFLICT','severity':'warn','message': f'Final Signal {fs_sig} vs MTF {mt_primary}.'})
                ideas.append('Konflikt -> Gewicht dynamisch reduzieren oder Confirmation Delay einbauen.')
    except Exception:
        pass
    return findings, ideas


def run_symbol_diagnostics(analysis):
    start = time.time()
    findings = []
    ideas = []
    if not analysis or analysis.get('error'):
        return {'status':'error','findings':[{'type':'ANALYSIS_ERROR','severity':'error','message':analysis.get('error') if analysis else 'unknown'}],'ideas':[], 'latency_ms': round((time.time()-start)*1000,2)}
    ts_find, ts_ideas = analyze_trade_setups(analysis.get('trade_setups'))
    findings.extend(ts_find); ideas.extend(ts_ideas)
    ai_find, ai_ideas = analyze_ai(analysis.get('ai_analysis'))
    findings.extend(ai_find); ideas.extend(ai_ideas)
    ex_find, ex_ideas = analyze_explainability(analysis.get('ai_explainability_meta'))
    findings.extend(ex_find); ideas.extend(ex_ideas)
    fc = (analysis.get('ai_analysis') or {}).get('feature_contributions')
    fc_find, fc_ideas = analyze_feature_contributions(fc)
    findings.extend(fc_find); ideas.extend(fc_ideas)
    cs_find, cs_ideas = analyze_consistency(analysis.get('final_score'), analysis.get('multi_timeframe'))
    findings.extend(cs_find); ideas.extend(cs_ideas)
    # Score heuristics for readiness
    error_ct = sum(1 for f in findings if f['severity']=='error')
    warn_ct = sum(1 for f in findings if f['severity']=='warn')
    readiness = 'GOOD'
    if error_ct:
        readiness = 'BAD'
    elif warn_ct > 2:
        readiness = 'ATTENTION'
    summary = {
        'status': 'ok',
        'readiness': readiness,
        'error_count': error_ct,
        'warning_count': warn_ct,
        'finding_count': len(findings),
        'idea_count': len(ideas),
        'latency_ms': round((time.time()-start)*1000,2),
        'findings': findings,
        'ideas': sorted(set(ideas))
    }
    return summary
