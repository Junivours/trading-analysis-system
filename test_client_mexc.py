#!/usr/bin/env python3
"""
Call Flask endpoints using Flask test client (no network/socket).
"""
import os
from pprint import pprint

# Ensure MEXC env are present (fallback to existing env)
os.environ.setdefault('MEXC_API_KEY', os.environ.get('MEXC_API_KEY',''))
os.environ.setdefault('MEXC_API_SECRET', os.environ.get('MEXC_API_SECRET',''))

from app import app as flask_app

print("🚀 Using Flask test client to POST /api/bot/run …")
with flask_app.test_client() as c:
    r = c.post('/api/bot/run', json={
        'symbol': 'BTCUSDT',
        'exchange': 'mexc',
        'paper': True,
        'equity': 1000,
        'risk_pct': 0.5
    })
    print('Status:', r.status_code)
    try:
        data = r.get_json()
    except Exception:
        data = {'raw': r.data.decode('utf-8','ignore')}
    pprint(data)
