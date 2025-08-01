#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Railway Deployment Starter für ULTIMATE Trading Analysis Pro
Dieser Starter löst Railway-spezifische Probleme und startet die Hauptanwendung
"""

import os
import sys
import subprocess

def main():
    """Hauptstarter für Railway Deployment"""
    print("🚀 ULTIMATE Trading Analysis Pro - Railway Starter")
    print("✅ Checking Python environment...")
    
    # Python Version Check
    print(f"Python Version: {sys.version}")
    print(f"Python Path: {sys.executable}")
    
    # Environment Variables
    port = os.environ.get('PORT', '5000')
    print(f"PORT: {port}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    
    # Überprüfe ob alle wichtigen Module verfügbar sind
    try:
        import flask
        print(f"✅ Flask: {flask.__version__}")
    except ImportError as e:
        print(f"❌ Flask import failed: {e}")
        return 1
    
    try:
        import pandas
        print(f"✅ Pandas: {pandas.__version__}")
    except ImportError as e:
        print(f"⚠️ Pandas import failed: {e}")
    
    try:
        import numpy
        print(f"✅ NumPy: {numpy.__version__}")
    except ImportError as e:
        print(f"⚠️ NumPy import failed: {e}")
    
    # Starte die Hauptanwendung
    print("🔥 Starting main application...")
    try:
        # Import und starte app.py
        from app import app
        app.run(
            host='0.0.0.0',
            port=int(port),
            debug=False
        )
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
