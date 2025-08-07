#!/usr/bin/env python3
"""
🚀 ULTIMATE TRADING SYSTEM V4 - STANDALONE EXECUTABLE
JAX-Powered Neural Networks + Multi-Timeframe Analysis
Professional 70/20/10 Trading Methodology
"""

import sys
import os
import webbrowser
import time
import threading
from threading import Timer

# Füge den aktuellen Pfad hinzu für Module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def start_browser():
    """Öffne Browser nach 3 Sekunden"""
    try:
        time.sleep(3)
        print("🌐 Öffne Trading Dashboard im Browser...")
        webbrowser.open('http://127.0.0.1:5000')
    except Exception as e:
        print(f"Browser konnte nicht geöffnet werden: {e}")

def main():
    """Hauptfunktion für die EXE"""
    try:
        print("🚀 ULTIMATE TRADING SYSTEM V4 wird gestartet...")
        print("=" * 60)
        print("🧠 JAX Neural Networks: AKTIVIERT")
        print("⏰ Multi-Timeframe Analysis: AKTIVIERT") 
        print("📊 Real-time Binance Data: AKTIVIERT")
        print("🎨 Professional UI: AKTIVIERT")
        print("=" * 60)
        print("⚡ Server startet auf: http://127.0.0.1:5000")
        print("🌐 Browser öffnet automatisch in 3 Sekunden...")
        print("❌ Zum Beenden: CTRL+C drücken")
        print("=" * 60)
        
        # Starte Browser in separatem Thread
        browser_thread = threading.Thread(target=start_browser, daemon=True)
        browser_thread.start()
        
        # Importiere und starte die Hauptanwendung
        from app_turbo_new import app
        
        # Starte Flask App
        app.run(
            debug=False,  # Debug aus für EXE
            host='127.0.0.1',  # Nur localhost für Sicherheit
            port=5000,
            use_reloader=False  # Reloader aus für EXE
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Trading System wird beendet...")
        print("💎 Danke für die Nutzung!")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Fehler beim Starten: {e}")
        print("📧 Bitte den Entwickler kontaktieren")
        input("Enter drücken zum Beenden...")
        sys.exit(1)

if __name__ == "__main__":
    main()
