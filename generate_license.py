"""
🔐 LIZENZ-GENERATOR
Nur für den Entwickler - Generiert Lizenzen für Kunden
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from license_manager import LicenseManager
import argparse
from datetime import datetime

def generate_customer_license():
    """Generiert Lizenz für Kunden"""
    
    print("🔐 TRADING SYSTEM - LIZENZ GENERATOR")
    print("=" * 50)
    
    # Eingaben
    user_id = input("👤 Kunden-ID eingeben: ")
    duration_days = int(input("📅 Gültigkeitsdauer (Tage): "))
    
    # Bestätigung
    print(f"\n📋 LIZENZ-DETAILS:")
    print(f"   Kunde: {user_id}")
    print(f"   Dauer: {duration_days} Tage")
    print(f"   Ablauf: {datetime.now().strftime('%d.%m.%Y')} + {duration_days} Tage")
    
    confirm = input("\n✅ Lizenz generieren? (ja/nein): ")
    if confirm.lower() not in ['ja', 'j', 'yes', 'y']:
        print("❌ Abgebrochen")
        return
    
    # Generiere Lizenz
    manager = LicenseManager()
    
    try:
        license_data = manager.generate_license(user_id, duration_days)
        
        # Speichere Lizenz-Datei
        license_filename = f"license_{user_id}_{datetime.now().strftime('%Y%m%d')}.txt"
        with open(license_filename, 'w') as f:
            f.write(license_data)
        
        print(f"\n✅ Lizenz generiert!")
        print(f"📁 Datei: {license_filename}")
        print("\n📧 ANWEISUNGEN FÜR KUNDEN:")
        print("1. Datei als '.license' (ohne .txt) im App-Ordner speichern")
        print("2. Trading System starten")
        print("3. Lizenz wird automatisch validiert")
        
        # Validierung testen
        print("\n🧪 LIZENZ-TEST:")
        with open(license_filename, 'r') as f:
            test_license = f.read()
        
        with open('.license', 'w') as f:
            f.write(test_license)
        
        is_valid, message = manager.validate_license()
        print(f"   Status: {'✅ GÜLTIG' if is_valid else '❌ UNGÜLTIG'}")
        print(f"   Info: {message}")
        
        # Cleanup
        if os.path.exists('.license'):
            os.remove('.license')
            
    except Exception as e:
        print(f"❌ Fehler: {e}")

def main():
    parser = argparse.ArgumentParser(description='Trading System License Generator')
    parser.add_argument('--user', help='User ID')
    parser.add_argument('--days', type=int, help='Duration in days')
    
    args = parser.parse_args()
    
    if args.user and args.days:
        # Command line mode
        manager = LicenseManager()
        license_data = manager.generate_license(args.user, args.days)
        print(license_data)
    else:
        # Interactive mode
        generate_customer_license()

if __name__ == "__main__":
    main()
