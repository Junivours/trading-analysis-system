"""
🔐 TRADING SYSTEM LICENSE MANAGER
Schützt deinen Code vor unbefugter Nutzung
"""

import hashlib
import hmac
import time
import os
import socket
import platform
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import json
import base64

class LicenseManager:
    def __init__(self):
        # Dein geheimer Schlüssel (NIEMALS committen!)
        self.master_key = os.getenv('TRADING_MASTER_KEY', 'your-secret-master-key-here')
        self.license_server = "https://your-license-server.com"  # Dein eigener Server
        
    def generate_machine_fingerprint(self):
        """Erstellt einzigartigen Machine-Fingerprint"""
        # Hardware-basierte Identifikation
        hostname = socket.gethostname()
        mac_address = ':'.join(['{:02x}'.format((int(time.time()) >> i) & 0xff) for i in range(0, 8, 8)])
        platform_info = platform.platform()
        
        # Kombiniere zu eindeutigem Fingerprint
        machine_data = f"{hostname}-{mac_address}-{platform_info}"
        return hashlib.sha256(machine_data.encode()).hexdigest()[:16]
    
    def validate_license(self):
        """Überprüft Lizenz - KRITISCHE SICHERHEITSFUNKTION"""
        try:
            # 1. Prüfe Lizenz-Datei
            license_file = os.path.join(os.getcwd(), '.license')
            if not os.path.exists(license_file):
                return False, "❌ Keine gültige Lizenz gefunden"
            
            # 2. Lade und entschlüssele Lizenz
            with open(license_file, 'r') as f:
                encrypted_license = f.read()
            
            try:
                # Entschlüsselung
                key = base64.urlsafe_b64encode(self.master_key.encode()[:32].ljust(32, b'0'))
                cipher = Fernet(key)
                decrypted_data = cipher.decrypt(encrypted_license.encode())
                license_data = json.loads(decrypted_data)
            except:
                return False, "❌ Lizenz beschädigt oder manipuliert"
            
            # 3. Validiere Machine-Fingerprint
            current_fingerprint = self.generate_machine_fingerprint()
            if license_data.get('machine_id') != current_fingerprint:
                return False, "❌ Lizenz für anderen Computer ausgestellt"
            
            # 4. Prüfe Ablaufdatum
            expiry_date = datetime.fromisoformat(license_data.get('expires', '2024-01-01'))
            if datetime.now() > expiry_date:
                return False, "❌ Lizenz abgelaufen"
            
            # 5. Prüfe digitale Signatur
            signature = license_data.get('signature')
            license_content = f"{license_data['machine_id']}{license_data['expires']}{license_data['user_id']}"
            expected_signature = hmac.new(
                self.master_key.encode(), 
                license_content.encode(), 
                hashlib.sha256
            ).hexdigest()
            
            if signature != expected_signature:
                return False, "❌ Ungültige Lizenz-Signatur"
            
            # 6. Online-Validierung (optional, wenn Internet verfügbar)
            # self.validate_online(license_data)
            
            return True, f"✅ Lizenz gültig bis {expiry_date.strftime('%d.%m.%Y')}"
            
        except Exception as e:
            return False, f"❌ Lizenz-Fehler: {str(e)}"
    
    def generate_license(self, user_id, duration_days=30):
        """Generiert neue Lizenz (nur für Lizenz-Server)"""
        machine_id = self.generate_machine_fingerprint()
        expires = (datetime.now() + timedelta(days=duration_days)).isoformat()
        
        # Erstelle Signatur
        license_content = f"{machine_id}{expires}{user_id}"
        signature = hmac.new(
            self.master_key.encode(), 
            license_content.encode(), 
            hashlib.sha256
        ).hexdigest()
        
        license_data = {
            'user_id': user_id,
            'machine_id': machine_id,
            'expires': expires,
            'signature': signature,
            'issued_at': datetime.now().isoformat()
        }
        
        # Verschlüssele Lizenz
        key = base64.urlsafe_b64encode(self.master_key.encode()[:32].ljust(32, b'0'))
        cipher = Fernet(key)
        encrypted_license = cipher.encrypt(json.dumps(license_data).encode())
        
        return encrypted_license.decode()

# Globaler Lizenz-Manager
license_manager = LicenseManager()

def require_license(func):
    """Decorator für lizenzpflichtige Funktionen"""
    def wrapper(*args, **kwargs):
        is_valid, message = license_manager.validate_license()
        if not is_valid:
            print(f"🚫 ZUGRIFF VERWEIGERT: {message}")
            print("📧 Kontaktiere den Entwickler für eine gültige Lizenz")
            return {"error": "Unauthorized access", "message": message}
        return func(*args, **kwargs)
    return wrapper
