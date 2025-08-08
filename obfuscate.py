"""
🔐 CODE OBFUSCATION SCRIPT
Verschleiert deinen Python-Code für Production
"""

import os
import shutil
import subprocess
import sys

def obfuscate_code():
    """Obfuskiert den Trading-Code für Deployment"""
    
    print("🔐 Starte Code-Obfuskation...")
    
    # 1. Erstelle Backup
    if os.path.exists('original_backup'):
        shutil.rmtree('original_backup')
    shutil.copytree('.', 'original_backup', ignore=shutil.ignore_patterns('original_backup', '__pycache__', '.git'))
    
    # 2. Erstelle obfuskierten Ordner
    if os.path.exists('obfuscated'):
        shutil.rmtree('obfuscated')
    os.makedirs('obfuscated')
    
    # 3. Obfuskiere Hauptdateien
    files_to_obfuscate = [
        'app.py',
        'license_manager.py'
    ]
    
    for file in files_to_obfuscate:
        if os.path.exists(file):
            print(f"🔒 Obfuskiere {file}...")
            
            # PyArmor Obfuskation
            cmd = [
                'pyarmor', 'gen', 
                '--output', 'obfuscated',
                '--pack', 'app.py',  # Bundle alles in eine Datei
                '--platform', 'linux.x86_64',  # Railway Platform
                '--restrict', '1',  # Strenge Einschränkungen
                file
            ]
            
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f"✅ {file} erfolgreich obfuskiert")
            except subprocess.CalledProcessError as e:
                print(f"❌ Fehler bei {file}: {e}")
    
    # 4. Kopiere andere notwendige Dateien
    other_files = [
        'requirements.txt',
        'Procfile',
        'runtime.txt',
        'README.md'
    ]
    
    for file in other_files:
        if os.path.exists(file):
            shutil.copy2(file, 'obfuscated/')
    
    # 5. Erstelle Production requirements.txt
    prod_requirements = """flask==2.3.3
requests==2.31.0
numpy==1.24.3
gunicorn==21.2.0
python-dotenv==1.0.0
cryptography==41.0.7
pyarmor==8.4.6
"""
    
    with open('obfuscated/requirements.txt', 'w') as f:
        f.write(prod_requirements)
    
    # 6. Erstelle verschleierte Startup-Datei
    startup_script = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os

# Lizenz-Check beim Start
try:
    from license_manager import license_manager
    is_valid, message = license_manager.validate_license()
    if not is_valid:
        print(f"🚫 UNAUTHORIZED: {message}")
        sys.exit(1)
    print(f"✅ License valid: {message}")
except ImportError:
    print("❌ License system not found")
    sys.exit(1)

# Starte obfuskierte App
if __name__ == "__main__":
    from app import app
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
'''
    
    with open('obfuscated/main.py', 'w') as f:
        f.write(startup_script)
    
    # 7. Update Procfile für obfuskierten Code
    with open('obfuscated/Procfile', 'w') as f:
        f.write('web: gunicorn main:app')
    
    print("✅ Code-Obfuskation abgeschlossen!")
    print("📁 Obfuskierter Code in 'obfuscated/' Ordner")
    print("🚀 Bereit für Railway Deployment")

if __name__ == "__main__":
    obfuscate_code()
