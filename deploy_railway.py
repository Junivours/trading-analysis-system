"""
🚀 RAILWAY DEPLOYMENT SCRIPT
Automatisiertes, sicheres Deployment
"""

import os
import shutil
import subprocess
import sys

def setup_railway_deployment():
    """Bereitet sicheres Railway Deployment vor"""
    
    print("🚀 Railway Deployment Setup...")
    
    # 1. Obfuskiere Code erst
    print("🔐 Starte Code-Obfuskation...")
    from obfuscate import obfuscate_code
    obfuscate_code()
    
    # 2. Erstelle Railway-spezifische Konfiguration
    railway_config = {
        "build": {
            "builder": "NIXPACKS"
        },
        "deploy": {
            "restartPolicyType": "ON_FAILURE",
            "restartPolicyMaxRetries": 3
        }
    }
    
    # 3. Erstelle nixpacks.toml für Railway
    nixpacks_config = """
[phases.setup]
nixPkgs = ["python310", "pip"]

[phases.install]
cmds = ["pip install -r requirements.txt"]

[phases.build]
cmds = ["echo 'Build completed'"]

[start]
cmd = "gunicorn main:app"
"""
    
    with open('obfuscated/nixpacks.toml', 'w') as f:
        f.write(nixpacks_config)
    
    # 4. Erstelle Railway-optimierte requirements.txt
    railway_requirements = """flask==2.3.3
gunicorn==21.2.0
requests==2.31.0
numpy==1.24.3
python-dotenv==1.0.0
cryptography==41.0.7
"""
    
    with open('obfuscated/requirements.txt', 'w') as f:
        f.write(railway_requirements)
    
    # 5. Erstelle deployment instructions
    instructions = """
🚀 RAILWAY DEPLOYMENT ANLEITUNG:

1. Railway Account erstellen: https://railway.app
2. GitHub Repository verbinden (nur obfuscated Ordner!)
3. Environment Variables setzen:
   - TRADING_MASTER_KEY=your-secret-key
   - FLASK_ENV=production
   - PORT=5000

4. Deployment starten
5. Custom Domain hinzufügen (Optional)

🔐 SICHERHEITS-CHECKLISTE:
✅ Code obfuskiert
✅ Lizenz-System aktiv
✅ Environment Variables gesetzt
✅ .env Dateien nicht committed
✅ Backup erstellt

📧 LIZENZ-GENERIERUNG:
- Kontaktiere dich selbst für neue Lizenzen
- Machine-ID wird automatisch generiert
- Lizenzen sind hardware-gebunden

🛡️ SCHUTZ-EBENEN:
1. Code-Obfuskation (PyArmor)
2. Lizenz-System mit Hardware-Binding
3. Environment-Variable Schutz
4. Digital Signatures
5. Online-Validierung (optional)
"""
    
    with open('DEPLOYMENT_INSTRUCTIONS.txt', 'w') as f:
        f.write(instructions)
    
    print("✅ Railway Deployment Setup abgeschlossen!")
    print("📁 Sichere Dateien in 'obfuscated/' Ordner")
    print("📋 Lies DEPLOYMENT_INSTRUCTIONS.txt für Details")

if __name__ == "__main__":
    setup_railway_deployment()
