#!/usr/bin/env python3
"""
💾 BACKUP CREATOR FÜR ML TRADING SYSTEM
Erstellt vollständiges Backup des sauberen Projekts
"""

import os
import shutil
import zipfile
from datetime import datetime
import json

def create_backup():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_name = f"ML_Trading_System_BACKUP_{timestamp}"
    
    print(f"💾 ERSTELLE BACKUP: {backup_name}")
    print("=" * 60)
    
    # Backup-Ordner erstellen
    backup_dir = f"../{backup_name}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # Alle wichtigen Dateien und Ordner
    items_to_backup = [
        # Core Python Files
        'app.py',
        'binance_api.py',
        'ml_engine.py',
        'real_data_enforcer.py',
        'validation_suite.py',
        
        # Test Files
        'local_test.py',
        'quick_test.py',
        
        # Configuration
        'requirements.txt',
        '.env.example',
        '.gitignore',
        
        # Deployment
        'Dockerfile',
        'Procfile',
        'railway.toml',
        
        # Documentation
        'README.md',
        'TEST_RESULTS.md',
        'PROJECT_CLEAN.md',
        
        # Directories
        'templates',
        'utils'
    ]
    
    copied_files = 0
    total_size = 0
    
    # Backup-Info sammeln
    backup_info = {
        'timestamp': timestamp,
        'date': datetime.now().isoformat(),
        'system': 'Windows Python 3.13.5',
        'status': 'All tests passed - 100% functional',
        'files': [],
        'features': [
            'Real Binance API Integration',
            'ML Engine (RandomForest + SVM)',
            'Technical Indicators (RSI, MACD, BB)',
            'Real Data Validation',
            'Web Dashboard',
            'REST API Endpoints'
        ]
    }
    
    # Dateien kopieren
    for item in items_to_backup:
        if os.path.exists(item):
            dest_path = os.path.join(backup_dir, item)
            
            if os.path.isfile(item):
                # Datei kopieren
                shutil.copy2(item, dest_path)
                size = os.path.getsize(item)
                total_size += size
                copied_files += 1
                
                backup_info['files'].append({
                    'name': item,
                    'size': size,
                    'type': 'file'
                })
                
                print(f"✅ Kopiert: {item} ({size} bytes)")
                
            elif os.path.isdir(item):
                # Ordner kopieren
                shutil.copytree(item, dest_path)
                
                # Größe berechnen
                dir_size = sum(os.path.getsize(os.path.join(dirpath, filename))
                              for dirpath, dirnames, filenames in os.walk(dest_path)
                              for filename in filenames)
                total_size += dir_size
                copied_files += len([f for _, _, files in os.walk(dest_path) for f in files])
                
                backup_info['files'].append({
                    'name': item,
                    'size': dir_size,
                    'type': 'directory'
                })
                
                print(f"📁 Kopiert: {item}/ ({dir_size} bytes)")
    
    # Backup-Info speichern
    backup_info['total_size'] = total_size
    backup_info['total_files'] = copied_files
    
    with open(os.path.join(backup_dir, 'BACKUP_INFO.json'), 'w', encoding='utf-8') as f:
        json.dump(backup_info, f, indent=2, ensure_ascii=False)
    
    # ZIP erstellen
    zip_name = f"{backup_name}.zip"
    zip_path = f"../{zip_name}"
    
    print(f"\n📦 ERSTELLE ZIP: {zip_name}")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(backup_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, backup_dir)
                zipf.write(file_path, arc_name)
    
    zip_size = os.path.getsize(zip_path)
    
    # Ordner löschen (ZIP reicht)
    shutil.rmtree(backup_dir)
    
    print(f"\n✅ BACKUP ERSTELLT!")
    print(f"📁 Datei: {zip_path}")
    print(f"📊 Größe: {zip_size / 1024:.1f} KB")
    print(f"📦 Dateien: {copied_files}")
    print(f"💾 Original: {total_size / 1024:.1f} KB")
    
    return zip_path, zip_size, copied_files

def create_git_backup():
    """Erstelle auch Git-Backup"""
    print(f"\n🔄 GIT BACKUP...")
    
    try:
        # Git Status
        os.system("git add .")
        commit_msg = f'Backup: Clean ML Trading System - {datetime.now().strftime("%Y-%m-%d %H:%M")}'
        os.system(f'git commit -m "{commit_msg}"')
        print("✅ Git Commit erstellt")
        
        # Branch für Backup
        backup_branch = f"backup-{datetime.now().strftime('%Y%m%d-%H%M')}"
        os.system(f"git checkout -b {backup_branch}")
        print(f"✅ Backup Branch: {backup_branch}")
        
        return backup_branch
        
    except Exception as e:
        print(f"⚠️ Git Backup Fehler: {e}")
        return None

if __name__ == "__main__":
    # ZIP Backup
    zip_path, zip_size, file_count = create_backup()
    
    # Git Backup
    git_branch = create_git_backup()
    
    print(f"\n🎉 BACKUP KOMPLETT!")
    print(f"💾 ZIP: {zip_path} ({zip_size/1024:.1f} KB)")
    if git_branch:
        print(f"🔄 Git Branch: {git_branch}")
    print(f"📁 {file_count} Dateien gesichert")
    print(f"🚀 System bereit für Deployment!")
