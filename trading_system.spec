# -*- mode: python ; coding: utf-8 -*-

import sys
import os

# Pfad zur Anwendung
app_path = os.path.abspath('.')

block_cipher = None

a = Analysis(
    ['trading_system.py'],
    pathex=[app_path],
    binaries=[],
    datas=[
        # Statische Dateien hinzufügen falls vorhanden
        ('favicon.ico', '.'),
    ],
    hiddenimports=[
        # Wichtige Module explizit hinzufügen
        'flask',
        'jax',
        'flax',
        'optax',
        'numpy',
        'requests',
        'json',
        'datetime',
        'time',
        'threading',
        'webbrowser',
        # JAX dependencies
        'jax.numpy',
        'jax._src',
        'flax.linen',
        'flax.training',
        # Andere wichtige Module
        'werkzeug',
        'werkzeug.serving',
        'werkzeug.utils',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Unnötige Module ausschließen für kleinere EXE
        'matplotlib',
        'tkinter',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'PySide6',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='TradingSystem_V4',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Komprimierung für kleinere Datei
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Console anzeigen für Logs
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon='favicon.ico',  # Icon entfernt wegen Fehler
    version_info={
        'version': (4, 0, 0, 0),
        'description': 'Ultimate Trading System V4 - JAX Neural Networks',
        'company': 'Trading Analysis System',
        'product': 'Ultimate Trading System',
        'copyright': '2025 Trading System',
    }
)
