"""Core trading logic modules package.
Currently exposes TechnicalAnalysis extracted from monolithic app.py.
Further components (advanced indicators, pattern detection, AI, risk) will be migrated here stepwise.
"""

from .technical_analysis import TechnicalAnalysis  # re-export for convenience
