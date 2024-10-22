"""
PumaGuard
"""

import importlib.metadata

try:
    __VERSION__ = importlib.metadata.version('pumaguard')
except importlib.metadata.PackageNotFoundError:
    __VERSION__ = 'undefined'
