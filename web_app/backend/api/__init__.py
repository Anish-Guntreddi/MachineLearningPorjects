"""
API Package
Contains all API route modules
"""

from .inference import router as inference_router
from .metrics import router as metrics_router

__all__ = ['inference_router', 'metrics_router']
