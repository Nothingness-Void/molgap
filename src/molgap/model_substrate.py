"""Compatibility module alias for the retained model substrate.

Both import paths share globals, including the private slot helpers.
"""
import sys

from .models import substrate as _implementation

sys.modules[__name__] = _implementation
