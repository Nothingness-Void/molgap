"""Compatibility alias for the fixed-record implementation in ``molgap.data``."""
import sys as _sys

from .data import fixed_records as _implementation

_sys.modules[__name__] = _implementation
