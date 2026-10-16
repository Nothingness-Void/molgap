"""Compatibility alias for the packed-graph implementation in ``molgap.data``."""
import sys as _sys

from .data import packed_graph as _implementation

# Share globals as well as objects so legacy monkeypatches reach the implementation.
_sys.modules[__name__] = _implementation
