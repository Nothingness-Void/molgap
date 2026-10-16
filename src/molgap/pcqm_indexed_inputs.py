"""Compatibility alias for the indexed-input implementation in ``molgap.data``."""
import sys as _sys

from .data import indexed_inputs as _implementation

# In particular, legacy patches to _pin must change IndexedRoleDataset globals.
_sys.modules[__name__] = _implementation
