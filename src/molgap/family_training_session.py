"""Compatibility alias for :mod:`molgap.training.session`."""
import sys as _sys
from importlib import import_module as _import_module

# One module object preserves private globals and old-path monkeypatches.
_sys.modules[__name__] = _import_module(".training.session", __package__)
