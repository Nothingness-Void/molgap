"""Alias for GPTrans's byte-pinned historical implementation.

The frozen architecture contract hashes ``molgap/gptrans.py`` itself. Keep that
single implementation at its original path; expose the organized namespace
without rewriting historical source identity or duplicating the model.
"""
import sys

from .. import gptrans as _implementation

sys.modules[__name__] = _implementation
