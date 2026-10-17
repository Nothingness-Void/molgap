"""Historical import alias only; no parallel implementation or operating policy."""
import sys
from . import training_runtime as _owner

sys.modules[__name__] = _owner
