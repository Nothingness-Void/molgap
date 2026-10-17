"""Historical import alias only; no parallel trainer or legacy execution policy."""
import sys
from . import gptrans_screen_training as _owner

sys.modules[__name__] = _owner
