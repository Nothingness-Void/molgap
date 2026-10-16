"""Model implementations, loaded only when their submodule is requested.

Import a selected family explicitly; importing this namespace loads no model
or training runtime.
"""

__all__ = ["gptrans", "substrate"]
