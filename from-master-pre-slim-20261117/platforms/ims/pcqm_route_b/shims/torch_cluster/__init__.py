"""IMS compatibility shim for the unavailable compiled torch-cluster wheel."""

from molgap.portable_radius import BACKEND, radius_graph

__version__ = "molgap-portable-1"

__all__ = ["BACKEND", "radius_graph"]

