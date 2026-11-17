"""Data readers, loaded only when the selected reader or export is requested."""

_LAZY_EXPORTS = {
    "PackedGraphDataset": ("packed_graph", "PackedGraphDataset"),
    "FIXED_KAGGLE_MANIFESTS": ("fixed_records", "FIXED_KAGGLE_MANIFESTS"),
    "graph_records": ("fixed_records", "graph_records"),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name):
    if name in _LAZY_EXPORTS:
        from importlib import import_module

        module, attribute = _LAZY_EXPORTS[name]
        return getattr(import_module(f".{module}", __name__), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
