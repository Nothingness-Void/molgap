"""Optional CPU snapshot copying; serialization and atomic IO stay with callers."""
from __future__ import annotations

import copy


_SCALARS = frozenset((type(None), bool, int, float, complex, str, bytes))


def checkpoint_to_cpu(value):
    """Match colab_k1_screen._cpu without recursive imports/scalar deepcopies.

    Tensor occurrences are detached CPU clones, never shared with the source or
    each other. Dict keys remain untouched and output dicts are plain, as in the
    original. List/tuple constructors and deepcopy fallback retain its semantics;
    this is not a memoized copier and does not add cyclic-container support.
    Exact built-in scalar roots need no torch import; subclasses use fallback.
    """
    if type(value) in _SCALARS:
        return value

    import torch

    def snapshot(item):
        if type(item) in _SCALARS:
            return item
        if torch.is_tensor(item):
            return item.detach().cpu().clone()
        if isinstance(item, dict):
            return {key: snapshot(child) for key, child in item.items()}
        if isinstance(item, (tuple, list)):
            return type(item)(snapshot(child) for child in item)
        return copy.deepcopy(item)

    return snapshot(value)
