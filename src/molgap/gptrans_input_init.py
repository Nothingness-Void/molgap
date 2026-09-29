"""A frozen-state, input-embedding-only initialization candidate for GPTrans-T."""
from __future__ import annotations

from collections.abc import Mapping

import torch

from .v4_runtime import state_dict_sha256


INPUT_EMBEDDING_KEYS = (
    *(f"atom_encoder.atom_embedding_list.{index}.weight" for index in range(9)),
    *(f"bond_encoder.bond_embedding_list.{index}.weight" for index in range(3)),
    "in_degree_encoder.weight",
    "out_degree_encoder.weight",
    "spatial_encoder.weight",
)
INITIALIZATION_SEED = 42
INITIALIZATION_STD = 0.02


def normal002_input_embedding_state(
    reference_state: Mapping[str, torch.Tensor], *, expected_reference_sha256: str
) -> dict[str, torch.Tensor]:
    """Return a copy with only the 15 input embedding tables reinitialized.

    The same tensor transform is used for asset-side hash preparation and for
    the remote model. Its private CPU generator does not advance training RNG.
    """
    if state_dict_sha256(reference_state) != expected_reference_sha256:
        raise RuntimeError("Frozen reference tensor state changed")
    missing = set(INPUT_EMBEDDING_KEYS) - set(reference_state)
    if missing:
        raise RuntimeError(f"Missing GPTrans input embedding tables: {sorted(missing)}")
    for name in INPUT_EMBEDDING_KEYS:
        tensor = reference_state[name]
        if tensor.device.type != "cpu" or tensor.ndim != 2 or not tensor.is_floating_point():
            raise RuntimeError(f"Invalid GPTrans input embedding tensor: {name}")

    candidate = {name: value.detach().clone() for name, value in reference_state.items()}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(INITIALIZATION_SEED)
    with torch.no_grad():
        for name in INPUT_EMBEDDING_KEYS:
            torch.nn.init.normal_(
                candidate[name], mean=0.0, std=INITIALIZATION_STD, generator=generator
            )

    changed = {
        name for name, original in reference_state.items()
        if not torch.equal(original, candidate[name])
    }
    if changed != set(INPUT_EMBEDDING_KEYS):
        raise RuntimeError(f"GPTrans input initialization changed unexpected keys: {sorted(changed)}")
    return candidate
