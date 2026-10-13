"""Frozen-product recovery and bond-supported triplet aggregation, no trainer."""
from pathlib import Path

import torch

from . import gptrans_triplet_communication as parent
from . import gptrans_triplet_followup as product
from .gptrans_local_relation import state_digest, valid_pairs
from .screen_policy import canonical_fingerprint
from .training_reproducibility import atomic_torch_save
from .v4_runtime import normalized_source_sha256

MODES = ("degree_local_triplet_product_v2_ema999", "degree_local_triplet_bond_route_ema999")
PARAMETERS = {MODES[0]: 5_885_185, MODES[1]: 5_880_961}
PARAMETER_CAP = max(PARAMETERS.values())
FORMAT = "molgap-triplet-routing-initial-v1"
PARENT_TENSOR_SHA256 = product.PARENT_TENSOR_SHA256
STATE_PINS = {
    MODES[0]: "64e5b2f5d733f7f91230bce4e6fea4d143a6b516e85b7d4dad56f76973d782ae",
    MODES[1]: PARENT_TENSOR_SHA256,
}
INVENTORY_PINS = {
    MODES[0]: "7a93c782dc4027a6ac497aa9c2066b8c74b4c17636e7f421b90952e54d0e3055",
    MODES[1]: "08e140d88de74bea9a2f19ed988dc1ba1bed4fcca7f7564fd23db81eea24c6c9",
}


def configuration(mode):
    if mode not in MODES:
        raise ValueError("Unknown triplet routing intervention")
    return {**parent.configuration(parent.MODES[0]),
        "parent_variant": parent.MODES[0],
        "relation_mechanism": "triplet-channelwise-product" if mode == MODES[0]
        else "bond-supported-triplet-aggregation",
        "value_pair_support": "all-valid" if mode == MODES[0] else "true-directed-bonds+self+virtual",
        "routing_normalization": "parent" if mode == MODES[0] else "conditional-over-supported-values;denominator-clamp1e-12",
        "initialization": "exact-pinned-parent-and-delta-tensors;no-cross-runtime-random-regeneration",
        "new_forward_rng": False, "whole_branch_drop": False}


def architecture_identity(mode):
    return canonical_fingerprint({"variant": mode, "config": configuration(mode),
        "parent_identity": parent.architecture_identity(parent.MODES[0]),
        "product_source": normalized_source_sha256(Path(product.__file__)),
        "source": normalized_source_sha256(Path(__file__))})


def bond_support(valid, edges):
    """Indices already include the virtual-node offset in the native ABI."""
    graph, source, target = edges
    support = torch.eye(valid.shape[-1], device=valid.device, dtype=torch.bool)[None].expand_as(valid).clone()
    support[graph, source, target] = True
    support[:, 0, :] = True
    support[:, :, 0] = True
    return support & valid


def bond_direction(value, bias, gate, valid, support, *, outward=False):
    # Factorize the junction-dependent mask; do not materialize B*N*N*N logits.
    dim = 1 if outward else 2
    probability = bias.masked_fill(~valid[..., None], torch.finfo(bias.dtype).min).softmax(dim=dim)
    probability = probability * valid[..., None]
    weighted = probability * torch.sigmoid(gate)
    supported_value = value * support[..., None, None]
    if outward:
        numerator = torch.einsum("bkih,bkjhd->bijhd", weighted, supported_value)
        denominator = torch.einsum("bkih,bkj->bijh", probability, support.to(value.dtype))
    else:
        numerator = torch.einsum("bikh,bjkhd->bijhd", weighted, supported_value)
        denominator = torch.einsum("bikh,bjk->bijh", probability, support.to(value.dtype))
    return (numerator / denominator.clamp_min(1e-12)[..., None]).masked_fill(~valid[..., None, None], 0)


class BondTripletBlock(parent.TripletLocalBlock):
    def __init__(self, core):
        super().__init__(core, parent.MODES[0])

    def forward(self, node, pair, padding, edges):
        valid = valid_pairs(padding)
        support = bond_support(valid, edges)
        values = self.norm(pair.permute(0, 2, 3, 1)).masked_fill(~valid[..., None], 0)
        returns = []
        for direction in ("in", "out"):
            value = getattr(self, direction + "_value")(values).unflatten(-1, (2, 8))
            bias = getattr(self, direction + "_bias")(values)
            gate = getattr(self, direction + "_gate")(values)
            returns.append(bond_direction(value, bias, gate, valid, support,
                                          outward=direction == "out").flatten(-2))
        delta = self.output(torch.cat(returns, -1)).masked_fill(~valid[..., None], 0).permute(0, 3, 1, 2)
        if getattr(self, "_capture_relation", False):
            self._relation_rms = (delta.detach().square().sum() / (valid.sum() * 32).clamp_min(1)).sqrt()
            self._capture_relation = False
        return self.core(node, pair + delta, padding, edges)


def construct(mode, base_state):
    configuration(mode)
    if mode == MODES[0]:
        model = product.construct(product.MODES[1], base_state)
    else:
        model = parent.construct(parent.MODES[0], base_state)
        with torch.random.fork_rng(devices=[]):
            for depth in parent.INSERTIONS:
                old = model.blocks[depth - 1]
                block = BondTripletBlock(old.core)
                block.load_state_dict(old.state_dict(), strict=True)
                model.blocks[depth - 1] = block
    model._molgap_author_variant = mode
    return model


def _inventory(state):
    return canonical_fingerprint({k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in state.items()})


def read_initial_facts(mode, path):
    """Qualify retained bytes, layout and parent, not a new runtime's RNG output."""
    configuration(mode)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model_state"]
    recovered = {k: v for k, v in state.items() if "_operand." not in k}
    checks = {
        "format": payload.get("format") == FORMAT,
        "variant": payload.get("variant") == mode,
        "architecture": payload.get("architecture_identity") == architecture_identity(mode),
        "full_tensor_pin": state_digest(state) == payload.get("state_sha256") == STATE_PINS[mode],
        "layout_pin": _inventory(state) == INVENTORY_PINS[mode],
        "parent_tensor_pin": state_digest(recovered) == payload.get("parent_tensor_sha256") == PARENT_TENSOR_SHA256,
        "parameter_inventory": sum(v.numel() for v in state.values()) == payload.get("parameters") == PARAMETERS[mode],
        "finite": all(torch.isfinite(v).all().item() for v in state.values()),
    }
    if not all(checks.values()):
        raise ValueError("Frozen triplet initialization failed: " + ",".join(k for k, ok in checks.items() if not ok))
    return {k: payload[k] for k in ("parameters", "state_sha256", "architecture_identity")} | {"qualification_checks": checks}


def freeze_initial(mode, base_path, output):
    """Re-envelope exact retained initial tensors; never execute a model."""
    from .constants import REPO_ROOT
    configuration(mode)
    old_mode = product.MODES[1] if mode == MODES[0] else parent.MODES[0]
    path = REPO_ROOT / "platforms/_records/kaggle/initializations" / (old_mode + "_initial.pt")
    state = torch.load(path, map_location="cpu", weights_only=True)["model_state"]
    if state_digest(state) != STATE_PINS[mode] or _inventory(state) != INVENTORY_PINS[mode]:
        raise ValueError("Retained initialization is not the pinned original")
    atomic_torch_save(Path(output), {"format": FORMAT, "variant": mode,
        "architecture_identity": architecture_identity(mode), "parameters": PARAMETERS[mode],
        "state_sha256": STATE_PINS[mode], "parent_tensor_sha256": PARENT_TENSOR_SHA256,
        "model_state": state})
    return read_initial_facts(mode, output)


def load_initial(mode, path):
    from .gptrans import OGBGPTransTiny
    facts = read_initial_facts(mode, path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = construct(mode, OGBGPTransTiny().state_dict())
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state"], strict=True)
    if sum(p.numel() for p in model.parameters()) != PARAMETERS[mode]:
        raise ValueError("Triplet routing parameter inventory differs")
    model._capacity_initial_sha256, model._capacity_parameters = facts["state_sha256"], facts["parameters"]
    return model


begin_capture = parent.begin_capture
diagnostics = parent.diagnostics
verify_connected_preflight = parent.verify_connected_preflight
