"""One parameter-free, molecule-wise deep local-residual amplitude falsifier.

The accepted local-bond implementation and its frozen tensors are reused.
This is not DeepNorm: neither the propagation core nor initialization changes.
"""
from pathlib import Path

import torch

from .gptrans_capacity import BondLocalBlock, construct as construct_local
from .screen_policy import canonical_fingerprint
from .training_reproducibility import atomic_torch_save
from .v4_runtime import normalized_source_sha256

MODE = "degree_bond_local_cap_ema999"
BASE_MODE = "degree_bond_local_ema999"
PARAMETER_CAP = 5_871_201
INITIAL_TENSOR_SHA256 = "a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b"
CAP = .25
INSERTIONS = tuple(range(4, 13))


def configuration(mode):
    if mode != MODE:
        raise ValueError("Unknown local-control intervention")
    from .gptrans_capacity import configuration as local_configuration
    return {**local_configuration(BASE_MODE), "controlled_layers": list(INSERTIONS),
            "maximum_local_update_input_rms_ratio": CAP,
            "reduction": "per-molecule-real-nodes-fp32-l2",
            "gradient": "differentiate-through-norm-and-clamp-no-detach",
            "denominator_floor": 1e-12, "train_inference_identical": True}


def architecture_identity(mode):
    return canonical_fingerprint({"variant": mode, "config": configuration(mode),
        "local_source": normalized_source_sha256(Path(__file__).with_name("gptrans_capacity.py")),
        "core_source": normalized_source_sha256(Path(__file__).with_name("gptrans.py")),
        "control_source": normalized_source_sha256(Path(__file__))})


def bound_update(node, update, padding):
    """Preserve direction; bound magnitude independently of padding/other graphs."""
    real = ~padding.reshape(node.shape[0], node.shape[1]).clone()
    real[:, 0] = False
    mask = real.unsqueeze(-1)
    clean_node = torch.where(mask, node.float(), 0.)
    clean_update = torch.where(mask, update.float(), 0.)
    input_norm = torch.linalg.vector_norm(clean_node, dim=(1, 2))
    update_norm = torch.linalg.vector_norm(clean_update, dim=(1, 2))
    scale = (CAP * input_norm / update_norm.clamp_min(1e-12)).clamp(max=1.)
    bounded = clean_update * scale[:, None, None]
    return bounded.to(update.dtype), (input_norm, update_norm, scale)


class ControlledBondLocalBlock(BondLocalBlock):
    def forward(self, node, pair, padding, edges):
        graph, source, target = edges
        clean = self.norm(node)
        message = self.output(self.message(torch.cat((clean[graph, source], clean[graph, target],
                            pair[graph, :, source, target]), -1)))
        flat_target = graph * node.shape[1] + target
        update = node.new_zeros(node.shape[0] * node.shape[1], node.shape[-1])
        update = torch.index_add(update, 0, flat_target, message).view_as(node)
        bounded, (input_norm, update_norm, scale) = bound_update(node, update, padding)
        if getattr(self, "_capture_control_diagnostics", False):
            raw = update_norm / input_norm.clamp_min(1e-12)
            self._control_diagnostics = torch.stack((raw, raw * scale, scale)).detach()
            self._capture_control_diagnostics = False
        return self.core(node + bounded, pair, padding)


def construct(mode, base_state):
    configuration(mode)
    model = construct_local(BASE_MODE, base_state)
    for depth, block in enumerate(model.blocks, 1):
        if depth in INSERTIONS:
            block.__class__ = ControlledBondLocalBlock
    model._molgap_author_variant = mode
    return model


def freeze_initial(mode, base_path, output):
    from .pcqm_gptrans_v4 import _state_sha256
    from .gptrans_author_variants import DEGREE_INITIAL_SHA256
    from .gptrans import OGBGPTransTiny
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    with torch.random.fork_rng(devices=[]):
        reference = OGBGPTransTiny()
    reference.load_state_dict(base["model_state"], strict=True)
    if _state_sha256(reference) != DEGREE_INITIAL_SHA256:
        raise ValueError("Accepted G1 initialization changed")
    model = construct(mode, base["model_state"])
    digest = _state_sha256(model)
    if digest != INITIAL_TENSOR_SHA256:
        raise ValueError("Control must preserve every accepted local initial tensor")
    atomic_torch_save(Path(output), {"format": "molgap-gptrans-local-control-initial-v1", "variant": mode,
        "architecture_identity": architecture_identity(mode), "parameters": PARAMETER_CAP,
        "state_sha256": digest, "model_state": model.state_dict()})
    return {"parameters": PARAMETER_CAP, "state_sha256": digest,
            "architecture_identity": architecture_identity(mode)}


def load_initial(mode, path):
    from .gptrans import OGBGPTransTiny
    from .pcqm_gptrans_v4 import _state_sha256
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("format") != "molgap-gptrans-local-control-initial-v1"
            or payload.get("variant") != mode
            or payload.get("architecture_identity") != architecture_identity(mode)
            or payload.get("state_sha256") != INITIAL_TENSOR_SHA256):
        raise ValueError("Frozen local-control initialization identity changed")
    with torch.random.fork_rng(devices=[]):
        base = OGBGPTransTiny().state_dict()
    model = construct(mode, base)
    model.load_state_dict(payload["model_state"], strict=True)
    count = sum(p.numel() for p in model.parameters())
    if count != PARAMETER_CAP or payload["parameters"] != count or _state_sha256(model) != INITIAL_TENSOR_SHA256:
        raise ValueError("Frozen local-control tensor/parameter identity changed")
    model._capacity_initial_sha256, model._capacity_parameters = INITIAL_TENSOR_SHA256, count
    return model
