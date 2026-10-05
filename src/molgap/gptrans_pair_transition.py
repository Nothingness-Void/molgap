"""Independent nonlinear real-pair transitions on the frozen corrected G1 core.

The original GPA, node FFN and readout remain intact. No graph construction,
training loop, platform operation or additional stochastic layer lives here.
"""
from pathlib import Path

import torch
from torch import nn

from .gptrans import OGBGPTransTiny
from .screen_policy import canonical_fingerprint
from .training_reproducibility import atomic_torch_save
from .v4_runtime import normalized_source_sha256

MODE = "degree_pair_transition_ema999"
PARAMETER_CAP = 5_263_841
INSERTIONS = (3, 6, 9, 12)


def configuration(mode):
    if mode != MODE:
        raise ValueError("Unknown pair-transition intervention")
    return {"node_channels": 256, "pair_channels": 32, "layers": 12, "heads": 8,
            "transition_before_blocks": list(INSERTIONS), "hidden_channels": 64,
            "pair_scope": "valid-real-atom-ordered-pairs-including-diagonal",
            "branch_norm": "LayerNorm32", "activation": "GELU", "return_init": "zero",
            "degree_scale": .0897, "dropout": .1, "drop_path": .1,
            "initialization": "seed42-addon;accepted-G1-core-preserved-v1"}


def architecture_identity(mode):
    return canonical_fingerprint({"core": normalized_source_sha256(Path(__file__).with_name("gptrans.py")),
        "variant": mode, "config": configuration(mode),
        "transition_module": normalized_source_sha256(Path(__file__))})


class PairTransitionBlock(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.norm = nn.LayerNorm(32)
        self.hidden = nn.Linear(32, 64)
        self.output = nn.Linear(64, 32)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def transition(self, pair, padding):
        # Exclude both padded queries/keys and the virtual-node row/column.
        valid_node = ~padding[:, 0, 0, :]
        real_node = valid_node & (torch.arange(pair.shape[-1], device=pair.device) > 0)
        valid = real_node[:, :, None] & real_node[:, None, :]
        values = pair.permute(0, 2, 3, 1).masked_fill(~valid[..., None], 0)
        delta = self.output(torch.nn.functional.gelu(self.hidden(self.norm(values))))
        delta = delta.masked_fill(~valid[..., None], 0).permute(0, 3, 1, 2)
        updated = pair + delta
        if getattr(self, "_capture_transition_diagnostics", False):
            count = (valid.sum() * pair.shape[1]).clamp_min(1)
            rms = lambda x: ((x.detach().square() * valid[:, None]).sum() / count).sqrt()
            self._transition_diagnostics = torch.stack((rms(pair), rms(delta), rms(updated)))
            self._capture_transition_diagnostics = False
        return updated

    def forward(self, node, pair, padding):
        # A pre-block12 transition can affect the graph token; a post-final
        # real-pair update would be disconnected from the frozen readout.
        return self.core(node, self.transition(pair, padding), padding)


def construct(mode, base_state):
    configuration(mode)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = OGBGPTransTiny()
        model.load_state_dict(base_state, strict=True)
        for index in INSERTIONS:
            model.blocks[index - 1] = PairTransitionBlock(model.blocks[index - 1])
    if sum(p.numel() for p in model.parameters()) != PARAMETER_CAP:
        raise ValueError("Frozen pair-transition parameter count changed")
    model._molgap_author_variant = mode
    return model


def freeze_initial(mode, base_path, output):
    from .pcqm_gptrans_v4 import _state_sha256
    from .gptrans_author_variants import DEGREE_INITIAL_SHA256
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    reference = OGBGPTransTiny()
    reference.load_state_dict(base["model_state"], strict=True)
    if _state_sha256(reference) != DEGREE_INITIAL_SHA256:
        raise ValueError("Accepted G1 initialization changed")
    model = construct(mode, base["model_state"])
    facts = {"parameters": PARAMETER_CAP, "state_sha256": _state_sha256(model),
             "architecture_identity": architecture_identity(mode)}
    atomic_torch_save(Path(output), {"format": "molgap-gptrans-pair-transition-initial-v1",
        "variant": mode, **facts, "model_state": model.state_dict()})
    return facts


def load_initial(mode, path):
    from .pcqm_gptrans_v4 import _state_sha256
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("format") != "molgap-gptrans-pair-transition-initial-v1"
            or payload.get("variant") != mode or payload.get("architecture_identity") != architecture_identity(mode)):
        raise ValueError("Frozen pair-transition initialization identity changed")
    with torch.random.fork_rng(devices=[]):
        base = OGBGPTransTiny().state_dict()
    model = construct(mode, base)
    model.load_state_dict(payload["model_state"], strict=True)
    if payload["parameters"] != PARAMETER_CAP or _state_sha256(model) != payload["state_sha256"]:
        raise ValueError("Frozen pair-transition tensor identity changed")
    # Retain the native trainer's full-initialization verification ABI.
    model._capacity_initial_sha256 = payload["state_sha256"]
    model._capacity_parameters = PARAMETER_CAP
    return model
