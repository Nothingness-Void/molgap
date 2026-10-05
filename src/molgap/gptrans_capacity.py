"""Bounded capacity and local-bond falsifiers on the corrected GPTrans core.

No runner, dataset construction or submission lives here. Wider tensors use
the same seed-42 constructor policy, not resized pretrained weights. Same-shape
core tensors in the FFN/local arms retain the accepted G1 initialization.
"""
from pathlib import Path

import torch
from torch import nn

from .gptrans import OGBGPTransTiny
from .training_reproducibility import atomic_torch_save
from .v4_runtime import normalized_source_sha256
from .screen_policy import canonical_fingerprint

MODES = ("degree_node352_ema999", "degree_pair64_ema999",
         "degree_ffn2_ema999", "degree_bond_local_ema999")
PARAMETER_CAP = 2 * 5_246_817


def configuration(mode):
    if mode not in MODES:
        raise ValueError("Unknown capacity intervention")
    return {"node_channels": 352 if mode == MODES[0] else 256,
            "pair_channels": 64 if mode == MODES[1] else 32,
            "layers": 12, "heads": 8, "ffn_ratio": 2 if mode == MODES[2] else 1,
            "local_bond_channels": 64 if mode == MODES[3] else 0,
            "degree_scale": .0897, "dropout": .1, "drop_path": .1,
            "initialization": "seed42-constructor;shared-shape-core-preserved-v1"}


def architecture_identity(mode):
    return canonical_fingerprint({"core": normalized_source_sha256(Path(__file__).with_name("gptrans.py")),
        "variant": mode, "config": configuration(mode),
        "capacity_module": normalized_source_sha256(Path(__file__))})


class BondLocalBlock(nn.Module):
    """Unnormalized true-bond sum before GPA, with an identity initial bypass.

Only accepted real directed bonds carry messages. There is no all-pairs edge
MLP and no new coordinate, path or feature cache.
"""
    def __init__(self, block, channels=256, pair_channels=32):
        super().__init__()
        self.core = block
        self.norm = nn.LayerNorm(channels)
        self.message = nn.Sequential(nn.Linear(2 * channels + pair_channels, 64), nn.GELU())
        self.output = nn.Linear(64, channels)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, node, pair, padding, edges):
        graph, source, target = edges
        clean = self.norm(node)
        message = self.output(self.message(torch.cat((clean[graph, source], clean[graph, target],
                            pair[graph, :, source, target]), -1)))
        # Strict deterministic mode selects Torch's deterministic CUDA reduction;
        # the owning preflight verifies two optimizer-inclusive repeats before use.
        flat_target = graph * node.shape[1] + target
        update = node.new_zeros(node.shape[0] * node.shape[1], node.shape[-1])
        update = torch.index_add(update, 0, flat_target, message)
        update = update.view_as(node)
        return self.core(node + update, pair, padding)


class BondLocalGPTrans(OGBGPTransTiny):
    def forward(self, x, edge_index, edge_attr, batch, random_walk_pe=None):
        node, pair, padding = self._dense_inputs(x, edge_index, edge_attr, batch)
        graph, source, target = self._local_edges(edge_index, batch, len(x))
        edges = (graph, source + 1, target + 1)
        for block in self.blocks:
            node, pair = block(node, pair, padding, edges)
        return self.readout(torch.cat((node[:, 0], pair[:, :, 0, 0]), -1))


def construct(mode, base_state=None):
    """Construct only; no data, forward pass or optimization is executed."""
    config = configuration(mode)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = OGBGPTransTiny(node_channels=config["node_channels"], pair_channels=config["pair_channels"])
        if mode in MODES[:2]:
            with torch.no_grad():
                model.in_degree_encoder.weight.mul_(.0897)
                model.out_degree_encoder.weight.mul_(.0897)
        elif base_state is not None:
            model.load_state_dict(base_state, strict=True)
        else:
            raise ValueError("Same-shape arms require the accepted G1 initial state")
        if mode == MODES[2]:
            for block in model.blocks:
                old = block.ffn
                first, last = nn.Linear(256, 512), nn.Linear(512, 256)
                with torch.no_grad():
                    first.weight[:256].copy_(old[0].weight)
                    first.bias[:256].copy_(old[0].bias)
                    last.weight[:, :256].copy_(old[3].weight)
                    last.weight[:, 256:].zero_()
                    last.bias.copy_(old[3].bias)
                block.ffn = nn.Sequential(first, nn.GELU(), nn.Dropout(.1), last, nn.Dropout(.1))
        if mode == MODES[3]:
            model.__class__ = BondLocalGPTrans
            model.blocks = nn.ModuleList(BondLocalBlock(block) for block in model.blocks)
    count = sum(p.numel() for p in model.parameters())
    if not 0 < count <= PARAMETER_CAP:
        raise ValueError(f"Parameter allowance exceeded: {count}")
    model._molgap_author_variant = mode
    return model


def freeze_initial(mode, base_path, output):
    from .pcqm_gptrans_v4 import _state_sha256
    from .gptrans_author_variants import DEGREE_INITIAL_SHA256
    base = torch.load(base_path, map_location="cpu", weights_only=True)
    # Verify actual tensors, not just a self-asserted payload digest.
    reference = OGBGPTransTiny()
    reference.load_state_dict(base["model_state"], strict=True)
    if _state_sha256(reference) != DEGREE_INITIAL_SHA256:
        raise ValueError("Accepted G1 initialization changed")
    model = construct(mode, base["model_state"])
    digest = _state_sha256(model)
    atomic_torch_save(Path(output), {"format": "molgap-gptrans-capacity-initial-v1", "variant": mode,
        "architecture_identity": architecture_identity(mode), "parameters": sum(p.numel() for p in model.parameters()),
        "state_sha256": digest, "model_state": model.state_dict()})
    return {"parameters": sum(p.numel() for p in model.parameters()), "state_sha256": digest,
            "architecture_identity": architecture_identity(mode)}


def load_initial(mode, path):
    from .pcqm_gptrans_v4 import _state_sha256
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("format") != "molgap-gptrans-capacity-initial-v1" or payload.get("variant") != mode
            or payload.get("architecture_identity") != architecture_identity(mode)):
        raise ValueError("Frozen capacity initialization identity changed")
    # Construct shared-shape cores before replacing modules, then load all frozen tensors.
    if mode in MODES[2:]:
        with torch.random.fork_rng(devices=[]):
            base = OGBGPTransTiny().state_dict()
    else:
        base = None
    model = construct(mode, base)
    model.load_state_dict(payload["model_state"], strict=True)
    count = sum(p.numel() for p in model.parameters())
    if count != payload["parameters"] or _state_sha256(model) != payload["state_sha256"]:
        raise ValueError("Frozen capacity tensor identity changed")
    model._capacity_initial_sha256 = payload["state_sha256"]
    model._capacity_parameters = count
    return model
