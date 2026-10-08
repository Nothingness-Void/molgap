"""Pure-2D triplet communication addons over the frozen local-bond parent.

TGT equations 1--6 motivate the two directions; no geometry teacher, distance
prediction stage or extra dropout is introduced. Native GPTrans owns training.
"""
from pathlib import Path
import math

import torch
from torch import nn

from .gptrans_capacity import construct as construct_local
from .gptrans_local_relation import (
    BASE_MODE, BASE_PARAMETERS, INSERTIONS, PARENT_FILE_SHA256,
    PARENT_TENSOR_SHA256, state_digest, valid_pairs,
)
from .screen_policy import canonical_fingerprint
from .training_reproducibility import atomic_torch_save, sha256_file
from .v4_runtime import normalized_source_sha256

MODES = ("degree_local_triplet_aggregate_ema999", "degree_local_triplet_attention_ema999")
HEADS, HEAD_DIM = 2, 8
FORMAT = "molgap-gptrans-triplet-communication-initial-v1"
PARAMETERS = {MODES[0]: BASE_PARAMETERS + 4 * 2440,
              MODES[1]: BASE_PARAMETERS + 4 * 4552}
PARAMETER_CAP = max(PARAMETERS.values())


def configuration(mode):
    if mode not in MODES:
        raise ValueError("Unknown triplet communication intervention")
    from .gptrans_capacity import configuration as parent_configuration
    return {**parent_configuration(BASE_MODE), "parent_variant": BASE_MODE,
        "relation_mechanism": "tgt-style-triplet-aggregation" if mode == MODES[0]
        else "tgt-style-triplet-attention", "changed_layers": list(INSERTIONS),
        "heads": HEADS, "head_channels": HEAD_DIM, "directions": ["inward", "outward"],
        "pair_scope": "all-valid-pairs-including-virtual", "third_pair_gate": True,
        "new_return_initialization": "zero", "new_forward_rng": False,
        "readout": "unchanged-virtual-node-plus-virtual-pair",
        "initialization": "retained-local-parent-exact;seed42-new-tensors-only-v1"}


def architecture_identity(mode):
    return canonical_fingerprint({"variant": mode, "config": configuration(mode),
        "sources": {name: normalized_source_sha256(Path(__file__).with_name(name))
                    for name in ("gptrans.py", "gptrans_capacity.py", "gptrans_local_relation.py", Path(__file__).name)}})


def projection_shapes(mode):
    configuration(mode)
    shapes = {name: (HEADS * HEAD_DIM, 32) for name in ("in_value", "out_value")}
    shapes.update({name: (HEADS, 32) for name in ("in_bias", "in_gate", "out_bias", "out_gate")})
    if mode == MODES[1]:
        shapes.update({name: (HEADS * HEAD_DIM, 32) for name in ("in_query", "in_key", "out_query", "out_key")})
    shapes["output"] = (32, 2 * HEADS * HEAD_DIM)
    return shapes


def masked_weights(logits, gate, mask, *, dim):
    # Finite masking also gives zero, rather than NaN, for an all-padding query.
    weights = logits.masked_fill(~mask, torch.finfo(logits.dtype).min).softmax(dim=dim)
    return weights * torch.sigmoid(gate) * mask


def aggregate_direction(value, bias, gate, valid, *, outward=False):
    """Quadratic-size intermediates; no B*N*N*N attention tensor."""
    weights = masked_weights(bias, gate, valid[..., None], dim=1 if outward else 2)
    return (torch.einsum("bkih,bkjhd->bijhd", weights, value) if outward else
            torch.einsum("bikh,bjkhd->bijhd", weights, value))


def attention_direction(value, query, key, bias, gate, valid, *, outward=False):
    """Query-dependent TGT triplet attention; includes the third pair bias/gate."""
    if outward:
        scores = torch.einsum("bijhd,bkjhd->bijkh", query, key) / math.sqrt(HEAD_DIM)
        bias, gate = bias.transpose(1, 2), gate.transpose(1, 2)
    else:
        scores = torch.einsum("bijhd,bjkhd->bijkh", query, key) / math.sqrt(HEAD_DIM)
    triple = valid[:, :, :, None] & valid[:, :, None, :]
    # valid is a Cartesian node mask; reversed third-pair support is equal.
    triple = triple & valid[:, None, :, :]
    weights = masked_weights(scores + bias[:, :, None, :, :],
                             gate[:, :, None, :, :], triple[..., None], dim=3)
    return (torch.einsum("bijkh,bkjhd->bijhd", weights, value) if outward else
            torch.einsum("bijkh,bjkhd->bijhd", weights, value))


class TripletLocalBlock(nn.Module):
    def __init__(self, core, mode):
        super().__init__()
        self.core, self.mode = core, mode
        self.norm = nn.LayerNorm(32)
        for name, (out_features, in_features) in projection_shapes(mode).items():
            setattr(self, name, nn.Linear(in_features, out_features))
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, node, pair, padding, edges):
        valid = valid_pairs(padding)
        values = self.norm(pair.permute(0, 2, 3, 1)).masked_fill(~valid[..., None], 0)
        returns = []
        for direction in ("in", "out"):
            value = getattr(self, direction + "_value")(values).unflatten(-1, (HEADS, HEAD_DIM))
            value = value.masked_fill(~valid[..., None, None], 0)
            bias, gate = getattr(self, direction + "_bias")(values), getattr(self, direction + "_gate")(values)
            if self.mode == MODES[0]:
                returned = aggregate_direction(value, bias, gate, valid, outward=direction == "out")
            else:
                query = getattr(self, direction + "_query")(values).unflatten(-1, (HEADS, HEAD_DIM))
                key = getattr(self, direction + "_key")(values).unflatten(-1, (HEADS, HEAD_DIM))
                returned = attention_direction(value, query, key, bias, gate, valid, outward=direction == "out")
            returns.append(returned.flatten(-2))
        delta = self.output(torch.cat(returns, -1)).masked_fill(~valid[..., None], 0).permute(0, 3, 1, 2)
        if getattr(self, "_capture_relation", False):
            self._relation_rms = (delta.detach().square().sum() / (valid.sum() * 32).clamp_min(1)).sqrt()
            self._capture_relation = False
        # Include virtual [0,0], ensuring a route to the unchanged final readout.
        return self.core(node, pair + delta, padding, edges)


def construct(mode, base_state):
    configuration(mode)
    model = construct_local(BASE_MODE, base_state)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        for depth in INSERTIONS:
            model.blocks[depth - 1] = TripletLocalBlock(model.blocks[depth - 1], mode)
    model._molgap_author_variant = mode
    return model


def freeze_initial(mode, base_path, output):
    """CPU tensor extension only, never model construction/execution."""
    from .constants import REPO_ROOT
    configuration(mode)
    parent_path = REPO_ROOT / "platforms/_records/kaggle/initializations/degree_bond_local_ema999_initial.pt"
    if sha256_file(parent_path) != PARENT_FILE_SHA256:
        raise ValueError("Accepted local-parent initial file changed")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)["model_state"]
    if state_digest(parent) != PARENT_TENSOR_SHA256:
        raise ValueError("Accepted parent tensors changed")
    state = {}
    for name, value in parent.items():
        fields = name.split(".")
        if name.startswith("blocks.") and int(fields[1]) + 1 in INSERTIONS:
            fields.insert(2, "core")
        state[".".join(fields)] = value.clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        for depth in INSERTIONS:
            prefix = f"blocks.{depth - 1}."
            state[prefix + "norm.weight"], state[prefix + "norm.bias"] = torch.ones(32), torch.zeros(32)
            for name, (out_features, in_features) in projection_shapes(mode).items():
                weight, bias = torch.empty(out_features, in_features), torch.empty(out_features)
                nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
                nn.init.uniform_(bias, -1 / math.sqrt(in_features), 1 / math.sqrt(in_features))
                if name == "output":
                    weight.zero_()
                    bias.zero_()
                state[prefix + name + ".weight"], state[prefix + name + ".bias"] = weight, bias
    if sum(v.numel() for v in state.values()) != PARAMETERS[mode]:
        raise ValueError("Triplet tensor inventory changed")
    facts = {"parameters": PARAMETERS[mode], "state_sha256": state_digest(state), "architecture_identity": architecture_identity(mode)}
    atomic_torch_save(Path(output), {"format": FORMAT, "variant": mode, **facts,
        "parent_tensor_sha256": PARENT_TENSOR_SHA256, "model_state": state})
    return facts


def read_initial_facts(mode, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model_state"]
    if (payload.get("format") != FORMAT or payload.get("variant") != mode
            or payload.get("architecture_identity") != architecture_identity(mode)
            or payload.get("parent_tensor_sha256") != PARENT_TENSOR_SHA256
            or payload.get("parameters") != PARAMETERS[mode]
            or state_digest(state) != payload.get("state_sha256")
            or sum(v.numel() for v in state.values()) != PARAMETERS[mode]):
        raise ValueError("Triplet initialization identity differs")
    recovered = {}
    for name, value in state.items():
        fields = name.split(".")
        if name.startswith("blocks.") and int(fields[1]) + 1 in INSERTIONS:
            if fields[2] != "core":
                continue
            del fields[2]
        recovered[".".join(fields)] = value
    if state_digest(recovered) != PARENT_TENSOR_SHA256:
        raise ValueError("Triplet arm altered accepted parent tensors")
    for depth in INSERTIONS:
        for suffix, shape in (("weight", (32, 32)), ("bias", (32,))):
            value = state[f"blocks.{depth - 1}.output.{suffix}"]
            if tuple(value.shape) != shape or torch.count_nonzero(value).item() != 0:
                raise ValueError("Triplet return must start at identity")
    if not all(torch.isfinite(v).all().item() for v in state.values()):
        raise ValueError("Nonfinite initial tensors")
    return {key: payload[key] for key in ("parameters", "state_sha256", "architecture_identity")}


def load_initial(mode, path):
    from .gptrans import OGBGPTransTiny
    facts = read_initial_facts(mode, path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = construct(mode, OGBGPTransTiny().state_dict())
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state"], strict=True)
    if sum(p.numel() for p in model.parameters()) != PARAMETERS[mode]:
        raise ValueError("Remote triplet parameter inventory changed")
    model._capacity_initial_sha256, model._capacity_parameters = facts["state_sha256"], facts["parameters"]
    return model


def relation_blocks(model):
    return [(d, b) for d, b in enumerate(model.blocks, 1) if isinstance(b, TripletLocalBlock)]


def begin_capture(model):
    for _, block in relation_blocks(model):
        block._capture_relation = True


def diagnostics(model):
    rows = relation_blocks(model)
    return {"changed_layers": [d for d, _ in rows], "mode": model._molgap_author_variant,
        "return_rms_first_scheduled_batch": [float(b._relation_rms.cpu()) for _, b in rows],
        "return_weight_norm_epoch_end": [float(b.output.weight.detach().norm().cpu()) for _, b in rows],
        "return_gradient_norm_last_batch": [float(b.output.weight.grad.detach().norm().cpu()) if b.output.weight.grad is not None else None for _, b in rows]}


def verify_connected_preflight(model):
    norms = [float(b.output.weight.grad.detach().norm().cpu()) if b.output.weight.grad is not None else 0. for _, b in relation_blocks(model)]
    if len(norms) != len(INSERTIONS) or not all(math.isfinite(v) and v > 0 for v in norms):
        raise ValueError("Triplet return disconnected/nonfinite after preflight warmup")
    return {"all_return_gradients_connected": True, "gradient_norms": norms}
