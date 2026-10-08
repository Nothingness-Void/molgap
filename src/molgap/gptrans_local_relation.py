"""Two isolated relation-flow deltas over the accepted real-bond local parent.

No new graph cache, trainer, readout or platform operation is defined here.
Initial artifacts are extended as CPU tensors, without executing a model.
"""
from pathlib import Path
import hashlib
import math

import torch
from torch import nn

from .gptrans_capacity import BondLocalBlock, construct as construct_local
from .screen_policy import canonical_fingerprint
from .training_reproducibility import atomic_torch_save, sha256_file
from .v4_runtime import normalized_source_sha256

MODES = ("degree_local_connected_pair_ema999", "degree_local_bond_return_ema999")
BASE_MODE = "degree_bond_local_ema999"
BASE_PARAMETERS = 5_871_201
PARENT_FILE_SHA256 = "d471d9f94ff2c10436261b7f70f3e2d9111978f1ac8aadfabea2da32a5fe9a1d"
PARENT_TENSOR_SHA256 = "a65034dae2d01d82eb0074eba8f2ee697d93ca259188ae45a4213e9041d1ed1b"
INSERTIONS = (3, 6, 9, 12)
PARAMETERS = {MODES[0]: BASE_PARAMETERS + 4 * 4256,
              MODES[1]: BASE_PARAMETERS + 12 * 2080}
PARAMETER_CAP = max(PARAMETERS.values())
FORMAT = "molgap-gptrans-local-relation-initial-v1"


def configuration(mode):
    if mode not in MODES:
        raise ValueError("Unknown local relation-flow intervention")
    from .gptrans_capacity import configuration as parent_configuration
    return {**parent_configuration(BASE_MODE), "parent_variant": BASE_MODE,
        "relation_mechanism": "valid-pair-transition-including-virtual" if mode == MODES[0]
        else "true-bond-message-to-pair-to-local-node",
        "changed_layers": list(INSERTIONS) if mode == MODES[0] else list(range(1, 13)),
        "new_return_initialization": "zero", "new_forward_rng": False,
        "pair_hidden_channels": 64 if mode == MODES[0] else None,
        "pair_scope": "all-valid-pairs-including-virtual" if mode == MODES[0] else "real-directed-bonds-only",
        "readout": "unchanged-virtual-node-plus-virtual-pair",
        "initialization": "retained-local-parent-exact;seed42-new-tensors-only-v1"}


def architecture_identity(mode):
    return canonical_fingerprint({"variant": mode, "config": configuration(mode),
        "core_source": normalized_source_sha256(Path(__file__).with_name("gptrans.py")),
        "parent_source": normalized_source_sha256(Path(__file__).with_name("gptrans_capacity.py")),
        "relation_source": normalized_source_sha256(Path(__file__))})


def state_digest(state):
    # Preserve the native initial-state byte-digest ABI (names + tensor bytes).
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode() + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def valid_pairs(padding):
    valid = ~padding[:, 0, 0, :]
    return valid[:, :, None] & valid[:, None, :]


def bond_pair_update(pair, edges, returned):
    graph, source, target = edges
    size = pair.shape[-1]
    flat_index = (graph * size + source) * size + target
    delta = pair.new_zeros(pair.shape[0] * size * size, pair.shape[1])
    delta = torch.index_add(delta, 0, flat_index, returned)
    return pair + delta.view(pair.shape[0], size, size, pair.shape[1]).permute(0, 3, 1, 2)


class ConnectedPairLocalBlock(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core
        self.norm = nn.LayerNorm(32)
        self.hidden = nn.Linear(32, 64)
        self.output = nn.Linear(64, 32)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, node, pair, padding, edges):
        valid = valid_pairs(padding)
        values = pair.permute(0, 2, 3, 1).masked_fill(~valid[..., None], 0)
        returned = self.output(torch.nn.functional.gelu(self.hidden(self.norm(values))))
        returned = returned.masked_fill(~valid[..., None], 0).permute(0, 3, 1, 2)
        if getattr(self, "_capture_relation", False):
            count = (valid.sum() * 32).clamp_min(1)
            self._relation_rms = ((returned.detach().square() * valid[:, None]).sum() / count).sqrt()
            self._capture_relation = False
        # Updating the virtual pair row AND [0,0] provides a final-block return.
        return self.core(node, pair + returned, padding, edges)


class BondPairReturnBlock(BondLocalBlock):
    def forward(self, node, pair, padding, edges):
        graph, source, target = edges
        clean = self.norm(node)
        hidden = self.message(torch.cat((clean[graph, source], clean[graph, target],
                                        pair[graph, :, source, target]), -1))
        returned = self.pair_return(hidden)
        updated = bond_pair_update(pair, edges, returned)
        if getattr(self, "_capture_relation", False):
            self._relation_rms = returned.detach().square().sum().div(max(returned.numel(), 1)).sqrt()
            self._capture_relation = False
        # Reuse the parent's message with updated pairs in THIS block. Without
        # this recomputation, the final real-pair-only return is disconnected.
        return super().forward(node, updated, padding, edges)


def construct(mode, base_state):
    configuration(mode)
    model = construct_local(BASE_MODE, base_state)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        if mode == MODES[0]:
            for depth in INSERTIONS:
                model.blocks[depth - 1] = ConnectedPairLocalBlock(model.blocks[depth - 1])
        else:
            for block in model.blocks:
                block.__class__ = BondPairReturnBlock
                block.pair_return = nn.Linear(64, 32)
                nn.init.zeros_(block.pair_return.weight)
                nn.init.zeros_(block.pair_return.bias)
    model._molgap_author_variant = mode
    return model


def freeze_initial(mode, base_path, output):
    """Extend accepted tensors only; do not construct/run a local model."""
    from .constants import REPO_ROOT
    configuration(mode)
    parent_path = REPO_ROOT / "platforms/_records/kaggle/initializations/degree_bond_local_ema999_initial.pt"
    if sha256_file(parent_path) != PARENT_FILE_SHA256:
        raise ValueError("Accepted local-parent initialization file changed")
    parent = torch.load(parent_path, map_location="cpu", weights_only=True)["model_state"]
    if state_digest(parent) != PARENT_TENSOR_SHA256:
        raise ValueError("Accepted local-parent tensor identity changed")
    state = {}
    for name, value in parent.items():
        fields = name.split(".")
        if mode == MODES[0] and name.startswith("blocks.") and int(fields[1]) + 1 in INSERTIONS:
            fields.insert(2, "core")
        state[".".join(fields)] = value.clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        for depth in INSERTIONS if mode == MODES[0] else range(1, 13):
            prefix = f"blocks.{depth - 1}."
            if mode == MODES[0]:
                state[prefix + "norm.weight"] = torch.ones(32)
                state[prefix + "norm.bias"] = torch.zeros(32)
                weight, bias = torch.empty(64, 32), torch.empty(64)
                nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
                nn.init.uniform_(bias, -1 / math.sqrt(32), 1 / math.sqrt(32))
                state[prefix + "hidden.weight"], state[prefix + "hidden.bias"] = weight, bias
                name = "output"
            else:
                name = "pair_return"
            state[prefix + name + ".weight"] = torch.zeros(32, 64)
            state[prefix + name + ".bias"] = torch.zeros(32)
    if sum(v.numel() for v in state.values()) != PARAMETERS[mode]:
        raise ValueError("Frozen tensor parameter inventory changed")
    facts = {"parameters": PARAMETERS[mode], "state_sha256": state_digest(state),
             "architecture_identity": architecture_identity(mode)}
    atomic_torch_save(Path(output), {"format": FORMAT, "variant": mode, **facts,
        "parent_tensor_sha256": PARENT_TENSOR_SHA256, "model_state": state})
    return facts


def read_initial_facts(mode, path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (payload.get("format") != FORMAT or payload.get("variant") != mode
            or payload.get("architecture_identity") != architecture_identity(mode)
            or payload.get("parent_tensor_sha256") != PARENT_TENSOR_SHA256
            or payload.get("parameters") != PARAMETERS[mode]
            or state_digest(payload["model_state"]) != payload["state_sha256"]):
        raise ValueError("Frozen relation initialization identity changed")
    # Verify every parent tensor; do not trust a self-asserted parent digest.
    recovered = {}
    for name, value in payload["model_state"].items():
        fields = name.split(".")
        if mode == MODES[0] and name.startswith("blocks.") and int(fields[1]) + 1 in INSERTIONS:
            if fields[2] != "core":
                continue
            del fields[2]
        elif mode == MODES[1] and ".pair_return." in name:
            continue
        recovered[".".join(fields)] = value
    if state_digest(recovered) != PARENT_TENSOR_SHA256:
        raise ValueError("Relation arm changed an accepted parent tensor")
    state = payload["model_state"]
    if sum(value.numel() for value in state.values()) != PARAMETERS[mode]:
        raise ValueError("Relation initialization tensor inventory changed")
    depths = INSERTIONS if mode == MODES[0] else range(1, 13)
    name = "output" if mode == MODES[0] else "pair_return"
    for depth in depths:
        for suffix, shape in (("weight", (32, 64)), ("bias", (32,))):
            value = state[f"blocks.{depth - 1}.{name}.{suffix}"]
            if tuple(value.shape) != shape or torch.count_nonzero(value).item() != 0:
                raise ValueError("Relation return must start at exact zero")
    if not all(torch.isfinite(value).all().item() for value in state.values()):
        raise ValueError("Nonfinite relation initialization tensor")
    return {key: payload[key] for key in ("parameters", "state_sha256", "architecture_identity")}


def load_initial(mode, path):
    from .gptrans import OGBGPTransTiny
    facts = read_initial_facts(mode, path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = construct(mode, OGBGPTransTiny().state_dict())
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state"], strict=True)
    if sum(p.numel() for p in model.parameters()) != PARAMETERS[mode]:
        raise ValueError("Remote relation parameter count differs")
    model._capacity_initial_sha256, model._capacity_parameters = facts["state_sha256"], facts["parameters"]
    return model


def relation_blocks(model):
    return [(depth, block) for depth, block in enumerate(model.blocks, 1)
            if isinstance(block, (ConnectedPairLocalBlock, BondPairReturnBlock))]


def begin_capture(model):
    for _, block in relation_blocks(model):
        block._capture_relation = True


def diagnostics(model):
    rows = relation_blocks(model)
    outputs = [b.output if isinstance(b, ConnectedPairLocalBlock) else b.pair_return for _, b in rows]
    return {"changed_layers": [d for d, _ in rows], "mode": model._molgap_author_variant,
        "return_rms_first_scheduled_batch": [float(b._relation_rms.cpu()) for _, b in rows],
        "return_weight_norm_epoch_end": [float(p.weight.detach().norm().cpu()) for p in outputs],
        "return_gradient_norm_last_batch": [float(p.weight.grad.detach().norm().cpu()) if p.weight.grad is not None else None for p in outputs]}


def verify_connected_preflight(model):
    outputs = [b.output if isinstance(b, ConnectedPairLocalBlock) else b.pair_return for _, b in relation_blocks(model)]
    norms = [float(p.weight.grad.detach().norm().cpu()) if p.weight.grad is not None else 0. for p in outputs]
    if not norms or not all(math.isfinite(v) and v > 0 for v in norms):
        raise ValueError("Relation return has a disconnected/nonfinite preflight gradient")
    return {"all_return_gradients_connected": True, "gradient_norms": norms}
