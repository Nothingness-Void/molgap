"""Two isolated interventions over the accepted triplet-aggregate model."""
from pathlib import Path
import torch
from torch import nn

from . import gptrans_triplet_communication as parent
from .gptrans_local_relation import state_digest, valid_pairs
from .screen_policy import canonical_fingerprint
from .training_reproducibility import atomic_torch_save
from .v4_runtime import normalized_source_sha256

MODES = ("degree_local_triplet_drop_ema999", "degree_local_triplet_product_ema999")
BASE_MODE = parent.MODES[0]
PARAMETERS = {MODES[0]: parent.PARAMETERS[BASE_MODE],
              MODES[1]: parent.PARAMETERS[BASE_MODE] + 4 * 2 * (32 * 16 + 16)}
PARAMETER_CAP = max(PARAMETERS.values())
FORMAT = "molgap-triplet-followup-initial-v1"
PARENT_TENSOR_SHA256 = "32df5f9a19d353a95407d6bd4d53a10a2165ada7bb02c32392dd15a85cb0fd9a"


def configuration(mode):
    if mode not in MODES:
        raise ValueError("Unknown triplet follow-up")
    return {**parent.configuration(BASE_MODE), "parent_variant": BASE_MODE,
        "relation_mechanism": "triplet-branch-drop-p01" if mode == MODES[0] else "triplet-channelwise-product",
        "branch_drop_probability": .1 if mode == MODES[0] else 0.,
        "branch_drop_scope": "per-molecule; triplet-return-only; train-only",
        "new_forward_rng": mode == MODES[0],
        "new_rng": "private-cpu-generator; layer-seeded; checkpointed-uint8-buffer",
        "composition": "unchanged" if mode == MODES[0] else "U(third-pair)*V(value-pair) before sum",
        "initialization": "accepted-aggregate-initial-tensors-exact; new-tensors-only"}


def architecture_identity(mode):
    return canonical_fingerprint({"variant": mode, "config": configuration(mode),
        "parent_identity": parent.architecture_identity(BASE_MODE),
        "source": normalized_source_sha256(Path(__file__))})


class PrivateBranchDrop(nn.Module):
    """Keep parent dropout draws unchanged, including after checkpoint resume.

    Only B scalar masks are sampled on CPU, not an N-cubed mask. Keeping the
    small RNG buffer on CPU avoids a device synchronization at every forward.
    Native state_dict/EMA/checkpoint owners already preserve integer buffers.
    """
    def __init__(self, depth):
        super().__init__()
        self.register_buffer("rng_state", torch.Generator().manual_seed(42000 + depth).get_state())

    def _apply(self, fn, recurse=True):
        state = self.rng_state
        super()._apply(fn, recurse=recurse)
        self.rng_state = state.cpu()
        return self

    def forward(self, delta):
        if not self.training:
            return delta
        generator = torch.Generator().set_state(self.rng_state)
        mask = (torch.rand(delta.shape[0], generator=generator) >= .1).to(delta)
        self.rng_state.copy_(generator.get_state())
        return delta * mask.view(-1, 1, 1, 1) / .9


def product_direction(value, operand, bias, gate, valid, *, outward=False):
    weights = parent.masked_weights(bias, gate, valid[..., None], dim=1 if outward else 2)
    left = operand * weights[..., None]
    return (torch.einsum("bkihd,bkjhd->bijhd", left, value) if outward else
            torch.einsum("bikhd,bjkhd->bijhd", left, value))


class FollowupBlock(parent.TripletLocalBlock):
    def __init__(self, core, mode, depth):
        super().__init__(core, BASE_MODE)
        self.followup_mode = mode
        if mode == MODES[0]:
            self.branch_drop = PrivateBranchDrop(depth)
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(43000 + depth)
                self.in_operand = nn.Linear(32, 16)
                self.out_operand = nn.Linear(32, 16)

    def forward(self, node, pair, padding, edges):
        valid = valid_pairs(padding)
        values = self.norm(pair.permute(0, 2, 3, 1)).masked_fill(~valid[..., None], 0)
        returns = []
        for direction in ("in", "out"):
            value = getattr(self, direction + "_value")(values).unflatten(-1, (2, 8))
            value = value.masked_fill(~valid[..., None, None], 0)
            bias = getattr(self, direction + "_bias")(values)
            gate = getattr(self, direction + "_gate")(values)
            if self.followup_mode == MODES[0]:
                returned = parent.aggregate_direction(value, bias, gate, valid, outward=direction == "out")
            else:
                operand = getattr(self, direction + "_operand")(values).unflatten(-1, (2, 8))
                returned = product_direction(value, operand, bias, gate, valid, outward=direction == "out")
            returns.append(returned.flatten(-2))
        delta = self.output(torch.cat(returns, -1)).masked_fill(~valid[..., None], 0).permute(0, 3, 1, 2)
        if self.followup_mode == MODES[0]:
            delta = self.branch_drop(delta)
        if getattr(self, "_capture_relation", False):
            self._relation_rms = (delta.detach().square().sum() / (valid.sum() * 32).clamp_min(1)).sqrt()
            self._capture_relation = False
        return self.core(node, pair + delta, padding, edges)


def _extras(mode, depth):
    if mode == MODES[0]:
        return {"branch_drop.rng_state": PrivateBranchDrop(depth).rng_state}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(43000 + depth)
        return {direction + "_operand." + k: v for direction in ("in", "out")
                for k, v in nn.Linear(32, 16).state_dict().items()}


def construct(mode, base_state):
    configuration(mode)
    model = parent.construct(BASE_MODE, base_state)
    with torch.random.fork_rng(devices=[]):
        for depth in parent.INSERTIONS:
            old = model.blocks[depth - 1]
            block = FollowupBlock(old.core, mode, depth)
            block.load_state_dict({**old.state_dict(), **_extras(mode, depth)}, strict=True)
            model.blocks[depth - 1] = block
    model._molgap_author_variant = mode
    return model


def _parent_initial():
    from .constants import REPO_ROOT
    path = REPO_ROOT / "platforms/_records/kaggle/initializations/degree_local_triplet_aggregate_ema999_initial.pt"
    facts = parent.read_initial_facts(BASE_MODE, path)
    if facts["state_sha256"] != PARENT_TENSOR_SHA256:
        raise ValueError("Accepted aggregate parent initialization changed")
    return path, facts


def freeze_initial(mode, base_path, output):
    configuration(mode)
    path, facts = _parent_initial()
    state = {k: v.clone() for k, v in torch.load(path, map_location="cpu", weights_only=True)["model_state"].items()}
    for depth in parent.INSERTIONS:
        state.update({f"blocks.{depth - 1}." + k: v for k, v in _extras(mode, depth).items()})
    result = {"parameters": PARAMETERS[mode], "state_sha256": state_digest(state),
              "architecture_identity": architecture_identity(mode)}
    atomic_torch_save(Path(output), {"format": FORMAT, "variant": mode, **result,
        "parent_tensor_sha256": facts["state_sha256"], "model_state": state})
    read_initial_facts(mode, output)
    return result


def read_initial_facts(mode, path):
    configuration(mode)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    state = payload["model_state"]
    extras = {f"blocks.{d - 1}." + k: v for d in parent.INSERTIONS for k, v in _extras(mode, d).items()}
    recovered = {k: v for k, v in state.items() if k not in extras}
    if (payload.get("format") != FORMAT or payload.get("variant") != mode
            or payload.get("architecture_identity") != architecture_identity(mode)
            or payload.get("parent_tensor_sha256") != PARENT_TENSOR_SHA256
            or payload.get("parameters") != PARAMETERS[mode]
            or state_digest(state) != payload.get("state_sha256")
            or state_digest(recovered) != PARENT_TENSOR_SHA256
            or any(k not in state or not torch.equal(state[k], v) for k, v in extras.items())
            or sum(v.numel() for k, v in state.items() if not k.endswith("rng_state")) != PARAMETERS[mode]
            or not all(torch.isfinite(v).all().item() for v in state.values())):
        raise ValueError("Follow-up initial state differs from frozen aggregate parent/delta")
    return {k: payload[k] for k in ("parameters", "state_sha256", "architecture_identity")}


def load_initial(mode, path):
    from .gptrans import OGBGPTransTiny
    facts = read_initial_facts(mode, path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = construct(mode, OGBGPTransTiny().state_dict())
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model_state"], strict=True)
    if sum(p.numel() for p in model.parameters()) != PARAMETERS[mode]:
        raise ValueError("Follow-up parameter inventory differs")
    model._capacity_initial_sha256, model._capacity_parameters = facts["state_sha256"], facts["parameters"]
    return model


begin_capture = parent.begin_capture
diagnostics = parent.diagnostics
verify_connected_preflight = parent.verify_connected_preflight
