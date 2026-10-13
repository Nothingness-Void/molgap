"""Three bounded mechanisms over the same retained triplet-aggregate parent.

Scientific deltas only. The existing shared family owns training and durability.
LEMON motivates correspondence, not a faithful reproduction or a Gap guarantee.
"""
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

from .gptrans_capacity import BondLocalGPTrans
from .gptrans_triplet_communication import construct as aggregate_factory, MODES as TRIPLETS
from .gptrans_local_relation import INSERTIONS
from .bond_message_carry import BondMessageCarry
from .v4_runtime import read_frozen_state_artifact, state_dict_sha256
from .training_reproducibility import atomic_torch_save, sha256_file

BASE, CYCLE, CARRY, AUX = MODES = (
    "degree_aggregate_shared_ema999", "degree_aggregate_cycle_ema999",
    "degree_aggregate_carry_ema999", "degree_aggregate_correspondence_ema999")
PARENT_FILE_SHA = "f391d94878e6cb949aba8afc9f39330a127a48a0f0d165dfca9dbf11755894dc"
PARENT_STATE_SHA = "32df5f9a19d353a95407d6bd4d53a10a2165ada7bb02c32392dd15a85cb0fd9a"
AUX_WEIGHT, TEMPERATURE, BONDS_PER_GRAPH = .01, .2, 8


def inherited_state(state):
    return {k: v for k, v in state.items() if not k.startswith((
        "cycle_embedding.", "bond_carry.", "atom_projection.", "bond_projection."))}


def base_state(state):
    result = {}
    for name, value in inherited_state(state).items():
        fields = name.split(".")
        if fields[0] == "blocks":
            if int(fields[1]) + 1 in INSERTIONS:
                if fields[2] != "core":
                    continue
                fields.pop(2)
            if fields[2] != "core":
                continue
            fields.pop(2)
        result[".".join(fields)] = value
    return result


def correspondence_layout(layout):
    """Bound public topology-only contrastive work to8undirected bonds per graph."""
    graph, source, target = layout["edges"]
    if graph.device.type != "cpu":
        raise ValueError("Correspondence layout must precede H2D")
    selected = []
    for index in range(layout["batch_size"]):
        ids = torch.where((graph == index) & (source < target))[0]
        if len(ids):
            # Uniform canonical ranks; no random draws or label-dependent choice.
            rank = torch.linspace(0, len(ids) - 1, min(BONDS_PER_GRAPH, len(ids))).round().long()
            selected.append(ids[rank])
    chosen = torch.cat(selected) if selected else torch.empty(0, dtype=torch.long)
    layout["correspondence_edges"] = (graph[chosen], source[chosen] + 1, target[chosen] + 1)
    return layout


def correspondence_loss(atom, bond):
    if atom.shape != bond.shape or atom.ndim != 2 or atom.shape[1] != 32:
        raise ValueError("Correspondence projection shape changed")
    if len(atom) < 2:
        return (atom.sum() + bond.sum()) * 0
    logits = F.normalize(atom, dim=-1) @ F.normalize(bond, dim=-1).T / TEMPERATURE
    positive = torch.arange(len(atom), device=atom.device)
    return (F.cross_entropy(logits, positive) + F.cross_entropy(logits.T, positive)) / 2


class CampaignGPTrans(BondLocalGPTrans):
    def forward(self, x, edge_index, edge_attr, batch, random_walk_pe=None):
        self._carry_previous, self._carry_active = None, True
        self._correspondence_loss = None
        try:
            node, pair, padding = self._dense_inputs(x, edge_index, edge_attr, batch)
            layout = self._execution_layout
            if self._campaign_mode == CYCLE:
                pair = pair + self.cycle_embedding(layout["cycle_categories"]).permute(0, 3, 1, 2)
            graph, source, target = self._local_edges(edge_index, batch, len(x))
            edges = graph, source + 1, target + 1
            for block in self.blocks:
                node, pair = block(node, pair, padding, edges)
            if self._campaign_mode == AUX and self.training:
                g, s, t = layout["correspondence_edges"]
                atom = self.atom_projection((node[g, s] + node[g, t]) * .5)
                bond = self.bond_projection((pair[g, :, s, t] + pair[g, :, t, s]) * .5)
                self._correspondence_loss = correspondence_loss(atom, bond)
            return self.readout(torch.cat((node[:, 0], pair[:, :, 0, 0]), -1))
        finally:
            self._carry_previous, self._carry_active = None, False


def construct(mode, state):
    if mode not in MODES:
        raise ValueError("Unknown four-arm campaign mechanism")
    model = aggregate_factory(TRIPLETS[0], base_state(state))
    model.__class__ = CampaignGPTrans
    model._campaign_mode = model._molgap_author_variant = mode
    model._correspondence_loss = model._carry_previous = None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        if mode == CYCLE:
            model.cycle_embedding = nn.Embedding(3, 32, padding_idx=0)
            nn.init.zeros_(model.cycle_embedding.weight)
        elif mode == CARRY:
            model.bond_carry = BondMessageCarry()
            for depth, outer in enumerate(model.blocks, 1):
                local = outer.core if depth in INSERTIONS else outer
                def carry_hook(_module, _inputs, value, depth=depth):
                    if not model._carry_active:
                        raise ValueError("Sparse memory used outside an owning forward")
                    carried = model.bond_carry(value, model._carry_previous, depth=depth)
                    model._carry_previous = carried
                    return carried
                local.message.register_forward_hook(carry_hook)
        elif mode == AUX:
            model.atom_projection = nn.Linear(model.node_channels, 32)
            model.bond_projection = nn.Linear(32, 32)
    return model


def freeze_initial(mode, parent, output):
    """Tensor extension only: no molecular constructor or forward locally."""
    if sha256_file(parent) != PARENT_FILE_SHA:
        raise ValueError("Retained aggregate initialization file changed")
    state, _ = read_frozen_state_artifact(parent, expected_state_sha256=PARENT_STATE_SHA)
    state = dict(state)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        if mode == CYCLE:
            state["cycle_embedding.weight"] = torch.zeros(3, 32)
        elif mode == CARRY:
            state["bond_carry.logits"] = torch.zeros(11)
        elif mode == AUX:
            width = state["node_norm.weight"].numel()
            for prefix, layer in (("atom_projection", nn.Linear(width, 32)),
                                  ("bond_projection", nn.Linear(32, 32))):
                state.update({prefix + "." + key: value for key, value in layer.state_dict().items()})
        elif mode != BASE:
            raise ValueError("Unknown campaign initialization")
    digest = state_dict_sha256(state)
    atomic_torch_save(Path(output), {"model_state": state, "state_sha256": digest,
        "parent_tensor_sha256": PARENT_STATE_SHA, "parameters": sum(v.numel() for v in state.values())})
    return {"state_sha256": digest, "file_sha256": sha256_file(output),
            "parameters": sum(v.numel() for v in state.values())}


def native_gradients(model):
    from .gptrans_triplet_communication import verify_connected_preflight
    result = verify_connected_preflight(model)
    if model._campaign_mode == BASE:
        return result
    prefixes = {CYCLE: ("cycle_embedding",), CARRY: ("bond_carry",),
                AUX: ("atom_projection", "bond_projection")}[model._campaign_mode]
    norms = {name: float(value.grad.detach().norm().cpu()) if value.grad is not None else 0.
             for name, value in model.named_parameters() if name.startswith(prefixes)}
    if not norms or not all(torch.isfinite(torch.tensor(v)) and v > 0 for v in norms.values()):
        raise ValueError("Campaign mechanism has disconnected or nonfinite gradients")
    if model._campaign_mode == CARRY and not (model.bond_carry.logits.grad.abs() > 0).all():
        raise ValueError("Not all11cross-depth coefficients are connected")
    return {**result, "mechanism_gradient_norms": norms}
