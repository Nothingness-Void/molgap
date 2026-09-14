"""Pairwise interactions between K1 edge recurrence and global-slot simplifications."""
from __future__ import annotations

from .k1_edge_memory import edge_step
from .pcqm_k1_variants import _return_allocation_update, _single_slot_processor_update

MODES = (
    "neural_atom_k1_edge_context_no_slot_attention",
    "neural_atom_k1_edge_context_uniform_return",
)
PARAMETERS = {MODES[0]: 3_608_897, MODES[1]: 3_658_817}


def _mix(model, layer, h, batch, *, diagnostics=False):
    if str(layer) not in model.base.neural_atom_mixers:
        return h, None
    mixer = model.base.neural_atom_mixers[str(layer)]
    if model.mode == MODES[0]:
        update, _, _, _, details = _single_slot_processor_update(mixer, None, h, batch)
    else:
        update, _, _, _, _, details = _return_allocation_update(
            mixer, h, batch, "uniform", remove_slot_attention=False
        )
    return h + update, details if diagnostics else None


def make_encoder(mode):
    if mode not in MODES:
        raise ValueError(mode)
    import torch.nn as nn
    from .qm9_neural_atom import make_encoder as frozen_encoder

    class EdgeSlotInteractionK1(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = frozen_encoder("neural_atom_k1")
            self.mode = mode
            if mode == MODES[0]:
                for mixer in self.base.neural_atom_mixers.values():
                    del mixer.slot_attention

        def encode(self, x, edge_index, edge_attr, batch, random_walk_pe):
            if tuple(random_walk_pe.shape) != (x.shape[0], self.base.rwse_dim):
                raise ValueError("RWSE shape changed")
            h = self.base._embed_nodes(x) + self.base.rwse_encoder(random_walk_pe.float())
            memory = self.base._embed_edges(edge_attr)
            for layer, (edge_update, block) in enumerate(
                zip(self.base.edge_updates, self.base.local_blocks), start=1
            ):
                memory, read = edge_step(edge_update, h, edge_index, memory, normalize_context=True)
                h = block(h, edge_index, batch, edge_attr=read)
                h, _ = _mix(self, layer, h, batch)
            return self.base._pool(h, batch)

        def forward(self, x, edge_index, edge_attr, batch, random_walk_pe):
            return self.base.head(self.encode(x, edge_index, edge_attr, batch, random_walk_pe))

    return EdgeSlotInteractionK1()


def check_mechanism(model, batch):
    import torch
    base = model.base
    with torch.no_grad():
        h = base._embed_nodes(batch.x) + base.rwse_encoder(batch.random_walk_pe.float())
        memory = base._embed_edges(batch.edge_attr)
        edge_checks, slot_checks = [], []
        for layer, (edge_update, block) in enumerate(zip(base.edge_updates, base.local_blocks), 1):
            before = memory.clone()
            memory, read = edge_step(edge_update, h, batch.edge_index, memory, normalize_context=True)
            source, target = batch.edge_index
            expected = before + edge_update.update(
                edge_update.output_norm(before) + edge_update.source(h[source]) + edge_update.target(h[target])
            )
            if not torch.equal(memory, expected) or not torch.equal(read, edge_update.output_norm(memory)):
                raise RuntimeError("Edge context/read equation failed")
            edge_checks.append({"layer": layer, "memory_rms": float(memory.square().mean().sqrt()),
                                "read_rms": float(read.square().mean().sqrt()),
                                "storage_differs_from_read": not torch.equal(memory, read)})
            h = block(h, batch.edge_index, batch.batch, edge_attr=read)
            h, details = _mix(model, layer, h, batch.batch, diagnostics=True)
            if details is not None:
                if model.mode == MODES[0]:
                    mass = details["assignment"].sum(dim=-1)
                    invariant = not hasattr(base.neural_atom_mixers[str(layer)], "slot_attention") and torch.allclose(mass, torch.ones_like(mass))
                else:
                    returned, valid = details["return_assignment"], details["valid"]
                    padded = returned.masked_select(~valid.unsqueeze(1))
                    invariant = torch.allclose(returned.sum(dim=-1), torch.ones_like(returned.sum(dim=-1))) and torch.equal(padded, torch.zeros_like(padded))
                if not invariant:
                    raise RuntimeError("Global-slot invariant failed")
                slot_checks.append({"layer": layer, "invariant": True})
        if len(edge_checks) != 9 or len(slot_checks) != 3:
            raise RuntimeError("Expected nine edge and three slot checks")
    return {"equations_verified": True, "real_bonds_only": True,
            "normalized_update_context": True,
            "slot_mode": "no-slot-attention" if model.mode == MODES[0] else "uniform-return",
            "layers": edge_checks, "slot_layers": slot_checks}
