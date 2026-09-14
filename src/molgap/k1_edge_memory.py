"""Separate persistent real-bond storage from its normalized read view."""
from __future__ import annotations

MODES = ("neural_atom_k1_edge_read_norm", "neural_atom_k1_edge_context_read_norm")
PARAMETERS = 3_658_817


def edge_step(update, h, edge_index, memory, *, normalize_context):
    """Reuse every frozen tensor; normalize reads, not the residual storage."""
    source, target = edge_index
    context_memory = update.output_norm(memory) if normalize_context else memory
    context = context_memory + update.source(h[source]) + update.target(h[target])
    memory = memory + update.update(context)
    return memory, update.output_norm(memory)


def make_encoder(mode):
    if mode not in MODES:
        raise ValueError(mode)
    import torch.nn as nn
    from .qm9_neural_atom import make_encoder as frozen_encoder

    class EdgeMemoryK1(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = frozen_encoder("neural_atom_k1")
            self.mode = mode

        def encode(self, x, edge_index, edge_attr, batch, random_walk_pe):
            if tuple(random_walk_pe.shape) != (x.shape[0], self.base.rwse_dim):
                raise ValueError("RWSE shape changed")
            h = self.base._embed_nodes(x)
            h = h + self.base.rwse_encoder(random_walk_pe.float())
            memory = self.base._embed_edges(edge_attr)
            for layer, (update, block) in enumerate(
                zip(self.base.edge_updates, self.base.local_blocks), start=1
            ):
                memory, read = edge_step(
                    update, h, edge_index, memory,
                    normalize_context=(self.mode == MODES[1]),
                )
                h = block(h, edge_index, batch, edge_attr=read)
                if str(layer) in self.base.neural_atom_mixers:
                    h = self.base.neural_atom_mixers[str(layer)](h, batch)
            return self.base._pool(h, batch)

        def forward(self, x, edge_index, edge_attr, batch, random_walk_pe):
            return self.base.head(self.encode(x, edge_index, edge_attr, batch, random_walk_pe))

    return EdgeMemoryK1()


def check_mechanism(model, batch):
    """Remote train-fixture checks; no held-out inference or local execution."""
    import torch
    base = model.base
    with torch.no_grad():
        h = base._embed_nodes(batch.x) + base.rwse_encoder(batch.random_walk_pe.float())
        initial = base._embed_edges(batch.edge_attr)
        memory = initial.clone()
        checks = []
        for layer, (update, block) in enumerate(zip(base.edge_updates, base.local_blocks), 1):
            before = memory.clone()
            memory, read = edge_step(update, h, batch.edge_index, memory,
                                     normalize_context=(model.mode == MODES[1]))
            source, target = batch.edge_index
            context_memory = update.output_norm(before) if model.mode == MODES[1] else before
            expected = before + update.update(context_memory + update.source(h[source]) + update.target(h[target]))
            if not torch.equal(memory, expected) or not torch.equal(read, update.output_norm(memory)):
                raise RuntimeError("Residual storage/read equation failed")
            if not torch.isfinite(memory).all() or not torch.isfinite(read).all():
                raise RuntimeError("Nonfinite real-bond state")
            checks.append({"layer": layer, "memory_rms": float(memory.square().mean().sqrt()),
                           "read_rms": float(read.square().mean().sqrt()),
                           "storage_differs_from_read": not torch.equal(memory, read)})
            h = block(h, batch.edge_index, batch.batch, edge_attr=read)
            if str(layer) in base.neural_atom_mixers:
                h = base.neural_atom_mixers[str(layer)](h, batch.batch)
        if len(checks) != 9 or not all(row["storage_differs_from_read"] for row in checks):
            raise RuntimeError("Persistent raw-state separation inactive")
    return {"equations_verified": True, "real_bonds_only": True,
            "normalized_update_context": model.mode == MODES[1], "layers": checks}


def snapshot_step(model, optimizer):
    import copy
    from .training_reproducibility import capture_rng_state
    return {"model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": capture_rng_state()}


def check_resume_equivalence(model, optimizer, snapshot, batch, mean, std, step):
    """Compare step two with an exact step-one restart on a train-only fixture."""
    import copy
    import torch
    from .training_reproducibility import capture_rng_state, restore_rng_state
    after = capture_rng_state()
    replay = copy.deepcopy(model).to("cuda").train()
    replay.load_state_dict(snapshot["model"])
    replay_optimizer = torch.optim.AdamW(replay.parameters(), lr=4e-4, weight_decay=1e-5)
    replay_optimizer.load_state_dict(snapshot["optimizer"])
    restore_rng_state(snapshot["rng"])
    step(replay, replay_optimizer, batch, mean, std)
    if not all(torch.equal(value, replay.state_dict()[key]) for key, value in model.state_dict().items()):
        raise RuntimeError("Interrupted/resumed optimizer state diverged")
    replay_rng = capture_rng_state()
    if not torch.equal(after["torch"], replay_rng["torch"]) or not all(
        torch.equal(a, b) for a, b in zip(after["cuda"], replay_rng["cuda"])
    ):
        raise RuntimeError("Interrupted/resumed random stream diverged")
    restore_rng_state(after)
    return True
