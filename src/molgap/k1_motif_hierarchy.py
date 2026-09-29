"""One topology-only motif exchange on the frozen K1 atom/bond encoder."""
from __future__ import annotations

MODE = "neural_atom_k1_motif_hierarchy"
MODES = (MODE,)
HIDDEN = 192
MOTIF = 64
LAYER = 6


def make_encoder(mode: str):
    if mode != MODE:
        raise ValueError(mode)
    import torch
    import torch.nn as nn
    from .qm9_neural_atom import make_encoder as make_k1

    class MotifK1(nn.Module):
        requires_motif_hierarchy = True

        def __init__(self):
            super().__init__()
            self.base = make_k1("neural_atom_k1")
            self.motif_project = nn.Linear(HIDDEN, MOTIF, bias=False)
            self.bond_embedding = nn.Embedding(4, 16)
            self.rule_embedding = nn.Embedding(8, 8)
            self.message = nn.Sequential(
                nn.Linear(MOTIF + 16 + 8, MOTIF), nn.SiLU(),
                nn.Linear(MOTIF, MOTIF),
            )
            self.motif_norm = nn.LayerNorm(MOTIF)
            self.return_projection = nn.Linear(MOTIF, HIDDEN, bias=False)
            nn.init.zeros_(self.return_projection.weight)

        def exchange(self, h, batch, membership, counts, sources, targets,
                     edge_counts, bond_types, rule_masks):
            counts, edge_counts = counts.view(-1).long(), edge_counts.view(-1).long()
            membership = membership.view(-1).long()
            sources, targets = sources.view(-1).long(), targets.view(-1).long()
            bond_types, rule_masks = bond_types.view(-1).long(), rule_masks.view(-1).long()
            graphs = int(batch.max()) + 1
            edges = sources.numel()
            if (counts.numel() != graphs or edge_counts.numel() != graphs
                    or membership.numel() != h.shape[0]
                    or int(edge_counts.sum()) != edges
                    or any(v.numel() != edges for v in (targets, bond_types, rule_masks))
                    or bool((counts < 1).any()) or bool((edge_counts < 0).any())):
                raise ValueError("Motif sidecar batch shape mismatch")
            if (bool((membership < 0).any())
                    or bool((membership >= counts[batch]).any())
                    or bool((bond_types < 0).any()) or bool((bond_types >= 4).any())
                    or bool((rule_masks < 1).any()) or bool((rule_masks >= 8).any())):
                raise ValueError("Motif sidecar category or membership mismatch")
            edge_graph = torch.repeat_interleave(
                torch.arange(graphs, device=batch.device), edge_counts
            )
            if edges and (bool((sources < 0).any()) or bool((targets < 0).any())
                          or bool((sources >= counts[edge_graph]).any())
                          or bool((targets >= counts[edge_graph]).any())):
                raise ValueError("Inter-motif edge leaves its molecule")
            offsets = counts.cumsum(0) - counts
            atom_motifs = membership + offsets[batch]
            total = int(counts.sum())
            projected = self.motif_project(h)
            pooled = h.new_zeros((total, MOTIF))
            pooled.index_add_(0, atom_motifs, projected)
            population = h.new_zeros((total, 1))
            population.index_add_(0, atom_motifs, h.new_ones((h.shape[0], 1)))
            if bool((population == 0).any()):
                raise ValueError("Empty motif in sidecar")
            pooled = pooled / population
            if edges:
                source = sources + offsets[edge_graph]
                target = targets + offsets[edge_graph]
                messages = self.message(torch.cat((
                    pooled[source], self.bond_embedding(bond_types),
                    self.rule_embedding(rule_masks),
                ), dim=-1))
                received = h.new_zeros((total, MOTIF))
                received.index_add_(0, target, messages)
                degree = h.new_zeros((total, 1))
                degree.index_add_(0, target, h.new_ones((edges, 1)))
                pooled = pooled + received / degree.clamp_min(1)
            return h + self.return_projection(self.motif_norm(pooled)[atom_motifs])

        def forward(self, x, edge_index, edge_attr, batch, random_walk_pe,
                    motif_membership, motif_count, motif_source, motif_target,
                    motif_edge_count, motif_bond_type, motif_bridge_rule_mask):
            if tuple(random_walk_pe.shape) != (x.shape[0], self.base.rwse_dim):
                raise ValueError("RWSE16 shape changed")
            h = self.base._embed_nodes(x)
            h = h + self.base.rwse_encoder(random_walk_pe.float())
            edge_state = self.base._embed_edges(edge_attr)
            for layer, (edge_update, block) in enumerate(
                zip(self.base.edge_updates, self.base.local_blocks), start=1
            ):
                edge_state = edge_update(h, edge_index, edge_state)
                h = block(h, edge_index, batch, edge_attr=edge_state)
                if str(layer) in self.base.neural_atom_mixers:
                    h = self.base.neural_atom_mixers[str(layer)](h, batch)
                if layer == LAYER:
                    h = self.exchange(
                        h, batch, motif_membership, motif_count, motif_source,
                        motif_target, motif_edge_count, motif_bond_type,
                        motif_bridge_rule_mask,
                    )
            return self.base.head(self.base._pool(h, batch))

    return MotifK1()
