"""Leaf family model/input/loss operations; no trainer, dataset or platform owner."""
from __future__ import annotations


def make_gptrans_backbone(variant="reference"):
    from .gptrans import OGBGPTransTiny
    from .gptrans_variants import apply_variant
    from .gptrans_memory import apply_memory_variant
    apply = apply_memory_variant if variant in {"memory_value", "memory_message"} else apply_variant
    model = OGBGPTransTiny(node_channels=256, pair_channels=32, num_layers=12,
        num_heads=8, shortest_path_cap=20, dropout=.1, drop_path=.1, layer_scale=1., n_targets=1)
    return apply(model, variant)


def k1_forward(model, batch):
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    ).view(-1)



def k1_batch_to_device(batch, device, *, training: bool, auxiliary: bool = False):
    """Transfer only model/loss inputs; keep source IDs and boundaries on CPU."""
    fields = ["x", "edge_index", "edge_attr", "batch", "random_walk_pe"]
    if training:
        fields.append("y")
    if auxiliary:
        if not training:
            raise ValueError("Auxiliary inputs belong to training only")
        fields.append("chemical_fingerprint")
    return batch.to(device, *fields, non_blocking=True)


def k1_optimizer_step(model, optimizer, batch, mean, std, mode="reference", *,
                    clip_norm=1.0, check_finite=False, clip_parameters=None, observer=None, forward=None):
    """Family loss hook on the scale-independent optimizer core."""
    import torch.nn.functional as functional
    from .family_training_core import optimizer_update
    call = forward or k1_forward

    def loss_builder():
        if mode == "clean_fingerprint":
            representation = model.encode(batch.x, batch.edge_index, batch.edge_attr,
                                           batch.batch, batch.random_walk_pe)
            prediction = model.head(representation).view(-1)
        else:
            prediction = call(model, batch)
        target = (batch.y.view(-1) - mean) / std
        loss = functional.l1_loss(prediction, target)
        absolute = (prediction.detach() - target).abs().sum()
        if mode in ("mean2", "mean2_clean_second"):
            from contextlib import nullcontext
            second_view = nullcontext()
            if mode == "mean2_clean_second":
                from .k1_clean_second import clean_second_view
                second_view = clean_second_view(model)
            with second_view:
                second = call(model, batch)
            loss = 0.5 * (loss + functional.l1_loss(second, target))
            absolute = 0.5 * (absolute + (second.detach() - target).abs().sum())
        if mode == "clean_fingerprint":
            loss = loss + 0.1 * functional.binary_cross_entropy_with_logits(
                model.clean_fingerprint_head(representation), batch.chemical_fingerprint.float())
        return loss, (absolute, int(target.numel()))

    loss, (absolute, rows) = optimizer_update(model, optimizer, loss_builder,
        clip_norm=clip_norm, check_finite=check_finite, clip_parameters=clip_parameters, observer=observer)
    return loss, absolute, rows
