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


SAM_RHO = 0.05  # Foret et al. 2010.01412 default; one fixed value, no tuning.


def _k1_sam_step(model, optimizer, batch, mean, std, call, *, clip_norm, check_finite, clip_parameters, observer):
    """Sharpness-aware step: gradient at w + rho*g/||g||, applied to the restored w.

    optimizer_update cannot restore weights between its backward and step, so SAM owns
    this branch. The perturbed pass must not move BatchNorm running statistics. It does
    consume extra CUDA RNG, so later dropout masks differ from the baseline arm.
    """
    import math
    import torch
    import torch.nn.functional as functional
    from torch.nn.modules.batchnorm import _BatchNorm
    from .family_training_core import update_scope
    target = (batch.y.view(-1) - mean) / std

    def evaluate():
        prediction = call(model, batch)
        return functional.l1_loss(prediction, target), (prediction.detach() - target).abs().sum()

    parameters = [p for p in model.parameters() if p.requires_grad]
    with update_scope(observer, "zero_grad"):
        optimizer.zero_grad(set_to_none=True)
    loss, absolute = evaluate()
    if check_finite and not bool(torch.isfinite(loss)):
        raise RuntimeError("Training loss became non-finite")
    with update_scope(observer, "backward"):
        loss.backward()
    grads = [p.grad for p in parameters if p.grad is not None]
    scale = SAM_RHO / (torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads])) + 1e-12)
    if check_finite and not bool(torch.isfinite(scale)):
        raise RuntimeError("SAM gradient became non-finite")
    originals = [p.detach().clone() for p in parameters]
    norms = [m for m in model.modules() if isinstance(m, _BatchNorm)]
    saved = [(m.momentum, None if m.num_batches_tracked is None else m.num_batches_tracked.clone()) for m in norms]
    try:
        with update_scope(observer, "sam_perturb"), torch.no_grad():
            for p in parameters:
                if p.grad is not None:
                    p.add_(p.grad * scale)
        for m in norms:
            m.momentum = 0.0
        optimizer.zero_grad(set_to_none=True)
        perturbed, _ = evaluate()
        if check_finite and not bool(torch.isfinite(perturbed)):
            raise RuntimeError("Perturbed loss became non-finite")
        with update_scope(observer, "backward"):
            perturbed.backward()
    finally:
        with torch.no_grad():
            for p, original in zip(parameters, originals):
                p.copy_(original)
            for m, (momentum, tracked) in zip(norms, saved):
                m.momentum = momentum
                if tracked is not None:
                    m.num_batches_tracked.copy_(tracked)
    if observer is not None and observer.after_backward is not None:
        observer.after_backward(model)
    with update_scope(observer, "gradient_clip"):
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters() if clip_parameters is None else clip_parameters, clip_norm)
    if observer is not None and observer.after_clip is not None:
        observer.after_clip(model, gradient_norm)
    if check_finite and not bool(torch.isfinite(gradient_norm)):
        raise RuntimeError("Training gradient became non-finite")
    with update_scope(observer, "optimizer"):
        optimizer.step()
    return loss.detach(), absolute, int(target.numel())


def k1_optimizer_step(model, optimizer, batch, mean, std, mode="reference", *,
                    clip_norm=1.0, check_finite=False, clip_parameters=None, observer=None, forward=None):
    """Family loss hook on the scale-independent optimizer core."""
    import torch.nn.functional as functional
    from .family_training_core import optimizer_update
    call = forward or k1_forward
    if mode == "sam":
        return _k1_sam_step(model, optimizer, batch, mean, std, call, clip_norm=clip_norm,
            check_finite=check_finite, clip_parameters=clip_parameters, observer=observer)

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


def policy(mode):
    """Registered mode policy: SAM keeps the reference normalized-L1 objective identity.

    Only the update rule (see _k1_sam_step) differs, so same-run comparison must not
    report a loss change.
    """
    from .family_training_policy import FamilyModePolicy
    if mode != "sam":
        raise ValueError("No family_model_ops policy for mode " + str(mode))
    return FamilyModePolicy(objective_name="reference")
