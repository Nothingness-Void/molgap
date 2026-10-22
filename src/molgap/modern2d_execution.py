"""Explicit prepared execution candidate; no admission, IO or training loop.

Validate topology on CPU before staging it. Keep labels out of the layout and
transfer each packed categorical tensor once, rather than reading GPU graph
sizes and regrouping edges on every forward. The historical model forward and
frozen recipes are not redirected by importing this module.
"""
from contextlib import contextmanager
from copy import copy

import torch

from .modern2d_execution_context import (
    PreparedTopology, prepare_topology, current_prepared, _PREPARED_CALL,
)


def prepared_forward(model, cpu_batch, *, device):
    """One opt-in model forward; caller retains optimizer/session ownership.

    This callable is for bounded execution qualification until a new source and
    native update/resume parity have been accepted. It grants no compute/role
    authority and never monkeypatches model.forward or a frozen recipe.
    """
    from .ppgt_author import PPGTAuthor
    from .esa_native import ESANative
    if type(model) not in {PPGTAuthor, ESANative}:
        raise TypeError("Prepared execution supports the exact PPGT/ESA models only")
    device = torch.device(device)
    model_device = next(model.parameters()).device
    if (device.type != model_device.type or
            (device.index is not None and device.index != model_device.index)):
        raise ValueError("Prepared execution requested device differs from model")
    # ESA expands categorical tokens on CPU, not after staging topology on GPU.
    stage_device = model_device if type(model) is PPGTAuthor else torch.device("cpu")
    prepared = prepare_topology(cpu_batch, device=stage_device,
        max_nodes=model.config.max_nodes, max_edges=model.config.max_edges)
    token = _PREPARED_CALL.set((model, cpu_batch, prepared))
    try:
        # __call__ is essential: substrate addons own pre-/post-forward frames.
        return model(cpu_batch)
    finally:
        _PREPARED_CALL.reset(token)


@contextmanager
def prepared_kernel_execution(kernel):
    """Reuse the actual FamilyKernel/Session for a bounded execution probe.

    Only generated CPU fixtures or already admitted train-only native scratch
    preflights may select this unqualified candidate. Formal scientific training
    is deliberately blocked, even with a valid historical native certificate.
    Enter before the existing profiler and exit after it; never wrap a live run.
    """
    from .family_training_kernel import FamilyKernel
    from .ppgt_author import PPGTAuthor
    from .esa_native import ESANative
    from .shared_model_adapter import validate_model_output
    if type(kernel) is not FamilyKernel or type(kernel.model) not in {PPGTAuthor, ESANative}:
        raise TypeError("Prepared execution requires the owning PPGT/ESA FamilyKernel")
    recipe = kernel.recipe
    synthetic = kernel.device.type == "cpu" and recipe.dataset.manifest.startswith("synthetic/")
    scratch = (kernel.device.type == "cuda" and recipe.optimizer_steps == 2 and
        not recipe.dataset.development and recipe.development_rows == 0 and
        recipe.selection == "final-live" and (kernel._qualification_admission is not None or
            (recipe.formal_interface and recipe.formal_training.phase == "native-preflight")))
    if not (synthetic or scratch):
        raise ValueError("Prepared execution is unqualified for scientific training; scratch admission required")
    if getattr(kernel, "_prepared_execution_active", False):
        raise RuntimeError("Prepared kernel execution cannot be nested")
    sentinel = object()
    names = ("_model_batch", "_model_forward", "_prepared_execution_active")
    saved = {name: vars(kernel).get(name, sentinel) for name in names}
    def stage(batch, *, training):
        borrowed = copy(batch)
        if training:
            # The existing normalized-gap update owns loss/backward/Adam. Only
            # its label vector moves here; the model owns prepared topology.
            borrowed.y = borrowed.y.to(kernel.device, non_blocking=True)
        return borrowed
    def forward(model, batch):
        output = prepared_forward(model, batch, device=kernel.device)
        return validate_model_output(output, rows=int(batch.num_graphs),
            device=next(model.parameters()).device)
    kernel._model_batch, kernel._model_forward = stage, forward
    kernel._prepared_execution_active = True
    try:
        yield {"execution_path": "modern2d-prepared-v2-candidate", "native_qualified": False}
    finally:
        for name, value in saved.items():
            if value is sentinel:
                vars(kernel).pop(name, None)
            else:
                setattr(kernel, name, value)
