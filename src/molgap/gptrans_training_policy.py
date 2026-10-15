"""Server research-mode semantics outside the reusable training lifecycle.

The existing addon registry selects this owner explicitly. Desktop registrations
may select another policy; neither branch/hostname nor data size selects it.
"""
from pathlib import Path
import time

from .family_training_policy import FamilyModePolicy


def construct(recipe, state):
    if recipe.mode.startswith("degree_aggregate_"):
        from .gptrans_research_campaign import construct
    else:
        from .gptrans_cycle_pair import construct
    return construct(recipe.mode, state)


def prepare_batch(batch, *, training):
    if training:
        from .gptrans_research_campaign import correspondence_layout
        batch.execution_layout = correspondence_layout(dict(batch.execution_layout))
    return batch


def correspondence_objective(model, batch, forward, recipe):
    from .gptrans_research_campaign import AUX_WEIGHT
    prediction = forward(model, batch).reshape(-1)
    gap = (prediction - (batch.y.reshape(-1).float() - recipe.target_mean_eV) / recipe.target_std_eV).abs()
    auxiliary = model._correspondence_loss
    if auxiliary is None:
        raise RuntimeError("Missing clean atom/bond correspondence loss")
    model._auxiliary_metric = auxiliary.detach()
    return gap.mean() + AUX_WEIGHT * auxiliary, (gap.detach().sum(), int(batch.num_graphs))


def correspondence_after_update(model):
    model._correspondence_loss = None


def attach_inputs(root, selection, data):
    from .gptrans_cycle_inputs import attach_sidecar
    return attach_sidecar(root, selection, data)


def refresh_resources(resource, completed_passes):
    from .family_pair_release import resource_estimate
    return {**resource, **resource_estimate(train_step_seconds=resource["maximum_observed_train_step_seconds"],
        forward_seconds=resource["maximum_observed_forward_seconds"], reserved_bytes=resource["reserved_bytes"],
        total_bytes=resource["total_bytes"], completed_passes=completed_passes)}


def policy(mode):
    correspondence = mode == "degree_aggregate_correspondence_ema999"
    cycle = mode in {"degree_local_cycle_common_ema999", "degree_aggregate_cycle_ema999"}
    return FamilyModePolicy(construct=construct,
        prepare_batch=prepare_batch if correspondence else None,
        objective=correspondence_objective if correspondence else None,
        after_update=correspondence_after_update if correspondence else None,
        attach_inputs=attach_inputs if cycle else None, input_sidecar_required=cycle,
        extra_development_weights=("live",), train_metric_unit="eV", process_timing=True,
        segment_passes=10, qualify_resources=qualify_resources, refresh_resources=refresh_resources,
        objective_name="aggregate-correspondence-0.01-tau0.2-bonds8" if correspondence else
            "aggregate-shared-gap-v1" if mode.startswith("degree_aggregate_") else "local-cycle-shared-v1")


def qualify_resources(recipe, data, tuning, initial):
    """Disposable native profile and zero-start proof, never a third trained arm."""
    import torch
    from .family_training_kernel import FamilyKernel
    from .family_training_loaders import FamilyLoaders
    from .family_pair_release import resource_estimate
    from .training_reproducibility import configure_fp32_determinism
    from .gptrans_execution_backend import bind_model_layout, prepared_forward
    from .family_training_runner import _probe_recipe
    probe = _probe_recipe(recipe)
    configure_fp32_determinism(recipe.seed)
    kernel = FamilyKernel(probe, initial, device="cuda")
    loaders = FamilyLoaders(probe, data, tuning)
    train_times, forward_times = [], []
    try:
        iterator = iter(loaders.train(0, 0))
        batch = next(iterator)
        zero_start = None
        if recipe.mode == "degree_local_cycle_common_ema999" or recipe.mode.startswith("degree_aggregate_"):
            from .gptrans_cycle_pair import construct, PARENT
            from .v4_runtime import read_frozen_state_artifact
            state, _ = read_frozen_state_artifact(initial, expected_state_sha256=recipe.initialization_sha256)
            if recipe.mode.startswith("degree_aggregate_"):
                from .gptrans_research_campaign import construct, BASE as PARENT, inherited_state
                parent_state = inherited_state(state)
            else:
                parent_state = {k: v for k, v in state.items() if k != "cycle_embedding.weight"}
            parent = construct(PARENT, parent_state)
            parent.load_state_dict(parent_state, strict=True)
            bind_model_layout(parent)
            parent.to("cuda").eval()
            kernel.model.eval()
            with torch.no_grad():
                actual = kernel.forward(batch)
                expected = prepared_forward(parent, kernel._gptrans_batch(batch, training=False))
            if not torch.equal(actual, expected):
                raise ValueError("Cycle zero-start encoder is not exactly the local parent on native inputs")
            zero_start = {"accepted": True, "maximum_prediction_delta": 0., "rows": int(batch.num_graphs)}
            del parent, state, parent_state, actual, expected
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        kernel.model.train()
        kernel.step(batch, check_finite=True)
        for index in range(20):
            torch.cuda.synchronize()
            began = time.perf_counter()
            batch = next(iterator)
            kernel.model.train()
            kernel.step(batch, check_finite=True)
            torch.cuda.synchronize()
            train_times.append(time.perf_counter() - began)
            began = time.perf_counter()
            kernel.model.eval()
            with torch.no_grad():
                kernel.forward(batch)
            torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - began)
        connected = None
        if recipe.mode.startswith("degree_aggregate_"):
            from .gptrans_research_campaign import native_gradients
            connected = native_gradients(kernel.model)
        return {**resource_estimate(train_step_seconds=max(train_times), forward_seconds=max(forward_times),
            reserved_bytes=torch.cuda.max_memory_reserved(), total_bytes=torch.cuda.get_device_properties(0).total_memory),
            "zero_start": zero_start, "connected": connected, "formal_training_exposure": 0, "disposable_optimizer_updates": 21}
    finally:
        loaders.close()
        kernel.close()
