"""Bounded native resource and fixed TRAIN observations for hop-shell K1.

The shared kernel owns updates, evaluation and weight views. These observations
do not participate in development selection or change the training cursor.
"""
from dataclasses import replace
from pathlib import Path
import gc
import math
import statistics
import time


TRAIN_ROWS = 4096
TRAIN_PASSES = (10, 20, 30, 40)
WARMUP_STEPS = 2
TIMED_STEPS = 8
MAXIMUM_OVERHEAD = 0.25


def qualify_resources(recipe, data, tuning, initial_state):
    """Measure optimizer-inclusive native steps on identical TRAIN-only batches."""
    import torch
    from torch_geometric.loader import DataLoader
    from .family_training_data import PassBatchSampler
    from .family_training_kernel import FamilyKernel
    from .training_reproducibility import capture_rng_state, restore_rng_state
    from .experiment_family_workflow import tensor_digest

    if recipe.family != "neural_atom_k1" or recipe.mode != "composed":
        raise ValueError("Hop-shell qualification requires composed K1")
    rng = capture_rng_state()
    began = time.perf_counter()
    timings, sources = {}, []
    reference = replace(recipe, mode="reference", addons=())
    try:
        sampler = PassBatchSampler(data.train_rows, recipe.batch_size, recipe.seed,
            algorithm=recipe.sampler, drop_last=recipe.drop_last)
        loader = DataLoader(data.train, batch_sampler=sampler, num_workers=0,
            generator=torch.Generator().manual_seed(recipe.seed))
        iterator = iter(loader)
        try:
            batches = [next(iterator) for _ in range(WARMUP_STEPS + TIMED_STEPS)]
        except StopIteration as exc:
            raise ValueError("Insufficient authorized TRAIN batches for native qualification") from exc
        finally:
            del iterator
        for batch in batches:
            index = getattr(batch, "source_idx", getattr(batch, "row_index", None))
            if index is None:
                raise ValueError("Native qualification requires TRAIN source identities")
            sources.append(index.reshape(-1).long().cpu())
        for label, configuration in (("reference", reference), ("candidate", recipe)):
            restore_rng_state(rng)
            gc.collect()
            torch.cuda.empty_cache()
            kernel = FamilyKernel(configuration, Path(initial_state), device="cuda")
            try:
                kernel.model.train()
                for batch in batches[:WARMUP_STEPS]:
                    kernel.step(batch, check_finite=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                samples = []
                for batch in batches[WARMUP_STEPS:]:
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    kernel.step(batch, check_finite=True)
                    torch.cuda.synchronize()
                    samples.append(time.perf_counter() - tick)
                timings[label] = {"median_step_seconds": statistics.median(samples),
                    "samples_seconds": samples,
                    "peak_memory_bytes": torch.cuda.max_memory_allocated()}
            finally:
                kernel.close()
                del kernel
                gc.collect()
        overhead = timings["candidate"]["median_step_seconds"] / timings["reference"]["median_step_seconds"] - 1
        if not math.isfinite(overhead):
            raise ValueError("Nonfinite native overhead measurement")
        elapsed = time.perf_counter() - began
        return {"format": "molgap-k1-hop-shell-resource-v1",
            "status": "accepted" if overhead <= MAXIMUM_OVERHEAD else "rejected",
            "recipe_identity": recipe.identity, "reference_recipe_identity": reference.identity,
            "dataset_manifest_sha256": recipe.dataset.manifest_sha256,
            "train_source_idx_sha256": tensor_digest(torch.cat(sources), role="source_idx"),
            "fixture": "first-ten-batches-frozen-pass-zero-order",
            "warmup_steps": WARMUP_STEPS, "timed_steps": TIMED_STEPS, "timings": timings,
            "synchronized_step_overhead_fraction": overhead,
            "maximum_overhead_fraction": MAXIMUM_OVERHEAD,
            "formal_training_exposure": 0, "development_consumed": False,
            "costs": [{"metric": "wall_seconds", "unit": "seconds", "value": elapsed,
                "scope": "native-resource-window-excludes-bootstrap-and-queue"}],
            "cpu_allocation_seconds": None, "queue_seconds": None}
    finally:
        restore_rng_state(rng)


def refresh_resources(resource, pass_index):
    """Retain the original measured gate when actual-state resume is qualified."""
    if (resource.get("status") != "accepted" or
            resource.get("maximum_overhead_fraction") != MAXIMUM_OVERHEAD or
            not math.isfinite(resource.get("synchronized_step_overhead_fraction", math.inf)) or
            resource["synchronized_step_overhead_fraction"] > MAXIMUM_OVERHEAD):
        raise ValueError("Missing accepted hop-shell native resource qualification")
    return {**resource, "resume_pass_index": pass_index}


def observe_pass(kernel, loaders, root, progress):
    """Evaluate a fixed ordered TRAIN cohort with the current live weights."""
    if progress.pass_index not in TRAIN_PASSES:
        return
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from .family_training_core import evaluate_development
    from .experiment_family_workflow import tensor_digest
    from .training_reproducibility import atomic_json, capture_rng_state, restore_rng_state, sha256_file
    from .v4_runtime import state_dict_sha256

    if loaders.data.train_rows < TRAIN_ROWS:
        raise ValueError("Fixed TRAIN cohort requires at least 4096 authorized rows")
    rng = capture_rng_state()
    original_state = state_dict_sha256(kernel.model.state_dict())
    began = time.perf_counter()
    try:
        batches = DataLoader(Subset(loaders.data.train, range(TRAIN_ROWS)),
            batch_size=kernel.recipe.batch_size, shuffle=False, num_workers=0,
            generator=torch.Generator().manual_seed(kernel.recipe.seed))
        result = kernel.training_state.evaluate("live", lambda model: evaluate_development(
            model, batches, forward=lambda model, batch: kernel.forward(batch, model=model),
            rows=TRAIN_ROWS, source_start=loaders.data.train_source_start,
            mean=kernel.recipe.target_mean_eV, std=kernel.recipe.target_std_eV))
        report = {"format": "molgap-k1-fixed-train-evaluation-v1", "role": "TRAIN",
            "weights": "live", "metric": "MAE", "unit": "eV", "target": "Gap",
            "model_mode": "eval", "rows": TRAIN_ROWS,
            "cohort": "first-4096-ordered-training-rows",
            "pass_index": progress.pass_index, "optimizer_step": progress.optimizer_step,
            "sample_presentations": progress.sample_presentations,
            "recipe_identity": kernel.recipe.identity,
            "live_state_sha256": original_state,
            "dataset_manifest_sha256": kernel.recipe.dataset.manifest_sha256,
            "mae_eV": result["mae_eV"],
            "source_idx_sha256": tensor_digest(result["source_idx"], role="source_idx"),
            "target_eV_sha256": tensor_digest(result["target_eV"], role="target"),
            "prediction_eV_sha256": tensor_digest(result["prediction_eV"], role="target"),
            "wall_seconds": time.perf_counter() - began,
            "selection_used": False, "formal_training_exposure": 0}
        if state_dict_sha256(kernel.model.state_dict()) != original_state:
            raise ValueError("Fixed TRAIN evaluation changed the live weights or buffers")
        folder = Path(root) / "train_cohort"
        atomic_json(folder / f"pass_{progress.pass_index:02d}.json", report)
        files = [folder / f"pass_{value:02d}.json" for value in TRAIN_PASSES
                 if (folder / f"pass_{value:02d}.json").is_file()]
        atomic_json(folder / "manifest.json", {"format": "molgap-fixed-train-receipts-v1",
            "recipe_identity": kernel.recipe.identity,
            "files": {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
                      for path in files}})
    finally:
        restore_rng_state(rng)


def policy(mode):
    from .family_training_policy import FamilyModePolicy
    if mode != "composed":
        raise ValueError("Unknown hop-shell observation mode")
    return FamilyModePolicy(qualify_resources=qualify_resources,
        refresh_resources=refresh_resources, observe_pass=observe_pass)
