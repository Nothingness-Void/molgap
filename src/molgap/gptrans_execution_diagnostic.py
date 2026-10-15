"""Bounded disposable GPTrans execution probes; no scientific training endpoint."""
from contextlib import contextmanager
from pathlib import Path
import json
import math
import time

KINDS = ("attribution", "timing")
VARIANTS = ("native", "prepared", "foreach_optimizer", "foreach_ema", "optimized")
WALL_SECONDS = 1200


def configuration(kind):
    if kind not in KINDS:
        raise ValueError("Unsupported diagnostic kind")
    return {"format": "molgap-gptrans-execution-probe-v1", "kind": kind,
        "seed": 42, "physical_batch": 128, "cpu_threads": 1,
        "attribution_steps": 10, "timing_repeats": 3, "warmup_steps": 120,
        "measured_steps": 500, "profile_steps": 16, "wall_ceiling_seconds": WALL_SECONDS,
        "allocated_t4_hours_ceiling": 2 * WALL_SECONDS / 3600,
        "scientific_training_executed": False, "development_role_read": False,
        "official_validation_role_read": False, "test_dev_role_read": False,
        "test_challenge_role_read": False}


def read_diagnostic(root, base):
    from .experiment_family_workflow import _artifact_path, _json
    path = Path(root) / base / "diagnostic.json"
    if not path.exists():
        return None
    record = _json(_artifact_path(Path(root).absolute(), base + "/diagnostic.json"))
    if record != configuration(record.get("kind")):
        raise ValueError("Frozen diagnostic dimensions or roles changed")
    return record


def phases(record):
    if record is None:
        return ("preflight", "train")
    return (("preflight", "diagnostic-0") if record["kind"] == "attribution" else
        ("preflight", "diagnostic-0", "diagnostic-1", "diagnostic-2", "diagnostic-profile"))


def phase_device(device, phase):
    return 1 - device if phase == "diagnostic-1" else device


def execution_extension(record):
    """Experiment policy returns neutral phases and its own evidence hooks."""
    from .experiment_execution import ExecutionExtension, ExecutionPhase
    def artifacts(root, arms):
        return {"diagnostic": record, "scientific_training_executed": False,
            "diagnostic_artifacts": {arm: diagnostic_artifact_refs(root / arm) for arm in arms}}
    return ExecutionExtension(tuple(ExecutionPhase(name,
        worker="molgap.experiment_training_worker" if name == "preflight" else "molgap.gptrans_execution_diagnostic",
        swap_devices=name == "diagnostic-1", separate_output=name != "preflight",
        preflight_binding=name != "preflight") for name in phases(record)),
        record, record["wall_ceiling_seconds"], artifacts,
        lambda root: {"diagnostic_summary": summarize_probes(root, record)})


def variants_for_arm(arm):
    if arm not in {"original", "optimized"}:
        raise ValueError("Unknown diagnostic arm")
    return VARIANTS[:3] if arm == "original" else VARIANTS[3:]


def _cpu(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu(child) for child in value)
    return value


@contextmanager
def observe_step(model):
    """Local model hook plus explicit pre/post-clip observer; no global patch."""
    from .family_training_core import UpdateObserver
    observed = {}
    def forward(module, inputs, output):
        observed["prediction"] = _cpu(output)
    def gradients(current):
        return {name: _cpu(parameter.grad) for name, parameter in current.named_parameters()
                if parameter.grad is not None}
    def after_backward(current):
        observed["raw_gradients"] = gradients(current)
    def after_clip(current, norm):
        observed["clipped_gradients"] = gradients(current)
        observed["gradient_norm"] = _cpu(norm)
    handle = model.register_forward_hook(forward)
    try:
        yield observed, UpdateObserver(after_backward=after_backward, after_clip=after_clip)
    finally:
        handle.remove()


def _state(variant, recipe, initial):
    import torch
    from . import pcqm_gptrans_v4 as owner
    if variant not in VARIANTS:
        raise ValueError("Unsupported execution factor")
    if variant == "native":
        model, optimizer, scheduler, ema = owner._make_training_state(initial, "reference")
        mean, std = (torch.tensor(value, device="cuda") for value in
                     (recipe.target_mean_eV, recipe.target_std_eV))
        def step(batch, check_finite, observer=None):
            return owner._optimizer_step(model, optimizer, ema,
                batch.to("cuda", non_blocking=True), mean, std, check_finite=check_finite, observer=observer)
    else:
        from .family_training_kernel import FamilyKernel
        kernel = FamilyKernel(recipe, initial, device="cuda")
        model = kernel.model
        if variant == "prepared":
            kernel.optimizer = torch.optim.AdamW(model.parameters(), lr=recipe.learning_rate,
                weight_decay=recipe.weight_decay, foreach=False, fused=False)
        if variant in {"prepared", "foreach_optimizer"}:
            kernel.ema = owner.ExponentialMovingAverage(model, recipe.ema_decay)
        kernel._qualification_memory_fill = variant != "optimized"
        optimizer, ema = kernel.optimizer, kernel.ema
        scheduler = owner.FrozenEpochScheduler(optimizer)
        def step(batch, check_finite, observer=None):
            return kernel.step(batch, check_finite=check_finite, observer=observer)[0]
    scheduler.step(0)
    model.train()
    return model, optimizer, scheduler, ema, step


def _load_train(inputs):
    from . import pcqm_gptrans_v4 as owner
    from .training_reproducibility import sha256_file
    matches = [p for p in Path(inputs).rglob("*manifest*.json")
               if p.is_file() and sha256_file(p) == owner.MANIFEST_SHA256]
    if len(matches) != 1:
        raise ValueError("Expected one accepted fixed100K manifest")
    assets = owner.validate_fixed_assets(matches[0].parent, matches[0], verify_content=True)
    graphs, shards = owner._load_datasets(assets.train_paths)
    if len(graphs) != owner.TRAIN_ROWS:
        raise ValueError("Training graph count changed")
    stats = owner._run_target_stats(shards, "reference", None)
    if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-12) for a, b in
           zip(stats, (5.338300266634226, 1.2750902063230574))):
        raise ValueError("Training-only target statistics changed")
    return graphs


def _snapshot(model, optimizer, scheduler, ema):
    from .gptrans_legacy_execution_profile import gptrans_state_snapshot
    return gptrans_state_snapshot(model, optimizer, scheduler, ema)


def run_probe(*, spec, package, source, staged, inputs, output, arm, phase, preflight):
    import torch
    from . import pcqm_gptrans_v4 as owner
    from .experiment_package import verify_experiment_source_package
    from .experiment_family_workflow import RunContext, _json, _check_context
    from .experiment_execution import validate_execution_plan, validate_staged_trajectory
    from .gptrans_infrastructure_comparison import comparison_base, INITIAL_ARTIFACT_SHA256
    from .family_training_runner import _recipe_configuration, _state_fingerprint
    from .gptrans_execution_analysis import _native_validator
    from .screen_policy import canonical_fingerprint
    from .research_memory.trace import atomic_write, json_bytes
    from .training_reproducibility import configure_fp32_determinism, atomic_torch_save, sha256_file
    manifest = verify_experiment_source_package(package)
    config = _json(staged / "experiment_launch.json")
    jobs = validate_execution_plan(spec, config["jobs"])
    job = next(j for j in jobs if j["arm_id"] == arm)
    base = comparison_base(spec)
    diagnostic = read_diagnostic(source, base)
    if diagnostic is None or phase not in phases(diagnostic)[1:]:
        raise ValueError("No frozen authorization for this diagnostic phase")
    context = RunContext.for_training(spec, package, expected_package_identity=manifest["package_identity"],
        arm_id=arm, account=config["account"], run_reference=config["run_reference"])
    trajectory = validate_staged_trajectory(spec, arm, staged,
        expected_sha256=config["prospective_sha256"][arm])
    if trajectory["state_at_start"]["source_commit"] != context.source_commit:
        raise ValueError("Diagnostic prospective source differs")
    record = _json(source / job["recipe"])
    if sha256_file(source / job["recipe"]) != context.training_recipe_sha256:
        raise ValueError("Diagnostic recipe differs from Spec")
    identity = canonical_fingerprint(record) if arm == "original" else _recipe_configuration(record).identity
    _native_validator(arm_id=arm, root=preflight, context=context, record=record,
        certificate=_json(preflight / "runtime_certificate.json"), identity=identity)
    initial = staged / job["initial_state"]
    if sha256_file(initial) != INITIAL_ARTIFACT_SHA256:
        raise ValueError("Diagnostic initialization bytes changed")
    shared_record = _json(source / base / "optimized/contract.json")
    recipe = _recipe_configuration(shared_record)
    settings = configure_fp32_determinism(42)
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise ValueError("Diagnostic needs one isolated T4")
    graphs = _load_train(inputs)
    output.mkdir(parents=True, exist_ok=True)
    def save(name, value):
        atomic_write(output / name, json_bytes(value))
    mode = diagnostic["kind"]
    variants = variants_for_arm(arm) if mode == "attribution" else ("native" if arm == "original" else "optimized",)
    result = {"format": "molgap-gptrans-probe-output-v1", "status": "RUNNING", "context": context.to_dict(),
        "diagnostic": diagnostic, "phase": phase, "cpu_threads": torch.get_num_threads(),
        "physical_device": __import__("os").environ["CUDA_VISIBLE_DEVICES"],
        "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(0), "determinism": settings}, "variants": []}
    save("progress.json", result)
    for variant in variants:
        configure_fp32_determinism(42)
        model, optimizer, scheduler, ema, step = _state(variant, recipe, initial)
        # Common starting RNG separates construction/loader ownership from update math.
        configure_fp32_determinism(42)
        from .experiment_family_workflow import tensor_safe_rng_state
        from .training_reproducibility import capture_rng_state
        from torch_geometric.loader import DataLoader
        loader = DataLoader(graphs, batch_sampler=owner.DeterministicEpochBatchSampler(len(graphs), 0),
            num_workers=1, persistent_workers=True, pin_memory=True, prefetch_factor=2,
            generator=torch.Generator().manual_seed(42))
        rows = []
        if mode == "attribution":
            steps, warmup = diagnostic["attribution_steps"], 0
        else:
            warmup = diagnostic["warmup_steps"]
            steps = warmup + (diagnostic["profile_steps"] if phase == "diagnostic-profile" else diagnostic["measured_steps"])
        profile = None
        if phase == "diagnostic-profile":
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            profile = torch.profiler.profile(activities=activities, record_shapes=False,
                profile_memory=False, with_stack=False)
        began = None
        def before_fetch(next_step):
            nonlocal began
            if next_step == warmup + 1 and mode == "timing":
                torch.cuda.synchronize()
                atomic_torch_save(output / "warmup_checkpoint.pt", _snapshot(model, optimizer, scheduler, ema))
                if profile is not None:
                    profile.__enter__()
                began = time.perf_counter()
        def update(batch, *, check_finite):
            index = len(rows) if mode == "attribution" else 0
            # CPU collations are independently bound before each owner consumes them.
            batch_hash = owner._batch_sha256(batch) if mode == "attribution" else None
            if mode == "attribution":
                rng_before = tensor_safe_rng_state(capture_rng_state())
                with observe_step(model) as (observed, observer):
                    loss = step(batch, True, observer=observer)
                payload = {**observed, **_snapshot(model, optimizer, scheduler, ema),
                    "loss": _cpu(loss), "batch_sha256": batch_hash,
                    "rng_before": rng_before, "step": index + 1, "variant": variant}
                from .experiment_family_workflow import _finite_tree
                _finite_tree(payload, "diagnostic step")
                path = output / f"{variant}-step-{index + 1:02d}.pt"
                atomic_torch_save(path, payload)
                rows.append({"step": index + 1, "batch_sha256": batch_hash,
                    "path": path.name, "sha256": sha256_file(path),
                    "state_fingerprint": _state_fingerprint(payload)})
                save(variant + "-steps.json", rows)
            else:
                loss = step(batch, check_finite)
            return loss
        from contextlib import closing
        from .family_training_core import BatchPlan, train_batches
        try:
            with closing(train_batches(model, loader,
                    BatchPlan(diagnostic["physical_batch"], stop_step=steps),
                    update, before_fetch=before_fetch)) as stream:
                for processed in stream:
                    if processed.optimizer_step % 100 == 0:
                        save("cursor.json", {"variant": variant, "completed_updates": processed.optimizer_step,
                            "scientific_training_executed": False})
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - began if began is not None else None
        finally:
            if profile is not None and began is not None:
                profile.__exit__(None, None, None)
                profile.export_chrome_trace(str(output / "trace.json"))
            shutdown = getattr(getattr(loader, "_iterator", None), "_shutdown_workers", None)
            if shutdown is not None:
                shutdown()
        state = _snapshot(model, optimizer, scheduler, ema)
        atomic_torch_save(output / (variant + "-last.pt"), state)
        result["variants"].append({"variant": variant, "updates": steps, "warmup_updates": warmup,
            "measured_updates": steps - warmup, "measured_wall_seconds": elapsed,
            "measurement_scope": "common one-worker loader through update return, including small cursor receipts; CUDA synchronized endpoints; tensor checkpoint IO excluded",
            "last_state_fingerprint": _state_fingerprint(state), "steps": rows})
        save("progress.json", result)
        del model, optimizer, scheduler, ema, step, state
        torch.cuda.empty_cache()
    result["status"] = "COMPLETE"
    result["files"] = [{"path": p.name, "sha256": sha256_file(p)} for p in sorted(output.iterdir())
        if p.is_file() and p.name not in {"output_manifest.json", "progress.json"}]
    save("output_manifest.json", result)
    return result


def diagnostic_artifact_refs(root):
    from .training_reproducibility import sha256_file
    refs = []
    for manifest in sorted(Path(root).glob("diagnostic-*/output_manifest.json")):
        record = json.loads(manifest.read_bytes())
        refs.append({"path": manifest.relative_to(root).as_posix(), "sha256": sha256_file(manifest)})
        for item in record["files"]:
            path = manifest.parent / item["path"]
            from .experiment_family_workflow import _artifact_path
            path = _artifact_path(Path(root).absolute(), path.relative_to(root).as_posix())
            if sha256_file(path) != item["sha256"]:
                raise ValueError("Diagnostic artifact changed before retention")
            refs.append({"path": path.relative_to(root).as_posix(), "sha256": item["sha256"]})
    return refs


def tensor_deltas(left, right):
    """Exact schema and finite-value checks, with unrounded observed differences."""
    import torch
    result = {"equal": True, "maximum_absolute_delta": 0., "worst_leaf": None, "tensor_count": 0}
    def visit(a, b, name):
        if torch.is_tensor(a):
            if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError("Diagnostic tensor schema differs: " + name)
            if a.is_floating_point() and (not bool(torch.isfinite(a).all()) or not bool(torch.isfinite(b).all())):
                raise ValueError("Nonfinite diagnostic tensor: " + name)
            result["tensor_count"] += 1
            equal = torch.equal(a, b)
            result["equal"] &= equal
            if not equal:
                delta = float((a.double() - b.double()).abs().max())
                if delta > result["maximum_absolute_delta"]:
                    result.update(maximum_absolute_delta=delta, worst_leaf=name)
        elif isinstance(a, dict):
            if not isinstance(b, dict) or a.keys() != b.keys():
                raise ValueError("Diagnostic state keys differ: " + name)
            for key in a:
                visit(a[key], b[key], name + "/" + str(key))
        elif isinstance(a, (list, tuple)):
            if type(a) is not type(b) or len(a) != len(b):
                raise ValueError("Diagnostic state sequence differs: " + name)
            for i, (x, y) in enumerate(zip(a, b)):
                visit(x, y, name + "/" + str(i))
        elif a != b:
            raise ValueError("Diagnostic non-tensor semantics differ: " + name)
    visit(left, right, "state")
    return result


def summarize_probes(output, diagnostic, *, publish=True):
    import torch
    from .research_memory.trace import atomic_write, json_bytes
    from .training_reproducibility import sha256_file
    torch.set_num_threads(1)
    manifests = list(Path(output).glob("*/diagnostic-*/output_manifest.json"))
    records = [json.loads(path.read_bytes()) for path in manifests]
    expected = 2 if diagnostic["kind"] == "attribution" else 8
    if len(records) != expected or any(r["status"] != "COMPLETE" for r in records):
        raise ValueError("Incomplete diagnostic phase set")
    summary = {"format": "molgap-gptrans-probe-summary-v1", "status": "COMPLETE",
        "kind": diagnostic["kind"], "scientific_training_executed": False,
        "scientific_acceptance": "NOT_EVALUATED", "replay_readiness": "NOT_APPLICABLE_EXECUTION_DIAGNOSTIC"}
    if diagnostic["kind"] == "timing":
        from .execution_analysis import speed
        import statistics
        paired = []
        for phase in ("diagnostic-0", "diagnostic-1", "diagnostic-2"):
            selected = {r["context"]["arm_id"]: r for r in records if r["phase"] == phase}
            if set(selected) != {"original", "optimized"}:
                raise ValueError("Missing paired timing phase")
            old, new = (selected[a]["variants"][0]["measured_wall_seconds"] for a in ("original", "optimized"))
            paired.append({"phase": phase, "original_seconds": old, "optimized_seconds": new,
                "physical_devices": {a: selected[a]["physical_device"] for a in selected}, **speed(old, new)})
        summary.update(paired_timings=paired,
            median_paired_speedup=statistics.median(r["speedup_factor"] for r in paired),
            scope="common one-worker loader and one CPU thread; training updates only; no development/checkpoint throughput claim")
    else:
        bound = {}
        for path, record in zip(manifests, records):
            for variant in record["variants"]:
                bound[variant["variant"]] = (path.parent, variant)
        if set(bound) != set(VARIANTS):
            raise ValueError("Missing frozen attribution factor")
        comparisons = []
        for old, new in zip(VARIANTS, VARIANTS[1:]):
            rows = []
            for index in range(diagnostic["attribution_steps"]):
                states = []
                for variant in (old, new):
                    root, record = bound[variant]
                    item = record["steps"][index]
                    path = root / item["path"]
                    if sha256_file(path) != item["sha256"]:
                        raise ValueError("Attribution checkpoint changed")
                    states.append(torch.load(path, map_location="cpu", weights_only=True))
                a, b = states
                if a["batch_sha256"] != b["batch_sha256"] or a["step"] != b["step"]:
                    raise ValueError("Attribution inputs/updates differ")
                compared = {key: tensor_deltas(a[key], b[key]) for key in
                    ("rng_before", "prediction", "loss", "raw_gradients", "clipped_gradients", "model", "ema", "rng_state")}
                compared["optimizer_moments"] = tensor_deltas(a["optimizer"]["state"], b["optimizer"]["state"])
                rows.append({"step": index + 1, "batch_sha256": a["batch_sha256"], "stages": compared})
            comparisons.append({"original_factor": old, "changed_factor": new, "steps": rows,
                "first_divergence": next(({"step": r["step"], "stage": key} for r in rows
                    for key in ("rng_before", "prediction", "loss", "raw_gradients", "clipped_gradients", "model", "optimizer_moments", "ema", "rng_state")
                    if not r["stages"][key]["equal"]), None)})
        summary["factor_comparisons"] = comparisons
        summary["scope"] = "ten disposable updates; later differences may propagate from earlier changed state; not a long-run MAE attribution"
    if not publish:
        return summary
    path = Path(output) / "diagnostic_summary.json"
    atomic_write(path, json_bytes(summary))
    return {"path": path.name, "sha256": sha256_file(path)}


def main():
    import argparse
    from .experiment_spec import ExperimentSpec
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("source-root", "package-dir", "input-root", "output", "launch", "preflight-output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=("original", "optimized"))
    parser.add_argument("--phase", required=True)
    args = parser.parse_args()
    spec = ExperimentSpec.from_json((args.package_dir / "experiment_spec.json").read_text())
    run_probe(spec=spec, package=args.package_dir, source=args.source_root,
        staged=args.launch.parent, inputs=args.input_root, output=args.output,
        arm=args.arm, phase=args.phase, preflight=args.preflight_output)


if __name__ == "__main__":
    main()
