"""Observe the retained K1/2 owner; no replacement epoch or optimizer loop."""
from functools import wraps
from types import SimpleNamespace

from .family_execution_profile import _ExecutionCapture, _ObservedBatches, ProfileConfig
from .screen_policy import canonical_fingerprint


class LegacyK1ExecutionProfiler(_ExecutionCapture):
    """Exclusive-process observer for the retained original K1 trainer."""

    def __init__(self, owner, recipe, output, config, *, model=None):
        import torch
        self.owner, self._model = owner, model
        proxy = SimpleNamespace(closed=False, progress=SimpleNamespace(optimizer_step=0),
            kernel=SimpleNamespace(device=torch.device("cuda" if model is None else next(model.parameters()).device),
                                   model=model),
            recipe=SimpleNamespace(identity=canonical_fingerprint(recipe), family="neural_atom_k1", batch_size=128))
        super().__init__(proxy, output, config)
        self._optimizers = set()
        self.on_pass = None

    def __enter__(self):
        if self._model is None and self.config.module_paths:
            raise ValueError("Deferred legacy models require empty module_paths")
        return super().__enter__()

    def _install(self):
        import torch
        from torch_geometric.data import Data
        from .experiment_family_workflow import FamilyOutputSession
        for obj, name, label in (
                (self.owner, "_forward", "forward"),
                (self.owner, "_evaluate", "development"),
                (Data, "to", "transfer"),
                (torch.autograd, "backward", "backward"),
                (torch.nn.utils, "clip_grad_norm_", "gradient_clip"),
                (FamilyOutputSession, "checkpoint", "checkpoint_publish"),
                (FamilyOutputSession, "selected", "selected_publish"),
                (FamilyOutputSession, "epoch_finished", "trace_publish")):
            self._wrap(obj, name, label)
        if self._model is not None:
            for path in self.config.module_paths:
                self._observe_module(self._model.get_submodule(path), "module/" + path)
        original_step = self.owner._optimizer_step
        @wraps(original_step)
        def step(model, optimizer, *args, **kwargs):
            self._begin_batch()
            if id(optimizer) not in self._optimizers:
                self._optimizers.add(id(optimizer))
                self._wrap(optimizer, "zero_grad", "zero_grad")
                self._wrap(optimizer, "step", "optimizer")
            with self.phase("train_update"):
                result = original_step(model, optimizer, *args, **kwargs)
            self._after_update()
            self.session.progress.optimizer_step = self._updates
            return result
        self._replace(self.owner, "_optimizer_step", step)
        original_loader = self.owner._train_loader
        @wraps(original_loader)
        def loader(*args, **kwargs):
            with self.phase("loader_setup"):
                source = original_loader(*args, **kwargs)
            return _ObservedBatches(source, self)
        self._replace(self.owner, "_train_loader", loader)
        if self.on_pass is not None:
            original_pass = FamilyOutputSession.epoch_finished
            @wraps(original_pass)
            def finished(outputs, **values):
                result = original_pass(outputs, **values)
                self.on_pass(outputs, values["epoch"])
                return result
            self._replace(FamilyOutputSession, "epoch_finished", finished)


def state_fingerprint(model, optimizer):
    from .experiment_family_workflow import tensor_safe_rng_state
    from .family_training_runner import _state_fingerprint
    from .training_reproducibility import capture_rng_state
    return _state_fingerprint({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "rng": tensor_safe_rng_state(capture_rng_state())})


def run_profiled_legacy(owner, runner, options):
    """Bind native proof and capture to the retained recipe and original runner."""
    from pathlib import Path
    import time
    from .experiment_family_workflow import _json, RunContext
    from .family_training_runner import _verify_profile_proof, normalized_file_sha
    from .training_reproducibility import atomic_json
    recipe = _json(Path(options["recipe_path"]))
    config = ProfileConfig.from_dict(recipe["profiling"])
    root = Path(options["output"])
    preflight = Path(options.get("preflight_dir") or root)
    certificate = _json(preflight / "runtime_certificate.json")
    _verify_profile_proof(preflight, SimpleNamespace(identity=canonical_fingerprint(recipe)),
        certificate.get("profile_proof"), required=True)
    profile_root = root / "profiles" / f"step_0_{time.time_ns()}"
    context = RunContext.for_training(options["spec"], options["package_dir"],
        expected_package_identity=options["expected_package_identity"], arm_id=options["arm_id"],
        account=options["account"], run_reference=options["run_reference"])
    profiler = LegacyK1ExecutionProfiler(owner, recipe, profile_root, config)
    from .k1_infrastructure_comparison import is_comparison
    if is_comparison(options["spec"]):
        from .experiment_execution import validate_staged_trajectory
        from .experiment_resume import create_resume_bundle
        trajectory = validate_staged_trajectory(options["spec"], options["arm_id"], Path(options["input_root"]))
        def retain(outputs, epoch):
            if epoch % 10 == 0 and epoch < 40:
                create_resume_bundle(root, context, trajectory,
                    root.parent / "checkpoint_chunks" / context.arm_id / f"pass_{epoch:02d}")
        profiler.on_pass = retain
    with profiler as observed:
        result = runner(**options)
    atomic_json(root / "execution_profile_receipt.json", {"context": context.to_dict(),
        "report_path": (profile_root / "profile.json").relative_to(root).as_posix(),
        "report_sha256": normalized_file_sha(profile_root / "profile.json"),
        "trace_sha256": observed.report["trace_sha256"], "capture_status": observed.report["status"]})
    return result
