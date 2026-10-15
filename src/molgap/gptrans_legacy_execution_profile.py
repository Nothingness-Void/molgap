"""Observe the retained GPTrans-T V4 loop without replacing its trainer.

The V4 owner remains the only implementation of model construction, sampling,
optimization, EMA, evaluation, and checkpoint semantics.  This module only
installs short-lived wrappers around those existing entry points so an opt-in
profile can be collected in the same worker and then removed completely.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
import time

from .family_execution_profile import _ExecutionCapture, _ObservedBatches, ProfileConfig
from .screen_policy import canonical_fingerprint


def _cpu_tree(value):
    """Copy a nested optimizer/EMA/RNG payload to CPU for durable proof."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(child) for child in value)
    return value


def gptrans_state_snapshot(model, optimizer, scheduler, ema):
    """Return the complete state that a two-step GPTrans proof compares."""
    from .experiment_family_workflow import tensor_safe_rng_state
    from .training_reproducibility import capture_rng_state

    return _cpu_tree({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "ema": ema.state_dict(),
        "rng_state": tensor_safe_rng_state(capture_rng_state()),
    })


def gptrans_state_fingerprint(state):
    """Fingerprint every model, optimizer, EMA, scheduler and RNG leaf."""
    from .family_training_runner import _state_fingerprint

    required = {"model", "optimizer", "scheduler", "ema", "rng_state"}
    if type(state) is not dict or set(state) != required:
        raise ValueError("GPTrans profile state is incomplete")
    return _state_fingerprint(state)


def _validate_native_profile_state(state):
    """Reject rehashed placeholders before admitting a native profile proof.

    ``gptrans_state_fingerprint`` intentionally remains a small, CPU-friendly
    structural hash for observer tests.  The native admission gate is stricter:
    it checks the state shape that the retained V4 owner actually serializes,
    including the two-step AdamW state and one isolated CUDA RNG stream.
    """
    import torch

    gptrans_state_fingerprint(state)

    model = state["model"]
    ema = state["ema"]
    if type(model) is not dict or not model:
        raise ValueError("Native GPTrans profile model state is empty")
    if type(ema) is not dict or not ema:
        raise ValueError("Native GPTrans profile EMA state is empty")
    if set(model) != set(ema):
        raise ValueError("Native GPTrans model/EMA tensor identities differ")

    def tensor_state(value, label):
        if (not torch.is_tensor(value) or value.device.type != "cpu"
                or value.numel() == 0):
            raise ValueError("Native GPTrans " + label + " tensor is empty or non-portable")
        if (value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()):
            raise ValueError("Native GPTrans " + label + " tensor is non-finite")

    for name, value in model.items():
        tensor_state(value, "model " + str(name))
        other = ema[name]
        tensor_state(other, "EMA " + str(name))
        if value.shape != other.shape or value.dtype != other.dtype:
            raise ValueError("Native GPTrans model/EMA tensor shapes differ")

    optimizer = state["optimizer"]
    if (type(optimizer) is not dict
            or set(optimizer) != {"state", "param_groups"}
            or type(optimizer["state"]) is not dict or not optimizer["state"]
            or type(optimizer["param_groups"]) is not list
            or not optimizer["param_groups"]):
        raise ValueError("Native GPTrans optimizer state is empty or incomplete")
    parameter_ids = set()
    for group in optimizer["param_groups"]:
        if type(group) is not dict or type(group.get("params")) is not list or not group["params"]:
            raise ValueError("Native GPTrans optimizer parameter groups are incomplete")
        if any(type(parameter_id) is not int for parameter_id in group["params"]):
            raise ValueError("Native GPTrans optimizer parameter IDs are invalid")
        parameter_ids.update(group["params"])
    if not set(optimizer["state"]).issubset(parameter_ids):
        raise ValueError("Native GPTrans optimizer state is not bound to parameters")
    for parameter_id, slot in optimizer["state"].items():
        if type(parameter_id) is not int or type(slot) is not dict:
            raise ValueError("Native GPTrans optimizer slots are incomplete")
        if not {"step", "exp_avg", "exp_avg_sq"} <= set(slot):
            raise ValueError("Native GPTrans optimizer moments are incomplete")
        step = slot["step"]
        if torch.is_tensor(step):
            if (step.device.type != "cpu" or step.numel() != 1
                    or not bool(torch.isfinite(step).all()) or float(step.item()) != 2.0):
                raise ValueError("Native GPTrans optimizer step is not exactly two")
        elif (type(step) not in {int, float} or isinstance(step, bool)
              or step != 2 or step != step):
            raise ValueError("Native GPTrans optimizer step is not exactly two")
        tensor_state(slot["exp_avg"], "optimizer exp_avg")
        tensor_state(slot["exp_avg_sq"], "optimizer exp_avg_sq")
        if slot["exp_avg"].shape != slot["exp_avg_sq"].shape:
            raise ValueError("Native GPTrans optimizer moments have different shapes")

    scheduler = state["scheduler"]
    if (type(scheduler) is not dict or set(scheduler) != {"epoch"}
            or type(scheduler["epoch"]) is not int or scheduler["epoch"] != 0):
        raise ValueError("Native GPTrans scheduler epoch is not zero")

    rng_state = state["rng_state"]
    if type(rng_state) is not dict or set(rng_state) != {"python", "numpy", "torch", "cuda"}:
        raise ValueError("Native GPTrans portable RNG state is incomplete")
    from .experiment_family_workflow import _validate_rng_state
    _validate_rng_state(rng_state)
    if type(rng_state["cuda"]) is not list or len(rng_state["cuda"]) != 1:
        raise ValueError("Native GPTrans portable RNG state needs one CUDA entry")
    return state


class LegacyGPTransExecutionProfiler(_ExecutionCapture):
    """Exclusive-process observer for the unchanged GPTrans-T owner."""

    def __init__(self, owner, recipe, output, config, *, model=None,
                 starting_optimizer_step=0):
        import torch

        if type(starting_optimizer_step) is not int or starting_optimizer_step < 0:
            raise ValueError("GPTrans profiler starting optimizer step is invalid")
        self.owner, self._model = owner, model
        self._starting_optimizer_step = starting_optimizer_step
        device = torch.device("cuda" if model is None else next(model.parameters()).device)
        proxy = SimpleNamespace(
            closed=False,
            progress=SimpleNamespace(optimizer_step=starting_optimizer_step),
            kernel=SimpleNamespace(device=device, model=model),
            recipe=SimpleNamespace(identity=canonical_fingerprint(recipe), family="gptrans_t", batch_size=128),
        )
        super().__init__(proxy, output, config)
        self._optimizers = set()
        self._schedulers = set()
        self._emas = set()
        self.on_epoch = None

    def __enter__(self):
        # The formal V4 arm constructs its model inside run_training.  Defer
        # module hooks until _make_training_state returns that model.
        if self._model is None and self.config.module_paths:
            saved = self.config
            self.config = replace(saved, module_paths=())
            try:
                return super().__enter__()
            finally:
                self.config = saved
        return super().__enter__()

    def _attach_model(self, model):
        if self._model is model:
            return
        self._model = model
        self.session.kernel.model = model
        for path in self.config.module_paths:
            self._observe_module(model.get_submodule(path), "module/" + path)

    def _observe_optimizer(self, optimizer):
        identity = id(optimizer)
        if identity in self._optimizers:
            return
        self._optimizers.add(identity)
        self._wrap(optimizer, "zero_grad", "zero_grad")
        self._wrap(optimizer, "step", "optimizer")

    def _observe_scheduler(self, scheduler):
        identity = id(scheduler)
        if identity in self._schedulers:
            return
        self._schedulers.add(identity)
        self._wrap(scheduler, "step", "schedule")

    def _observe_ema(self, ema):
        identity = id(ema)
        if identity in self._emas:
            return
        self._emas.add(identity)
        self._wrap(ema, "update", "ema")

    def _install(self):
        import torch
        from torch_geometric.data import Data

        for obj, name, label in (
            (self.owner, "_forward", "forward"),
            (self.owner, "_evaluate", "development"),
            (self.owner, "_save_checkpoint", "checkpoint_publish"),
            (Data, "to", "transfer"),
            (torch.autograd, "backward", "backward"),
            (torch.nn.utils, "clip_grad_norm_", "gradient_clip"),
        ):
            if hasattr(obj, name):
                self._wrap(obj, name, label)

        # The formal arm creates (model, optimizer, scheduler, ema) lazily.
        # Observing this factory keeps hooks local to the one owner invocation.
        original_factory = self.owner._make_training_state

        @wraps(original_factory)
        def factory(*args, **kwargs):
            result = original_factory(*args, **kwargs)
            if type(result) is not tuple or len(result) != 4:
                raise RuntimeError("GPTrans training state factory ABI changed")
            model, optimizer, scheduler, ema = result
            self._attach_model(model)
            self._observe_optimizer(optimizer)
            self._observe_scheduler(scheduler)
            self._observe_ema(ema)
            return result

        self._replace(self.owner, "_make_training_state", factory)

        original_step = self.owner._optimizer_step

        @wraps(original_step)
        def step(model, optimizer, ema, *args, **kwargs):
            self._begin_batch()
            self._attach_model(model)
            self._observe_optimizer(optimizer)
            self._observe_ema(ema)
            with self.phase("train_update"):
                result = original_step(model, optimizer, ema, *args, **kwargs)
            self._after_update()
            self.session.progress.optimizer_step = self._starting_optimizer_step + self._updates
            return result

        self._replace(self.owner, "_optimizer_step", step)

        original_loader = self.owner._training_loader

        @wraps(original_loader)
        def loader(*args, **kwargs):
            with self.phase("loader_setup"):
                source = original_loader(*args, **kwargs)
            return _ObservedBatches(source, self)

        self._replace(self.owner, "_training_loader", loader)

        if self.on_epoch is not None:
            original_epoch = getattr(self.owner, "_v5_append_checkpoint_observation", None)
            if original_epoch is not None:
                @wraps(original_epoch)
                def epoch(*args, **kwargs):
                    result = original_epoch(*args, **kwargs)
                    self.on_epoch(*args, **kwargs)
                    return result
                self._replace(self.owner, "_v5_append_checkpoint_observation", epoch)


def _profile_config(recipe):
    value = recipe.get("profiling")
    return None if value is None else ProfileConfig.from_dict(value)


def _proof_artifact(root: Path, record: dict, label: str):
    import torch
    from .experiment_family_workflow import _artifact_path
    from .family_training_runner import normalized_file_sha

    if (type(record) is not dict or set(record) != {"path", "sha256", "fingerprint"}
            or type(record["path"]) is not str):
        raise ValueError("Malformed GPTrans profile " + label + " artifact")
    path = _artifact_path(root, record["path"])
    if normalized_file_sha(path) != record["sha256"]:
        raise ValueError("GPTrans profile " + label + " artifact hash changed")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if gptrans_state_fingerprint(state) != record["fingerprint"]:
        raise ValueError("GPTrans profile " + label + " state fingerprint changed")
    return state


def verify_profile_state_proof(root, recipe, proof, *, required):
    """Verify exact profile/plain state equality before formal training."""
    if proof is None:
        if required:
            raise ValueError("Profiled GPTrans recipe lacks state proof")
        return None
    expected = {"format", "status", "recipe_identity", "updates", "plain", "profiled", "equal"}
    if type(proof) is not dict or set(proof) != expected:
        raise ValueError("Malformed GPTrans profile state proof")
    if (proof["format"] != "molgap-gptrans-profile-state-proof-v1"
            or proof["status"] != "accepted"
            or proof["recipe_identity"] != canonical_fingerprint(recipe)
            or proof["updates"] != 2
            or proof["equal"] is not True):
        raise ValueError("GPTrans profile state proof binding differs from recipe")
    if (type(proof["plain"]) is not dict or type(proof["profiled"]) is not dict
            or proof["plain"].get("path") == proof["profiled"].get("path")):
        raise ValueError("GPTrans profile state proof reuses one state artifact")
    root = Path(root).absolute()
    plain = _proof_artifact(root, proof["plain"], "plain state")
    profiled = _proof_artifact(root, proof["profiled"], "profiled state")
    _validate_native_profile_state(plain)
    _validate_native_profile_state(profiled)
    if gptrans_state_fingerprint(plain) != gptrans_state_fingerprint(profiled):
        raise ValueError("GPTrans profiled/plain state differs")
    return proof


def _state_record(root: Path, path: Path, state: dict):
    from .family_training_runner import normalized_file_sha

    return {"path": path.relative_to(root).as_posix(),
            "sha256": normalized_file_sha(path),
            "fingerprint": gptrans_state_fingerprint(state)}


def qualify_profile_state(*, owner, recipe, train_graphs, train_shards, initial_state_path,
                          output: Path, target_transform_path=None):
    """Run equal two-step profile/plain scratch states on the real train role."""
    import torch
    from .training_reproducibility import atomic_json, atomic_torch_save, configure_fp32_determinism

    config = _profile_config(recipe)
    if config is None:
        return None, None
    output = Path(output).absolute()
    profile_root = output / "profile_qualification"
    if profile_root.exists():
        raise ValueError("Profile qualification output already exists; reconcile before retry")
    mean_value, std_value = owner._run_target_stats(train_shards, recipe.get("mode", "reference"), target_transform_path)
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")

    def run(observe):
        configure_fp32_determinism(owner.SEED)
        model, optimizer, scheduler, ema = owner._make_training_state(initial_state_path, recipe.get("mode", "reference"))
        context = (LegacyGPTransExecutionProfiler(owner, recipe, profile_root, replace(config, warmup_steps=0, active_steps=2), model=model)
                   if observe else nullcontext())
        with context as profiler:
            if profiler is not None:
                profiler._observe_scheduler(scheduler)
            scheduler.step(0)
            loader = owner._training_loader(train_graphs, 0)
            iterator = iter(loader)
            try:
                for _ in range(2):
                    batch = next(iterator)
                    batch = batch.to("cuda", non_blocking=True)
                    owner._optimizer_step(model, optimizer, ema, batch, mean, std, check_finite=True)
            finally:
                close = getattr(iterator, "close", None)
                if close is not None:
                    close()
        state = gptrans_state_snapshot(model, optimizer, scheduler, ema)
        return state, profiler

    plain, _ = run(False)
    profiled, observed = run(True)
    if observed.report.get("status") != "COMPLETE" or observed.report.get("captured_updates") != 2:
        raise ValueError("GPTrans native profiler did not capture exactly two updates")
    if not observed.report.get("device_events_observed"):
        raise ValueError("GPTrans native profiler did not capture CUDA events")
    if gptrans_state_fingerprint(plain) != gptrans_state_fingerprint(profiled):
        raise ValueError("GPTrans profiled/plain optimizer state differs")

    plain_path = profile_root / "plain-state.pt"
    profiled_path = profile_root / "profiled-state.pt"
    atomic_torch_save(plain_path, plain)
    atomic_torch_save(profiled_path, profiled)
    proof = {"format": "molgap-gptrans-profile-state-proof-v1", "status": "accepted",
        "recipe_identity": canonical_fingerprint(recipe), "updates": 2,
        "plain": _state_record(output, plain_path, plain),
        "profiled": _state_record(output, profiled_path, profiled), "equal": True}
    atomic_json(profile_root / "state_proof.json", proof)
    return {"report": "profile_qualification/profile.json",
            "sha256": _file_sha(output / "profile_qualification/profile.json")}, proof


def _file_sha(path):
    from .family_training_runner import normalized_file_sha
    return normalized_file_sha(path)


def _retained_optimizer_step(owner, recipe, context, root: Path):
    """Read only the retained V4 epoch marker before installing observers.

    The owner still performs its complete checkpoint validation and restore in
    ``run_training``.  This small metadata read gives the profile the correct
    global step for a resumed invocation without duplicating that restore path.
    """
    checkpoint_path = Path(root) / "last_checkpoint.pt"
    if not checkpoint_path.is_file():
        return 0
    from .v4_runtime import torch_load_compat

    checkpoint = torch_load_compat(checkpoint_path, map_location="cpu", weights_only=False)
    if type(checkpoint) is not dict:
        raise ValueError("GPTrans resumed profile checkpoint is not a mapping")
    if checkpoint.get("format") != owner.CHECKPOINT_FORMAT:
        raise ValueError("GPTrans resumed profile checkpoint format changed")
    if checkpoint.get("variant", "reference") != recipe.get("mode", "reference"):
        raise ValueError("GPTrans resumed profile checkpoint variant changed")
    if checkpoint.get("source_archive_sha256") != context.source_archive_sha256:
        raise ValueError("GPTrans resumed profile checkpoint source changed")
    epoch = checkpoint.get("epoch")
    if type(epoch) is not int or epoch < 0 or epoch >= owner.EPOCHS:
        raise ValueError("GPTrans resumed profile checkpoint epoch is invalid")
    batches = getattr(owner, "BATCHES_PER_EPOCH", None)
    if type(batches) is not int or batches <= 0:
        raise ValueError("GPTrans owner exposes no valid epoch batch count")
    return (epoch + 1) * batches


def run_profiled_legacy(*, owner, recipe, root: Path, preflight: Path, context, runner):
    """Verify native proof, observe the formal owner invocation, and return it."""
    from .family_training_runner import _verify_profile_proof
    from .training_reproducibility import atomic_json

    config = _profile_config(recipe)
    if config is None:
        return runner()
    root = Path(root).absolute()
    preflight = Path(preflight).absolute()
    certificate = _read_json(preflight / "runtime_certificate.json")
    identity = SimpleNamespace(identity=canonical_fingerprint(recipe))
    _verify_profile_proof(preflight, identity, certificate.get("profile_proof"), required=True)
    verify_profile_state_proof(preflight, recipe, certificate.get("profile_state_proof"), required=True)
    starting_step = _retained_optimizer_step(owner, recipe, context, root)
    profile_root = root / "profiles" / f"step_{starting_step}_{time.time_ns()}"
    with LegacyGPTransExecutionProfiler(owner, recipe, profile_root, config,
                                        starting_optimizer_step=starting_step) as observed:
        result = runner()
    atomic_json(Path(root) / "execution_profile_receipt.json", {
        "context": context.to_dict(),
        "recipe_identity": canonical_fingerprint(recipe),
        "report_path": profile_root.relative_to(root).joinpath("profile.json").as_posix(),
        "report_sha256": _file_sha(profile_root / "profile.json"),
        "trace_sha256": observed.report["trace_sha256"],
        "capture_status": observed.report["status"],
    })
    return result


def _read_json(path: Path):
    from .experiment_family_workflow import _json
    return _json(Path(path))

