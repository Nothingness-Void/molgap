"""Opt-in observation of the actual shared session in an exclusive worker.

No model, update kernel, sampler or persistence algorithm is duplicated here.
The caller owns admission, data roles, a bounded run and its evidence closure.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
import threading
import time


_OWNER = threading.Lock()
_MISSING = object()


def profile_artifact_refs(root):
    """Pin only known observer/proof outputs for selective platform retrieval."""
    from .experiment_launch import _safe_local
    from .training_reproducibility import sha256_file

    root = Path(root).absolute()
    _safe_local(root)
    names = ("execution_profile_receipt.json", "runtime_certificate.json",
             "preflight.json", "runtime_manifest.json", "runtime_provenance.json",
             "repeat-a.pt", "repeat-b.pt", "resumed.pt", "resumed-first.pt",
             "fill-enabled.pt")
    paths = [root / name for name in names if (root / name).is_file()]
    for directory in ("profiles", "profile_qualification"):
        folder = root / directory
        _safe_local(folder)
        if folder.exists():
            for path in folder.rglob("*"):
                _safe_local(path)
                if path.is_file() and path.suffix in {".json", ".pt", ".txt"}:
                    paths.append(path)
    return [{"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)}
            for path in sorted(set(paths))]


@dataclass(frozen=True)
class ProfileConfig:
    warmup_steps: int = 2
    active_steps: int = 8
    record_shapes: bool = True
    profile_memory: bool = False
    with_stack: bool = False
    row_limit: int = 50
    module_paths: tuple[str, ...] = ()
    ppgt_phases: bool = False

    def __post_init__(self):
        for name, minimum in (("warmup_steps", 0), ("active_steps", 1), ("row_limit", 1)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError("Invalid profile setting: " + name)
        for name in ("record_shapes", "profile_memory", "with_stack", "ppgt_phases"):
            if type(getattr(self, name)) is not bool:
                raise ValueError("Expected profile boolean: " + name)
        if (type(self.module_paths) is not tuple or
                any(type(path) is not str or not path for path in self.module_paths) or
                len(set(self.module_paths)) != len(self.module_paths)):
            raise ValueError("Expected unique, nonempty module paths")

    @classmethod
    def from_dict(cls, value):
        if type(value) is not dict or set(value) - set(cls.__dataclass_fields__):
            raise ValueError("Unknown profiler configuration fields")
        value = dict(value)
        if "module_paths" in value:
            if type(value["module_paths"]) is not list:
                raise ValueError("JSON module_paths must be a list")
            value["module_paths"] = tuple(value["module_paths"])
        return cls(**value)


class _ObservedBatches:
    def __init__(self, source, owner):
        self.source, self.owner = source, owner

    def __len__(self):
        return len(self.source)

    def __iter__(self):
        capture_off = self.owner._finished
        if capture_off:
            self.owner._synchronize()
            self.owner._span = (self.owner._updates, len(self.source), time.perf_counter())
        self.owner._begin_batch()
        with self.owner.phase("loader_start"):
            iterator = iter(self.source)
        try:
            while True:
                self.owner._begin_batch()
                try:
                    with self.owner.phase("loader_wait"):
                        batch = next(iterator)
                except StopIteration:
                    return
                yield batch
        finally:
            self.owner._span = None
            close = getattr(iterator, "close", None)
            if close is not None:
                close()


class _ExecutionCapture:
    """Temporarily observe a session; the ordinary trainer has no added hooks.

    Use in a dedicated, single-training-thread worker. Instance hooks and the
    update observer retain the existing ABI without replacing global backward,
    clipping or transport functions.
    Warmup uses the existing run's updates; profiling never adds an update.
    Ending the capture does not stop the run or alter its checkpoint cadence.
    """

    def __init__(self, session, output, config=ProfileConfig()):
        if type(config) is not ProfileConfig:
            raise TypeError("Expected ProfileConfig")
        self.session, self.output, self.config = session, Path(output), config
        self._used = self._active = self._finished = False
        self._prof = None
        self._updates = self._captured = 0
        self._phases, self._stack = {}, []
        self._elapsed = None
        self._events = []
        self._thread = None
        self._train_spans = []
        self._span = None

    def _after_update(self):
        self._updates += 1
        if self._active:
            self._captured += 1
        if self._span is not None:
            updates, expected, began = self._span
            if self._updates - updates == expected:
                self._synchronize()
                self._retain_train_span(updates, time.perf_counter() - began)
                self._span = None

    def _retain_train_span(self, updates, elapsed):
        from .training_reproducibility import atomic_json
        steps = self._updates - updates
        self._train_spans.append({"start_relative_update": updates,
            "optimizer_updates": steps, "sample_presentations": steps * self.session.recipe.batch_size,
            "wall_seconds": elapsed, "scope": "first loader request through last update return; capture off; observer wrappers installed"})
        atomic_json(self.output / "training_spans.json", {"format": "molgap-capture-off-training-spans-v1",
            "recipe_identity": self.session.recipe.identity, "spans": self._train_spans})

    @contextmanager
    def phase(self, label):
        if not self._active or threading.get_ident() != self._thread:
            yield
            return
        import torch
        frame = [time.perf_counter(), 0.0]
        self._stack.append(frame)
        try:
            with torch.profiler.record_function("molgap/" + label):
                yield
        finally:
            elapsed = time.perf_counter() - frame[0]
            self._stack.pop()
            if self._stack:
                self._stack[-1][1] += elapsed
            row = self._phases.setdefault(label, {"calls": 0, "host_total_seconds": 0.0,
                "host_self_seconds": 0.0})
            row["calls"] += 1
            row["host_total_seconds"] += elapsed
            row["host_self_seconds"] += max(0.0, elapsed - frame[1])

    def _replace(self, obj, name, replacement):
        original = vars(obj).get(name, _MISSING)
        def restore():
            if original is _MISSING:
                delattr(obj, name)
            else:
                setattr(obj, name, original)
        setattr(obj, name, replacement)
        self._cleanup.callback(restore)

    def _wrap(self, obj, name, label):
        original = getattr(obj, name)
        @wraps(original)
        def observed(*args, **kwargs):
            with self.phase(label):
                return original(*args, **kwargs)
        self._replace(obj, name, observed)

    def _observe_module(self, module, label):
        # Keep the forward ABI intact: native cache dispatch inspects it.
        scopes = threading.local()
        def before(*args):
            scope = self.phase(label)
            scope.__enter__()
            if not hasattr(scopes, "stack"):
                scopes.stack = []
            scopes.stack.append(scope)
        def after(*args):
            stack = getattr(scopes, "stack", ())
            if stack:
                stack.pop().__exit__(None, None, None)
        self._cleanup.callback(module.register_forward_pre_hook(before, prepend=True).remove)
        self._cleanup.callback(module.register_forward_hook(after, always_call=True).remove)

    def _install(self):
        import torch
        session, kernel = self.session, self.session.kernel
        for obj, name, label in (
            (kernel.model, "forward", "forward"),
            (kernel, "evaluate", "development"),
            (session, "_flush_metrics", "metric_sync"),
            (session, "selected", "selected_publish"),
            (session, "pass_finished", "trace_publish"),
            (session, "_finish_pass", "pass_end"),
            (session.schedule, "before_step", "schedule_before"),
            (session.schedule, "after_step", "schedule_after"),
        ):
            self._wrap(obj, name, label)
        if getattr(session, "checkpoint_builder", None) is None:
            self._wrap(session, "state_dict", "checkpoint_state")
            self._wrap(session, "checkpoint", "checkpoint_publish")
        else:
            self._wrap(session, "_checkpoint_state", "checkpoint_state")
            self._wrap(session, "checkpoint_builder", "checkpoint_publish")
        from .family_training_core import UpdateObserver
        if getattr(kernel, "update_observer", None) is not None:
            raise ValueError("Profiler requires an unobserved update owner")
        self._replace(kernel, "update_observer", UpdateObserver(scope=self.phase))
        from .training_reproducibility import observe_finite_validation
        self._cleanup.enter_context(observe_finite_validation(self.phase))
        if kernel.recipe.family == "gptrans_t":
            self._wrap(kernel, "_gptrans_batch", "layout_and_transfer")
            if hasattr(kernel.model, "_shortest_path"):
                self._wrap(kernel.model, "_shortest_path", "shortest_path")
        if getattr(kernel.recipe, "integration_preview", False):
            self._wrap(kernel, "_model_batch", "topology_transfer")
        for path in self.config.module_paths:
            self._observe_module(kernel.model.get_submodule(path), "module/" + path)
        if self.config.ppgt_phases:
            self._install_ppgt_phases(kernel.model)
        original_step, original_loader = kernel.step, session.loaders.train
        @wraps(original_step)
        def step(*args, **kwargs):
            with self.phase("train_update"):
                result = original_step(*args, **kwargs)
            self._after_update()
            return result
        @wraps(original_loader)
        def loader(*args, **kwargs):
            with self.phase("loader_setup"):
                source = original_loader(*args, **kwargs)
            return _ObservedBatches(source, self)
        self._replace(kernel, "step", step)
        self._replace(session.loaders, "train", loader)

    def _install_ppgt_phases(self, model):
        # Only the exclusive profiler owns these temporary module-level wraps.
        # Observe the existing implementation, never substitute a faster formula.
        from . import ppgt_author, ppgt_author_features
        if type(model) is not ppgt_author.PPGTAuthor:
            raise ValueError("PPGT phase capture requires the unchanged PPGTAuthor model")
        for obj, name, label in (
            (ppgt_author, "validated_topology_views", "ppgt/topology_views"),
            (model, "graph_forward", "ppgt/graph_forward"),
            (ppgt_author, "positional_inputs", "ppgt/positional_inputs"),
            (ppgt_author_features, "rrwp", "ppgt/rrwp"),
            (ppgt_author, "sinusoidal_pe", "ppgt/sinusoidal_pe"),
        ):
            self._wrap(obj, name, label)
        self._observe_module(model.pe_mlp, "ppgt/pe_mlp")
        for block in model.stem:
            self._observe_module(block, "ppgt/pair_stem")
        for block in model.blocks:
            self._observe_module(block, "ppgt/backbone")
            self._observe_module(block.attention, "ppgt/attention")
            self._observe_module(block.FFN, "ppgt/ffn")
        self._observe_module(model.head, "ppgt/readout")

    def _synchronize(self):
        if self.session.kernel.device.type == "cuda":
            import torch
            torch.cuda.synchronize(self.session.kernel.device)

    def _begin_batch(self):
        if self._finished:
            return
        if self._active and self._captured >= self.config.active_steps:
            self._finish_capture()
        elif not self._active and self._updates >= self.config.warmup_steps:
            import torch
            self._synchronize()
            self._prof = torch.profiler.profile(activities=self._activities,
                record_shapes=self.config.record_shapes, profile_memory=self.config.profile_memory,
                with_stack=self.config.with_stack)
            self._prof.__enter__()
            self._stamp = time.perf_counter()
            self._active = True

    def _finish_capture(self):
        if self._prof is None or self._finished:
            return
        self._active = False
        self._finished = True
        try:
            self._synchronize()
            self._elapsed = time.perf_counter() - self._stamp
        finally:
            self._prof.__exit__(None, None, None)
        self._events = list(self._prof.events())
        self._prof.export_chrome_trace(str(self.output / "trace.json"))

    def __enter__(self):
        import torch
        if self._used or self.session.closed:
            raise RuntimeError("Profiler needs a fresh context and an open session")
        if not _OWNER.acquire(blocking=False):
            raise RuntimeError("Only one shared-session profiler per process")
        self._cleanup = ExitStack()
        self._cleanup.callback(_OWNER.release)
        self._used, self._thread = True, threading.get_ident()
        try:
            device = self.session.kernel.device
            self._activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                if torch.profiler.ProfilerActivity.CUDA not in torch.profiler.supported_activities():
                    raise RuntimeError("CUDA profiler activity unavailable in this runtime")
                self._activities.append(torch.profiler.ProfilerActivity.CUDA)
            elif device.type != "cpu":
                raise ValueError("Profiler supports CPU or the CUDA/HIP Torch backend")
            for path in self.config.module_paths:
                self.session.kernel.model.get_submodule(path)
            self.output.mkdir(parents=True, exist_ok=False)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            self._start_step = self.session.progress.optimizer_step
            self._install()
            return self
        except BaseException:
            self._cleanup.close()
            raise

    def _report(self, error):
        import torch
        from .training_reproducibility import atomic_json, sha256_file
        cuda_observed = any(str(event.device_type).endswith("CUDA") for event in self._events)
        averages = list(self._prof.key_averages(group_by_input_shape=self.config.record_shapes)) if self._events else []
        operators = []
        for event in averages:
            if event.key.startswith("molgap/"):
                continue
            device_time = getattr(event, "self_device_time_total", None)
            if device_time is None:
                device_time = getattr(event, "self_cuda_time_total", None)
            operators.append({"name": event.key, "calls": event.count,
                "self_cpu_us": event.self_cpu_time_total,
                "self_device_us": device_time if cuda_observed else None,
                "input_shapes": event.input_shapes})
        phases = [{"name": name, **row} for name, row in self._phases.items()]
        completed = self._captured == self.config.active_steps
        self.report = {"format": "molgap-family-execution-profile-v1",
            "status": "ERROR" if error else "COMPLETE" if completed else "PARTIAL",
            "error": None if error is None else {"type": type(error).__name__, "message": str(error)},
            "recipe_identity": self.session.recipe.identity, "family": self.session.recipe.family,
            "config": asdict(self.config), "starting_optimizer_step": self._start_step,
            "observed_updates": self._updates, "captured_updates": self._captured,
            "capture_start_optimizer_step": self._start_step + self.config.warmup_steps if self._prof else None,
            "capture_wall_seconds": self._elapsed,
            "capture_off_training_spans": self._train_spans,
            "runtime": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                "hip": torch.version.hip, "device": str(self.session.kernel.device),
                "device_name": torch.cuda.get_device_name(self.session.kernel.device)
                    if self.session.kernel.device.type == "cuda" else None,
                "cpu_threads": torch.get_num_threads()},
            "device_events_observed": cuda_observed,
            "context_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.session.kernel.device)
                if self.session.kernel.device.type == "cuda" else None,
            "context_peak_reserved_bytes": torch.cuda.max_memory_reserved(self.session.kernel.device)
                if self.session.kernel.device.type == "cuda" else None,
            "phases_by_host_self_time": sorted(phases, key=lambda row: row["host_self_seconds"], reverse=True),
            "operators_by_self_cpu": sorted(operators, key=lambda row: row["self_cpu_us"], reverse=True)[:self.config.row_limit],
            "operators_by_self_device": sorted(operators, key=lambda row: row["self_device_us"] or 0,
                reverse=True)[:self.config.row_limit] if cuda_observed else [],
            "trace_sha256": sha256_file(self.output / "trace.json") if (self.output / "trace.json").is_file() else None,
            "limitations": ["Instrumented timings are not unprofiled throughput or billed GPU time.",
                "Host phases include Python and waits; GPU kernels run asynchronously.",
                "Nested inclusive phase times must not be added; host self time excludes observed children.",
                "Device kernel durations may overlap; their sum is not GPU busy wall time.",
                "Unvisited phases have no measurement; missing device events are unknown, not zero.",
                "Update counts are successful kernel returns; a failing update may have partially changed state.",
                "DataLoader subprocess work is seen as main-process wait, not worker CPU attribution.",
                "The caller owns real-input authority, provenance, stop budget and terminal evidence."]}
        if getattr(self.session.recipe, "integration_preview", False):
            self.report.update(integration_status="INTEGRATION_PENDING",
                execution_scope="INTEGRATION_PREVIEW_ONLY", native_execution_released=False)
            self.report["limitations"].append(
                "Modern2D session preview observations do not qualify family fidelity, real inputs or native training.")
        atomic_json(self.output / "profile.json", self.report)

    def __exit__(self, exc_type, error, traceback):
        try:
            self._finish_capture()
            self._report(error)
        except BaseException as diagnostic_error:
            if error is None:
                raise
            if hasattr(error, "add_note"):
                error.add_note("Profiler finalization also failed: " + repr(diagnostic_error))
        finally:
            self._cleanup.close()
        return False


class FamilyExecutionProfiler(_ExecutionCapture):
    """Observe the owning shared session without changing its training loop."""

    def __init__(self, session, output, config=ProfileConfig()):
        from .family_training_session import FamilyTrainingSession
        if not isinstance(session, FamilyTrainingSession):
            raise TypeError("Expected a shared session")
        super().__init__(session, output, config)
