"""Opt-in Live/Raw-EMA/Clean-EMA state actions for a trusted family trainer.

Calibration changes a dedicated Clean model in place. It never changes existing
recipes or role authority; a future contract must explicitly bind this policy.
The owning session still checkpoints optimizer, schedule, cursor and RNG.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from ..screen_policy import canonical_fingerprint


STATE_FORMAT = "molgap-family-training-state-v1"
CALIBRATION_FORMAT = "molgap-clean-bn-policy-v1"


def _digest(value, name):
    if (type(value) is not str or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise ValueError("Expected SHA256: " + name)


@dataclass(frozen=True)
class CalibrationPolicy:
    input_identity: str
    training_role_identity: str
    source_order_sha256: str
    batches: int
    batch_size: int = 128
    role: str = "train"
    estimator: str = "cumulative-batch"

    def __post_init__(self):
        for name in ("input_identity", "training_role_identity", "source_order_sha256"):
            _digest(getattr(self, name), name)
        if self.role != "train":
            raise ValueError("Clean-BN consumes only an explicitly bound training role")
        if self.estimator != "cumulative-batch":
            raise ValueError("Unsupported Clean-BN statistics estimator")
        if type(self.batches) is not int or not 1 <= self.batches <= 10000:
            raise ValueError("Calibration requires a finite positive batch budget")
        if type(self.batch_size) is not int or not 2 <= self.batch_size <= 8192:
            raise ValueError("Calibration requires a fixed physical batch boundary")

    def to_dict(self):
        return {"format": CALIBRATION_FORMAT, **asdict(self)}

    @property
    def identity(self):
        return canonical_fingerprint(self.to_dict())

    @classmethod
    def from_dict(cls, value):
        expected = {"format", "input_identity", "training_role_identity", "source_order_sha256",
                    "batches", "batch_size", "role", "estimator"}
        if type(value) is not dict or set(value) != expected or value["format"] != CALIBRATION_FORMAT:
            raise ValueError("Calibration policy schema changed")
        return cls(**{name: item for name, item in value.items() if name != "format"})


def source_order_sha256(indices):
    """Hash exact signed-int64 row order; never silently cast a float identity."""
    import hashlib
    import torch
    if (not isinstance(indices, torch.Tensor) or indices.ndim != 1 or
            indices.dtype != torch.int64 or not indices.numel() or bool((indices < 0).any())):
        raise ValueError("Calibration source indices must be a nonnegative int64 vector")
    values = indices.detach().cpu().contiguous().numpy().astype("<i8", copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


def _copy_state(state, *, cpu=False):
    return {name: value.detach().cpu().clone() if cpu else value.detach().clone()
            for name, value in state.items()}


def _validate_state(state, expected, label):
    import torch
    if type(state) is not dict or state.keys() != expected.keys() or not state:
        raise ValueError(label + " tensor keys changed")
    for name, value in state.items():
        target = expected[name]
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype or
                ((value.is_floating_point() or value.is_complex()) and not bool(torch.isfinite(value).all()))):
            raise ValueError(label + " tensor schema/nonfinite state: " + name)


def _state_identity(state):
    from ..v4_runtime import state_dict_sha256
    return state_dict_sha256(state)


def validate_training_state(state, live, *, raw_available, calibration_policy=None, allow_missing_clean=False):
    """Pure checkpoint-component validation shared by sessions and state restore."""
    expected = {"format", "live", "raw_ema", "raw_revision", "clean"}
    if (type(state) is not dict or set(state) != expected or state["format"] != STATE_FORMAT or
            type(state["raw_revision"]) is not int or state["raw_revision"] < 0 or
            (state["raw_ema"] is None) != (not raw_available)):
        raise ValueError("Training state schema/policy changed")
    _validate_state(state["live"], live, "Saved Live")
    if state["raw_ema"] is not None:
        _validate_state(state["raw_ema"], live, "Saved Raw EMA")
    clean = state["clean"]
    if clean is not None:
        if (type(clean) is not dict or set(clean) != {"model", "calibration", "stale"} or
                type(clean["stale"]) is not bool or state["raw_ema"] is None):
            raise ValueError("Saved Clean EMA schema changed")
        _validate_state(clean["model"], live, "Saved Clean EMA")
        record = clean["calibration"]
        keys = {"policy", "policy_identity", "raw_revision", "raw_state_identity", "clean_state_identity", "identity"}
        if type(record) is not dict or set(record) != keys:
            raise ValueError("Saved calibration record changed")
        policy = CalibrationPolicy.from_dict(record["policy"])
        if (record["policy_identity"] != policy.identity or
                type(record["raw_revision"]) is not int or not 0 <= record["raw_revision"] <= state["raw_revision"] or
                record["clean_state_identity"] != _state_identity(clean["model"]) or
                record["identity"] != canonical_fingerprint({key: value for key, value in record.items() if key != "identity"})):
            raise ValueError("Saved calibration identity changed")
        _digest(record["raw_state_identity"], "raw_state_identity")
        stale = (record["raw_revision"] != state["raw_revision"] or
                 record["raw_state_identity"] != _state_identity(state["raw_ema"]))
        if stale != clean["stale"]:
            raise ValueError("Saved calibration staleness changed")
        if calibration_policy is not None and (
                type(calibration_policy) is not CalibrationPolicy or calibration_policy.identity != policy.identity):
            raise ValueError("Saved calibration differs from requested policy")
    elif calibration_policy is not None and not allow_missing_clean:
        raise ValueError("Saved checkpoint has no requested Clean EMA")


class FamilyTrainingState:
    """Exclusive-worker state owner; JSON cannot select factory or forward hooks.

    ``clean_factory`` creates an independent, same-schema model using the owning
    family factory/layout. No model copy is allocated until Clean is requested.
    Call ``update_raw_ema`` after the existing optimizer step, never in addition
    to an EMA update already performed by an existing kernel.
    """
    def __init__(self, live_model, raw_ema=None, *, clean_factory=None, weight_view=None):
        import torch
        from ..gptrans_execution_hotpath import ReusedWeightView
        if not isinstance(live_model, torch.nn.Module):
            raise TypeError("Live state requires a model")
        if clean_factory is not None and not callable(clean_factory):
            raise TypeError("Clean model factory must be a trusted callable")
        self.live_model, self.raw_ema = live_model, raw_ema
        self.clean_factory, self.clean_model = clean_factory, None
        self.weight_view = weight_view if weight_view is not None else ReusedWeightView()
        self.raw_revision, self.calibration = 0, None
        _validate_state(dict(live_model.state_dict()), live_model.state_dict(), "Live")
        if raw_ema is not None:
            _validate_state(dict(raw_ema.state_dict()), live_model.state_dict(), "Raw EMA")

    def _raw_state(self):
        if self.raw_ema is None:
            raise ValueError("Raw EMA is unavailable")
        state = dict(self.raw_ema.state_dict())
        _validate_state(state, self.live_model.state_dict(), "Raw EMA")
        return state

    def _new_clean_model(self):
        import torch
        if self.clean_factory is None:
            raise ValueError("Clean EMA requires an explicit independent model factory")
        model = self.clean_factory()
        if not isinstance(model, torch.nn.Module) or model is self.live_model:
            raise ValueError("Clean factory must return an independent model")
        live = self.live_model.state_dict()
        devices = {value.device for value in live.values()}
        if len(devices) != 1:
            raise ValueError("Clean state requires one explicitly owned device")
        model.to(next(iter(devices)))
        _validate_state(dict(model.state_dict()), live, "Clean factory")
        def storage_key(value):
            # Views can begin at different offsets in one overlapping storage.
            return value.device, value.untyped_storage().data_ptr()
        owned = {storage_key(value) for value in (*self.live_model.parameters(), *self.live_model.buffers())
                 if value.numel()}
        if self.raw_ema is not None:
            owned.update(storage_key(value) for value in self._raw_state().values() if value.numel())
        if any(value.numel() and storage_key(value) in owned
               for value in (*model.parameters(), *model.buffers())):
            raise ValueError("Clean factory aliases Live/Raw tensor storage")
        return model

    @property
    def clean_status(self):
        if self.clean_model is None or self.calibration is None:
            return "missing"
        if (self.raw_revision != self.calibration["raw_revision"] or
                _state_identity(self._raw_state()) != self.calibration["raw_state_identity"]):
            return "stale"
        return "ready"

    def update_raw_ema(self):
        if self.raw_ema is None:
            raise ValueError("Raw EMA is unavailable")
        self.raw_ema.update(self.live_model)
        self.raw_revision += 1

    def calibrate(self, policy, batches, *, forward, source_indices):
        """Native bounded BN action; successful Clean buffers remain installed.

        Only the declared prefix is consumed. Its source order and every batch
        boundary must match. Callers own admission, loader private RNG and data
        immutability; global RNG and model modes are restored here.
        """
        import torch
        from copy import deepcopy
        from torch.nn.modules.batchnorm import _BatchNorm
        from ..training_reproducibility import capture_rng_state, restore_rng_state
        if type(policy) is not CalibrationPolicy or not callable(forward) or not callable(source_indices):
            raise TypeError("Typed policy and trusted forward/source-index hooks are required")
        raw = self._raw_state()
        raw_identity = _state_identity(raw)
        raw_revision = self.raw_revision
        rng = capture_rng_state()
        previous_model, previous_calibration = self.clean_model, self.calibration
        model, previous = None, None
        modes, momenta = [], []
        iterator = None
        def close_iterator():
            nonlocal iterator
            current, iterator = iterator, None
            close = getattr(current, "close", None)
            if close is not None:
                close()
        try:
            model = previous_model if previous_model is not None else self._new_clean_model()
            previous = _copy_state(model.state_dict())
            modes = [(module, module.training) for module in model.modules()]
            bn = [(name, module) for name, module in model.named_modules()
                  if isinstance(module, _BatchNorm) and module.track_running_stats]
            if not bn:
                raise ValueError("Clean-BN requires tracked BatchNorm buffers")
            momenta = [(module, module.momentum) for _, module in bn]
            model.load_state_dict(raw, strict=True)
            model.eval()
            for _, module in bn:
                module.reset_running_stats()
                module.momentum, module.training = None, True
            order = []
            iterator = iter(batches)
            with torch.no_grad():
                for _ in range(policy.batches):
                    try:
                        batch = next(iterator)
                    except StopIteration as error:
                        raise ValueError("Calibration ended before its declared batch budget") from error
                    indices = source_indices(batch)
                    source_order_sha256(indices)
                    if indices.numel() != policy.batch_size or (
                            hasattr(batch, "num_graphs") and int(batch.num_graphs) != policy.batch_size):
                        raise ValueError("Calibration batch boundary differs from policy")
                    order.append(indices.detach().cpu().clone())
                    result = forward(model, batch)
                    if (not isinstance(result, torch.Tensor) or result.numel() != policy.batch_size or
                            not bool(torch.isfinite(result).all())):
                        raise ValueError("Calibration forward shape/nonfinite output")
            # Stream cleanup is part of the action: failure cannot publish Clean.
            close_iterator()
            if source_order_sha256(torch.cat(order)) != policy.source_order_sha256:
                raise ValueError("Calibration source order differs from policy")
            if any(int(module.num_batches_tracked) < 1 for _, module in bn):
                raise ValueError("Calibration did not reach every tracked BatchNorm")
            allowed = {".".join(filter(None, (name, key))) for name, _ in bn
                       for key in ("running_mean", "running_var", "num_batches_tracked")}
            calibrated = dict(model.state_dict())
            _validate_state(calibrated, raw, "Calibrated")
            if any(not torch.equal(value, raw[name]) for name, value in calibrated.items() if name not in allowed):
                raise ValueError("Calibration changed tensors outside BatchNorm buffers")
            if self.raw_revision != raw_revision or _state_identity(self._raw_state()) != raw_identity:
                raise ValueError("Raw EMA changed during calibration")
            record = {"policy": policy.to_dict(), "policy_identity": policy.identity,
                      "raw_revision": raw_revision, "raw_state_identity": raw_identity,
                      "clean_state_identity": _state_identity(calibrated)}
            record["identity"] = canonical_fingerprint(record)
            self.clean_model, self.calibration = model, record
            return deepcopy(record)
        except Exception:
            if model is not None and previous is not None:
                model.load_state_dict(previous, strict=True)
            self.clean_model, self.calibration = previous_model, previous_calibration
            raise
        finally:
            try:
                for module, training in modes:
                    module.training = training
                for module, momentum in momenta:
                    module.momentum = momentum
                close_iterator()
            finally:
                restore_rng_state(rng)

    def evaluate(self, weights, evaluator, *, calibration_policy=None):
        """Call an owning evaluator; Clean requires a matching fresh policy."""
        import torch
        if not callable(evaluator):
            raise TypeError("An owning evaluator is required")
        if weights == "clean-ema":
            if (type(calibration_policy) is not CalibrationPolicy or self.clean_status != "ready" or
                    calibration_policy.identity != self.calibration["policy_identity"]):
                raise ValueError("Clean EMA is missing, stale or differently calibrated")
            model = self.clean_model
            if _state_identity(model.state_dict()) != self.calibration["clean_state_identity"]:
                raise ValueError("Clean EMA tensors changed after calibration")
            modes = [(module, module.training) for module in model.modules()]
            try:
                model.eval()
                with torch.no_grad():
                    result = evaluator(model)
                if _state_identity(model.state_dict()) != self.calibration["clean_state_identity"]:
                    raise ValueError("Evaluation changed calibrated Clean EMA tensors")
                return result
            finally:
                for module, training in modes:
                    module.training = training
        if weights not in {"live", "raw-ema"}:
            raise ValueError("Unknown model weight state")
        if weights == "raw-ema":
            self._raw_state()
        modes = [(module, module.training) for module in self.live_model.modules()]
        try:
            with self.weight_view.use(self.live_model, self.raw_ema, "live" if weights == "live" else "ema"):
                self.live_model.eval()
                with torch.no_grad():
                    return evaluator(self.live_model)
        finally:
            for module, training in modes:
                module.training = training

    def calibrate_and_evaluate(self, policy, batches, *, forward, source_indices, evaluator):
        self.calibrate(policy, batches, forward=forward, source_indices=source_indices)
        return self.evaluate("clean-ema", evaluator, calibration_policy=policy)

    def calibration_state_dict(self):
        """Session metadata; Live/Raw tensors already live in its resume envelope."""
        from copy import deepcopy
        clean = None
        if self.clean_model is not None:
            current = dict(self.clean_model.state_dict())
            _validate_state(current, self.live_model.state_dict(), "Clean EMA checkpoint")
            if _state_identity(current) != self.calibration["clean_state_identity"]:
                raise ValueError("Clean EMA tensors changed before checkpoint")
            clean = {"model": _copy_state(self.clean_model.state_dict(), cpu=True),
                     "calibration": deepcopy(self.calibration), "stale": self.clean_status != "ready"}
        return {"format": STATE_FORMAT, "raw_revision": self.raw_revision, "clean": clean}

    def state_dict(self):
        return {**self.calibration_state_dict(),
                "live": _copy_state(self.live_model.state_dict(), cpu=True),
                "raw_ema": _copy_state(self._raw_state(), cpu=True) if self.raw_ema is not None else None}

    def load_state_dict(self, state, *, calibration_policy=None):
        """Validate first; restore Live/Raw/Clean atomically on loading failure."""
        from copy import deepcopy
        from ..training_reproducibility import capture_rng_state, restore_rng_state
        live = dict(self.live_model.state_dict())
        validate_training_state(state, live, raw_available=self.raw_ema is not None,
                                calibration_policy=calibration_policy)
        clean = state["clean"]
        rng = capture_rng_state()
        previous_live = _copy_state(live)
        previous_raw = _copy_state(self._raw_state()) if self.raw_ema is not None else None
        previous_clean, previous_record, previous_revision = self.clean_model, self.calibration, self.raw_revision
        try:
            # Prepare an independent destination before changing live training state.
            clean_model = self._new_clean_model() if clean is not None else None
            if clean_model is not None:
                import torch
                from torch.nn.modules.batchnorm import _BatchNorm
                bn = [(name, module) for name, module in clean_model.named_modules()
                      if isinstance(module, _BatchNorm) and module.track_running_stats]
                if not bn:
                    raise ValueError("Saved Clean-BN model has no tracked BatchNorm")
                if any(int(clean["model"][".".join(filter(None, (name, "num_batches_tracked")))]) < 1
                       for name, _ in bn):
                    raise ValueError("Saved Clean-BN buffers have no calibration exposure")
                allowed = {".".join(filter(None, (name, key))) for name, _ in bn
                           for key in ("running_mean", "running_var", "num_batches_tracked")}
                if not clean["stale"] and any(not torch.equal(value, state["raw_ema"][name])
                        for name, value in clean["model"].items() if name not in allowed):
                    raise ValueError("Saved Clean-BN differs from Raw EMA outside BN buffers")
                clean_model.load_state_dict(clean["model"], strict=True)
            self.live_model.load_state_dict(state["live"], strict=True)
            if self.raw_ema is not None:
                self.raw_ema.load_state_dict(state["raw_ema"])
            self.clean_model, self.calibration = clean_model, deepcopy(clean["calibration"]) if clean is not None else None
            self.raw_revision = state["raw_revision"]
        except Exception:
            self.live_model.load_state_dict(previous_live, strict=True)
            if self.raw_ema is not None:
                self.raw_ema.load_state_dict(previous_raw)
            self.clean_model, self.calibration, self.raw_revision = previous_clean, previous_record, previous_revision
            raise
        finally:
            restore_rng_state(rng)

    def save(self, path):
        from ..training_reproducibility import atomic_torch_save
        atomic_torch_save(Path(path), self.state_dict())

    def load(self, path, *, calibration_policy=None):
        import torch
        self.load_state_dict(torch.load(Path(path), map_location="cpu", weights_only=True),
                             calibration_policy=calibration_policy)

# Retained pickles name the historical module even though execution lives here.
for _compat_object in (
    _digest, CalibrationPolicy, source_order_sha256, _copy_state, _validate_state,
    _state_identity, validate_training_state, FamilyTrainingState,
):
    _compat_object.__module__ = "molgap.family_training_state"
del _compat_object
