"""CPU-only snapshot/evaluation guards around the existing family session.

No model implementation, checkpoint loader, training loop, selection or RML
writer. Real checkpoint evaluation remains a separately authorized owning action.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib

from .common.wire import canonical_fingerprint
from .cross_scale_plan import digest, exact, number


def state_digest(value):
    """Semantic tensor-state digest independent of serialization storage aliases."""
    import torch
    h = hashlib.sha256()
    def visit(item):
        if isinstance(item, torch.Tensor):
            if item.device.type != "cpu":
                raise ValueError("This infrastructure acceptance only permits CPU tensors")
            h.update(str((item.dtype, tuple(item.shape))).encode())
            h.update(item.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            h.update(b"dict")
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            h.update(type(item).__name__.encode())
            for part in item:
                visit(part)
        elif item is None or type(item) in (int, float, bool, str):
            h.update(repr((type(item).__name__, item)).encode())
        else:
            raise ValueError("Unsupported state component: " + type(item).__name__)
    visit(value)
    return h.hexdigest()


@dataclass(frozen=True)
class EvaluationPermit:
    role_identity: str
    row_sha256: str
    target_sha256: str
    authority_sha256: str
    role_history_sha256: str
    rows: int
    source_start: int
    role: str = "development"
    protected: bool = False

    def __post_init__(self):
        if (self.role != "development" or self.protected is not False or not self.role_identity.startswith("internal_development") or
                any(term in self.role_identity.lower() for term in ("official", "test", "ood", "p8", "external", "shadow", "protected"))):
            raise ValueError("ROLE_BLOCKED: only an authorized internal development role")
        for key in ("row_sha256", "target_sha256", "authority_sha256", "role_history_sha256"):
            digest(getattr(self, key), key)
        number(self.rows, "role rows", integer=True, minimum=1)
        number(self.source_start, "source start", integer=True)

    def verify(self, authority, history):
        if canonical_fingerprint(authority) != self.authority_sha256 or canonical_fingerprint(history) != self.role_history_sha256:
            raise ValueError("Role authority/use ledger tampered")
        exact(authority, {"role_identity", "row_sha256", "operations", "protected", "scope"}, "evaluation authority")
        if (authority["role_identity"] != self.role_identity or authority["row_sha256"] != self.row_sha256 or
                authority["operations"] != ["prediction_input", "labels_read", "metric_computed"] or
                authority["protected"] is not False or authority["scope"] != "CPU_SYNTHETIC_ONLY"):
            raise ValueError("ROLE_BLOCKED: no CPU synthetic evaluation authority")
        exact(history, {"role_identity", "row_sha256", "usage"}, "role history")
        if history != {"role_identity": self.role_identity, "row_sha256": self.row_sha256, "usage": "selection-used"}:
            raise ValueError("Role history mismatch; not an independent holdout")


def capture_checkpoint(session):
    """Use the existing state and cursor validator; never round up a partial pass."""
    from .family_training_session import FamilyTrainingSession, validate_session_resume
    if not isinstance(session, FamilyTrainingSession):
        raise TypeError("Existing FamilyTrainingSession required")
    state = deepcopy(session.state_dict())
    validate_session_resume(session.recipe, state)
    return state


class ExactStepSnapshotSink:
    """Compose with the owning checkpoint callback, never with a second loop.

    The frozen recipe must actually checkpoint the requested points. A missing
    point is reported, not reconstructed by evaluating the nearest epoch.
    Publication uses a caller-supplied existing atomic artifact owner.
    """
    def __init__(self, recipe, milestones, publish):
        if not callable(publish) or not milestones or list(milestones) != sorted(set(milestones)):
            raise ValueError("Trusted publisher and ordered milestone schedule required")
        for step in milestones:
            number(step, "snapshot step", integer=True, minimum=1)
            if step > recipe.optimizer_steps:
                raise ValueError("Snapshot outside frozen recipe")
        self.recipe, self.milestones, self.publish = recipe, tuple(milestones), publish
        self.observed = {}

    def __call__(self, progress, state):
        from .family_training_session import validate_session_resume
        if progress.optimizer_step not in self.milestones:
            return
        validate_session_resume(self.recipe, state)
        if (state["optimizer_step"] != progress.optimizer_step or state["sample_presentations"] != progress.sample_presentations or
                (state["cursor"]["pass_index"], state["cursor"]["next_batch"]) != (progress.pass_index, progress.next_batch)):
            raise ValueError("Snapshot progress differs from acknowledged checkpoint")
        identity = state_digest(state)
        if progress.optimizer_step in self.observed and self.observed[progress.optimizer_step] != identity:
            raise ValueError("Conflicting exact-step snapshot")
        self.publish(progress, deepcopy(state))
        self.observed[progress.optimizer_step] = identity

    def report(self):
        missing = sorted(set(self.milestones) - set(self.observed))
        return {"status": "OBSERVED" if not missing else "INCOMPLETE", "observed": dict(self.observed),
                "missing_steps": missing, "exposure_inferred": False}


def evaluate_readonly(session, checkpoint, *, permit, authority, role_history,
                      batches_factory, weights="live"):
    from .family_training_core import evaluate_development
    from .family_training_session import validate_session_resume
    from .experiment_family_workflow import tensor_digest, tensor_safe_rng_state
    from .training_reproducibility import capture_rng_state, restore_rng_state
    import torch

    if type(permit) is not EvaluationPermit or not callable(batches_factory):
        raise TypeError("Typed permit and trusted existing loader factory required")
    permit.verify(authority, role_history)  # Must precede even opening the loader.
    validate_session_resume(session.recipe, checkpoint)
    before = deepcopy(session.state_dict())
    checkpoint_sha = state_digest(checkpoint)
    before_sha = state_digest(before)
    rng = tensor_safe_rng_state(capture_rng_state())
    modes = [(module, module.training) for module in session.kernel.model.modules()]
    if any(p.device.type != "cpu" for p in session.kernel.model.parameters()):
        raise ValueError("CPU synthetic evaluator only; native owning adapter not qualified")
    if weights == "live":
        state = checkpoint["model"]
    elif weights == "ema" and checkpoint["ema"] is not None:
        state = checkpoint["ema"]
    else:
        raise ValueError("Unavailable weight state; Clean-BN requires separate calibration evidence")
    model = deepcopy(session.kernel.model)
    model.load_state_dict(state, strict=True)
    model.eval()
    try:
        with torch.no_grad():
            result = evaluate_development(model, batches_factory(),
                forward=lambda selected, batch: session.kernel.forward(batch, model=selected),
                rows=permit.rows, source_start=permit.source_start,
                mean=session.recipe.target_mean_eV, std=session.recipe.target_std_eV)
        if (tensor_digest(result["source_idx"], role="source_idx") != permit.row_sha256 or
                tensor_digest(result["target_eV"], role="target") != permit.target_sha256):
            raise ValueError("Evaluated source/target identity differs from authorized cohort")
        return {**result, "optimizer_step": checkpoint["optimizer_step"],
                "sample_presentations": checkpoint["sample_presentations"],
                "complete_passes": checkpoint["cursor"]["pass_index"],
                "next_batch": checkpoint["cursor"]["next_batch"], "weights": weights,
                "checkpoint_state_sha256": checkpoint_sha, "selection_updated": False,
                "role_usage": "selection-used", "rml_written": False}
    finally:
        restore_rng_state(rng)
        for module, mode in modes:
            module.training = mode
        if state_digest(checkpoint) != checkpoint_sha or state_digest(session.state_dict()) != before_sha:
            raise RuntimeError("Read-only evaluation mutated checkpoint/session state")
