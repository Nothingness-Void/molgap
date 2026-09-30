"""Opt-in event wiring for owning trainers; no trainer or release authority.

Legacy artifacts remain unchanged. A new recipe must pin the actual trainer
and acceptance requirements before execution. Interrupted prefixes are never
reconstructed from legacy summary rows.
"""
from __future__ import annotations

import hashlib
import struct
import os
import tarfile
from pathlib import Path

from .experiment_family_workflow import (
    FamilyOutputSession, RunContext, _json, _metric_semantics, tensor_safe_rng_state,
)
from .experiment_package import _spec
from .research_memory.trace import file_digest


def acknowledged_order_digest(indices) -> str:
    """Hash the completed epoch's acknowledged permutation, not prefetched work."""
    digest = hashlib.sha256()
    for index in indices:
        digest.update(struct.pack("<q", int(index)))
    return digest.hexdigest()


def _output_namespace(output: Path, native_output: Path):
    from .experiment_launch import _safe_local
    output, native_output = Path(output).absolute(), Path(native_output).absolute()
    _safe_local(output)
    _safe_local(native_output)
    if output.resolve() != (native_output / "family_outputs").resolve():
        raise ValueError("Family outputs must use native_output/family_outputs; native files cannot be overwritten")


def _same_resume_tree(left, right) -> bool:
    """Compare translated resume state, ignoring tensor device but not values/types."""
    import torch
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
                and left.dtype == right.dtype and left.shape == right.shape
                and torch.equal(left.detach().cpu(), right.detach().cpu()))
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_resume_tree(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(_same_resume_tree(a, b) for a, b in zip(left, right))
    return left == right


def bind_training_outputs(spec, *, package_dir: Path, expected_package_identity: str,
                          arm_id: str, account: str, run_reference: str,
                          output: Path, native_output: Path, contract: Path) -> TrainingOutputHooks:
    # Check before constructing a session, which publishes files immediately.
    _output_namespace(output, native_output)
    spec = _spec(spec)
    declaration = spec.to_dict()
    if declaration["schema_version"] != "molgap-experiment-spec-v2":
        raise ValueError("Trainer output wiring requires a per-arm prospective Spec v2")
    context = RunContext.for_training(spec, package_dir,
        expected_package_identity=expected_package_identity, arm_id=arm_id,
        account=account, run_reference=run_reference)
    arm = next(a for a in declaration["arms"] if a["arm_id"] == arm_id)
    if arm["training"]["overrides"]:
        raise ValueError("Output wiring cannot apply training recipe overrides")
    adapters = {"gptrans_t": "gptrans-v1", "neural_atom_k1": "k1-v1",
                "edge_state_gps": "edge-state-v1"}
    plan = next(p for p in declaration["prospective"]["arms"] if p["arm_id"] == arm_id)
    recipe = _json(Path(contract))
    if context.family_name == "gptrans_t":
        from .gptrans_adapter import gptrans_metadata
        variant = gptrans_metadata(spec, arm_id).variant
        if variant not in {"reference", "degree_scale", "path_bond_mean"}:
            raise ValueError("Variant has no qualified new-protocol trainer wiring")
        trainer = "pcqm_gptrans_v4"
    elif context.family_name == "neural_atom_k1":
        if arm["addons"]:
            raise ValueError("Historical K1 runner cannot execute model-only registry addons")
        trainer, variant = "pcqm_k1_variants_runner", "neural_atom_k1_v4"
    else:
        # This profile supports artifacts, not an invented base/depth epoch loop.
        trainer, variant = "edge_state_training_core", "experiment_owned"
    if recipe.get("trainer_binding") != {"name": trainer, "variant": variant}:
        raise ValueError("Frozen recipe does not bind the actual owning trainer/variant")
    session = FamilyOutputSession(output, context, adapter=adapters[context.family_name],
        contract=contract, trajectory_id=plan["trajectory_id"],
        metric_semantics=_metric_semantics(recipe))
    return TrainingOutputHooks(session, trainer=trainer, variant=variant)


class TrainingOutputHooks:
    def __init__(self, session: FamilyOutputSession, *, trainer: str, variant: str):
        self.session, self.trainer, self.variant = session, trainer, variant

    @property
    def binding(self) -> dict:
        return {"context": self.session.context.to_dict(), "adapter": self.session.adapter,
                "trainer": self.trainer, "variant": self.variant}

    def validate_start(self, *, trainer: str, variant: str, source_commit: str,
                       source_archive_sha256: str, epochs: int, steps: int,
                       samples: int, development_rows: int, trajectory_id: str | None,
                       native_output: Path):
        _output_namespace(self.session.root, native_output)
        context, expected = self.session.context, self.session.expected
        if (trainer, variant, source_commit, source_archive_sha256) != (
                self.trainer, self.variant, context.source_commit, context.source_archive_sha256):
            raise ValueError("Trainer output source/variant binding mismatch")
        if trajectory_id != self.session.stage.recorder.record["trajectory_id"]:
            raise ValueError("Trainer output prospective trajectory mismatch")
        actual = dict(epochs=epochs, optimizer_steps=steps, sample_presentations=samples,
                      development_rows=development_rows, precision="fp32")
        if any(expected[k] != v for k, v in actual.items()):
            raise ValueError("Frozen output recipe differs from executable training exposure")

    def validate_resume(self, checkpoint: dict | None, *, completed_epochs: int,
                        optimizer_step: int, sample_presentations: int):
        observations = [r for r in self.session.stage.recorder.record["observations"]
                        if r["event"] == "observation"]
        if completed_epochs == 0:
            if checkpoint is not None or observations:
                raise ValueError("Existing output prefix requires explicit resume reconciliation")
            return
        if checkpoint is None or checkpoint.get("family_output_binding") != self.binding:
            raise ValueError("Resume lacks identical native checkpoint/family output binding")
        if len(observations) != completed_epochs or (
                observations[-1]["optimizer_step"], observations[-1]["sample_presentations"]) != (
                optimizer_step, sample_presentations):
            raise ValueError("Native checkpoint and retained family trace prefix disagree")
        checkpoint_path = self.session.root / "last_checkpoint.pt"
        events = [r for r in self.session.stage.recorder.record["observations"] if r["event"] == "checkpoint"]
        if not checkpoint_path.is_file() or not events or events[-1]["checkpoint_identity"] != "sha256:" + file_digest(checkpoint_path):
            raise ValueError("Retained family checkpoint hash disagrees with its acknowledged event")
        import torch
        from .v4_runtime import state_dict_sha256
        retained = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if retained.get("context") != self.session.context.to_dict() or (
                retained.get("optimizer_step"), retained.get("sample_presentations")) != (
                optimizer_step, sample_presentations):
            raise ValueError("Retained family checkpoint source/counters disagree")
        for key in ("model", "ema") if self.session.adapter == "gptrans-v1" else ("model",):
            if state_dict_sha256(retained[key]) != state_dict_sha256(checkpoint[key]):
                raise ValueError("Native/family resume model or EMA state disagrees")
        for key in ("optimizer", "scheduler", "rng_state"):
            native = tensor_safe_rng_state(checkpoint[key]) if key == "rng_state" else checkpoint[key]
            if not _same_resume_tree(retained.get(key), native):
                raise ValueError("Native/family resume " + key + " state disagrees")

    def completed_epoch(self, *, epoch: int, optimizer_step: int, sample_presentations: int,
                        train_mae_eV: float, live_dev_mae_eV: float,
                        checkpoint: dict, sampler_order_sha256: str,
                        selected: dict | None = None, ema_dev_mae_eV: float | None = None,
                        learning_rate: float | None = None, wall_time_seconds: float | None = None,
                        cumulative_wall_time_seconds: float | None = None):
        # Epoch conversion occurs at this boundary only; old native rows stay zero-based.
        if checkpoint.get("family_output_binding") != self.binding:
            raise ValueError("Native checkpoint lacks the frozen family output binding")
        self.session.epoch_finished(epoch=epoch + 1, optimizer_step=optimizer_step,
            sample_presentations=sample_presentations, live_train_metric=train_mae_eV,
            live_dev_metric=live_dev_mae_eV, ema_dev_metric=ema_dev_mae_eV,
            learning_rate=learning_rate, wall_time_seconds=wall_time_seconds,
            cumulative_wall_time_seconds=cumulative_wall_time_seconds)
        if selected is not None:
            self.session.selected(epoch=epoch + 1, optimizer_step=optimizer_step, **selected)
        self.session.checkpoint(model_state=checkpoint["model"], optimizer_state=checkpoint["optimizer"],
            scheduler_state=checkpoint.get("scheduler"), ema_state=checkpoint.get("ema"),
            rng_state=checkpoint["rng_state"], optimizer_step=optimizer_step,
            sample_presentations=sample_presentations,
            cursor={"epoch": epoch + 1, "next_batch": 0,
                    "sampler_order_sha256": sampler_order_sha256})
        recorder = self.session.stage.recorder
        recorder.checkpoint_event("sha256:" + file_digest(self.session.root / "last_checkpoint.pt"))

    def complete(self, *, hardware: str) -> dict:
        context = self.session.context
        return self.session.complete(hardware=hardware, runtime={
            "platform": context.platform, "account": context.account, "precision": "fp32",
            "source_commit": context.source_commit,
            "source_archive_sha256": context.source_archive_sha256})

    def archive_prefix(self, *, epoch: int):
        """Retain retrievable chunks beside native outputs, never a private worker cache."""
        from .experiment_launch import _safe_local
        path = self.session.root.parent / f"family_recovery_epoch_{epoch + 1:02d}.tar"
        _safe_local(path.absolute())
        temporary = path.with_suffix(".tar.tmp")
        with tarfile.open(temporary, "w") as archive:
            for source in sorted(self.session.root.iterdir()):
                _safe_local(source.absolute())
                if source.is_file():
                    archive.add(source, arcname="family_outputs/" + source.name)
        os.replace(temporary, path)
