"""Session artifact translation using existing atomic IO, trace and inspection owners."""
from __future__ import annotations

import math
import os
import time
from copy import deepcopy
from pathlib import Path

from .experiment_family_workflow import (
    OUTPUT_FORMAT, ROLES, StageRecorder, _artifact_path, _check_context, _costs, _digest,
    _finite_tree, _inspect_predictions, _json, _publish, _runtime, write_output_manifest,
    write_resume_state,
)
from .experiment_launch import _safe_local, publish_immutable_bytes
from .research_memory.trace import atomic_write, file_digest, json_bytes, sync_directory, validate_canonical_trace
from .family_training_session import validate_session_resume
from .training_reproducibility import atomic_torch_save


_COMMIT_FILES = {"selected_model.pt", "development_predictions.pt", "canonical_trace.json", "last_checkpoint.pt"}


def _recover_commit(root, context):
    """Recover only a hash-bound producer transaction, never truncate evidence."""
    companion = root / "canonical_trace.json.context.json"
    journal = root / "session_commit.json"
    contract = root / "training_contract.json"
    for path in (companion, journal, contract, root / "session_pending"):
        _safe_local(path.absolute())
    if contract.exists() and file_digest(_artifact_path(root, contract.name)) != context.training_recipe_sha256:
        raise ValueError("Retained session contract differs from context")
    if companion.exists():
        _check_context(_json(_artifact_path(root, companion.name)).get("context"), context)
    if journal.exists() and not journal.is_file():
        raise ValueError("Session commit journal is not a file")
    if not journal.exists():
        if not companion.exists() and any(
                (root / name).exists() or (root / "session_pending" / name).exists()
                or (root / "session_pending" / (name + ".previous")).exists() for name in _COMMIT_FILES):
            raise ValueError("Pending session artifact identity is unavailable; preserve for reconciliation")
        return
    record = _json(_artifact_path(root, journal.name))
    if set(record) != {"format", "context", "entries"} or record["format"] != "molgap-session-commit-v1":
        raise ValueError("Invalid session commit journal")
    _check_context(record["context"], context)
    entries = record["entries"]
    if (type(entries) is not list or not entries or any(type(entry) is not dict for entry in entries)
            or entries[-1].get("name") != "last_checkpoint.pt"
            or len({entry.get("name") for entry in entries}) != len(entries)):
        raise ValueError("Invalid session commit ordering")
    for entry in entries:
        if type(entry) is not dict or set(entry) != {"name", "previous_sha256", "next_sha256"} or entry["name"] not in _COMMIT_FILES:
            raise ValueError("Invalid session commit artifact")
        _digest(entry["next_sha256"], "next_sha256")
        if entry["previous_sha256"] is not None:
            _digest(entry["previous_sha256"], "previous_sha256")
        for path in (root / entry["name"], root / "session_pending" / entry["name"],
                     root / "session_pending" / (entry["name"] + ".previous")):
            _safe_local(path.absolute())
            if path.exists() and not path.is_file():
                raise ValueError("Session commit artifact is not a file")
    # Publication of the last checkpoint is the commit point. If absent/old,
    # restore the preceding selected/trace/checkpoint bundle using retained bytes.
    observed_hashes = {entry["name"]: file_digest(root / entry["name"])
        if (root / entry["name"]).is_file() else None for entry in entries}
    committed = observed_hashes["last_checkpoint.pt"] == entries[-1]["next_sha256"]
    for entry in entries:
        visible = root / entry["name"]
        backup = root / "session_pending" / (entry["name"] + ".previous")
        observed = observed_hashes[entry["name"]]
        if committed:
            if observed != entry["next_sha256"]:
                raise ValueError("Committed session bundle is incomplete; manual reconciliation required")
        elif entry["previous_sha256"] is None:
            if observed not in (None, entry["next_sha256"]):
                raise ValueError("Unbound session artifact during rollback")
        elif observed != entry["previous_sha256"]:
            if observed not in (None, entry["next_sha256"]) or not backup.is_file() or file_digest(backup) != entry["previous_sha256"]:
                raise ValueError("Previous session bundle is unavailable; manual reconciliation required")
    # Validate the whole transaction before making any recovery edit.
    if not committed:
        for entry in entries:
            visible = root / entry["name"]
            backup = root / "session_pending" / (entry["name"] + ".previous")
            if entry["previous_sha256"] is None:
                visible.unlink(missing_ok=True)
            elif observed_hashes[entry["name"]] != entry["previous_sha256"]:
                os.replace(backup, visible)
        sync_directory(root)
    journal.unlink()
    sync_directory(root)
    for entry in entries:
        for suffix in ("", ".previous"):
            (root / "session_pending" / (entry["name"] + suffix)).unlink(missing_ok=True)


def _assert_model_schema(selected, resume):
    import torch
    if (not isinstance(selected, dict) or selected.keys() != resume.keys() or any(
            not isinstance(value, torch.Tensor) or value.shape != resume[key].shape or value.dtype != resume[key].dtype
            for key, value in selected.items())):
        raise ValueError("Selected model tensor schema differs from resume")


class SessionOutputs:
    def __init__(self, root, context, *, contract, adapter, trajectory_id, train_role_identity,
                 resume_trajectory=None):
        from .family_training_runner import read_recipe
        self.root, self.context, self.adapter = Path(root).absolute(), context, adapter
        self.resume_trajectory = resume_trajectory
        self.bounded_pair = False
        self.pending_pass = None
        _safe_local(self.root)
        if file_digest(contract) != context.training_recipe_sha256:
            raise ValueError("Session contract differs from the frozen Spec")
        record, self.recipe = read_recipe(contract)
        from .family_training_recipe import BOUNDED_MODES
        self.bounded_pair = self.recipe.mode in BOUNDED_MODES
        self.expected = record["acceptance_requirements"]
        self.root.mkdir(parents=True, exist_ok=True)
        if train_role_identity != record["train_role_identity"]:
            raise ValueError("Training metric role differs from frozen contract")
        _recover_commit(self.root, self.context)
        publish_immutable_bytes(self.root / "training_contract.json", Path(contract).read_bytes())
        self.pending = self.root / "session_pending"
        _safe_local(self.pending.absolute())
        self.pending.mkdir(exist_ok=True)
        # Discard only identified producer scratch that preceded journal publication.
        for name in _COMMIT_FILES:
            for suffix in ("", ".previous"):
                path = self.pending / (name + suffix)
                _safe_local(path.absolute())
                path.unlink(missing_ok=True)
        semantics = {"live_train_metric": {"metric": "MAE", "unit": "normalized-Gap",
            "target": "Gap", "role_identity": train_role_identity, "weights": "live", "direction": "minimize"},
            "live_dev_metric": None, "ema_dev_metric": None}
        if self.recipe.dataset.development:
            weights = self.recipe.selection.rsplit("-", 1)[1]
            semantics["ema_dev_metric" if weights == "ema" else "live_dev_metric"] = {
                "metric": "MAE", "unit": "eV", "target": "Gap", "role_identity": record["development_role_identity"],
                "weights": weights, "direction": "minimize"}
            if self.bounded_pair and self.recipe.ema_decay is not None:
                semantics["live_dev_metric"] = {**semantics["ema_dev_metric"], "weights": "live"}
        if self.bounded_pair:
            semantics["live_train_metric"]["unit"] = "eV"
        self.stage = StageRecorder(self.root / "canonical_trace.json", context, "downstream",
            semantics, trajectory_id=trajectory_id,
            device_time_semantics="sum_over_devices" if self.bounded_pair else None)
        rows = self.stage.recorder.record["observations"]
        self.previous_cumulative = rows[-1]["cumulative_wall_time_seconds"] if rows and self.bounded_pair else 0.
        self.segment_started = self.previous_observation = time.perf_counter()

    def selected(self, progress, weights, state, evaluation):
        _finite_tree(state, "selected.model")
        atomic_torch_save(self.pending / "selected_model.pt", {"context": self.context.to_dict(),
            "model": state, "optimizer_step": progress.optimizer_step,
            "epoch_or_pass": progress.optimizer_step / self.recipe.loop_plan.steps_per_pass,
            "weights": weights, "selection": self.recipe.selection,
            "development_mae_eV": evaluation["mae_eV"] if evaluation is not None else None})
        if evaluation is not None:
            atomic_torch_save(self.pending / "development_predictions.pt", {"context": self.context.to_dict(),
                **{key: evaluation[key].detach().cpu() for key in ("source_idx", "target_eV", "prediction_eV")}})

    def checkpoint(self, progress, state):
        if self.pending_pass is not None and self.pending_pass[0].optimizer_step != progress.optimizer_step:
            raise ValueError("Pass metrics differ from acknowledged checkpoint")
        changed = [name for name in ("selected_model.pt", "development_predictions.pt", "canonical_trace.json")
                   if (self.pending / name).is_file()]
        destination = self.pending if changed or self.pending_pass is not None else self.root
        write_resume_state(destination / "last_checkpoint.pt", self.context, adapter=self.adapter,
            model_state=state["model"], optimizer_state=state["optimizer"], rng_state=state["rng_state"],
            cursor=state["cursor"], optimizer_step=progress.optimizer_step,
            sample_presentations=progress.sample_presentations,
            scheduler_state=state["scheduler"], ema_state=state["ema"])
        observed = None
        if self.pending_pass is not None:
            _, fields = self.pending_pass
            observed = time.perf_counter()
            duration = observed - self.previous_observation
            cumulative = self.previous_cumulative + observed - self.segment_started
            # Bind the staged trace to the next checkpoint before publishing either.
            recorder = deepcopy(self.stage.recorder)
            recorder.path = self.pending / "canonical_trace.json"
            recorder.append_observation(optimizer_step=progress.optimizer_step, sample_presentations=progress.sample_presentations,
                epoch_or_pass=progress.pass_index, **fields,
                checkpoint_identity=file_digest(destination / "last_checkpoint.pt"),
                wall_time_seconds=duration, cumulative_wall_time_seconds=cumulative,
                device_time_seconds=duration, cumulative_device_time_seconds=cumulative)
            if "canonical_trace.json" not in changed:
                changed.append("canonical_trace.json")
        if changed:
            names = [*changed, "last_checkpoint.pt"]
            for name in names:
                _safe_local((self.root / name).absolute())
            entries = [{"name": name, "previous_sha256": file_digest(self.root / name) if (self.root / name).is_file() else None,
                        "next_sha256": file_digest(self.pending / name)} for name in names]
            atomic_write(self.root / "session_commit.json", json_bytes({"format": "molgap-session-commit-v1",
                "context": self.context.to_dict(), "entries": entries}))
            # Keep the preceding generation until checkpoint-last publication commits.
            for name in names:
                if (self.root / name).is_file():
                    os.replace(self.root / name, self.pending / (name + ".previous"))
                os.replace(self.pending / name, self.root / name)
            sync_directory(self.root)
            _recover_commit(self.root, self.context)
            if "canonical_trace.json" in changed:
                self.stage.recorder.record = validate_canonical_trace(_json(self.root / "canonical_trace.json"))
        if observed is not None:
            self.previous_observation, self.pending_pass = observed, None
        if (self.resume_trajectory is not None and progress.next_batch == 0 and
                progress.pass_index > 0 and progress.pass_index % 10 == 0 and progress.status != "complete"):
            from .experiment_resume import create_resume_bundle
            destination = self.root.parent / "checkpoint_chunks" / self.context.arm_id / f"pass_{progress.pass_index:02d}"
            create_resume_bundle(self.root, self.context, self.resume_trajectory, destination)

    def pass_finished(self, progress, metrics):
        fields = {"live_train_metric": metrics["normalized_train_gap_mae"]}
        if metrics["development"] is not None:
            fields["ema_dev_metric" if self.recipe.selection.endswith("ema") else "live_dev_metric"] = metrics["development"]["mae_eV"]
        if metrics.get("development_live") is not None:
            fields["live_dev_metric"] = metrics["development_live"]["mae_eV"]
        if self.bounded_pair:
            fields["live_train_metric"] *= self.recipe.target_std_eV
            self.pending_pass = (progress, fields)
            return
        recorder = deepcopy(self.stage.recorder)
        recorder.path = self.pending / "canonical_trace.json"
        recorder.append_observation(optimizer_step=progress.optimizer_step, sample_presentations=progress.sample_presentations,
            epoch_or_pass=progress.pass_index, **fields)

    def retained_state(self):
        import torch
        state = torch.load(_artifact_path(self.root, "last_checkpoint.pt"), map_location="cpu", weights_only=True)
        _check_context(state.pop("context"), self.context)
        state.setdefault("ema", None)
        progress = validate_session_resume(self.recipe, state)
        rows = [row for row in self.stage.recorder.record["observations"] if row["event"] == "observation"]
        if any(row["optimizer_step"] > progress.optimizer_step for row in rows):
            raise ValueError("Trace is ahead of the durable checkpoint; reconcile before resume")
        if state["cursor"]["selected_step"] is not None:
            selected = torch.load(_artifact_path(self.root, "selected_model.pt"), map_location="cpu", weights_only=True)
            _check_context(selected.get("context"), self.context)
            if selected.get("optimizer_step") != state["cursor"]["selected_step"]:
                raise ValueError("Retained selected endpoint differs from resume history")
        return state

    def complete(self, progress, *, runtime, costs, target_identity=None):
        if progress.status != "complete":
            raise ValueError("Partial work is not a terminal endpoint")
        rows = [row for row in self.stage.recorder.record["observations"] if row["event"] == "observation"]
        if not rows or rows[-1]["optimizer_step"] != progress.optimizer_step:
            self.stage.observe(optimizer_step=progress.optimizer_step, sample_presentations=progress.sample_presentations,
                epoch_or_pass=progress.optimizer_step / self.recipe.loop_plan.steps_per_pass)
        artifacts = {"selected_model": "selected_model.pt", "resume": "last_checkpoint.pt",
            "trace": "canonical_trace.json", "contract": "training_contract.json"}
        if self.recipe.dataset.development:
            artifacts["predictions"] = "development_predictions.pt"
        write_output_manifest(self.root, self.context, adapter=self.adapter, artifacts=artifacts,
            progress={"optimizer_steps": progress.optimizer_step, "sample_presentations": progress.sample_presentations,
                "completed_passes": progress.pass_index, "next_batch": progress.next_batch, "status": progress.status},
            runtime=runtime, costs=costs)
        return inspect_session_output(self.root, context=self.context, expected=self.expected, target_identity=target_identity)


def inspect_session_output(root, *, context, expected, target_identity=None):
    """Mechanical CPU inspection only; no model construction, prediction or promotion."""
    import torch
    from .experiment_family_artifacts import artifact_adapter
    from .family_training_runner import read_recipe
    from .v4_runtime import state_dict_sha256
    blockers, observed = [], {}
    try:
        root = Path(root).absolute()
        _safe_local(root)
        manifest = _json(_artifact_path(root, "output_manifest.json"))
        if set(manifest) != {"format", "context", "adapter", "artifacts", "progress", "runtime", "costs"} or manifest["format"] != OUTPUT_FORMAT:
            raise ValueError("Unsupported session output manifest")
        _check_context(manifest["context"], context)
        profile = artifact_adapter(manifest["adapter"], (context.family_name, context.family_version))
        if profile.protocol != "family-session":
            raise ValueError("Not a shared-session output profile")
        artifacts, paths = manifest["artifacts"], {}
        if type(artifacts) is not dict:
            raise ValueError("Invalid session artifact roles")
        for role, entry in artifacts.items():
            if type(entry) is not dict or set(entry) != {"path", "sha256"}:
                raise ValueError("Invalid artifact pointer")
            _digest(entry["sha256"], role)
            paths[role] = _artifact_path(root, entry["path"])
            if file_digest(paths[role]) != entry["sha256"]:
                raise ValueError("Artifact hash mismatch: " + role)
        if len({path.as_posix().casefold() for path in paths.values()}) != len(paths):
            raise ValueError("Artifact roles alias each other")
        if artifacts["contract"]["sha256"] != context.training_recipe_sha256:
            raise ValueError("Unpinned session contract")
        contract, recipe = read_recipe(paths["contract"])
        if expected != contract["acceptance_requirements"]:
            raise ValueError("Expectations differ from the frozen contract")
        if set(paths) != (ROLES if recipe.dataset.development else ROLES - {"predictions"}):
            raise ValueError("Artifact roles differ from selected data roles")
        _runtime(manifest["runtime"], context, "fp32")
        _costs(manifest["costs"])
        state = torch.load(paths["resume"], map_location="cpu", weights_only=True)
        if type(state) is not dict:
            raise ValueError("Invalid resume payload")
        _check_context(state.pop("context"), context)
        state.setdefault("ema", None)
        progress = validate_session_resume(recipe, state)
        actual = {"optimizer_steps": progress.optimizer_step, "sample_presentations": progress.sample_presentations,
            "completed_passes": progress.pass_index, "next_batch": progress.next_batch, "status": progress.status}
        if progress.status != "complete" or manifest["progress"] != actual:
            raise ValueError("Missing complete acknowledged endpoint")
        trace = validate_canonical_trace(_json(paths["trace"]))
        if trace["run_id"] != context.logical_run_id + ":" + context.arm_id + ":downstream":
            raise ValueError("Trace run/stage identity mismatch")
        companion = _json(_artifact_path(root, artifacts["trace"]["path"] + ".context.json"))
        _check_context(companion.get("context"), context)
        if companion.get("stage_id") != "downstream":
            raise ValueError("Trace context stage mismatch")
        rows = [row for row in trace["observations"] if row["event"] == "observation"]
        expected_steps = list(range(recipe.loop_plan.steps_per_pass, recipe.optimizer_steps + 1,
                                    recipe.loop_plan.steps_per_pass))
        if recipe.optimizer_steps % recipe.loop_plan.steps_per_pass:
            expected_steps.append(recipe.optimizer_steps)
        if [row["optimizer_step"] for row in rows] != expected_steps:
            raise ValueError("Trace is missing complete-pass or partial-terminal exposure")
        per_pass = recipe.loop_plan.steps_per_pass * recipe.batch_size if recipe.drop_last else recipe.train_rows
        for row in rows:
            passes, offset = divmod(row["optimizer_step"], recipe.loop_plan.steps_per_pass)
            if (row["sample_presentations"] != passes * per_pass + min(offset * recipe.batch_size, per_pass)
                    or row["epoch_or_pass"] != row["optimizer_step"] / recipe.loop_plan.steps_per_pass):
                raise ValueError("Trace sample/pass exposure mismatch")
        selected = torch.load(paths["selected_model"], map_location="cpu", weights_only=True)
        if type(selected) is not dict:
            raise ValueError("Invalid selected payload")
        _check_context(selected.get("context"), context)
        _finite_tree(selected, "selected")
        if (not isinstance(selected.get("model"), dict) or not selected["model"] or
                any(not isinstance(value, torch.Tensor) for value in selected["model"].values()) or
                selected.get("optimizer_step") != state["cursor"]["selected_step"] or
                selected.get("selection") != recipe.selection or
                selected.get("weights") != recipe.selection.rsplit("-", 1)[1] or
                selected.get("epoch_or_pass") != selected["optimizer_step"] / recipe.loop_plan.steps_per_pass):
            raise ValueError("Selected checkpoint identity/policy mismatch")
        _assert_model_schema(selected["model"], state["model"])
        if recipe.dataset.development:
            observed.update(_inspect_predictions(paths["predictions"], context=context, expected=expected,
                target_identity=target_identity))
            weights = selected["weights"]
            key = "ema_dev_metric" if weights == "ema" else "live_dev_metric"
            definition = {"metric": "MAE", "unit": "eV", "target": "Gap",
                "role_identity": contract["development_role_identity"], "weights": weights, "direction": "minimize"}
            if trace["metric_semantics"][key] != definition:
                raise ValueError("Selected development metric role/weight semantics differ")
            values = [row[key] for row in rows]
            if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
                raise ValueError("Development trace lacks selected-weight observations")
            at = next(row for row in rows if row["optimizer_step"] == selected["optimizer_step"])
            for value in (selected.get("development_mae_eV"), state["cursor"]["best_development_mae_eV"], at[key], min(values)):
                if type(value) not in (int, float) or not math.isclose(value, observed["development_mae_eV"], rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError("Selected predictions/trace/resume metric disagree")
        else:
            if (selected.get("development_mae_eV") is not None or any(
                    trace["metric_semantics"][key] is not None or any(row[key] is not None for row in rows)
                    for key in ("live_dev_metric", "ema_dev_metric"))):
                raise ValueError("Final-only output fabricated development evidence")
            final = state["ema"] if selected["weights"] == "ema" else state["model"]
            if state_dict_sha256(selected["model"]) != state_dict_sha256(final):
                raise ValueError("Final selected weights differ from the resume endpoint")
        observed.update(progress=actual, artifacts=artifacts, costs=manifest["costs"],
            selected_optimizer_step=selected["optimizer_step"], selected_weights=selected["weights"],
            producer_platform_version=manifest["context"]["platform_version"])
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, EOFError, StopIteration) as exc:
        blockers.append(str(exc))
    return {"status": "BLOCKED" if blockers else "MECHANICALLY_VERIFIED", "blockers": blockers,
        "context": context.to_dict(), "observed": observed, "scientific_acceptance": "NOT_EVALUATED",
        "replay_readiness": "NOT_EVALUATED", "runtime_qualification": "NOT_EVALUATED"}
