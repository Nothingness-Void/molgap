"""Session artifact translation using existing atomic IO, trace and inspection owners."""
from __future__ import annotations

import json
import math
import os
import time
from copy import deepcopy
from pathlib import Path

from .experiment_family_workflow import (
    DATA_OPEN_FILE, OUTPUT_FORMAT, ROLES, StageRecorder, _artifact_path, _check_context, _costs, _digest,
    _finite_tree, _inspect_predictions, _json, _runtime, tensor_digest,
    write_output_manifest, write_resume_state,
)
from .experiment_family_artifacts import SESSION_OPTIONAL_ARTIFACTS
from .experiment_launch import _safe_local, publish_immutable_bytes
from .research_memory.trace import atomic_write, file_digest, json_bytes, sync_directory, validate_canonical_trace
from .research_memory.roles import validate_observed_role_truth
from .research_memory.schemas import validate_role_event
from .family_training_session import validate_session_resume
from .training_reproducibility import atomic_torch_save


_COMMIT_FILES = {"selected_model.pt", "development_predictions.pt", "canonical_trace.json", "last_checkpoint.pt", "final_development_predictions.pt"}


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


class RoleAudit:
    """Persist the observed role facts for one shared training session.

    The sidecar deliberately uses the existing terminal role-audit mapping
    (``context``, ``role_use`` and ``roles``).  Event records are emitted only
    after a loader has yielded a real batch or an output callback has received
    a computed metric/selection.  Permit and role-history bytes bind the input
    identity, but never create an observed event by themselves.
    """

    PROTECTED_ROLES = ("official_validation", "test_dev", "test_challenge")

    def _resume_action_id(self, resume_trajectory):
        if resume_trajectory is None:
            return "A001"
        if isinstance(resume_trajectory, (bytes, bytearray)):
            value = json.loads(bytes(resume_trajectory).decode("utf-8"))
        elif isinstance(resume_trajectory, dict):
            value = resume_trajectory
        else:
            value = _json(Path(resume_trajectory))
        actions = value.get("actions") if isinstance(value, dict) else None
        if not isinstance(actions, list):
            raise ValueError("Resume trajectory lacks prospective actions for role audit")
        matches = [action for action in actions
                   if isinstance(action, dict) and self.run_id in action.get("run_ids", [])]
        if len(matches) != 1:
            raise ValueError("Resume trajectory lacks one action for the bound downstream run")
        action_id = matches[0].get("action_id")
        if not isinstance(action_id, str) or not action_id:
            raise ValueError("Resume trajectory action lacks an identity")
        return action_id

    @staticmethod
    def _evidence_ref(path):
        """Use a repository locator when the output is inside a checkout."""
        for parent in (path.parent, *path.parents):
            if (parent / ".git").exists():
                return path.relative_to(parent).as_posix()
        return path.name

    def __init__(self, root, context, *, trajectory_id, train_role_identity,
                 development_role_identity=None, action_id=None, resume_trajectory=None):
        self.root = Path(root).absolute()
        self.path = self.root / "role_audit.json"
        self.context = context
        self.trajectory_id = trajectory_id
        self.run_id = context.logical_run_id + ":" + context.arm_id + ":downstream"
        self.action_id = action_id or self._resume_action_id(resume_trajectory)
        self.evidence_ref = self._evidence_ref(self.path)
        self._bindings = {}
        self.roles = []
        # A missing protected-role observation is unknown until the admitted
        # train/development streams have actually completed.  The loader
        # boundary, rather than possession of a permit or absence of a log,
        # is what lets the terminal claim those roles were untouched.
        self.role_use = {role: "unknown" for role in self.PROTECTED_ROLES}
        _safe_local(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.path.is_file():
            self._load()
        elif (self.root / "last_checkpoint.pt").is_file():
            raise ValueError(
                "Role audit sidecar missing for an existing checkpoint; refusing scientific resume")
        else:
            self._write()

    def _load(self):
        audit = _json(self.path)
        if audit.get("context") != self.context.to_dict():
            raise ValueError("Role audit differs from the frozen run/source/attempt context")
        roles, role_use = audit.get("roles"), audit.get("role_use")
        if not isinstance(roles, list) or not isinstance(role_use, dict):
            raise ValueError("Role audit requires observed role_use and existing-schema roles")
        rebased = False
        loaded_roles = []
        for event in roles:
            event = validate_role_event(event)
            if (event["trajectory_id"], event["run_id"], event["action_id"]) != (
                    self.trajectory_id, self.run_id, self.action_id):
                raise ValueError("Role audit event differs from its session identity")
            if event["evidence_ref"] != self.evidence_ref:
                reference_name = event["evidence_ref"].replace("\\", "/").rsplit("/", 1)[-1]
                if reference_name != self.path.name:
                    raise ValueError("Role audit event source differs from its sidecar")
                # Portable resume changes the output root.  Rebind only the
                # evidence locator; all event identities and observed facts
                # remain byte-for-byte semantic data from the source run.
                event = dict(event)
                event["evidence_ref"] = self.evidence_ref
                rebased = True
            loaded_roles.append(event)
        for role in self.PROTECTED_ROLES:
            if role_use.get(role) not in {"untouched", "consumed", "not_applicable", "unknown"}:
                raise ValueError("Role audit lacks explicit protected-role truth: " + role)
        validate_observed_role_truth([role_use], loaded_roles)
        self.roles = [dict(event) for event in loaded_roles]
        self.role_use = dict(role_use)
        for event in self.roles:
            logical_role = ("development" if event["role_name"] in {"common_dev", "development"}
                            or event["access_kind"] in {"selection_used", "metric_computed"}
                            and event["role_name"] not in {"train", "train_100k", "train_500k"}
                            else "train")
            self._bindings.setdefault(logical_role, (event["role_name"],
                event["dataset_identity"], event["row_manifest_hash"]))
        if rebased:
            self._write()

    def _write(self):
        value = {"context": self.context.to_dict(), "role_use": dict(self.role_use),
                 "roles": [dict(event) for event in self.roles]}
        validate_observed_role_truth([value["role_use"],], value["roles"])
        atomic_write(self.path, json_bytes(value))

    @staticmethod
    def _source_row_hash(source_indices):
        if source_indices is None:
            return None
        import torch
        if isinstance(source_indices, range):
            source_indices = torch.arange(source_indices.start, source_indices.stop,
                                          source_indices.step, dtype=torch.int64)
        elif isinstance(source_indices, torch.Tensor):
            source_indices = source_indices.reshape(-1)
            if source_indices.dtype != torch.int64:
                raise ValueError("Role audit source indices must be int64")
        else:
            source_indices = torch.as_tensor(tuple(source_indices), dtype=torch.int64)
        return tensor_digest(source_indices, role="source_idx")

    def bind_dataset(self, logical_role, dataset, *, manifest_sha256=None, source_indices=None):
        """Bind actual dataset identity before any access event is emitted."""
        permit = getattr(dataset, "permit", None)
        role_name = getattr(permit, "role", None)
        if not isinstance(role_name, str) or not role_name:
            role_name = "train" if logical_role == "train" else "development"
        dataset_identity = getattr(permit, "version_identity", None)
        row_manifest_hash = getattr(permit, "manifest_sha256", None)
        if not isinstance(dataset_identity, str) or not dataset_identity:
            dataset_identity = manifest_sha256
        if not isinstance(dataset_identity, str) or not dataset_identity:
            raise ValueError("Role audit dataset lacks an admitted manifest identity: " + str(logical_role))
        if not isinstance(row_manifest_hash, str) or not row_manifest_hash:
            row_manifest_hash = self._source_row_hash(source_indices) or manifest_sha256
        if not isinstance(row_manifest_hash, str) or not row_manifest_hash:
            raise ValueError("Role audit dataset lacks an admitted row identity: " + str(logical_role))
        binding = (role_name, dataset_identity, row_manifest_hash)
        previous = self._bindings.get(logical_role)
        if previous is not None and previous != binding:
            raise ValueError("Role audit dataset identity changed during a session")
        self._bindings[logical_role] = binding
        self.role_use.setdefault(role_name, "unknown")
        self._write()

    def _binding(self, logical_role):
        binding = self._bindings.get(logical_role)
        if binding is not None:
            return binding
        raise ValueError("Role audit lacks an admitted dataset binding: " + str(logical_role))

    def observe(self, logical_role, access_kinds, *, selection_used=False):
        """Record only facts established by a yielded batch or owner callback."""
        role_name, dataset_identity, row_manifest_hash = self._binding(logical_role)
        access_kinds = list(access_kinds)
        if selection_used and "selection_used" not in access_kinds:
            access_kinds.append("selection_used")
        existing = {(event["role_name"], event["access_kind"]): event for event in self.roles}
        changed = False
        for access_kind in access_kinds:
            key = (role_name, access_kind)
            if key in existing:
                continue
            event = {
                "schema": "molgap-role-event-v1",
                "role_event_id": f"role-{self.trajectory_id}-{role_name}-{access_kind}",
                "trajectory_id": self.trajectory_id,
                "action_id": self.action_id,
                "run_id": self.run_id,
                "dataset_identity": dataset_identity,
                "row_manifest_hash": row_manifest_hash,
                "role_name": role_name,
                "access_kind": access_kind,
                "selection_used": bool(selection_used or access_kind == "selection_used"),
                "evidence_ref": self.evidence_ref,
            }
            self.roles.append(validate_role_event(event))
            changed = True
        if changed:
            self.role_use[role_name] = "consumed"
            self._write()

    def observe_batch(self, logical_role):
        kinds = ("training_membership", "prediction_input", "labels_read") \
            if logical_role == "train" else ("prediction_input", "labels_read")
        self.observe(logical_role, kinds)

    def observe_metrics(self, logical_role, *, selection_used=False):
        kinds = ["metric_computed"]
        if selection_used:
            kinds.append("selection_used")
        self.observe(logical_role, kinds, selection_used=selection_used)

    def validate(self, *, development):
        validate_observed_role_truth([self.role_use], self.roles)
        required = {"train": {"training_membership", "prediction_input", "labels_read", "metric_computed"}}
        if development:
            required["development"] = {"prediction_input", "labels_read", "metric_computed"}
        for logical_role, access_kinds in required.items():
            role_name = self._binding(logical_role)[0]
            if self.role_use.get(role_name) != "consumed":
                raise ValueError("Role audit lacks observed execution for " + logical_role)
            observed_kinds = {event["access_kind"] for event in self.roles
                              if event["role_name"] == role_name}
            if not access_kinds <= observed_kinds:
                raise ValueError("Role audit lacks observed access for " + logical_role)
        # Only a complete admitted loader boundary proves that protected roles
        # were untouched.  A missing sidecar or an unobserved stream remains
        # unknown and fails closed at terminal staging.
        for role in self.PROTECTED_ROLES:
            self.role_use[role] = "untouched"
        self._write()

    def pointer(self):
        return {"path": self.path.name, "sha256": file_digest(self.path)}


class SessionOutputs:
    def __init__(self, root, context, *, contract, adapter, trajectory_id, train_role_identity,
                 resume_trajectory=None, action_id=None):
        from .family_training_runner import read_recipe
        from .v4_bundle import _payload
        from .v4_runtime import normalized_source_sha256
        self.root, self.context, self.adapter = Path(root).absolute(), context, adapter
        self.resume_trajectory = resume_trajectory
        self.bounded_pair = False
        self.pending_pass = None
        _safe_local(self.root)
        if normalized_source_sha256(Path(contract)) != context.training_recipe_sha256:
            raise ValueError("Session contract differs from the frozen Spec")
        record, self.recipe = read_recipe(contract)
        if self.recipe.formal_interface:
            from .experiment_family_artifacts import artifact_adapter
            from .experiment_execution import MODERN2D_FORMAL_PROFILE
            if (adapter != MODERN2D_FORMAL_PROFILE or
                    self.recipe.interface != (context.family_name, context.family_version)):
                raise ValueError("Formal output family/version/profile differs from its training contract")
            artifact_adapter(adapter, self.recipe.interface)
        if self.recipe.integration_preview:
            from .experiment_family_artifacts import artifact_adapter
            from .experiment_execution import MODERN2D_PREVIEW_PROFILE
            if (adapter != MODERN2D_PREVIEW_PROFILE or
                    self.recipe.interface != (context.family_name, context.family_version)):
                raise ValueError("Integration preview output family/version/profile differs from contract")
            artifact_adapter(adapter, self.recipe.interface)
        from .family_training_policy import mode_policy
        self.policy = mode_policy(self.recipe.family, self.recipe.mode)
        # Formal Modern2D references need the same checkpoint-bound time axes as
        # timed addon modes; a reference policy must not silently erase evidence.
        self.bounded_pair = self.policy.process_timing or self.recipe.formal_interface
        self.expected = record["acceptance_requirements"]
        self.root.mkdir(parents=True, exist_ok=True)
        if train_role_identity != record["train_role_identity"]:
            raise ValueError("Training metric role differs from frozen contract")
        self.role_audit = RoleAudit(self.root, context, trajectory_id=trajectory_id,
            train_role_identity=train_role_identity,
            development_role_identity=record.get("development_role_identity")
            if self.recipe.dataset.development else None, action_id=action_id,
            resume_trajectory=resume_trajectory)
        _recover_commit(self.root, self.context)
        publish_immutable_bytes(self.root / "training_contract.json", _payload(Path(contract)))
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
            weights = self.recipe.selection_weights
            semantics["ema_dev_metric" if weights in {"ema", "clean-ema"} else "live_dev_metric"] = {
                "metric": "MAE", "unit": "eV", "target": "Gap", "role_identity": record["development_role_identity"],
                "weights": weights, "direction": "minimize"}
            for extra in dict.fromkeys((*self.policy.extra_development_weights,
                                        *self.recipe.development_reports)):
                semantics["ema_dev_metric" if extra == "ema" else "live_dev_metric"] = {
                    "metric": "MAE", "unit": "eV", "target": "Gap",
                    "role_identity": record["development_role_identity"],
                    "weights": extra, "direction": "minimize"}
        semantics["live_train_metric"]["unit"] = self.policy.train_metric_unit
        self.stage = StageRecorder(self.root / "canonical_trace.json", context, "downstream",
            semantics, trajectory_id=trajectory_id,
            device_time_semantics="sum_over_devices" if self.bounded_pair else None)
        from .training_logging import TrainingLogger, TrainingLogPolicy
        self.logger = TrainingLogger(self.root, context, self.recipe,
            TrainingLogPolicy.from_dict(record.get("logging")),
            observations=self.stage.recorder.record["observations"],
            observation_train_unit=semantics["live_train_metric"]["unit"])
        rows = self.stage.recorder.record["observations"]
        self.previous_cumulative = rows[-1]["cumulative_wall_time_seconds"] if rows and self.bounded_pair else 0.
        self.segment_started = self.previous_observation = time.perf_counter()

    def batch_finished(self, progress, *, learning_rates):
        self.logger.progress(progress, learning_rates=learning_rates)

    def selected(self, progress, weights, state, evaluation):
        _finite_tree(state, "selected.model")
        if evaluation is not None:
            self.role_audit.observe_metrics("development", selection_used=True)
        atomic_torch_save(self.pending / "selected_model.pt", {"context": self.context.to_dict(),
            "model": state, "optimizer_step": progress.optimizer_step,
            "epoch_or_pass": progress.optimizer_step / self.recipe.loop_plan.steps_per_pass,
            "weights": weights, "selection": self.recipe.selection,
            "development_mae_eV": evaluation["mae_eV"] if evaluation is not None else None})
        if evaluation is not None:
            atomic_torch_save(self.pending / "development_predictions.pt", {"context": self.context.to_dict(),
                **{key: evaluation[key].detach().cpu() for key in ("source_idx", "target_eV", "prediction_eV")}})

    def checkpoint_from_builder(self, progress, state_builder):
        """Consume a trusted synchronous producer; the writer validates all state."""
        if not callable(state_builder):
            raise TypeError("Checkpoint state builder must be callable")
        self.checkpoint(progress, state_builder())

    def checkpoint(self, progress, state):
        if self.pending_pass is not None and self.pending_pass[0].optimizer_step != progress.optimizer_step:
            raise ValueError("Pass metrics differ from acknowledged checkpoint")
        changed = [name for name in ("selected_model.pt", "development_predictions.pt", "canonical_trace.json", "final_development_predictions.pt")
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
            if self.recipe.formal_interface:
                # Record the observed checkpoint LR, not a reconstructed schedule.
                rates = [group["lr"] for group in state["optimizer"]["param_groups"]]
                if len(rates) != 1:
                    raise ValueError("Formal trace requires one observed optimizer learning rate")
                fields = {**fields, "learning_rate": rates[0]}
            observed = time.perf_counter()
            duration = observed - self.previous_observation
            cumulative = self.previous_cumulative + observed - self.segment_started
            # Bind the staged trace to the next checkpoint before publishing either.
            recorder = deepcopy(self.stage.recorder)
            recorder.path = self.pending / "canonical_trace.json"
            recorder.append_observation(optimizer_step=progress.optimizer_step, sample_presentations=progress.sample_presentations,
                epoch_or_pass=progress.optimizer_step / self.recipe.loop_plan.steps_per_pass, **fields,
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
                progress.pass_index > 0 and progress.pass_index % self.policy.chunk_passes == 0 and progress.status != "complete"):
            from .experiment_resume import create_resume_bundle
            destination = self.root.parent / "checkpoint_chunks" / self.context.arm_id / f"pass_{progress.pass_index:02d}"
            create_resume_bundle(self.root, self.context, self.resume_trajectory, destination)
        self.logger.checkpoint_finished(progress)

    def pass_finished(self, progress, metrics):
        self.logger.pass_finished(progress, metrics)
        self.role_audit.observe_metrics("train", selection_used=False)
        if metrics["development"] is not None:
            self.role_audit.observe_metrics("development", selection_used=False)
            if (getattr(self.recipe.formal_training, "evaluation_every_steps", None) is not None
                    and progress.optimizer_step == self.recipe.optimizer_steps):
                atomic_torch_save(self.pending / "final_development_predictions.pt",
                    {"context": self.context.to_dict(), **{key: metrics["development"][key].detach().cpu()
                        for key in ("source_idx", "target_eV", "prediction_eV")}})
        fields = {"live_train_metric": metrics["normalized_train_gap_mae"]}
        if metrics["development"] is not None:
            fields["ema_dev_metric" if self.recipe.selection.endswith("ema") else "live_dev_metric"] = metrics["development"]["mae_eV"]
        if metrics.get("development_live") is not None:
            fields["live_dev_metric"] = metrics["development_live"]["mae_eV"]
        if metrics.get("development_ema") is not None:
            fields["ema_dev_metric"] = metrics["development_ema"]["mae_eV"]
        if self.policy.train_metric_unit == "eV":
            fields["live_train_metric"] *= self.recipe.target_std_eV
        if self.bounded_pair:
            self.pending_pass = (progress, fields)
            return
        recorder = deepcopy(self.stage.recorder)
        recorder.path = self.pending / "canonical_trace.json"
        recorder.append_observation(optimizer_step=progress.optimizer_step, sample_presentations=progress.sample_presentations,
            epoch_or_pass=progress.optimizer_step / self.recipe.loop_plan.steps_per_pass, **fields)

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
        self.role_audit.validate(development=self.recipe.dataset.development)
        rows = [row for row in self.stage.recorder.record["observations"] if row["event"] == "observation"]
        if not rows or rows[-1]["optimizer_step"] != progress.optimizer_step:
            self.stage.observe(optimizer_step=progress.optimizer_step, sample_presentations=progress.sample_presentations,
                epoch_or_pass=progress.optimizer_step / self.recipe.loop_plan.steps_per_pass)
        artifacts = {"selected_model": "selected_model.pt", "resume": "last_checkpoint.pt",
            "trace": "canonical_trace.json", "contract": "training_contract.json",
            "trace_context": "canonical_trace.json.context.json"}
        if self.recipe.dataset.development:
            artifacts["predictions"] = "development_predictions.pt"
            if getattr(self.recipe.formal_training, "evaluation_every_steps", None) is not None:
                artifacts["final_predictions"] = "final_development_predictions.pt"
        # Bind the recorder's sidecar only when this worker created it for the
        # current output root; an arbitrary file must never become evidence.
        from .family_training_runner import _DATA_OPEN
        if _DATA_OPEN["path"] == self.root / DATA_OPEN_FILE and (self.root / DATA_OPEN_FILE).is_file():
            artifacts["data_open"] = DATA_OPEN_FILE
        artifacts["role_audit"] = "role_audit.json"
        native_names = ("runtime_certificate.json", "runtime_provenance.json")
        native_present = tuple((self.root / name).is_file() for name in native_names)
        if native_present == (True, True):
            for name in native_names:
                artifacts[name[:-5]] = name
        elif native_present != (False, False):
            raise ValueError("Runtime certificate/provenance sidecars must be published as a pair")
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
        # A retained output may reference an addon retired from the active
        # catalog.  It is parsed for identity/receipt checks only and never
        # executed by this inspection path.
        contract, recipe = read_recipe(paths["contract"], _frozen_read=True)
        if recipe.interface != (context.family_name, context.family_version):
            raise ValueError("Session contract family/version differs from context")
        if recipe.formal_interface:
            from .experiment_execution import MODERN2D_FORMAL_PROFILE
            if manifest["adapter"] != MODERN2D_FORMAL_PROFILE:
                raise ValueError("Formal training requires its distinct output profile")
            observed.update(execution_scope="CONFIGURED_SCREEN" if recipe.formal_training.configured else "FORMAL_100K_TRAINING",
                scientific_acceptance="NOT_EVALUATED", replay_readiness="NOT_EVALUATED")
        if recipe.integration_preview:
            from .experiment_execution import MODERN2D_PREVIEW_PROFILE
            if manifest["adapter"] != MODERN2D_PREVIEW_PROFILE:
                raise ValueError("Integration preview requires its explicit output profile")
            observed.update(integration_status="INTEGRATION_PENDING",
                execution_scope="INTEGRATION_PREVIEW_ONLY", native_execution_released=False)
        if expected != contract["acceptance_requirements"]:
            raise ValueError("Expectations differ from the frozen contract")
        required_roles = ROLES if recipe.dataset.development else ROLES - {"predictions"}
        runtime_roles = {"runtime_certificate", "runtime_provenance"}
        if (root / "role_audit.json").is_file() and "role_audit" not in artifacts:
            raise ValueError("Role audit sidecar is not bound by output manifest")
        if "role_audit" in artifacts and artifacts["role_audit"]["path"] != "role_audit.json":
            raise ValueError("Role audit must bind the fixed role_audit.json sidecar")
        optional_roles = SESSION_OPTIONAL_ARTIFACTS
        present_runtime_roles = set(paths) & runtime_roles
        if present_runtime_roles not in (set(), runtime_roles):
            raise ValueError("Runtime certificate/provenance artifact pair is incomplete")
        if "trace_context" in paths:
            required_roles = required_roles | {"trace_context"}
            if artifacts["trace_context"]["path"] != artifacts["trace"]["path"] + ".context.json":
                raise ValueError("Trace companion path differs from its bound trace")
        if "data_open" in paths:
            required_roles = required_roles | {"data_open"}
            _check_context(_json(paths["data_open"]).get("context"), context)
        if set(paths) != required_roles | (set(paths) & optional_roles):
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
        observation_interval = getattr(recipe.formal_training, "evaluation_every_steps", None) or recipe.loop_plan.steps_per_pass
        expected_steps = list(range(observation_interval, recipe.optimizer_steps + 1,
                                    observation_interval))
        if recipe.optimizer_steps % observation_interval:
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
                selected.get("weights") != recipe.selection_weights or
                selected.get("epoch_or_pass") != selected["optimizer_step"] / recipe.loop_plan.steps_per_pass):
            raise ValueError("Selected checkpoint identity/policy mismatch")
        _assert_model_schema(selected["model"], state["model"])
        if recipe.dataset.development:
            observed.update(_inspect_predictions(paths["predictions"], context=context, expected=expected,
                target_identity=target_identity))
            weights = selected["weights"]
            key = "ema_dev_metric" if weights in {"ema", "clean-ema"} else "live_dev_metric"
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
            if getattr(recipe.formal_training, "evaluation_every_steps", None) is not None:
                if "final_predictions" not in paths or artifacts["final_predictions"]["path"] != "final_development_predictions.pt":
                    raise ValueError("Step-clock terminal requires its final live prediction payload")
                final = _inspect_predictions(paths["final_predictions"], context=context, expected=expected,
                    target_identity=target_identity)
                if not math.isclose(final["development_mae_eV"], rows[-1][key], rel_tol=1e-6, abs_tol=1e-8):
                    raise ValueError("Final predictions differ from the terminal trace metric")
                observed["final_development_mae_eV"] = final["development_mae_eV"]
        else:
            if (selected.get("development_mae_eV") is not None or any(
                    trace["metric_semantics"][key] is not None or any(row[key] is not None for row in rows)
                    for key in ("live_dev_metric", "ema_dev_metric"))):
                raise ValueError("Final-only output fabricated development evidence")
            final = (state["cursor"]["weight_state"]["clean"]["model"] if selected["weights"] == "clean-ema"
                     else state["ema"] if selected["weights"] == "ema" else state["model"])
            if state_dict_sha256(selected["model"]) != state_dict_sha256(final):
                raise ValueError("Final selected weights differ from the resume endpoint")
        observed.update(progress=actual, artifacts=artifacts, costs=manifest["costs"],
            selected_optimizer_step=selected["optimizer_step"], selected_weights=selected["weights"],
            producer_platform_version=manifest["context"]["platform_version"])
        audit_path = root / "role_audit.json"
        if audit_path.is_file():
            audit = _json(audit_path)
            _check_context(audit.get("context"), context)
            audit_roles, audit_use = audit.get("roles"), audit.get("role_use")
            if not isinstance(audit_roles, list) or not isinstance(audit_use, dict):
                raise ValueError("Role audit requires observed role_use and existing-schema roles")
            for event in audit_roles:
                validate_role_event(event)
            validate_observed_role_truth([audit_use], audit_roles)
            audit_ref = audit_path.relative_to(root).as_posix()
            observed["role_audit"] = {"path": audit_ref, "sha256": file_digest(audit_path),
                                       "role_use": audit_use, "event_count": len(audit_roles)}
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, EOFError, StopIteration) as exc:
        blockers.append(str(exc))
    return {"status": "BLOCKED" if blockers else "MECHANICALLY_VERIFIED", "blockers": blockers,
        "context": context.to_dict(), "observed": observed, "scientific_acceptance": "NOT_EVALUATED",
        "replay_readiness": "NOT_EVALUATED", "runtime_qualification": "NOT_EVALUATED"}
