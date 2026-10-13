"""Session artifact translation using existing atomic IO, trace and inspection owners."""
from __future__ import annotations

import math
import time
from pathlib import Path

from .experiment_family_workflow import (
    OUTPUT_FORMAT, ROLES, StageRecorder, _artifact_path, _check_context, _costs, _digest,
    _finite_tree, _inspect_predictions, _json, _publish, _runtime, write_output_manifest,
    write_resume_state,
)
from .experiment_launch import _safe_local, publish_immutable_bytes
from .research_memory.trace import file_digest, validate_canonical_trace
from .family_training_session import validate_session_resume
from .training_reproducibility import atomic_torch_save


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
        self.bounded_pair = self.recipe.mode in {"degree_bond_local_ema999", "degree_local_cycle_common_ema999"}
        self.expected = record["acceptance_requirements"]
        self.root.mkdir(parents=True, exist_ok=True)
        publish_immutable_bytes(self.root / "training_contract.json", Path(contract).read_bytes())
        semantics = {"live_train_metric": {"metric": "MAE", "unit": "normalized-Gap",
            "target": "Gap", "role_identity": train_role_identity, "weights": "live", "direction": "minimize"},
            "live_dev_metric": None, "ema_dev_metric": None}
        if train_role_identity != record["train_role_identity"]:
            raise ValueError("Training metric role differs from frozen contract")
        if self.recipe.dataset.development:
            weights = self.recipe.selection.rsplit("-", 1)[1]
            semantics["ema_dev_metric" if weights == "ema" else "live_dev_metric"] = {
                "metric": "MAE", "unit": "eV", "target": "Gap", "role_identity": record["development_role_identity"],
                "weights": weights, "direction": "minimize"}
            if self.recipe.mode in {"degree_bond_local_ema999", "degree_local_cycle_common_ema999"} and self.recipe.ema_decay is not None:
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
        atomic_torch_save(self.root / "selected_model.pt", {"context": self.context.to_dict(),
            "model": state, "optimizer_step": progress.optimizer_step,
            "epoch_or_pass": progress.optimizer_step / self.recipe.loop_plan.steps_per_pass,
            "weights": weights, "selection": self.recipe.selection,
            "development_mae_eV": evaluation["mae_eV"] if evaluation is not None else None})
        if evaluation is not None:
            atomic_torch_save(self.root / "development_predictions.pt", {"context": self.context.to_dict(),
                **{key: evaluation[key].detach().cpu() for key in ("source_idx", "target_eV", "prediction_eV")}})

    def checkpoint(self, progress, state):
        write_resume_state(self.root / "last_checkpoint.pt", self.context, adapter=self.adapter,
            model_state=state["model"], optimizer_state=state["optimizer"], rng_state=state["rng_state"],
            cursor=state["cursor"], optimizer_step=progress.optimizer_step,
            sample_presentations=progress.sample_presentations,
            scheduler_state=state["scheduler"], ema_state=state["ema"])
        if self.pending_pass is not None:
            acknowledged, fields = self.pending_pass
            if acknowledged.optimizer_step != progress.optimizer_step:
                raise ValueError("Pass metrics differ from acknowledged checkpoint")
            observed = time.perf_counter()
            duration = observed - self.previous_observation
            cumulative = self.previous_cumulative + observed - self.segment_started
            self.stage.observe(optimizer_step=progress.optimizer_step, sample_presentations=progress.sample_presentations,
                epoch_or_pass=progress.pass_index, **fields,
                checkpoint_identity=file_digest(self.root / "last_checkpoint.pt"),
                wall_time_seconds=duration, cumulative_wall_time_seconds=cumulative,
                device_time_seconds=duration, cumulative_device_time_seconds=cumulative)
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
        self.stage.observe(optimizer_step=progress.optimizer_step, sample_presentations=progress.sample_presentations,
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
