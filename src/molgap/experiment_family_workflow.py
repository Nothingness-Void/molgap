"""Shared producer/inspector hooks; existing Spec and RML remain authoritative.

No training, model construction, platform request or scientific decision occurs
here. A verified output is a mechanical result, never a replay-ready claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

from .experiment_family_artifacts import artifact_adapter
from .experiment_launch import _safe_local, read_launch_receipt, publish_immutable_bytes, build_launch_receipt
from .experiment_spec import ExperimentSpec, _digest, _identifier, _text
from .research_memory.paths import repo_local_path
from .research_memory.trace import (
    RMLTraceRecorder, file_digest, json_bytes, validate_canonical_trace,
)
from .screen_policy import canonical_fingerprint


OUTPUT_FORMAT = "molgap-family-output-v1"
ROLES = frozenset({"predictions", "selected_model", "resume", "trace", "contract"})
EXPECTED = frozenset({"epochs", "optimizer_steps", "sample_presentations",
                      "development_rows", "source_idx_sha256", "target_sha256", "precision"})


def _json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    value = json.loads(path.read_bytes(), object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    if type(value) is not dict:
        raise ValueError("Expected JSON object")
    return value


def _positive(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label}: expected positive integer")


def _artifact_path(root: Path, value: str) -> Path:
    if type(value) is not str or "\\" in value or ":" in value or value.startswith("/"):
        raise ValueError("Expected relative POSIX artifact path")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("Invalid artifact path")
    _safe_local((root / value).absolute())
    path = repo_local_path(root, value)
    if not path.is_file():
        raise ValueError(f"Missing artifact: {value}")
    return path


@dataclass(frozen=True)
class RunContext:
    experiment_id: str
    logical_run_id: str
    arm_id: str
    arm_identity: str
    spec_identity: str
    family_name: str
    family_version: str
    source_commit: str
    source_archive_sha256: str
    package_identity: str
    training_recipe_sha256: str
    platform: str
    account: str
    run_reference: str
    platform_version: str | None

    def __post_init__(self):
        for field in ("experiment_id", "logical_run_id", "arm_id"):
            _identifier(getattr(self, field), field)
        for field in ("arm_identity", "spec_identity", "source_archive_sha256", "package_identity", "training_recipe_sha256"):
            _digest(getattr(self, field), field)
        for field in ("family_name", "family_version", "platform", "account", "run_reference"):
            _text(getattr(self, field), field)
        if self.platform_version is not None:
            _text(self.platform_version, "platform_version")
        import re
        if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.source_commit):
            raise ValueError("Expected full source commit")
        if any(c in self.run_reference for c in "?@#") or self.account != self.run_reference.split("/")[0]:
            raise ValueError("Run reference/account mismatch or credential-bearing reference")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def for_training(cls, spec: ExperimentSpec, package_dir: Path, *,
                     expected_package_identity: str, arm_id: str,
                     account: str, run_reference: str) -> RunContext:
        """Producer context before a submitter returns a version.

        The account/slug are frozen declarations, not a platform observation.
        Unknown version remains null; acceptance binds the real receipt later.
        """
        receipt = build_launch_receipt(spec, Path(package_dir), expected_package_identity=expected_package_identity)
        declaration = spec.to_dict()
        arm = next((a for a in declaration["arms"] if a["arm_id"] == arm_id), None)
        if arm is None:
            raise ValueError("Unknown producer arm")
        binding = receipt["binding"]
        return cls(declaration["experiment_id"], declaration["logical_run_id"], arm_id,
                   canonical_fingerprint(arm), spec.identity, arm["family"]["name"],
                   arm["family"]["version"], binding["source_commit"], binding["source_archive_sha256"],
                   binding["package_identity"], arm["training"]["recipe"]["sha256"], declaration["platform"]["name"], account,
                   run_reference, None)
    @classmethod
    def from_launch(cls, spec: ExperimentSpec, receipt_path: Path, package_dir: Path,
                    *, expected_package_identity: str, arm_id: str) -> RunContext:
        receipt = read_launch_receipt(Path(receipt_path).absolute(), spec, Path(package_dir),
                                      expected_package_identity=expected_package_identity)
        if receipt["submission_state"] not in {"ACCEPTED", "RECONCILED_EXISTING"}:
            raise ValueError("RunContext requires a reconciled observed launch")
        runs = receipt["physical_runs"]["value"]
        if runs is None:
            raise ValueError("Missing physical arm mapping")
        run = next((item for item in runs if arm_id in item["arm_ids"]), None)
        if run is None or run["platform_version"]["value"] is None:
            raise ValueError("Missing exact physical run/version")
        declaration = spec.to_dict()
        arm = next(item for item in declaration["arms"] if item["arm_id"] == arm_id)
        binding = receipt["binding"]
        reference = run["canonical_reference"]
        return cls(declaration["experiment_id"], declaration["logical_run_id"], arm_id,
                   canonical_fingerprint(arm), spec.identity, arm["family"]["name"],
                   arm["family"]["version"], binding["source_commit"],
                   binding["source_archive_sha256"], binding["package_identity"], arm["training"]["recipe"]["sha256"],
                   declaration["platform"]["name"], reference.split("/")[0], reference,
                   run["platform_version"]["value"])


def _check_context(observed: dict, expected: RunContext) -> None:
    if type(observed) is not dict or set(observed) != set(expected.to_dict()):
        raise ValueError("Output context missing identity fields")
    values = expected.to_dict()
    for key, value in observed.items():
        if key == "platform_version" and value is None:
            continue  # Missing producer observation is retained, never backfilled.
        if value != values[key]:
            raise ValueError("Output context/source/arm mismatch: " + key)


class StageRecorder:
    """Use one trace per phase; counters are acknowledged processed work.

    Explicit stage/run identities prevent pretraining and downstream metrics or
    optimizer resets from being silently combined into one curve.
    """
    def __init__(self, path: Path, context: RunContext, stage_id: str,
                 metric_semantics: dict, *, trajectory_id: str | None = None):
        _identifier(stage_id, "stage_id")
        _safe_local(Path(path).absolute())
        self.context = context
        _publish(Path(path).with_suffix(Path(path).suffix + ".context.json"),
                 {"context": context.to_dict(), "stage_id": stage_id})
        self.recorder = RMLTraceRecorder(path,
            trajectory_id=trajectory_id or context.experiment_id + "-" + context.arm_id,
            run_id=context.logical_run_id + ":" + context.arm_id + ":" + stage_id,
            metric_semantics=metric_semantics)
        # The companion is immutable and disallows reuse with another source/arm.

    def observe(self, *, optimizer_step: int, sample_presentations: int,
                epoch_or_pass: int | float, **fields) -> None:
        _positive(optimizer_step, "optimizer_step")
        _positive(sample_presentations, "sample_presentations")
        if isinstance(epoch_or_pass, bool) or not isinstance(epoch_or_pass, (int, float)) or not math.isfinite(epoch_or_pass) or epoch_or_pass < 0:
            raise ValueError("Invalid epoch_or_pass")
        self.recorder.append_observation(optimizer_step=optimizer_step,
            sample_presentations=sample_presentations, epoch_or_pass=epoch_or_pass, **fields)


def _publish(path: Path, value: dict) -> Path:
    """Immutable metadata publication, with identical retries allowed."""
    _safe_local(path.absolute())
    publish_immutable_bytes(path.absolute(), json_bytes(value))
    return path


def write_output_manifest(output_dir: Path, context: RunContext, *, adapter: str,
                          artifacts: dict[str, str], progress: dict,
                          runtime: dict, costs: list[dict]) -> Path:
    root = Path(output_dir).absolute()
    _safe_local(root)
    artifact_adapter(adapter, (context.family_name, context.family_version))
    if set(artifacts) != ROLES:
        raise ValueError("Output requires exactly predictions/selected_model/resume/trace/contract")
    entries = {role: {"path": name, "sha256": file_digest(_artifact_path(root, name))}
               for role, name in artifacts.items()}
    if len({item["path"].casefold() for item in entries.values()}) != len(ROLES):
        raise ValueError("Artifact roles must bind distinct files")
    return _publish(root / "output_manifest.json", {
        "format": OUTPUT_FORMAT, "context": context.to_dict(), "adapter": adapter,
        "artifacts": entries, "progress": progress, "runtime": runtime, "costs": costs,
    })


def write_selected_state(path: Path, context: RunContext, *, model_state: dict,
                         epoch: int, optimizer_step: int, weights: str,
                         development_mae_eV: float) -> Path:
    """Producer hook; the owning trainer selects the endpoint and its weights."""
    from .training_reproducibility import atomic_torch_save, assert_finite_state_dict
    _safe_local(Path(path).absolute())
    _positive(epoch, "epoch")
    _positive(optimizer_step, "optimizer_step")
    if weights not in {"live", "ema"} or isinstance(development_mae_eV, bool) or not isinstance(development_mae_eV, (float, int)) or not math.isfinite(development_mae_eV) or development_mae_eV < 0:
        raise ValueError("Invalid selected weight/metric semantics")
    assert_finite_state_dict(model_state, label="selected_model")
    atomic_torch_save(Path(path), {"context": context.to_dict(), "model": model_state,
        "epoch": epoch, "optimizer_step": optimizer_step, "weights": weights,
        "development_mae_eV": development_mae_eV})
    return Path(path)


def write_resume_state(path: Path, context: RunContext, *, adapter: str, model_state: dict,
                       optimizer_state: dict, rng_state: dict, cursor: dict,
                       optimizer_step: int, sample_presentations: int,
                       scheduler_state: dict | None = None, ema_state: dict | None = None) -> Path:
    """Atomic output translation; resume loading stays with the owning trainer."""
    from .training_reproducibility import atomic_torch_save, assert_finite_state_dict
    _safe_local(Path(path).absolute())
    profile = artifact_adapter(adapter, (context.family_name, context.family_version))
    _positive(optimizer_step, "optimizer_step")
    _positive(sample_presentations, "sample_presentations")
    state = {"context": context.to_dict(), "model": model_state, "optimizer": optimizer_state,
             "rng_state": tensor_safe_rng_state(rng_state), "cursor": cursor, "optimizer_step": optimizer_step,
             "sample_presentations": sample_presentations}
    if scheduler_state is not None:
        state["scheduler"] = scheduler_state
    if ema_state is not None:
        state["ema"] = ema_state
    if any(not isinstance(state.get(key), dict) or not state[key] for key in profile.resume_keys):
        raise ValueError("Incomplete checkpoint/resume state")
    assert_finite_state_dict(model_state, label="resume.model")
    if ema_state is not None:
        assert_finite_state_dict(ema_state, label="resume.ema")
    atomic_torch_save(Path(path), state)
    return Path(path)


def tensor_safe_rng_state(state: dict) -> dict:
    """Translate the existing RNG helper's NumPy array to safe primitive data."""
    import copy
    result = copy.deepcopy(state)
    numpy_state = result.get("numpy")
    if isinstance(numpy_state, tuple) and len(numpy_state) == 5 and hasattr(numpy_state[1], "tolist"):
        result["numpy"] = (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:])
    _validate_rng_state(result)
    return result


def _validate_rng_state(state):
    import torch
    if not isinstance(state, dict) or not {"python", "numpy", "torch", "cuda"} <= set(state):
        raise ValueError("Incomplete Python/NumPy/Torch/CUDA RNG resume state")
    for value in (state["torch"], *state["cuda"]):
        if not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.ndim != 1 or not len(value):
            raise ValueError("Invalid torch/CUDA RNG state")
    python = state["python"]
    numpy = state["numpy"]
    if not isinstance(python, tuple) or len(python) != 3 or not isinstance(python[1], tuple):
        raise ValueError("Invalid Python RNG state")
    if not isinstance(numpy, tuple) or len(numpy) != 5 or not isinstance(numpy[1], list) or not numpy[1] or any(type(n) is not int or not 0 <= n < 2**32 for n in numpy[1]):
        raise ValueError("Invalid tensor-safe NumPy RNG state")


def _finite_tree(value, label):
    import torch
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite checkpoint state: " + label)
    elif isinstance(value, dict):
        for key, child in value.items():
            _finite_tree(child, label + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            _finite_tree(child, label)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite checkpoint state: " + label)


class FamilyOutputSession:
    """Small event adapter for an existing trainer; no replacement training loop.

    The trainer provides acknowledged counters, selected weights, predictions,
    sampler cursor and actual runtime observations. This session handles their
    persistence and translation consistently for both registered families.
    """
    def __init__(self, output_dir: Path, context: RunContext, *, adapter: str,
                 contract: Path, trajectory_id: str, metric_semantics: dict):
        import time
        self.started = time.perf_counter()
        self.root, self.context, self.adapter = Path(output_dir).absolute(), context, adapter
        _safe_local(self.root)
        _safe_local(Path(contract).absolute())
        artifact_adapter(adapter, (context.family_name, context.family_version))
        if file_digest(contract) != context.training_recipe_sha256:
            raise ValueError("Session contract/Spec recipe mismatch")
        self.expected = _json(Path(contract)).get("acceptance_requirements")
        if not isinstance(self.expected, dict) or set(self.expected) != EXPECTED:
            raise ValueError("Owning recipe lacks frozen acceptance_requirements")
        self.root.mkdir(parents=True, exist_ok=True)
        publish_immutable_bytes(self.root / "training_contract.json", Path(contract).read_bytes())
        self.stage = StageRecorder(self.root / "canonical_trace.json", context, "downstream",
                                   metric_semantics, trajectory_id=trajectory_id)

    def epoch_finished(self, *, epoch: int, optimizer_step: int, sample_presentations: int, **metrics):
        self.stage.observe(epoch_or_pass=epoch, optimizer_step=optimizer_step,
                           sample_presentations=sample_presentations, **metrics)

    def selected(self, *, model_state: dict, epoch: int, optimizer_step: int, weights: str,
                 prediction_eV, target_eV, source_idx):
        from .training_reproducibility import atomic_torch_save
        mae = (prediction_eV.detach().cpu().double() - target_eV.detach().cpu().double()).abs().mean().item()
        write_selected_state(self.root / "selected_model.pt", self.context, model_state=model_state,
            epoch=epoch, optimizer_step=optimizer_step, weights=weights, development_mae_eV=mae)
        atomic_torch_save(self.root / "development_predictions.pt", {
            "context": self.context.to_dict(),
            "prediction_eV": prediction_eV.detach().cpu(), "target_eV": target_eV.detach().cpu(),
            "source_idx": source_idx.detach().cpu()})

    def checkpoint(self, *, model_state: dict, optimizer_state: dict, cursor: dict,
                   optimizer_step: int, sample_presentations: int, rng_state: dict,
                   scheduler_state: dict | None = None, ema_state: dict | None = None):
        write_resume_state(self.root / "last_checkpoint.pt", self.context, adapter=self.adapter,
            model_state=model_state, optimizer_state=optimizer_state, rng_state=rng_state,
            cursor=cursor, optimizer_step=optimizer_step, sample_presentations=sample_presentations,
            scheduler_state=scheduler_state, ema_state=ema_state)

    def complete(self, *, runtime: dict, hardware: str) -> dict:
        import time
        rows = self.stage.recorder.record["observations"]
        if not rows:
            raise ValueError("Cannot complete a session without observed training work")
        progress = {"epochs": len(rows), "optimizer_steps": rows[-1]["optimizer_step"],
                    "sample_presentations": rows[-1]["sample_presentations"]}
        costs = [
            {"metric": "wall_seconds", "unit": "seconds", "value": time.perf_counter() - self.started,
             "status": "measured", "semantics": "process_wall", "hardware": hardware},
            {"metric": "device_seconds", "unit": "seconds", "value": None,
             "status": "missing", "semantics": "allocated_device", "hardware": hardware},
        ]
        write_output_manifest(self.root, self.context, adapter=self.adapter,
            artifacts={"predictions": "development_predictions.pt", "selected_model": "selected_model.pt",
                       "resume": "last_checkpoint.pt", "trace": "canonical_trace.json", "contract": "training_contract.json"},
            progress=progress, runtime=runtime, costs=costs)
        return inspect_output(self.root, context=self.context, expected=self.expected)


def tensor_digest(tensor, *, role: str) -> str:
    """Canonical row/target identity: little-endian int64 or float64 CPU bytes."""
    import torch
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 1:
        raise ValueError("Expected one-dimensional tensor")
    if role == "source_idx":
        if tensor.dtype != torch.int64:
            raise ValueError("source_idx must be int64")
        array = tensor.detach().cpu().contiguous().numpy().astype("<i8", copy=False)
    elif role == "target":
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError("target must be finite floating point")
        array = tensor.detach().cpu().double().contiguous().numpy().astype("<f8", copy=False)
    else:
        raise ValueError("Unknown digest role")
    return hashlib.sha256(array.tobytes()).hexdigest()


def _runtime(runtime: dict, context: RunContext, precision: str):
    required = {"platform", "account", "precision", "source_commit", "source_archive_sha256"}
    if type(runtime) is not dict or set(runtime) != required:
        raise ValueError("Runtime requires explicit platform/account/precision/source identity")
    for key in required - {"precision"}:
        if runtime[key] != getattr(context, key):
            raise ValueError(f"Runtime identity mismatch: {key}")
    if runtime["precision"] != precision:
        raise ValueError("Runtime precision mismatch")


def _costs(costs):
    if type(costs) is not list or not costs:
        raise ValueError("Missing native cost observations")
    seen = set()
    for cost in costs:
        if type(cost) is not dict or set(cost) != {"metric", "unit", "value", "status", "semantics", "hardware"}:
            raise ValueError("Cost requires native metric/unit/status/semantics/hardware")
        if cost["metric"] in seen:
            raise ValueError("Duplicate native cost metric")
        seen.add(cost["metric"])
        if cost["status"] == "missing":
            if cost["value"] is not None:
                raise ValueError("Missing cost must remain null")
        elif cost["status"] in {"measured", "estimated"}:
            if isinstance(cost["value"], bool) or not isinstance(cost["value"], (int, float)) or not math.isfinite(cost["value"]) or cost["value"] < 0:
                raise ValueError("Invalid native cost")
        else:
            raise ValueError("Unknown cost status")
        allowed = {("wall_seconds", "seconds", "process_wall"),
                   ("device_seconds", "seconds", "allocated_device"),
                   ("device_seconds", "seconds", "device_busy")}
        if (cost["metric"], cost["unit"], cost["semantics"]) not in allowed:
            raise ValueError("Unsupported or conflated native cost semantics")
        _text(cost["hardware"], "cost.hardware")


def inspect_output(output_dir: Path, *, context: RunContext, expected: dict) -> dict:
    """Inspect pinned retained files on CPU, using safe tensor-only loading.

    Expected row/target hashes and exposure must come from the frozen contract,
    not from these outputs. No reference, role, runtime performance or causal
    qualification is inferred from this mechanical inspection.
    """
    import torch
    from .training_reproducibility import assert_finite_state_dict
    root = Path(output_dir).absolute()
    _safe_local(root)
    blockers, observed = [], {}
    try:
        if type(expected) is not dict or set(expected) != EXPECTED:
            raise ValueError("Missing or unknown acceptance expectations")
        for key in ("epochs", "optimizer_steps", "sample_presentations", "development_rows"):
            _positive(expected[key], key)
        for key in ("source_idx_sha256", "target_sha256"):
            _digest(expected[key], key)
        if expected["precision"] not in {"fp32", "fp16", "bf16"}:
            raise ValueError("Unsupported precision")
        manifest = _json(_artifact_path(root, "output_manifest.json"))
        if set(manifest) != {"format", "context", "adapter", "artifacts", "progress", "runtime", "costs"} or manifest["format"] != OUTPUT_FORMAT:
            raise ValueError("Unsupported output manifest")
        _check_context(manifest["context"], context)
        observed["producer_platform_version"] = manifest["context"]["platform_version"]
        profile = artifact_adapter(manifest["adapter"], (context.family_name, context.family_version))
        artifacts = manifest["artifacts"]
        if type(artifacts) is not dict or set(artifacts) != ROLES:
            raise ValueError("Missing or unknown artifact roles")
        paths = {}
        for role, entry in artifacts.items():
            if type(entry) is not dict or set(entry) != {"path", "sha256"}:
                raise ValueError("Invalid artifact binding")
            _digest(entry["sha256"], role + ".sha256")
            paths[role] = _artifact_path(root, entry["path"])
            if file_digest(paths[role]) != entry["sha256"]:
                raise ValueError(f"Artifact hash mismatch: {role}")
        if len(set(paths.values())) != len(ROLES):
            raise ValueError("Artifact roles alias each other")
        if artifacts["contract"]["sha256"] != context.training_recipe_sha256:
            raise ValueError("Acceptance contract is not pinned by the Spec recipe")
        contract = _json(paths["contract"])
        if contract.get("acceptance_requirements") != expected:
            raise ValueError("Supplied expectations disagree with the frozen recipe contract")
        _runtime(manifest["runtime"], context, expected["precision"])
        _costs(manifest["costs"])
        progress = manifest["progress"]
        if type(progress) is not dict or set(progress) != {"epochs", "optimizer_steps", "sample_presentations"}:
            raise ValueError("Missing explicit progress counters")
        for key, value in progress.items():
            if type(value) is not int or value != expected[key]:
                raise ValueError(f"Progress mismatch: {key}")
        payload = torch.load(paths["predictions"], map_location="cpu", weights_only=True)
        if type(payload) is not dict or not {"source_idx", "target_eV", "prediction_eV", "context"} <= payload.keys():
            raise ValueError("Unsupported predictions payload")
        _check_context(payload["context"], context)
        if any(value is True for key, value in payload.items() if key in {"official_validation_used", "test_dev_used", "test_challenge_used"}):
            raise ValueError("Protected role consumption declared")
        tensors = [payload[key] for key in ("source_idx", "target_eV", "prediction_eV")]
        if any(not isinstance(t, torch.Tensor) or t.ndim != 1 or len(t) != expected["development_rows"] for t in tensors):
            raise ValueError("Prediction/target/row shape mismatch")
        idx, target, prediction = tensors
        if idx.dtype != torch.int64 or len(torch.unique(idx)) != len(idx):
            raise ValueError("Invalid or duplicate source rows")
        if not prediction.is_floating_point() or not torch.isfinite(prediction).all():
            raise ValueError("Nonfinite or nonfloating predictions")
        if tensor_digest(idx, role="source_idx") != expected["source_idx_sha256"]:
            raise ValueError("Development row identity mismatch")
        if tensor_digest(target, role="target") != expected["target_sha256"]:
            raise ValueError("Development target identity mismatch")
        observed["development_mae_eV"] = (prediction.double() - target.double()).abs().mean().item()
        trace = validate_canonical_trace(_json(paths["trace"]))
        rows = [r for r in trace["observations"] if r["event"] == "observation"]
        if not rows or any(r["optimizer_step"] is None or r["sample_presentations"] is None or r["epoch_or_pass"] is None for r in rows):
            raise ValueError("Trace missing observed exposure/epoch counters")
        last = rows[-1]
        if last["optimizer_step"] != progress["optimizer_steps"] or last["sample_presentations"] != progress["sample_presentations"]:
            raise ValueError("Trace/terminal exposure mismatch")
        if len(rows) != progress["epochs"]:
            raise ValueError("Protocol requires one observation per completed epoch")
        if [r["epoch_or_pass"] for r in rows] != list(range(1, progress["epochs"] + 1)):
            raise ValueError("Trace epochs must be one-based contiguous completed epochs")
        expected_run = context.logical_run_id + ":" + context.arm_id + ":downstream"
        if trace["run_id"] != expected_run:
            raise ValueError("Trace arm/stage identity mismatch")
        for role in ("selected_model", "resume"):
            state = torch.load(paths[role], map_location="cpu", weights_only=True)
            if type(state) is not dict:
                raise ValueError(f"{role} context identity mismatch")
            _check_context(state.get("context"), context)
            if not isinstance(state.get("model"), dict) or not state["model"] or not all(isinstance(t, torch.Tensor) for t in state["model"].values()):
                raise ValueError(f"{role}: missing tensor model state")
            assert_finite_state_dict(state["model"], label=role)
            if role == "selected_model":
                step, epoch = state.get("optimizer_step"), state.get("epoch")
                if type(epoch) is not int or not 1 <= epoch <= progress["epochs"] or type(step) is not int or step not in {r["optimizer_step"] for r in rows}:
                    raise ValueError("Selected endpoint is outside the retained trace")
                if state.get("weights") not in {"live", "ema"} or (state["weights"] == "ema" and not profile.ema_required):
                    raise ValueError("Unsupported selected weight semantics")
                selected = rows[epoch - 1]
                if selected["optimizer_step"] != step:
                    raise ValueError("Selected epoch/step disagree")
                metric = "ema_dev_metric" if state["weights"] == "ema" else "live_dev_metric"
                semantics = trace["metric_semantics"][metric]
                if not semantics or semantics.get("unit") != "eV" or semantics.get("weights") != state["weights"]:
                    raise ValueError("Selected development metric semantics missing or mismatched")
                for value in (state.get("development_mae_eV"), selected[metric]):
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not math.isclose(value, observed["development_mae_eV"], rel_tol=1e-6, abs_tol=1e-8):
                        raise ValueError("Selected trace/checkpoint/prediction metric mismatch")
                observed.update(selected_epoch=epoch, selected_optimizer_step=step, selected_weights=state["weights"])
            else:
                if any(not isinstance(state.get(key), dict) or not state[key] for key in profile.resume_keys):
                    raise ValueError("Incomplete checkpoint/resume state")
                if profile.ema_required:
                    assert_finite_state_dict(state["ema"], label="resume.ema")
                optimizer = state["optimizer"]
                if not isinstance(optimizer.get("state"), dict) or not isinstance(optimizer.get("param_groups"), list) or not optimizer["param_groups"]:
                    raise ValueError("Incomplete optimizer resume state")
                _validate_rng_state(state["rng_state"])
                _finite_tree(state, "resume")
                cursor = state.get("cursor")
                if type(cursor) is not dict or set(cursor) != {"epoch", "next_batch", "sampler_order_sha256"}:
                    raise ValueError("Missing acknowledged sampler cursor")
                if type(cursor["epoch"]) is not int or cursor["epoch"] != progress["epochs"] or type(cursor["next_batch"]) is not int or cursor["next_batch"] < 0:
                    raise ValueError("Resume cursor/epoch mismatch")
                _digest(cursor["sampler_order_sha256"], "resume.sampler_order_sha256")
                _positive(state.get("optimizer_step"), "resume.optimizer_step")
                _positive(state.get("sample_presentations"), "resume.sample_presentations")
                if state.get("optimizer_step") != progress["optimizer_steps"] or state.get("sample_presentations") != progress["sample_presentations"]:
                    raise ValueError("Resume/terminal exposure mismatch")
        observed.update(progress=progress, costs=manifest["costs"], artifacts=artifacts)
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, EOFError) as exc:
        blockers.append(str(exc))
    return {"status": "BLOCKED" if blockers else "MECHANICALLY_VERIFIED", "blockers": blockers,
            "context": context.to_dict(), "observed": observed,
            "scientific_acceptance": "NOT_EVALUATED", "replay_readiness": "NOT_EVALUATED",
            "runtime_qualification": "NOT_EVALUATED"}


def check_acceptance_plan(spec: ExperimentSpec, repo_root: Path, plan: dict) -> dict:
    """Prelaunch capability and retained-reference check, without execution.

    The owning contract still provides comparison_prelaunch and reference_bundle.
    A passing plan only checks availability and schemas, not trainer execution.
    """
    from .comparison_readiness import validate_comparison_prelaunch, validate_reference_bundle, reference_bundle_digest
    declaration = spec.to_dict()
    expected_arms = {a["arm_id"]: a for a in declaration["arms"]}
    results = []
    if type(plan) is not dict or set(plan) != {"format", "spec_identity", "arms"} or plan["format"] != "molgap-family-acceptance-plan-v1" or plan["spec_identity"] != spec.identity:
        raise ValueError("Acceptance plan/spec mismatch")
    entries = plan["arms"]
    if type(entries) is not list or [e.get("arm_id") for e in entries] != list(expected_arms):
        raise ValueError("Acceptance plan requires exactly all Spec arms in order")
    for entry in entries:
        blockers = []
        try:
            if set(entry) != {"arm_id", "adapter", "expected", "contract", "comparison_prelaunch", "reference_bundle", "reference_artifacts"}:
                raise ValueError("Missing or unknown acceptance plan fields")
            arm = expected_arms[entry["arm_id"]]
            artifact_adapter(entry["adapter"], (arm["family"]["name"], arm["family"]["version"]))
            if type(entry["expected"]) is not dict or set(entry["expected"]) != EXPECTED:
                raise ValueError("Incomplete terminal expectations")
            for key in ("epochs", "optimizer_steps", "sample_presentations", "development_rows"):
                _positive(entry["expected"][key], key)
            for key in ("source_idx_sha256", "target_sha256"):
                _digest(entry["expected"][key], key)
            if entry["expected"]["precision"] not in {"fp32", "fp16", "bf16"}:
                raise ValueError("Unsupported precision")
            def pinned(pointer):
                if type(pointer) is not dict or set(pointer) != {"path", "sha256"}:
                    raise ValueError("Expected pinned repository artifact")
                _digest(pointer["sha256"], "plan artifact digest")
                path = _artifact_path(Path(repo_root).absolute(), pointer["path"])
                if file_digest(path) != pointer["sha256"]:
                    raise ValueError("Plan artifact hash mismatch")
                return path
            contract_path = pinned(entry["contract"])
            if entry["contract"]["sha256"] != arm["training"]["recipe"]["sha256"] or _json(contract_path).get("acceptance_requirements") != entry["expected"]:
                raise ValueError("Acceptance expectations are not pinned by the owning Spec recipe")
            comparison = validate_comparison_prelaunch(_json(pinned(entry["comparison_prelaunch"])))
            if not comparison["prelaunch_ready"]:
                raise ValueError("Owning comparison prelaunch is blocked")
            bundle = validate_reference_bundle(_json(pinned(entry["reference_bundle"])))
            if comparison["reference_bundle_id"] != bundle["reference_bundle_id"] or comparison["reference_bundle_sha256"] != reference_bundle_digest(bundle):
                raise ValueError("Comparison/reference bundle mismatch")
            predictions = bundle["prediction_manifest"]
            if predictions["source_idx_sha256"] != entry["expected"]["source_idx_sha256"] or predictions["target_sha256"] != entry["expected"]["target_sha256"] or predictions["row_count"] != entry["expected"]["development_rows"]:
                raise ValueError("Reference/expected development identity mismatch")
            bindings = entry["reference_artifacts"]
            from .comparison_readiness import REQUIRED_REFERENCE_ARTIFACTS
            if type(bindings) is not dict or set(bindings) != REQUIRED_REFERENCE_ARTIFACTS | {"predictions"}:
                raise ValueError("Incomplete retained reference artifacts")
            for pointer in bindings.values():
                pinned(pointer)
            if bindings["predictions"]["sha256"] != predictions["prediction_sha256"]:
                raise ValueError("Retained reference prediction hash mismatch")
            owners = {"runtime_certificate": "runtime_certificate_ref", "row_manifest": "row_manifest_ref",
                      "target_manifest": "target_manifest_ref", "trace_manifest": "trace_manifest_ref",
                      "role_history": "role_history_ref", "target_transform_asset": "target_transform_asset_ref",
                      "cost_records": "cost_records_ref", "acceptance": "acceptance_ref", "decision": "decision_ref"}
            for artifact, field in owners.items():
                if repo_local_path(repo_root, bindings[artifact]["path"]) != repo_local_path(repo_root, bundle[field]):
                    raise ValueError("Reference artifact/owner pointer mismatch: " + artifact)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            blockers.append(str(exc))
        results.append({"arm_id": entry["arm_id"], "status": "BLOCKED" if blockers else "AVAILABLE", "blockers": blockers})
    return {"status": "BLOCKED" if any(r["blockers"] for r in results) else "ACCEPTANCE_INPUTS_AVAILABLE",
            "spec_identity": spec.identity, "arms": results, "trainer_execution": "NOT_VERIFIED"}


def close_verified_outputs(repo_root: Path, spec: ExperimentSpec, descriptor, *, outputs: dict,
                           execute: bool = True) -> dict:
    """Reinspect every arm, then delegate the unchanged terminal/RML transaction.

    outputs maps arm_id to {context, output_dir, expected}. Descriptor artifact
    paths and SHA must agree with the inspected raw files before any RML write.
    """
    from .experiment_terminal import execute_terminal_descriptor, translate_terminal_descriptor
    if type(execute) is not bool:
        raise ValueError("Expected explicit boolean execution switch")
    arms = spec.to_dict()["arms"]
    if set(outputs) != {a["arm_id"] for a in arms}:
        raise ValueError("Closure requires independent output inputs for every arm")
    # Existing translator verifies prospective, evidence, terminal, path and SHA.
    translate_terminal_descriptor(repo_root, spec, descriptor)
    reports = {}
    for arm in descriptor.to_dict()["arms"]:
        item = outputs[arm["arm_id"]]
        report = inspect_output(item["output_dir"], context=item["context"], expected=item["expected"])
        reports[arm["arm_id"]] = report
        if report["status"] == "BLOCKED":
            continue
        ctx = item["context"]
        expected_arm = next(a for a in arms if a["arm_id"] == ctx.arm_id)
        if ctx.spec_identity != spec.identity or ctx.arm_identity != canonical_fingerprint(expected_arm):
            raise ValueError("Verified output/Spec mismatch")
        identity = arm["observed"]["identity"]
        for key, value in (("source_commit", ctx.source_commit), ("source_package_sha256", ctx.source_archive_sha256)):
            if identity[key]["value"] != value:
                raise ValueError("Descriptor/verified source identity mismatch")
        if identity["platform"]["run_reference"]["value"] != ctx.run_reference:
            raise ValueError("Descriptor/verified physical run mismatch")
        for role, output_role in (("predictions", "predictions"), ("checkpoint", "selected_model"), ("trace", "trace")):
            binding = arm["observed"]["artifacts"][role]
            checked = report["observed"]["artifacts"][output_role]
            if binding["sha256"] != checked["sha256"]:
                raise ValueError("Descriptor/verified artifact mismatch: " + role)
            inspected_path = _artifact_path(Path(item["output_dir"]).absolute(), checked["path"])
            if repo_local_path(repo_root, binding["locator"]) != inspected_path:
                raise ValueError("Descriptor/verified artifact path mismatch: " + role)
            if role == "trace" and ("trace" not in arm or repo_local_path(repo_root, arm["trace"]) != inspected_path):
                raise ValueError("Descriptor closure trace differs from the inspected trace")
        for field, counter in (("epoch", "epochs"), ("step", "optimizer_steps"), ("samples", "sample_presentations")):
            if arm["observed"]["progress"][field]["value"] != report["observed"]["progress"][counter]:
                raise ValueError("Descriptor/verified progress mismatch")
    if any(r["status"] == "BLOCKED" for r in reports.values()):
        return {"status": "BLOCKED", "arms": reports, "executed": False}
    if not execute:
        return {"status": "MECHANICALLY_VERIFIED", "arms": reports, "executed": False}
    results = execute_terminal_descriptor(repo_root, spec, descriptor)
    return {"status": "COMPLETE" if all(r.get("pipeline_status") == "COMPLETE" for r in results) else "INCOMPLETE",
            "arms": reports, "results": results, "executed": True}


def prepare_terminal_outputs(spec: ExperimentSpec, repo_root: Path, supplied: dict, *,
                             receipt_path: Path, package_dir: Path,
                             expected_package_identity: str) -> dict:
    """Confine CLI output inputs in the owning core, keeping CLI dispatch thin."""
    if type(supplied) is not dict or set(supplied) != {a["arm_id"] for a in spec.to_dict()["arms"]}:
        raise ValueError("Expected every independent Spec arm")
    outputs = {}
    for arm_id, item in supplied.items():
        if type(item) is not dict or set(item) != {"output_dir", "expected"}:
            raise ValueError("Expected per-arm output_dir/expected inputs")
        outputs[arm_id] = {
            "context": RunContext.from_launch(spec, receipt_path, package_dir,
                expected_package_identity=expected_package_identity, arm_id=arm_id),
            "expected": item["expected"], "output_dir": repo_local_path(repo_root, item["output_dir"])}
    return outputs


def build_verified_terminal_descriptor(repo_root: Path, spec: ExperimentSpec, *, outputs: dict,
                                       locations: dict):
    """Translate mechanical observations without per-experiment evidence glue.

    locations supplies existing trajectory, terminal, canonical trace and run ID.
    Scientific acceptance/V5/cost/role records remain the owning pipeline inputs;
    this helper neither invents nor rewrites their decisions.
    """
    from .experiment_terminal import TerminalDescriptor, translate_terminal_descriptor
    from .experiment_spec import TERMINAL_PROTOCOL
    root = Path(repo_root).resolve()
    declaration = spec.to_dict()
    ids = {a["arm_id"] for a in declaration["arms"]}
    if set(outputs) != ids or set(locations) != ids:
        raise ValueError("Descriptor requires every independently inspected arm")
    entries = []
    fact = lambda value: {"value": value, "missing_reason": None}
    for arm in declaration["arms"]:
        arm_id = arm["arm_id"]
        item = outputs[arm_id]
        ctx = item["context"]
        if ctx.spec_identity != spec.identity or ctx.arm_identity != canonical_fingerprint(arm):
            raise ValueError("Output/Spec arm binding mismatch")
        report = inspect_output(item["output_dir"], context=ctx, expected=item["expected"])
        if report["status"] != "MECHANICALLY_VERIFIED":
            raise ValueError("Output inspection blocked: " + "; ".join(report["blockers"]))
        where = locations[arm_id]
        if set(where) != {"trajectory_id", "run_id", "trajectory", "terminal", "trace"}:
            raise ValueError("Expected existing per-arm trajectory/terminal/trace locations")
        observation = report["observed"]
        def binding(role):
            entry = observation["artifacts"][role]
            path = _artifact_path(Path(item["output_dir"]).absolute(), entry["path"])
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError("RML closure artifacts must be retained below repository") from exc
            return {"status": "available", "locator": relative, "sha256": entry["sha256"], "missing_reason": None}
        progress = observation["progress"]
        identity = {
            "experiment_id": ctx.experiment_id, "logical_run_id": ctx.logical_run_id,
            "arm_id": arm_id, "spec_identity": ctx.spec_identity,
            "family": fact(arm["family"]),
            "recipe_identity": fact(canonical_fingerprint(arm["training"]["recipe"])),
            "data_identity": fact(canonical_fingerprint(arm["data"])),
            "split_identity": fact(canonical_fingerprint({"split": arm["data"]["split"], "roles": arm["data"]["roles"]})),
            "feature_identity": fact(arm["data"]["feature_sha256"]), "target": fact(arm["data"]["target"]),
            "initialization_identity": fact(canonical_fingerprint(arm["initialization"])),
            "source_commit": fact(ctx.source_commit), "source_package_sha256": fact(ctx.source_archive_sha256),
            "attempt_id": fact(ctx.arm_id + "-v" + ctx.platform_version),
            "platform": {"name": ctx.platform, "run_reference": fact(ctx.run_reference)},
        }
        # A contract binding is a declaration. Completion has independent file,
        # shape and exposure checks above; causal validity is still unevaluated.
        costs = [{"metric": c["metric"], "unit": c["unit"], "value": c["value"],
                  "status": "measurement_missing" if c["status"] == "missing" else c["status"],
                  "reason": None if c["status"] == "measured" else "Native " + c["hardware"] + ": " + c["semantics"]}
                 for c in observation["costs"]]
        missing = ["runtime_qualification_not_evaluated", "scientific_comparison_not_evaluated"]
        if observation.get("producer_platform_version") is None:
            missing.append("producer_platform_version_not_observed")
        metrics = {"status": "missing", "locator": None, "sha256": None,
                   "missing_reason": "Owning scientific metric record is separate from mechanical inspection"}
        entry = {"arm_id": arm_id, "arm_identity": ctx.arm_identity, **where,
                 "observed": {"identity": identity,
                    "terminal": {"status": fact("complete"), "exit_reason": fact("verified_outputs_complete")},
                    "artifacts": {"metrics": metrics, "predictions": binding("predictions"),
                                  "checkpoint": binding("selected_model"), "trace": binding("trace")},
                    "progress": {"epoch": fact(progress["epochs"]), "step": fact(progress["optimizer_steps"]),
                                 "samples": fact(progress["sample_presentations"])},
                    "costs": costs, "missing_evidence": missing}}
        if repo_local_path(root, where["trace"]) != repo_local_path(root, entry["observed"]["artifacts"]["trace"]["locator"]):
            raise ValueError("Descriptor closure trace differs from the inspected trace")
        entries.append(entry)
    descriptor = TerminalDescriptor(spec, {"schema_version": TERMINAL_PROTOCOL,
        "spec_identity": spec.identity, "experiment_id": declaration["experiment_id"],
        "logical_run_id": declaration["logical_run_id"], "arms": entries})
    translate_terminal_descriptor(root, spec, descriptor)
    return descriptor
