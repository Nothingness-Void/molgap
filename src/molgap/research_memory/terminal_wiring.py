"""Terminal-side trace closure and orchestration contract.

Enforces that experiments with retained training trace evidence cannot silently
finalize with trace_status=unavailable.  Maintains RML core 1:1 finalization model:
1 trajectory + 1 terminal artifact + 0 or 1 canonical trace = 1 RML finalization.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from molgap.evidence_pointers import load_json_object
from .paths import repo_local_path
from .pipeline import finalize_rebuild_backtest
from .recovery import recover_trace
from .trace import atomic_write, canonicalize_trace, file_digest, json_bytes, load_canonical_trace


def is_trace_artifact(artifact: Mapping[str, Any]) -> bool:
    """Check whether an artifact declaration in V5 evidence represents a retained training trace.

    Prioritizes semantic fields (role, kind, artifact_type, purpose, category).
    Explicitly excludes non-training traces (analysis, debug, inference, profile, eval).
    Uses filename heuristics only as a backward-compatible fallback.
    """
    if not isinstance(artifact, Mapping):
        return False

    name = str(artifact.get("name", "")).lower()
    locator = str(artifact.get("locator", "")).lower()

    # 1. Semantic fields evaluation (primary truth)
    for key in ("artifact_type", "kind", "role", "category", "purpose"):
        val = str(artifact.get(key, "")).lower()
        if val in {"training_trace", "train_trace", "training_log", "training_history"}:
            return True
        if any(
            non in val for non in ("inference", "analysis", "debug", "profile", "eval", "test")
        ):
            return False

    # 2. Explicit exclusion of non-training traces by name or locator
    non_training_markers = ("analysis", "debug", "inference", "profile", "eval", "test_trace")
    if any(m in name for m in non_training_markers):
        return False
    if any(m in locator for m in non_training_markers):
        return False

    # 3. Filename / locator heuristics for training traces (compatibility fallback)
    if name in {"trace", "training_trace", "raw_trace", "train_trace", "train_log"}:
        return True
    if name.endswith("_trace"):
        return True
    for ext in (
        "/trace.json",
        "\\trace.json",
        "trace.json",
        "/trace.csv",
        "\\trace.csv",
        "trace.csv",
        "train_log.csv",
    ):
        if locator.endswith(ext):
            return True

    return False


def inspect_trace_retention_evidence(
    evidence: Mapping[str, Any],
    arm_identifier: str | None = None,
) -> tuple[bool, dict[str, Any] | None]:
    """Inspect V5 evidence artifacts to determine whether retained training trace is declared.

    Returns:
        (trace_declared, matching_artifact_dict_or_None)
    """
    artifacts = evidence.get("artifacts", [])
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        return False, None

    trace_artifacts = [dict(a) for a in artifacts if is_trace_artifact(a)]
    if not trace_artifacts:
        return False, None

    if len(trace_artifacts) == 1:
        # If arm_identifier is given, verify it does not conflict with sibling arm artifacts
        if arm_identifier:
            arm_clean = arm_identifier.lower().replace("-", "_")
            loc_clean = str(trace_artifacts[0].get("locator", "")).lower().replace("-", "_")
            name_clean = str(trace_artifacts[0].get("name", "")).lower().replace("-", "_")
            if arm_clean not in loc_clean and arm_clean not in name_clean:
                sibling_arm_artifacts = [
                    a
                    for a in artifacts
                    if arm_clean in str(a.get("name", "")).lower().replace("-", "_")
                    or arm_clean in str(a.get("locator", "")).lower().replace("-", "_")
                ]
                if sibling_arm_artifacts:
                    raise ValueError(
                        f"FAIL CLOSED: arm identifier '{arm_identifier}' does not match "
                        f"the single declared trace artifact '{trace_artifacts[0].get('locator')}'"
                    )
        return True, trace_artifacts[0]

    # Multiple trace artifacts declared (e.g. multi-arm joint experiment).
    if arm_identifier:
        arm_clean = arm_identifier.lower().replace("-", "_")
        matched = [
            a
            for a in trace_artifacts
            if arm_clean in str(a.get("locator", "")).lower().replace("-", "_")
            or arm_clean in str(a.get("name", "")).lower().replace("-", "_")
        ]
        if len(matched) == 1:
            candidate = matched[0]
            # Sibling provenance compatibility check:
            # If evidence has other artifacts for this arm (e.g. best_model, predictions),
            # verify that the candidate trace aligns with the arm identifier
            sibling_arm_artifacts = [
                a
                for a in artifacts
                if (
                    arm_clean in str(a.get("name", "")).lower().replace("-", "_")
                    or arm_clean in str(a.get("locator", "")).lower().replace("-", "_")
                )
                and not is_trace_artifact(a)
            ]
            if sibling_arm_artifacts:
                cand_parent = Path(candidate["locator"]).parent.as_posix().lower()
                if arm_clean not in cand_parent and arm_clean not in str(candidate.get("name", "")).lower():
                    raise ValueError(
                        f"FAIL CLOSED: candidate trace '{candidate['locator']}' does not match "
                        f"sibling arm artifact provenance for arm '{arm_identifier}'"
                    )
            return True, candidate
        if not matched:
            raise ValueError(
                f"FAIL CLOSED: retention evidence declares {len(trace_artifacts)} traces, "
                f"but none matched arm identifier '{arm_identifier}': "
                f"{[a.get('locator') for a in trace_artifacts]}"
            )
        raise ValueError(
            f"FAIL CLOSED: ambiguous trace retention evidence for arm '{arm_identifier}': "
            f"{[a.get('locator') for a in matched]}"
        )

    # Multiple trace artifacts present without arm identifier
    raise ValueError(
        f"FAIL CLOSED: ambiguous multi-arm evidence ({len(trace_artifacts)} traces declared); "
        f"explicit arm_identifier is required: "
        f"{[a.get('locator') for a in trace_artifacts]}"
    )


def resolve_metric_semantics(
    repo_root: str | Path,
    trajectory_obj: Mapping[str, Any],
    terminal_obj: Mapping[str, Any],
    raw_data: Mapping[str, Any],
    recovery_spec: Mapping[str, Any] | None = None,
    has_ema_metric: bool = False,
) -> dict[str, Any]:
    """Resolve authoritative metric semantics without guessing scientific defaults.

    Sources checked in order:
    1. explicit recovery_spec["metric_semantics"]
    2. raw_data["metric_semantics"]
    3. terminal_obj.get("evidence", {}).get("metric_semantics")
    4. contract["metric_semantics"] from trajectory["state_at_start"]["contract_refs"]
    5. contract target/metric declarations (contract["target"], contract["metric"], contract["unit"])
    """
    root = Path(repo_root).resolve()

    # 1. recovery_spec
    if recovery_spec and isinstance(recovery_spec, Mapping) and recovery_spec.get("metric_semantics"):
        ms = recovery_spec["metric_semantics"]
        if isinstance(ms, Mapping) and ms.get("live_train_metric"):
            return dict(ms)

    # 2. raw_data
    if isinstance(raw_data, Mapping) and raw_data.get("metric_semantics"):
        ms = raw_data["metric_semantics"]
        if isinstance(ms, Mapping) and ms.get("live_train_metric"):
            return dict(ms)

    # 3. evidence
    ev = terminal_obj.get("evidence", {}) if isinstance(terminal_obj, Mapping) else {}
    if isinstance(ev, Mapping) and ev.get("metric_semantics"):
        ms = ev["metric_semantics"]
        if isinstance(ms, Mapping) and ms.get("live_train_metric"):
            return dict(ms)

    # 4 & 5. contract
    state = trajectory_obj.get("state_at_start", {}) if isinstance(trajectory_obj, Mapping) else {}
    contract_refs = state.get("contract_refs", [])
    contract_obj: dict[str, Any] = {}
    if contract_refs:
        c_path = repo_local_path(root, contract_refs[0])
        if c_path.is_file():
            try:
                contract_obj = load_json_object(c_path)
            except Exception:
                contract_obj = {}

    if contract_obj.get("metric_semantics"):
        ms = contract_obj["metric_semantics"]
        if isinstance(ms, Mapping) and ms.get("live_train_metric"):
            return dict(ms)

    # Check explicit contract metric declarations
    target = contract_obj.get("target")
    metric = contract_obj.get("metric")
    unit = contract_obj.get("unit")
    if target and metric and unit:
        direction = contract_obj.get("direction", "minimize")
        train_role = contract_obj.get("train_role", "internal_train")
        dev_role = contract_obj.get("development_role", "internal_development")
        return {
            "live_train_metric": {
                "metric": str(metric),
                "unit": str(unit),
                "target": str(target),
                "role_identity": str(train_role),
                "weights": "live",
                "direction": str(direction),
            },
            "live_dev_metric": None,
            "ema_dev_metric": {
                "metric": str(metric),
                "unit": str(unit),
                "target": str(target),
                "role_identity": str(dev_role),
                "weights": "ema",
                "direction": str(direction),
            }
            if has_ema_metric
            else None,
        }

    # Authoritative determination failed
    raise ValueError(
        "FAIL CLOSED: cannot determine authoritative metric semantics for raw trace "
        "(missing metric/unit/target in recovery_spec, raw trace, evidence, or contract). "
        "Please provide an explicit recovery_spec or authoritative contract metric semantics."
    )


def resolve_trace_for_terminal_arm(
    repo_root: str | Path,
    trajectory: str | Path | Mapping[str, Any],
    terminal: str | Path | Mapping[str, Any],
    *,
    arm_identifier: str | None = None,
    trace_source: str | Path | None = None,
    canonical_trace: str | Path | None = None,
    canonical_trace_output: str | Path | None = None,
    recovery_spec: dict[str, Any] | None = None,
) -> tuple[bool, Path | None]:
    """Resolve and validate the canonical trace for an arm based on retention evidence.

    Returns:
        (trace_required, canonical_trace_path_or_None)
    """
    root = Path(repo_root).resolve()

    # Load trajectory identity
    if isinstance(trajectory, (str, Path)):
        traj_data = load_json_object(repo_local_path(root, trajectory))
    else:
        traj_data = dict(trajectory)
    trajectory_id = traj_data["trajectory_id"]

    # Load terminal package & evidence
    if isinstance(terminal, (str, Path)):
        term_data = load_json_object(repo_local_path(root, terminal))
    else:
        term_data = dict(terminal)
    run_id = term_data["run_id"]
    evidence = term_data.get("evidence", {})

    trace_declared, declared_artifact = inspect_trace_retention_evidence(
        evidence, arm_identifier=arm_identifier
    )
    trace_required = trace_declared or (trace_source is not None) or (canonical_trace is not None)

    if not trace_required:
        return False, None

    # Step 1: Caller directly provided canonical trace
    if canonical_trace is not None:
        c_path = repo_local_path(root, canonical_trace)
        if not c_path.is_file():
            raise FileNotFoundError(
                f"FAIL CLOSED: declared canonical trace file does not exist: {c_path}"
            )
        trace_record = load_canonical_trace(c_path)
        if trace_record.get("trajectory_id") != trajectory_id:
            raise ValueError(
                f"FAIL CLOSED: canonical trace trajectory_id mismatch: "
                f"trace has '{trace_record.get('trajectory_id')}', but arm trajectory is '{trajectory_id}'"
            )
        if trace_record.get("run_id") != run_id:
            raise ValueError(
                f"FAIL CLOSED: canonical trace run_id mismatch: "
                f"trace has '{trace_record.get('run_id')}', but arm terminal run_id is '{run_id}'"
            )

        # Enforce provenance and digest binding to retention evidence
        if declared_artifact is not None:
            decl_loc = str(declared_artifact.get("locator", "")).replace("\\", "/")
            decl_sha = str(declared_artifact.get("sha256", ""))
            c_rel_posix = c_path.relative_to(root).as_posix()

            # Situation A: declared artifact IS the canonical trace itself
            is_declared_itself = (c_rel_posix == decl_loc) or (
                decl_loc.endswith(".json") and c_path.name == Path(decl_loc).name
            )

            actual_digest = file_digest(c_path)

            if is_declared_itself:
                if decl_sha and actual_digest != decl_sha:
                    raise ValueError(
                        f"FAIL CLOSED: explicit canonical trace digest mismatch with retention evidence: "
                        f"expected {decl_sha}, got {actual_digest}"
                    )
            else:
                # Situation B: declared artifact is raw trace (or another artifact).
                # The passed canonical trace must prove in its provenance that it binds to declared_artifact!
                provenance_list = trace_record.get("provenance", [])
                if not isinstance(provenance_list, Sequence) or isinstance(
                    provenance_list, (str, bytes)
                ):
                    raise ValueError(
                        f"FAIL CLOSED: explicit canonical trace missing valid provenance array "
                        f"to bind against declared trace artifact '{decl_loc}'"
                    )

                matched_provenance = False
                for prov in provenance_list:
                    if not isinstance(prov, Mapping):
                        continue
                    p_src = str(prov.get("source", "")).replace("\\", "/")
                    p_sha = str(prov.get("sha256", ""))
                    src_matches = (
                        (p_src == decl_loc)
                        or p_src.endswith(decl_loc)
                        or decl_loc.endswith(p_src)
                    )
                    sha_matches = (not decl_sha) or (p_sha.lower() == decl_sha.lower())
                    if src_matches and sha_matches:
                        matched_provenance = True
                        break

                if not matched_provenance:
                    raise ValueError(
                        f"FAIL CLOSED: explicit canonical trace provenance does not bind to declared trace artifact "
                        f"'{decl_loc}' with SHA256 '{decl_sha}'"
                    )

        return True, c_path

    # Step 2: Resolve raw trace source
    if trace_source is not None:
        source_path = repo_local_path(root, trace_source)
    elif declared_artifact is not None:
        source_path = repo_local_path(root, declared_artifact["locator"])
    else:
        raise ValueError(
            "FAIL CLOSED: training trace is required by retention evidence, "
            "but no trace source path could be resolved."
        )

    if not source_path.is_file():
        raise FileNotFoundError(
            f"FAIL CLOSED: declared retained trace file missing: {source_path}"
        )

    # Verify digest against declared evidence if present
    if declared_artifact and declared_artifact.get("sha256"):
        current_digest = file_digest(source_path)
        if current_digest != declared_artifact["sha256"]:
            raise ValueError(
                f"FAIL CLOSED: retained trace file digest mismatch for {source_path}: "
                f"expected {declared_artifact['sha256']}, got {current_digest}"
            )

    # Step 3: Check if source_path is already a valid canonical trace
    is_canonical = False
    try:
        existing = load_canonical_trace(source_path)
        is_canonical = existing.get("schema") == "molgap-trace-v1"
    except Exception:
        is_canonical = False

    if is_canonical:
        if existing.get("trajectory_id") != trajectory_id:
            raise ValueError(
                f"FAIL CLOSED: trace trajectory_id mismatch for {source_path}: "
                f"trace has '{existing.get('trajectory_id')}', but arm trajectory is '{trajectory_id}'"
            )
        if existing.get("run_id") != run_id:
            raise ValueError(
                f"FAIL CLOSED: trace run_id mismatch for {source_path}: "
                f"trace has '{existing.get('run_id')}', but arm terminal run_id is '{run_id}'"
            )
        return True, source_path

    # Step 4: Canonicalize raw trace
    if canonical_trace_output is not None:
        output_path = repo_local_path(root, canonical_trace_output)
    else:
        output_path = repo_local_path(
            root, source_path.parent / f"canonical_trace_{trajectory_id}.json"
        )

    if recovery_spec is not None and "field_mapping" in recovery_spec:
        full_spec = dict(recovery_spec)
        full_spec.setdefault("trajectory_id", trajectory_id)
        full_spec.setdefault("run_id", run_id)
        full_spec.setdefault("rows_key", "rows")
        recover_trace([source_path], full_spec, output_path, repo_root=root)
    else:
        # Standard observation extractor for raw training trace JSON
        raw_data = json.loads(source_path.read_text(encoding="utf-8"))
        rows = raw_data.get("rows") if isinstance(raw_data, dict) else raw_data
        if not isinstance(rows, list):
            raise ValueError(
                f"FAIL CLOSED: raw trace {source_path} has unknown structure (no rows array)"
            )

        observations = []
        for idx, r in enumerate(rows):
            obs = {
                "sequence": idx,
                "event": "observation",
                "epoch_or_pass": r.get("epoch", r.get("epoch_or_pass", idx)),
                "optimizer_step": r.get("optimizer_step", r.get("step")),
                "sample_presentations": r.get("sample_presentations"),
                "learning_rate": r.get("learning_rate", r.get("lr")),
                "live_train_metric": r.get(
                    "train_mae_eV", r.get("train_loss", r.get("live_train_metric"))
                ),
                "live_dev_metric": r.get("live_dev_metric"),
                "ema_dev_metric": r.get(
                    "development_mae_eV", r.get("dev_mae_eV", r.get("ema_dev_metric"))
                ),
                "wall_time_seconds": r.get("elapsed_seconds", r.get("wall_time_seconds")),
            }
            observations.append(obs)

        has_ema = any(o.get("ema_dev_metric") is not None for o in observations)

        metric_semantics = resolve_metric_semantics(
            repo_root=root,
            trajectory_obj=traj_data,
            terminal_obj=term_data,
            raw_data=raw_data if isinstance(raw_data, dict) else {},
            recovery_spec=recovery_spec,
            has_ema_metric=has_ema,
        )

        trace_record = canonicalize_trace(
            {
                "trajectory_id": trajectory_id,
                "run_id": run_id,
                "metric_semantics": metric_semantics,
                "observations": observations,
                "provenance": [
                    {
                        "source": source_path.relative_to(root).as_posix(),
                        "sha256": file_digest(source_path),
                        "recovered_at_terminal_closure": True,
                    }
                ],
            }
        )
        atomic_write(output_path, json_bytes(trace_record))

    validated = load_canonical_trace(output_path)
    if validated.get("trajectory_id") != trajectory_id:
        raise ValueError(
            f"FAIL CLOSED: canonicalized trace trajectory_id mismatch for {output_path}"
        )
    if validated.get("run_id") != run_id:
        raise ValueError(
            f"FAIL CLOSED: canonicalized trace run_id mismatch for {output_path}"
        )

    return True, output_path


def build_default_trace_manifest(
    repo_root: str | Path,
    trajectory: Mapping[str, Any],
    terminal: Mapping[str, Any],
    trace_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a default trace manifest when terminal package omitted one.

    Derives facts strictly from frozen contract, prospective trajectory, terminal package,
    and mechanical trace observations. Never synthesizes speculative training/dataset provenance.
    """
    root = Path(repo_root).resolve()
    traj_id = trajectory["trajectory_id"]
    run_id = terminal["run_id"]
    state = trajectory.get("state_at_start", {})
    contract_refs = state.get("contract_refs", [])
    contract_ref = contract_refs[0] if contract_refs else ""
    reference_id = state.get("reference_ids", [""])[0] if state.get("reference_ids") else ""

    # Load frozen contract if available to extract authoritative identities
    contract_obj: dict[str, Any] = {}
    if contract_ref:
        c_path = repo_local_path(root, contract_ref)
        if c_path.is_file():
            try:
                contract_obj = load_json_object(c_path)
            except Exception:
                contract_obj = {}

    # Derive identities strictly from authoritative sources without guessing
    dataset_identity = (
        contract_obj.get("data_role_fingerprint")
        or contract_obj.get("dataset_identity")
        or "unspecified_in_contract"
    )
    row_split_identity = (
        contract_obj.get("row_order_fingerprint")
        or contract_obj.get("row_split_identity")
        or "unspecified_in_contract"
    )
    model_identity = (
        contract_obj.get("model_id")
        or contract_obj.get("model_identity")
        or "unspecified_in_contract"
    )
    scientific_contract = (
        contract_obj.get("benchmark_id")
        or contract_obj.get("format")
        or contract_ref
        or "unspecified_in_contract"
    )
    optimizer_identity = contract_obj.get("optimizer", "unspecified_in_contract")
    lr_schedule_identity = contract_obj.get("lr_schedule", "unspecified_in_contract")
    target_transform_identity = contract_obj.get("target_transform", "unspecified_in_contract")
    precision_identity = contract_obj.get("precision", "unspecified_in_contract")

    # Mechanical facts from trace observations
    observations = trace_record.get("observations", [])
    has_step = any(o.get("optimizer_step") is not None for o in observations)
    has_samples = any(o.get("sample_presentations") is not None for o in observations)
    has_ema = any(o.get("ema_dev_metric") is not None for o in observations)
    has_live_dev = any(o.get("live_dev_metric") is not None for o in observations)

    max_steps = max(
        (o.get("optimizer_step") for o in observations if o.get("optimizer_step") is not None),
        default=None,
    )
    max_samples = max(
        (
            o.get("sample_presentations")
            for o in observations
            if o.get("sample_presentations") is not None
        ),
        default=None,
    )

    x_axis = "optimizer_steps" if has_step else ("presentations" if has_samples else "epochs")

    return {
        "schema": "molgap-trace-manifest-v1",
        "trajectory_id": traj_id,
        "run_id": run_id,
        "comparison_role": "candidate",
        "reference_id": reference_id,
        "contract_ref": contract_ref,
        "terminal_evidence_ref": "pending",
        "model_identity": str(model_identity),
        "evaluation_role_identity": "internal_development",
        "selection_semantics": "best_development_metric",
        "weight_semantics": "ema_and_live" if has_ema else "live",
        "metric_semantics": "Gap MAE in eV",
        "presentation_semantics_ref": contract_ref,
        "trace_artifact_ref": "pending",
        "x_axis": x_axis,
        "exposure": {
            "optimizer_steps": max_steps,
            "sample_presentations": max_samples,
        },
        "comparability_identity": {
            "scientific_contract": str(scientific_contract),
            "dataset_identity": str(dataset_identity),
            "row_split_identity": str(row_split_identity),
            "architecture_identity": str(model_identity),
            "optimizer_identity": str(optimizer_identity),
            "lr_schedule_identity": str(lr_schedule_identity),
            "target_transform_identity": str(target_transform_identity),
            "precision_identity": str(precision_identity),
            "ema_semantics": "ema" if has_ema else "none",
            "evaluation_role_identity": "internal_development",
            "selection_role_identity": "best_development_metric",
            "terminal_endpoint_identity": f"endpoint_{max_steps or len(observations)}",
            "matched_architecture_required": False,
            "x_axis_semantics": x_axis,
        },
        "backtest_eligibility": {
            "eligible": False,
            "exclusion_reasons": ["closure_synthesized_manifest_not_prelaunch_calibrated"],
        },
        "trace_fields": {
            "epoch_or_pass": True,
            "optimizer_step": bool(has_step),
            "sample_presentations": bool(has_samples),
            "learning_rate": True,
            "live_train_metric": True,
            "live_dev_metric": bool(has_live_dev),
            "ema_dev_metric": bool(has_ema),
            "checkpoint_identity": False,
        },
    }


def close_terminal_arm(
    repo_root: str | Path,
    trajectory: str | Path,
    terminal: str | Path,
    *,
    trace: str | Path | None = None,
    arm_identifier: str | None = None,
    trace_source: str | Path | None = None,
    canonical_trace_output: str | Path | None = None,
    recovery_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute terminal trace closure and invoke terminal-pipeline with strict fail-closed guard."""
    root = Path(repo_root).resolve()
    term_obj = load_json_object(repo_local_path(root, terminal))
    evidence = term_obj.get("evidence", {})

    trace_required, auto_trace = resolve_trace_for_terminal_arm(
        root,
        trajectory,
        terminal,
        arm_identifier=arm_identifier,
        trace_source=trace_source,
        canonical_trace=trace,
        canonical_trace_output=canonical_trace_output,
        recovery_spec=recovery_spec,
    )

    terminal_to_use = terminal
    if trace_required:
        # Strict guard: caller must not supply None or fail to resolve canonical trace
        final_trace = trace or auto_trace
        if final_trace is None:
            raise ValueError(
                "FAIL CLOSED: retained training trace is required by evidence, "
                "but no canonical trace was provided or resolved. "
                "Refusing silent finalization with trace_status=unavailable."
            )
        # Verify trace identity and validity before passing to pipeline
        trace_rec = load_canonical_trace(repo_local_path(root, final_trace))
        traj_obj = load_json_object(repo_local_path(root, trajectory))
        if trace_rec.get("trajectory_id") != traj_obj.get("trajectory_id"):
            raise ValueError(
                f"FAIL CLOSED: trace trajectory_id mismatch: "
                f"'{trace_rec.get('trajectory_id')}' != '{traj_obj.get('trajectory_id')}'"
            )
        if trace_rec.get("run_id") != term_obj.get("run_id"):
            raise ValueError(
                f"FAIL CLOSED: trace run_id mismatch: "
                f"'{trace_rec.get('run_id')}' != '{term_obj.get('run_id')}'"
            )
        if term_obj.get("trace_manifest") is None:
            manifest = build_default_trace_manifest(root, traj_obj, term_obj, trace_rec)
            term_obj_copy = copy.deepcopy(term_obj)
            term_obj_copy["trace_manifest"] = manifest
            term_path = repo_local_path(root, terminal)
            target_term_path = (
                term_path.parent / f"terminal_with_manifest_{traj_obj['trajectory_id']}.json"
            )
            target_term_path.write_bytes(json_bytes(term_obj_copy))
            terminal_to_use = target_term_path
    else:
        final_trace = trace

    result = finalize_rebuild_backtest(root, trajectory, terminal_to_use, trace=final_trace)

    if trace_required:
        if result.get("pipeline_status") != "COMPLETE":
            raise RuntimeError(
                f"FAIL CLOSED: terminal-pipeline failed for required trace: {result.get('error')}"
            )
        # Verify durable receipt has trace_status = available
        traj_path = repo_local_path(root, trajectory)
        receipt_path = traj_path.parent / "rml_finalized/finalization.json"
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("trace_status") != "available":
                raise RuntimeError(
                    f"FAIL CLOSED: trace was required but final receipt recorded "
                    f"trace_status='{receipt.get('trace_status')}'"
                )

    return result


def close_terminal_multi_arm(
    repo_root: str | Path,
    arms: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Execute terminal closure independently across multiple arms of an experiment.

    Each arm maintains its own 1 trajectory : 0/1 trace finalization transaction.
    """
    root = Path(repo_root).resolve()
    results = []
    for arm in arms:
        res = close_terminal_arm(
            repo_root=root,
            trajectory=arm["trajectory"],
            terminal=arm["terminal"],
            trace=arm.get("trace"),
            arm_identifier=arm.get("arm_identifier") or arm.get("arm"),
            trace_source=arm.get("trace_source") or arm.get("raw_trace_path"),
            canonical_trace_output=arm.get("canonical_trace_output"),
            recovery_spec=arm.get("recovery_spec"),
        )
        results.append(res)
    return results
