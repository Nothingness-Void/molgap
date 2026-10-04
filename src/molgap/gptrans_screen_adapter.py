"""Spec-bound call arguments for the existing GPTrans trainer, not admission.

This module imports no torch/model code. It neither executes training nor
certifies input sidecars, release gates, RML plans or replay readiness.
"""
from __future__ import annotations

from pathlib import Path

from .experiment_package import _spec, verify_experiment_source_package
from .gptrans_adapter import gptrans_metadata


def gptrans_screen_arguments(
    spec, arm_id: str, *, package_dir: Path, dataset_root: Path,
    manifest_path: Path, initial_state_path: Path, target_transform_path: Path,
    platform_id: str, path_sidecar_root: Path | None = None,
) -> dict:
    """Resolve a single arm's preflight/training calls without new trainer code.

    Only the input interventions already supported by the V5 trainer are
    qualified here. Registered construction addons are not automatically
    executable V5 screens. All bytes/roles and compute authority are checked
    by the existing release gate, trainer and independent acceptance.
    """
    spec = _spec(spec)
    metadata = gptrans_metadata(spec, arm_id)
    declaration = spec.to_dict()
    if declaration["schema_version"] != "molgap-experiment-spec-v2":
        raise ValueError("Training argument wiring requires per-arm Spec v2 plans")
    if metadata.variant not in {"degree_scale", "path_bond_mean", "degree_path_bond_mean", "degree_scale_ema999",
                                "degree_group_decay_ema999", "degree_path_endpoints_ema999", "degree_pair_depth_scale_ema999", "degree_path_bond_mean_ema999",
                                "degree_node_mean_readout_ema999", "degree_bond_mean_readout_ema999", "degree_decay001_ema999",
                                "degree_node352_ema999", "degree_pair64_ema999", "degree_ffn2_ema999", "degree_bond_local_ema999"}:
        raise ValueError("This variant has no qualified per-arm V5 trace wiring")
    arm = next(item for item in declaration["arms"] if item["arm_id"] == arm_id)
    if arm["training"]["overrides"]:
        raise ValueError("The frozen GPTrans trainer does not accept recipe overrides")
    if arm["initialization"]["kind"] != "frozen_state":
        raise ValueError("GPTrans screen requires a frozen initial state")
    if not platform_id or type(platform_id) is not str:
        raise ValueError("Explicit platform provenance is required")
    if (metadata.variant in {"path_bond_mean", "degree_path_bond_mean", "degree_path_bond_mean_ema999"}) != (path_sidecar_root is not None):
        raise ValueError("Only the path arm requires/consumes an accepted path sidecar")
    package = verify_experiment_source_package(package_dir)
    if package["spec_identity"] != spec.identity:
        raise ValueError("Executable source package differs from the arm Spec")
    plan = next(item for item in declaration["prospective"]["arms"] if item["arm_id"] == arm_id)
    common = {
        "dataset_root": Path(dataset_root), "manifest_path": Path(manifest_path),
        "source_archive": Path(package_dir) / "source.tar.gz",
        "source_archive_sha256": package["archive_sha256"],
        "source_commit": package["source_commit"], "platform_id": platform_id,
        "initial_state_path": Path(initial_state_path), "variant": metadata.variant,
        "target_transform_path": Path(target_transform_path),
        "path_sidecar_root": Path(path_sidecar_root) if path_sidecar_root is not None else None,
    }
    return {
        "spec_identity": spec.identity, "arm_id": arm_id,
        "preflight": common,
        "training": {**common, "v5_audit": True, "trajectory_id": plan["trajectory_id"],
                     "logical_run_id": f"{declaration['logical_run_id']}:{arm_id}"},
        "limitations": ["Call arguments only: not a release gate or execution certificate.",
                        "Persist Spec/arm binding and validate it before any resume."],
    }
