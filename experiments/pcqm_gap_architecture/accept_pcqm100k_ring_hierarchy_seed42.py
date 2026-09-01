"""No-inference acceptance for the ring hierarchy screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


COMPARATOR = "ogb_distance_angle_triangle_edge_state_gps9"
CANDIDATE = "ogb_distance_angle_ring_hierarchy_triangle_edge_state_gps9"
SEED = 42
EXPECTED_PARAMETERS = {COMPARATOR: 4_891_057, CANDIDATE: 4_949_097}
EXPECTED_COMPARATOR_MAE = 0.1353926807641983
EXPECTED_COMPARATOR_HASHES = {
    "metrics.json": "a5ad6ab31df1b5753864860c4bd3352ee21874b3dfcd33d5a4841158736ccb70",
    "best_model.pt": "015f470b687a717690bce9ce3ef4f4198ceecba53e0d3ccf3d10606981c118db",
    "checkpoint.pt": "1c961b8d1962158ac217d2b433b963c018cbe815d32e7d76d95088a43c110f91",
    "trace.json": "b5c6be205ec76c0fd05cc707f031329f2e83fd6b2e0381bcf44b75afe0d02dec",
    "validation_payload.pt": "48d187e7adaf30b67cc2e33ec81656d576a80de22a6ad516aecc2711e07f149c",
}
EXPECTED_RESUME_MANIFEST_SHA256 = (
    "9d0f4ccc5f315dd5c7f5fe9305bb6cd36f1bd88659bffeea96711525678c77f9"
)
EXPECTED_VALIDATION_ROW_SHA256 = (
    "4045acbbb0e359f11e0479cac3e24f1b038a7392f0fc4eabc382da68ef83882b"
)
EXPECTED_VALIDATION_TARGET_SHA256 = (
    "7920d73338f063d2fab6ceca5f124dcc7fe2c2863d87ca718f77fc7b707c3a94"
)
BASE_CONTRACT = {
    "batch_size": 48,
    "learning_rate": 1.6e-4,
    "weight_decay": 1.0e-6,
    "max_epochs": 40,
    "patience": 8,
    "precision": "fp32",
    "target": "gap",
    "geometry": "ETKDGv3+MMFF94s-single-conformer-bottom-fusion",
    "ring_hierarchy": "symmsssr-ring64+shared-four-point-update+rank32-ring-to-atom",
    "ring_update_layers": [2, 4, 6, 8],
    "ring_channels": 64,
    "ring_feature_channels": 12,
    "ring_edge_channels": 4,
    "ring_to_atom_exchange_rank": 32,
    "pooling": "atom_mean_only",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accept(root: Path, expected_source_commit: str, expected_cache_sha256: str) -> dict:
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    selection_path = root / "selection.json"
    require(selection_path.is_file(), "missing selection.json")
    selection = (
        json.loads(selection_path.read_text(encoding="utf-8"))
        if selection_path.is_file()
        else {}
    )
    required = {
        "format": "molgap-pcqm-gap100k-ring-hierarchy-seed42-v1",
        "complete": True,
        "source_commit": expected_source_commit,
        "ring_cache_aggregate_sha256": expected_cache_sha256,
        "seed": SEED,
        "candidates": [COMPARATOR, CANDIDATE],
        "search_budget_s": 14_400,
        "atomic_checkpoints": True,
        "independently_retrievable_outputs": True,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "molecular_research_server_accessed": False,
        "seed43_44_submitted": False,
        "full_data_authorized": False,
    }
    for key, value in required.items():
        require(selection.get(key) == value, key)

    preflight_path = root / "preflight.json"
    require(preflight_path.is_file(), "missing preflight.json")
    if preflight_path.is_file():
        preflight_document = json.loads(preflight_path.read_text(encoding="utf-8"))
        require(
            preflight_document.get("format")
            == "molgap-pcqm-gap100k-ring-hierarchy-preflight-v1",
            "preflight format",
        )
        require(preflight_document.get("complete") is True, "preflight complete")
        require(
            preflight_document.get("source_commit") == expected_source_commit,
            "preflight source",
        )
        require(
            preflight_document.get("ring_cache_aggregate_sha256")
            == expected_cache_sha256,
            "preflight cache",
        )

    preflight = selection.get("preflight", [])
    require(isinstance(preflight, list) and len(preflight) == 2, "preflight rows")
    if isinstance(preflight, list):
        require(
            [row.get("candidate") for row in preflight]
            == [COMPARATOR, CANDIDATE],
            "preflight identities",
        )
        for row in preflight:
            identity = row.get("candidate")
            require(
                row.get("parameter_count") == EXPECTED_PARAMETERS.get(identity),
                f"preflight parameters {identity}",
            )
            for key in (
                "initial_function_match",
                "shared_backbone_parameters_match",
                "ring_injection_zero",
                "ring_return_gradient_nonzero",
                "finite_prediction",
                "finite_loss",
                "finite_gradients",
            ):
                require(row.get(key) is True, f"preflight {identity} {key}")
            require(
                row.get("shared_backbone_mismatches") == [],
                f"preflight backbone {identity}",
            )
            require(
                row.get("ring_injection_nonzero_parameters") == [],
                f"preflight injection {identity}",
            )

    frozen = selection.get("frozen_comparator", {})
    frozen_metrics = frozen.get("metrics", {}) if isinstance(frozen, dict) else {}
    require(frozen.get("source_kernel") == "nothingnessvoid/molgap-pcqm-sparse-torsion-s42", "frozen kernel")
    require(frozen.get("source_kernel_version") == 3, "frozen kernel version")
    require(
        frozen.get("resume_manifest_sha256") == EXPECTED_RESUME_MANIFEST_SHA256,
        "frozen resume manifest",
    )
    require(frozen.get("artifact_sha256") == EXPECTED_COMPARATOR_HASHES, "frozen artifact hashes")
    comparator_required = {
        "candidate": COMPARATOR,
        "complete": True,
        "seed": SEED,
        "parameter_count": EXPECTED_PARAMETERS[COMPARATOR],
        "validation_gap_mae_eV": EXPECTED_COMPARATOR_MAE,
        "validation_rows": 10_000,
        "validation_row_index_sha256": EXPECTED_VALIDATION_ROW_SHA256,
        "validation_target_sha256": EXPECTED_VALIDATION_TARGET_SHA256,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    for key, value in comparator_required.items():
        require(frozen_metrics.get(key) == value, f"frozen comparator {key}")

    runs = selection.get("runs", [])
    require(isinstance(runs, list) and len(runs) == 1, "candidate run rows")
    candidate_metrics_path = root / "results" / CANDIDATE / "metrics.json"
    require(candidate_metrics_path.is_file(), "missing candidate metrics")
    candidate_metrics = (
        json.loads(candidate_metrics_path.read_text(encoding="utf-8"))
        if candidate_metrics_path.is_file()
        else {}
    )
    if isinstance(runs, list) and len(runs) == 1:
        require(runs[0] == candidate_metrics, "selection candidate mismatch")
    candidate_required = {
        "format": "molgap-pcqm-gap100k-ring-hierarchy-run-v1",
        "complete": True,
        "candidate": CANDIDATE,
        "source_commit": expected_source_commit,
        "ring_cache_aggregate_sha256": expected_cache_sha256,
        "seed": SEED,
        "parameter_count": EXPECTED_PARAMETERS[CANDIDATE],
        "parameter_budget": 5_000_000,
        "validation_rows": 10_000,
        "validation_row_index_sha256": EXPECTED_VALIDATION_ROW_SHA256,
        "validation_target_sha256": EXPECTED_VALIDATION_TARGET_SHA256,
        "contract": BASE_CONTRACT,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    for key, value in candidate_required.items():
        require(candidate_metrics.get(key) == value, f"candidate {key}")
    mae = candidate_metrics.get("validation_gap_mae_eV")
    require(isinstance(mae, (int, float)) and math.isfinite(mae), "candidate MAE")
    artifacts = candidate_metrics.get("artifacts", {})
    for name in ("best_model", "checkpoint", "validation_payload", "trace"):
        relative = artifacts.get(name)
        path = root / relative if isinstance(relative, str) else root / "__missing__"
        require(path.is_file(), f"missing candidate {name}")
        if path.is_file():
            require(
                sha256_file(path) == artifacts.get(f"{name}_sha256"),
                f"candidate {name} hash",
            )

    comparison = []
    if isinstance(mae, (int, float)) and math.isfinite(mae):
        delta = float(mae) - EXPECTED_COMPARATOR_MAE
        comparison = [
            {
                "seed": SEED,
                "comparator_validation_gap_mae_eV": EXPECTED_COMPARATOR_MAE,
                "candidate_validation_gap_mae_eV": float(mae),
                "candidate_minus_comparator_eV": delta,
            },
            {
                "seed": "mean",
                "comparator_validation_gap_mae_eV": EXPECTED_COMPARATOR_MAE,
                "candidate_validation_gap_mae_eV": float(mae),
                "candidate_minus_comparator_eV": delta,
            },
        ]
    passed = bool(comparison) and comparison[0]["candidate_minus_comparator_eV"] < 0
    require(selection.get("paired_comparison") == comparison, "paired comparison")
    require(selection.get("scientific_gate_passed") is passed, "scientific gate")
    require(
        selection.get("selected_candidate") == (CANDIDATE if passed else COMPARATOR),
        "selected candidate",
    )
    return {
        "format": "molgap-pcqm-gap100k-ring-hierarchy-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": expected_source_commit,
        "ring_cache_aggregate_sha256": expected_cache_sha256,
        "candidate_parameter_count": EXPECTED_PARAMETERS[CANDIDATE],
        "paired_comparison": comparison,
        "scientific_gate_passed": passed,
        "selected_candidate": CANDIDATE if passed else COMPARATOR,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("expected_source_commit")
    parser.add_argument("expected_cache_sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(args.root, args.expected_source_commit, args.expected_cache_sha256)
    output = args.output or args.root / "acceptance.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

