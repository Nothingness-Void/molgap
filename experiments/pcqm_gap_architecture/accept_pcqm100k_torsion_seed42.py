"""No-inference acceptance for the paired sparse torsion seed-42 screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


COMPARATOR = "ogb_distance_angle_triangle_edge_state_gps9"
CANDIDATE = "ogb_distance_angle_torsion_triangle_edge_state_gps9"
CANDIDATES = (COMPARATOR, CANDIDATE)
SEED = 42
EXPECTED_PARAMETER_COUNTS = {
    COMPARATOR: 4_891_057,
    CANDIDATE: 4_902_081,
}
BASE_CONTRACT = {
    "batch_size": 48,
    "learning_rate": 1.6e-4,
    "weight_decay": 1.0e-6,
    "max_epochs": 40,
    "patience": 8,
    "precision": "fp32",
    "target": "gap",
    "geometry": "ETKDGv3+MMFF94s-single-conformer-bottom-fusion",
}
EXPECTED_GEOMETRY_SOURCE_COMMIT = "e083bee19ee6a13cd9f72e91229752a9d5f56389"
EXPECTED_PARENT_GRAPH_SHA256 = (
    "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
)
EXPECTED_PARENT_WEDGE_SHA256 = (
    "dc62b8289b0d85bd71a2eca9a16b6223f53206dd9a901670bb799125eff77406"
)
EXPECTED_GEOMETRY_SHA256 = (
    "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_contract(candidate: str) -> dict:
    return {
        **BASE_CONTRACT,
        "torsion": (
            "none"
            if candidate == COMPARATOR
            else "sparse-16-fixed-periodic-shared-gated-update"
        ),
    }


def accept(root: Path, expected_source_commit: str, expected_torsion_sha256: str) -> dict:
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    errors: list[str] = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    required = {
        "format": "molgap-pcqm-gap100k-sparse-torsion-seed42-v1",
        "complete": True,
        "source_commit": expected_source_commit,
        "geometry_source_commit": EXPECTED_GEOMETRY_SOURCE_COMMIT,
        "parent_graph_cache_aggregate_sha256": EXPECTED_PARENT_GRAPH_SHA256,
        "parent_wedge_cache_aggregate_sha256": EXPECTED_PARENT_WEDGE_SHA256,
        "parent_geometry_cache_aggregate_sha256": EXPECTED_GEOMETRY_SHA256,
        "torsion_cache_aggregate_sha256": expected_torsion_sha256,
        "seed": SEED,
        "candidates": list(CANDIDATES),
        "search_budget_s": 23_400,
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
    require(
        0 <= float(selection.get("torsion_cache_valid_geometry_rows", -1)) <= 110_000,
        "valid geometry row count",
    )

    preflight = selection.get("preflight", [])
    require(isinstance(preflight, list) and len(preflight) == 2, "preflight rows")
    if isinstance(preflight, list):
        require(
            [row.get("candidate") for row in preflight] == list(CANDIDATES),
            "preflight identities",
        )
        for row in preflight:
            candidate = row.get("candidate")
            require(
                row.get("parameter_count") == EXPECTED_PARAMETER_COUNTS.get(candidate),
                f"preflight parameters {candidate}",
            )
            for key in (
                "initial_function_match",
                "shared_backbone_parameters_match",
                "torsion_injection_zero",
                "finite_prediction",
                "finite_loss",
                "finite_gradients",
            ):
                require(row.get(key) is True, f"preflight {candidate} {key}")
            require(
                row.get("shared_backbone_mismatches") == [],
                f"preflight {candidate} shared backbone mismatches",
            )
            require(
                row.get("torsion_nonzero_parameters") == [],
                f"preflight {candidate} nonzero torsion injection",
            )

    runs = selection.get("runs", [])
    require(isinstance(runs, list) and len(runs) == 2, "run rows")
    if isinstance(runs, list):
        require(
            [(row.get("seed"), row.get("candidate")) for row in runs]
            == [(SEED, COMPARATOR), (SEED, CANDIDATE)],
            "run order and identities",
        )

    accepted_rows = []
    row_hashes = set()
    target_hashes = set()
    for candidate in CANDIDATES:
        matching = [
            row
            for row in runs
            if row.get("seed") == SEED and row.get("candidate") == candidate
        ]
        require(len(matching) == 1, f"missing run {candidate}")
        metrics_path = root / "results" / candidate / "metrics.json"
        require(metrics_path.is_file(), f"missing metrics {candidate}")
        if not metrics_path.is_file() or len(matching) != 1:
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        require(matching[0] == metrics, f"selection mismatch {candidate}")
        require(
            metrics.get("format") == "molgap-pcqm-gap100k-sparse-torsion-paired-run-v1",
            f"metrics format {candidate}",
        )
        require(metrics.get("complete") is True, f"complete {candidate}")
        require(metrics.get("source_commit") == expected_source_commit, f"source {candidate}")
        require(
            metrics.get("torsion_cache_aggregate_sha256") == expected_torsion_sha256,
            f"torsion cache {candidate}",
        )
        require(metrics.get("seed") == SEED, f"seed {candidate}")
        require(
            metrics.get("parameter_count") == EXPECTED_PARAMETER_COUNTS[candidate],
            f"parameters {candidate}",
        )
        require(metrics.get("parameter_budget") == 5_200_000, f"budget {candidate}")
        require(metrics.get("validation_rows") == 10_000, f"validation rows {candidate}")
        require(
            metrics.get("contract") == expected_contract(candidate),
            f"contract {candidate}",
        )
        require(metrics.get("official_validation_role_read") is False, f"official valid {candidate}")
        require(metrics.get("test_dev_role_read") is False, f"test-dev {candidate}")
        value = metrics.get("validation_gap_mae_eV")
        require(
            isinstance(value, (int, float)) and math.isfinite(value),
            f"MAE {candidate}",
        )
        row_hash = metrics.get("validation_row_index_sha256")
        target_hash = metrics.get("validation_target_sha256")
        require(isinstance(row_hash, str) and len(row_hash) == 64, f"row hash {candidate}")
        require(isinstance(target_hash, str) and len(target_hash) == 64, f"target hash {candidate}")
        row_hashes.add(row_hash)
        target_hashes.add(target_hash)
        artifacts = metrics.get("artifacts", {})
        for name in ("best_model", "checkpoint", "validation_payload", "trace"):
            relative = artifacts.get(name)
            path = root / relative if isinstance(relative, str) else root / "__missing__"
            require(path.is_file(), f"missing {name} {candidate}")
            if path.is_file():
                require(
                    sha256_file(path) == artifacts.get(f"{name}_sha256"),
                    f"hash {name} {candidate}",
                )
        accepted_rows.append(metrics)

    require(len(row_hashes) == 1, "validation row identity differs")
    require(len(target_hashes) == 1, "validation targets differ")
    by_candidate = {row["candidate"]: row for row in accepted_rows}
    comparison = []
    if len(by_candidate) == 2:
        comparator = by_candidate[COMPARATOR]["validation_gap_mae_eV"]
        candidate = by_candidate[CANDIDATE]["validation_gap_mae_eV"]
        delta = candidate - comparator
        comparison = [
            {
                "seed": SEED,
                "comparator_validation_gap_mae_eV": comparator,
                "candidate_validation_gap_mae_eV": candidate,
                "candidate_minus_comparator_eV": delta,
            },
            {
                "seed": "mean",
                "comparator_validation_gap_mae_eV": comparator,
                "candidate_validation_gap_mae_eV": candidate,
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
    result = {
        "format": "molgap-pcqm-gap100k-sparse-torsion-seed42-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": expected_source_commit,
        "torsion_cache_aggregate_sha256": expected_torsion_sha256,
        "paired_comparison": comparison,
        "scientific_gate_passed": passed,
        "selected_candidate": CANDIDATE if passed else COMPARATOR,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "molecular_research_server_accessed": False,
    }
    if errors:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-torsion-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(args.root, args.expected_source_commit, args.expected_torsion_sha256)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
