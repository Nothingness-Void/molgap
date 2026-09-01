"""No-inference acceptance for paired geometry-confirmation seeds 43/44."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


COMPARATOR = "ogb_sparse_triangle_edge_state_gps9"
CANDIDATE = "ogb_distance_angle_triangle_edge_state_gps9"
CANDIDATES = (COMPARATOR, CANDIDATE)
SEEDS = (43, 44)
SEED42_COMPARATOR_MAE = 0.13790177369117737
SEED42_CANDIDATE_MAE = 0.13559719920158386
EXPECTED_PARAMETER_COUNTS = {
    COMPARATOR: 4_878_257,
    CANDIDATE: 4_891_057,
}
BASE_CONTRACT = {
    "batch_size": 48,
    "learning_rate": 1.6e-4,
    "weight_decay": 1.0e-6,
    "max_epochs": 40,
    "patience": 8,
    "precision": "fp32",
    "target": "gap",
}


def expected_contract(candidate: str) -> dict:
    return {
        **BASE_CONTRACT,
        "geometry": (
            "none"
            if candidate == COMPARATOR
            else "ETKDGv3+MMFF94s-single-conformer-bottom-fusion"
        ),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accept(
    root: Path,
    expected_source_commit: str,
    expected_geometry_sha256: str,
) -> dict:
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    errors = []

    def require(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    require(
        selection.get("format")
        == "molgap-pcqm-gap100k-geometry-bottom-fusion-multiseed-v1",
        "format",
    )
    require(selection.get("complete") is True, "complete")
    require(selection.get("source_commit") == expected_source_commit, "source")
    require(
        selection.get("geometry_cache_aggregate_sha256")
        == expected_geometry_sha256,
        "geometry cache SHA",
    )
    require(
        isinstance(selection.get("geometry_valid_fraction"), (int, float))
        and selection["geometry_valid_fraction"] >= 0.99,
        "geometry valid fraction",
    )
    require(selection.get("seeds") == list(SEEDS), "seeds")
    require(selection.get("candidates") == list(CANDIDATES), "candidates")
    require(selection.get("search_budget_s") == 39_600, "search budget")
    require(selection.get("official_validation_role_read") is False, "official valid")
    require(selection.get("test_dev_role_read") is False, "test-dev")

    preflight = selection.get("preflight", [])
    require(isinstance(preflight, list) and len(preflight) == 2, "preflight rows")
    if isinstance(preflight, list):
        require(
            [row.get("candidate") for row in preflight] == list(CANDIDATES),
            "preflight identities",
        )
        for row in preflight:
            for key in ("finite_prediction", "finite_loss", "finite_gradients"):
                require(row.get(key) is True, f"preflight {row.get('candidate')} {key}")
            parameter_count = row.get("parameter_count")
            require(
                parameter_count
                == EXPECTED_PARAMETER_COUNTS.get(row.get("candidate")),
                f"preflight {row.get('candidate')} parameters",
            )

    runs = selection.get("runs", [])
    expected_pairs = [(seed, candidate) for seed in SEEDS for candidate in CANDIDATES]
    require(isinstance(runs, list) and len(runs) == 4, "run rows")
    if isinstance(runs, list):
        require(
            [(row.get("seed"), row.get("candidate")) for row in runs]
            == expected_pairs,
            "run order and identities",
        )

    accepted_rows = []
    row_index_hashes = set()
    target_hashes = set()
    for seed, candidate in expected_pairs:
        metrics_path = root / "results" / f"seed{seed}" / candidate / "metrics.json"
        require(metrics_path.is_file(), f"missing metrics seed{seed}/{candidate}")
        if not metrics_path.is_file():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        matching = [
            row
            for row in runs
            if row.get("seed") == seed and row.get("candidate") == candidate
        ]
        require(len(matching) == 1 and matching[0] == metrics, f"selection mismatch {seed}/{candidate}")
        require(
            metrics.get("format")
            == "molgap-pcqm-gap100k-geometry-paired-run-v1",
            f"metrics format {seed}/{candidate}",
        )
        require(metrics.get("complete") is True, f"complete {seed}/{candidate}")
        require(metrics.get("source_commit") == expected_source_commit, f"source {seed}/{candidate}")
        require(
            metrics.get("contract") == expected_contract(candidate),
            f"contract {seed}/{candidate}",
        )
        require(metrics.get("validation_rows") == 10_000, f"validation rows {seed}/{candidate}")
        require(metrics.get("official_validation_role_read") is False, f"official valid {seed}/{candidate}")
        require(metrics.get("test_dev_role_read") is False, f"test-dev {seed}/{candidate}")
        parameter_count = metrics.get("parameter_count")
        require(
            parameter_count == EXPECTED_PARAMETER_COUNTS[candidate],
            f"parameters {seed}/{candidate}",
        )
        value = metrics.get("validation_gap_mae_eV")
        require(
            isinstance(value, (int, float)) and math.isfinite(value),
            f"MAE {seed}/{candidate}",
        )
        row_hash = metrics.get("validation_row_index_sha256")
        target_hash = metrics.get("validation_target_sha256")
        require(isinstance(row_hash, str) and len(row_hash) == 64, f"row hash {seed}/{candidate}")
        require(isinstance(target_hash, str) and len(target_hash) == 64, f"target hash {seed}/{candidate}")
        row_index_hashes.add(row_hash)
        target_hashes.add(target_hash)
        artifacts = metrics.get("artifacts", {})
        for name in ("best_model", "checkpoint", "validation_payload", "trace"):
            relative = artifacts.get(name)
            path = root / relative if isinstance(relative, str) else root / "__missing__"
            require(path.is_file(), f"missing {name} {seed}/{candidate}")
            if path.is_file():
                require(
                    sha256_file(path) == artifacts.get(f"{name}_sha256"),
                    f"hash {name} {seed}/{candidate}",
                )
        accepted_rows.append(metrics)

    require(len(row_index_hashes) == 1, "validation row identity differs")
    require(len(target_hashes) == 1, "validation targets differ")

    by_key = {(row["seed"], row["candidate"]): row for row in accepted_rows}
    pairs = [
        {
            "seed": 42,
            "comparator_validation_gap_mae_eV": SEED42_COMPARATOR_MAE,
            "candidate_validation_gap_mae_eV": SEED42_CANDIDATE_MAE,
            "candidate_minus_comparator_eV": (
                SEED42_CANDIDATE_MAE - SEED42_COMPARATOR_MAE
            ),
        }
    ]
    if len(by_key) == 4:
        for seed in SEEDS:
            comparator = by_key[(seed, COMPARATOR)]["validation_gap_mae_eV"]
            candidate = by_key[(seed, CANDIDATE)]["validation_gap_mae_eV"]
            pairs.append(
                {
                    "seed": seed,
                    "comparator_validation_gap_mae_eV": comparator,
                    "candidate_validation_gap_mae_eV": candidate,
                    "candidate_minus_comparator_eV": candidate - comparator,
                }
            )
    mean_comparator = sum(
        row["comparator_validation_gap_mae_eV"] for row in pairs
    ) / len(pairs)
    mean_candidate = sum(
        row["candidate_validation_gap_mae_eV"] for row in pairs
    ) / len(pairs)
    mean_row = {
        "seed": "mean",
        "comparator_validation_gap_mae_eV": mean_comparator,
        "candidate_validation_gap_mae_eV": mean_candidate,
        "candidate_minus_comparator_eV": mean_candidate - mean_comparator,
    }
    expected_comparison = [*pairs, mean_row]
    multiseed_gate_passed = (
        len(pairs) == 3
        and all(row["candidate_minus_comparator_eV"] < 0 for row in pairs)
        and mean_candidate < mean_comparator
    )
    expected_seed42_reference = {
        "seed": 42,
        "candidate": COMPARATOR,
        "comparator_validation_gap_mae_eV": SEED42_COMPARATOR_MAE,
        "geometry_candidate": CANDIDATE,
        "candidate_validation_gap_mae_eV": SEED42_CANDIDATE_MAE,
        "candidate_minus_comparator_eV": (
            SEED42_CANDIDATE_MAE - SEED42_COMPARATOR_MAE
        ),
    }
    require(
        selection.get("seed42_reference") == expected_seed42_reference,
        "seed42 reference",
    )
    require(
        selection.get("paired_comparison") == expected_comparison,
        "paired comparison arithmetic",
    )
    require(
        selection.get("multiseed_gate_passed") is multiseed_gate_passed,
        "multiseed gate arithmetic",
    )
    require(
        selection.get("selected_candidate")
        == (CANDIDATE if multiseed_gate_passed else COMPARATOR),
        "selected candidate",
    )

    result = {
        "format": "molgap-pcqm-gap100k-geometry-multiseed-acceptance-v1",
        "accepted": not errors,
        "errors": errors,
        "source_commit": expected_source_commit,
        "geometry_cache_aggregate_sha256": expected_geometry_sha256,
        "paired_comparison": expected_comparison,
        "multiseed_gate_passed": multiseed_gate_passed,
        "selected_candidate": CANDIDATE if multiseed_gate_passed else COMPARATOR,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    if errors:
        raise RuntimeError(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-geometry-sha256", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = accept(
        args.root,
        args.expected_source_commit,
        args.expected_geometry_sha256,
    )
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
