"""No-training audit of local specialist behavior in frozen PCQM predictors."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import average_precision_score, roc_auc_score

from .pcqm_k1_explainability import (
    DEVELOPMENT_ROWS,
    EXPECTED_GEOMETRY_SHA256,
    EXPECTED_MANIFEST_SHA256,
    _load_development_descriptors,
    sha256_file,
)


REFERENCE = "neural_atom_k1_v4"
FOLDS = 5
TOP_STRUCTURE_EXPERTS = 5
ORACLE_PAIR_GAIN_GATE_EV = 0.005
ORACLE_MULTI_GAIN_GATE_EV = 0.010
WIN_RATE_GATE = 0.05
WIN_MARGIN_GATE_EV = 0.010
AP_LIFT_GATE = 0.05
CLUSTER_ROWS_GATE = 500
CLUSTER_UPLIFT_GATE = 0.10


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fields = list(rows[0]) if rows else []
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _atomic_torch(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _prediction_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype=np.float32).tobytes()).hexdigest()


def _model_id(path: Path) -> str:
    if path.name == "development_predictions.pt":
        return path.parent.parent.name
    return path.parent.name if path.name == "best_development_payload.pt" else path.stem


def _pool_class(model_id: str, paths: list[str]) -> str:
    text = " ".join([model_id, *paths]).lower()
    if "gptrans" in text or model_id in {
        "pair_prenorm",
        "centered_logits",
        "memory_message",
        "memory_value",
        "training",
    }:
        return "contextual_gptrans_contract"
    if "pair_token_mose" in text:
        return "contextual_feature_identity_mismatch"
    return "primary_k1_v4_lineage"


def _discover_predictions(
    record_root: Path,
    packaged_payload_root: Path,
    source_idx: torch.Tensor,
    target: torch.Tensor,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    candidates = sorted(
        set(record_root.rglob("best_development_payload.pt"))
        | set(record_root.rglob("development_predictions.pt"))
        | set(packaged_payload_root.glob("*.pt"))
    )
    aligned: dict[str, dict[str, Any]] = {}
    rejected: dict[str, int] = {}
    for path in candidates:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or not {
                "source_idx",
                "target_eV",
                "prediction_eV",
            }.issubset(payload):
                rejected["missing_prediction_fields"] = rejected.get(
                    "missing_prediction_fields", 0
                ) + 1
                continue
            payload_source = payload["source_idx"].view(-1).long()
            payload_target = payload["target_eV"].view(-1).float()
            prediction = payload["prediction_eV"].view(-1).float()
            if not torch.equal(payload_source, source_idx):
                rejected["source_idx_mismatch"] = rejected.get(
                    "source_idx_mismatch", 0
                ) + 1
                continue
            if not torch.equal(payload_target, target):
                rejected["target_mismatch"] = rejected.get("target_mismatch", 0) + 1
                continue
            if not torch.isfinite(prediction).all():
                rejected["nonfinite_prediction"] = rejected.get(
                    "nonfinite_prediction", 0
                ) + 1
                continue
            prediction_hash = _prediction_sha256(prediction.numpy())
            record = aligned.setdefault(
                prediction_hash,
                {
                    "model_id": _model_id(path),
                    "prediction": prediction.numpy().astype(np.float64),
                    "paths": [],
                    "payload_sha256": sha256_file(path),
                },
            )
            record["paths"].append(path.as_posix())
            # Prefer the original result over the convenience explainability copy.
            if packaged_payload_root not in path.parents:
                record["model_id"] = _model_id(path)
                record["payload_sha256"] = sha256_file(path)
        except Exception:
            rejected["unreadable_or_incompatible"] = rejected.get(
                "unreadable_or_incompatible", 0
            ) + 1

    predictions: dict[str, np.ndarray] = {}
    inventory: list[dict[str, Any]] = []
    used_names: dict[str, int] = {}
    for prediction_hash, record in sorted(
        aligned.items(), key=lambda item: (item[1]["model_id"], item[0])
    ):
        base_name = record["model_id"]
        suffix = used_names.get(base_name, 0)
        used_names[base_name] = suffix + 1
        model_id = base_name if suffix == 0 else f"{base_name}__{suffix + 1}"
        predictions[model_id] = record["prediction"]
        pool_class = _pool_class(model_id, record["paths"])
        inventory.append(
            {
                "model_id": model_id,
                "pool_class": pool_class,
                "prediction_sha256": prediction_hash,
                "payload_sha256": record["payload_sha256"],
                "canonical_path": sorted(
                    record["paths"],
                    key=lambda value: (
                        "k1_explainability_payloads_v1" in value,
                        len(value),
                        value,
                    ),
                )[0],
                "alias_paths": sorted(record["paths"]),
                "aligned_rows": int(len(record["prediction"])),
                "mae_eV": float(
                    np.abs(record["prediction"] - target.numpy()).mean()
                ),
            }
        )
    if REFERENCE not in predictions:
        raise RuntimeError("Immutable K1-v4 reference prediction was not discovered")
    return predictions, inventory, {
        "candidate_files_scanned": len(candidates),
        "unique_aligned_predictions": len(predictions),
        "rejected": rejected,
    }


def _load_source_rows(source_csv: Path, source_idx: np.ndarray, target: np.ndarray) -> pd.DataFrame:
    stop = int(source_idx.max()) + 1
    frame = pd.read_csv(source_csv, nrows=stop)
    rows = frame.iloc[source_idx].reset_index(drop=True)
    if not np.array_equal(rows["idx"].to_numpy(np.int64), source_idx):
        raise RuntimeError("Official source CSV index does not align with prediction rows")
    gap = rows["homolumogap"].to_numpy(np.float64)
    if not np.allclose(gap, target, rtol=0.0, atol=5e-7):
        raise RuntimeError(
            "Official source targets do not align with payload targets: "
            f"max_abs_delta={np.max(np.abs(gap - target))}"
        )
    return rows


def _chemical_features(smiles: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    descriptor_rows: list[dict[str, Any]] = []
    fingerprints = np.zeros((len(smiles), 1024), dtype=np.uint8)
    valid = np.zeros(len(smiles), dtype=bool)
    for index, value in enumerate(smiles.astype(str)):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            # The accepted graph cache can contain a source SMILES that RDKit's
            # current sanitiser rejects (for example, unusual hypervalence).
            # Keep the row in Oracle analysis, but do not fabricate chemistry
            # descriptors or a fingerprint for it.
            descriptor_rows.append(
                {
                    "molecular_weight": np.nan,
                    "atom_count_rdkit": np.nan,
                    "heavy_atom_count": np.nan,
                    "aromatic_atom_fraction_rdkit": np.nan,
                    "ring_count": np.nan,
                    "aromatic_ring_count": np.nan,
                    "heteroatom_count": np.nan,
                    "nitrogen_count": np.nan,
                    "oxygen_count": np.nan,
                    "sulfur_count": np.nan,
                    "halogen_count": np.nan,
                    "rotatable_bond_count": np.nan,
                    "formal_charge_abs": np.nan,
                    "fraction_csp3": np.nan,
                    "conjugated_bond_fraction_rdkit": np.nan,
                    "rdkit_valid": 0.0,
                    "scaffold": "",
                }
            )
            continue
        valid[index] = True
        atoms = list(molecule.GetAtoms())
        bonds = list(molecule.GetBonds())
        atom_count = max(len(atoms), 1)
        atomic_numbers = np.asarray([atom.GetAtomicNum() for atom in atoms])
        descriptor_rows.append(
            {
                "molecular_weight": float(Descriptors.MolWt(molecule)),
                "atom_count_rdkit": float(len(atoms)),
                "heavy_atom_count": float(molecule.GetNumHeavyAtoms()),
                "aromatic_atom_fraction_rdkit": float(
                    sum(atom.GetIsAromatic() for atom in atoms) / atom_count
                ),
                "ring_count": float(rdMolDescriptors.CalcNumRings(molecule)),
                "aromatic_ring_count": float(
                    rdMolDescriptors.CalcNumAromaticRings(molecule)
                ),
                "heteroatom_count": float(Lipinski.NumHeteroatoms(molecule)),
                "nitrogen_count": float(np.count_nonzero(atomic_numbers == 7)),
                "oxygen_count": float(np.count_nonzero(atomic_numbers == 8)),
                "sulfur_count": float(np.count_nonzero(atomic_numbers == 16)),
                "halogen_count": float(
                    np.count_nonzero(np.isin(atomic_numbers, [9, 17, 35, 53]))
                ),
                "rotatable_bond_count": float(Lipinski.NumRotatableBonds(molecule)),
                "formal_charge_abs": float(
                    sum(abs(atom.GetFormalCharge()) for atom in atoms)
                ),
                "fraction_csp3": float(rdMolDescriptors.CalcFractionCSP3(molecule)),
                "conjugated_bond_fraction_rdkit": float(
                    sum(bond.GetIsConjugated() for bond in bonds) / max(len(bonds), 1)
                ),
                "rdkit_valid": 1.0,
                "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=molecule),
            }
        )
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), fingerprints[index])
    frame = pd.DataFrame(descriptor_rows)
    clusters = np.full(len(smiles), -1, dtype=np.int64)
    clusters[valid] = MiniBatchKMeans(
        n_clusters=64,
        random_state=42,
        batch_size=2048,
        n_init=3,
        max_iter=100,
    ).fit_predict(fingerprints[valid].astype(np.float32))
    return frame, clusters


def _fold_ids(source_idx: np.ndarray) -> np.ndarray:
    # Stable integer mixing; this is diagnostic partitioning, not a new role.
    mixed = (source_idx.astype(np.uint64) * np.uint64(11400714819323198485))
    return (mixed % np.uint64(FOLDS)).astype(np.int64)


def _pairwise_rows(
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    inventory_by_name: dict[str, dict[str, Any]],
    folds: np.ndarray,
) -> list[dict[str, Any]]:
    base_error = np.abs(predictions[REFERENCE] - target)
    rows: list[dict[str, Any]] = []
    for name, prediction in sorted(predictions.items()):
        if name == REFERENCE:
            continue
        expert_error = np.abs(prediction - target)
        equal_average_error = np.abs(
            0.5 * (prediction + predictions[REFERENCE]) - target
        )
        gain = base_error - expert_error
        wins = gain > 0
        losses = gain < 0
        oracle_error = np.minimum(base_error, expert_error)
        fold_gains = [
            float(base_error[folds == fold].mean() - oracle_error[folds == fold].mean())
            for fold in range(FOLDS)
        ]
        row = {
            "model_id": name,
            "pool_class": inventory_by_name[name]["pool_class"],
            "base_mae_eV": float(base_error.mean()),
            "expert_mae_eV": float(expert_error.mean()),
            "expert_delta_vs_base_eV": float(expert_error.mean() - base_error.mean()),
            "equal_average_mae_eV": float(equal_average_error.mean()),
            "equal_average_gain_vs_base_eV": float(
                base_error.mean() - equal_average_error.mean()
            ),
            "oracle_mae_eV": float(oracle_error.mean()),
            "oracle_gain_eV": float(base_error.mean() - oracle_error.mean()),
            "expert_win_rate": float(wins.mean()),
            "mean_winning_margin_eV": float(gain[wins].mean()) if wins.any() else 0.0,
            "median_winning_margin_eV": float(np.median(gain[wins])) if wins.any() else 0.0,
            "mean_losing_margin_eV": float((-gain[losses]).mean()) if losses.any() else 0.0,
            "gain_p05_eV": float(np.quantile(gain, 0.05)),
            "gain_p50_eV": float(np.quantile(gain, 0.50)),
            "gain_p95_eV": float(np.quantile(gain, 0.95)),
            "residual_correlation_with_base": float(
                np.corrcoef(prediction - target, predictions[REFERENCE] - target)[0, 1]
            ),
            "fold_oracle_gain_min_eV": float(min(fold_gains)),
            "fold_oracle_gain_max_eV": float(max(fold_gains)),
            "fold_oracle_gains_eV": fold_gains,
        }
        row["pair_oracle_gate_pass"] = bool(
            row["oracle_gain_eV"] >= ORACLE_PAIR_GAIN_GATE_EV
            and row["expert_win_rate"] >= WIN_RATE_GATE
            and row["mean_winning_margin_eV"] >= WIN_MARGIN_GATE_EV
            and row["fold_oracle_gain_min_eV"] > 0
        )
        rows.append(row)
    return rows


def _greedy_oracle(
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    names: list[str],
) -> list[dict[str, Any]]:
    base_error = np.abs(predictions[REFERENCE] - target)
    current = base_error.copy()
    remaining = set(names) - {REFERENCE}
    curve = [
        {
            "expert_count": 1,
            "added_model": REFERENCE,
            "oracle_mae_eV": float(current.mean()),
            "gain_vs_base_eV": 0.0,
            "incremental_gain_eV": 0.0,
        }
    ]
    while remaining:
        best_name = min(
            remaining,
            key=lambda name: np.minimum(
                current, np.abs(predictions[name] - target)
            ).mean(),
        )
        updated = np.minimum(current, np.abs(predictions[best_name] - target))
        curve.append(
            {
                "expert_count": len(curve) + 1,
                "added_model": best_name,
                "oracle_mae_eV": float(updated.mean()),
                "gain_vs_base_eV": float(base_error.mean() - updated.mean()),
                "incremental_gain_eV": float(current.mean() - updated.mean()),
            }
        )
        current = updated
        remaining.remove(best_name)
    return curve


def _safe_auc(labels: np.ndarray, score: np.ndarray) -> dict[str, float | None]:
    prevalence = float(labels.mean())
    if labels.min() == labels.max() or np.all(score == score[0]):
        return {
            "prevalence": prevalence,
            "average_precision": None,
            "average_precision_lift": None,
            "roc_auc": None,
            "orientation": None,
        }
    positive_ap = float(average_precision_score(labels, score))
    negative_ap = float(average_precision_score(labels, -score))
    if positive_ap >= negative_ap:
        oriented = score
        ap = positive_ap
        orientation = "high"
    else:
        oriented = -score
        ap = negative_ap
        orientation = "low"
    return {
        "prevalence": prevalence,
        "average_precision": ap,
        "average_precision_lift": float(ap - prevalence),
        "roc_auc": float(roc_auc_score(labels, oriented)),
        "orientation": orientation,
    }


def _descriptor_signal(labels: np.ndarray, values: np.ndarray) -> dict[str, Any]:
    finite = np.isfinite(values)
    labels = labels[finite]
    values = values[finite]
    if len(values) == 0:
        return {
            "valid_rows": 0,
            "prevalence": None,
            "average_precision": None,
            "average_precision_lift": None,
            "roc_auc": None,
            "orientation": None,
            "winner_mean": None,
            "loser_mean": None,
            "standardized_mean_difference": None,
            "quintile_win_rates": [],
            "quintile_spread": None,
        }
    auc = _safe_auc(labels, values)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, 6)))
    if len(edges) < 2:
        rates = [float(labels.mean())]
    else:
        groups = np.digitize(values, edges[1:-1], right=True)
        rates = [float(labels[groups == group].mean()) for group in range(len(edges) - 1)]
    winner = values[labels]
    loser = values[~labels]
    pooled = float(values.std(ddof=1))
    return {
        "valid_rows": int(len(values)),
        **auc,
        "winner_mean": float(winner.mean()) if len(winner) else None,
        "loser_mean": float(loser.mean()) if len(loser) else None,
        "standardized_mean_difference": (
            float((winner.mean() - loser.mean()) / pooled)
            if len(winner) and len(loser) and pooled > 0
            else None
        ),
        "quintile_win_rates": rates,
        "quintile_spread": float(max(rates) - min(rates)),
    }


def _cluster_signal(
    labels: np.ndarray,
    clusters: np.ndarray,
    folds: np.ndarray,
) -> dict[str, Any]:
    prevalence = float(labels.mean())
    records = []
    for cluster in sorted(np.unique(clusters[clusters >= 0])):
        mask = clusters == cluster
        if mask.sum() < 100:
            continue
        fold_uplifts = []
        for fold in range(FOLDS):
            fold_mask = folds == fold
            local = mask & fold_mask
            fold_uplifts.append(
                float(labels[local].mean() - labels[fold_mask].mean())
                if local.any()
                else None
            )
        records.append(
            {
                "cluster": int(cluster),
                "rows": int(mask.sum()),
                "win_rate": float(labels[mask].mean()),
                "uplift": float(labels[mask].mean() - prevalence),
                "fold_uplifts": fold_uplifts,
                "stable_enrichment": bool(
                    mask.sum() >= CLUSTER_ROWS_GATE
                    and labels[mask].mean() - prevalence >= CLUSTER_UPLIFT_GATE
                    and all(value is not None and value > 0 for value in fold_uplifts)
                ),
            }
        )
    records.sort(key=lambda item: item["uplift"], reverse=True)
    return {
        "cluster_count_reported": len(records),
        "stable_enriched_cluster_count": sum(
            item["stable_enrichment"] for item in records
        ),
        "top_enriched": records[:10],
        "bottom_enriched": records[-10:],
    }


def _scaffold_signal(labels: np.ndarray, scaffolds: pd.Series) -> dict[str, Any]:
    frame = pd.DataFrame({"scaffold": scaffolds.fillna(""), "winner": labels})
    grouped = (
        frame.groupby("scaffold", dropna=False)["winner"]
        .agg(["size", "mean"])
        .reset_index()
    )
    grouped = grouped[(grouped["scaffold"] != "") & (grouped["size"] >= 20)]
    grouped = grouped.sort_values("mean", ascending=False)
    return {
        "unique_scaffolds": int(frame["scaffold"].nunique()),
        "recurrent_scaffolds_ge20": int(len(grouped)),
        "top_enriched": [
            {
                "scaffold": row.scaffold,
                "rows": int(row.size),
                "win_rate": float(row.mean),
            }
            for row in grouped.head(10).itertuples(index=False)
        ],
        "bottom_enriched": [
            {
                "scaffold": row.scaffold,
                "rows": int(row.size),
                "win_rate": float(row.mean),
            }
            for row in grouped.tail(10).itertuples(index=False)
        ],
    }


def _structure_report(
    target: np.ndarray,
    predictions: dict[str, np.ndarray],
    pairwise: list[dict[str, Any]],
    descriptors: pd.DataFrame,
    graph_descriptors: dict[str, np.ndarray],
    clusters: np.ndarray,
    folds: np.ndarray,
    primary_names: list[str],
) -> dict[str, Any]:
    base_error = np.abs(predictions[REFERENCE] - target)
    top = [
        row["model_id"]
        for row in sorted(
            (row for row in pairwise if row["model_id"] in primary_names),
            key=lambda row: row["oracle_gain_eV"],
            reverse=True,
        )[:TOP_STRUCTURE_EXPERTS]
    ]
    scalar_features = {
        column: descriptors[column].to_numpy(np.float64)
        for column in descriptors.columns
        if column != "scaffold"
    }
    scalar_features.update(
        {
            name: values
            for name, values in graph_descriptors.items()
            if name != "target_gap_eV"
        }
    )
    strict_stack = np.stack([predictions[name] for name in primary_names], axis=1)
    ensemble_std = strict_stack.std(axis=1)
    experts: dict[str, Any] = {}
    for name in top:
        prediction = predictions[name]
        expert_error = np.abs(prediction - target)
        labels = expert_error < base_error
        disagreement = np.abs(prediction - predictions[REFERENCE])
        signals = {
            feature: _descriptor_signal(labels, values)
            for feature, values in scalar_features.items()
        }
        signals["abs_prediction_disagreement"] = _descriptor_signal(
            labels, disagreement
        )
        signals["all_primary_prediction_std"] = _descriptor_signal(
            labels, ensemble_std
        )
        best_signal_name, best_signal = max(
            signals.items(),
            key=lambda item: (
                item[1]["average_precision_lift"]
                if item[1]["average_precision_lift"] is not None
                else -np.inf
            ),
        )
        cluster = _cluster_signal(labels, clusters, folds)
        structure_gate = bool(
            best_signal["average_precision_lift"] is not None
            and best_signal["average_precision_lift"] >= AP_LIFT_GATE
            and cluster["stable_enriched_cluster_count"] > 0
        )
        experts[name] = {
            "win_prevalence": float(labels.mean()),
            "best_scalar_signal": best_signal_name,
            "best_scalar_signal_metrics": best_signal,
            "disagreement_signal": signals["abs_prediction_disagreement"],
            "descriptor_signals": signals,
            "morgan_clusters": cluster,
            "scaffolds": _scaffold_signal(labels, descriptors["scaffold"]),
            "structure_gate_pass": structure_gate,
        }
    return {
        "top_experts_by_pairwise_oracle_gain": top,
        "experts": experts,
        "embedding_features_available": False,
        "seed_variance_available": False,
        "train_reference_ood_distance_available": False,
        "structure_gate_any_pass": any(
            item["structure_gate_pass"] for item in experts.values()
        ),
    }


def run_specialist_audit(
    *,
    cache_root: Path,
    source_csv: Path,
    record_root: Path,
    packaged_payload_root: Path,
    output_root: Path,
    matrix_output: Path,
) -> dict[str, Any]:
    source_idx, target, graph_descriptors, cache_manifest = _load_development_descriptors(
        cache_root
    )
    source_tensor = torch.from_numpy(source_idx)
    target_tensor = torch.from_numpy(target.astype(np.float32))
    predictions, inventory, discovery = _discover_predictions(
        record_root, packaged_payload_root, source_tensor, target_tensor
    )
    inventory_by_name = {item["model_id"]: item for item in inventory}
    source_rows = _load_source_rows(source_csv, source_idx, target)
    chemical_descriptors, clusters = _chemical_features(source_rows["smiles"])
    folds = _fold_ids(source_idx)

    primary_names = [
        item["model_id"]
        for item in inventory
        if item["pool_class"] == "primary_k1_v4_lineage"
    ]
    expanded_names = list(predictions)
    if REFERENCE not in primary_names:
        raise RuntimeError("K1-v4 reference is not in the primary pool")

    pairwise = _pairwise_rows(target, predictions, inventory_by_name, folds)
    primary_curve = _greedy_oracle(target, predictions, primary_names)
    expanded_curve = _greedy_oracle(target, predictions, expanded_names)
    structure = _structure_report(
        target,
        predictions,
        pairwise,
        chemical_descriptors,
        graph_descriptors,
        clusters,
        folds,
        primary_names,
    )
    primary_pair_passes = [
        row
        for row in pairwise
        if row["model_id"] in primary_names and row["pair_oracle_gate_pass"]
    ]
    primary_multi_gain = primary_curve[-1]["gain_vs_base_eV"]
    asset_gate = bool(
        len(primary_names) >= 4
        and len(source_idx) == DEVELOPMENT_ROWS
        and cache_manifest["geometry_aggregate_sha256"] == EXPECTED_GEOMETRY_SHA256
    )
    oracle_gate = bool(
        primary_pair_passes
        and primary_multi_gain >= ORACLE_MULTI_GAIN_GATE_EV
    )
    if not asset_gate:
        verdict = "INCONCLUSIVE_ASSET_FAILURE"
    elif not oracle_gate:
        verdict = "NO_GO_INSUFFICIENT_ORACLE"
    elif not structure["structure_gate_any_pass"]:
        verdict = "NO_GO_STRUCTURE_NOT_LEARNABLE"
    else:
        verdict = "GO_TO_ROUTER_BASELINE"

    matrix_payload = {
        "format": "molgap-pcqm-specialist-prediction-matrix-v1",
        "source_idx": torch.from_numpy(source_idx),
        "smiles": source_rows["smiles"].tolist(),
        "target_eV": torch.from_numpy(target.astype(np.float32)),
        "model_ids": list(predictions),
        "predictions_eV": torch.from_numpy(
            np.stack([predictions[name] for name in predictions], axis=1).astype(
                np.float32
            )
        ),
        "morgan_cluster_64": torch.from_numpy(clusters),
        "fold_id": torch.from_numpy(folds),
    }
    _atomic_torch(matrix_output, matrix_payload)

    asset_report = {
        "format": "molgap-pcqm-specialist-asset-audit-v1",
        "complete": True,
        "cache_manifest_sha256": sha256_file(cache_root / "manifest.json"),
        "expected_cache_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "geometry_aggregate_sha256": cache_manifest["geometry_aggregate_sha256"],
        "source_csv_sha256": sha256_file(source_csv),
        "source_idx_sha256": _tensor_sha256(source_tensor),
        "target_sha256": _tensor_sha256(target_tensor),
        "rows": int(len(source_idx)),
        "source_idx_min": int(source_idx.min()),
        "source_idx_max": int(source_idx.max()),
        "model_count": len(predictions),
        "primary_pool_count": len(primary_names),
        "contextual_pool_count": len(predictions) - len(primary_names),
        "discovery": discovery,
        "models": inventory,
        "role_and_leakage": {
            "analysis_role": "official-train-derived-development",
            "role_reused_for_prior_model_selection": True,
            "unbiased_final_evaluation": False,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
            "test_challenge_role_read": False,
            "label_aware_oracle": True,
            "router_training_executed": False,
            "model_training_executed": False,
            "model_inference_executed": False,
        },
        "matrix_artifact": {
            "path": matrix_output.as_posix(),
            "sha256": sha256_file(matrix_output),
            "rows": int(len(source_idx)),
            "models": len(predictions),
        },
    }
    oracle_report = {
        "format": "molgap-pcqm-specialist-oracle-v1",
        "reference": REFERENCE,
        "base_mae_eV": float(
            np.abs(predictions[REFERENCE] - target).mean()
        ),
        "pairwise": pairwise,
        "primary_greedy_curve": primary_curve,
        "expanded_contextual_greedy_curve": expanded_curve,
        "gate_thresholds": {
            "pair_oracle_gain_eV": ORACLE_PAIR_GAIN_GATE_EV,
            "multi_oracle_gain_eV": ORACLE_MULTI_GAIN_GATE_EV,
            "win_rate": WIN_RATE_GATE,
            "mean_winning_margin_eV": WIN_MARGIN_GATE_EV,
            "folds": FOLDS,
        },
        "primary_pair_gate_pass_count": len(primary_pair_passes),
        "primary_multi_oracle_gain_eV": float(primary_multi_gain),
        "oracle_capacity_gate_pass": oracle_gate,
        "oracle_is_non_deployable": True,
    }
    reason_by_verdict = {
        "INCONCLUSIVE_ASSET_FAILURE": (
            "The aligned asset set did not satisfy the frozen minimum identity gate."
        ),
        "NO_GO_INSUFFICIENT_ORACLE": (
            "Existing experts did not provide enough label-Oracle headroom to justify "
            "Router development."
        ),
        "NO_GO_STRUCTURE_NOT_LEARNABLE": (
            "Oracle headroom is large, but no top specialist passed the frozen "
            "inference-visible scalar plus stable Morgan-cluster structure gate."
        ),
        "GO_TO_ROUTER_BASELINE": (
            "Oracle capacity and frozen structure-identifiability gates both passed."
        ),
    }
    decision = {
        "format": "molgap-pcqm-specialist-audit-decision-v1",
        "verdict": verdict,
        "asset_gate_pass": asset_gate,
        "oracle_capacity_gate_pass": oracle_gate,
        "structure_gate_pass": structure["structure_gate_any_pass"],
        "phase_d_authorized_by_this_run": False,
        "new_training_authorized": False,
        "reason": reason_by_verdict[verdict],
        "scope_note": (
            "Phase A-C only. A future Router baseline requires a separate frozen "
            "protocol even when all feasibility gates pass."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_root / "asset_audit.json", asset_report)
    _atomic_json(output_root / "oracle_report.json", oracle_report)
    _atomic_json(output_root / "structure_report.json", structure)
    _atomic_json(output_root / "decision.json", decision)
    _atomic_csv(output_root / "pairwise_oracle.csv", pairwise)
    _atomic_csv(output_root / "primary_greedy_curve.csv", primary_curve)
    result = {
        "format": "molgap-pcqm-specialist-audit-completion-v1",
        "complete": True,
        "decision": decision,
        "artifacts": {
            path.name: sha256_file(path)
            for path in sorted(output_root.iterdir())
            if path.is_file() and path.name != "completion_manifest.json"
        },
    }
    _atomic_json(output_root / "completion_manifest.json", result)
    return result
