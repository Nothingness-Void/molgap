"""Train and evaluate learned GPS7/GPS9/GPS11 prediction-level routing."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit

from molgap.multi2d_router_fusion import (
    DESCRIPTOR_COLUMNS,
    EXPERTS,
    TARGETS,
    GateTrainingConfig,
    fit_dense_soft_gate,
    fit_predispatch_router,
    metric_block,
    predict_dense_gate,
    predict_hard_route,
    route_cost,
)
from molgap.multi2d import targetwise_oracle
from molgap.router import paired_bootstrap_mean


COMMON_COLUMNS = {
    "gps7": "repaired_2m_d_gps7_seed42",
    "gps9": "repaired_2m_d_gps9_seed42",
    "gps11_160": "repaired_2m_d_gps11_160_seed42",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def atomic_csv(payload: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    payload.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_torch(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_holdout(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "molgap-three-gps-holdout-v1":
        raise ValueError(f"Unsupported holdout payload: {payload.get('format')}")
    if tuple(payload.get("expert_names", [])) != EXPERTS:
        raise ValueError("Holdout expert order does not match the Router contract")
    if tuple(payload.get("target_names", [])) != TARGETS:
        raise ValueError("Holdout target order does not match the Router contract")
    result = {}
    for split in ("validation", "test"):
        block = payload[split]
        source_idx = block["source_idx"].numpy().astype(np.int64)
        targets = block["targets"].numpy().astype(np.float32)
        predictions = block["predictions"].numpy().astype(np.float32)
        if (
            len(source_idx) != len(np.unique(source_idx))
            or predictions.shape != (len(source_idx), len(EXPERTS), len(TARGETS))
            or targets.shape != (len(source_idx), len(TARGETS))
            or not np.isfinite(predictions).all()
            or not np.isfinite(targets).all()
        ):
            raise ValueError(f"Invalid or non-finite {split} payload")
        result[split] = {
            "source_idx": source_idx,
            "targets": targets,
            "predictions": predictions,
        }
    if np.intersect1d(
        result["validation"]["source_idx"],
        result["test"]["source_idx"],
    ).size:
        raise ValueError("Holdout validation/test source_idx overlap")
    return result


def aligned_manifest_rows(
    manifest: pd.DataFrame,
    source_idx: np.ndarray,
) -> pd.DataFrame:
    if "manifest_row" not in manifest or manifest.manifest_row.duplicated().any():
        raise ValueError("Manifest must contain unique manifest_row")
    indexed = manifest.set_index("manifest_row", drop=False)
    missing = np.setdiff1d(source_idx, indexed.index.to_numpy())
    if len(missing):
        raise ValueError(f"Manifest misses {len(missing):,} source_idx rows")
    rows = indexed.loc[source_idx].reset_index(drop=True)
    if not np.array_equal(rows.manifest_row.to_numpy(np.int64), source_idx):
        raise ValueError("Manifest row alignment failed")
    required = {"scaffold", *DESCRIPTOR_COLUMNS}
    if missing_columns := required.difference(rows.columns):
        raise ValueError(f"Manifest misses columns: {sorted(missing_columns)}")
    return rows


def descriptor_array(frame: pd.DataFrame) -> np.ndarray:
    values = frame.loc[:, DESCRIPTOR_COLUMNS].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Descriptor table contains non-finite values")
    return values


def smiles_descriptors(smiles: pd.Series) -> np.ndarray:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski

    rows = []
    for value in smiles.astype(str):
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid external SMILES: {value!r}")
        rows.append(
            (
                Descriptors.MolWt(molecule),
                molecule.GetNumHeavyAtoms(),
                Lipinski.NumAromaticRings(molecule),
                Lipinski.NumRotatableBonds(molecule),
                Lipinski.NumHeteroatoms(molecule),
            )
        )
    return np.asarray(rows, dtype=np.float32)


def load_common_predictions(paths: dict[str, Path]) -> tuple[pd.DataFrame, np.ndarray]:
    frames = {name: pd.read_csv(path) for name, path in paths.items()}
    identity = ["eval_set", "cid", "smiles", *TARGETS]
    reference = frames["gps7"].loc[:, identity]
    predictions = []
    for name in EXPERTS:
        frame = frames[name]
        if not frame.loc[:, identity].equals(reference):
            raise ValueError(f"External common identities differ for {name}")
        prefix = COMMON_COLUMNS[name]
        predictions.append(
            frame.loc[:, [f"{prefix}_{target}" for target in TARGETS]]
            .to_numpy(np.float32)
        )
    stack = np.stack(predictions, axis=1)
    if not np.isfinite(stack).all():
        raise ValueError("External predictions contain non-finite values")
    return reference.copy(), stack


def prediction_delta(
    targets: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict:
    result = {}
    for index, target in enumerate((*TARGETS, "average")):
        if target == "average":
            delta = np.abs(candidate - targets).mean(axis=1) - np.abs(
                baseline - targets
            ).mean(axis=1)
        else:
            delta = np.abs(candidate[:, index] - targets[:, index]) - np.abs(
                baseline[:, index] - targets[:, index]
            )
        result[target] = paired_bootstrap_mean(
            delta,
            n_bootstrap=10_000,
            seed=42 + index,
        )
    return result


def evaluate_methods(
    targets: np.ndarray,
    predictions: np.ndarray,
    dense: np.ndarray,
    hard: np.ndarray,
) -> dict:
    methods = {
        expert: predictions[:, index]
        for index, expert in enumerate(EXPERTS)
    }
    methods["equal_gps7_gps9"] = predictions[:, :2].mean(axis=1)
    methods["equal_three"] = predictions.mean(axis=1)
    methods["dense_soft_gate"] = dense
    methods["predispatch_hard_route"] = hard
    oracle, _ = targetwise_oracle(
        targets,
        {expert: predictions[:, index] for index, expert in enumerate(EXPERTS)},
    )
    methods["targetwise_oracle"] = oracle
    return {
        "metrics": {
            name: metric_block(targets, value)
            for name, value in methods.items()
        },
        "delta_vs_gps9": {
            name: prediction_delta(targets, predictions[:, 1], value)
            for name, value in methods.items()
            if name not in {"gps9", "targetwise_oracle"}
        },
    }


def mean_seed_predictions(
    models: list,
    predictions: np.ndarray,
    descriptors: np.ndarray | None = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    outputs, aux = [], []
    for model in models:
        if descriptors is None:
            output, weight = predict_dense_gate(model, predictions)
            outputs.append(output)
            aux.append(weight)
        else:
            output, selected, _ = predict_hard_route(
                model, predictions, descriptors
            )
            outputs.append(output)
            aux.append(selected)
    return np.mean(outputs, axis=0), aux


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gps7-common", type=Path, required=True)
    parser.add_argument("--gps9-common", type=Path, required=True)
    parser.add_argument("--gps11-common", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    atomic_json({"status": "loading"}, args.out_dir / "progress.json")

    holdout = load_holdout(args.holdout)
    manifest = pd.read_parquet(args.manifest)
    validation_rows = aligned_manifest_rows(
        manifest, holdout["validation"]["source_idx"]
    )
    test_rows = aligned_manifest_rows(manifest, holdout["test"]["source_idx"])
    descriptors_validation = descriptor_array(validation_rows)
    descriptors_test = descriptor_array(test_rows)
    groups = validation_rows.scaffold.fillna("").astype(str).to_numpy()
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    router_train, router_validation = next(
        splitter.split(np.arange(len(groups)), groups=groups)
    )
    if set(groups[router_train]).intersection(groups[router_validation]):
        raise RuntimeError("Router train/validation scaffold overlap")
    atomic_json(
        {
            "status": "training",
            "router_train_rows": len(router_train),
            "router_validation_rows": len(router_validation),
            "router_train_scaffolds": len(set(groups[router_train])),
            "router_validation_scaffolds": len(set(groups[router_validation])),
        },
        args.out_dir / "progress.json",
    )

    dense_models, route_models, training = [], [], {}
    for seed in (42, 43, 44):
        config = GateTrainingConfig(seed=seed)
        dense, dense_report = fit_dense_soft_gate(
            holdout["validation"]["predictions"],
            holdout["validation"]["targets"],
            router_train,
            router_validation,
            config=config,
            device=args.device,
        )
        route, route_report = fit_predispatch_router(
            holdout["validation"]["predictions"],
            holdout["validation"]["targets"],
            descriptors_validation,
            router_train,
            router_validation,
            config=config,
            device=args.device,
        )
        dense_models.append(dense)
        route_models.append(route)
        training[str(seed)] = {
            "dense": dense_report,
            "predispatch": route_report,
        }
        atomic_torch(
            {
                "kind": "three_gps_dense_soft_gate",
                "seed": seed,
                "experts": EXPERTS,
                "targets": TARGETS,
                "config": dense_report["config"],
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in dense.state_dict().items()
                },
            },
            args.out_dir / f"dense_seed{seed}.pt",
        )
        atomic_torch(
            {
                "kind": "three_gps_predispatch_router",
                "seed": seed,
                "experts": EXPERTS,
                "targets": TARGETS,
                "descriptor_columns": DESCRIPTOR_COLUMNS,
                "config": route_report["config"],
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in route.state_dict().items()
                },
            },
            args.out_dir / f"predispatch_seed{seed}.pt",
        )

    test_dense, test_dense_weights = mean_seed_predictions(
        dense_models, holdout["test"]["predictions"]
    )
    test_hard_seed, test_selected = mean_seed_predictions(
        route_models,
        holdout["test"]["predictions"],
        descriptors_test,
    )
    # Majority-vote target routes preserve actual sparse encoder-call semantics.
    selected_stack = np.stack(test_selected)
    test_selected_majority = np.apply_along_axis(
        lambda value: np.bincount(value, minlength=len(EXPERTS)).argmax(),
        axis=0,
        arr=selected_stack,
    )
    expert_first = holdout["test"]["predictions"].transpose(0, 2, 1)
    test_hard = np.take_along_axis(
        expert_first,
        test_selected_majority[..., None],
        axis=2,
    )[..., 0]
    atomic_torch(
        {
            "format": "molgap-three-gps-frozen-2d-test-v1",
            "source_idx": torch.from_numpy(holdout["test"]["source_idx"]),
            "targets": torch.from_numpy(holdout["test"]["targets"]),
            "expert_names": EXPERTS,
            "target_names": TARGETS,
            "expert_predictions": torch.from_numpy(
                holdout["test"]["predictions"]
            ),
            "dense_prediction": torch.from_numpy(test_dense.astype(np.float32)),
            "equal_gps7_gps9_prediction": torch.from_numpy(
                holdout["test"]["predictions"][:, :2].mean(axis=1).astype(
                    np.float32
                )
            ),
            "dense_weights": torch.from_numpy(
                np.mean(np.stack(test_dense_weights), axis=0).astype(np.float32)
            ),
            "hard_prediction": torch.from_numpy(test_hard.astype(np.float32)),
            "hard_selection": torch.from_numpy(
                test_selected_majority.astype(np.int64)
            ),
            "split_seed": 42,
            "router_seeds": (42, 43, 44),
        },
        args.out_dir / "frozen_2d_test_payload.pt",
    )
    internal = evaluate_methods(
        holdout["test"]["targets"],
        holdout["test"]["predictions"],
        test_dense,
        test_hard,
    )
    internal["hard_route_cost"] = route_cost(test_selected_majority)
    internal["seed_mean_hard_prediction_metrics"] = metric_block(
        holdout["test"]["targets"],
        test_hard_seed,
    )

    common_frame, common_predictions = load_common_predictions(
        {
            "gps7": args.gps7_common,
            "gps9": args.gps9_common,
            "gps11_160": args.gps11_common,
        }
    )
    common_descriptors = smiles_descriptors(common_frame.smiles)
    common_dense, _ = mean_seed_predictions(dense_models, common_predictions)
    _, common_selected = mean_seed_predictions(
        route_models,
        common_predictions,
        common_descriptors,
    )
    common_selected = np.stack(common_selected)
    common_selected_majority = np.apply_along_axis(
        lambda value: np.bincount(value, minlength=len(EXPERTS)).argmax(),
        axis=0,
        arr=common_selected,
    )
    common_hard = np.take_along_axis(
        common_predictions.transpose(0, 2, 1),
        common_selected_majority[..., None],
        axis=2,
    )[..., 0]
    common_targets = common_frame.loc[:, TARGETS].to_numpy(np.float32)
    external = {}
    for scope in ("all", "ood1000", "p8_targeted_hard"):
        mask = (
            np.ones(len(common_frame), dtype=bool)
            if scope == "all"
            else common_frame.eval_set.eq(scope).to_numpy()
        )
        external[scope] = evaluate_methods(
            common_targets[mask],
            common_predictions[mask],
            common_dense[mask],
            common_hard[mask],
        )
        external[scope]["hard_route_cost"] = route_cost(
            common_selected_majority[mask]
        )

    output = common_frame.copy()
    for expert_index, expert in enumerate(EXPERTS):
        for target_index, target in enumerate(TARGETS):
            output[f"{expert}_{target}"] = common_predictions[
                :, expert_index, target_index
            ]
    for target_index, target in enumerate(TARGETS):
        output[f"dense_soft_gate_{target}"] = common_dense[:, target_index]
        output[f"predispatch_hard_route_{target}"] = common_hard[:, target_index]
        output[f"predispatch_selected_{target}"] = common_selected_majority[
            :, target_index
        ]
    atomic_csv(output, args.out_dir / "external_predictions.csv")

    metrics = {
        "experiment": "repaired_2m_three_gps_learned_router_fusion",
        "status": "pilot_complete_not_production",
        "experts": EXPERTS,
        "targets": TARGETS,
        "split": {
            "base_split_seed": 42,
            "base_validation_rows": len(validation_rows),
            "base_test_rows": len(test_rows),
            "router_train_rows": len(router_train),
            "router_validation_rows": len(router_validation),
            "router_train_validation_scaffold_overlap": 0,
        },
        "training": training,
        "internal_test": internal,
        "external": external,
        "inputs": {
            "holdout": str(args.holdout),
            "holdout_sha256": sha256(args.holdout),
            "manifest": str(args.manifest),
            "manifest_sha256": sha256(args.manifest),
            "common_predictions": {
                name: {
                    "path": str(path),
                    "sha256": sha256(path),
                }
                for name, path in {
                    "gps7": args.gps7_common,
                    "gps9": args.gps9_common,
                    "gps11_160": args.gps11_common,
                }.items()
            },
        },
        "artifacts": {
            "frozen_2d_test_payload": str(
                args.out_dir / "frozen_2d_test_payload.pt"
            )
        },
        "sealed_20k_used": False,
        "production_registry_changed": False,
        "pcqm_status": (
            "not_evaluated_existing_pcqm_csvs_only_store_gap; "
            "dense gate requires all three target predictions"
        ),
    }
    atomic_json(metrics, args.out_dir / "metrics.json")
    atomic_json(
        {
            "status": "complete",
            "metrics": str(args.out_dir / "metrics.json"),
            "external_predictions": str(
                args.out_dir / "external_predictions.csv"
            ),
        },
        args.out_dir / "progress.json",
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
