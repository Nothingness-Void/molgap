"""Frozen-embedding fusion and routing for the QM9 architecture screen."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
import torch.nn as nn

from .fusion import BoundedResidualFusionHead, FusionHead
from .qm9_screen import _metrics, set_seed


def validation_selected_prediction_blend(
    first_predictions_path: Path,
    second_predictions_path: Path,
    target_payload_path: Path,
    grid_points: int = 101,
) -> dict:
    """Select one global prediction-blend weight on validation only."""
    if grid_points < 2:
        raise ValueError("grid_points must be at least 2")
    first = torch.load(
        first_predictions_path, map_location="cpu", weights_only=False
    )
    second = torch.load(
        second_predictions_path, map_location="cpu", weights_only=False
    )
    targets = torch.load(
        target_payload_path, map_location="cpu", weights_only=False
    )
    aligned = {}
    for role in ("validation", "test"):
        source_idx = first[f"{role}_source_idx"]
        if not torch.equal(source_idx, second[f"{role}_source_idx"]):
            raise ValueError(f"Unaligned prediction-blend {role} rows")
        positions = {
            int(value): i
            for i, value in enumerate(targets[role]["source_idx"].tolist())
        }
        missing = [
            int(value)
            for value in source_idx.tolist()
            if int(value) not in positions
        ]
        if missing:
            raise ValueError(
                f"Missing {len(missing)} prediction-blend {role} targets"
            )
        selected = torch.tensor(
            [positions[int(value)] for value in source_idx.tolist()]
        )
        aligned[role] = {
            "first": first[f"{role}_predictions"],
            "second": second[f"{role}_predictions"],
            "target": targets[role]["targets"][selected],
            "source_idx": source_idx,
        }

    validation = aligned["validation"]
    candidates = []
    for index in range(grid_points):
        weight = index / (grid_points - 1)
        prediction = (
            weight * validation["first"]
            + (1.0 - weight) * validation["second"]
        )
        candidates.append({
            "first_weight": weight,
            "validation_average_mae_eV": _metrics(
                prediction.numpy(), validation["target"].numpy()
            )["average"]["mae"],
        })
    selected = min(
        candidates,
        key=lambda value: value["validation_average_mae_eV"],
    )
    weight = selected["first_weight"]
    test = aligned["test"]
    prediction = weight * test["first"] + (1.0 - weight) * test["second"]
    return {
        "blend": "validation_selected_global_prediction_weight",
        "selection_role": "validation",
        "grid_points": grid_points,
        "first_weight": weight,
        "second_weight": 1.0 - weight,
        "validation_average_mae_eV": selected[
            "validation_average_mae_eV"
        ],
        "aligned_rows": {
            role: len(value["source_idx"])
            for role, value in aligned.items()
        },
        "test_metrics": {
            "first": _metrics(
                test["first"].numpy(), test["target"].numpy()
            ),
            "second": _metrics(
                test["second"].numpy(), test["target"].numpy()
            ),
            "blend": _metrics(
                prediction.numpy(), test["target"].numpy()
            ),
        },
    }


def evaluate_gap_threshold_route(
    base_predictions_path: Path,
    dual_predictions_path: Path,
    target_payload_path: Path,
) -> dict:
    base = torch.load(
        base_predictions_path, map_location="cpu", weights_only=False
    )
    dual = torch.load(
        dual_predictions_path, map_location="cpu", weights_only=False
    )
    targets = torch.load(
        target_payload_path, map_location="cpu", weights_only=False
    )

    roles = {}
    for role, prefix in (("validation", "validation"), ("test", "test")):
        source_key = f"{prefix}_source_idx"
        prediction_key = f"{prefix}_predictions"
        source_idx = base[source_key]
        if not torch.equal(source_idx, dual[source_key]):
            raise ValueError(f"Unaligned routed {role} prediction rows")
        target_positions = {
            int(value): i
            for i, value in enumerate(targets[role]["source_idx"].tolist())
        }
        missing = [
            int(value)
            for value in source_idx.tolist()
            if int(value) not in target_positions
        ]
        if missing:
            raise ValueError(f"Missing {len(missing)} routed {role} target rows")
        selected = torch.tensor(
            [target_positions[int(value)] for value in source_idx.tolist()]
        )
        roles[role] = {
            "base": base[prediction_key],
            "dual": dual[prediction_key],
            "target": targets[role]["targets"][selected],
        }

    validation = roles["validation"]
    gap = validation["base"][:, 2]
    thresholds = torch.quantile(gap, torch.linspace(0.0, 1.0, 101)).unique()
    candidates = []
    for mode in ("below", "above"):
        for threshold in thresholds:
            mask = gap <= threshold if mode == "below" else gap >= threshold
            prediction = torch.where(
                mask[:, None], validation["dual"], validation["base"]
            )
            value = _metrics(
                prediction.numpy(), validation["target"].numpy()
            )["average"]["mae"]
            candidates.append({
                "mode": mode,
                "threshold_eV": float(threshold),
                "validation_average_mae_eV": value,
                "validation_route_fraction": float(mask.float().mean()),
            })
    candidates.extend([
        {
            "mode": "never",
            "threshold_eV": None,
            "validation_average_mae_eV": _metrics(
                validation["base"].numpy(), validation["target"].numpy()
            )["average"]["mae"],
            "validation_route_fraction": 0.0,
        },
        {
            "mode": "always",
            "threshold_eV": None,
            "validation_average_mae_eV": _metrics(
                validation["dual"].numpy(), validation["target"].numpy()
            )["average"]["mae"],
            "validation_route_fraction": 1.0,
        },
    ])
    selected = min(
        candidates,
        key=lambda value: (
            value["validation_average_mae_eV"],
            value["validation_route_fraction"],
        ),
    )

    test = roles["test"]
    if selected["mode"] == "never":
        mask = torch.zeros(len(test["base"]), dtype=torch.bool)
    elif selected["mode"] == "always":
        mask = torch.ones(len(test["base"]), dtype=torch.bool)
    elif selected["mode"] == "below":
        mask = test["base"][:, 2] <= selected["threshold_eV"]
    else:
        mask = test["base"][:, 2] >= selected["threshold_eV"]
    routed = torch.where(mask[:, None], test["dual"], test["base"])
    return {
        "route_feature": "base_predicted_gap_eV",
        "selection_role": "validation",
        "selected": selected,
        "test_route_fraction": float(mask.float().mean()),
        "test_metrics": {
            "base": _metrics(test["base"].numpy(), test["target"].numpy()),
            "always_dual": _metrics(
                test["dual"].numpy(), test["target"].numpy()
            ),
            "routed": _metrics(routed.numpy(), test["target"].numpy()),
        },
    }


def _aligned_fusion_payloads(
    topology_payload: Path,
    geometry_payload: Path,
) -> dict:
    left = torch.load(topology_payload, map_location="cpu", weights_only=False)
    right = torch.load(geometry_payload, map_location="cpu", weights_only=False)
    aligned = {}
    for role in ("train", "validation", "test"):
        left_pos = {
            int(value): i
            for i, value in enumerate(left[role]["source_idx"].tolist())
        }
        right_pos = {
            int(value): i
            for i, value in enumerate(right[role]["source_idx"].tolist())
        }
        common = sorted(set(left_pos).intersection(right_pos))
        if not common:
            raise ValueError(f"No common fusion rows for {role}")
        left_index = torch.tensor([left_pos[index] for index in common])
        right_index = torch.tensor([right_pos[index] for index in common])
        left_targets = left[role]["targets"][left_index]
        right_targets = right[role]["targets"][right_index]
        if not torch.equal(left_targets, right_targets):
            raise ValueError(f"Mismatched fusion targets for {role}")
        aligned[role] = {
            "h2": left[role]["embeddings"][left_index],
            "h3": right[role]["embeddings"][right_index],
            "target": left_targets,
            "baseline": left[role]["predictions"][left_index],
            "source_idx": torch.tensor(common),
        }
    return aligned


def train_standard_fusion(
    *,
    topology_payload: Path,
    geometry_payload: Path,
    epochs: int,
    seed: int,
    hidden: int = 192,
    batch_size: int = 512,
    fusion_type: str = "gate",
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aligned = _aligned_fusion_payloads(topology_payload, geometry_payload)
    train_target = aligned["train"]["target"]
    mean = train_target.mean(dim=0)
    std = train_target.std(dim=0).clamp_min(1e-6)
    model = FusionHead(
        fusion_type,
        hidden=hidden,
        dropout=0.0,
        dim_2d=aligned["train"]["h2"].shape[1],
        dim_3d=aligned["train"]["h3"].shape[1],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-5
    )
    criterion = nn.L1Loss()

    def loader(role, shuffle):
        item = aligned[role]
        dataset = torch.utils.data.TensorDataset(
            item["h2"], item["h3"], (item["target"] - mean) / std
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

    @torch.no_grad()
    def evaluate(role):
        model.eval()
        outputs = []
        for h2, h3, _ in loader(role, False):
            outputs.append(model(h2.to(device), h3.to(device)).cpu())
        prediction = torch.cat(outputs) * std + mean
        return prediction, _metrics(
            prediction.numpy(), aligned[role]["target"].numpy()
        )

    best_state = None
    best_mae = float("inf")
    best_epoch = -1
    log = []
    wait = 0
    for epoch in range(epochs):
        started = time.perf_counter()
        model.train()
        for h2, h3, target in loader("train", True):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model(h2.to(device), h3.to(device)), target.to(device)
            )
            loss.backward()
            optimizer.step()
        _, metrics = evaluate("validation")
        value = metrics["average"]["mae"]
        improved = value < best_mae
        if improved:
            best_mae = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        log.append({
            "epoch": epoch,
            "validation_average_mae_eV": value,
            "elapsed_s": time.perf_counter() - started,
            "selected": improved,
        })
        if wait >= 10:
            break
    if best_state is None:
        raise RuntimeError("Fusion produced no checkpoint")
    model.load_state_dict(best_state)
    validation_prediction, _ = evaluate("validation")
    prediction, test_metrics = evaluate("test")
    baseline_metrics = _metrics(
        aligned["test"]["baseline"].numpy(),
        aligned["test"]["target"].numpy(),
    )
    return {
        "fusion": f"standard_embedding_{fusion_type}",
        "fusion_type": fusion_type,
        "hidden": hidden,
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "seed": seed,
        "aligned_rows": {
            role: len(value["source_idx"]) for role, value in aligned.items()
        },
        "best_epoch": best_epoch,
        "best_validation_average_mae_eV": best_mae,
        "test_metrics": test_metrics,
        "aligned_topology_baseline_metrics": baseline_metrics,
        "fusion_minus_topology_average_mae_eV": (
            test_metrics["average"]["mae"]
            - baseline_metrics["average"]["mae"]
        ),
        "log": log,
        "validation_predictions": validation_prediction,
        "validation_source_idx": aligned["validation"]["source_idx"],
        "predictions": prediction,
        "source_idx": aligned["test"]["source_idx"],
    }


def train_bounded_residual_fusion(
    *,
    topology_payload: Path,
    geometry_payload: Path,
    epochs: int,
    seed: int,
    hidden: int = 192,
    batch_size: int = 512,
    max_abs_correction_eV: float = 0.2,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aligned = _aligned_fusion_payloads(topology_payload, geometry_payload)
    model = BoundedResidualFusionHead(
        hidden=hidden,
        dim_2d=aligned["train"]["h2"].shape[1],
        dim_3d=aligned["train"]["h3"].shape[1],
        max_abs_correction_eV=max_abs_correction_eV,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-4, weight_decay=1e-5
    )
    criterion = nn.L1Loss()

    def loader(role, shuffle):
        item = aligned[role]
        dataset = torch.utils.data.TensorDataset(
            item["h2"], item["h3"], item["baseline"], item["target"]
        )
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle
        )

    @torch.no_grad()
    def evaluate(role):
        model.eval()
        outputs = []
        for h2, h3, baseline, _ in loader(role, False):
            outputs.append(
                model(
                    h2.to(device),
                    h3.to(device),
                    baseline.to(device),
                ).cpu()
            )
        prediction = torch.cat(outputs)
        metrics = _metrics(
            prediction.numpy(), aligned[role]["target"].numpy()
        )
        return prediction, metrics

    baseline_validation = _metrics(
        aligned["validation"]["baseline"].numpy(),
        aligned["validation"]["target"].numpy(),
    )
    best_state = copy.deepcopy(model.state_dict())
    best_mae = baseline_validation["average"]["mae"]
    best_epoch = -1
    log = [{
        "epoch": -1,
        "validation_average_mae_eV": best_mae,
        "elapsed_s": 0.0,
        "selected": True,
    }]
    wait = 0
    for epoch in range(epochs):
        started = time.perf_counter()
        model.train()
        for h2, h3, baseline, target in loader("train", True):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                h2.to(device), h3.to(device), baseline.to(device)
            )
            loss = criterion(prediction, target.to(device))
            loss.backward()
            optimizer.step()
        _, metrics = evaluate("validation")
        value = metrics["average"]["mae"]
        improved = value < best_mae
        if improved:
            best_mae = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        log.append({
            "epoch": epoch,
            "validation_average_mae_eV": value,
            "elapsed_s": time.perf_counter() - started,
            "selected": improved,
        })
        if wait >= 10:
            break

    model.load_state_dict(best_state)
    validation_prediction, _ = evaluate("validation")
    prediction, test_metrics = evaluate("test")
    baseline = aligned["test"]["baseline"]
    baseline_metrics = _metrics(
        baseline.numpy(), aligned["test"]["target"].numpy()
    )
    correction = prediction - baseline
    return {
        "fusion": "bounded_residual_identity_path",
        "seed": seed,
        "max_abs_correction_eV": max_abs_correction_eV,
        "aligned_rows": {
            role: len(value["source_idx"]) for role, value in aligned.items()
        },
        "best_epoch": best_epoch,
        "best_validation_average_mae_eV": best_mae,
        "test_metrics": test_metrics,
        "aligned_topology_baseline_metrics": baseline_metrics,
        "fusion_minus_topology_average_mae_eV": (
            test_metrics["average"]["mae"]
            - baseline_metrics["average"]["mae"]
        ),
        "test_correction_mean_abs_eV": correction.abs().mean(dim=0).tolist(),
        "test_correction_max_abs_eV": correction.abs().amax(dim=0).tolist(),
        "log": log,
        "validation_predictions": validation_prediction,
        "validation_source_idx": aligned["validation"]["source_idx"],
        "predictions": prediction,
        "source_idx": aligned["test"]["source_idx"],
    }
