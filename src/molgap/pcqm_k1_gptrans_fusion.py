"""Leakage-controlled scalar blend of full K1 and GPTrans-T predictions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


TRAINING_ROOT = Path("/lustre/home/users/sm2/chou/molgap-k1-gpttrans-full")
VALID_GRAPH_ROOT = Path(
    "/lustre/home/users/sm2/chou/molgap-pcqm-edge-state-full/rich_full/graphs"
)
FUSION_CONTRACT_PATH = Path(__file__).resolve().parents[2] / (
    "experiments/pcqm_k1_gptrans_full_fusion/fusion_contract.json"
)
EXPECTED_VALID_ROWS = 73_545
OFFICIAL_ROW_MANIFEST_SHA256 = (
    "c329fbde935324118f4f62aa6e2b01b209d6f54642549637853419deb1761c4b"
)
OFFICIAL_ARCHIVE_SHA256 = (
    "628ae612a3de752160929d93d1584a75257ae156e7b13d91b133d8db62e6d701"
)
CALIBRATION_MODULUS = 5
CALIBRATION_REMAINDER = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def calibration_mask(source_idx: np.ndarray) -> np.ndarray:
    """Use one deterministic fifth for fitting; keep four fifths untouched."""
    indices = np.asarray(source_idx, dtype=np.int64)
    if indices.ndim != 1 or len(np.unique(indices)) != len(indices):
        raise ValueError("source_idx must be a one-dimensional unique vector")
    return indices % CALIBRATION_MODULUS == CALIBRATION_REMAINDER


def fit_convex_blend_weight(
    k1_prediction: np.ndarray,
    gptrans_prediction: np.ndarray,
    target: np.ndarray,
) -> tuple[float, float]:
    """Select K1's convex weight on the frozen 0.001-wide MAE grid."""
    k1 = np.asarray(k1_prediction, dtype=np.float64)
    gp = np.asarray(gptrans_prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if k1.ndim != 1 or gp.shape != k1.shape or truth.shape != k1.shape or not len(k1):
        raise ValueError("blend inputs must be nonempty aligned vectors")
    if not (np.isfinite(k1).all() and np.isfinite(gp).all() and np.isfinite(truth).all()):
        raise ValueError("blend inputs must be finite")
    candidates = np.linspace(0.0, 1.0, 1001, dtype=np.float64)
    residuals = gp[None, :] + candidates[:, None] * (k1 - gp)[None, :] - truth[None, :]
    losses = np.abs(residuals).mean(axis=1)
    minimum = float(losses.min())
    tied = np.flatnonzero(np.isclose(losses, minimum, rtol=0.0, atol=1e-12))
    winner = min(tied.tolist(), key=lambda idx: (abs(candidates[idx] - 0.5), candidates[idx]))
    return float(candidates[winner]), float(losses[winner])


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.shape != target.shape or not prediction.size:
        raise ValueError("MAE inputs must be nonempty aligned vectors")
    if not (np.isfinite(prediction).all() and np.isfinite(target).all()):
        raise ValueError("MAE inputs must be finite")
    return float(np.abs(prediction - target).mean())


def _accepted_valid_shards(graph_root: Path) -> tuple[dict, list[dict]]:
    graph_root = graph_root.resolve()
    acceptance_path = graph_root / "acceptance.json"
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    expected = {
        "format": "molgap-pcqm4mv2-official-edge-state-graph-acceptance-v2",
        "status": "accepted",
        "archive_sha256": OFFICIAL_ARCHIVE_SHA256,
        "source_manifest_sha256": OFFICIAL_ROW_MANIFEST_SHA256,
        "feature_schema": "ogb",
        "node_feature_dim": 9,
        "edge_feature_dim": 3,
        "rwse_dim": 16,
        "counts": {"train": 3_378_606, "valid": EXPECTED_VALID_ROWS},
        "failures": 0,
        "test_graphs_built": False,
        "external_data_used": False,
    }
    if any(acceptance.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Official-valid graph acceptance identity changed")

    report_records = acceptance.get("reports", [])
    if len(report_records) != 75:
        raise RuntimeError("Official graph report inventory changed")
    valid_records = []
    for shard_index in range(67, 75):
        record = report_records[shard_index]
        report_path = (graph_root / record["report_path"]).resolve()
        report_path.relative_to(graph_root)
        if record.get("shard_index") != shard_index or sha256_file(report_path) != record.get("report_sha256"):
            raise RuntimeError(f"Official-valid report hash failed: shard {shard_index}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") != "complete"
            or report.get("feature_schema") != "ogb"
            or report.get("node_feature_dim") != 9
            or report.get("edge_feature_dim") != 3
            or report.get("rwse_dim") != 16
            or report.get("failures")
        ):
            raise RuntimeError(f"Official-valid report contract failed: shard {shard_index}")
        for item in report.get("files", []):
            if item.get("role") != "valid":
                continue
            path = (graph_root / item["path"]).resolve()
            path.relative_to(graph_root)
            if (
                path.stat().st_size != int(item["bytes"])
                or sha256_file(path) != item["sha256"]
            ):
                raise RuntimeError(f"Official-valid shard hash failed: {path.name}")
            valid_records.append(
                {
                    "shard_index": shard_index,
                    "path": str(path),
                    "rows": int(item["rows"]),
                    "bytes": int(item["bytes"]),
                    "sha256": item["sha256"],
                    "report_sha256": record["report_sha256"],
                }
            )
    if len(valid_records) != 8 or sum(item["rows"] for item in valid_records) != EXPECTED_VALID_ROWS:
        raise RuntimeError("Official-valid shard counts changed")
    return acceptance, valid_records


def _validate_fusion_contract(acceptance: dict, graph_acceptance_sha256: str) -> dict:
    contract = json.loads(FUSION_CONTRACT_PATH.read_text(encoding="utf-8"))
    role = contract.get("evaluation_role", {})
    calibration = contract.get("calibration", {})
    if (
        contract.get("format") != "molgap-pcqm-k1-gptrans-fusion-contract-v1"
        or role.get("graph_acceptance_sha256") != graph_acceptance_sha256
        or role.get("official_row_manifest_sha256") != acceptance.get("source_manifest_sha256")
        or role.get("rows") != EXPECTED_VALID_ROWS
        or role.get("validation_role_consumed_for_this_single_study") is not True
        or role.get("test_dev_read") is not False
        or role.get("test_challenge_read") is not False
        or calibration.get("calibration_predicate") != "source_idx % 5 == 0"
        or calibration.get("holdout_predicate") != "source_idx % 5 != 0"
        or calibration.get("fitted_parameters") != 1
        or calibration.get("alpha_grid_step") != 0.001
        or calibration.get("objective") != "calibration MAE in eV"
    ):
        raise RuntimeError("Frozen fusion contract does not match the accepted graph cache")
    return contract


def _load_model_bundles(training_root: Path):
    import torch

    from .pcqm_gptrans_full_runner import (
        EXPECTED_INITIAL_SHA256 as GPTRANS_INITIAL_SHA256,
        MODEL_ID as GPTRANS_MODEL_ID,
        TRAINING_CONTRACT_SHA256 as GPTRANS_CONTRACT_SHA256,
        _make_model as make_gptrans,
        _state_sha256 as state_sha256,
    )
    from .pcqm_k1_full_runner import (
        EXPECTED_INITIAL_MODEL_SHA256 as K1_INITIAL_SHA256,
        EXPECTED_PARAMETERS as K1_PARAMETERS,
        TRAINING_CONTRACT_SHA256 as K1_CONTRACT_SHA256,
        _state_sha256 as k1_state_sha256,
    )
    from .qm9_neural_atom import make_encoder
    from .training_reproducibility import configure_fp32_determinism

    k1_dir = training_root / "outputs/k1/full"
    gp_dir = training_root / "outputs/gptrans/full"
    k1_accept = json.loads((k1_dir / "acceptance.json").read_text(encoding="utf-8"))
    gp_accept = json.loads((gp_dir / "acceptance.json").read_text(encoding="utf-8"))
    if k1_accept.get("accepted") is not True or gp_accept.get("accepted") is not True:
        raise RuntimeError("Both base-model acceptances are required")
    if k1_accept.get("training_contract_sha256") != K1_CONTRACT_SHA256:
        raise RuntimeError("K1 acceptance contract differs")
    if gp_accept.get("training_contract_sha256") != GPTRANS_CONTRACT_SHA256:
        raise RuntimeError("GPTrans-T acceptance contract differs")

    bundles = {}
    for name, directory, acceptance in (
        ("k1", k1_dir, k1_accept),
        ("gptrans", gp_dir, gp_accept),
    ):
        bundle_path = directory / "model_bundle.pt"
        bundle_sha = sha256_file(bundle_path)
        if bundle_sha != acceptance.get("model_bundle_sha256"):
            raise RuntimeError(f"{name} model bundle SHA changed")
        bundles[name] = torch.load(bundle_path, map_location="cpu", weights_only=False)

    configure_fp32_determinism(42)
    k1_model = make_encoder("neural_atom_k1")
    if sum(parameter.numel() for parameter in k1_model.parameters()) != K1_PARAMETERS:
        raise RuntimeError("K1 parameter count changed")
    if k1_state_sha256(k1_model) != K1_INITIAL_SHA256:
        raise RuntimeError("K1 seed-42 initial state changed")
    k1_bundle = bundles["k1"]
    if k1_bundle.get("training_contract_sha256") != K1_CONTRACT_SHA256:
        raise RuntimeError("K1 bundle contract differs")
    k1_model.load_state_dict(k1_bundle["state_dict"], strict=True)

    configure_fp32_determinism(42)
    gp_model = make_gptrans()
    if sum(parameter.numel() for parameter in gp_model.parameters()) != 5_246_817:
        raise RuntimeError("GPTrans-T parameter count changed")
    if state_sha256(gp_model) != GPTRANS_INITIAL_SHA256:
        raise RuntimeError("GPTrans-T seed-42 initial state changed")
    gp_bundle = bundles["gptrans"]
    if gp_bundle.get("training_contract_sha256") != GPTRANS_CONTRACT_SHA256:
        raise RuntimeError("GPTrans-T bundle contract differs")
    gp_model.load_state_dict(gp_bundle["state_dict"], strict=True)

    if k1_bundle.get("model_id") != "neural_atom_k1" or gp_bundle.get("model_id") != GPTRANS_MODEL_ID:
        raise RuntimeError("Base-model bundle identity changed")
    return k1_model, gp_model, bundles, k1_accept, gp_accept


def _predict_shard(path: Path, *, k1_model, gp_model, bundles, device):
    import torch
    from torch_geometric.data import InMemoryDataset
    from torch_geometric.loader import DataLoader

    class PackedGraphDataset(InMemoryDataset):
        def __init__(self, shard_path: Path):
            super().__init__(root=None)
            self.data, self.slices = torch.load(
                shard_path, map_location="cpu", weights_only=False
            )

    dataset = PackedGraphDataset(path)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)
    k1_mean = float(bundles["k1"]["target_stats"]["mean_eV"])
    k1_std = float(bundles["k1"]["target_stats"]["sample_std_eV"])
    gp_mean = float(bundles["gptrans"]["target_stats"]["mean_eV"])
    gp_std = float(bundles["gptrans"]["target_stats"]["sample_std_eV"])
    indices, targets, k1_predictions, gp_predictions = [], [], [], []
    k1_model.eval()
    gp_model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            k1_normalized = k1_model(
                batch.x,
                batch.edge_index,
                batch.edge_attr,
                batch.batch,
                batch.random_walk_pe,
            ).view(-1)
            gp_normalized = gp_model(
                batch.x, batch.edge_index, batch.edge_attr, batch.batch
            ).view(-1)
            k1_predictions.append((k1_normalized.float() * k1_std + k1_mean).cpu())
            gp_predictions.append((gp_normalized.float() * gp_std + gp_mean).cpu())
            targets.append(batch.y.view(-1).float().cpu())
            indices.append(batch.source_idx.view(-1).long().cpu())
    return {
        "source_idx": torch.cat(indices),
        "target_eV": torch.cat(targets),
        "k1_prediction_eV": torch.cat(k1_predictions),
        "gptrans_prediction_eV": torch.cat(gp_predictions),
    }


def run_fusion_study(
    *,
    training_root: Path = TRAINING_ROOT,
    graph_root: Path = VALID_GRAPH_ROOT,
    output: Path,
) -> dict:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Fusion study requires one visible A100")
    if "A100" not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError(f"Expected A100, got {torch.cuda.get_device_name(0)}")

    training_root = Path(training_root).resolve()
    graph_root = Path(graph_root).resolve()
    output = Path(output).resolve()
    if not output.is_relative_to(training_root):
        raise ValueError("Fusion output must remain under the authorized experiment root")
    acceptance, valid_shards = _accepted_valid_shards(graph_root)
    graph_acceptance_sha256 = sha256_file(graph_root / "acceptance.json")
    _validate_fusion_contract(acceptance, graph_acceptance_sha256)
    k1_model, gp_model, bundles, k1_accept, gp_accept = _load_model_bundles(training_root)
    identity = {
        "format": "molgap-pcqm-k1-gptrans-fusion-identity-v1",
        "fusion_code_sha256": sha256_file(Path(__file__)),
        "fusion_contract_sha256": sha256_file(FUSION_CONTRACT_PATH),
        "k1_bundle_sha256": k1_accept["model_bundle_sha256"],
        "k1_training_contract_sha256": k1_accept["training_contract_sha256"],
        "gptrans_bundle_sha256": gp_accept["model_bundle_sha256"],
        "gptrans_training_contract_sha256": gp_accept["training_contract_sha256"],
        "graph_acceptance_sha256": graph_acceptance_sha256,
        "official_row_manifest_sha256": acceptance["source_manifest_sha256"],
        "calibration_rule": "source_idx_mod_5_eq_0_v1",
        "blend_grid": {"k1_weight_min": 0.0, "k1_weight_max": 1.0, "step": 0.001},
        "precision": "fp32",
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("Existing fusion output identity differs")
    else:
        atomic_json(identity_path, identity)

    device = torch.device("cuda")
    k1_model.to(device)
    gp_model.to(device)
    parts_dir = output / "prediction_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    completed_parts: dict[str, dict] = {}
    for record in valid_shards:
        part_path = parts_dir / f"valid_{record['shard_index']:04d}.pt"
        sidecar_path = part_path.with_suffix(".json")
        expected_part = {
            "shard_index": record["shard_index"],
            "graph_sha256": record["sha256"],
            "k1_bundle_sha256": identity["k1_bundle_sha256"],
            "gptrans_bundle_sha256": identity["gptrans_bundle_sha256"],
            "rows": record["rows"],
        }
        if part_path.exists() or sidecar_path.exists():
            if part_path.is_file() and not sidecar_path.exists():
                payload = torch.load(part_path, map_location="cpu", weights_only=False)
                if (
                    len(payload.get("source_idx", [])) != record["rows"]
                    or not torch.isfinite(payload.get("target_eV", torch.tensor(float("nan")))).all()
                ):
                    raise RuntimeError(f"Orphan prediction shard is invalid: {part_path.name}")
                sidecar = {
                    "format": "molgap-pcqm-k1-gptrans-fusion-prediction-part-v1",
                    **expected_part,
                    "sha256": sha256_file(part_path),
                }
                atomic_json(sidecar_path, sidecar)
                completed_parts[part_path.name] = sidecar
                continue
            if not part_path.is_file() or not sidecar_path.is_file():
                raise RuntimeError(f"Incomplete prediction shard exists: {part_path.name}")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if any(sidecar.get(key) != value for key, value in expected_part.items()):
                raise RuntimeError(f"Prediction shard identity differs: {part_path.name}")
            if sha256_file(part_path) != sidecar.get("sha256"):
                raise RuntimeError(f"Prediction shard hash differs: {part_path.name}")
            completed_parts[part_path.name] = sidecar
            continue

        payload = _predict_shard(
            Path(record["path"]),
            k1_model=k1_model,
            gp_model=gp_model,
            bundles=bundles,
            device=device,
        )
        if len(payload["source_idx"]) != record["rows"]:
            raise RuntimeError(f"Prediction row count changed: {part_path.name}")
        if not torch.isfinite(payload["target_eV"]).all():
            raise RuntimeError(f"Non-finite official-valid labels: {part_path.name}")
        atomic_torch_save(part_path, payload)
        sidecar = {
            "format": "molgap-pcqm-k1-gptrans-fusion-prediction-part-v1",
            **expected_part,
            "sha256": sha256_file(part_path),
        }
        atomic_json(sidecar_path, sidecar)
        completed_parts[part_path.name] = sidecar
        atomic_json(output / "progress.json", {
            "status": "predicting",
            "completed_parts": sorted(completed_parts),
            "expected_parts": len(valid_shards),
            "rows_completed": sum(item["rows"] for item in completed_parts.values()),
        })

    ordered_names = [f"valid_{record['shard_index']:04d}.pt" for record in valid_shards]
    if sorted(completed_parts) != sorted(ordered_names):
        raise RuntimeError("Prediction part inventory is incomplete")
    payloads = [
        torch.load(parts_dir / name, map_location="cpu", weights_only=False)
        for name in ordered_names
    ]
    source_idx = torch.cat([item["source_idx"] for item in payloads]).numpy().astype(np.int64)
    target = torch.cat([item["target_eV"] for item in payloads]).numpy().astype(np.float64)
    k1_prediction = torch.cat([item["k1_prediction_eV"] for item in payloads]).numpy().astype(np.float64)
    gp_prediction = torch.cat([item["gptrans_prediction_eV"] for item in payloads]).numpy().astype(np.float64)
    order = np.argsort(source_idx, kind="stable")
    source_idx, target = source_idx[order], target[order]
    k1_prediction, gp_prediction = k1_prediction[order], gp_prediction[order]
    if (
        len(source_idx) != EXPECTED_VALID_ROWS
        or len(np.unique(source_idx)) != EXPECTED_VALID_ROWS
        or not np.isfinite(target).all()
    ):
        raise RuntimeError("Official-valid source_idx coverage or labels changed")

    calibration = calibration_mask(source_idx)
    holdout = ~calibration
    alpha_k1, calibration_mae = fit_convex_blend_weight(
        k1_prediction[calibration], gp_prediction[calibration], target[calibration]
    )
    equal_blend = 0.5 * (k1_prediction + gp_prediction)
    calibrated_blend = alpha_k1 * k1_prediction + (1.0 - alpha_k1) * gp_prediction
    metrics = {
        "format": "molgap-pcqm-k1-gptrans-fusion-study-v1",
        "status": "complete",
        "models": {
            "k1_bundle_sha256": identity["k1_bundle_sha256"],
            "k1_training_contract_sha256": identity["k1_training_contract_sha256"],
            "gptrans_bundle_sha256": identity["gptrans_bundle_sha256"],
            "gptrans_training_contract_sha256": identity["gptrans_training_contract_sha256"],
        },
        "official_valid": {
            "rows": len(source_idx),
            "graph_acceptance_sha256": graph_acceptance_sha256,
            "official_row_manifest_sha256": acceptance["source_manifest_sha256"],
            "source_idx_sha256": hashlib.sha256(source_idx.tobytes()).hexdigest(),
            "k1_mae_eV": mae(k1_prediction, target),
            "gptrans_t_mae_eV": mae(gp_prediction, target),
            "fixed_equal_blend_mae_eV": mae(equal_blend, target),
        },
        "calibration_holdout": {
            "split": "source_idx modulo 5 equals 0 for calibration; all other rows held out",
            "calibration_rows": int(calibration.sum()),
            "holdout_rows": int(holdout.sum()),
            "optimized_objective": "MAE eV",
            "grid_step": 0.001,
            "k1_weight": alpha_k1,
            "gptrans_t_weight": 1.0 - alpha_k1,
            "calibration_k1_mae_eV": mae(k1_prediction[calibration], target[calibration]),
            "calibration_gptrans_t_mae_eV": mae(gp_prediction[calibration], target[calibration]),
            "calibration_equal_blend_mae_eV": mae(equal_blend[calibration], target[calibration]),
            "calibration_calibrated_blend_mae_eV": calibration_mae,
            "holdout_k1_mae_eV": mae(k1_prediction[holdout], target[holdout]),
            "holdout_gptrans_t_mae_eV": mae(gp_prediction[holdout], target[holdout]),
            "holdout_equal_blend_mae_eV": mae(equal_blend[holdout], target[holdout]),
            "holdout_calibrated_blend_mae_eV": mae(
                calibrated_blend[holdout], target[holdout]
            ),
        },
        "official_validation_role_read": True,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "production_registry_changed": False,
    }
    prediction_path = output / "official_valid_predictions.pt"
    atomic_torch_save(prediction_path, {
        "source_idx": torch.from_numpy(source_idx.copy()),
        "target_eV": torch.from_numpy(target.astype(np.float32)),
        "k1_prediction_eV": torch.from_numpy(k1_prediction.astype(np.float32)),
        "gptrans_t_prediction_eV": torch.from_numpy(gp_prediction.astype(np.float32)),
        "fixed_equal_blend_eV": torch.from_numpy(equal_blend.astype(np.float32)),
        "calibrated_blend_eV": torch.from_numpy(calibrated_blend.astype(np.float32)),
        "calibration_mask": torch.from_numpy(calibration.copy()),
    })
    metrics["prediction_sha256"] = sha256_file(prediction_path)
    atomic_json(output / "metrics.json", metrics)
    artifacts = {
        "identity.json": sha256_file(identity_path),
        "metrics.json": sha256_file(output / "metrics.json"),
        "official_valid_predictions.pt": metrics["prediction_sha256"],
        **{
            f"prediction_parts/{name}": sidecar["sha256"]
            for name, sidecar in sorted(completed_parts.items())
        },
        **{
            f"prediction_parts/{Path(name).with_suffix('.json').name}": sha256_file(
                parts_dir / Path(name).with_suffix(".json").name
            )
            for name in sorted(completed_parts)
        },
    }
    completion = {
        "format": "molgap-pcqm-k1-gptrans-fusion-completion-v1",
        "status": "complete",
        "accepted": False,
        "identity": identity,
        "artifacts": artifacts,
        "metrics": metrics,
    }
    atomic_json(output / "completion_manifest.json", completion)
    atomic_json(output / "progress.json", {
        "status": "complete",
        "completed_parts": sorted(completed_parts),
        "expected_parts": len(valid_shards),
        "rows_completed": len(source_idx),
        "completion_manifest_sha256": sha256_file(output / "completion_manifest.json"),
    })
    return completion


def accept_fusion_study(
    *,
    training_root: Path = TRAINING_ROOT,
    graph_root: Path = VALID_GRAPH_ROOT,
    output: Path,
) -> dict:
    """Verify fusion identities, row alignment, metrics, and durable artifacts."""
    import torch

    from .pcqm_gptrans_full_runner import TRAINING_CONTRACT_SHA256 as GP_CONTRACT_SHA256
    from .pcqm_k1_full_runner import TRAINING_CONTRACT_SHA256 as K1_CONTRACT_SHA256

    training_root = Path(training_root).resolve()
    graph_root = Path(graph_root).resolve()
    output = Path(output).resolve()
    if not output.is_relative_to(training_root):
        raise ValueError("Fusion output must remain under the authorized experiment root")

    graph_acceptance, valid_shards = _accepted_valid_shards(graph_root)
    graph_acceptance_sha = sha256_file(graph_root / "acceptance.json")
    _validate_fusion_contract(graph_acceptance, graph_acceptance_sha)
    identity_path = output / "identity.json"
    metrics_path = output / "metrics.json"
    predictions_path = output / "official_valid_predictions.pt"
    completion_path = output / "completion_manifest.json"
    progress_path = output / "progress.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))

    expected_identity = {
        "format": "molgap-pcqm-k1-gptrans-fusion-identity-v1",
        "fusion_contract_sha256": sha256_file(FUSION_CONTRACT_PATH),
        "graph_acceptance_sha256": graph_acceptance_sha,
        "official_row_manifest_sha256": graph_acceptance["source_manifest_sha256"],
        "calibration_rule": "source_idx_mod_5_eq_0_v1",
        "blend_grid": {"k1_weight_min": 0.0, "k1_weight_max": 1.0, "step": 0.001},
        "precision": "fp32",
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        raise RuntimeError("Fusion identity or role contract changed")
    if identity.get("fusion_code_sha256") != sha256_file(Path(__file__)):
        raise RuntimeError("Fusion implementation hash differs from the executed source")

    base_acceptances = {}
    for name, contract_sha in (("k1", K1_CONTRACT_SHA256), ("gptrans", GP_CONTRACT_SHA256)):
        base_dir = training_root / "outputs" / name / "full"
        base_accept = json.loads((base_dir / "acceptance.json").read_text(encoding="utf-8"))
        bundle_path = base_dir / "model_bundle.pt"
        if (
            base_accept.get("accepted") is not True
            or base_accept.get("training_contract_sha256") != contract_sha
            or sha256_file(bundle_path) != base_accept.get("model_bundle_sha256")
        ):
            raise RuntimeError(f"Base model acceptance failed: {name}")
        base_acceptances[name] = base_accept
    for name, bundle_key, contract_key in (
        ("k1", "k1_bundle_sha256", "k1_training_contract_sha256"),
        ("gptrans", "gptrans_bundle_sha256", "gptrans_training_contract_sha256"),
    ):
        base_accept = base_acceptances[name]
        if (
            identity.get(bundle_key) != base_accept.get("model_bundle_sha256")
            or identity.get(contract_key) != base_accept.get("training_contract_sha256")
        ):
            raise RuntimeError(f"Fusion identity differs from accepted {name}")

    if (
        completion.get("format") != "molgap-pcqm-k1-gptrans-fusion-completion-v1"
        or completion.get("status") != "complete"
        or completion.get("accepted") is not False
        or completion.get("identity") != identity
        or completion.get("metrics") != metrics
    ):
        raise RuntimeError("Fusion completion manifest is incomplete or mismatched")
    if (
        metrics.get("format") != "molgap-pcqm-k1-gptrans-fusion-study-v1"
        or metrics.get("status") != "complete"
        or metrics.get("official_validation_role_read") is not True
        or metrics.get("test_dev_role_read") is not False
        or metrics.get("test_challenge_role_read") is not False
        or metrics.get("production_registry_changed") is not False
    ):
        raise RuntimeError("Fusion metrics violate the frozen role boundary")
    if (
        progress.get("status") != "complete"
        or progress.get("expected_parts") != len(valid_shards)
        or progress.get("rows_completed") != EXPECTED_VALID_ROWS
        or progress.get("completion_manifest_sha256") != sha256_file(completion_path)
    ):
        raise RuntimeError("Fusion progress marker is incomplete")

    for relative, expected_sha in completion.get("artifacts", {}).items():
        artifact_path = (output / relative).resolve()
        artifact_path.relative_to(output)
        if not artifact_path.is_file() or sha256_file(artifact_path) != expected_sha:
            raise RuntimeError(f"Fusion artifact SHA failed: {relative}")
    expected_artifacts = {
        "identity.json",
        "metrics.json",
        "official_valid_predictions.pt",
    }
    for record in valid_shards:
        name = f"valid_{record['shard_index']:04d}"
        expected_artifacts.add(f"prediction_parts/{name}.pt")
        expected_artifacts.add(f"prediction_parts/{name}.json")
    if set(completion.get("artifacts", {})) != expected_artifacts:
        raise RuntimeError("Fusion completion artifact inventory changed")

    final = torch.load(predictions_path, map_location="cpu", weights_only=False)
    expected_keys = {
        "source_idx", "target_eV", "k1_prediction_eV", "gptrans_t_prediction_eV",
        "fixed_equal_blend_eV", "calibrated_blend_eV", "calibration_mask",
    }
    if set(final) != expected_keys:
        raise RuntimeError("Final prediction payload schema changed")
    final_arrays = {key: value.detach().cpu().numpy() for key, value in final.items()}
    indices = final_arrays["source_idx"].astype(np.int64, copy=False)
    if (
        len(indices) != EXPECTED_VALID_ROWS
        or not np.array_equal(indices, np.sort(indices, kind="stable"))
        or len(np.unique(indices)) != EXPECTED_VALID_ROWS
    ):
        raise RuntimeError("Final official-valid source_idx coverage changed")

    calibration = calibration_mask(indices)
    if not np.array_equal(final_arrays["calibration_mask"].astype(bool), calibration):
        raise RuntimeError("Fusion calibration/holdout membership changed")
    target = final_arrays["target_eV"].astype(np.float64)
    k1 = final_arrays["k1_prediction_eV"].astype(np.float64)
    gp = final_arrays["gptrans_t_prediction_eV"].astype(np.float64)
    equal = 0.5 * (k1 + gp)
    calibrated_alpha, calibration_mae = fit_convex_blend_weight(
        k1[calibration], gp[calibration], target[calibration]
    )
    calibrated = calibrated_alpha * k1 + (1.0 - calibrated_alpha) * gp
    if not np.array_equal(final_arrays["fixed_equal_blend_eV"], equal.astype(np.float32)):
        raise RuntimeError("Fixed equal-blend predictions do not reproduce")
    if not np.allclose(
        final_arrays["calibrated_blend_eV"], calibrated.astype(np.float32), rtol=0.0, atol=1e-7
    ):
        raise RuntimeError("Calibrated blend predictions do not reproduce")
    heldout = ~calibration
    split_metrics = metrics.get("calibration_holdout", {})
    if (
        int(calibration.sum()) != split_metrics.get("calibration_rows")
        or int(heldout.sum()) != split_metrics.get("holdout_rows")
        or calibrated_alpha != split_metrics.get("k1_weight")
        or metrics.get("official_valid", {}).get("rows") != EXPECTED_VALID_ROWS
        or metrics.get("official_valid", {}).get("source_idx_sha256")
        != hashlib.sha256(indices.tobytes()).hexdigest()
    ):
        raise RuntimeError("Fusion split or selected scalar weight changed")

    recomputed = {
        ("official_valid", "k1_mae_eV"): mae(k1, target),
        ("official_valid", "gptrans_t_mae_eV"): mae(gp, target),
        ("official_valid", "fixed_equal_blend_mae_eV"): mae(equal, target),
        ("calibration_holdout", "calibration_k1_mae_eV"): mae(k1[calibration], target[calibration]),
        ("calibration_holdout", "calibration_gptrans_t_mae_eV"): mae(gp[calibration], target[calibration]),
        ("calibration_holdout", "calibration_equal_blend_mae_eV"): mae(equal[calibration], target[calibration]),
        ("calibration_holdout", "calibration_calibrated_blend_mae_eV"): calibration_mae,
        ("calibration_holdout", "holdout_k1_mae_eV"): mae(k1[heldout], target[heldout]),
        ("calibration_holdout", "holdout_gptrans_t_mae_eV"): mae(gp[heldout], target[heldout]),
        ("calibration_holdout", "holdout_equal_blend_mae_eV"): mae(equal[heldout], target[heldout]),
        ("calibration_holdout", "holdout_calibrated_blend_mae_eV"): mae(
            calibrated[heldout], target[heldout]
        ),
    }
    for (section, key), expected_value in recomputed.items():
        actual = metrics.get(section, {}).get(key)
        if actual is None or not np.isclose(float(actual), expected_value, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"Fusion metric does not reproduce: {section}.{key}")

    part_rows = 0
    prediction_parts = []
    for record in valid_shards:
        part_path = output / "prediction_parts" / f"valid_{record['shard_index']:04d}.pt"
        sidecar_path = part_path.with_suffix(".json")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if (
            sidecar.get("shard_index") != record["shard_index"]
            or sidecar.get("graph_sha256") != record["sha256"]
            or sidecar.get("rows") != record["rows"]
            or sidecar.get("k1_bundle_sha256") != identity["k1_bundle_sha256"]
            or sidecar.get("gptrans_bundle_sha256") != identity["gptrans_bundle_sha256"]
            or sidecar.get("sha256") != sha256_file(part_path)
        ):
            raise RuntimeError(f"Fusion prediction part identity failed: {part_path.name}")
        part = torch.load(part_path, map_location="cpu", weights_only=False)
        part_rows += len(part["source_idx"])
        if len(part["source_idx"]) != record["rows"]:
            raise RuntimeError(f"Fusion prediction part count failed: {part_path.name}")
        prediction_parts.append(part)
    if part_rows != EXPECTED_VALID_ROWS:
        raise RuntimeError("Prediction parts do not cover official validation")
    joined = {
        key: torch.cat([part[key] for part in prediction_parts]).numpy()
        for key in ("source_idx", "target_eV", "k1_prediction_eV", "gptrans_prediction_eV")
    }
    part_order = np.argsort(joined["source_idx"], kind="stable")
    for part_key, final_key in (
        ("source_idx", "source_idx"),
        ("target_eV", "target_eV"),
        ("k1_prediction_eV", "k1_prediction_eV"),
        ("gptrans_prediction_eV", "gptrans_t_prediction_eV"),
    ):
        if not np.array_equal(joined[part_key][part_order], final_arrays[final_key]):
            raise RuntimeError(f"Final prediction payload does not match parts: {part_key}")

    result = {
        "format": "molgap-pcqm-k1-gptrans-fusion-acceptance-v1",
        "accepted": True,
        "acceptance_scope": "mechanical_artifact_and_protocol_checks_only",
        "identity_sha256": sha256_file(identity_path),
        "completion_manifest_sha256": sha256_file(completion_path),
        "metrics_sha256": sha256_file(metrics_path),
        "prediction_sha256": sha256_file(predictions_path),
        "official_valid_rows": EXPECTED_VALID_ROWS,
        "calibration_rows": int(calibration.sum()),
        "holdout_rows": int(heldout.sum()),
        "k1_weight": calibrated_alpha,
        "gptrans_t_weight": 1.0 - calibrated_alpha,
        "official_validation_role_read": True,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "scientific_advancement_decided": False,
    }
    atomic_json(output / "acceptance.json", result)
    return result


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "accept"):
        child = subparsers.add_parser(command)
        child.add_argument("--training-root", type=Path, default=TRAINING_ROOT)
        child.add_argument("--graph-root", type=Path, default=VALID_GRAPH_ROOT)
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run_fusion_study(
            training_root=args.training_root,
            graph_root=args.graph_root,
            output=args.output,
        )
    else:
        result = accept_fusion_study(
            training_root=args.training_root,
            graph_root=args.graph_root,
            output=args.output,
        )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
