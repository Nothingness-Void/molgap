"""Durable training and external evaluation for conservative 2D/3D fusion."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from .conservative_fusion_payload import (
    EXTERNAL_FORMAT,
    TRAINING_FORMAT,
    sha256_file,
)
from .hierarchical_fusion import (
    ConservativeFusionConfig,
    ConservativeHierarchicalResidualHead,
    fit_conservative_hierarchical_fusion,
    predict_conservative_hierarchical_fusion,
)
from .router import paired_bootstrap_mean


TARGETS = ("homo", "lumo", "gap")


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _tensor(payload: dict, name: str, shape: tuple[int | None, ...]) -> torch.Tensor:
    value = payload.get(name)
    if not isinstance(value, torch.Tensor) or value.ndim != len(shape):
        raise ValueError(f"Payload tensor {name} is missing or has the wrong rank")
    for actual, expected in zip(value.shape, shape):
        if expected is not None and actual != expected:
            raise ValueError(f"Payload tensor {name} has shape {tuple(value.shape)}")
    if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"Payload tensor {name} contains non-finite values")
    return value.cpu()


def _load_training(path: Path) -> dict[str, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != TRAINING_FORMAT:
        raise ValueError("Training payload format differs")
    source_idx = _tensor(payload, "source_idx", (None,))
    rows = source_idx.numel()
    values = {
        "source_idx": source_idx.numpy().astype(np.int64),
        "targets": _tensor(payload, "targets", (rows, 3)).numpy().astype(np.float32),
        "equal": _tensor(payload, "equal_prediction", (rows, 3)).numpy().astype(np.float32),
        "dense": _tensor(payload, "dense_prediction", (rows, 3)).numpy().astype(np.float32),
        "context": _tensor(payload, "context", (rows, None)).numpy().astype(np.float32),
        "train": _tensor(payload, "train_indices", (None,)).numpy().astype(np.int64),
        "validation": _tensor(payload, "validation_indices", (None,)).numpy().astype(np.int64),
        "test": _tensor(payload, "test_indices", (None,)).numpy().astype(np.int64),
    }
    if len(np.unique(values["source_idx"])) != rows:
        raise ValueError("Training source_idx values are duplicated")
    split = np.concatenate((values["train"], values["validation"], values["test"]))
    if len(split) != rows or not np.array_equal(np.sort(split), np.arange(rows)):
        raise ValueError("Training split is not an exact partition")
    return values


def _load_external(path: Path, context_dim: int) -> dict[str, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != EXTERNAL_FORMAT:
        raise ValueError("External payload format differs")
    source_idx = _tensor(payload, "source_idx", (None,))
    rows = source_idx.numel()
    values = {
        "source_idx": source_idx.numpy().astype(np.int64),
        "targets": _tensor(payload, "targets", (rows, 3)).numpy().astype(np.float32),
        "equal": _tensor(payload, "equal_prediction", (rows, 3)).numpy().astype(np.float32),
        "dense": _tensor(payload, "dense_prediction", (rows, 3)).numpy().astype(np.float32),
        "routed_v4": _tensor(payload, "routed_v4_prediction", (rows, 3)).numpy().astype(np.float32),
        "context": _tensor(payload, "context", (rows, context_dim)).numpy().astype(np.float32),
        "scope": _tensor(payload, "scope", (rows,)).numpy().astype(np.int8),
    }
    if not set(np.unique(values["scope"])).issubset({1, 2}):
        raise ValueError("External scope codes differ")
    return values


def _metrics(targets: np.ndarray, prediction: np.ndarray) -> dict:
    error = np.abs(prediction - targets)
    return {
        **{
            target: {"mae_eV": float(error[:, index].mean())}
            for index, target in enumerate(TARGETS)
        },
        "average": {"mae_eV": float(error.mean())},
    }


def _paired_delta(
    targets: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict:
    report = {}
    for index, target in enumerate((*TARGETS, "average")):
        if target == "average":
            delta = np.abs(candidate - targets).mean(axis=1) - np.abs(
                baseline - targets
            ).mean(axis=1)
        else:
            delta = np.abs(candidate[:, index] - targets[:, index]) - np.abs(
                baseline[:, index] - targets[:, index]
            )
        report[target] = paired_bootstrap_mean(
            delta, n_bootstrap=10_000, seed=42 + index
        )
    return report


def _load_model(path: Path, device: torch.device) -> ConservativeHierarchicalResidualHead:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("kind") != "conservative_hierarchical_2d_3d_residual":
        raise ValueError(f"Unexpected model kind in {path}")
    config = ConservativeFusionConfig(**payload["config"])
    state = payload["state_dict"]
    model = ConservativeHierarchicalResidualHead(
        len(state["feature_center"]),
        state["feature_center"].numpy(),
        state["feature_scale"].numpy(),
        config,
    )
    model.load_state_dict(state)
    return model.to(device).eval()


def run_conservative_fusion(
    *,
    training_payload_path: Path,
    external_payload_path: Path,
    checkpoint_dir: Path,
    results_dir: Path,
    device: str = "cuda",
    seeds: tuple[int, ...] = (42, 43, 44),
    config: ConservativeFusionConfig | None = None,
) -> dict:
    """Train or resume six heads, then run the fixed external gate."""
    started = time.time()
    device_object = torch.device(device)
    if device_object.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    training_hash = sha256_file(training_payload_path)
    external_hash = sha256_file(external_payload_path)
    training = _load_training(training_payload_path)
    external = _load_external(external_payload_path, training["context"].shape[1])
    config_base = config or ConservativeFusionConfig()
    contract = {
        "format": "molgap-conservative-fusion-run-v1",
        "training_payload_sha256": training_hash,
        "external_payload_sha256": external_hash,
        "rows": len(training["targets"]),
        "context_dim": training["context"].shape[1],
        "seeds": list(seeds),
        "config": asdict(config_base),
    }
    _atomic_json({"status": "running", **contract}, results_dir / "progress.json")

    internal_results, internal_predictions = {}, {}
    external_predictions, accepted_models = {}, {}
    for base_name in ("equal", "dense"):
        test_predictions = []
        models = []
        seed_reports = {}
        for seed in seeds:
            seed_config = replace(config_base, seed=seed)
            best_path = checkpoint_dir / f"{base_name}_seed{seed}.best.pt"
            last_path = checkpoint_dir / f"{base_name}_seed{seed}.last.pt"
            seed_result_path = results_dir / f"{base_name}_seed{seed}.json"
            if best_path.is_file() and seed_result_path.is_file():
                print(f"reusing accepted {base_name}/seed{seed}", flush=True)
                saved = json.loads(seed_result_path.read_text(encoding="utf-8"))
                if (
                    saved.get("training_payload_sha256") != training_hash
                    or saved.get("config") != asdict(seed_config)
                    or saved.get("model_sha256") != sha256_file(best_path)
                ):
                    raise ValueError(f"Completed seed contract differs: {base_name}/{seed}")
                model = _load_model(best_path, device_object)
                report = saved["training"]
            else:
                print(f"training {base_name}/seed{seed}", flush=True)
                contract_id = hashlib.sha256(
                    json.dumps(
                        {
                            **contract,
                            "base": base_name,
                            "seed": seed,
                            "config": asdict(seed_config),
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest()
                model, report = fit_conservative_hierarchical_fusion(
                    training[base_name],
                    training["context"],
                    training["targets"],
                    training["train"],
                    training["validation"],
                    config=seed_config,
                    device=device_object,
                    checkpoint_path=last_path,
                    progress_path=results_dir / f"{base_name}_seed{seed}.progress.json",
                    resume=last_path.is_file(),
                    contract_id=contract_id,
                    progress_label=f"{base_name}/seed{seed}",
                )
                model_payload = {
                    "kind": report["kind"],
                    "base": base_name,
                    "seed": seed,
                    "training_payload_sha256": training_hash,
                    "contract_id": contract_id,
                    "config": report["config"],
                    "state_dict": {
                        name: value.detach().cpu()
                        for name, value in model.state_dict().items()
                    },
                }
                _atomic_torch(model_payload, best_path)
                saved = {
                    "status": "accepted",
                    "base": base_name,
                    "seed": seed,
                    "training_payload_sha256": training_hash,
                    "config": asdict(seed_config),
                    "model_sha256": sha256_file(best_path),
                    "training": report,
                }
                _atomic_json(saved, seed_result_path)
                print(
                    f"completed {base_name}/seed{seed}: "
                    f"best={report['best_validation_mae_eV']:.6f}@{report['best_epoch']}",
                    flush=True,
                )
            prediction = predict_conservative_hierarchical_fusion(
                model,
                training[base_name][training["test"]],
                training["context"][training["test"]],
            )[0]
            test_predictions.append(prediction)
            models.append(model)
            seed_reports[str(seed)] = report
        ensemble = np.mean(test_predictions, axis=0)
        targets_test = training["targets"][training["test"]]
        base_test = training[base_name][training["test"]]
        internal_results[base_name] = {
            "base_metrics": _metrics(targets_test, base_test),
            "fusion_metrics": _metrics(targets_test, ensemble),
            "fusion_minus_base": _paired_delta(targets_test, base_test, ensemble),
            "training": seed_reports,
        }
        internal_predictions[base_name] = {
            "base": base_test,
            "fusion": ensemble,
        }
        external_predictions[base_name] = np.mean(
            [
                predict_conservative_hierarchical_fusion(
                    model, external[base_name], external["context"]
                )[0]
                for model in models
            ],
            axis=0,
        )
        accepted_models[base_name] = [
            {
                "seed": seed,
                "path": str(checkpoint_dir / f"{base_name}_seed{seed}.best.pt"),
                "sha256": sha256_file(
                    checkpoint_dir / f"{base_name}_seed{seed}.best.pt"
                ),
            }
            for seed in seeds
        ]

    scopes = {
        "all": np.ones(len(external["targets"]), dtype=bool),
        "ood1000": external["scope"] == 1,
        "p8_targeted_hard": external["scope"] == 2,
    }
    external_results = {}
    for scope_name, mask in scopes.items():
        targets = external["targets"][mask]
        methods = {
            "routed_v4_500k": external["routed_v4"][mask],
            "repaired_2m_equal_2d": external["equal"][mask],
            "repaired_2m_dense_2d": external["dense"][mask],
            "conservative_equal_2d3d": external_predictions["equal"][mask],
            "conservative_dense_2d3d": external_predictions["dense"][mask],
        }
        external_results[scope_name] = {
            "rows": int(mask.sum()),
            "methods": {name: _metrics(targets, value) for name, value in methods.items()},
            "fusion_delta_vs_own_2d": {
                base: _paired_delta(
                    targets,
                    methods[f"repaired_2m_{base}_2d"],
                    methods[f"conservative_{base}_2d3d"],
                )
                for base in ("equal", "dense")
            },
        }
    gates = {}
    for base in ("equal", "dense"):
        all_delta = external_results["all"]["fusion_delta_vs_own_2d"][base][
            "average"
        ]["delta"]
        hard_delta = external_results["p8_targeted_hard"][
            "fusion_delta_vs_own_2d"
        ][base]["average"]["delta"]
        gates[base] = {
            "all_average_delta_eV": all_delta,
            "p8_hard_average_delta_eV": hard_delta,
            "passed": all_delta <= 0.0005 and hard_delta <= -0.001,
        }

    internal_prediction_path = results_dir / "internal_test_predictions.pt"
    _atomic_torch(
        {
            "format": "molgap-conservative-fusion-internal-predictions-v1",
            "source_idx": torch.from_numpy(training["source_idx"][training["test"]]),
            "targets": torch.from_numpy(training["targets"][training["test"]]),
            "equal_2d": torch.from_numpy(internal_predictions["equal"]["base"]),
            "equal_2d3d": torch.from_numpy(internal_predictions["equal"]["fusion"]),
            "dense_2d": torch.from_numpy(internal_predictions["dense"]["base"]),
            "dense_2d3d": torch.from_numpy(internal_predictions["dense"]["fusion"]),
        },
        internal_prediction_path,
    )
    external_prediction_path = results_dir / "external_predictions.pt"
    _atomic_torch(
        {
            "format": "molgap-conservative-fusion-external-predictions-v1",
            "source_idx": torch.from_numpy(external["source_idx"]),
            "scope": torch.from_numpy(external["scope"]),
            "targets": torch.from_numpy(external["targets"]),
            "routed_v4_500k": torch.from_numpy(external["routed_v4"]),
            "repaired_2m_equal_2d": torch.from_numpy(external["equal"]),
            "repaired_2m_dense_2d": torch.from_numpy(external["dense"]),
            "conservative_equal_2d3d": torch.from_numpy(
                external_predictions["equal"]
            ),
            "conservative_dense_2d3d": torch.from_numpy(
                external_predictions["dense"]
            ),
        },
        external_prediction_path,
    )
    prediction_artifacts = {
        "internal_test_predictions": {
            "path": str(internal_prediction_path),
            "sha256": sha256_file(internal_prediction_path),
        },
        "external_predictions": {
            "path": str(external_prediction_path),
            "sha256": sha256_file(external_prediction_path),
        },
    }
    result = {
        "format": "molgap-conservative-fusion-result-v1",
        "status": "accepted" if any(item["passed"] for item in gates.values()) else "rejected",
        "device": str(device_object),
        "gpu": torch.cuda.get_device_name(0) if device_object.type == "cuda" else None,
        "elapsed_s": time.time() - started,
        "contract": contract,
        "internal": internal_results,
        "external": external_results,
        "promotion_gate": gates,
        "models": accepted_models,
        "prediction_artifacts": prediction_artifacts,
        "sealed_20k_used": False,
        "production_registry_changed": False,
    }
    metrics_path = results_dir / "metrics.json"
    _atomic_json(result, metrics_path)
    completion = {
        "status": "complete",
        "decision": result["status"],
        "metrics": {
            "path": str(metrics_path),
            "sha256": sha256_file(metrics_path),
        },
        "models": accepted_models,
        "prediction_artifacts": prediction_artifacts,
        "training_payload_sha256": training_hash,
        "external_payload_sha256": external_hash,
    }
    _atomic_json(completion, results_dir / "completion_manifest.json")
    _atomic_json({"status": "complete", **completion}, results_dir / "progress.json")
    return result
