"""Identity-safe evaluation of equal-weight prediction ensembles."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Iterable


TARGETS = ("homo", "lumo", "gap")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _prediction_columns(
    fieldnames: list[str],
    model_hint: str,
    targets: tuple[str, ...],
) -> dict[str, str]:
    matches: dict[str, str] = {}
    for target in targets:
        candidates = [
            name
            for name in fieldnames
            if name.endswith(f"_{target}") and model_hint in name
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one {model_hint!r} prediction column for {target}, "
                f"found {candidates}"
            )
        matches[target] = candidates[0]
    return matches


def _read_predictions(
    path: Path,
    *,
    model_hint: str,
    pcqm: bool,
) -> tuple[list[tuple[str, ...]], list[dict[str, float]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        targets = ("gap",) if pcqm else TARGETS
        prediction_columns = _prediction_columns(fields, model_hint, targets)
        identity_fields = ("cid", "smiles") if pcqm else ("eval_set", "cid", "smiles")
        missing = sorted(set(identity_fields + targets) - set(fields))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        identities: list[tuple[str, ...]] = []
        rows: list[dict[str, float]] = []
        for row in reader:
            identity = tuple(row[name] for name in identity_fields)
            values = {
                target: float(row[target])
                for target in targets
                if target in prediction_columns
            }
            values.update(
                {
                    f"pred_{target}": float(row[column])
                    for target, column in prediction_columns.items()
                }
            )
            if not all(math.isfinite(value) for value in values.values()):
                raise ValueError(f"Non-finite prediction row in {path}: {identity}")
            identities.append(identity)
            rows.append(values)
    if len(identities) != len(set(identities)):
        raise ValueError(f"Duplicate identities in {path}")
    return identities, rows


def _metrics(rows: list[dict[str, float]], predictions: list[dict[str, float]]) -> dict:
    per_target = {}
    for target in TARGETS:
        errors = [
            abs(prediction[target] - row[target])
            for row, prediction in zip(rows, predictions, strict=True)
        ]
        per_target[target] = {"mae_eV": sum(errors) / len(errors)}
    per_target["average"] = {
        "mae_eV": sum(per_target[target]["mae_eV"] for target in TARGETS) / 3
    }
    return per_target


def evaluate_equal_ensemble(
    common_paths: Iterable[Path],
    pcqm_paths: Iterable[Path],
    *,
    model_hint: str = "repaired_2m_d_gps7_seed",
) -> dict:
    common_paths = list(common_paths)
    pcqm_paths = list(pcqm_paths)
    if len(common_paths) < 2 or len(common_paths) != len(pcqm_paths):
        raise ValueError("Equal ensemble needs matching common/PCQM files for >=2 seeds")

    common_payloads = [
        _read_predictions(path, model_hint=model_hint, pcqm=False)
        for path in common_paths
    ]
    pcqm_payloads = [
        _read_predictions(path, model_hint=model_hint, pcqm=True)
        for path in pcqm_paths
    ]
    for label, payloads in (("common", common_payloads), ("pcqm", pcqm_payloads)):
        reference = payloads[0][0]
        for identities, _ in payloads[1:]:
            if identities != reference:
                raise ValueError(f"{label} seed prediction identities are not aligned")

    common_ids, common_rows = common_payloads[0]
    common_predictions = []
    for index in range(len(common_rows)):
        common_predictions.append(
            {
                target: sum(
                    payload[1][index][f"pred_{target}"]
                    for payload in common_payloads
                )
                / len(common_payloads)
                for target in TARGETS
            }
        )
    scopes = {"common": list(range(len(common_rows)))}
    for scope in ("ood1000", "p8_targeted_hard"):
        scopes[scope] = [
            index for index, identity in enumerate(common_ids) if identity[0] == scope
        ]
    common_metrics = {}
    for scope, indices in scopes.items():
        if not indices:
            continue
        common_metrics[scope] = {
            "n": len(indices),
            "metrics": _metrics(
                [common_rows[index] for index in indices],
                [common_predictions[index] for index in indices],
            ),
        }

    _, pcqm_rows = pcqm_payloads[0]
    pcqm_predictions = [
        {
            "gap": sum(
                payload[1][index]["pred_gap"] for payload in pcqm_payloads
            )
            / len(pcqm_payloads)
        }
        for index in range(len(pcqm_rows))
    ]
    pcqm_gap_errors = [
        abs(prediction["gap"] - row["gap"])
        for row, prediction in zip(pcqm_rows, pcqm_predictions, strict=True)
    ]
    return {
        "experiment": "retention_d_three_seed_equal_ensemble",
        "status": "accuracy_mode_candidate",
        "seeds": len(common_paths),
        "weights": [1.0 / len(common_paths)] * len(common_paths),
        "compute": {
            "gps7_encoder_passes_per_molecule": float(len(common_paths)),
        },
        "common": common_metrics,
        "pcqm": {
            "n": len(pcqm_rows),
            "gap": {"mae_eV": sum(pcqm_gap_errors) / len(pcqm_gap_errors)},
        },
        "inputs": [
            {
                "common_path": path.as_posix(),
                "common_sha256": sha256_file(path),
                "pcqm_path": pcqm.as_posix(),
                "pcqm_sha256": sha256_file(pcqm),
            }
            for path, pcqm in zip(common_paths, pcqm_paths, strict=True)
        ],
        "production_registry_changed": False,
        "sealed_20k_used": False,
    }
