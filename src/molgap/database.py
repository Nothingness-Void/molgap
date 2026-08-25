"""Auditable batch inference for the Track A property database.

The database build deliberately keeps row validation separate from model
inference.  A molecule can be outside the trained domain and still receive a
finite screening prediction; only rows that cannot be parsed or graphed are
rejected before inference.  The raw ledger therefore remains useful for audit
and for producing a later filtered release view.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from .constants import MODEL_REGISTRY, REPO_ROOT, TARGET_COLS
from .pubchemqc import ALLOWED_ELEMENTS
from .utils import canonicalize_smiles, safe_mol


SCHEMA_VERSION = 1
FORMAT = "molgap-database-build-v1"
TRAINED_MW_RANGE = (200.0, 1000.0)
DEFAULT_MODEL_KEY = "repaired_2m_dense_2d"

_LEDGER_COLUMNS = (
    "input_row",
    "source_id",
    "dedup_key",
    "duplicate_identity",
    "validity",
    "valid_smiles",
    "canonical_smiles",
    "elements",
    "unsupported_elements",
    "mw",
    "allowed_elements",
    "mw_in_trained_range",
    "graph_success",
    "in_domain",
    "applicability_reason",
    "prediction_status",
    "rejection_reason",
    "expert_count",
    "homo",
    "lumo",
    "gap",
    "expert_disagreement_homo_eV",
    "expert_disagreement_lumo_eV",
    "expert_disagreement_gap_eV",
)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str, *, overwrite: bool) -> None:
    """Write one file via same-directory temp file and ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _json_list(values: Iterable[str]) -> str:
    return json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))


def _normalise_source_id(value: object, fallback: str) -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _resolve_id_column(frame: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"ID column {requested!r} is not present in the input")
        return requested
    for candidate in ("source_id", "cid", "id", "name"):
        if candidate in frame.columns:
            return candidate
    return None


def _inspect_smiles(smiles: object) -> dict[str, object]:
    """Return deterministic parsing and applicability facts for one SMILES."""
    mol = safe_mol(smiles)
    if mol is None:
        raw = "" if smiles is None else str(smiles).strip()
        return {
            "valid_smiles": False,
            "canonical_smiles": None,
            "elements": (),
            "unsupported_elements": (),
            "mw": None,
            "allowed_elements": False,
            "mw_in_trained_range": False,
            "in_domain": False,
            "applicability_reason": "invalid_smiles",
            "dedup_key": f"invalid::{raw}",
        }

    canonical = canonicalize_smiles(smiles)
    if canonical is None:
        return {
            "valid_smiles": False,
            "canonical_smiles": None,
            "elements": (),
            "unsupported_elements": (),
            "mw": None,
            "allowed_elements": False,
            "mw_in_trained_range": False,
            "in_domain": False,
            "applicability_reason": "invalid_smiles",
            "dedup_key": f"invalid::{str(smiles).strip()}",
        }

    from rdkit.Chem import Descriptors

    elements = tuple(sorted({atom.GetSymbol() for atom in mol.GetAtoms()}))
    unsupported = tuple(sorted(set(elements).difference(ALLOWED_ELEMENTS)))
    mw = float(Descriptors.MolWt(mol))
    mw_in_range = bool(TRAINED_MW_RANGE[0] <= mw <= TRAINED_MW_RANGE[1])
    allowed = not unsupported
    reasons = []
    if unsupported:
        reasons.append("unsupported_elements")
    if not mw_in_range:
        reasons.append("mw_out_of_range")
    return {
        "valid_smiles": True,
        "canonical_smiles": canonical,
        "elements": elements,
        "unsupported_elements": unsupported,
        "mw": mw,
        "allowed_elements": allowed,
        "mw_in_trained_range": mw_in_range,
        "in_domain": bool(allowed and mw_in_range),
        "applicability_reason": "|".join(reasons) if reasons else "ok",
        "dedup_key": canonical,
    }


def _model_asset_paths(model_key: str) -> list[tuple[str, Path]]:
    """Collect checkpoint paths that define a registry model, in stable order."""
    if model_key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model registry key: {model_key!r}")

    seen: set[str] = set()
    assets: list[tuple[str, Path]] = []

    def visit(key: str) -> None:
        if key in seen:
            return
        seen.add(key)
        spec = MODEL_REGISTRY[key]
        checkpoint = spec.get("checkpoint")
        if checkpoint is not None:
            assets.append((key, Path(checkpoint)))
        for child in spec.get("experts", ()):
            visit(str(child))
        for child in spec.get("components", ()):
            visit(str(child))
        for index, gate in enumerate(spec.get("gates", ())):
            assets.append((f"{key}.gate{index}", Path(gate)))

    visit(model_key)
    return assets


def model_manifest(model_key: str) -> dict[str, object]:
    """Describe model custody and hashes without loading the model."""
    assets = []
    for role, path in _model_asset_paths(model_key):
        exists = path.is_file()
        assets.append(
            {
                "role": role,
                "path": _relative_path(path),
                "exists": exists,
                "sha256": sha256_file(path) if exists else None,
            }
        )
    digest_input = json.dumps(assets, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    return {
        "key": model_key,
        "version": f"{model_key}:{fingerprint[:12]}",
        "registry_kind": MODEL_REGISTRY[model_key].get("kind"),
        "assets": assets,
        "all_assets_present": all(bool(item["exists"]) for item in assets),
    }


def _base_row(
    source_row: pd.Series,
    input_row: int,
    source_id: str,
    inspected: dict[str, object],
) -> dict[str, object]:
    """Build one ledger row before model prediction is joined."""
    row = source_row.to_dict()
    row.update(
        {
            "input_row": input_row,
            "source_id": source_id,
            "dedup_key": inspected["dedup_key"],
            "duplicate_identity": False,
            "validity": inspected["valid_smiles"],
            "valid_smiles": inspected["valid_smiles"],
            "canonical_smiles": inspected["canonical_smiles"],
            "elements": _json_list(inspected["elements"]),
            "unsupported_elements": _json_list(inspected["unsupported_elements"]),
            "mw": inspected["mw"],
            "allowed_elements": inspected["allowed_elements"],
            "mw_in_trained_range": inspected["mw_in_trained_range"],
            "graph_success": False,
            "in_domain": inspected["in_domain"],
            "applicability_reason": inspected["applicability_reason"],
            "prediction_status": "rejected",
            "rejection_reason": "invalid_smiles"
            if not inspected["valid_smiles"]
            else None,
            "expert_count": 0,
            **{target: None for target in TARGET_COLS},
            **{
                f"expert_disagreement_{target}_eV": None
                for target in TARGET_COLS
            },
        }
    )
    return row


def _ensure_output_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Put source columns first and append the stable ledger contract."""
    ordered = list(frame.columns)
    for column in _LEDGER_COLUMNS:
        if column not in ordered:
            ordered.append(column)
    return frame.loc[:, ordered]


def build_database_ledger(
    frame: pd.DataFrame,
    *,
    model_key: str = DEFAULT_MODEL_KEY,
    models: dict | None = None,
    batch_size: int = 256,
    device: object | None = None,
    smiles_column: str = "smiles",
    id_column: str | None = None,
    graph_builder: Callable[[str], object | None] | None = None,
    predictor: Callable[..., tuple] | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Validate and predict an input frame while retaining every source row.

    ``graph_builder`` and ``predictor`` are injectable so the row contract can
    be tested without model checkpoints.  Production callers should leave both
    unset and use the public repaired-2M inference API.
    """
    if smiles_column not in frame.columns:
        raise ValueError(f"Input is missing the required SMILES column: {smiles_column!r}")
    if not frame.columns.is_unique:
        raise ValueError("Input CSV has duplicate column names")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    resolved_id = _resolve_id_column(frame, id_column)
    if graph_builder is None:
        from .graphs import smiles_to_2d_pyg

        graph_builder = smiles_to_2d_pyg

    rows: list[dict[str, object]] = []
    predictable_source_rows: list[int] = []
    predictable_smiles: list[str] = []
    for input_row, (_, source_row) in enumerate(frame.iterrows()):
        fallback_id = f"input_row_{input_row:08d}"
        source_id = _normalise_source_id(
            source_row[resolved_id] if resolved_id is not None else None,
            fallback_id,
        )
        inspected = _inspect_smiles(source_row[smiles_column])
        row = _base_row(source_row, input_row, source_id, inspected)
        if inspected["valid_smiles"]:
            graph = graph_builder(str(inspected["canonical_smiles"]))
            row["graph_success"] = graph is not None
            if graph is None:
                row["rejection_reason"] = "graph_failed"
            else:
                predictable_source_rows.append(input_row)
                predictable_smiles.append(str(inspected["canonical_smiles"]))
        rows.append(row)

    output = pd.DataFrame(rows)
    if output.empty:
        output = pd.DataFrame(columns=[*frame.columns, *_LEDGER_COLUMNS])
    duplicate_counts = Counter(output["dedup_key"].tolist())
    output["duplicate_identity"] = output["dedup_key"].map(
        lambda key: duplicate_counts[key] > 1
    )

    if predictable_smiles:
        if predictor is None:
            from .inference import predict_smiles_batch_repaired_2m_2d

            predictor = predict_smiles_batch_repaired_2m_2d
        result = predictor(
            predictable_smiles,
            models=models,
            batch_size=batch_size,
            return_expert_predictions=True,
            device=device,
            key=model_key,
        )
        if len(result) == 3:
            valid_idx, predictions, expert_predictions = result
        elif len(result) == 2:
            valid_idx, predictions = result
            expert_predictions = None
        else:
            raise ValueError("Inference predictor must return two or three values")

        valid_idx = np.asarray(valid_idx, dtype=int)
        predictions = np.asarray(predictions, dtype=np.float32)
        if predictions.ndim != 2 or predictions.shape[1] != len(TARGET_COLS):
            raise ValueError(
                "Inference returned an unexpected prediction shape: "
                f"{predictions.shape}"
            )
        if len(valid_idx) != len(predictions):
            raise ValueError("Inference valid-index and prediction lengths differ")
        if len(valid_idx) and (
            valid_idx.min() < 0 or valid_idx.max() >= len(predictable_smiles)
        ):
            raise ValueError("Inference returned an index outside the candidate rows")

        expert_array = None
        if expert_predictions is not None:
            expert_array = np.asarray(expert_predictions, dtype=np.float32)
            if expert_array.ndim != 3 or expert_array.shape[0] != len(predictions):
                raise ValueError(
                    "Inference returned an unexpected expert prediction shape: "
                    f"{expert_array.shape}"
                )

        returned = set(valid_idx.tolist())
        for prediction_position, (local_index, prediction) in enumerate(
            zip(valid_idx.tolist(), predictions)
        ):
            source_row = predictable_source_rows[local_index]
            target_row = output.index[source_row]
            if not np.isfinite(prediction).all():
                output.at[target_row, "rejection_reason"] = "non_finite_prediction"
                continue
            output.at[target_row, "prediction_status"] = "predicted"
            output.at[target_row, "rejection_reason"] = None
            output.at[target_row, "expert_count"] = (
                int(expert_array.shape[1]) if expert_array is not None else 0
            )
            for target_index, target in enumerate(TARGET_COLS):
                output.at[target_row, target] = float(prediction[target_index])
            if expert_array is not None:
                disagreement = np.std(
                    expert_array[prediction_position], axis=0
                )
                if np.isfinite(disagreement).all():
                    for target_index, target in enumerate(TARGET_COLS):
                        output.at[
                            target_row,
                            f"expert_disagreement_{target}_eV",
                        ] = float(disagreement[target_index])

        for local_index, source_row in enumerate(predictable_source_rows):
            if local_index not in returned:
                output.at[source_row, "rejection_reason"] = "prediction_failed"

    # A prediction that failed its finite-value check must not be labelled as
    # predicted even if the public API returned its index.
    output.loc[
        output["prediction_status"] != "predicted", TARGET_COLS
    ] = None
    output = _ensure_output_schema(output)

    predicted = output[output["prediction_status"] == "predicted"]
    finite_predictions = (
        bool(np.isfinite(predicted[TARGET_COLS].to_numpy(dtype=float)).all())
        if len(predicted)
        else True
    )
    reason_counts = {
        str(key): int(value)
        for key, value in output["rejection_reason"].fillna("").value_counts().items()
        if key
    }
    stats = {
        "input_rows": int(len(output)),
        "predicted_rows": int(len(predicted)),
        "rejected_rows": int(len(output) - len(predicted)),
        "valid_smiles_rows": int(output["valid_smiles"].sum()),
        "graph_success_rows": int(output["graph_success"].sum()),
        "in_domain_rows": int(output["in_domain"].sum()),
        "out_of_domain_rows": int((~output["in_domain"]).sum()),
        "unique_dedup_keys": int(output["dedup_key"].nunique(dropna=False)),
        "duplicate_identity_rows": int(output["duplicate_identity"].sum()),
        "rejection_reason_counts": reason_counts,
        "all_predicted_values_finite": finite_predictions,
    }
    if not finite_predictions:
        raise ValueError("Ledger contains non-finite values in predicted rows")
    return output, stats


def _read_input(
    path: Path,
    *,
    max_rows: int | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {path}")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive")
    frame = pd.read_csv(path, keep_default_na=False)
    total_rows = len(frame)
    if max_rows is not None:
        frame = frame.iloc[:max_rows].copy()
    return frame, {
        "path": _relative_path(path),
        "sha256": sha256_file(path),
        "rows_total": int(total_rows),
        "rows_processed": int(len(frame)),
        "max_rows": max_rows,
    }


def run_database_build(
    input_csv: Path,
    out_dir: Path,
    *,
    model_key: str = DEFAULT_MODEL_KEY,
    smiles_column: str = "smiles",
    id_column: str | None = None,
    max_rows: int | None = None,
    batch_size: int = 256,
    device: object | None = None,
    models: dict | None = None,
    graph_builder: Callable[[str], object | None] | None = None,
    predictor: Callable[..., tuple] | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Run one atomic database build and return its machine-readable manifest."""
    model = model_manifest(model_key)
    if models is None and not model["all_assets_present"]:
        missing = [asset["path"] for asset in model["assets"] if not asset["exists"]]
        raise FileNotFoundError(
            "Cannot run production inference because model assets are missing: "
            + ", ".join(missing)
        )

    input_csv = Path(input_csv)
    out_dir = Path(out_dir)
    frame, source = _read_input(input_csv, max_rows=max_rows)
    ledger, stats = build_database_ledger(
        frame,
        model_key=model_key,
        models=models,
        batch_size=batch_size,
        device=device,
        smiles_column=smiles_column,
        id_column=id_column,
        graph_builder=graph_builder,
        predictor=predictor,
    )

    predictions_path = out_dir / "predictions.csv"
    manifest_path = out_dir / "manifest.json"
    if not overwrite:
        existing = [
            str(path)
            for path in (predictions_path, manifest_path)
            if path.exists()
        ]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing output(s): " + ", ".join(existing)
            )
    csv_text = ledger.to_csv(index=False, lineterminator="\n")
    _atomic_write_text(predictions_path, csv_text, overwrite=overwrite)

    resolved_id = _resolve_id_column(frame, id_column)
    manifest: dict[str, object] = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "built_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source": {
            **source,
            "smiles_column": smiles_column,
            "id_column": resolved_id,
            "fallback_id": "input_row_{row:08d}",
        },
        "model": model,
        "task": {
            "targets": list(TARGET_COLS),
            "units": "eV",
            "reference": "B3LYP/6-31G* gas-phase Kohn-Sham values",
            "model_key": model_key,
        },
        "applicability": {
            "allowed_elements": sorted(ALLOWED_ELEMENTS),
            "trained_mw_range": list(TRAINED_MW_RANGE),
            "out_of_domain_policy": "retain row and flag; do not silently filter",
        },
        "inference": {
            "batch_size": int(batch_size),
            "device": str(device) if device is not None else "model_default",
            "expert_disagreement": {
                "definition": "population standard deviation across direct expert predictions",
                "units": "eV",
                "calibrated": False,
            },
        },
        "row_contract": {
            "ledger_preserves_input_rows": True,
            "dedup_key": "canonical_smiles, or invalid::<raw_smiles> when parsing fails",
            "prediction_status_values": ["predicted", "rejected"],
            "rejection_reason_values": [
                "invalid_smiles",
                "graph_failed",
                "prediction_failed",
                "non_finite_prediction",
            ],
        },
        "row_counts": stats,
        "outputs": {
            "predictions_csv": {
                "path": _relative_path(predictions_path),
                "sha256": sha256_file(predictions_path),
                "rows": int(len(ledger)),
            }
        },
    }
    _atomic_write_text(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        overwrite=overwrite,
    )
    manifest["outputs"]["manifest_json"] = {
        "path": _relative_path(manifest_path),
        "sha256": sha256_file(manifest_path),
    }
    return manifest
