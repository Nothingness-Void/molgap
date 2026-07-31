"""Public-API smoke test for the repaired-2M pure-2D presets.

Exercises the three input classes required by the freeze package: ordinary
in-domain SMILES, invalid SMILES that must be dropped rather than imputed, and
out-of-domain molecules (elements outside CHONSFCl, or MW far outside the
training window) that must still return finite numbers while being flagged.

Applicability is reported, not enforced: the loader has no gate, so this record
is what tells a reader which rows are outside the trained domain.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

from molgap.constants import EVALUATE_DIR, MODEL_REGISTRY, TARGET_COLS
from molgap.inference import (
    load_repaired_2m_2d,
    predict_smiles_batch_repaired_2m_2d,
)
from molgap.pubchemqc import ALLOWED_ELEMENTS


TARGETS = tuple(TARGET_COLS)
PRESETS = ("repaired_2m_dense_2d", "repaired_2m_equal_2d")
DEFAULT_OUTPUT = (
    EVALUATE_DIR / "project_freeze" / "public_api_smoke_test" / "repaired_2m_smoke_test.json"
)
# The corpus is CHONSFCl at MW 200-1000; predictions outside that window are
# extrapolation, not a supported claim.
TRAINED_MW_RANGE = (200.0, 1000.0)

# In-domain: CHONSFCl only, MW inside the trained 200-1000 window. These are
# the rows a supported prediction may be claimed for.
VALID_SMILES = (
    "Clc1ccc(cc1)C(=O)Nc1ccccc1",
    "O=C(Nc1ccccc1)c1ccc(cc1)S(=O)(=O)N",
    "COc1ccc(cc1)C(=O)Nc1ccc(F)cc1",
    "c1ccc(cc1)c1ccc(cc1)c1ccccc1",
    "O=C1c2ccccc2C(=O)c2ccccc21",
    "c1ccc2c(c1)[nH]c1ccc3c(c21)cccc3",
)
INVALID_SMILES = (
    "not_a_smiles",
    "c1ccccc",
    "C(C)(C)(C)(C)C",
    "",
    "[Xx]",
)
# Two out-of-domain axes: unsupported elements and molecular weight far outside
# the trained window. Both must survive inference without crashing.
OOD_SMILES = (
    "CC[Si](C)(C)C",
    "c1ccc(cc1)[Se]c1ccccc1",
    "CC(=O)O[B-](F)(F)F",
    "C",
    "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC",
)


def atomic_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def applicability(smiles: str) -> dict[str, object]:
    """Report the domain signals a reader needs, without gating the prediction."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return {"parsed": False}
    elements = {atom.GetSymbol() for atom in molecule.GetAtoms()}
    weight = float(Descriptors.MolWt(molecule))
    unsupported = sorted(elements.difference(ALLOWED_ELEMENTS))
    return {
        "parsed": True,
        "mw": weight,
        "elements": sorted(elements),
        "unsupported_elements": unsupported,
        "mw_in_trained_range": bool(
            TRAINED_MW_RANGE[0] <= weight <= TRAINED_MW_RANGE[1]
        ),
        "in_domain": bool(
            not unsupported and TRAINED_MW_RANGE[0] <= weight <= TRAINED_MW_RANGE[1]
        ),
    }


def run_case(
    name: str,
    smiles: tuple[str, ...],
    models: dict,
    *,
    expect_all_valid: bool,
    expect_none_valid: bool,
    expect_all_in_domain: bool = False,
) -> dict[str, object]:
    valid_idx, predictions = predict_smiles_batch_repaired_2m_2d(
        list(smiles), models=models
    )
    if predictions.shape[1] != len(TARGETS):
        raise RuntimeError(f"{name}: prediction width is {predictions.shape[1]}")
    if len(valid_idx) != len(predictions):
        raise RuntimeError(f"{name}: index and prediction lengths differ")
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"{name}: non-finite prediction returned")
    if expect_all_valid and len(valid_idx) != len(smiles):
        dropped = [smiles[i] for i in range(len(smiles)) if i not in set(valid_idx.tolist())]
        raise RuntimeError(f"{name}: expected every row to be valid, dropped {dropped}")
    if expect_none_valid and len(valid_idx):
        kept = [smiles[i] for i in valid_idx.tolist()]
        raise RuntimeError(f"{name}: invalid SMILES were not dropped: {kept}")
    if len(valid_idx) and (valid_idx.min() < 0 or valid_idx.max() >= len(smiles)):
        raise RuntimeError(f"{name}: returned index outside the input range")

    rows = []
    for position, source_index in enumerate(valid_idx.tolist()):
        row: dict[str, object] = {
            "input_index": int(source_index),
            "smiles": smiles[source_index],
            "applicability": applicability(smiles[source_index]),
        }
        row.update(
            {
                target: float(predictions[position, index])
                for index, target in enumerate(TARGETS)
            }
        )
        rows.append(row)
    if expect_all_in_domain:
        outside = [
            row["smiles"]
            for row in rows
            if not row["applicability"]["in_domain"]
        ]
        if outside:
            raise RuntimeError(
                f"{name}: suite is meant to be in-domain but these rows are not: {outside}"
            )
    return {
        "input_rows": len(smiles),
        "valid_rows": int(len(valid_idx)),
        "dropped_rows": int(len(smiles) - len(valid_idx)),
        "dropped_inputs": [
            smiles[index]
            for index in range(len(smiles))
            if index not in set(valid_idx.tolist())
        ],
        "all_finite": True,
        "predictions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = None if args.device is None else torch.device(args.device)

    presets: dict[str, object] = {}
    for key in PRESETS:
        models = load_repaired_2m_2d(device, key=key)
        presets[key] = {
            "encoder_passes": int(MODEL_REGISTRY[key]["encoder_passes"]),
            "cases": {
                "valid": run_case(
                    f"{key}/valid",
                    VALID_SMILES,
                    models,
                    expect_all_valid=True,
                    expect_none_valid=False,
                    expect_all_in_domain=True,
                ),
                "invalid": run_case(
                    f"{key}/invalid",
                    INVALID_SMILES,
                    models,
                    expect_all_valid=False,
                    expect_none_valid=True,
                ),
                "ood": run_case(
                    f"{key}/ood",
                    OOD_SMILES,
                    models,
                    expect_all_valid=False,
                    expect_none_valid=False,
                ),
            },
        }

    result = {
        "schema_version": 1,
        "status": "complete",
        "check": "public_api_valid_invalid_ood_smoke_test",
        "trained_mw_range": list(TRAINED_MW_RANGE),
        "trained_elements": sorted(ALLOWED_ELEMENTS),
        "note": (
            "Out-of-domain rows still return finite predictions; the loader "
            "reports applicability rather than refusing them. Do not present an "
            "out-of-domain value as a calibrated prediction."
        ),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device or ("cuda" if torch.cuda.is_available() else "cpu")),
        },
        "presets": presets,
    }
    atomic_json(result, args.output)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
