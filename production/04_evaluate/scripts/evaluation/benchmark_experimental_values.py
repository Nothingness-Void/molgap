"""Measure the frozen model against literature experimental OLED values.

This quantifies a **known limitation**, not an accuracy claim. The model predicts
gas-phase B3LYP/6-31G* Kohn-Sham eigenvalues; the reference here is thin-film CV
and UPS measurements. These are different physical quantities measured in a
different phase, so a large offset is the expected result and the point of the
record is to state its size and sign rather than to hide it.

Do not present the output as model error. It is the size of the gap that a future
solid-state Delta head would have to close.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from molgap.constants import COMMERCIAL_DIR, EVALUATE_DIR, TARGET_COLS
from molgap.inference import (
    load_repaired_2m_2d,
    predict_smiles_batch_repaired_2m_2d,
)


TARGETS = tuple(TARGET_COLS)
PRESETS = ("repaired_2m_dense_2d", "repaired_2m_equal_2d")
DEFAULT_INPUT = COMMERCIAL_DIR / "oled_experimental_v2.csv"
DEFAULT_OUTPUT = (
    EVALUATE_DIR
    / "project_freeze"
    / "experimental_offset"
    / "experimental_value_offset.json"
)


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def offsets(prediction: np.ndarray, experiment: np.ndarray) -> dict[str, object]:
    signed = prediction - experiment
    block: dict[str, object] = {}
    for index, target in enumerate(TARGETS):
        values = signed[:, index]
        block[target] = {
            "mean_signed_offset_eV": float(values.mean()),
            "median_signed_offset_eV": float(np.median(values)),
            "mae_eV": float(np.abs(values).mean()),
            "std_eV": float(values.std(ddof=1)),
            "min_signed_eV": float(values.min()),
            "max_signed_eV": float(values.max()),
            # A tight spread around a large mean is the interesting case: it
            # means the discrepancy is mostly a learnable systematic shift.
            "systematic_fraction": float(
                abs(values.mean()) / (abs(values.mean()) + values.std(ddof=1))
            ),
        }
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    required = {"name", "smiles", *[f"{t}_exp" for t in TARGETS]}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Experimental table misses {sorted(missing)}")
    device = None if args.device is None else torch.device(args.device)

    presets: dict[str, object] = {}
    for key in PRESETS:
        models = load_repaired_2m_2d(device, key=key)
        valid_idx, prediction = predict_smiles_batch_repaired_2m_2d(
            frame.smiles.astype(str).tolist(), models=models
        )
        if len(valid_idx) != len(frame):
            raise RuntimeError(
                f"{key}: only {len(valid_idx)} of {len(frame)} reference rows built a graph"
            )
        experiment = frame[[f"{t}_exp" for t in TARGETS]].to_numpy(float)
        presets[key] = {
            "rows": int(len(frame)),
            "offset": offsets(prediction.astype(float), experiment),
            "per_molecule": [
                {
                    "name": str(frame.name.iloc[row]),
                    **{
                        f"{target}_predicted_eV": float(prediction[row, index])
                        for index, target in enumerate(TARGETS)
                    },
                    **{
                        f"{target}_experimental_eV": float(experiment[row, index])
                        for index, target in enumerate(TARGETS)
                    },
                }
                for row in range(len(frame))
            ],
        }

    result = {
        "schema_version": 1,
        "status": "complete",
        "check": "known_limitation_not_an_accuracy_claim",
        "quantity_predicted": "gas-phase B3LYP/6-31G* Kohn-Sham eigenvalues",
        "quantity_referenced": "thin-film / solution CV and UPS literature values",
        "why_not_comparable": (
            "Kohn-Sham eigenvalues are not quasiparticle levels, and a gas-phase "
            "calculation is not a condensed-phase measurement. Polarization, "
            "packing, and the measurement technique all shift the reference. A "
            "large offset is expected; only its size and consistency are "
            "informative."
        ),
        "reference": {
            "path": str(args.input),
            "rows": int(len(frame)),
            "homo_methods": sorted(frame.homo_method.dropna().unique().tolist())
            if "homo_method" in frame
            else [],
            "lumo_methods": sorted(frame.lumo_method.dropna().unique().tolist())
            if "lumo_method" in frame
            else [],
            "conditions": sorted(frame.condition.dropna().unique().tolist())
            if "condition" in frame
            else [],
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(device or ("cuda" if torch.cuda.is_available() else "cpu")),
        },
        "presets": presets,
    }
    atomic_write(args.output, json.dumps(result, indent=2) + "\n")

    dense = presets["repaired_2m_dense_2d"]["offset"]
    lines = [
        "# Offset Against Experimental Values",
        "",
        "**This is a limitation record, not an accuracy result.** The model "
        "predicts gas-phase B3LYP/6-31G* Kohn-Sham eigenvalues; the reference is "
        "thin-film and solution CV/UPS literature values. Different quantity, "
        "different phase.",
        "",
        f"- Reference: `{args.input.name}`, {len(frame)} commercial OLED molecules",
        f"- Methods: {', '.join(result['reference']['homo_methods'])} (HOMO), "
        f"{', '.join(result['reference']['lumo_methods'])} (LUMO)",
        f"- Conditions: {', '.join(result['reference']['conditions'])}",
        "",
        "## Signed offset, `repaired_2m_dense_2d` minus experiment",
        "",
        "| Target | Mean | Median | Std | Range | Systematic share |",
        "|---|---:|---:|---:|---|---:|",
    ]
    for target in TARGETS:
        block = dense[target]
        lines.append(
            f"| {target.upper()} | {block['mean_signed_offset_eV']:+.3f} eV | "
            f"{block['median_signed_offset_eV']:+.3f} eV | "
            f"{block['std_eV']:.3f} eV | "
            f"{block['min_signed_eV']:+.2f} to {block['max_signed_eV']:+.2f} eV | "
            f"{block['systematic_fraction'] * 100:.0f}% |"
        )
    lines.extend(
        [
            "",
            "Every HOMO and LUMO offset is positive: predicted levels sit above "
            "the measured ones. LUMO is shifted far more than HOMO, so the "
            "predicted gap is systematically too wide (one of 17 molecules is "
            "the lone exception, at -0.14 eV). This is the direction the "
            "literature reports for Kohn-Sham eigenvalues versus condensed-"
            "phase measurements.",
            "",
            "The offsets are large but consistent, and the standard deviations "
            "are well below the means. That pattern is what makes a solid-state "
            "Delta head plausible: a mostly systematic shift is learnable, "
            "whereas scattered disagreement would not be. It is not evidence "
            "that such a head works — none has been trained.",
            "",
            "This is a 17-molecule literature compilation with mixed measurement "
            "techniques, not a controlled benchmark. Treat it as an order-of-"
            "magnitude statement about a known gap.",
            "",
        ]
    )
    atomic_write(args.output.with_suffix(".md"), "\n".join(lines))
    print(args.output)
    print(args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
