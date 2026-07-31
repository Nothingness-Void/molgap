"""Compare DFT and ML prediction cost on the same ten commercial OLED molecules.

The DFT side is parsed from the retained Phase 5 Gaussian 16 outputs; nothing is
recomputed. The ML side is timed live on the same SMILES with the registered
repaired-2M presets. Reporting both on one identical molecule set is the point:
a speedup quoted across different molecules is not a speedup.

Three DFT cost scopes are reported separately because they answer different
questions:

- `opt_freq` is what was actually run for the Phase 5 validation (`opt freq`);
- `opt` is geometry optimization alone;
- `geometry_step` is one optimization step, the closest available proxy for a
  single-point SCF at this level of theory.

The model was trained on PubChemQC single-point B3LYP/6-31G* labels, so
`geometry_step` is the honest per-label comparison and `opt_freq` is the honest
"what a chemist actually runs" comparison. Do not quote one and label it the
other.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics as st
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from molgap.constants import COMMERCIAL_DIR, EVALUATE_DIR, PRODUCTION_HISTORY_DIR
from molgap.inference import (
    load_repaired_2m_2d,
    predict_smiles_batch_repaired_2m_2d,
)


PRESETS = ("repaired_2m_dense_2d", "repaired_2m_equal_2d")
DEFAULT_GAUSSIAN_DIR = (
    PRODUCTION_HISTORY_DIR / "phase5" / "gaussian_validation" / "gjf"
)
DEFAULT_MOLECULES = COMMERCIAL_DIR / "gaussian_validation_10.csv"
DEFAULT_REFERENCE = (
    PRODUCTION_HISTORY_DIR / "phase5" / "gaussian_validation" / "ml_vs_gaussian.csv"
)
DEFAULT_OUTPUT = (
    EVALUATE_DIR / "project_freeze" / "cost_comparison" / "dft_vs_ml_cost.json"
)

TIME_LINE = re.compile(
    r"(Job cpu time|Elapsed time):\s+(\d+) days\s+(\d+) hours\s+"
    r"(\d+) minutes\s+([\d.]+) seconds"
)
PROCESSORS = re.compile(r"total of\s+(\d+) processors")
STEP = re.compile(r"Step number\s+\d+")
ROUTE = re.compile(r"^\s#[pP]?\s*(.+)$", re.MULTILINE)


def _seconds(match: re.Match) -> float:
    return (
        int(match.group(2)) * 86400
        + int(match.group(3)) * 3600
        + int(match.group(4)) * 60
        + float(match.group(5))
    )


def parse_gaussian_log(path: Path) -> dict[str, object]:
    """Extract per-link timings from one Gaussian 16 log.

    `opt freq` emits one timing block per link, so the first block is the
    optimization and the second is the frequency job. A log with a different
    number of blocks is rejected rather than silently mis-attributed.
    """
    text = path.read_text(errors="ignore")
    wall, cpu = [], []
    for match in TIME_LINE.finditer(text):
        (cpu if match.group(1).startswith("Job cpu") else wall).append(_seconds(match))
    if len(wall) != 2 or len(cpu) != 2:
        raise ValueError(
            f"{path.name}: expected 2 opt+freq timing blocks, found "
            f"{len(wall)} wall / {len(cpu)} cpu"
        )
    if "Normal termination" not in text:
        raise ValueError(f"{path.name}: no normal termination")
    processors = PROCESSORS.findall(text)
    steps = len(STEP.findall(text))
    if steps < 1:
        raise ValueError(f"{path.name}: no optimization steps found")
    route = ROUTE.search(text)
    return {
        "molecule": path.stem,
        "route": route.group(1).strip() if route else None,
        "processors": int(processors[0]) if processors else None,
        "optimization_steps": steps,
        "opt_wall_s": wall[0],
        "freq_wall_s": wall[1],
        "opt_cpu_s": cpu[0],
        "freq_cpu_s": cpu[1],
        "opt_freq_wall_s": wall[0] + wall[1],
        "opt_freq_cpu_s": cpu[0] + cpu[1],
        "geometry_step_wall_s": wall[0] / steps,
    }


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": float(st.median(values)),
        "mean": float(st.mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def time_preset(
    key: str,
    smiles: list[str],
    *,
    repeats: int,
    scale_rows: int,
    device: torch.device | None,
) -> dict[str, object]:
    models = load_repaired_2m_2d(device, key=key)
    resolved = models["device"]

    def synchronize() -> None:
        if resolved.type == "cuda":
            torch.cuda.synchronize(resolved)

    def run(values: list[str], batch_size: int) -> float:
        samples = []
        for _ in range(repeats):
            synchronize()
            start = time.perf_counter()
            valid_idx, predictions = predict_smiles_batch_repaired_2m_2d(
                values, models=models, batch_size=batch_size
            )
            synchronize()
            samples.append(time.perf_counter() - start)
            if len(valid_idx) != len(values) or not np.isfinite(predictions).all():
                raise RuntimeError(f"{key}: incomplete or non-finite prediction")
        return float(st.median(samples))

    run(smiles, 256)  # warmup

    singles = {value: run([value], 1) for value in smiles}
    batch_s = run(smiles, 256)
    scaled = [smiles[index % len(smiles)] for index in range(scale_rows)]
    scaled_s = run(scaled, 256)
    return {
        "registry_key": key,
        "device": str(resolved),
        "single_molecule_s": summarize(list(singles.values())),
        "batch_of_set": {
            "rows": len(smiles),
            "total_s": batch_s,
            "s_per_molecule": batch_s / len(smiles),
        },
        "batch_scaled": {
            "rows": scale_rows,
            "total_s": scaled_s,
            "s_per_molecule": scaled_s / scale_rows,
        },
    }


def markdown(result: dict[str, object]) -> str:
    dft = result["dft"]
    lines = [
        "# DFT Versus ML Prediction Cost",
        "",
        "Same ten commercial OLED molecules on both sides. The DFT numbers are "
        "parsed from retained Gaussian 16 logs; nothing was recomputed.",
        "",
        f"- DFT: `{dft['route']}`, Gaussian 16, "
        f"{dft['processors_used']} shared-memory cores per job",
        f"- ML: `{result['ml'][0]['device']}`, warm model, "
        f"{result['repeats']} timed repeats",
        f"- Molecules: {dft['molecules']}",
        "",
        "## DFT cost per molecule",
        "",
        "| Scope | Median | Mean | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key, unit in (
        ("Full `opt freq`, wall clock", "opt_freq_wall_s", "min"),
        ("Geometry optimization, wall clock", "opt_wall_s", "min"),
        ("One geometry step, wall clock", "geometry_step_wall_s", "s"),
        ("Full `opt freq`, core-hours", "opt_freq_core_h", "core-h"),
    ):
        block = dft["summary"][key]
        divisor = 60.0 if unit == "min" else 1.0
        lines.append(
            f"| {label} | {block['median'] / divisor:.2f} {unit} | "
            f"{block['mean'] / divisor:.2f} {unit} | "
            f"{block['min'] / divisor:.2f} {unit} | "
            f"{block['max'] / divisor:.2f} {unit} |"
        )
    lines.extend(
        [
            "",
            "## ML cost per molecule",
            "",
            "| Preset | Passes | Single call | Batch of 10 | Batch of "
            f"{result['ml'][0]['batch_scaled']['rows']:,} |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for block in result["ml"]:
        lines.append(
            f"| `{block['registry_key']}` | {block['encoder_passes']} | "
            f"{block['single_molecule_s']['median'] * 1000:.1f} ms | "
            f"{block['batch_of_set']['s_per_molecule'] * 1000:.2f} ms/mol | "
            f"{block['batch_scaled']['s_per_molecule'] * 1000:.2f} ms/mol |"
        )
    lines.extend(["", "## Speedup", "", "| Preset | vs full `opt freq` | vs one geometry step |", "|---|---:|---:|"])
    for block in result["ml"]:
        speedup = block["speedup"]
        lines.append(
            f"| `{block['registry_key']}` (batched) | "
            f"{speedup['vs_opt_freq_wall_batched']:,.0f}x | "
            f"{speedup['vs_geometry_step_wall_batched']:,.0f}x |"
        )
    lines.extend(
        [
            "",
            "The model was trained on PubChemQC single-point B3LYP/6-31G* labels, "
            "so the geometry-step column is the honest per-label comparison and "
            "the `opt freq` column is the honest \"what a chemist actually runs\" "
            "comparison. Quote whichever you mean, and say which one it is.",
            "",
        ]
    )
    if result.get("accuracy"):
        lines.extend(
            [
                "## Accuracy on these same molecules",
                "",
                "Mean absolute error against the Gaussian reference, in eV:",
                "",
                "| Model | HOMO | LUMO | Gap | Average |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for name, block in result["accuracy"].items():
            lines.append(
                f"| {name} | {block['homo_mae_eV']:.3f} | "
                f"{block['lumo_mae_eV']:.3f} | {block['gap_mae_eV']:.3f} | "
                f"{block['average_mae_eV']:.3f} |"
            )
        lines.extend(
            [
                "",
                "This is a ten-molecule spot check against a different DFT "
                "protocol (`B3LYP/6-31G(d)` opt+freq geometries, not PubChemQC "
                "PM6 geometries), so it is an illustrative agreement check, not "
                "the accepted accuracy evidence. Accepted metrics are in "
                "`../track_a_final_decision.md`.",
                "",
                "Two of the ten contain elements outside the trained CHONSFCl "
                "set and are flagged in the JSON record.",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gaussian-dir", type=Path, default=DEFAULT_GAUSSIAN_DIR)
    parser.add_argument("--molecules", type=Path, default=DEFAULT_MOLECULES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--scale-rows", type=int, default=1000)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    logs = sorted(args.gaussian_dir.glob("*.out"))
    if not logs:
        raise FileNotFoundError(f"No Gaussian logs under {args.gaussian_dir}")
    parsed = [parse_gaussian_log(path) for path in logs]

    routes = {row["route"] for row in parsed}
    if len(routes) != 1:
        raise ValueError(f"Mixed DFT routes, not comparable as one number: {routes}")
    processors = sorted({row["processors"] for row in parsed})

    frame = pd.read_csv(args.molecules)
    smiles = frame.smiles.astype(str).tolist()
    device = None if args.device is None else torch.device(args.device)

    from molgap.constants import MODEL_REGISTRY

    ml = []
    predictions = {}
    for key in PRESETS:
        block = time_preset(
            key,
            smiles,
            repeats=args.repeats,
            scale_rows=args.scale_rows,
            device=device,
        )
        block["encoder_passes"] = int(MODEL_REGISTRY[key]["encoder_passes"])
        ml.append(block)
        models = load_repaired_2m_2d(device, key=key)
        _, values = predict_smiles_batch_repaired_2m_2d(smiles, models=models)
        predictions[key] = values

    summary = {
        key: summarize([row[key] for row in parsed])
        for key in (
            "opt_wall_s",
            "freq_wall_s",
            "opt_cpu_s",
            "freq_cpu_s",
            "opt_freq_wall_s",
            "opt_freq_cpu_s",
            "geometry_step_wall_s",
        )
    }
    summary["opt_freq_core_h"] = summarize(
        [row["opt_freq_cpu_s"] / 3600 for row in parsed]
    )

    for block in ml:
        batched = block["batch_scaled"]["s_per_molecule"]
        single = block["single_molecule_s"]["median"]
        block["speedup"] = {
            "vs_opt_freq_wall_batched": summary["opt_freq_wall_s"]["median"] / batched,
            "vs_opt_freq_wall_single_call": summary["opt_freq_wall_s"]["median"] / single,
            "vs_geometry_step_wall_batched": (
                summary["geometry_step_wall_s"]["median"] / batched
            ),
        }
        block["projected_one_million_gpu_hours"] = batched * 1_000_000 / 3600

    accuracy = None
    if args.reference.is_file():
        reference = pd.read_csv(args.reference)
        truth = reference[
            ["homo_gaussian", "lumo_gaussian", "gap_gaussian"]
        ].to_numpy(float)
        historical = reference[["homo_pred", "lumo_pred", "gap_pred"]].to_numpy(float)
        accuracy = {}
        for name, values in (
            ("Phase 5 SchNet (historical)", historical),
            ("repaired-2M equal", predictions["repaired_2m_equal_2d"]),
            ("repaired-2M dense", predictions["repaired_2m_dense_2d"]),
        ):
            errors = np.abs(values - truth)
            accuracy[name] = {
                "homo_mae_eV": float(errors[:, 0].mean()),
                "lumo_mae_eV": float(errors[:, 1].mean()),
                "gap_mae_eV": float(errors[:, 2].mean()),
                "average_mae_eV": float(errors.mean()),
            }

    from molgap.pubchemqc import ALLOWED_ELEMENTS
    from rdkit import Chem

    out_of_domain = []
    for name, value in zip(frame.name, smiles):
        molecule = Chem.MolFromSmiles(value)
        unsupported = sorted(
            {atom.GetSymbol() for atom in molecule.GetAtoms()}.difference(
                ALLOWED_ELEMENTS
            )
        )
        if unsupported:
            out_of_domain.append({"molecule": name, "unsupported_elements": unsupported})

    result = {
        "schema_version": 1,
        "status": "complete",
        "comparison": "same_ten_commercial_oled_molecules",
        "repeats": args.repeats,
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "dft": {
            "engine": "Gaussian 16",
            "route": routes.pop(),
            "recomputed": False,
            "source": str(args.gaussian_dir),
            "molecules": len(parsed),
            "processors_used": (
                str(processors[0])
                if len(processors) == 1
                else f"{processors[0]}-{processors[-1]}"
            ),
            "per_molecule": parsed,
            "summary": summary,
        },
        "ml": ml,
        "accuracy": accuracy,
        "out_of_domain_molecules": out_of_domain,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, payload in (
        (args.output, json.dumps(result, indent=2) + "\n"),
        (args.output.with_suffix(".md"), markdown(result)),
    ):
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    print(args.output)
    print(args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
