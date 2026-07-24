"""Build one durable quarter of the PubChemQC 100K second-conformer cache."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


WORK = Path("/kaggle/working")
SHARD_ROWS = 5_000
SHARD_ID = 1
SHARD_COUNT = 4


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_input(name: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name}, found {matches}")
    return matches[0]


def build_one(row: tuple[int, str, float, float, float]):
    source_idx, smiles, homo, lumo, gap = row
    import numpy as np
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return source_idx, None
    molecule = Chem.AddHs(molecule)
    params = AllChem.ETKDGv3()
    params.randomSeed = int((43 * 1_000_003 + source_idx) % 2_147_483_647)
    if AllChem.EmbedMolecule(molecule, params) != 0:
        return source_idx, None
    try:
        AllChem.MMFFOptimizeMolecule(molecule, maxIters=200)
    except Exception:
        pass
    try:
        AllChem.ComputeGasteigerCharges(molecule)
        charges = [
            float(atom.GetProp("_GasteigerCharge"))
            if atom.HasProp("_GasteigerCharge")
            else 0.0
            for atom in molecule.GetAtoms()
        ]
        charges = [value if np.isfinite(value) else 0.0 for value in charges]
    except Exception:
        charges = [0.0] * molecule.GetNumAtoms()
    conformer = molecule.GetConformer()
    return source_idx, {
        "z": [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
        "pos": conformer.GetPositions().astype("float32"),
        "charges": charges,
        "y": [homo, lumo, gap],
    }


def main() -> None:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "rdkit==2025.3.5",
            "torch-geometric==2.6.1",
        ]
    )
    import numpy as np
    import pandas as pd
    import torch
    from torch_geometric.data import Data

    if SHARD_ID is None:
        raise RuntimeError("SHARD_ID must be embedded when packaging the kernel")
    shard_id = int(SHARD_ID)
    shard_count = SHARD_COUNT
    split_path = find_input("split.csv")
    split = pd.read_csv(split_path)
    positions = np.array_split(np.arange(len(split)), shard_count)[shard_id]
    selected = split.iloc[positions].copy().reset_index(drop=True)
    run_dir = WORK / f"second_conformer_r{shard_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    progress_path = run_dir / "progress.json"
    completed_parts = []
    failures: list[int] = []

    for start in range(0, len(selected), SHARD_ROWS):
        stop = min(start + SHARD_ROWS, len(selected))
        output = run_dir / f"graphs_{start:06d}_{stop:06d}.pt"
        report = output.with_suffix(".json")
        if output.exists() and report.exists():
            completed_parts.append(json.loads(report.read_text(encoding="utf-8")))
            continue
        work = [
            (
                int(row.source_idx),
                str(row.canonical_smiles),
                float(row.homo),
                float(row.lumo),
                float(row.gap),
            )
            for row in selected.iloc[start:stop].itertuples(index=False)
        ]
        graphs = []
        part_failures = []
        with ProcessPoolExecutor(max_workers=4) as pool:
            for source_idx, payload in pool.map(build_one, work, chunksize=50):
                if payload is None:
                    part_failures.append(int(source_idx))
                    continue
                graphs.append(
                    Data(
                        z=torch.tensor(payload["z"], dtype=torch.long),
                        pos=torch.tensor(payload["pos"], dtype=torch.float32),
                        charges=torch.tensor(payload["charges"], dtype=torch.float32),
                        y=torch.tensor(payload["y"], dtype=torch.float32).view(1, 3),
                        source_idx=torch.tensor([source_idx], dtype=torch.long),
                    )
                )
        temporary = output.with_suffix(".pt.tmp")
        torch.save(graphs, temporary)
        os.replace(temporary, output)
        part_report = {
            "start": start,
            "stop": stop,
            "requested": len(work),
            "succeeded": len(graphs),
            "failed": len(part_failures),
            "failure_source_idx": part_failures,
            "path": output.name,
            "sha256": sha256(output),
        }
        atomic_json(part_report, report)
        completed_parts.append(part_report)
        failures.extend(part_failures)
        atomic_json(
            {
                "status": "running",
                "shard_id": shard_id,
                "completed_rows": stop,
                "total_rows": len(selected),
                "succeeded": sum(part["succeeded"] for part in completed_parts),
                "failed": sum(part["failed"] for part in completed_parts),
            },
            progress_path,
        )
        print(
            f"r{shard_id} {stop}/{len(selected)} "
            f"success={sum(part['succeeded'] for part in completed_parts)}",
            flush=True,
        )

    source_idx = [
        int(value)
        for value in selected.source_idx.to_numpy()
        if int(value) not in set(failures)
    ]
    completion = {
        "status": "complete",
        "shard_id": shard_id,
        "shard_count": shard_count,
        "split_csv_sha256": sha256(split_path),
        "partition_start": int(positions[0]),
        "partition_stop": int(positions[-1]) + 1,
        "requested": int(len(selected)),
        "succeeded": len(source_idx),
        "failed": len(failures),
        "unique_source_idx": len(source_idx) == len(set(source_idx)),
        "finite_labels": bool(
            np.isfinite(selected[["homo", "lumo", "gap"]].to_numpy()).all()
        ),
        "parts": completed_parts,
    }
    atomic_json(completion, run_dir / "completion_manifest.json")
    atomic_json({"status": "complete", **completion}, progress_path)
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
