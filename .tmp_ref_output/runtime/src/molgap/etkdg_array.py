"""Framework-neutral ETKDG shard construction for CPU-only clusters."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

TARGETS = ("homo", "lumo", "gap")
START_METHOD = "spawn"
CHUNKSIZE = 16
HEARTBEAT = 1_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_etkdg_record(item):
    """Return one deterministic MMFF200-refined ETKDGv3 record."""
    source_idx, smiles, target, seed = item
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return source_idx, None
        molecule = Chem.AddHs(molecule)
        embedded = False
        for attempt in range(2):
            parameters = AllChem.ETKDGv3()
            parameters.randomSeed = int(
                (seed * 1_000_003 + source_idx * 97 + attempt)
                % 2_147_483_647
            )
            if AllChem.EmbedMolecule(molecule, parameters) == 0:
                embedded = True
                break
        if not embedded:
            return source_idx, None
        try:
            AllChem.MMFFOptimizeMolecule(molecule, maxIters=200)
        except Exception:
            pass
        AllChem.ComputeGasteigerCharges(molecule)
        charges = []
        for atom in molecule.GetAtoms():
            value = (
                float(atom.GetProp("_GasteigerCharge"))
                if atom.HasProp("_GasteigerCharge")
                else 0.0
            )
            charges.append(value if np.isfinite(value) and abs(value) < 1e6 else 0.0)
        return source_idx, {
            "z": np.asarray(
                [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
                dtype=np.int16,
            ),
            "pos": np.asarray(
                molecule.GetConformer().GetPositions(), dtype=np.float32
            ),
            "charges": np.asarray(charges, dtype=np.float32),
            "y": np.asarray(target, dtype=np.float32),
        }
    except Exception:
        return source_idx, None


def build_records_parallel(work: list[tuple], *, workers: int, label: str):
    if workers <= 1:
        iterator = map(build_etkdg_record, work)
        for completed, result in enumerate(iterator, start=1):
            if completed % HEARTBEAT == 0 or completed == len(work):
                print(f"{label} completed={completed}/{len(work)}", flush=True)
            yield result
        return
    context = mp.get_context(START_METHOD)
    with context.Pool(processes=workers) as pool:
        iterator = pool.imap_unordered(
            build_etkdg_record,
            work,
            chunksize=CHUNKSIZE,
        )
        for completed, result in enumerate(iterator, start=1):
            if completed % HEARTBEAT == 0 or completed == len(work):
                print(f"{label} completed={completed}/{len(work)}", flush=True)
            yield result


def build_secondary_raw_shard(
    *,
    manifest_path: Path,
    output_dir: Path,
    shard_index: int,
    workers: int = 14,
) -> dict:
    """Build one hash-bound secondary shard as portable numeric arrays."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "molgap-secondary-etkdg-array-v1":
        raise ValueError("unsupported secondary array manifest")
    matches = [
        item
        for item in manifest.get("shards", [])
        if int(item["shard_index"]) == int(shard_index)
    ]
    if len(matches) != 1:
        raise ValueError(f"shard index {shard_index} is absent or duplicated")
    contract = matches[0]
    input_path = manifest_path.parent / contract["input_path"]
    if sha256(input_path) != contract["input_sha256"]:
        raise ValueError(f"array input SHA256 mismatch: {input_path}")

    raw_path = output_dir / Path(contract["output_path"]).with_suffix(".npz")
    report_path = output_dir / "reports" / (
        Path(contract["output_path"]).stem + ".json"
    )
    if raw_path.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") == "complete"
            and report.get("format") == "molgap-secondary-etkdg-raw-v1"
            and report.get("seed") == manifest["seed"]
            and report.get("input_sha256") == contract["input_sha256"]
            and report.get("sha256") == sha256(raw_path)
        ):
            print(f"reuse completed raw shard: {raw_path.name}", flush=True)
            return report

    primary_failures = {
        int(value) for value in contract["primary_failure_source_idx"]
    }
    work = []
    actual_source_idx = []
    with gzip.open(input_path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            source_idx = int(row["source_idx"])
            actual_source_idx.append(source_idx)
            if source_idx in primary_failures:
                continue
            work.append(
                (
                    source_idx,
                    row["canonical_smiles"],
                    [float(row[target]) for target in TARGETS],
                    int(manifest["seed"]),
                )
            )
    expected_source_idx = list(
        range(int(contract["start"]), int(contract["stop"]))
    )
    if actual_source_idx != expected_source_idx:
        raise ValueError(f"array input identity mismatch: {input_path}")
    if len(work) != int(contract["primary_graphs"]):
        raise ValueError(
            "primary alignment mismatch: "
            f"requested={len(work)} expected={contract['primary_graphs']}"
        )

    records = []
    failures = []
    for source_idx, record in build_records_parallel(
        work,
        workers=max(1, workers),
        label=f"secondary raw shard {shard_index + 1}/100 ETKDG",
    ):
        if record is None:
            failures.append(int(source_idx))
        else:
            records.append((int(source_idx), record))
    records.sort(key=lambda item: item[0])
    atom_counts = np.asarray(
        [len(record["z"]) for _, record in records], dtype=np.int64
    )
    atom_ptr = np.empty(len(records) + 1, dtype=np.int64)
    atom_ptr[0] = 0
    np.cumsum(atom_counts, out=atom_ptr[1:])
    arrays = {
        "source_idx": np.asarray(
            [source_idx for source_idx, _ in records], dtype=np.int64
        ),
        "atom_ptr": atom_ptr,
        "z": np.concatenate([record["z"] for _, record in records]),
        "pos": np.concatenate([record["pos"] for _, record in records]),
        "charges": np.concatenate(
            [record["charges"] for _, record in records]
        ),
        "y": np.stack([record["y"] for _, record in records]),
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_suffix(raw_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, raw_path)
    report = {
        "status": "complete",
        "format": "molgap-secondary-etkdg-raw-v1",
        "view": "independent_secondary_etkdg",
        "shard_index": int(shard_index),
        "start": int(contract["start"]),
        "stop": int(contract["stop"]),
        "seed": int(manifest["seed"]),
        "requested": len(work),
        "built": len(records),
        "failed": len(failures),
        "failure_source_idx": sorted(failures),
        "graphs": len(records),
        "atoms": int(atom_ptr[-1]),
        "path": raw_path.name,
        "bytes": raw_path.stat().st_size,
        "sha256": sha256(raw_path),
        "input_sha256": contract["input_sha256"],
        "source_csv_sha256": manifest["source_csv_sha256"],
        "primary_acceptance_sha256": manifest["primary_acceptance_sha256"],
        "primary_shard_sha256": contract["primary_graph_sha256"],
        "primary_sidecar_sha256": contract["primary_sidecar_sha256"],
    }
    atomic_json(report, report_path)
    print(
        f"secondary raw shard {shard_index + 1}/100 "
        f"requested={len(work)} built={len(records)} failed={len(failures)}",
        flush=True,
    )
    return report
