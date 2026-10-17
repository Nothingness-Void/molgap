"""Durable Colab workflow for repaired-2M ETKDG graphs and lightweight SchNet."""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .schnet import SchNetWrapper

TARGETS = ("homo", "lumo", "gap")
REPAIRED_2M_SHA256 = (
    "0c7d19de211016bebc2aa8b3030665e8c5f239baebbdf21c01598e0ddf3777c3"
)
TRAINING_APPROVAL_TOKEN = "P8_BOUNDED_RESIDUAL_PASSED"
LIGHT_SCHNET_CONFIG = {
    "hidden_channels": 176,
    "num_filters": 160,
    "num_interactions": 6,
    "num_gaussians": 50,
    "cutoff": 6.0,
    "dropout": 0.0,
}
ROUTE_B_SCHNET_CONFIG = {
    "hidden_channels": 176,
    "num_filters": 160,
    "num_interactions": 6,
    "num_gaussians": 50,
    "cutoff": 10.0,
    "dropout": 0.05,
}
GRAPH_BUILD_START_METHOD = "spawn"
GRAPH_BUILD_CHUNKSIZE = 16
GRAPH_BUILD_HEARTBEAT = 1_000
GRAPH_SHARD_PATTERN = re.compile(r"graphs_(\d{7})_(\d{7})\.pt$")


def atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_etkdg_record(item):
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


def _record_to_data(source_idx: int, record: dict) -> Data:
    return Data(
        z=torch.from_numpy(record["z"].astype(np.int64, copy=False)),
        pos=torch.from_numpy(record["pos"]),
        charges=torch.from_numpy(record["charges"]),
        y=torch.from_numpy(record["y"]).view(1, 3),
        source_idx=torch.tensor([source_idx], dtype=torch.long),
    )


def _build_etkdg(item):
    source_idx, record = _build_etkdg_record(item)
    return (
        source_idx,
        None if record is None else _record_to_data(source_idx, record),
    )


def _build_etkdg_parallel(
    work: list[tuple],
    *,
    workers: int,
    progress_label: str,
):
    """Build ETKDG graphs without inheriting the parent's multi-GB caches."""
    if not work:
        return
    if workers <= 1:
        iterator = map(_build_etkdg, work)
        for completed, result in enumerate(iterator, start=1):
            if completed % GRAPH_BUILD_HEARTBEAT == 0 or completed == len(work):
                print(
                    f"{progress_label} completed={completed}/{len(work)}",
                    flush=True,
                )
            yield result
        return

    context = mp.get_context(GRAPH_BUILD_START_METHOD)
    with context.Pool(processes=workers) as pool:
        iterator = pool.imap_unordered(
            _build_etkdg,
            work,
            chunksize=GRAPH_BUILD_CHUNKSIZE,
        )
        for completed, result in enumerate(iterator, start=1):
            if completed % GRAPH_BUILD_HEARTBEAT == 0 or completed == len(work):
                print(
                    f"{progress_label} completed={completed}/{len(work)}",
                    flush=True,
                )
            yield result


def _build_etkdg_record_parallel(
    work: list[tuple],
    *,
    workers: int,
    progress_label: str,
):
    if not work:
        return
    if workers <= 1:
        iterator = map(_build_etkdg_record, work)
        for completed, result in enumerate(iterator, start=1):
            if completed % GRAPH_BUILD_HEARTBEAT == 0 or completed == len(work):
                print(
                    f"{progress_label} completed={completed}/{len(work)}",
                    flush=True,
                )
            yield result
        return
    context = mp.get_context(GRAPH_BUILD_START_METHOD)
    with context.Pool(processes=workers) as pool:
        iterator = pool.imap_unordered(
            _build_etkdg_record,
            work,
            chunksize=GRAPH_BUILD_CHUNKSIZE,
        )
        for completed, result in enumerate(iterator, start=1):
            if completed % GRAPH_BUILD_HEARTBEAT == 0 or completed == len(work):
                print(
                    f"{progress_label} completed={completed}/{len(work)}",
                    flush=True,
                )
            yield result


def _load_tables(repaired_csv: Path, original_csv: Path):
    columns = ["cid", "canonical_smiles", *TARGETS]
    repaired = pd.read_csv(repaired_csv, usecols=columns)
    original = pd.read_csv(original_csv, usecols=columns)
    if len(repaired) != 2_000_000 or len(original) != 1_000_000:
        raise ValueError(
            f"unexpected CSV rows: repaired={len(repaired)} original={len(original)}"
        )
    if repaired["cid"].nunique() != len(repaired):
        raise ValueError("repaired-2M CID values are not unique")
    if original["cid"].nunique() != len(original):
        raise ValueError("original-1M CID values are not unique")
    return repaired, original


def build_graph_shards(
    *,
    repaired_csv: Path,
    original_csv: Path,
    original_graph_cache: Path,
    output_dir: Path,
    shard_size: int = 20_000,
    workers: int = 8,
    seed: int = 42,
    verify_repaired_sha256: bool = True,
) -> dict:
    """Reuse aligned original-1M graphs and build only new repaired-2M rows."""
    if verify_repaired_sha256:
        actual = sha256(repaired_csv)
        if actual != REPAIRED_2M_SHA256:
            raise ValueError(f"repaired-2M SHA256 mismatch: {actual}")
    repaired, original = _load_tables(repaired_csv, original_csv)
    old_position = {
        int(cid): index for index, cid in enumerate(original["cid"].to_numpy())
    }

    print(f"loading reusable original graph cache: {original_graph_cache}", flush=True)
    original_graphs = torch.load(
        original_graph_cache, map_location="cpu", weights_only=False
    )
    reusable_graphs = {}
    for graph in original_graphs:
        index = int(graph.source_idx.view(-1)[0])
        if index in reusable_graphs:
            raise ValueError(f"duplicate original graph source_idx {index}")
        reusable_graphs[index] = graph
    del original_graphs

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    print(
        "graph worker protocol "
        f"start_method={GRAPH_BUILD_START_METHOD} "
        f"workers={max(1, workers)} chunksize={GRAPH_BUILD_CHUNKSIZE}",
        flush=True,
    )
    processed_reports = []
    total_shards = (len(repaired) + shard_size - 1) // shard_size
    for shard_index, start in enumerate(range(0, len(repaired), shard_size)):
        stop = min(start + shard_size, len(repaired))
        graph_path = output_dir / f"graphs_{start:07d}_{stop:07d}.pt"
        report_path = reports_dir / f"graphs_{start:07d}_{stop:07d}.json"
        if graph_path.exists() and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                report.get("status") == "complete"
                and report.get("start") == start
                and report.get("stop") == stop
                and report.get("sha256") == sha256(graph_path)
            ):
                processed_reports.append(report)
                print(
                    f"reuse completed shard {shard_index + 1}/{total_shards}: "
                    f"{graph_path.name}",
                    flush=True,
                )
                continue

        rows = repaired.iloc[start:stop]
        graphs = []
        work = []
        reused = 0
        for offset, row in enumerate(rows.itertuples(index=False)):
            source_idx = start + offset
            old_index = old_position.get(int(row.cid))
            old_graph = reusable_graphs.get(old_index) if old_index is not None else None
            if old_graph is not None:
                if str(original.iloc[old_index].canonical_smiles) != str(
                    row.canonical_smiles
                ):
                    raise ValueError(f"CID/SMILES identity mismatch at {source_idx}")
                graph = old_graph.clone()
                graph.source_idx = torch.tensor([source_idx], dtype=torch.long)
                graph.y = torch.tensor(
                    [row.homo, row.lumo, row.gap], dtype=torch.float32
                ).view(1, 3)
                graphs.append(graph)
                reused += 1
            else:
                work.append(
                    (
                        source_idx,
                        str(row.canonical_smiles),
                        [float(row.homo), float(row.lumo), float(row.gap)],
                        seed,
                    )
                )

        built = 0
        failures = []
        if work:
            progress_label = (
                f"shard {shard_index + 1}/{total_shards} ETKDG"
            )
            for source_idx, graph in _build_etkdg_parallel(
                work,
                workers=max(1, workers),
                progress_label=progress_label,
            ):
                if graph is None:
                    failures.append(int(source_idx))
                else:
                    graphs.append(graph)
                    built += 1
        graphs.sort(key=lambda graph: int(graph.source_idx.view(-1)[0]))
        atomic_torch_save(graphs, graph_path)
        report = {
            "status": "complete",
            "start": start,
            "stop": stop,
            "requested": stop - start,
            "reused": reused,
            "built": built,
            "failed": len(failures),
            "failure_source_idx": failures,
            "graphs": len(graphs),
            "path": graph_path.name,
            "bytes": graph_path.stat().st_size,
            "sha256": sha256(graph_path),
        }
        atomic_json(report, report_path)
        processed_reports.append(report)
        atomic_json(
            {
                "status": "running",
                "completed_shards": shard_index + 1,
                "total_shards": total_shards,
                "processed_rows": stop,
                "requested": sum(item["requested"] for item in processed_reports),
                "reused": sum(item["reused"] for item in processed_reports),
                "built": sum(item["built"] for item in processed_reports),
                "failed": sum(item["failed"] for item in processed_reports),
            },
            output_dir / "progress.json",
        )
        print(
            f"shard {shard_index + 1}/{total_shards} rows={start}:{stop} "
            f"reused={reused} built={built} failed={len(failures)}",
            flush=True,
        )

    completion = {
        "status": "complete",
        "repaired_csv": str(repaired_csv),
        "repaired_csv_sha256": sha256(repaired_csv),
        "original_csv": str(original_csv),
        "original_graph_cache": str(original_graph_cache),
        "shard_size": shard_size,
        "shards": len(processed_reports),
        "requested": sum(item["requested"] for item in processed_reports),
        "reused": sum(item["reused"] for item in processed_reports),
        "built": sum(item["built"] for item in processed_reports),
        "failed": sum(item["failed"] for item in processed_reports),
        "graphs": sum(item["graphs"] for item in processed_reports),
    }
    atomic_json(completion, output_dir / "build_completion.json")
    return completion


def build_secondary_graph_shards(
    *,
    repaired_csv: Path,
    primary_graph_dir: Path,
    output_dir: Path,
    workers: int = 8,
    seed: int = 314_159,
    verify_repaired_sha256: bool = True,
) -> dict:
    """Build an independent ETKDG view for every accepted primary graph."""
    if verify_repaired_sha256:
        actual = sha256(repaired_csv)
        if actual != REPAIRED_2M_SHA256:
            raise ValueError(f"repaired-2M SHA256 mismatch: {actual}")
    repaired = pd.read_csv(
        repaired_csv, usecols=["canonical_smiles", *TARGETS]
    )
    if len(repaired) != 2_000_000:
        raise ValueError(f"unexpected repaired rows: {len(repaired)}")
    primary_paths = sorted(primary_graph_dir.glob("graphs_*.pt"))
    if len(primary_paths) != 100:
        raise ValueError(
            f"expected 100 accepted primary shards, found {len(primary_paths)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    print(
        "secondary graph worker protocol "
        f"start_method={GRAPH_BUILD_START_METHOD} "
        f"workers={max(1, workers)} chunksize={GRAPH_BUILD_CHUNKSIZE}",
        flush=True,
    )
    reports = []
    for shard_index, primary_path in enumerate(primary_paths):
        graph_path = output_dir / primary_path.name
        report_path = reports_dir / f"{primary_path.stem}.json"
        if graph_path.exists() and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                report.get("status") == "complete"
                and report.get("seed") == seed
                and report.get("sha256") == sha256(graph_path)
            ):
                reports.append(report)
                print(
                    f"reuse completed secondary shard "
                    f"{shard_index + 1}/100: {graph_path.name}",
                    flush=True,
                )
                continue
        primary_graphs = torch.load(
            primary_path, map_location="cpu", weights_only=False
        )
        work = []
        for graph in primary_graphs:
            source_idx = int(graph.source_idx.view(-1)[0])
            row = repaired.iloc[source_idx]
            work.append(
                (
                    source_idx,
                    str(row.canonical_smiles),
                    [float(row.homo), float(row.lumo), float(row.gap)],
                    seed,
                )
            )
        secondary = []
        failures = []
        progress_label = f"secondary shard {shard_index + 1}/100 ETKDG"
        for source_idx, graph in _build_etkdg_parallel(
            work,
            workers=max(1, workers),
            progress_label=progress_label,
        ):
            if graph is None:
                failures.append(int(source_idx))
            else:
                secondary.append(graph)
        secondary.sort(key=lambda graph: int(graph.source_idx.view(-1)[0]))
        atomic_torch_save(secondary, graph_path)
        report = {
            "status": "complete",
            "view": "independent_secondary_etkdg",
            "seed": seed,
            "requested": len(primary_graphs),
            "reused": 0,
            "built": len(secondary),
            "failed": len(failures),
            "failure_source_idx": failures,
            "graphs": len(secondary),
            "path": graph_path.name,
            "bytes": graph_path.stat().st_size,
            "sha256": sha256(graph_path),
            "primary_shard_sha256": sha256(primary_path),
        }
        atomic_json(report, report_path)
        reports.append(report)
        atomic_json(
            {
                "status": "running",
                "view": "independent_secondary_etkdg",
                "seed": seed,
                "completed_shards": shard_index + 1,
                "total_shards": 100,
                "requested": sum(item["requested"] for item in reports),
                "built": sum(item["built"] for item in reports),
                "failed": sum(item["failed"] for item in reports),
            },
            output_dir / "progress.json",
        )
        print(
            f"secondary shard {shard_index + 1}/100 "
            f"requested={len(primary_graphs)} built={len(secondary)} "
            f"failed={len(failures)}",
            flush=True,
        )
    completion = {
        "status": "complete",
        "view": "independent_secondary_etkdg",
        "seed": seed,
        "shards": len(reports),
        "requested": sum(item["requested"] for item in reports),
        "reused": 0,
        "built": sum(item["built"] for item in reports),
        "graphs": sum(item["built"] for item in reports),
        "failed": sum(item["failed"] for item in reports),
        "source_csv_sha256": sha256(repaired_csv),
    }
    atomic_json(completion, output_dir / "build_completion.json")
    return completion


def prepare_secondary_array_inputs(
    *,
    repaired_csv: Path,
    primary_acceptance: Path,
    primary_metadata_root: Path,
    output_dir: Path,
    start_shard: int = 40,
    stop_shard: int = 100,
    seed: int = 314_159,
    expected_shards: int = 100,
    expected_source_rows: int = 2_000_000,
) -> dict:
    """Materialize hash-bound CSV inputs for independent secondary array jobs."""
    source_sha = sha256(repaired_csv)
    if source_sha != REPAIRED_2M_SHA256:
        raise ValueError(f"repaired-2M SHA256 mismatch: {source_sha}")
    acceptance = json.loads(primary_acceptance.read_text(encoding="utf-8"))
    if not acceptance.get("accepted") or not acceptance.get("immutable"):
        raise ValueError("primary graph acceptance is not accepted and immutable")
    shards = acceptance.get("shards", [])
    if (
        acceptance.get("expected_shards") != expected_shards
        or len(shards) != expected_shards
    ):
        raise ValueError(
            f"expected {expected_shards} accepted primary shards, "
            f"found {len(shards)}"
        )
    if acceptance.get("source_csv_sha256") != source_sha:
        raise ValueError("primary acceptance source CSV SHA256 mismatch")
    if not (0 <= start_shard < stop_shard <= len(shards)):
        raise ValueError(
            f"invalid shard interval: start={start_shard} stop={stop_shard}"
        )

    table = pd.read_csv(repaired_csv, usecols=["canonical_smiles", *TARGETS])
    if len(table) != expected_source_rows:
        raise ValueError(f"unexpected repaired rows: {len(table)}")
    table.insert(0, "source_idx", np.arange(len(table), dtype=np.int64))

    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    manifest_shards = []
    for shard_index in range(start_shard, stop_shard):
        accepted = shards[shard_index]
        match = GRAPH_SHARD_PATTERN.fullmatch(str(accepted["path"]))
        if match is None:
            raise ValueError(f"invalid primary shard path: {accepted['path']}")
        start, stop = (int(value) for value in match.groups())
        sidecar_path = primary_metadata_root / accepted["sidecar_path"]
        if not sidecar_path.is_file():
            raise FileNotFoundError(sidecar_path)
        sidecar_sha = sha256(sidecar_path)
        if sidecar_sha != accepted["sidecar_sha256"]:
            raise ValueError(f"primary sidecar SHA256 mismatch: {sidecar_path}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if (
            sidecar.get("status") != "complete"
            or sidecar.get("start") != start
            or sidecar.get("stop") != stop
            or sidecar.get("sha256") != accepted["sha256"]
            or sidecar.get("graphs") != accepted["rows"]
        ):
            raise ValueError(f"primary sidecar contract mismatch: {sidecar_path}")
        failures = sorted(int(value) for value in sidecar["failure_source_idx"])
        if len(failures) != int(sidecar["failed"]):
            raise ValueError(f"primary failure count mismatch: {sidecar_path}")

        input_path = inputs_dir / f"rows_{start:07d}_{stop:07d}.csv.gz"
        temporary = input_path.with_suffix(input_path.suffix + ".tmp")
        table.iloc[start:stop].to_csv(
            temporary,
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        os.replace(temporary, input_path)
        manifest_shards.append(
            {
                "shard_index": shard_index,
                "start": start,
                "stop": stop,
                "input_path": input_path.relative_to(output_dir).as_posix(),
                "input_rows": stop - start,
                "input_bytes": input_path.stat().st_size,
                "input_sha256": sha256(input_path),
                "output_path": accepted["path"],
                "primary_graph_sha256": accepted["sha256"],
                "primary_sidecar_path": accepted["sidecar_path"],
                "primary_sidecar_sha256": sidecar_sha,
                "primary_graphs": int(accepted["rows"]),
                "primary_failure_source_idx": failures,
            }
        )

    manifest = {
        "format": "molgap-secondary-etkdg-array-v1",
        "status": "prepared",
        "seed": int(seed),
        "source_rows": len(table),
        "source_csv_sha256": source_sha,
        "primary_acceptance_sha256": sha256(primary_acceptance),
        "start_shard": start_shard,
        "stop_shard": stop_shard,
        "shards": manifest_shards,
    }
    atomic_json(manifest, output_dir / "manifest.json")
    return manifest


def build_secondary_graph_shard_from_array_manifest(
    *,
    manifest_path: Path,
    output_dir: Path,
    shard_index: int,
    workers: int = 8,
) -> dict:
    """Build one secondary shard without loading the primary graph cache."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "molgap-secondary-etkdg-array-v1":
        raise ValueError("unsupported secondary array manifest")
    matches = [
        item for item in manifest.get("shards", [])
        if int(item["shard_index"]) == int(shard_index)
    ]
    if len(matches) != 1:
        raise ValueError(f"shard index {shard_index} is absent or duplicated")
    contract = matches[0]
    input_path = manifest_path.parent / contract["input_path"]
    if sha256(input_path) != contract["input_sha256"]:
        raise ValueError(f"array input SHA256 mismatch: {input_path}")

    graph_path = output_dir / contract["output_path"]
    report_path = output_dir / "reports" / (
        Path(contract["output_path"]).stem + ".json"
    )
    if graph_path.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("status") == "complete"
            and report.get("seed") == manifest["seed"]
            and report.get("input_sha256") == contract["input_sha256"]
            and report.get("primary_shard_sha256")
            == contract["primary_graph_sha256"]
            and report.get("sha256") == sha256(graph_path)
        ):
            print(f"reuse completed array shard: {graph_path.name}", flush=True)
            return report

    table = pd.read_csv(input_path)
    expected_source_idx = np.arange(
        int(contract["start"]), int(contract["stop"]), dtype=np.int64
    )
    actual_source_idx = table["source_idx"].to_numpy(dtype=np.int64)
    if (
        len(table) != int(contract["input_rows"])
        or not np.array_equal(actual_source_idx, expected_source_idx)
    ):
        raise ValueError(f"array input identity mismatch: {input_path}")
    primary_failures = set(
        int(value) for value in contract["primary_failure_source_idx"]
    )
    work = []
    for row in table.itertuples(index=False):
        source_idx = int(row.source_idx)
        if source_idx in primary_failures:
            continue
        work.append(
            (
                source_idx,
                str(row.canonical_smiles),
                [float(row.homo), float(row.lumo), float(row.gap)],
                int(manifest["seed"]),
            )
        )
    if len(work) != int(contract["primary_graphs"]):
        raise ValueError(
            "primary alignment mismatch: "
            f"requested={len(work)} expected={contract['primary_graphs']}"
        )

    secondary = []
    failures = []
    progress_label = f"secondary array shard {shard_index + 1}/100 ETKDG"
    for source_idx, graph in _build_etkdg_parallel(
        work,
        workers=max(1, workers),
        progress_label=progress_label,
    ):
        if graph is None:
            failures.append(int(source_idx))
        else:
            secondary.append(graph)
    secondary.sort(key=lambda graph: int(graph.source_idx.view(-1)[0]))
    atomic_torch_save(secondary, graph_path)
    report = {
        "status": "complete",
        "view": "independent_secondary_etkdg",
        "array_manifest_format": manifest["format"],
        "shard_index": int(shard_index),
        "start": int(contract["start"]),
        "stop": int(contract["stop"]),
        "seed": int(manifest["seed"]),
        "requested": len(work),
        "reused": 0,
        "built": len(secondary),
        "failed": len(failures),
        "failure_source_idx": sorted(failures),
        "graphs": len(secondary),
        "path": graph_path.name,
        "bytes": graph_path.stat().st_size,
        "sha256": sha256(graph_path),
        "input_sha256": contract["input_sha256"],
        "source_csv_sha256": manifest["source_csv_sha256"],
        "primary_acceptance_sha256": manifest["primary_acceptance_sha256"],
        "primary_shard_sha256": contract["primary_graph_sha256"],
        "primary_sidecar_sha256": contract["primary_sidecar_sha256"],
    }
    atomic_json(report, report_path)
    print(
        f"secondary array shard {shard_index + 1}/100 "
        f"requested={len(work)} built={len(secondary)} "
        f"failed={len(failures)}",
        flush=True,
    )
    return report


def build_secondary_raw_shard_from_array_manifest(
    *,
    manifest_path: Path,
    output_dir: Path,
    shard_index: int,
    workers: int = 8,
) -> dict:
    """Build one framework-neutral secondary shard for later PyG conversion."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "molgap-secondary-etkdg-array-v1":
        raise ValueError("unsupported secondary array manifest")
    matches = [
        item for item in manifest.get("shards", [])
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
            print(f"reuse completed raw array shard: {raw_path.name}", flush=True)
            return report

    primary_failures = set(
        int(value) for value in contract["primary_failure_source_idx"]
    )
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
    for source_idx, record in _build_etkdg_record_parallel(
        work,
        workers=max(1, workers),
        progress_label=f"secondary raw shard {shard_index + 1}/100 ETKDG",
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
        "z": np.concatenate(
            [record["z"] for _, record in records], dtype=np.int16
        ),
        "pos": np.concatenate(
            [record["pos"] for _, record in records], dtype=np.float32
        ),
        "charges": np.concatenate(
            [record["charges"] for _, record in records], dtype=np.float32
        ),
        "y": np.stack(
            [record["y"] for _, record in records]
        ).astype(np.float32, copy=False),
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
    return report


def convert_secondary_raw_shard_to_pyg(
    *,
    raw_path: Path,
    raw_report_path: Path,
    output_dir: Path,
) -> dict:
    """Convert an accepted framework-neutral shard to the standard PyG form."""
    report = json.loads(raw_report_path.read_text(encoding="utf-8"))
    if (
        report.get("status") != "complete"
        or report.get("format") != "molgap-secondary-etkdg-raw-v1"
        or report.get("sha256") != sha256(raw_path)
    ):
        raise ValueError("raw secondary shard report or SHA256 is invalid")
    with np.load(raw_path, allow_pickle=False) as arrays:
        source_idx = arrays["source_idx"]
        atom_ptr = arrays["atom_ptr"]
        z = arrays["z"]
        pos = arrays["pos"]
        charges = arrays["charges"]
        y = arrays["y"]
    if (
        len(source_idx) != int(report["graphs"])
        or len(atom_ptr) != len(source_idx) + 1
        or int(atom_ptr[-1]) != len(z)
        or len(z) != len(pos)
        or len(z) != len(charges)
        or y.shape != (len(source_idx), 3)
        or np.any(np.diff(source_idx) <= 0)
    ):
        raise ValueError("raw secondary shard array contract is invalid")
    graphs = []
    for index, value in enumerate(source_idx):
        start, stop = int(atom_ptr[index]), int(atom_ptr[index + 1])
        graphs.append(
            Data(
                z=torch.from_numpy(z[start:stop].astype(np.int64, copy=False)),
                pos=torch.from_numpy(pos[start:stop]),
                charges=torch.from_numpy(charges[start:stop]),
                y=torch.from_numpy(y[index]).view(1, 3),
                source_idx=torch.tensor([int(value)], dtype=torch.long),
            )
        )
    graph_path = output_dir / Path(report["path"]).with_suffix(".pt")
    atomic_torch_save(graphs, graph_path)
    pyg_report = {
        key: value for key, value in report.items()
        if key not in {"format", "atoms", "path", "bytes", "sha256"}
    }
    pyg_report.update(
        {
            "path": graph_path.name,
            "bytes": graph_path.stat().st_size,
            "sha256": sha256(graph_path),
            "raw_path": raw_path.name,
            "raw_sha256": report["sha256"],
        }
    )
    atomic_json(
        pyg_report,
        output_dir / "reports" / f"{graph_path.stem}.json",
    )
    return pyg_report


def validate_graph_shards(output_dir: Path) -> dict:
    graph_paths = sorted(output_dir.glob("graphs_*.pt"))
    if len(graph_paths) != 100:
        raise ValueError(f"expected 100 graph shards, found {len(graph_paths)}")
    seen = set()
    graphs = 0
    for index, path in enumerate(graph_paths, start=1):
        report_path = output_dir / "reports" / f"{path.stem}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if sha256(path) != report["sha256"]:
            raise ValueError(f"SHA256 mismatch: {path}")
        part = torch.load(path, map_location="cpu", weights_only=False)
        if len(part) != report["graphs"]:
            raise ValueError(f"graph count mismatch: {path}")
        for graph in part:
            source_idx = int(graph.source_idx.view(-1)[0])
            if source_idx in seen:
                raise ValueError(f"duplicate source_idx {source_idx}")
            seen.add(source_idx)
            if not torch.isfinite(graph.pos).all() or not torch.isfinite(graph.y).all():
                raise ValueError(f"non-finite graph {source_idx}")
            target = graph.y.view(-1)
            if abs(float(target[2] - (target[1] - target[0]))) > 1e-5:
                raise ValueError(f"Gap identity failed for {source_idx}")
        graphs += len(part)
        print(f"validated shard {index}/{len(graph_paths)}", flush=True)
    completion = json.loads(
        (output_dir / "build_completion.json").read_text(encoding="utf-8")
    )
    if graphs != completion["graphs"] or graphs != len(seen):
        raise ValueError("global graph accounting failed")
    result = {
        "status": "accepted",
        "shards": len(graph_paths),
        "graphs": graphs,
        "failed": completion["failed"],
        "unique_source_idx": True,
        "finite_coordinates_and_labels": True,
        "gap_identity": True,
    }
    atomic_json(result, output_dir / "validation.json")
    return result


def _forward(model, batch):
    charges = batch.charges if hasattr(batch, "charges") else None
    return model(batch.z, batch.pos, batch.batch, charges=charges)


def _load_all_shards(graph_dir: Path) -> list:
    graphs = []
    for index, path in enumerate(sorted(graph_dir.glob("graphs_*.pt")), start=1):
        graphs.extend(torch.load(path, map_location="cpu", weights_only=False))
        print(f"loaded training shard {index}/100: total={len(graphs):,}", flush=True)
    graphs.sort(key=lambda graph: int(graph.source_idx.view(-1)[0]))
    return graphs


def split_graphs_by_source_idx(
    graphs: list,
    *,
    source_rows: int,
    seed: int,
) -> tuple[list, list, list]:
    """Reproduce the 2D split before filtering failed ETKDG rows."""
    if source_rows <= 0:
        raise ValueError("source_rows must be positive")
    permutation = np.random.RandomState(seed).permutation(source_rows)
    n_train = int(0.8 * source_rows)
    n_validation = int(0.1 * source_rows)
    roles = np.full(source_rows, 2, dtype=np.int8)
    roles[permutation[:n_train]] = 0
    roles[permutation[n_train : n_train + n_validation]] = 1
    split = ([], [], [])
    seen = set()
    for graph in graphs:
        source_idx = int(graph.source_idx.view(-1)[0])
        if source_idx < 0 or source_idx >= source_rows:
            raise ValueError(f"source_idx {source_idx} outside source row contract")
        if source_idx in seen:
            raise ValueError(f"duplicate source_idx {source_idx}")
        seen.add(source_idx)
        split[int(roles[source_idx])].append(graph)
    if any(not part for part in split):
        raise RuntimeError("one or more source-aligned split roles are empty")
    return split


def train_light_schnet(
    *,
    graph_dir: Path,
    checkpoint_dir: Path,
    result_dir: Path,
    approval_token: str,
    epochs: int = 20,
    batch_size: int = 128,
    num_workers: int = 2,
    seed: int = 42,
    source_rows: int = 2_000_000,
    model_config: dict | None = None,
) -> dict:
    if approval_token != TRAINING_APPROVAL_TOKEN:
        raise PermissionError("full training gate is not approved")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for full SchNet training")
    validation = json.loads(
        (graph_dir / "validation.json").read_text(encoding="utf-8")
    )
    if validation.get("status") != "accepted":
        raise RuntimeError("graph cache has not passed strict validation")

    graphs = _load_all_shards(graph_dir)
    train_set, validation_set, test_set = split_graphs_by_source_idx(
        graphs,
        source_rows=source_rows,
        seed=seed,
    )
    selected_model_config = dict(model_config or LIGHT_SCHNET_CONFIG)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    model = SchNetWrapper(**selected_model_config, use_charges=True).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=2.1150972021685588e-4, weight_decay=1.4656553886225336e-5
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda")
    criterion = nn.L1Loss()
    loader_generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    last_path = checkpoint_dir / "repaired_2m_light_schnet_last.pt"
    best_path = checkpoint_dir / "repaired_2m_light_schnet_best.pt"
    start_epoch = 0
    best_validation = float("inf")
    best_epoch = -1
    log = []
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        if (
            checkpoint["model_config"] != selected_model_config
            or int(checkpoint.get("source_rows", -1)) != source_rows
        ):
            raise RuntimeError("checkpoint model configuration mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_validation = float(checkpoint["best_validation"])
        best_epoch = int(checkpoint["best_epoch"])
        log = list(checkpoint["log"])
        random.setstate(checkpoint["python_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        loader_generator.set_state(checkpoint["loader_generator_state"])
        print(
            f"resuming epoch {start_epoch}; "
            f"best={best_validation:.6f}@{best_epoch}",
            flush=True,
        )

    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        model.train()
        train_total = 0.0
        train_count = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                loss = criterion(_forward(model, batch), batch.y.view(-1, 3))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(optimizer)
            scaler.update()
            train_total += float(loss.detach()) * batch.num_graphs
            train_count += batch.num_graphs
        model.eval()
        validation_total = 0.0
        validation_count = 0
        with torch.no_grad():
            for batch in validation_loader:
                batch = batch.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    loss = criterion(_forward(model, batch), batch.y.view(-1, 3))
                validation_total += float(loss) * batch.num_graphs
                validation_count += batch.num_graphs
        scheduler.step()
        train_mae = train_total / train_count
        validation_mae = validation_total / validation_count
        improved = validation_mae < best_validation
        if improved:
            best_validation = validation_mae
            best_epoch = epoch
            atomic_torch_save(model.state_dict(), best_path)
        row = {
            "epoch": epoch,
            "train_mae_eV": train_mae,
            "validation_mae_eV": validation_mae,
            "best_validation_mae_eV": best_validation,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.perf_counter() - started,
            "selected": improved,
        }
        log.append(row)
        atomic_torch_save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_validation": best_validation,
                "best_epoch": best_epoch,
                "log": log,
                "model_config": selected_model_config,
                "source_rows": source_rows,
                "split_seed": seed,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
                "loader_generator_state": loader_generator.get_state(),
            },
            last_path,
        )
        atomic_json(
            {
                "status": "running",
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_mae_eV": best_validation,
            },
            result_dir / "training_progress.json",
        )
        print(
            f"ep{epoch:03d} train={train_mae:.5f} val={validation_mae:.5f} "
            f"best={best_validation:.5f}@{best_epoch} "
            f"{row['elapsed_s']:.1f}s{' *' if improved else ''}",
            flush=True,
        )

    model.load_state_dict(
        torch.load(best_path, map_location=device, weights_only=True)
    )
    predictions = []
    targets = []
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                predictions.append(_forward(model, batch).float().cpu())
            targets.append(batch.y.view(-1, 3).float().cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    error = (prediction - target).abs()
    metrics = {
        name: {"mae_eV": float(error[:, index].mean())}
        for index, name in enumerate(("HOMO", "LUMO", "Gap"))
    }
    metrics["average"] = {"mae_eV": float(error.mean())}
    result = {
        "status": "complete",
        "model_config": selected_model_config,
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "split": {
            "seed": seed,
            "source_rows": source_rows,
            "train": len(train_set),
            "validation": len(validation_set),
            "test": len(test_set),
        },
        "best_epoch": best_epoch,
        "best_validation_mae_eV": best_validation,
        "test_metrics": metrics,
        "log": log,
        "artifacts": {
            "best": str(best_path),
            "best_sha256": sha256(best_path),
            "last": str(last_path),
            "last_sha256": sha256(last_path),
        },
    }
    atomic_json(result, result_dir / "training_metrics.json")
    return result


def export_embeddings(
    *,
    graph_dir: Path,
    best_checkpoint: Path,
    output_dir: Path,
    batch_size: int = 128,
    num_workers: int = 2,
    model_config: dict | None = None,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for embedding export")
    device = torch.device("cuda")
    selected_model_config = dict(model_config or LIGHT_SCHNET_CONFIG)
    model = SchNetWrapper(**selected_model_config, use_charges=True).to(device)
    model.load_state_dict(
        torch.load(best_checkpoint, map_location=device, weights_only=True)
    )
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for index, graph_path in enumerate(
        sorted(graph_dir.glob("graphs_*.pt")), start=1
    ):
        output_path = output_dir / f"embeddings_{graph_path.stem[7:]}.pt"
        report_path = output_path.with_suffix(".json")
        if output_path.exists() and report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("sha256") == sha256(output_path):
                reports.append(report)
                continue
        graphs = torch.load(graph_path, map_location="cpu", weights_only=False)
        embeddings = []
        predictions = []
        source_indices = []
        targets = []
        with torch.no_grad():
            for batch in DataLoader(
                graphs,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
                persistent_workers=num_workers > 0,
            ):
                batch = batch.to(device, non_blocking=True)
                with torch.amp.autocast("cuda"):
                    embedding = model.encode(
                        batch.z,
                        batch.pos,
                        batch.batch,
                        charges=getattr(batch, "charges", None),
                    )
                    prediction = model.head(embedding)
                embeddings.append(embedding.float().cpu())
                predictions.append(prediction.float().cpu())
                source_indices.append(batch.source_idx.view(-1).long().cpu())
                targets.append(batch.y.view(-1, 3).float().cpu())
        payload = {
            "embeddings": torch.cat(embeddings),
            "predictions": torch.cat(predictions),
            "source_idx": torch.cat(source_indices),
            "targets": torch.cat(targets),
            "checkpoint_sha256": sha256(best_checkpoint),
        }
        atomic_torch_save(payload, output_path)
        report = {
            "status": "complete",
            "part": index,
            "rows": len(payload["source_idx"]),
            "path": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256(output_path),
        }
        atomic_json(report, report_path)
        reports.append(report)
        print(f"embedding part {index}/100 rows={report['rows']}", flush=True)
    result = {
        "status": "complete",
        "parts": len(reports),
        "rows": sum(report["rows"] for report in reports),
        "checkpoint_sha256": sha256(best_checkpoint),
    }
    atomic_json(result, output_dir / "completion.json")
    return result
