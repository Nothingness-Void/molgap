"""Reusable utilities for distilling aligned molecular embedding teachers."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .fusion import DualGPSFusionHead


def atomic_torch_save(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def atomic_json_write(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_indices(graphs: Sequence[object], start: int, end: int) -> torch.Tensor:
    return torch.as_tensor(
        [int(graphs[index].source_idx.view(-1)[0]) for index in range(start, end)],
        dtype=torch.long,
    )


def extract_gps_embedding_parts(
    model: nn.Module,
    graphs: Sequence[object],
    *,
    model_path: Path,
    out_dir: Path,
    device: torch.device,
    batch_size: int = 256,
    chunk_size: int = 50_000,
    storage_dtype: torch.dtype = torch.float16,
) -> dict:
    """Extract resumable, independently retrievable GPS embedding chunks."""
    if batch_size <= 0 or chunk_size <= 0 or not graphs:
        raise ValueError("batch_size, chunk_size, and graphs must be non-empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_hash = sha256_file(model_path)
    graph_count = len(graphs)
    manifest_path = out_dir / "manifest.json"
    records = []
    model.eval()
    for part_number, start in enumerate(range(0, graph_count, chunk_size)):
        end = min(start + chunk_size, graph_count)
        part_path = out_dir / f"part-{part_number:03d}.pt"
        expected_source = torch.arange(start, end, dtype=torch.long)
        if part_path.is_file():
            payload = torch.load(part_path, map_location="cpu", weights_only=False)
            valid = (
                payload.get("model_sha256") == model_hash
                and payload.get("source_start") == start
                and payload.get("source_end") == end
                and torch.equal(payload.get("source_idx"), expected_source)
                and tuple(payload.get("embeddings", torch.empty(0)).shape)
                == (end - start, 192)
            )
            if valid:
                records.append(
                    {
                        "part": part_number,
                        "start": start,
                        "end": end,
                        "rows": end - start,
                        "path": str(part_path),
                        "bytes": part_path.stat().st_size,
                        "sha256": sha256_file(part_path),
                    }
                )
                print(f"Reused {part_path}", flush=True)
                continue

        actual_source = _source_indices(graphs, start, end)
        if not torch.equal(actual_source, expected_source):
            raise ValueError(
                f"Graph source_idx is not contiguous at rows {start:,}:{end:,}"
            )
        loader = DataLoader(
            graphs[start:end], batch_size=batch_size, shuffle=False, num_workers=0
        )
        embeddings = []
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(device)
                with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                    encoded = model.encode(
                        batch.x, batch.edge_index, batch.edge_attr, batch.batch
                    )
                embeddings.append(encoded.to(dtype=storage_dtype).cpu())
        payload = {
            "format": "molgap-gps-embedding-part-v1",
            "model_sha256": model_hash,
            "source_start": start,
            "source_end": end,
            "source_idx": expected_source,
            "embeddings": torch.cat(embeddings),
        }
        atomic_torch_save(payload, part_path)
        record = {
            "part": part_number,
            "start": start,
            "end": end,
            "rows": end - start,
            "path": str(part_path),
            "bytes": part_path.stat().st_size,
            "sha256": sha256_file(part_path),
        }
        records.append(record)
        atomic_json_write(
            {
                "format": "molgap-gps-embedding-manifest-v1",
                "complete": False,
                "model": str(model_path),
                "model_sha256": model_hash,
                "rows": graph_count,
                "embedding_dim": 192,
                "storage_dtype": str(storage_dtype),
                "parts": records,
            },
            manifest_path,
        )
        print(f"Saved {part_path}: rows {start:,}:{end:,}", flush=True)

    manifest = {
        "format": "molgap-gps-embedding-manifest-v1",
        "complete": True,
        "model": str(model_path),
        "model_sha256": model_hash,
        "rows": graph_count,
        "embedding_dim": 192,
        "storage_dtype": str(storage_dtype),
        "parts": records,
    }
    atomic_json_write(manifest, manifest_path)
    return manifest


@dataclass(frozen=True)
class TeacherEmbeddingSpec:
    name: str
    gps7_dir: Path
    gps9_dir: Path
    head_path: Path


def _load_embedding_manifest(path: Path) -> dict:
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != "molgap-gps-embedding-manifest-v1"
        or not manifest.get("complete")
    ):
        raise ValueError(f"Incomplete embedding manifest: {manifest_path}")
    return manifest


def _part_fingerprint(entries: Sequence[dict], head_hashes: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in [*(entry["sha256"] for entry in entries), *head_hashes]:
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def build_teacher_target_parts(
    specs: Sequence[TeacherEmbeddingSpec],
    *,
    out_dir: Path,
    device: torch.device,
    batch_size: int = 8192,
) -> dict:
    """Average dual-GPS expert predictions into resumable soft-target chunks."""
    if len(specs) < 2:
        raise ValueError("Distillation requires at least two teacher experts")
    manifests = [
        (_load_embedding_manifest(spec.gps7_dir), _load_embedding_manifest(spec.gps9_dir))
        for spec in specs
    ]
    rows = int(manifests[0][0]["rows"])
    reference_parts = manifests[0][0]["parts"]
    for pair in manifests:
        for manifest in pair:
            if int(manifest["rows"]) != rows:
                raise ValueError("Teacher embedding row counts differ")
            if [(p["start"], p["end"]) for p in manifest["parts"]] != [
                (p["start"], p["end"]) for p in reference_parts
            ]:
                raise ValueError("Teacher embedding chunk boundaries differ")

    heads = []
    head_hashes = []
    for spec in specs:
        head = DualGPSFusionHead(hidden=192)
        head.load_state_dict(
            torch.load(spec.head_path, map_location=device, weights_only=True)
        )
        heads.append(head.to(device).eval())
        head_hashes.append(sha256_file(spec.head_path))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    records = []
    for part_number, reference in enumerate(reference_parts):
        start, end = int(reference["start"]), int(reference["end"])
        part_path = out_dir / f"part-{part_number:03d}.pt"
        input_entries = []
        for gps7_manifest, gps9_manifest in manifests:
            input_entries.extend(
                [gps7_manifest["parts"][part_number], gps9_manifest["parts"][part_number]]
            )
        fingerprint = _part_fingerprint(input_entries, head_hashes)
        expected_source = torch.arange(start, end, dtype=torch.long)
        if part_path.is_file():
            payload = torch.load(part_path, map_location="cpu", weights_only=False)
            valid = (
                payload.get("fingerprint") == fingerprint
                and torch.equal(payload.get("source_idx"), expected_source)
                and tuple(payload.get("targets", torch.empty(0)).shape)
                == (end - start, 3)
            )
            if valid:
                records.append(
                    {
                        "part": part_number,
                        "start": start,
                        "end": end,
                        "rows": end - start,
                        "path": str(part_path),
                        "bytes": part_path.stat().st_size,
                        "sha256": sha256_file(part_path),
                    }
                )
                print(f"Reused teacher targets {part_path}", flush=True)
                continue

        expert_predictions = []
        for expert_index, spec in enumerate(specs):
            gps7_entry = manifests[expert_index][0]["parts"][part_number]
            gps9_entry = manifests[expert_index][1]["parts"][part_number]
            gps7 = torch.load(gps7_entry["path"], map_location="cpu", weights_only=False)
            gps9 = torch.load(gps9_entry["path"], map_location="cpu", weights_only=False)
            if not torch.equal(gps7["source_idx"], gps9["source_idx"]):
                raise ValueError(f"Misaligned teacher embeddings for {spec.name}")
            chunks = []
            with torch.inference_mode():
                for offset in range(0, end - start, batch_size):
                    stop = min(offset + batch_size, end - start)
                    with torch.amp.autocast(
                        "cuda", enabled=torch.cuda.is_available()
                    ):
                        prediction = heads[expert_index](
                            gps7["embeddings"][offset:stop].float().to(device),
                            gps9["embeddings"][offset:stop].float().to(device),
                        )
                    chunks.append(prediction.float().cpu())
            expert_predictions.append(torch.cat(chunks))
        targets = torch.stack(expert_predictions).mean(dim=0)
        payload = {
            "format": "molgap-distillation-target-part-v1",
            "fingerprint": fingerprint,
            "source_idx": expected_source,
            "targets": targets,
            "experts": [spec.name for spec in specs],
        }
        atomic_torch_save(payload, part_path)
        records.append(
            {
                "part": part_number,
                "start": start,
                "end": end,
                "rows": end - start,
                "path": str(part_path),
                "bytes": part_path.stat().st_size,
                "sha256": sha256_file(part_path),
            }
        )
        atomic_json_write(
            {
                "format": "molgap-distillation-target-manifest-v1",
                "complete": False,
                "rows": rows,
                "experts": [spec.name for spec in specs],
                "parts": records,
            },
            manifest_path,
        )
        print(f"Saved teacher targets {part_path}", flush=True)

    manifest = {
        "format": "molgap-distillation-target-manifest-v1",
        "complete": True,
        "rows": rows,
        "experts": [spec.name for spec in specs],
        "head_sha256": dict(zip([spec.name for spec in specs], head_hashes)),
        "parts": records,
    }
    atomic_json_write(manifest, manifest_path)
    return manifest


def load_teacher_targets(manifest_path: Path) -> torch.Tensor:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format") != "molgap-distillation-target-manifest-v1"
        or not manifest.get("complete")
    ):
        raise ValueError(f"Incomplete teacher target manifest: {manifest_path}")
    rows = int(manifest["rows"])
    targets = torch.empty((rows, 3), dtype=torch.float32)
    covered = 0
    for entry in manifest["parts"]:
        payload = torch.load(entry["path"], map_location="cpu", weights_only=False)
        indices = payload["source_idx"].long()
        if int(indices[0]) != covered or int(indices[-1]) != int(entry["end"]) - 1:
            raise ValueError(f"Non-contiguous target part: {entry['path']}")
        targets[indices] = payload["targets"].float()
        covered = int(entry["end"])
    if covered != rows or not torch.isfinite(targets).all():
        raise ValueError(f"Teacher targets cover {covered:,} of {rows:,} rows")
    return targets


def merge_embedding_prefix(
    manifest_path: Path, *, rows: int, out_path: Path
) -> dict:
    """Merge an aligned embedding prefix for a later 2D+3D fusion job."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or rows <= 0 or rows > int(manifest["rows"]):
        raise ValueError("Invalid complete embedding manifest or prefix length")
    embeddings = torch.empty((rows, 192), dtype=torch.float16)
    covered = 0
    for entry in manifest["parts"]:
        if covered >= rows:
            break
        payload = torch.load(entry["path"], map_location="cpu", weights_only=False)
        indices = payload["source_idx"].long()
        keep = indices < rows
        embeddings[indices[keep]] = payload["embeddings"][keep].to(torch.float16)
        covered = min(int(entry["end"]), rows)
    if covered != rows or not torch.isfinite(embeddings).all():
        raise ValueError(f"Embedding prefix covers {covered:,} of {rows:,} rows")
    payload = {
        "format": "molgap-gps-embedding-prefix-v1",
        "model_sha256": manifest["model_sha256"],
        "embeddings": embeddings,
        "source_idx": torch.arange(rows, dtype=torch.long),
    }
    atomic_torch_save(payload, out_path)
    return {
        "path": str(out_path),
        "rows": rows,
        "embedding_dim": 192,
        "dtype": "float16",
        "bytes": out_path.stat().st_size,
        "sha256": sha256_file(out_path),
    }
