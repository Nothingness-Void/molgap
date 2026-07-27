"""Strict acceptance for Route B SchNet and repaired-2M 3D graph artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from .pubchemqc_architecture import MODEL_CONFIG
from .schnet import SchNetWrapper


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _split_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source_idx", "split", "cid"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Split CSV requires {sorted(required)}")
    result = {int(row["source_idx"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Split CSV contains duplicate source_idx")
    return result


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    return hashlib.sha256(tensor.numpy().tobytes()).hexdigest()


def accept_schnet_output(
    output_dir: Path,
    split_csv: Path,
    accepted_payload_path: Path,
    report_path: Path,
    *,
    expected_split_sha256: str,
) -> dict:
    split_hash = sha256_file(split_csv)
    if split_hash != expected_split_sha256:
        raise ValueError("SchNet split hash differs from frozen contract")
    completion_path = output_dir / "completion_manifest.json"
    metrics_path = output_dir / "metrics.json"
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    embeddings_path = output_dir / "embeddings.pt"
    for path in (
        completion_path,
        metrics_path,
        best_path,
        last_path,
        embeddings_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("model_config") != MODEL_CONFIG:
        raise ValueError("SchNet metrics architecture is not 176/160/6 contract")
    for path in (best_path, last_path, embeddings_path):
        declared = completion["artifacts"].get(path.name)
        if not declared or declared["sha256"] != sha256_file(path):
            raise ValueError(f"SchNet artifact hash mismatch: {path.name}")
        if int(declared["bytes"]) != path.stat().st_size:
            raise ValueError(f"SchNet artifact byte count mismatch: {path.name}")

    best = torch.load(best_path, map_location="cpu", weights_only=False)
    if best.get("model_config") != MODEL_CONFIG:
        raise ValueError("SchNet checkpoint architecture is not 176/160/6")
    model = SchNetWrapper(**MODEL_CONFIG, use_charges=True)
    model.load_state_dict(best["model"], strict=True)
    payload = torch.load(embeddings_path, map_location="cpu", weights_only=False)
    split = _split_rows(split_csv)
    accepted = {}
    role_reports = {}
    for role in ("train", "validation", "test"):
        item = payload[role]
        source_idx = item["source_idx"].long()
        embeddings = item["embeddings"].float()
        targets = item["targets"].float()
        if embeddings.ndim != 2 or embeddings.shape[1] != 176:
            raise ValueError(f"{role} embedding dimension is not 176")
        if len(source_idx) != len(embeddings) or len(targets) != len(source_idx):
            raise ValueError(f"{role} SchNet payload row counts differ")
        if len(set(source_idx.tolist())) != len(source_idx):
            raise ValueError(f"{role} SchNet source_idx contains duplicates")
        if not torch.isfinite(embeddings).all() or not torch.isfinite(targets).all():
            raise ValueError(f"{role} SchNet payload contains non-finite values")
        indices = source_idx.tolist()
        if indices != sorted(indices):
            raise ValueError(f"{role} SchNet source_idx is not in canonical order")
        if any(
            index not in split or split[index]["split"].lower() != role
            for index in indices
        ):
            raise ValueError(f"{role} SchNet source_idx violates the frozen split")
        declared_rows = int(metrics["aligned_rows"][role])
        if len(indices) != declared_rows:
            raise ValueError(f"{role} SchNet row count differs from metrics")
        expected_targets = torch.tensor(
            [
                [
                    float(split[index]["homo"]),
                    float(split[index]["lumo"]),
                    float(split[index]["gap"]),
                ]
                for index in indices
            ],
            dtype=torch.float32,
        )
        if not torch.allclose(targets, expected_targets, atol=1e-6, rtol=0.0):
            raise ValueError(f"{role} SchNet targets differ from frozen split")
        cids = [split[index]["cid"] for index in indices]
        identity_hash = hashlib.sha256(
            "\n".join(f"{index},{cid}" for index, cid in zip(indices, cids)).encode()
        ).hexdigest()
        frozen_role_rows = sum(
            row["split"].lower() == role for row in split.values()
        )
        accepted[role] = {
            "source_idx": source_idx,
            "cid": cids,
            "embeddings": embeddings,
            "targets": targets,
        }
        role_reports[role] = {
            "rows": len(source_idx),
            "embedding_dim": int(embeddings.shape[1]),
            "identity_sha256": identity_hash,
            "targets_sha256": _tensor_sha256(targets),
            "excluded_from_frozen_split": frozen_role_rows - len(source_idx),
        }
    _atomic_torch(accepted_payload_path, accepted)
    report = {
        "accepted": True,
        "variant": metrics["variant"],
        "model_config": MODEL_CONFIG,
        "checkpoint_loadable": True,
        "predictions_finite": all(
            math.isfinite(float(value["mae_eV"]))
            for role in metrics["metrics"].values()
            for view in role.values()
            for value in view.values()
        ),
        "split_sha256": split_hash,
        "roles": role_reports,
        "artifacts": {
            "best": sha256_file(best_path),
            "last": sha256_file(last_path),
            "embeddings": sha256_file(embeddings_path),
            "accepted_payload": sha256_file(accepted_payload_path),
        },
        "fusion_unblocked_for_this_branch": True,
    }
    _atomic_json(report_path, report)
    return report


def accept_schnet_pair(primary_report: Path, augmented_report: Path) -> dict:
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (primary_report, augmented_report)
    ]
    variants = {report["variant"] for report in reports}
    accepted = all(report.get("accepted") for report in reports)
    aligned = all(
        reports[0]["roles"][role]["rows"] == reports[1]["roles"][role]["rows"]
        and reports[0]["roles"][role]["identity_sha256"]
        == reports[1]["roles"][role]["identity_sha256"]
        and reports[0]["roles"][role]["targets_sha256"]
        == reports[1]["roles"][role]["targets_sha256"]
        for role in ("train", "validation", "test")
    )
    return {
        "accepted": accepted and aligned and variants == {"primary", "augmented"},
        "variants": sorted(variants),
        "branch_alignment": aligned,
        "fusion_unblocked": (
            accepted and aligned and variants == {"primary", "augmented"}
        ),
    }


def align_gps_embeddings_to_reference(
    raw_embeddings_path: Path,
    reference_payload_path: Path,
    accepted_payload_path: Path,
    report_path: Path,
    *,
    expected_dim: int,
    expected_sha256: str,
    name: str,
) -> dict:
    raw_hash = sha256_file(raw_embeddings_path)
    if raw_hash != expected_sha256:
        raise ValueError(f"{name} raw embedding hash differs from contract")
    raw = torch.load(raw_embeddings_path, map_location="cpu", weights_only=False)
    embeddings = raw["embeddings"].float()
    source_idx = raw["source_idx"].long()
    if embeddings.ndim != 2 or embeddings.shape[1] != expected_dim:
        raise ValueError(f"{name} embedding dimension differs from contract")
    if len(embeddings) != len(source_idx):
        raise ValueError(f"{name} embedding row counts differ")
    if len(set(source_idx.tolist())) != len(source_idx):
        raise ValueError(f"{name} contains duplicate source_idx")
    if not torch.isfinite(embeddings).all():
        raise ValueError(f"{name} contains non-finite embeddings")
    positions = {int(index): offset for offset, index in enumerate(source_idx)}
    reference = torch.load(
        reference_payload_path, map_location="cpu", weights_only=False
    )
    accepted = {}
    roles = {}
    for role in ("train", "validation", "test"):
        indices = reference[role]["source_idx"].long()
        missing = [int(index) for index in indices if int(index) not in positions]
        if missing:
            raise ValueError(f"{name}/{role} is missing {len(missing)} reference rows")
        take = torch.tensor(
            [positions[int(index)] for index in indices], dtype=torch.long
        )
        aligned = embeddings[take].contiguous()
        accepted[role] = {
            "source_idx": indices.clone(),
            "cid": list(reference[role]["cid"]),
            "embeddings": aligned,
            "targets": reference[role]["targets"].float().clone(),
        }
        roles[role] = {
            "rows": len(indices),
            "embedding_dim": expected_dim,
            "identity_sha256": hashlib.sha256(
                "\n".join(
                    f"{int(index)},{cid}"
                    for index, cid in zip(indices, reference[role]["cid"])
                ).encode()
            ).hexdigest(),
            "targets_sha256": _tensor_sha256(accepted[role]["targets"]),
        }
    _atomic_torch(accepted_payload_path, accepted)
    report = {
        "accepted": True,
        "name": name,
        "raw_rows": len(source_idx),
        "embedding_dim": expected_dim,
        "raw_sha256": raw_hash,
        "reference_payload_sha256": sha256_file(reference_payload_path),
        "accepted_payload_sha256": sha256_file(accepted_payload_path),
        "roles": roles,
    }
    _atomic_json(report_path, report)
    return report


@torch.no_grad()
def extract_schnet_view_for_reference(
    checkpoint_path: Path,
    graph_cache_path: Path,
    reference_payload_path: Path,
    accepted_payload_path: Path,
    report_path: Path,
    *,
    batch_size: int = 128,
) -> dict:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if checkpoint.get("model_config") != MODEL_CONFIG:
        raise ValueError("SchNet checkpoint architecture differs from contract")
    reference = torch.load(
        reference_payload_path, map_location="cpu", weights_only=False
    )
    required = {
        int(index)
        for role in ("train", "validation", "test")
        for index in reference[role]["source_idx"]
    }
    graphs = torch.load(graph_cache_path, map_location="cpu", weights_only=False)
    selected = {}
    for graph in graphs:
        index = int(graph.source_idx.view(-1)[0])
        if index in required:
            if index in selected:
                raise ValueError("Primary graph cache contains duplicate source_idx")
            selected[index] = graph
    missing = required - set(selected)
    if missing:
        raise ValueError(f"Primary graph cache is missing {len(missing)} rows")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SchNetWrapper(**MODEL_CONFIG, use_charges=True).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    accepted = {}
    roles = {}
    for role in ("train", "validation", "test"):
        indices = reference[role]["source_idx"].long()
        role_graphs = [selected[int(index)] for index in indices]
        embeddings = []
        targets = []
        observed = []
        for batch in DataLoader(
            role_graphs, batch_size=batch_size, shuffle=False, num_workers=0
        ):
            batch = batch.to(device)
            charges = batch.charges if hasattr(batch, "charges") else None
            embeddings.append(
                model.encode(
                    batch.z, batch.pos, batch.batch, charges=charges
                ).float().cpu()
            )
            targets.append(batch.y.view(-1, 3).float().cpu())
            observed.append(batch.source_idx.view(-1).long().cpu())
        role_embeddings = torch.cat(embeddings)
        role_targets = torch.cat(targets)
        role_indices = torch.cat(observed)
        if not torch.equal(role_indices, indices):
            raise ValueError(f"{role} extracted SchNet source_idx order differs")
        if not torch.allclose(
            role_targets,
            reference[role]["targets"].float(),
            atol=1e-6,
            rtol=0.0,
        ):
            raise ValueError(f"{role} extracted SchNet targets differ")
        if not torch.isfinite(role_embeddings).all():
            raise ValueError(f"{role} extracted SchNet embeddings are non-finite")
        accepted[role] = {
            "source_idx": indices.clone(),
            "cid": list(reference[role]["cid"]),
            "embeddings": role_embeddings,
            "targets": role_targets,
        }
        roles[role] = {
            "rows": len(indices),
            "embedding_dim": int(role_embeddings.shape[1]),
            "embedding_sha256": _tensor_sha256(role_embeddings),
        }
    _atomic_torch(accepted_payload_path, accepted)
    report = {
        "accepted": True,
        "view": "primary_conformer_only",
        "device": str(device),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "graph_cache_sha256": sha256_file(graph_cache_path),
        "reference_payload_sha256": sha256_file(reference_payload_path),
        "accepted_payload_sha256": sha256_file(accepted_payload_path),
        "roles": roles,
    }
    _atomic_json(report_path, report)
    return report


def accept_repaired_3d_graphs(
    shard_dir: Path,
    source_csv: Path,
    output_manifest: Path,
    *,
    expected_shards: int = 100,
    pattern: str = "graphs_*.pt",
) -> dict:
    shards = sorted(shard_dir.glob(pattern))
    if len(shards) != expected_shards:
        raise ValueError(f"Expected {expected_shards} graph shards, found {len(shards)}")
    with source_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"cid", "canonical_smiles", "homo", "lumo", "gap"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Source CSV requires columns {sorted(required)}")
        source_rows = [
            (
                str(row["cid"]),
                str(row["canonical_smiles"]),
                torch.tensor(
                    [
                        float(row["homo"]),
                        float(row["lumo"]),
                        float(row["gap"]),
                    ],
                    dtype=torch.float32,
                ),
            )
            for row in reader
        ]
    source_cids = [row[0] for row in source_rows]
    if len(set(source_cids)) != len(source_cids):
        raise ValueError("Source CSV contains duplicate CID values")

    seen_source = set()
    seen_cid = set()
    rows = 0
    reused = built = failed = 0
    graphs_with_embedded_cid = 0
    graphs_with_embedded_smiles = 0
    entries = []
    for path in shards:
        sidecar_candidates = (
            path.parent / "reports" / f"{path.stem}.json",
            path.with_suffix(".json"),
        )
        existing_sidecars = [
            candidate for candidate in sidecar_candidates if candidate.is_file()
        ]
        if not existing_sidecars:
            raise FileNotFoundError(
                "No graph sidecar found; checked "
                + ", ".join(str(candidate) for candidate in sidecar_candidates)
            )
        if len(existing_sidecars) > 1:
            hashes = {sha256_file(candidate) for candidate in existing_sidecars}
            if len(hashes) != 1:
                raise ValueError(f"Conflicting graph sidecars for {path.name}")
        sidecar = existing_sidecars[0]
        report = json.loads(sidecar.read_text(encoding="utf-8"))
        required_report = {
            "status",
            "start",
            "stop",
            "requested",
            "reused",
            "built",
            "failed",
            "failure_source_idx",
            "graphs",
            "path",
            "bytes",
            "sha256",
        }
        if not required_report.issubset(report):
            missing = sorted(required_report - set(report))
            raise ValueError(f"Graph sidecar missing {missing}: {sidecar}")
        start = int(report["start"])
        stop = int(report["stop"])
        if (
            report["status"] != "complete"
            or start < 0
            or stop <= start
            or int(report["requested"]) != stop - start
            or str(report["path"]) != path.name
            or int(report["bytes"]) != path.stat().st_size
            or str(report["sha256"]) != sha256_file(path)
        ):
            raise ValueError(f"Graph sidecar contract mismatch: {sidecar}")

        graphs = torch.load(path, map_location="cpu", weights_only=False)
        shard_source_idx = set()
        for graph in graphs:
            index = int(graph.source_idx.view(-1)[0])
            if index < 0 or index >= len(source_rows):
                raise ValueError(f"source_idx outside source CSV in {path.name}")
            if index < start or index >= stop:
                raise ValueError(f"source_idx outside shard range in {path.name}")
            cid, smiles, expected_target = source_rows[index]
            if hasattr(graph, "cid"):
                graph_cid = str(
                    int(graph.cid.view(-1)[0])
                    if isinstance(graph.cid, torch.Tensor)
                    else graph.cid
                )
                if graph_cid != cid:
                    raise ValueError(f"CID/source_idx mismatch in {path.name}")
                graphs_with_embedded_cid += 1
            graph_smiles = None
            if hasattr(graph, "canonical_smiles"):
                graph_smiles = str(graph.canonical_smiles)
            elif hasattr(graph, "smiles"):
                graph_smiles = str(graph.smiles)
            if graph_smiles is not None:
                if graph_smiles != smiles:
                    raise ValueError(f"SMILES/source_idx mismatch in {path.name}")
                graphs_with_embedded_smiles += 1
            if index in seen_source or cid in seen_cid:
                raise ValueError(f"Duplicate graph identity in {path.name}")
            if not cid or not smiles or graph.pos.ndim != 2 or graph.pos.shape[1] != 3:
                raise ValueError(f"Invalid graph structure in {path.name}")
            if not torch.isfinite(graph.pos).all() or not torch.isfinite(graph.y).all():
                raise ValueError(f"Non-finite graph values in {path.name}")
            observed_target = graph.y.detach().cpu().float().view(-1)
            if observed_target.shape != expected_target.shape or not torch.allclose(
                observed_target, expected_target, atol=1e-6, rtol=0.0
            ):
                raise ValueError(f"Target/source_idx mismatch in {path.name}")
            seen_source.add(index)
            seen_cid.add(cid)
            shard_source_idx.add(index)
        if int(report["graphs"]) != len(graphs):
            raise ValueError(f"Sidecar row count mismatch for {path.name}")
        shard_reused = int(report["reused"])
        shard_built = int(report["built"])
        shard_failed = int(report["failed"])
        if shard_reused + shard_built != len(graphs):
            raise ValueError(f"Sidecar graph accounting mismatch for {path.name}")
        if shard_reused + shard_built + shard_failed != int(report["requested"]):
            raise ValueError(f"Sidecar source accounting mismatch for {path.name}")
        failure_source_idx = [int(value) for value in report["failure_source_idx"]]
        if (
            len(failure_source_idx) != shard_failed
            or len(set(failure_source_idx)) != shard_failed
            or any(index < start or index >= stop for index in failure_source_idx)
            or shard_source_idx.intersection(failure_source_idx)
        ):
            raise ValueError(f"Invalid failure_source_idx in {path.name}")
        reused += shard_reused
        built += shard_built
        failed += shard_failed
        rows += len(graphs)
        entries.append(
            {
                "path": path.name,
                "rows": len(graphs),
                "sha256": str(report["sha256"]),
                "sidecar_path": sidecar.relative_to(shard_dir).as_posix(),
                "sidecar_sha256": sha256_file(sidecar),
            }
        )
    if reused + built + failed != len(source_rows):
        raise ValueError("Reused + built + failed does not reconcile source rows")
    if rows != reused + built:
        raise ValueError("Accepted graph rows do not reconcile reused + built")
    manifest = {
        "accepted": True,
        "immutable": True,
        "expected_shards": expected_shards,
        "shards": entries,
        "source_rows": len(source_rows),
        "accepted_rows": rows,
        "reused": reused,
        "built": built,
        "failed": failed,
        "unique_source_idx": len(seen_source),
        "unique_cid": len(seen_cid),
        "identity_source": "source_csv_by_source_idx",
        "graphs_with_embedded_cid": graphs_with_embedded_cid,
        "graphs_with_embedded_smiles": graphs_with_embedded_smiles,
        "target_alignment": True,
        "source_csv_sha256": sha256_file(source_csv),
    }
    _atomic_json(output_manifest, manifest)
    return manifest
