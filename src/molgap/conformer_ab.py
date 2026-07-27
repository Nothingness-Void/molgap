"""Paired ETKDG versus ETKDG+MMFF construction and frozen-model evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable

import torch

from .pubchemqc_architecture import MODEL_CONFIG, evaluate_view
from .route_b_fusion import build_fusion_head
from .schnet import SchNetWrapper
from .utils import compute_gasteiger_charges

PROTOCOLS = {
    "bare_etkdg": 0,
    "etkdgv3_mmff200": 200,
}
ROLES = ("train", "validation", "test")
TARGET_NAMES = ("homo", "lumo", "gap")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_aligned_rows(
    split_csv: Path,
    reference_payload: Path,
    counts: dict[str, int],
) -> list[dict]:
    """Select deterministic role-stratified rows accepted by every Route B expert."""
    if set(counts) != set(ROLES) or any(counts[role] <= 0 for role in ROLES):
        raise ValueError(f"counts must contain positive values for {ROLES}")
    reference = torch.load(reference_payload, map_location="cpu", weights_only=False)
    accepted = {
        role: set(reference[role]["source_idx"].long().tolist()) for role in ROLES
    }
    selected: dict[str, list[dict]] = {role: [] for role in ROLES}
    with split_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            role = row["split"].strip().lower()
            if role not in selected or len(selected[role]) >= counts[role]:
                continue
            source_idx = int(row["source_idx"])
            if source_idx not in accepted[role]:
                continue
            selected[role].append(
                {
                    "source_idx": source_idx,
                    "role": role,
                    "cid": str(row["cid"]),
                    "smiles": row.get("canonical_smiles") or row["smiles"],
                    "targets": [
                        float(row["homo"]),
                        float(row["lumo"]),
                        float(row["gap"]),
                    ],
                }
            )
    missing = {
        role: counts[role] - len(selected[role])
        for role in ROLES
        if len(selected[role]) != counts[role]
    }
    if missing:
        raise ValueError(f"not enough accepted rows for requested split: {missing}")
    return [row for role in ROLES for row in selected[role]]


def write_selection(path: Path, rows: list[dict]) -> dict:
    payload = {
        "format": "molgap-conformer-ab-selection-v1",
        "rows": len(rows),
        "counts": {
            role: sum(row["role"] == role for row in rows) for role in ROLES
        },
        "source_idx_sha256": hashlib.sha256(
            ",".join(str(row["source_idx"]) for row in rows).encode("ascii")
        ).hexdigest(),
        "items": rows,
    }
    _atomic_json(path, payload)
    return payload


def _row_seed(base_seed: int, source_idx: int) -> int:
    return int((base_seed * 1_000_003 + source_idx) % 2_147_483_647)


def _build_one(work: tuple[dict, str, int]) -> dict:
    row, protocol, base_seed = work
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from torch_geometric.data import Data

    started = time.perf_counter()
    molecule = Chem.MolFromSmiles(row["smiles"])
    if molecule is None:
        return {"source_idx": row["source_idx"], "error": "parse_failed"}
    molecule = Chem.AddHs(molecule)
    embedded = False
    embed_started = time.perf_counter()
    for attempt in range(2):
        parameters = AllChem.ETKDGv3()
        parameters.randomSeed = _row_seed(base_seed + attempt, row["source_idx"])
        if AllChem.EmbedMolecule(molecule, parameters) == 0:
            embedded = True
            break
    embed_s = time.perf_counter() - embed_started
    if not embedded:
        return {
            "source_idx": row["source_idx"],
            "error": "embed_failed",
            "embed_s": embed_s,
        }

    mmff_code = None
    optimize_s = 0.0
    mmff_iters = PROTOCOLS[protocol]
    if mmff_iters:
        optimize_started = time.perf_counter()
        try:
            mmff_code = int(
                AllChem.MMFFOptimizeMolecule(molecule, maxIters=mmff_iters)
            )
        except Exception:
            mmff_code = -2
        optimize_s = time.perf_counter() - optimize_started

    graph_started = time.perf_counter()
    conformer = molecule.GetConformer()
    graph = Data(
        z=torch.tensor(
            [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
            dtype=torch.long,
        ),
        pos=torch.tensor(conformer.GetPositions(), dtype=torch.float32),
        charges=torch.tensor(
            compute_gasteiger_charges(molecule), dtype=torch.float32
        ),
        y=torch.tensor(row["targets"], dtype=torch.float32).view(1, 3),
        source_idx=torch.tensor([row["source_idx"]], dtype=torch.long),
    )
    graph_s = time.perf_counter() - graph_started
    return {
        "source_idx": row["source_idx"],
        "graph": graph,
        "embed_s": embed_s,
        "optimize_s": optimize_s,
        "graph_s": graph_s,
        "total_s": time.perf_counter() - started,
        "mmff_code": mmff_code,
    }


def build_protocol_cache(
    rows: list[dict],
    protocol: str,
    output_dir: Path,
    *,
    workers: int,
    shard_size: int = 1_000,
    base_seed: int = 42,
) -> dict:
    """Build a resumable graph cache and retain per-stage construction timings."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}")
    if workers <= 0 or shard_size <= 0:
        raise ValueError("workers and shard_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for shard_number, begin in enumerate(range(0, len(rows), shard_size)):
            shard_rows = rows[begin : begin + shard_size]
            graph_path = output_dir / f"graphs-{shard_number:03d}.pt"
            report_path = output_dir / f"graphs-{shard_number:03d}.json"
            if graph_path.exists() and report_path.exists():
                report = json.loads(report_path.read_text(encoding="utf-8"))
                if report.get("complete") and report.get("requested") == len(shard_rows):
                    reports.append(report)
                    continue
            wall_started = time.perf_counter()
            work = ((row, protocol, base_seed) for row in shard_rows)
            results = list(executor.map(_build_one, work, chunksize=10))
            successes = [result for result in results if "graph" in result]
            graphs = [result.pop("graph") for result in successes]
            _atomic_torch(graph_path, graphs)
            mmff_counts: dict[str, int] = {}
            for result in successes:
                key = str(result["mmff_code"])
                mmff_counts[key] = mmff_counts.get(key, 0) + 1
            report = {
                "complete": True,
                "protocol": protocol,
                "shard": shard_number,
                "requested": len(shard_rows),
                "succeeded": len(successes),
                "failed": len(results) - len(successes),
                "wall_s": time.perf_counter() - wall_started,
                "worker_cpu_s": {
                    field: sum(float(result.get(field, 0.0)) for result in results)
                    for field in ("embed_s", "optimize_s", "graph_s", "total_s")
                },
                "mmff_return_codes": mmff_counts,
                "failure_source_idx": [
                    int(result["source_idx"])
                    for result in results
                    if "graph" not in result
                ],
                "graph_path": graph_path.name,
                "graph_sha256": _sha256(graph_path),
            }
            _atomic_json(report_path, report)
            reports.append(report)
            print(
                f"{protocol} shard={shard_number:03d} "
                f"rows={len(successes)}/{len(shard_rows)} "
                f"wall={report['wall_s']:.1f}s",
                flush=True,
            )
    summary = _summarize_build(protocol, reports, workers)
    _atomic_json(output_dir / "summary.json", summary)
    return summary


def _summarize_build(protocol: str, reports: list[dict], workers: int) -> dict:
    requested = sum(report["requested"] for report in reports)
    succeeded = sum(report["succeeded"] for report in reports)
    wall_s = sum(report["wall_s"] for report in reports)
    cpu = {
        field: sum(report["worker_cpu_s"][field] for report in reports)
        for field in ("embed_s", "optimize_s", "graph_s", "total_s")
    }
    codes: dict[str, int] = {}
    for report in reports:
        for code, count in report["mmff_return_codes"].items():
            codes[code] = codes.get(code, 0) + count
    return {
        "format": "molgap-conformer-build-summary-v1",
        "protocol": protocol,
        "mmff_iters": PROTOCOLS[protocol],
        "requested": requested,
        "succeeded": succeeded,
        "failed": requested - succeeded,
        "workers": workers,
        "wall_s": wall_s,
        "molecules_per_second": succeeded / wall_s,
        "worker_cpu_s": cpu,
        "mean_worker_total_s": cpu["total_s"] / max(succeeded, 1),
        "mmff_return_codes": codes,
        "complete": requested > 0 and requested == succeeded,
    }


def load_graph_cache(path: Path, accepted: set[int] | None = None) -> list:
    graphs = []
    for part in sorted(path.glob("graphs-*.pt")):
        for graph in torch.load(part, map_location="cpu", weights_only=False):
            source_idx = int(graph.source_idx.view(-1)[0])
            if accepted is None or source_idx in accepted:
                graphs.append(graph)
    graphs.sort(key=lambda graph: int(graph.source_idx.view(-1)[0]))
    return graphs


def _load_schnet(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["model_config"] != MODEL_CONFIG:
        raise ValueError(f"unexpected SchNet config in {checkpoint_path}")
    model = SchNetWrapper(**checkpoint["model_config"], use_charges=True).to(device)
    model.load_state_dict(checkpoint["model"])
    return (
        model,
        checkpoint["target_mean"].float().to(device),
        checkpoint["target_std"].float().to(device),
    )


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    values = (prediction.float() - target.float()).abs().mean(dim=0)
    return {
        **{name: float(values[index]) for index, name in enumerate(TARGET_NAMES)},
        "average": float(values.mean()),
    }


def _subset_expert(payload_path: Path, source_indices: torch.Tensor) -> torch.Tensor:
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)["test"]
    lookup = {
        int(source_idx): index
        for index, source_idx in enumerate(payload["source_idx"].long().tolist())
    }
    missing = [int(value) for value in source_indices if int(value) not in lookup]
    if missing:
        raise ValueError(f"{payload_path.name} misses {len(missing)} selected rows")
    order = torch.tensor(
        [lookup[int(value)] for value in source_indices], dtype=torch.long
    )
    return payload["embeddings"][order].float()


@torch.no_grad()
def evaluate_frozen_route_b(
    protocol_dirs: dict[str, Path],
    *,
    gps9_payload: Path,
    gps11_payload: Path,
    accepted_primary_payload: Path | None,
    accepted_augmented_payload: Path | None,
    primary_checkpoint: Path,
    augmented_checkpoint: Path,
    head_checkpoints: Iterable[Path],
    output_path: Path,
    batch_size: int = 128,
) -> dict:
    """Evaluate identical frozen encoders/heads while changing only coordinates."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head_paths = list(head_checkpoints)
    if not head_paths:
        raise ValueError("at least one Route B head checkpoint is required")
    models = {
        "schnet_primary": _load_schnet(primary_checkpoint, device),
        "schnet_augmented": _load_schnet(augmented_checkpoint, device),
    }
    selection = json.loads(
        (output_path.parent / "selection.json").read_text(encoding="utf-8")
    )
    role_by_source = {
        int(row["source_idx"]): row["role"] for row in selection["items"]
    }
    protocol_payloads = {}
    common_indices = None
    for protocol, cache_dir in protocol_dirs.items():
        if protocol not in PROTOCOLS:
            raise ValueError(f"unknown protocol {protocol!r}")
        graphs = load_graph_cache(cache_dir)
        test_graphs = [
            graph
            for graph in graphs
            if role_by_source[int(graph.source_idx.view(-1)[0])] == "test"
        ]
        indices = torch.tensor(
            [int(graph.source_idx.view(-1)[0]) for graph in test_graphs],
            dtype=torch.long,
        )
        if common_indices is None:
            common_indices = indices
        elif not torch.equal(indices, common_indices):
            raise ValueError("protocol test source_idx alignment differs")
        targets = torch.cat([graph.y.view(1, 3) for graph in test_graphs]).float()
        encoded = {}
        standalone = {}
        for name, (model, mean, std) in models.items():
            view = evaluate_view(model, test_graphs, batch_size, device, mean, std)
            if not torch.equal(view["source_idx"], indices):
                raise ValueError(f"{name} source_idx order differs")
            encoded[name] = view["embeddings"].float()
            standalone[name] = _metrics(view["predictions"], targets)
        protocol_payloads[protocol] = {
            "source_idx": indices,
            "targets": targets,
            "embeddings": encoded,
            "standalone_schnet": standalone,
        }
    assert common_indices is not None
    gps = {
        "gps9": _subset_expert(gps9_payload, common_indices),
        "gps11_160": _subset_expert(gps11_payload, common_indices),
    }
    if (accepted_primary_payload is None) != (accepted_augmented_payload is None):
        raise ValueError("both accepted SchNet payloads must be provided together")
    if accepted_primary_payload is not None:
        reference = protocol_payloads[next(iter(protocol_dirs))]
        protocol_payloads["accepted_current_mmff"] = {
            "source_idx": common_indices,
            "targets": reference["targets"],
            "embeddings": {
                "schnet_primary": _subset_expert(
                    accepted_primary_payload, common_indices
                ),
                "schnet_augmented": _subset_expert(
                    accepted_augmented_payload, common_indices
                ),
            },
            "standalone_schnet": None,
        }
    result_protocols = {}
    ensemble_predictions = {}
    for protocol, payload in protocol_payloads.items():
        inputs = {**gps, **payload["embeddings"]}
        seed_predictions = []
        seed_metrics = []
        for checkpoint_path in head_paths:
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            head = build_fusion_head(
                checkpoint["fusion_mode"],
                checkpoint["input_dims"],
                checkpoint["hidden"],
                checkpoint["correction_scale_eV"],
            ).to(device)
            head.load_state_dict(checkpoint["model"])
            head.eval()
            prediction = head(
                {name: value.to(device) for name, value in inputs.items()}
            ).cpu()
            seed_predictions.append(prediction)
            seed_metrics.append(
                {
                    "checkpoint": str(checkpoint_path),
                    "metrics": _metrics(prediction, payload["targets"]),
                }
            )
        ensemble = torch.stack(seed_predictions).mean(dim=0)
        ensemble_predictions[protocol] = ensemble
        result_protocols[protocol] = {
            "rows": len(common_indices),
            "standalone_schnet": payload["standalone_schnet"],
            "route_b_by_seed": seed_metrics,
            "route_b_seed_metric_mean": {
                key: sum(item["metrics"][key] for item in seed_metrics)
                / len(seed_metrics)
                for key in (*TARGET_NAMES, "average")
            },
            "route_b_equal_seed_ensemble": _metrics(
                ensemble, payload["targets"]
            ),
        }
    paired_comparisons = {}
    for candidate in ("etkdgv3_mmff200", "accepted_current_mmff"):
        if candidate not in ensemble_predictions:
            continue
        target = protocol_payloads["bare_etkdg"]["targets"]
        error_delta = (
            (ensemble_predictions[candidate] - target).abs()
            - (ensemble_predictions["bare_etkdg"] - target).abs()
        )
        molecule_delta = error_delta.mean(dim=1)
        standard_error = float(
            molecule_delta.std(unbiased=True) / len(molecule_delta) ** 0.5
        )
        paired_comparisons[f"{candidate}_minus_bare_etkdg"] = {
            "rows": len(molecule_delta),
            "mean_average_absolute_error_delta_eV": float(molecule_delta.mean()),
            "normal_95pct_ci_eV": [
                float(molecule_delta.mean()) - 1.96 * standard_error,
                float(molecule_delta.mean()) + 1.96 * standard_error,
            ],
            "fraction_molecules_improved": float(
                (molecule_delta < 0).float().mean()
            ),
            "fraction_molecules_improved_by_gt_1e-4_eV": float(
                (molecule_delta < -1e-4).float().mean()
            ),
            "fraction_molecules_degraded_by_gt_1e-4_eV": float(
                (molecule_delta > 1e-4).float().mean()
            ),
            "target_mae_delta_eV": {
                name: float(error_delta[:, index].mean())
                for index, name in enumerate(TARGET_NAMES)
            },
        }
    result = {
        "format": "molgap-conformer-frozen-route-b-ab-v1",
        "evaluation_role": "test",
        "device": str(device),
        "frozen_architecture": [
            "GPS9-192",
            "GPS11-160",
            "SchNet-176/160/6-primary",
            "SchNet-176/160/6-two-conformer-trained-primary-view",
            "bounded-residual-head-0.10eV",
        ],
        "protocols": result_protocols,
        "paired_comparisons": paired_comparisons,
    }
    _atomic_json(output_path, result)
    return result


def analyze_tradeoff(
    build_summaries: dict[str, dict],
    evaluation: dict,
    output_path: Path,
) -> dict:
    bare = build_summaries["bare_etkdg"]
    mmff = build_summaries["etkdgv3_mmff200"]
    bare_metric = evaluation["protocols"]["bare_etkdg"][
        "route_b_equal_seed_ensemble"
    ]
    mmff_metric = evaluation["protocols"]["etkdgv3_mmff200"][
        "route_b_equal_seed_ensemble"
    ]
    extra_per_molecule = (
        mmff["wall_s"] / mmff["succeeded"]
        - bare["wall_s"] / bare["succeeded"]
    )
    deltas = {
        key: mmff_metric[key] - bare_metric[key]
        for key in (*TARGET_NAMES, "average")
    }
    relative_average_improvement = -deltas["average"] / bare_metric["average"]
    result = {
        "format": "molgap-conformer-tradeoff-v1",
        "speed": {
            "bare_wall_s_50k": bare["wall_s"],
            "mmff_wall_s_50k": mmff["wall_s"],
            "mmff_to_bare_ratio": mmff["wall_s"] / bare["wall_s"],
            "extra_s_per_molecule": extra_per_molecule,
            "extrapolated_extra_hours": {
                "100k": extra_per_molecule * 100_000 / 3600,
                "1m": extra_per_molecule * 1_000_000 / 3600,
            },
        },
        "accuracy": {
            "bare": bare_metric,
            "mmff": mmff_metric,
            "mmff_minus_bare_mae_eV": deltas,
            "mmff_relative_average_mae_improvement": relative_average_improvement,
            "paired_test": evaluation.get("paired_comparisons", {}).get(
                "etkdgv3_mmff200_minus_bare_etkdg"
            ),
        },
        "interpretation": (
            "Negative MAE deltas favor MMFF; positive deltas favor bare ETKDG. "
            "This is a frozen-checkpoint inference A/B, not a retrained ceiling test."
        ),
    }
    _atomic_json(output_path, result)
    return result
