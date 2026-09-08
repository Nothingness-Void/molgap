"""Local atom/bond/functional-group supervision for the PCQM Gap100K EdgeState model."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional


PARENT_GEOMETRY_CACHE_SHA256 = (
    "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
)
PARENT_GRAPH_CACHE_SHA256 = (
    "eb7c843e33f430ac755bc575d80153aba87677cea1ad5bb0dcf73cca906e2c21"
)
TRAIN_GRAPHS = 100_000
VALIDATION_GRAPHS = 10_000

FUNCTIONAL_GROUP_SMARTS = {
    "aromatic": "[a]",
    "carbonyl": "[CX3]=[OX1]",
    "amide": "[NX3][CX3](=[OX1])",
    "ester": "[CX3](=[OX1])[OX2][#6]",
    "carboxyl": "[CX3](=[OX1])[OX2H1]",
    "nitrile": "[CX2]#N",
    "amine": "[NX3;H0,H1,H2;!$(NC=O)]",
    "alcohol": "[OX2H][#6]",
    "ether": "[OD2]([#6])[#6]",
    "alkene": "[CX3]=[CX3]",
    "alkyne": "[CX2]#[CX2]",
    "ring_heteroatom": "[n,o,s;r]",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _torch_load(path: Path, *, map_location="cpu", weights_only=False):
    """Load on both upstream Torch and older DTK forks."""
    import torch

    try:
        return torch.load(
            path, map_location=map_location, weights_only=weights_only
        )
    except TypeError:
        return torch.load(path, map_location=map_location)


def functional_group_contract_sha256() -> str:
    payload = json.dumps(
        FUNCTIONAL_GROUP_SMARTS, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _functional_group_labels(smiles: str):
    import torch
    from rdkit import Chem

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("PCQM SMILES did not sanitize")
    labels = torch.zeros(
        (molecule.GetNumAtoms(), len(FUNCTIONAL_GROUP_SMARTS)),
        dtype=torch.uint8,
    )
    for column, smarts in enumerate(FUNCTIONAL_GROUP_SMARTS.values()):
        query = Chem.MolFromSmarts(smarts)
        if query is None:
            raise RuntimeError(f"Invalid frozen SMARTS: {smarts}")
        for match in molecule.GetSubstructMatches(query):
            labels[list(match), column] = 1
    return labels


def _verify_parent_manifest(manifest: dict) -> None:
    required = {
        "format": "molgap-pcqm-gap100k-etkdg-geometry-cache-v1",
        "complete": True,
        "aggregate_sha256": PARENT_GEOMETRY_CACHE_SHA256,
        "parent_graph_cache_aggregate_sha256": PARENT_GRAPH_CACHE_SHA256,
        "train_graphs": TRAIN_GRAPHS,
        "validation_graphs": VALIDATION_GRAPHS,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    checks = {key: manifest.get(key) == value for key, value in required.items()}
    if not all(checks.values()):
        raise RuntimeError(f"Parent PCQM cache contract changed: {checks}")


def build_local_label_cache(
    source_csv: Path,
    parent_cache_root: Path,
    output_root: Path,
    *,
    source_commit: str,
) -> dict:
    """Build a compact sidecar aligned to the accepted 100K/10K graph cache."""
    import torch

    from .pcqm_gap_data import read_official_train_prefix

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is True:
            return manifest
        raise RuntimeError("Incomplete final local-label manifest exists")

    parent_manifest_path = parent_cache_root / "manifest.json"
    parent = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    _verify_parent_manifest(parent)
    frame = read_official_train_prefix(source_csv)
    shards = []
    role_counts = {"train": 0, "validation": 0}
    positive_counts = torch.zeros(len(FUNCTIONAL_GROUP_SMARTS), dtype=torch.long)
    atom_rows = 0
    for parent_shard in parent["shards"]:
        role = str(parent_shard["role"])
        parent_path = parent_cache_root / parent_shard["file"]
        if sha256_file(parent_path) != parent_shard["sha256"]:
            raise RuntimeError(f"Parent shard hash changed: {parent_path.name}")
        graphs = _torch_load(parent_path, map_location="cpu", weights_only=False)
        entries = []
        for graph in graphs:
            row_index = int(graph.row_index.view(-1)[0])
            row = frame.iloc[row_index]
            if int(row.idx) != row_index:
                raise RuntimeError("PCQM row identity changed")
            labels = _functional_group_labels(str(row.smiles))
            if labels.shape[0] != graph.x.shape[0]:
                raise RuntimeError(
                    f"SMILES/graph atom alignment changed at row {row_index}"
                )
            entries.append({"row_index": row_index, "functional_group_y": labels})
            positive_counts += labels.sum(dim=0).long()
            atom_rows += int(labels.shape[0])
        output_name = parent_path.stem + "-local-labels.pt"
        output_path = output_root / output_name
        _atomic_torch_save(output_path, entries)
        record = {
            "role": role,
            "file": output_name,
            "parent_file": parent_path.name,
            "parent_sha256": parent_shard["sha256"],
            "graph_count": len(entries),
            "sha256": sha256_file(output_path),
        }
        shards.append(record)
        role_counts[role] += len(entries)
        atomic_json(
            output_root / "progress.json",
            {
                "format": "molgap-pcqm-gap100k-local-label-progress-v1",
                "complete": False,
                "source_commit": source_commit,
                "completed_shards": shards,
                "role_counts": role_counts,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
                "model_inference_executed": False,
            },
        )
        del graphs, entries

    if role_counts != {"train": TRAIN_GRAPHS, "validation": VALIDATION_GRAPHS}:
        raise RuntimeError(f"Local-label role counts are incomplete: {role_counts}")
    aggregate = hashlib.sha256()
    for item in shards:
        aggregate.update(
            f"{item['role']}\t{item['file']}\t{item['sha256']}\n".encode("ascii")
        )
    manifest = {
        "format": "molgap-pcqm-gap100k-local-label-cache-v1",
        "complete": True,
        "source_commit": source_commit,
        "parent_geometry_cache_aggregate_sha256": PARENT_GEOMETRY_CACHE_SHA256,
        "parent_graph_cache_aggregate_sha256": PARENT_GRAPH_CACHE_SHA256,
        "functional_group_names": list(FUNCTIONAL_GROUP_SMARTS),
        "functional_group_smarts": FUNCTIONAL_GROUP_SMARTS,
        "functional_group_contract_sha256": functional_group_contract_sha256(),
        "functional_group_positive_counts": positive_counts.tolist(),
        "atom_rows": atom_rows,
        "train_graphs": role_counts["train"],
        "validation_graphs": role_counts["validation"],
        "shards": shards,
        "aggregate_sha256": aggregate.hexdigest(),
        "gpu_used": False,
        "model_inference_executed": False,
        "gap_labels_used_for_auxiliary_targets": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "molecular_research_server_accessed": False,
    }
    atomic_json(manifest_path, manifest)
    return manifest


def accept_local_label_cache(root: Path, *, expected_source_commit: str) -> dict:
    """Mechanically validate the sidecar without running a model."""
    import torch

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    checks = {
        "format": manifest.get("format")
        == "molgap-pcqm-gap100k-local-label-cache-v1",
        "complete": manifest.get("complete") is True,
        "source_commit": manifest.get("source_commit") == expected_source_commit,
        "parent_geometry": manifest.get("parent_geometry_cache_aggregate_sha256")
        == PARENT_GEOMETRY_CACHE_SHA256,
        "parent_graph": manifest.get("parent_graph_cache_aggregate_sha256")
        == PARENT_GRAPH_CACHE_SHA256,
        "smarts": manifest.get("functional_group_contract_sha256")
        == functional_group_contract_sha256(),
        "train_graphs": manifest.get("train_graphs") == TRAIN_GRAPHS,
        "validation_graphs": manifest.get("validation_graphs")
        == VALIDATION_GRAPHS,
        "gpu_unused": manifest.get("gpu_used") is False,
        "no_model_inference": manifest.get("model_inference_executed") is False,
        "gap_not_auxiliary": manifest.get("gap_labels_used_for_auxiliary_targets")
        is False,
        "official_validation_sealed": manifest.get("official_validation_role_read")
        is False,
        "test_dev_sealed": manifest.get("test_dev_role_read") is False,
        "ims_unaccessed": manifest.get("molecular_research_server_accessed")
        is False,
    }
    aggregate = hashlib.sha256()
    observed = {"train": 0, "validation": 0}
    shard_checks = []
    for item in manifest.get("shards", []):
        path = root / item["file"]
        file_hash = sha256_file(path) if path.is_file() else None
        entries = (
            _torch_load(path, map_location="cpu", weights_only=False)
            if path.is_file()
            else []
        )
        valid_entries = len(entries) == int(item["graph_count"])
        for entry in entries:
            labels = entry.get("functional_group_y")
            valid_entries = valid_entries and (
                labels is not None
                and labels.ndim == 2
                and labels.shape[1] == len(FUNCTIONAL_GROUP_SMARTS)
                and labels.dtype == torch.uint8
                and bool(((labels == 0) | (labels == 1)).all())
            )
        ok = file_hash == item.get("sha256") and valid_entries
        shard_checks.append(ok)
        observed[item["role"]] += len(entries)
        aggregate.update(
            f"{item['role']}\t{item['file']}\t{item['sha256']}\n".encode("ascii")
        )
    checks["all_shards"] = bool(shard_checks) and all(shard_checks)
    checks["role_counts"] = observed == {
        "train": TRAIN_GRAPHS,
        "validation": VALIDATION_GRAPHS,
    }
    checks["aggregate"] = aggregate.hexdigest() == manifest.get("aggregate_sha256")
    acceptance = {
        "format": "molgap-pcqm-gap100k-local-label-acceptance-v1",
        "accepted": all(checks.values()),
        "checks": checks,
        "source_commit": manifest.get("source_commit"),
        "aggregate_sha256": manifest.get("aggregate_sha256"),
        "train_graphs": observed["train"],
        "validation_graphs": observed["validation"],
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(root / "acceptance.json", acceptance)
    if not acceptance["accepted"]:
        raise RuntimeError(f"Local-label cache acceptance failed: {checks}")
    return acceptance


# The paired GPU screen changes only the training objective. The inference model
# and the data roles remain the accepted PCQM Gap100K EdgeState contract.
MODEL_SEED = 42
BATCH_SIZE = 48
LEARNING_RATE = 1.6e-4
WEIGHT_DECAY = 1e-6
SCRATCH_EPOCHS = 40
PRETRAIN_EPOCHS = 20
FINETUNE_EPOCHS = 20
MASK_RATE = 0.15
EXPECTED_MODEL_PARAMETERS = 4_771_073
MIN_PAIRED_GAIN_EV = 0.003


def _state_sha256(model) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _capture_rng_state(*, shuffle_generator, mask_generator=None) -> dict:
    import random
    import numpy as np
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "shuffle_generator": shuffle_generator.get_state(),
    }
    if mask_generator is not None:
        state["mask_generator"] = mask_generator.get_state()
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict, *, shuffle_generator, mask_generator=None) -> None:
    import random
    import numpy as np
    import torch

    required = {"python", "numpy", "torch", "shuffle_generator"}
    if not required.issubset(state):
        raise RuntimeError("Resume checkpoint is missing RNG state")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    shuffle_generator.set_state(state["shuffle_generator"])
    if mask_generator is not None:
        if "mask_generator" not in state:
            raise RuntimeError("Pretraining checkpoint is missing mask RNG state")
        mask_generator.set_state(state["mask_generator"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _find_cache(
    format_name: str, explicit_root: Optional[Path] = None
) -> tuple[Path, dict]:
    if explicit_root is not None:
        root = explicit_root.resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("format") != format_name:
            raise RuntimeError(
                f"Expected {format_name} at {root}, found {manifest.get('format')}"
            )
        return root, manifest
    candidates = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("format") == format_name:
            candidates.append((path.parent, manifest))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {format_name} cache, found {candidates}")
    return candidates[0]


def load_training_roles(
    expected_label_sha256: str,
    *,
    graph_root: Optional[Path] = None,
    label_root: Optional[Path] = None,
):
    """Load and align the immutable graph cache and compact label sidecar."""
    import torch

    graph_root, graph_manifest = _find_cache(
        "molgap-pcqm-gap100k-etkdg-geometry-cache-v1", graph_root
    )
    _verify_parent_manifest(graph_manifest)
    label_root, label_manifest = _find_cache(
        "molgap-pcqm-gap100k-local-label-cache-v1", label_root
    )
    if label_manifest.get("aggregate_sha256") != expected_label_sha256:
        raise RuntimeError("Local-label cache identity changed")
    if (
        label_manifest.get("parent_geometry_cache_aggregate_sha256")
        != PARENT_GEOMETRY_CACHE_SHA256
        or label_manifest.get("functional_group_contract_sha256")
        != functional_group_contract_sha256()
    ):
        raise RuntimeError("Local-label parent or SMARTS identity changed")
    graph_by_name = {item["file"]: item for item in graph_manifest["shards"]}
    labels_by_parent = {item["parent_file"]: item for item in label_manifest["shards"]}
    if set(graph_by_name) != set(labels_by_parent):
        raise RuntimeError("Graph and local-label shard sets differ")
    roles = {"train": [], "validation": []}
    for graph_name, graph_item in graph_by_name.items():
        label_item = labels_by_parent[graph_name]
        graph_path = graph_root / graph_name
        label_path = label_root / label_item["file"]
        if sha256_file(graph_path) != graph_item["sha256"]:
            raise RuntimeError(f"Graph shard hash changed: {graph_name}")
        if sha256_file(label_path) != label_item["sha256"]:
            raise RuntimeError(f"Local-label shard hash changed: {label_path.name}")
        graphs = _torch_load(graph_path, map_location="cpu", weights_only=False)
        labels = _torch_load(label_path, map_location="cpu", weights_only=False)
        if len(graphs) != len(labels):
            raise RuntimeError(f"Sidecar length changed: {graph_name}")
        for graph, label in zip(graphs, labels):
            row_index = int(graph.row_index.view(-1)[0])
            if row_index != int(label["row_index"]):
                raise RuntimeError(f"Sidecar row alignment changed: {row_index}")
            graph.row_id = graph.row_index.clone()
            graph.functional_group_y = label["functional_group_y"].float()
            if tuple(graph.functional_group_y.shape) != (
                graph.num_nodes,
                len(FUNCTIONAL_GROUP_SMARTS),
            ):
                raise RuntimeError(f"Functional-group shape changed: {row_index}")
        roles[graph_item["role"]].extend(graphs)
    if {key: len(value) for key, value in roles.items()} != {
        "train": TRAIN_GRAPHS,
        "validation": VALIDATION_GRAPHS,
    }:
        raise RuntimeError("Loaded PCQM role counts changed")
    return roles, graph_manifest, label_manifest


def _make_encoder():
    from .pcqm_gap_architecture import make_pcqm_gap_encoder

    return make_pcqm_gap_encoder("ogb_edge_state_structural_gps9")


def _forward(model, batch):
    return model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    ).view(-1)


class LocalHierarchyHeads:
    """Training-only heads attached to the final node and real-bond states."""

    def __init__(self, model):
        import torch.nn as nn
        from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

        self.node_state = None
        self.edge_state = None
        self.atom_heads = nn.ModuleList(
            nn.Linear(192, classes) for classes in get_atom_feature_dims()
        )
        self.bond_heads = nn.ModuleList(
            nn.Linear(64, classes) for classes in get_bond_feature_dims()
        )
        self.group_head = nn.Linear(192, len(FUNCTIONAL_GROUP_SMARTS))
        self.module = nn.ModuleDict(
            {
                "atom_heads": self.atom_heads,
                "bond_heads": self.bond_heads,
                "group_head": self.group_head,
            }
        )
        self.handles = [
            model.convs[-1].register_forward_hook(self._capture_node),
            model.edge_updates[-1].register_forward_hook(self._capture_edge),
        ]

    def _capture_node(self, _module, _inputs, output):
        self.node_state = output

    def _capture_edge(self, _module, _inputs, output):
        self.edge_state = output

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _loader(graphs, *, shuffle: bool, seed: int, generator=None):
    import torch
    from torch_geometric.loader import DataLoader

    return DataLoader(
        graphs,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator or torch.Generator().manual_seed(seed),
        num_workers=0,
        pin_memory=True,
    )


def _target_stats(graphs) -> tuple[float, float]:
    import torch

    values = torch.tensor([float(graph.y.view(-1)[0]) for graph in graphs])
    return float(values.mean()), float(values.std(unbiased=False).clamp_min(1e-6))


def _evaluate(model, loader, mean, std, device):
    import torch

    model.eval()
    error = 0.0
    rows = 0
    predictions = []
    targets = []
    row_ids = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            prediction = _forward(model, batch) * std + mean
            target = batch.y.view(-1)
            error += float((prediction - target).abs().sum())
            rows += int(target.numel())
            predictions.append(prediction.cpu())
            targets.append(target.cpu())
            row_ids.append(batch.row_id.view(-1).cpu())
    return {
        "mae_eV": error / rows,
        "prediction": torch.cat(predictions),
        "target": torch.cat(targets),
        "row_id": torch.cat(row_ids),
    }


def _train_gap(
    model,
    roles,
    run_dir: Path,
    *,
    epochs: int,
    stage: str,
    initial_hash: str,
    label_sha256: str,
):
    import time
    import numpy as np
    import torch
    import torch.nn.functional as functional

    device = torch.device("cuda:0")
    model = model.to(device)
    mean, std = _target_stats(roles["train"])
    mean_tensor = torch.tensor(mean, device=device)
    std_tensor = torch.tensor(std, device=device)
    shuffle_generator = torch.Generator().manual_seed(MODEL_SEED)
    train_loader = _loader(
        roles["train"],
        shuffle=True,
        seed=MODEL_SEED,
        generator=shuffle_generator,
    )
    validation_loader = _loader(
        roles["validation"], shuffle=False, seed=MODEL_SEED
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    trace = []
    best = float("inf")
    best_epoch = -1
    start_epoch = 0
    checkpoint_path = run_dir / f"{stage}_checkpoint.pt"
    best_model_path = run_dir / f"{stage}_best_model.pt"
    if checkpoint_path.is_file():
        checkpoint = _torch_load(
            checkpoint_path, map_location=device, weights_only=False
        )
        checks = {
            "stage": checkpoint.get("stage") == stage,
            "seed": checkpoint.get("seed") == MODEL_SEED,
            "max_epochs": checkpoint.get("max_epochs") == epochs,
            "initial_hash": checkpoint.get("initial_encoder_sha256")
            == initial_hash,
            "label_cache": checkpoint.get("local_label_cache_aggregate_sha256")
            == label_sha256,
        }
        if not all(checks.values()):
            raise RuntimeError(f"Gap resume checkpoint contract changed: {checks}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        trace = list(checkpoint["trace"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if len(trace) != start_epoch or start_epoch > epochs:
            raise RuntimeError("Gap resume trace does not match checkpoint epoch")
        if trace:
            best_row = min(trace, key=lambda row: row["validation_gap_mae_eV"])
            best = float(best_row["validation_gap_mae_eV"])
            best_epoch = int(best_row["epoch"])
            if not best_model_path.is_file():
                raise RuntimeError("Gap resume checkpoint has no best model")
        _restore_rng_state(
            checkpoint.get("rng_state", {}),
            shuffle_generator=shuffle_generator,
        )
        print(f"resuming {stage} at epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        started = time.perf_counter()
        absolute = 0.0
        rows = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = _forward(model, batch)
            target = (batch.y.view(-1) - mean_tensor) / std_tensor
            loss = functional.l1_loss(prediction, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite Gap loss in {stage} epoch {epoch}")
            loss.backward()
            optimizer.step()
            absolute += float(
                (prediction.detach() * std_tensor + mean_tensor - batch.y.view(-1))
                .abs()
                .sum()
            )
            rows += int(batch.y.numel())
        scheduler.step()
        validation = _evaluate(
            model, validation_loader, mean_tensor, std_tensor, device
        )
        elapsed = time.perf_counter() - started
        improved = validation["mae_eV"] < best
        if improved:
            best = validation["mae_eV"]
            best_epoch = epoch
            _atomic_torch_save(
                best_model_path, model.state_dict()
            )
        row = {
            "epoch": epoch,
            "train_gap_mae_eV": absolute / rows,
            "validation_gap_mae_eV": validation["mae_eV"],
            "elapsed_s": elapsed,
            "graphs_per_s": TRAIN_GRAPHS / elapsed,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "improved": improved,
        }
        trace.append(row)
        atomic_json(run_dir / f"{stage}_trace.json", {"epochs": trace})
        _atomic_torch_save(
            checkpoint_path,
            {
                "stage": stage,
                "epoch": epoch,
                "max_epochs": epochs,
                "seed": MODEL_SEED,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "initial_encoder_sha256": initial_hash,
                "local_label_cache_aggregate_sha256": label_sha256,
                "trace": trace,
                "rng_state": _capture_rng_state(
                    shuffle_generator=shuffle_generator
                ),
            },
        )
        print(
            f"{stage} ep{epoch:02d} train={row['train_gap_mae_eV']:.6f} "
            f"val={validation['mae_eV']:.6f}eV {elapsed:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
    model.load_state_dict(
        torch.load(
            best_model_path,
            map_location=device,
            weights_only=True,
        )
    )
    validation = _evaluate(model, validation_loader, mean_tensor, std_tensor, device)
    payload_path = run_dir / f"{stage}_validation_payload.pt"
    _atomic_torch_save(
        payload_path,
        {
            "prediction_eV": validation["prediction"],
            "target_eV": validation["target"],
            "row_id": validation["row_id"],
            "stage": stage,
            "seed": MODEL_SEED,
            "local_label_cache_aggregate_sha256": label_sha256,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        },
    )
    return model, {
        "best_epoch": best_epoch,
        "best_validation_gap_mae_eV": validation["mae_eV"],
        "epochs_completed": len(trace),
        "mean_epoch_seconds": float(np.mean([row["elapsed_s"] for row in trace])),
        "mean_graphs_per_s": float(np.mean([row["graphs_per_s"] for row in trace])),
        "best_model_sha256": sha256_file(best_model_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_payload_sha256": sha256_file(payload_path),
    }


def _undirected_edge_mask(batch, generator):
    import torch

    if batch.edge_index.shape[1] == 0:
        return torch.zeros(0, dtype=torch.bool)
    source, target = batch.edge_index.cpu()
    node_count = int(batch.x.shape[0])
    keys = torch.minimum(source, target) * node_count + torch.maximum(source, target)
    _, inverse = torch.unique(keys, sorted=False, return_inverse=True)
    selected = torch.rand(int(inverse.max()) + 1, generator=generator) < MASK_RATE
    mask = selected[inverse]
    if not bool(mask.any()):
        mask[inverse == inverse[0]] = True
    return mask


def _pretrain(model, roles, run_dir: Path, *, initial_hash: str, label_sha256: str):
    import time
    import numpy as np
    import torch
    import torch.nn.functional as functional

    device = torch.device("cuda:0")
    model = model.to(device)
    heads = LocalHierarchyHeads(model)
    heads.module = heads.module.to(device)
    parameters = list(model.parameters()) + list(heads.module.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=PRETRAIN_EPOCHS, eta_min=1e-6
    )
    shuffle_generator = torch.Generator().manual_seed(MODEL_SEED)
    loader = _loader(
        roles["train"],
        shuffle=True,
        seed=MODEL_SEED,
        generator=shuffle_generator,
    )
    mask_generator = torch.Generator().manual_seed(MODEL_SEED + 17)
    trace = []
    start_epoch = 0
    checkpoint_path = run_dir / "pretrain_checkpoint.pt"
    if checkpoint_path.is_file():
        checkpoint = _torch_load(
            checkpoint_path, map_location=device, weights_only=False
        )
        checks = {
            "stage": checkpoint.get("stage") == "local_hierarchy_pretrain",
            "seed": checkpoint.get("seed") == MODEL_SEED,
            "max_epochs": checkpoint.get("max_epochs") == PRETRAIN_EPOCHS,
            "initial_hash": checkpoint.get("initial_encoder_sha256")
            == initial_hash,
            "label_cache": checkpoint.get("local_label_cache_aggregate_sha256")
            == label_sha256,
        }
        if not all(checks.values()):
            raise RuntimeError(
                f"Pretraining resume checkpoint contract changed: {checks}"
            )
        model.load_state_dict(checkpoint["model"])
        heads.module.load_state_dict(checkpoint["heads"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        trace = list(checkpoint["trace"])
        start_epoch = int(checkpoint["epoch"]) + 1
        if len(trace) != start_epoch or start_epoch > PRETRAIN_EPOCHS:
            raise RuntimeError(
                "Pretraining resume trace does not match checkpoint epoch"
            )
        _restore_rng_state(
            checkpoint.get("rng_state", {}),
            shuffle_generator=shuffle_generator,
            mask_generator=mask_generator,
        )
        print(f"resuming local hierarchy at epoch {start_epoch}", flush=True)
    for epoch in range(start_epoch, PRETRAIN_EPOCHS):
        model.train()
        heads.module.train()
        totals = {"loss": 0.0, "atom": 0.0, "bond": 0.0, "group": 0.0}
        batches = 0
        started = time.perf_counter()
        for batch in loader:
            original_x = batch.x.clone()
            original_edge = batch.edge_attr.clone()
            node_mask = torch.rand(
                batch.x.shape[0], generator=mask_generator
            ) < MASK_RATE
            if not bool(node_mask.any()):
                node_mask[0] = True
            edge_mask = _undirected_edge_mask(batch, mask_generator)
            batch.x[node_mask] = 0
            if edge_mask.numel():
                batch.edge_attr[edge_mask] = 0
            batch = batch.to(device, non_blocking=True)
            node_mask = node_mask.to(device)
            edge_mask = edge_mask.to(device)
            original_x = original_x.to(device)
            original_edge = original_edge.to(device)
            optimizer.zero_grad(set_to_none=True)
            _forward(model, batch)
            atom_loss = sum(
                functional.cross_entropy(
                    head(heads.node_state[node_mask]), original_x[node_mask, column]
                )
                for column, head in enumerate(heads.atom_heads)
            ) / len(heads.atom_heads)
            if edge_mask.numel() and bool(edge_mask.any()):
                bond_loss = sum(
                    functional.cross_entropy(
                        head(heads.edge_state[edge_mask]),
                        original_edge[edge_mask, column],
                    )
                    for column, head in enumerate(heads.bond_heads)
                ) / len(heads.bond_heads)
            else:
                bond_loss = atom_loss.new_zeros(())
            group_loss = functional.binary_cross_entropy_with_logits(
                heads.group_head(heads.node_state[node_mask]),
                batch.functional_group_y[node_mask],
            )
            loss = atom_loss + bond_loss + 0.5 * group_loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite hierarchy loss at epoch {epoch}")
            loss.backward()
            optimizer.step()
            for key, value in (
                ("loss", loss),
                ("atom", atom_loss),
                ("bond", bond_loss),
                ("group", group_loss),
            ):
                totals[key] += float(value.detach())
            batches += 1
        scheduler.step()
        elapsed = time.perf_counter() - started
        row = {
            "epoch": epoch,
            **{key: value / batches for key, value in totals.items()},
            "elapsed_s": elapsed,
            "graphs_per_s": TRAIN_GRAPHS / elapsed,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        trace.append(row)
        atomic_json(run_dir / "pretrain_trace.json", {"epochs": trace})
        _atomic_torch_save(
            run_dir / "pretrain_checkpoint.pt",
            {
                "stage": "local_hierarchy_pretrain",
                "epoch": epoch,
                "max_epochs": PRETRAIN_EPOCHS,
                "seed": MODEL_SEED,
                "model": model.state_dict(),
                "heads": heads.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "initial_encoder_sha256": initial_hash,
                "local_label_cache_aggregate_sha256": label_sha256,
                "gap_labels_read": False,
                "trace": trace,
                "rng_state": _capture_rng_state(
                    shuffle_generator=shuffle_generator,
                    mask_generator=mask_generator,
                ),
            },
        )
        print(
            f"local_hierarchy ep{epoch:02d} loss={row['loss']:.6f} "
            f"atom={row['atom']:.6f} bond={row['bond']:.6f} "
            f"group={row['group']:.6f} {elapsed:.1f}s",
            flush=True,
        )
    heads.close()
    result = {
        "epochs_completed": len(trace),
        "mean_epoch_seconds": float(np.mean([row["elapsed_s"] for row in trace])),
        "mean_graphs_per_s": float(np.mean([row["graphs_per_s"] for row in trace])),
        "training_head_parameters": sum(
            parameter.numel() for parameter in heads.module.parameters()
        ),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "final_encoder_sha256": _state_sha256(model),
        "gap_labels_read": False,
    }
    del heads
    return model, result


def run_worker(
    role: str,
    output_root: Path,
    *,
    label_sha256: str,
    source_commit: str,
    graph_root: Optional[Path] = None,
    label_root: Optional[Path] = None,
):
    import torch

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Worker {role} sees {torch.cuda.device_count()} GPUs")
    _set_seed(MODEL_SEED)
    roles, graph_manifest, label_manifest = load_training_roles(
        label_sha256, graph_root=graph_root, label_root=label_root
    )
    model = _make_encoder()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_MODEL_PARAMETERS:
        raise RuntimeError(f"EdgeState parameter count changed: {parameter_count}")
    initial_hash = _state_sha256(model)
    run_dir = output_root / role
    run_dir.mkdir(parents=True, exist_ok=True)
    if role == "scratch":
        _, gap = _train_gap(
            model,
            roles,
            run_dir,
            epochs=SCRATCH_EPOCHS,
            stage="scratch_gap",
            initial_hash=initial_hash,
            label_sha256=label_sha256,
        )
        result = {"gap": gap}
    elif role == "pretrained":
        model, pretraining = _pretrain(
            model,
            roles,
            run_dir,
            initial_hash=initial_hash,
            label_sha256=label_sha256,
        )
        _, gap = _train_gap(
            model,
            roles,
            run_dir,
            epochs=FINETUNE_EPOCHS,
            stage="finetune_gap",
            initial_hash=initial_hash,
            label_sha256=label_sha256,
        )
        result = {"pretraining": pretraining, "gap": gap}
    else:
        raise ValueError(f"Unknown worker role: {role}")
    payload = {
        "format": "molgap-pcqm-gap100k-edgestate-local-hierarchy-worker-v1",
        "complete": True,
        "role": role,
        "source_commit": source_commit,
        "seed": MODEL_SEED,
        "architecture": "ogb_edge_state_structural_gps9",
        "parameter_count": parameter_count,
        "initial_encoder_sha256": initial_hash,
        "parent_geometry_cache_aggregate_sha256": graph_manifest["aggregate_sha256"],
        "local_label_cache_aggregate_sha256": label_manifest["aggregate_sha256"],
        "encoder_sample_exposure_epochs": SCRATCH_EPOCHS,
        "gap_training_epochs": SCRATCH_EPOCHS if role == "scratch" else FINETUNE_EPOCHS,
        "pretraining_epochs": 0 if role == "scratch" else PRETRAIN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "precision": "FP32",
        "execution_platform": os.environ.get(
            "MOLGAP_EXECUTION_PLATFORM", "unspecified"
        ),
        "accelerator_name": torch.cuda.get_device_name(0),
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "molecular_research_server_accessed": False,
        **result,
    }
    atomic_json(run_dir / "metrics.json", payload)
    return payload


def finalize_paired_outputs(
    output_root: Path,
    *,
    label_sha256: str,
    source_commit: str,
    elapsed_s: Optional[float] = None,
):
    """Combine independently durable workers without running either encoder."""
    scratch = json.loads((output_root / "scratch" / "metrics.json").read_text())
    pretrained = json.loads(
        (output_root / "pretrained" / "metrics.json").read_text()
    )
    checks = {
        "source_commit": all(
            item.get("source_commit") == source_commit
            for item in (scratch, pretrained)
        ),
        "label_cache": all(
            item.get("local_label_cache_aggregate_sha256") == label_sha256
            for item in (scratch, pretrained)
        ),
        "same_initial_encoder": scratch.get("initial_encoder_sha256")
        == pretrained.get("initial_encoder_sha256"),
        "same_platform": scratch.get("execution_platform")
        == pretrained.get("execution_platform"),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Paired worker identities changed: {checks}")
    delta = (
        pretrained["gap"]["best_validation_gap_mae_eV"]
        - scratch["gap"]["best_validation_gap_mae_eV"]
    )
    selection = {
        "format": "molgap-pcqm-gap100k-edgestate-local-hierarchy-selection-v1",
        "complete": True,
        "source_commit": source_commit,
        "seed": MODEL_SEED,
        "gpu_names": [
            scratch.get("accelerator_name"),
            pretrained.get("accelerator_name"),
        ],
        "execution_platform": scratch.get("execution_platform"),
        "architecture": "ogb_edge_state_structural_gps9",
        "parameter_count": EXPECTED_MODEL_PARAMETERS,
        "initial_encoder_sha256": scratch["initial_encoder_sha256"],
        "parent_geometry_cache_aggregate_sha256": PARENT_GEOMETRY_CACHE_SHA256,
        "local_label_cache_aggregate_sha256": label_sha256,
        "scratch_validation_gap_mae_eV": scratch["gap"][
            "best_validation_gap_mae_eV"
        ],
        "pretrained_validation_gap_mae_eV": pretrained["gap"][
            "best_validation_gap_mae_eV"
        ],
        "pretrained_minus_scratch_eV": delta,
        "minimum_paired_gain_eV": MIN_PAIRED_GAIN_EV,
        "passes_seed42_nomination_gate": delta <= -MIN_PAIRED_GAIN_EV,
        "equal_encoder_sample_exposure_epochs": SCRATCH_EPOCHS,
        "shadow_audit_read": False,
        "seed43_44_submitted": False,
        "scale_up_submitted": False,
        "elapsed_s": elapsed_s,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "molecular_research_server_accessed": False,
    }
    atomic_json(output_root / "selection.json", selection)
    atomic_json(
        output_root / "progress.json",
        {
            "format": "molgap-pcqm-gap100k-edgestate-local-hierarchy-progress-v1",
            "complete": True,
            "source_commit": source_commit,
            "elapsed_s": elapsed_s,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        },
    )
    return selection


def run_paired_screen(output_root: Path, *, label_sha256: str, source_commit: str):
    """Run scratch and local-hierarchy training on isolated Kaggle T4 workers."""
    import subprocess
    import sys
    import time

    output_root.mkdir(parents=True, exist_ok=True)
    names_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu_names = [line.strip() for line in names_result.stdout.splitlines() if line.strip()]
    if len(gpu_names) != 2 or any("T4" not in name for name in gpu_names):
        raise RuntimeError(f"Expected Kaggle T4x2, found {gpu_names}")
    started = time.perf_counter()
    workers = []
    for device, role in enumerate(("scratch", "pretrained")):
        environment = os.environ.copy()
        environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        environment["CUDA_VISIBLE_DEVICES"] = str(device)
        environment["MOLGAP_LOCAL_HIERARCHY_WORKER"] = role
        environment["MOLGAP_LOCAL_HIERARCHY_OUTPUT"] = str(output_root)
        environment["MOLGAP_LOCAL_LABEL_SHA256"] = label_sha256
        environment["MOLGAP_SOURCE_COMMIT"] = source_commit
        workers.append(
            (
                role,
                subprocess.Popen([sys.executable, str(Path(sys.argv[0]).resolve())], env=environment),
            )
        )
    wall_budget_s = 39_600
    while any(process.poll() is None for _, process in workers):
        time.sleep(10)
        failed = [
            role
            for role, process in workers
            if process.poll() is not None and process.returncode != 0
        ]
        if failed:
            for _, process in workers:
                if process.poll() is None:
                    process.terminate()
            raise RuntimeError(f"Local-hierarchy worker failed: {failed}")
        if time.perf_counter() - started > wall_budget_s:
            for _, process in workers:
                if process.poll() is None:
                    process.terminate()
            raise TimeoutError("Paired local-hierarchy screen exceeded 11 hours")
        atomic_json(
            output_root / "progress.json",
            {
                "format": "molgap-pcqm-gap100k-edgestate-local-hierarchy-progress-v1",
                "complete": False,
                "source_commit": source_commit,
                "gpu_names": gpu_names,
                "worker_exitcodes": {
                    role: process.poll() for role, process in workers
                },
                "elapsed_s": time.perf_counter() - started,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
    return finalize_paired_outputs(
        output_root,
        label_sha256=label_sha256,
        source_commit=source_commit,
        elapsed_s=time.perf_counter() - started,
    )


def accept_paired_screen(
    root: Path,
    *,
    expected_source_commit: str,
    expected_label_sha256: str,
) -> dict:
    """Recompute terminal metrics from tensors without executing a model."""
    import torch

    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    worker_metrics = {
        role: json.loads((root / role / "metrics.json").read_text(encoding="utf-8"))
        for role in ("scratch", "pretrained")
    }
    recomputed = {}
    payload_hashes = {}
    for role, stage in (("scratch", "scratch_gap"), ("pretrained", "finetune_gap")):
        path = root / role / f"{stage}_validation_payload.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        mae = float((payload["prediction_eV"] - payload["target_eV"]).abs().mean())
        recomputed[role] = {
            "mae_eV": mae,
            "row_id": payload["row_id"],
            "target": payload["target_eV"],
        }
        payload_hashes[role] = sha256_file(path)
    delta = recomputed["pretrained"]["mae_eV"] - recomputed["scratch"]["mae_eV"]
    checks = {
        "selection_format": selection.get("format")
        == "molgap-pcqm-gap100k-edgestate-local-hierarchy-selection-v1",
        "complete": selection.get("complete") is True,
        "source_commit": selection.get("source_commit") == expected_source_commit,
        "label_cache": selection.get("local_label_cache_aggregate_sha256")
        == expected_label_sha256,
        "parent_cache": selection.get("parent_geometry_cache_aggregate_sha256")
        == PARENT_GEOMETRY_CACHE_SHA256,
        "parameter_count": selection.get("parameter_count")
        == EXPECTED_MODEL_PARAMETERS,
        "same_initial_encoder": worker_metrics["scratch"].get(
            "initial_encoder_sha256"
        )
        == worker_metrics["pretrained"].get("initial_encoder_sha256")
        == selection.get("initial_encoder_sha256"),
        "equal_exposure": all(
            item.get("encoder_sample_exposure_epochs") == SCRATCH_EPOCHS
            for item in worker_metrics.values()
        ),
        "scratch_epochs": worker_metrics["scratch"]["gap"].get(
            "epochs_completed"
        )
        == SCRATCH_EPOCHS,
        "pretrain_epochs": worker_metrics["pretrained"]["pretraining"].get(
            "epochs_completed"
        )
        == PRETRAIN_EPOCHS,
        "finetune_epochs": worker_metrics["pretrained"]["gap"].get(
            "epochs_completed"
        )
        == FINETUNE_EPOCHS,
        "row_identity": torch.equal(
            recomputed["scratch"]["row_id"], recomputed["pretrained"]["row_id"]
        ),
        "target_identity": torch.equal(
            recomputed["scratch"]["target"], recomputed["pretrained"]["target"]
        ),
        "scratch_mae": abs(
            recomputed["scratch"]["mae_eV"]
            - worker_metrics["scratch"]["gap"]["best_validation_gap_mae_eV"]
        )
        < 1e-8,
        "pretrained_mae": abs(
            recomputed["pretrained"]["mae_eV"]
            - worker_metrics["pretrained"]["gap"]["best_validation_gap_mae_eV"]
        )
        < 1e-8,
        "delta": abs(delta - selection.get("pretrained_minus_scratch_eV", 1.0))
        < 1e-8,
        "official_validation_sealed": all(
            item.get("official_validation_role_read") is False
            for item in worker_metrics.values()
        ),
        "test_dev_sealed": all(
            item.get("test_dev_role_read") is False
            for item in worker_metrics.values()
        ),
        "shadow_unread": selection.get("shadow_audit_read") is False,
        "no_scale_up": selection.get("scale_up_submitted") is False,
    }
    acceptance = {
        "format": "molgap-pcqm-gap100k-edgestate-local-hierarchy-acceptance-v1",
        "accepted": all(checks.values()),
        "checks": checks,
        "source_commit": expected_source_commit,
        "local_label_cache_aggregate_sha256": expected_label_sha256,
        "scratch_validation_gap_mae_eV": recomputed["scratch"]["mae_eV"],
        "pretrained_validation_gap_mae_eV": recomputed["pretrained"]["mae_eV"],
        "pretrained_minus_scratch_eV": delta,
        "passes_seed42_nomination_gate": delta <= -MIN_PAIRED_GAIN_EV,
        "payload_sha256": payload_hashes,
        "model_inference_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(root / "acceptance.json", acceptance)
    if not acceptance["accepted"]:
        raise RuntimeError(f"Paired local-hierarchy acceptance failed: {checks}")
    return acceptance
