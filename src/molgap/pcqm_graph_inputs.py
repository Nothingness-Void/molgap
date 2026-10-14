"""Accepted pure-2D PCQM graph inputs for the shared screen trainer.

This module is deliberately small.  It owns the boundary between a frozen
topology cache and a model-facing PyG batch; it does not build graphs and it
never opens a geometry asset.  A generic family therefore receives exactly the
same rows, targets, feature schema, and role separation as the existing
screening owners.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

from . import pcqm_topology as topology
from .pcqm_topology import graph_source_idx, load_packed_topology_shard, validate_ogb_gap_batch
from .screen_policy import canonical_fingerprint
from .training_reproducibility import sha256_file


MANIFEST_FORMAT = "molgap-pcqm4mv2-fixed-subset-v1"
FEATURE_SCHEMA = "ogb-atom9-bond3-rwse16-v1"
ROLES = ("train", "development")
PHYSICAL_BATCH = 128
SEED = 42
SAMPLER = "seed42-epoch-global-randperm-v1"

# These fields are not part of the pure topology ABI.  Rejecting them here
# catches a geometry cache accidentally copied into a graph-screen payload.
FORBIDDEN_FIELDS = frozenset({
    "pos", "positions", "geometry", "edge_distance", "edge_distances",
    "wedge_angle_cos", "wedge_edge_ids", "geometry_valid", "conformer",
})


def _digest(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_relative(value: object, label: str) -> str:
    if type(value) is not str or not value or "\\" in value or ":" in value or value.startswith("/"):
        raise ValueError(f"{label} must be a relative POSIX path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} has an unsafe path")
    return value


def _manifest_bytes(path: Path) -> tuple[dict, str]:
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Missing fixed graph manifest: {path}")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Fixed graph manifest is not JSON") from exc
    if type(value) is not dict:
        raise ValueError("Fixed graph manifest must be an object")
    return value, digest


def _accepted_manifest_identity(manifest: dict) -> None:
    """Require the repository's accepted fixed-data fingerprint.

    Tests and local smoke fixtures may add a checked-in equivalent to the
    topology acceptance table.  This keeps the generic owner from silently
    accepting a platform-local rebuilt graph cache.
    """
    identity = manifest.get("identity")
    if type(identity) is not dict or type(identity.get("name")) is not str:
        raise ValueError("Fixed graph manifest lacks an immutable dataset identity")
    known = {
        digest
        for values in topology.PREFLIGHT_DATASETS.values()
        for digest in values.values()
    }
    if canonical_fingerprint(manifest) not in known:
        raise ValueError("Fixed graph manifest is not an accepted immutable dataset")


def _validate_manifest_shape(manifest: dict) -> list[dict]:
    required = {"format", "status", "identity", "source", "assets"}
    if not required <= set(manifest) or manifest["format"] != MANIFEST_FORMAT or manifest["status"] != "complete":
        raise ValueError("Unsupported fixed graph manifest")
    source = manifest["source"]
    if type(source) is not dict or source.get("official_archive_sha256") != topology.OFFICIAL_ARCHIVE_SHA256 or source.get("official_row_manifest_sha256") != topology.OFFICIAL_ROW_MANIFEST_SHA256 or source.get("external_data_used") is not False:
        raise ValueError("Fixed graph source identity or external-data flag changed")
    contract = manifest.get("graph_contract")
    if contract is not None:
        if (type(contract) is not dict or contract.get("feature_schema") not in {"ogb", FEATURE_SCHEMA}
                or contract.get("node_feature_dim") != 9 or contract.get("edge_feature_dim") != 3
                or contract.get("rwse_dim") != 16):
            raise ValueError("Fixed graph feature contract changed")
    for sealed in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        if sealed in manifest and manifest[sealed] is not False:
            raise ValueError("Protected graph role was consumed")
    assets = manifest["assets"]
    # Geometry may be inventoried beside topology in the accepted fixed
    # manifest.  It is intentionally ignored and never opened by this owner.
    if type(assets) is not dict or "topology" not in assets or type(assets["topology"]) is not list or not assets["topology"]:
        raise ValueError("Fixed graph manifest lacks topology shard inventory")
    _accepted_manifest_identity(manifest)
    records = []
    seen = set()
    for item in assets["topology"]:
        if type(item) is not dict:
            raise ValueError("Invalid topology shard record")
        required_record = {"role", "file", "rows", "source_idx_min", "source_idx_max", "sha256", "bytes"}
        if not required_record <= set(item):
            raise ValueError("Topology shard inventory fields changed")
        role = item["role"]
        if role not in ROLES:
            raise ValueError("Topology shards must declare train/development roles")
        seen.add(role)
        _safe_relative(item["file"], "topology.file")
        _digest(item["sha256"], "topology.sha256")
        if type(item["rows"]) is not int or item["rows"] < PHYSICAL_BATCH:
            raise ValueError("Each graph role needs at least one physical batch")
        if type(item["source_idx_min"]) is not int or type(item["source_idx_max"]) is not int or item["source_idx_min"] < 0 or item["source_idx_max"] - item["source_idx_min"] + 1 != item["rows"]:
            raise ValueError("Topology source-index range does not match rows")
        if type(item["bytes"]) is not int or item["bytes"] <= 0:
            raise ValueError("Topology shard size is invalid")
        if any(token in item["file"].casefold() for token in ("geometry", "conformer", "position")):
            raise ValueError("Geometry asset selected for graph screen")
        records.append(dict(item))
    if set(seen) != set(ROLES):
        raise ValueError("Fixed graph manifest must declare train and development roles")
    role_manifest = manifest.get("roles")
    if role_manifest is not None:
        if type(role_manifest) is not dict or set(role_manifest) != set(ROLES):
            raise ValueError("Fixed graph role inventory changed")
        for role in ROLES:
            declaration = role_manifest[role]
            if type(declaration) is not dict or not {"source_idx_start", "source_idx_stop", "rows"} <= set(declaration):
                raise ValueError("Fixed graph role range is incomplete")
            role_records = sorted((item for item in records if item["role"] == role),
                                  key=lambda item: item["source_idx_min"])
            if (sum(item["rows"] for item in role_records) != declaration["rows"]
                    or role_records[0]["source_idx_min"] != declaration["source_idx_start"]
                    or role_records[-1]["source_idx_max"] + 1 != declaration["source_idx_stop"]):
                raise ValueError("Fixed graph role range differs from topology inventory")
    return records


def locate_manifest(root: Path, *, expected_sha256: str | None = None) -> Path:
    """Locate exactly one staged manifest, optionally by its frozen digest."""
    root = Path(root).resolve()
    if root.is_file():
        candidates = [root]
    else:
        candidates = sorted(path for path in root.rglob("*.json") if path.is_file() and (path.name in {"manifest.json", "fixed.json"} or path.name.endswith(".manifest.json")))
    if expected_sha256 is not None:
        _digest(expected_sha256, "manifest_sha256")
        candidates = [path for path in candidates if sha256_file(path) == expected_sha256]
    if len(candidates) != 1:
        raise ValueError("Expected exactly one immutable fixed graph manifest")
    return candidates[0]


@dataclass(frozen=True)
class GraphRole:
    role: str
    path: Path
    dataset: object
    source_idx: torch.Tensor
    target_eV: torch.Tensor
    source_idx_sha256: str
    target_sha256: str
    paths: tuple[Path, ...] = ()

    @property
    def rows(self) -> int:
        return int(self.source_idx.numel())

    @property
    def exposed_rows(self) -> int:
        """Rows consumed by complete physical batches under the frozen policy."""
        return self.rows - self.rows % PHYSICAL_BATCH

    @property
    def optimizer_steps(self) -> int:
        return self.exposed_rows // PHYSICAL_BATCH


@dataclass(frozen=True)
class GraphInputs:
    root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: dict
    roles: Mapping[str, GraphRole]

    def role(self, name: str) -> GraphRole:
        try:
            return self.roles[name]
        except KeyError as exc:
            raise ValueError(f"Unknown graph role: {name}") from exc

    def train_order(self, epoch: int, *, seed: int = SEED) -> torch.Tensor:
        return epoch_permutation(self.role("train").rows, epoch, seed=seed)

    def train_loader(self, epoch: int, *, batch_size: int = PHYSICAL_BATCH, seed: int = SEED):
        if batch_size != PHYSICAL_BATCH:
            raise ValueError("Graph screen physical batch is fixed at 128")
        from torch.utils.data import Subset
        from torch_geometric.loader import DataLoader

        role = self.role("train")
        order = self.train_order(epoch, seed=seed)[:role.exposed_rows].tolist()
        return DataLoader(Subset(role.dataset, order), batch_size=batch_size,
                          shuffle=False, drop_last=True, num_workers=0)

    def development_loader(self, *, batch_size: int = PHYSICAL_BATCH):
        if batch_size != PHYSICAL_BATCH:
            raise ValueError("Graph screen physical batch is fixed at 128")
        from torch_geometric.loader import DataLoader

        return DataLoader(self.role("development").dataset, batch_size=batch_size,
                          shuffle=False, num_workers=0)


def _tensor_digest(tensor: torch.Tensor, *, role: str) -> str:
    if role == "source_idx":
        if tensor.dtype != torch.int64:
            tensor = tensor.to(torch.int64)
        values = tensor.detach().cpu().contiguous().numpy().astype("<i8", copy=False)
    else:
        values = tensor.detach().cpu().double().contiguous().numpy().astype("<f8", copy=False)
    return hashlib.sha256(values.tobytes()).hexdigest()


def epoch_permutation(rows: int, epoch: int, *, seed: int = SEED) -> torch.Tensor:
    if type(rows) is not int or rows < PHYSICAL_BATCH or type(epoch) is not int or epoch < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Invalid graph screen sampler dimensions")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch - 1)
    return torch.randperm(rows, generator=generator, device="cpu", dtype=torch.int64)


def sampler_order_sha256(rows: int, epoch: int, *, seed: int = SEED) -> str:
    return _tensor_digest(epoch_permutation(rows, epoch, seed=seed), role="source_idx")


def _read_shard(root: Path, record: dict):
    path = (root / record["file"]).resolve()
    if not path.is_file() or not path.is_relative_to(root):
        raise ValueError("Topology shard escapes the staged input root")
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise ValueError("Topology shard changed after manifest acceptance")
    dataset = load_packed_topology_shard(path)
    if len(dataset) != record["rows"]:
        raise ValueError("Topology shard row count differs from the manifest")
    source_values = []
    targets = []
    for index in range(len(dataset)):
        graph = dataset[index]
        graph_fields = set(graph.keys())
        if FORBIDDEN_FIELDS & graph_fields:
            raise ValueError("Geometry field found in a pure topology graph")
        source = graph_source_idx(graph).reshape(-1)
        target = getattr(graph, "y", None)
        if source.numel() != 1 or not torch.is_tensor(target) or target.reshape(-1).numel() != 1 or not torch.isfinite(target).all():
            raise ValueError("Topology graph lacks one finite source index and Gap target")
        source_values.append(int(source.item()))
        targets.append(float(target.reshape(-1)[0].item()))
    source_idx = torch.tensor(source_values, dtype=torch.int64)
    target_eV = torch.tensor(targets, dtype=torch.float64)
    expected = torch.arange(record["source_idx_min"], record["source_idx_max"] + 1, dtype=torch.int64)
    if not torch.equal(source_idx, expected):
        raise ValueError("Topology source index order differs from the frozen manifest")
    from torch_geometric.loader import DataLoader
    batch = next(iter(DataLoader(dataset, batch_size=PHYSICAL_BATCH, shuffle=False, num_workers=0)))
    if batch.x.device.type != "cpu" or validate_ogb_gap_batch(batch) != PHYSICAL_BATCH:
        raise ValueError("Topology shard failed the accepted CPU OGB batch check")
    loaded_source = graph_source_idx(batch).reshape(-1).to(torch.int64)
    if not torch.equal(loaded_source, source_idx[:PHYSICAL_BATCH]):
        raise ValueError("Topology DataLoader changed source-index identity")
    return path, dataset, source_idx, target_eV


def _read_role(root: Path, records: list[dict]) -> GraphRole:
    if not records:
        raise ValueError("Missing topology shards for graph role")
    records = sorted(records, key=lambda item: item["source_idx_min"])
    datasets, sources, targets, paths = [], [], [], []
    expected_start = records[0]["source_idx_min"]
    for record in records:
        if record["source_idx_min"] != expected_start:
            raise ValueError("Topology role shards have a gap or overlap")
        path, dataset, source_idx, target_eV = _read_shard(root, record)
        paths.append(path); datasets.append(dataset); sources.append(source_idx); targets.append(target_eV)
        expected_start = record["source_idx_max"] + 1
    source_idx = torch.cat(sources)
    target_eV = torch.cat(targets)
    from torch.utils.data import ConcatDataset
    dataset = ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    return GraphRole(records[0]["role"], paths[0], dataset, source_idx, target_eV,
                     _tensor_digest(source_idx, role="source_idx"),
                     _tensor_digest(target_eV, role="target"), tuple(paths))


def load_graph_inputs(root: Path, *, manifest_path: Path | None = None,
                      expected_manifest_sha256: str | None = None,
                      expected_identity: dict | str | None = None) -> GraphInputs:
    """Load and validate the immutable topology-only train/development pair."""
    root = Path(root).resolve()
    manifest_path = Path(manifest_path).resolve() if manifest_path is not None else locate_manifest(root, expected_sha256=expected_manifest_sha256)
    if root.is_dir() and not manifest_path.is_relative_to(root):
        raise ValueError("Fixed graph manifest escapes the supplied input root")
    manifest, digest = _manifest_bytes(manifest_path)
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise ValueError("Fixed graph manifest digest mismatch")
    records = _validate_manifest_shape(manifest)
    if expected_identity is not None:
        if isinstance(expected_identity, str):
            if digest != expected_identity and canonical_fingerprint(manifest) != expected_identity:
                raise ValueError("Graph manifest identity differs from the frozen recipe")
        elif type(expected_identity) is dict and manifest.get("identity") != expected_identity:
            raise ValueError("Graph manifest dataset identity differs from the frozen recipe")
        else:
            raise ValueError("Invalid expected graph manifest identity")
    # Staged dataset mounts commonly place ``manifest.json`` beside a
    # ``store/`` directory below the caller's broader input mount.  Inventory
    # paths are relative to that manifest, never to the process working root.
    data_root = manifest_path.parent
    roles = {role: _read_role(data_root, [record for record in records if record["role"] == role]) for role in ROLES}
    if set(roles) != set(ROLES):
        raise ValueError("Graph screen requires train and development roles")
    return GraphInputs(root, manifest_path, digest, manifest, roles)


__all__ = [
    "FEATURE_SCHEMA", "FORBIDDEN_FIELDS", "GraphInputs", "GraphRole",
    "MANIFEST_FORMAT", "PHYSICAL_BATCH", "ROLES", "SAMPLER", "SEED",
    "epoch_permutation", "load_graph_inputs", "locate_manifest",
    "sampler_order_sha256",
]
