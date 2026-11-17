"""Shared OGB topology checks for EdgeState training and CPU input preflight."""
from __future__ import annotations

from pathlib import Path


# These are the accepted IMS fixed-dataset canonical identities. The K1 full
# identity also appears in its frozen trainer, which must not be rewritten.
PREFLIGHT_DATASETS = {
    ("neural_atom_k1", "1"): {
        "ogb-train-full": (
            "7d358a679d299bd5de6c5822e42dcef34b0e0b9fa9598c684709509c0b94021d"
        ),
    },
    ("edge_state_gps", "1"): {
        "ogb-train-100k": (
            "b087fb07b12b7610b0934283384e2f6eb2110422a3fdc2210c93be1deccf6879"
        ),
        "ogb-train-500k-scnet-v1": (
            "384553c8268447b8f7b83f5104073309e7f12755ab3fc29796dbf78f1914320b"
        ),
        "ogb-train-1m": (
            "e339e82f903d1381c65fd061c23e7403b72f16a79114a84b9822349555ae4017"
        ),
    },
}
OFFICIAL_ARCHIVE_SHA256 = (
    "628ae612a3de752160929d93d1584a75257ae156e7b13d91b133d8db62e6d701"
)
OFFICIAL_ROW_MANIFEST_SHA256 = (
    "c329fbde935324118f4f62aa6e2b01b209d6f54642549637853419deb1761c4b"
)


def graph_source_idx(batch):
    """Accepted PCQM graphs use row_index; newer callers may use source_idx."""
    import torch

    source = getattr(batch, "source_idx", None)
    row = getattr(batch, "row_index", None)
    if torch.is_tensor(row) and torch.is_tensor(getattr(batch, "ptr", None)):
        # PyG increments any key containing "index" by preceding node counts.
        offsets = batch.ptr[:-1].reshape(row.shape)
        row = row - offsets
    if source is None:
        source = row
    elif row is not None and (not torch.is_tensor(row) or not torch.equal(source, row)):
        raise ValueError("source_idx and row_index disagree")
    if not torch.is_tensor(source):
        raise ValueError("EdgeState batch is missing source_idx/row_index")
    return source


def validate_ogb_gap_batch(batch) -> int:
    """Validate a model-facing OGB atom9/bond3/RWSE16 Gap batch."""
    import torch

    from .ogb_features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS

    fields = ("x", "edge_index", "edge_attr", "batch", "random_walk_pe", "y")
    if any(not torch.is_tensor(getattr(batch, name, None)) for name in fields):
        raise ValueError("EdgeState batch is missing required tensor fields")
    source_field = graph_source_idx(batch)
    x, edges, bonds, groups = batch.x, batch.edge_index, batch.edge_attr, batch.batch
    pe, target, source = batch.random_walk_pe, batch.y.reshape(-1), source_field.reshape(-1)
    integers = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if (x.ndim != 2 or x.shape[1] != len(ATOM_FEATURE_DIMS) or x.shape[0] == 0
            or x.dtype not in integers):
        raise ValueError("Expected nonempty OGB atom9 categorical features")
    if (edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype not in integers
            or bonds.ndim != 2 or bonds.shape != (edges.shape[1], len(BOND_FEATURE_DIMS))
            or bonds.dtype not in integers):
        raise ValueError("Expected aligned OGB bond3 features and edge_index")
    if (groups.ndim != 1 or groups.numel() != x.shape[0] or groups.dtype not in integers
            or groups.numel() == 0):
        raise ValueError("Invalid graph batch vector")
    if len({value.device for value in (x, edges, bonds, groups, pe, target, source)}) != 1:
        raise ValueError("Batch tensors must be on one device")
    if bool(((groups < 0) | (groups >= x.shape[0])).any()):
        raise ValueError("Graph IDs must be contiguous from zero")
    count = int(groups.max()) + 1
    if count < 1 or not torch.equal(torch.unique(groups), torch.arange(count, device=groups.device)):
        raise ValueError("Graph IDs must be contiguous from zero")
    if (pe.shape != (x.shape[0], 16) or not pe.is_floating_point()
            or not bool(torch.isfinite(pe).all())):
        raise ValueError("Expected finite RWSE16 per atom")
    if (batch.y.shape not in ((count,), (count, 1)) or not target.is_floating_point()
            or not bool(torch.isfinite(target).all())):
        raise ValueError("Expected one finite Gap target in eV per graph")
    if (source_field.shape not in ((count,), (count, 1)) or source.dtype not in integers
            or bool((source < 0).any()) or torch.unique(source).numel() != count):
        raise ValueError("Expected unique nonnegative source_idx per graph")
    if edges.numel() and bool(((edges < 0) | (edges >= x.shape[0])).any()):
        raise ValueError("edge_index refers to a nonexistent atom")
    if edges.numel() and bool((groups[edges[0].long()] != groups[edges[1].long()]).any()):
        raise ValueError("edge_index crosses graph boundaries")
    if bool(((x < 0) | (x >= x.new_tensor(ATOM_FEATURE_DIMS))).any()):
        raise ValueError("OGB atom feature outside its categorical range")
    if bool(((bonds < 0) | (bonds >= bonds.new_tensor(BOND_FEATURE_DIMS))).any()):
        raise ValueError("OGB bond feature outside its categorical range")
    return count


def load_packed_topology_shard(path: Path):
    """Load a trusted, SHA-verified PyG (data, slices) shard on CPU."""
    from torch_geometric.data import InMemoryDataset

    from .v4_runtime import torch_load_compat

    class PackedGraphDataset(InMemoryDataset):
        def __init__(self, source: Path):
            super().__init__(root=None)
            self.data, self.slices = torch_load_compat(
                source, map_location="cpu", weights_only=False, mmap=True,
            )

    return PackedGraphDataset(Path(path))


def inspect_selected_topology_shards(data_root: Path, entry: dict, arm: dict):
    """Verify one immutable topology shard per role; never claim full-role coverage."""
    import hashlib
    import json
    import torch
    from torch_geometric.loader import DataLoader

    from .screen_policy import canonical_fingerprint

    family = (arm["family"]["name"], arm["family"]["version"])
    accepted = PREFLIGHT_DATASETS[family]
    fixed = data_root / entry["fixed_manifest"]["path"]
    manifest = json.loads(fixed.read_text(encoding="utf-8"))
    dataset_name = manifest.get("identity", {}).get("name")
    if dataset_name not in accepted or canonical_fingerprint(manifest) != accepted[dataset_name]:
        raise ValueError("Not the accepted frozen PCQM topology manifest")
    roles = ("train",) if family == ("neural_atom_k1", "1") else ("train", "development")
    if (manifest.get("format") != "molgap-pcqm4mv2-fixed-subset-v1"
            or manifest.get("status") != "complete"
            or manifest.get("identity", {}).get("name") != dataset_name
            or manifest.get("source", {}).get("official_archive_sha256") != OFFICIAL_ARCHIVE_SHA256
            or manifest["source"].get("official_row_manifest_sha256") != OFFICIAL_ROW_MANIFEST_SHA256
            or manifest["source"].get("external_data_used") is not False):
        raise ValueError("Frozen topology dataset identity changed")
    records = manifest["assets"]["topology"]
    if len(entry["files"]) != len(roles) or {item["role"] for item in entry["files"]} != set(roles):
        raise ValueError("Select exactly one topology shard per declared role")
    observations = {}
    for item in entry["files"]:
        matching = [record for record in records if record["file"] == item["path"]]
        if len(matching) != 1 or any(item[key] != matching[0][key] for key in ("role", "sha256", "bytes")):
            raise ValueError("Selected shard differs from frozen topology inventory")
        record = matching[0]
        path = data_root / item["path"]
        # The parent hashed the private copy. Recheck immediately before pickle load.
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != item["bytes"] or digest.hexdigest() != item["sha256"]:
            raise ValueError("Selected topology shard changed after staging")
        graphs = load_packed_topology_shard(path)
        rows = record["rows"]
        if type(rows) is not int or rows < 128 or len(graphs) != rows:
            raise ValueError("Selected topology shard row count mismatch")
        source = getattr(graphs._data, "row_index", None)
        if source is None:
            source = getattr(graphs._data, "source_idx", None)
        if (not torch.is_tensor(source) or source.numel() != rows
                or not torch.equal(source.reshape(-1).long(), torch.arange(
                    record["source_idx_min"], record["source_idx_max"] + 1, dtype=torch.long))):
            raise ValueError("Selected topology source indices differ from frozen inventory")
        target = getattr(graphs._data, "y", None)
        if (not torch.is_tensor(target) or target.numel() != rows
                or not bool(torch.isfinite(target).all())):
            raise ValueError("Selected topology Gap targets are incomplete or nonfinite")
        batch = next(iter(DataLoader(graphs, batch_size=128, shuffle=False, num_workers=0)))
        if validate_ogb_gap_batch(batch) != 128 or batch.x.device.type != "cpu":
            raise ValueError("Selected topology CPU loader batch invalid")
        if not torch.equal(graph_source_idx(batch).reshape(-1).long(), source.reshape(-1)[:128].long()):
            raise ValueError("Selected topology loader changed source order")
        observations[item["role"]] = {
            "graphs": int(batch.num_graphs), "rows": rows, "device": "cpu",
            "shard": item["path"], "sha256": item["sha256"],
            "source_idx_min": record["source_idx_min"], "source_idx_max": record["source_idx_max"],
        }
        del batch, graphs, source, target
    return observations
