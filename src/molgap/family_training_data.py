"""Pinned PCQM graph views, independent of mount path, scale and platform.

Only already-built topology shards are loaded. Callers must separately obtain
role authority; a manifest pin verifies identity, not authorization.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .experiment_family_workflow import _artifact_path, _json
from .training_reproducibility import sha256_file
from .v4_runtime import sample_std_compat


@dataclass(frozen=True)
class DatasetSelection:
    manifest: str
    manifest_sha256: str
    development: bool

    def __post_init__(self):
        from .experiment_spec import _digest
        _digest(self.manifest_sha256, "dataset manifest")
        if type(self.development) is not bool:
            raise ValueError("Development selection must be an exact bool")
        if (not isinstance(self.manifest, str) or not self.manifest or
                any(part in {"", ".", ".."} for part in self.manifest.split("/")) or
                "\\" in self.manifest or ":" in self.manifest or self.manifest.startswith("/")):
            raise ValueError("Expected a confined relative manifest path")


@dataclass(frozen=True)
class TrainingData:
    train: object
    development: object | None
    manifest_sha256: str
    train_rows: int
    development_rows: int
    train_source_start: int
    development_source_start: int | None
    target_mean_eV: float
    target_std_eV: float


def resolve_training_manifest(root: Path, selection: DatasetSelection) -> Path:
    """Resolve metadata by its frozen hash, independent of platform mount layout.

    An existing declared path must match; a wrong manifest is never hidden by
    discovery. Only missing locations permit a unique metadata match. Shards
    remain unopened until the owning loader validates the selected roles.
    """
    if type(selection) is not DatasetSelection:
        raise TypeError("Expected DatasetSelection")
    root = Path(root).absolute()
    declared = root / selection.manifest
    if declared.exists():
        path = _artifact_path(root, selection.manifest)
        if sha256_file(path) != selection.manifest_sha256:
            raise ValueError("Dataset manifest SHA differs from the frozen selection")
        return path
    candidates = []
    for candidate in root.rglob(Path(selection.manifest).name):
        path = _artifact_path(root, candidate.relative_to(root).as_posix())
        if sha256_file(path) == selection.manifest_sha256:
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f"Expected one pinned training manifest, found {len(candidates)}")
    return candidates[0]


def load_training_data(root: Path, selection: DatasetSelection) -> TrainingData:
    """Verify the selected view and shard bytes before trusted deserialization.

    This reads only selected official-train/internal-development shards. It
    never traverses geometry or official-validation/test assets, even if those
    assets are present in the fixed dataset store. Data preparation remains CPU.
    """
    import torch
    from torch.utils.data import ConcatDataset
    from .k1_screen_training import _PackedGraphDatasetFactory

    if type(selection) is not DatasetSelection:
        raise TypeError("Expected DatasetSelection")
    path = resolve_training_manifest(root, selection)
    # Manifest file paths are relative to their dataset, not sibling mounts.
    root = path.parent
    manifest = _json(path)
    legacy_mirror = manifest.get("format") == "molgap-pcqm4mv2-kaggle-fixed-subset-v1"
    if (manifest.get("format") not in {"molgap-pcqm4mv2-fixed-subset-v1",
                                      "molgap-pcqm4mv2-kaggle-fixed-subset-v1"} or
            manifest.get("status") != "complete"):
        raise ValueError("Expected the existing complete fixed-subset manifest")
    if legacy_mirror and any(manifest.get(key) is not False for key in (
            "official_validation_role_read", "test_dev_role_read", "test_challenge_role_read")):
        raise ValueError("Mirror protected-role declarations must be explicitly false")
    source = manifest.get("source", {})
    if source.get("external_data_used") is not False:
        raise ValueError("Fixed PCQM training must declare no external data")
    official_rows = source.get("official_train_rows")
    if type(official_rows) is not int or official_rows < 2:
        raise ValueError("Missing official train boundary")
    graph = manifest.get("graph_contract", {})
    if (graph.get("feature_schema") != "ogb" or graph.get("node_feature_dim") != 9
            or graph.get("edge_feature_dim") != 3 or graph.get("rwse_dim") != 16):
        raise ValueError("Expected accepted OGB atom9/bond3/RWSE16 topology")
    records = (manifest.get("geometry_shards", []) if legacy_mirror else
               manifest.get("assets", {}).get("topology", []))
    if not isinstance(records, list) or not records:
        raise ValueError("Selected view has no topology shards")
    requested = ("train", "development") if selection.development else ("train",)
    role_ranges = {}
    for role in requested:
        declared = manifest.get("roles", {}).get(role)
        if type(declared) is not dict:
            raise ValueError("Selected role is absent: " + role)
        start, stop, rows = (declared.get(key) for key in ("source_idx_start", "source_idx_stop", "rows"))
        if (any(type(v) is not int for v in (start, stop, rows)) or
                not 0 <= start < stop <= official_rows or stop - start != rows):
            raise ValueError("Role is not a confined official-train range: " + role)
        role_ranges[role] = (start, stop)
    if selection.development:
        a, b = role_ranges["train"], role_ranges["development"]
        if max(a[0], b[0]) < min(a[1], b[1]):
            raise ValueError("Training and development roles overlap")
    datasets, labels = {}, {}
    for role, (start, stop) in role_ranges.items():
        parts, values, cursor = [], [], start
        for record in records:
            if record.get("role") != role:
                continue
            count = record.get("rows")
            if (type(count) is not int or count < 1 or
                    record.get("source_idx_min") != cursor or
                    record.get("source_idx_max") != cursor + count - 1 or
                    cursor + count > stop):
                raise ValueError("Shard order/count differs from its selected role")
            shard_path = _artifact_path(root, record["file"])
            if sha256_file(shard_path) != record["sha256"]:
                raise ValueError("Topology shard SHA differs from its manifest")
            # Reuse the existing topology-only view; never build/use cached 3D.
            shard = _PackedGraphDatasetFactory.load(shard_path, mmap=True)
            if len(shard) != count:
                raise ValueError("Decoded graph count differs from manifest")
            indices = shard._data.source_idx.reshape(-1)
            if indices.dtype != torch.int64 or not torch.equal(indices, torch.arange(cursor, cursor + count)):
                raise ValueError("Decoded source_idx differs from declared row order")
            target = shard._data.y.reshape(-1)
            if target.numel() != count or not target.is_floating_point() or not torch.isfinite(target).all():
                raise ValueError("Decoded Gap labels are incomplete or nonfinite")
            parts.append(shard)
            values.append(target.double())
            cursor += count
        if cursor != stop:
            raise ValueError("Selected role is missing topology shards")
        datasets[role], labels[role] = ConcatDataset(parts), torch.cat(values)
    targets = labels["train"]
    if targets.numel() < 2:
        raise ValueError("Target normalization requires at least two train rows")
    development_range = role_ranges.get("development")
    return TrainingData(datasets["train"], datasets.get("development"),
        selection.manifest_sha256, len(datasets["train"]),
        len(datasets["development"]) if selection.development else 0,
        role_ranges["train"][0], development_range[0] if development_range else None,
        float(targets.mean()), float(sample_std_compat(targets).clamp_min(1e-6)))


class PassBatchSampler:
    """Existing deterministic order policies parameterized by rows and seed."""
    def __init__(self, rows, batch_size, seed, *, algorithm, pass_index=0, start_batch=0,
                 drop_last=True):
        if type(drop_last) is not bool:
            raise ValueError("Tail policy must be an exact boolean")
        count = rows // batch_size if type(rows) is int and type(batch_size) is int and batch_size > 0 else 0
        if not drop_last and type(rows) is int and type(batch_size) is int and batch_size > 0:
            count = (rows + batch_size - 1) // batch_size
        if (any(type(v) is not int for v in (rows, batch_size, seed, pass_index, start_batch))
                or rows < batch_size or batch_size < 1 or seed < 0 or pass_index < 0
                or not 0 <= start_batch <= count):
            raise ValueError("Invalid sampler dimensions or resume cursor")
        if algorithm not in {"python-epoch-shuffle", "torch-pass-randperm"}:
            raise ValueError("Unsupported frozen sampler algorithm")
        self.rows, self.batch_size, self.seed = rows, batch_size, seed
        self.algorithm, self.pass_index, self.start_batch = algorithm, pass_index, start_batch
        self.drop_last, self.batches_per_pass = drop_last, count

    def __len__(self):
        return self.batches_per_pass - self.start_batch

    def __iter__(self):
        if self.algorithm == "python-epoch-shuffle":
            import random
            order = list(range(self.rows))
            random.Random(self.seed * 1_000_003 + self.pass_index).shuffle(order)
        else:
            import torch
            generator = torch.Generator().manual_seed(self.seed + self.pass_index)
            order = torch.randperm(self.rows, generator=generator).tolist()
        for offset in range(self.start_batch, self.batches_per_pass):
            start = offset * self.batch_size
            yield order[start:start + self.batch_size]
