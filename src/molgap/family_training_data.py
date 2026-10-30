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
    train_source_start: int | None
    development_source_start: int | None
    target_mean_eV: float
    target_std_eV: float
    synthetic: bool = False
    train_source_indices: tuple[int, ...] | None = None
    development_source_indices: tuple[int, ...] | None = None
    indexed: bool = False


def resolve_pinned_manifest(root, selection):
    """Mount paths are provenance; resolve a unique immutable manifest by its pin."""
    if type(selection) is not DatasetSelection:
        raise TypeError("Expected DatasetSelection")
    root = Path(root).absolute()
    direct = root / selection.manifest
    if direct.exists():
        path = _artifact_path(root, selection.manifest)
        if sha256_file(path) != selection.manifest_sha256:
            raise ValueError("Dataset manifest SHA differs from the frozen selection")
        return path
    candidates = []
    for candidate in root.rglob(Path(selection.manifest).name):
        path = _artifact_path(root, candidate.relative_to(root).as_posix())
        if sha256_file(path) == selection.manifest_sha256:
            candidates.append(path)
    if len(candidates) > 1:
        # Source attachments retain the binding for evidence, but do not carry
        # its indexed payloads. Only a uniquely data-backed mount may execute.
        from .research_memory.paths import repo_local_path
        from .common.io import _safe_local
        backed = []
        for path in candidates:
            binding = _json(path)
            if binding.get("format") == "molgap-indexed-training-input-v1":
                data_root = repo_local_path(path.parent, binding["data_root"])
                _safe_local(data_root)
                if data_root.is_dir():
                    backed.append(path)
        candidates = backed
    if len(candidates) != 1:
        raise ValueError("Expected one uniquely mounted pinned manifest")
    return _artifact_path(root, candidates[0].relative_to(root).as_posix())


# Desktop public spelling; both callers use the same confined resolver.
resolve_training_manifest = resolve_pinned_manifest


def load_training_data(root: Path, selection: DatasetSelection) -> TrainingData:
    """Verify the selected view and shard bytes before trusted deserialization.

    This reads only selected official-train/internal-development shards. It
    never selects coordinates or official-validation/test roles. Accepted legacy
    packed files incidentally deserialize coordinates; model transfer excludes
    them. Data preparation remains CPU.
    """
    import torch
    from torch.utils.data import ConcatDataset
    from .packed_graph_dataset import PackedGraphDataset

    if type(selection) is not DatasetSelection:
        raise TypeError("Expected DatasetSelection")
    root = Path(root).absolute()
    path = resolve_pinned_manifest(root, selection)
    if sha256_file(path) != selection.manifest_sha256:
        raise ValueError("Dataset manifest SHA differs from the frozen selection")
    manifest = _json(path)
    if manifest.get("format") == "molgap-indexed-training-input-v1":
        return load_indexed_training_data(root, selection)
    from .pcqm_fixed_records import graph_records
    records = graph_records(manifest, selection.manifest_sha256)
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
            shard_path = _artifact_path(path.parent, record["file"])
            if sha256_file(shard_path) != record["sha256"]:
                raise ValueError("Topology shard SHA differs from its manifest")
            shard = PackedGraphDataset(shard_path, mmap=True)
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


def load_indexed_training_data(root: Path, selection: DatasetSelection) -> TrainingData:
    """Consume explicit accepted V2 maps; never infer an interval or reselect.

    The small binding is frozen as the recipe's dataset manifest. Each role
    additionally carries the independently supplied acceptance/manifest pins,
    authority and selection history consumed by the existing indexed owner.
    No network, graph construction, Audit IO or training happens here.
    """
    import torch
    from .data.indexed_inputs import AcceptedIndexedInputPermit, IndexedInputPermit, IndexedRoleDataset
    from .common.io import _safe_local
    from .research_memory.paths import repo_local_path
    path = resolve_pinned_manifest(root, selection)
    binding = _json(path)
    if (type(binding) is not dict or set(binding) not in (
            {"format", "data_root", "scope", "train", "development"},
            {"format", "data_root", "scope", "train", "development", "normalization_reference"}) or
            binding["format"] != "molgap-indexed-training-input-v1" or
            binding["scope"] not in {"CPU_SYNTHETIC_ONLY", "ACCEPTED_INDEXED_INPUT_V1"}):
        raise ValueError("Unknown indexed family input binding")
    relative = binding["data_root"]
    if type(relative) is not str or not relative or "\\" in relative or ":" in relative or any(
            part in {"", ".", ".."} for part in relative.split("/")):
        raise ValueError("Indexed data root must be confined and explicit")
    _safe_local(path.parent / relative)
    data_root = repo_local_path(path.parent, relative)
    if selection.development and binding["development"] is None:
        raise ValueError("Indexed selected development differs from input binding")
    datasets = {}
    for role in ("train", "development") if selection.development else ("train",):
        record = binding[role]
        if type(record) is not dict or set(record) != {"permit", "authority", "role_history"}:
            raise ValueError("Indexed role lacks its exact authority/history binding")
        kind = IndexedInputPermit if binding["scope"] == "CPU_SYNTHETIC_ONLY" else AcceptedIndexedInputPermit
        permit = kind(**record["permit"])
        if permit.role not in ({"train_100k", "train_500k"} if role == "train" else {"common_dev"}):
            raise ValueError("ROLE_BLOCKED: indexed role substituted in family input")
        datasets[role] = IndexedRoleDataset(data_root, permit, authority=record["authority"],
                                             role_history=record["role_history"])
    train, dev = datasets["train"], datasets.get("development")
    if dev is not None and (train.permit.version_identity != dev.permit.version_identity or
            train.permit.acceptance_sha256 != dev.permit.acceptance_sha256 or
            set(train.source_indices) & set(dev.source_indices)):
        raise ValueError("Indexed train/development versions overlap or disagree")
    targets = torch.cat(list(train.selected_targets()))
    if len(targets) < 2 or not bool(torch.isfinite(targets).all()):
        raise ValueError("Indexed train-only normalization needs finite labels")
    mean, std = indexed_target_statistics(targets, train.source_indices,
        binding.get("normalization_reference"))
    return TrainingData(train, dev, selection.manifest_sha256, len(train), len(dev) if dev else 0,
        None, None, mean, std,
        synthetic=binding["scope"] == "CPU_SYNTHETIC_ONLY", train_source_indices=train.source_indices,
        development_source_indices=dev.source_indices if dev else None, indexed=True)


def indexed_target_statistics(targets, source_indices, reference=None):
    """Verify a scale study's pinned nested-train transform against actual labels."""
    import torch
    from .experiment_family_workflow import tensor_digest
    if reference is not None:
        fields = {"role", "rows", "source_index_sha256", "mean_eV", "std_eV"}
        if (type(reference) is not dict or set(reference) != fields or
                reference["role"] != "train_100k" or reference["rows"] != 100000 or
                len(source_indices) != 500000 or len(targets) != 500000):
            raise ValueError("Frozen scale normalization requires exact nested100K source")
        indices = torch.tensor(source_indices[:100000], dtype=torch.int64)
        if tensor_digest(indices, role="source_idx") != reference["source_index_sha256"]:
            raise ValueError("Frozen normalization source membership/order differs")
        targets = targets[:100000]
    mean, std = float(targets.mean()), float(sample_std_compat(targets).clamp_min(1e-6))
    if reference is not None and (mean, std) != (reference["mean_eV"], reference["std_eV"]):
        raise ValueError("Frozen normalization source labels/statistics differ")
    return mean, std


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
