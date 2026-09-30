"""CPU-only inputs for the bounded GPTrans author/local attribution arms.

The module deliberately stops at data preparation.  It does not construct a
GPTrans model, read targets, run inference, or make a scientific acceptance
decision.  G2 stores one deterministic directed shortest path for every
non-bonded ordered pair whose distance is in ``[2, 20]``.  The runtime can use
the stored categorical histogram with the existing ``BondEncoder`` tables,
without adding hop-position parameters or changing the direct-bond branch.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from .gptrans_initial_scale_preflight import (
    INITIAL_MODEL_SHA256,
    INITIAL_STATE_SHA256,
)
from .pcqm_gptrans_v4 import _load_datasets, validate_fixed_assets
from .training_reproducibility import (
    atomic_json,
    atomic_torch_save,
    sha256_file,
)


MAX_DISTANCE = 20
CHUNK_ROWS = 1_000
DEGREE_SCALE = 0.0897
INITIAL_STATE_FORMAT = "molgap-gptrans-t-seed42-initial-state-v1"
INITIAL_SCALED_FORMAT = "molgap-gptrans-t-seed42-degree-scaled-initial-state-v1"
SIDECAR_FORMAT = "molgap-gptrans-author-path-sidecar-v1"
SHARD_FORMAT = "molgap-gptrans-author-path-shard-v1"

PATH_PAIR_SRC = "path_pair_src"
PATH_PAIR_DST = "path_pair_dst"
PATH_PAIR_DISTANCE = "path_pair_distance"
PATH_PAIR_PTR = "path_pair_ptr"
PATH_EDGE_ATTR = "path_edge_attr"
PATH_PAIR_HIST = "path_pair_hist"

RUNTIME_COMPACT_SCHEMA = "molgap-gptrans-author-runtime-compact-v1"
RUNTIME_COMPACT_BYTES_CAP = 4 * 1024**3
RUNTIME_COMPACT_KEYS = (
    "graph_pair_ptr",
    "source_index",
    PATH_PAIR_SRC,
    PATH_PAIR_DST,
    PATH_PAIR_DISTANCE,
    PATH_PAIR_HIST,
)

_RUNTIME_COMPACT_DTYPE_BYTES = {
    "graph_pair_ptr": 8,
    "source_index": 8,
    PATH_PAIR_SRC: 8,
    PATH_PAIR_DST: 8,
    PATH_PAIR_DISTANCE: 8,
    PATH_PAIR_HIST: 2,
}

_SCALED_KEYS = (
    "in_degree_encoder.weight",
    "out_degree_encoder.weight",
)


def _runtime_compact_cache_estimate(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Estimate the tensor-only runtime cache before loading any chunk."""

    dimensions = get_bond_feature_dims()
    histogram_width = sum(dimensions)
    tensor_bytes = {key: 0 for key in RUNTIME_COMPACT_KEYS}
    rows = pairs = 0
    for record in records:
        row_count = int(record["rows"])
        pair_count = int(record["pairs"])
        if row_count < 0 or pair_count < 0:
            raise RuntimeError("sidecar runtime cache counts must be non-negative")
        rows += row_count
        pairs += pair_count
        tensor_bytes["graph_pair_ptr"] += (row_count + 1) * _RUNTIME_COMPACT_DTYPE_BYTES["graph_pair_ptr"]
        tensor_bytes["source_index"] += row_count * _RUNTIME_COMPACT_DTYPE_BYTES["source_index"]
        for key in (PATH_PAIR_SRC, PATH_PAIR_DST, PATH_PAIR_DISTANCE):
            tensor_bytes[key] += pair_count * _RUNTIME_COMPACT_DTYPE_BYTES[key]
        tensor_bytes[PATH_PAIR_HIST] += (
            pair_count * histogram_width * _RUNTIME_COMPACT_DTYPE_BYTES[PATH_PAIR_HIST]
        )
    estimated_bytes = sum(tensor_bytes.values())
    return {
        "schema": RUNTIME_COMPACT_SCHEMA,
        "keys": list(RUNTIME_COMPACT_KEYS),
        "dtypes": {
            "graph_pair_ptr": "torch.int64",
            "source_index": "torch.int64",
            PATH_PAIR_SRC: "torch.int64",
            PATH_PAIR_DST: "torch.int64",
            PATH_PAIR_DISTANCE: "torch.int64",
            PATH_PAIR_HIST: "torch.int16",
        },
        "histogram_width": histogram_width,
        "chunks": len(records),
        "rows": rows,
        "pairs": pairs,
        "tensor_bytes": tensor_bytes,
        "estimated_bytes": estimated_bytes,
        "cap_bytes": RUNTIME_COMPACT_BYTES_CAP,
        "within_cap": estimated_bytes <= RUNTIME_COMPACT_BYTES_CAP,
    }


def digest(path: Path) -> str:
    """Return the SHA-256 digest of one file.

    This is intentionally the repository's shared file-digest primitive under
    the short name used by the CPU acceptance scripts.
    """

    return sha256_file(Path(path))


def _tensor_digest(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def state_digest(state: Mapping[str, torch.Tensor]) -> str:
    """Hash a state dict with the same identity rule as the frozen preflight."""

    hasher = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not isinstance(name, str) or not torch.is_tensor(value):
            raise TypeError("state must map string names to tensors")
        hasher.update(name.encode("utf-8") + b"\0")
        hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def get_bond_feature_dims() -> tuple[int, int, int]:
    """Return OGB's three categorical bond-table dimensions."""

    from ogb.graphproppred.mol_encoder import (
        get_bond_feature_dims as _get_bond_feature_dims,
    )

    dimensions = tuple(int(value) for value in _get_bond_feature_dims())
    if dimensions != (5, 6, 2):
        raise RuntimeError(f"Unexpected OGB bond feature dimensions: {dimensions}")
    return dimensions


def _histogram_offsets(dimensions: Sequence[int]) -> tuple[int, ...]:
    offsets = [0]
    for dimension in dimensions:
        if int(dimension) <= 0:
            raise ValueError("bond feature dimensions must be positive")
        offsets.append(offsets[-1] + int(dimension))
    return tuple(offsets)


def _looks_like_edge_index(value: Any) -> bool:
    try:
        array = torch.as_tensor(value)
    except Exception:
        return False
    return array.ndim == 2 and (array.shape[0] == 2 or array.shape[1] == 2)


def _edge_records(edges: Any) -> list[tuple[int, int, tuple[int, int, int]]]:
    """Normalize PyG ``(edge_index, edge_attr)`` or edge triples."""

    edge_index = None
    edge_attr = None
    if isinstance(edges, Mapping):
        edge_index = edges.get("edge_index")
        edge_attr = edges.get("edge_attr")
    elif hasattr(edges, "edge_index") and hasattr(edges, "edge_attr"):
        edge_index = getattr(edges, "edge_index")
        edge_attr = getattr(edges, "edge_attr")
    elif isinstance(edges, (tuple, list)) and len(edges) == 2 and _looks_like_edge_index(edges[0]):
        edge_index, edge_attr = edges

    dimensions = get_bond_feature_dims()
    if edge_index is not None:
        index = torch.as_tensor(edge_index)
        attributes = torch.as_tensor(edge_attr)
        if index.ndim != 2:
            raise ValueError("edge_index must be a rank-2 array")
        if index.shape[0] == 2:
            sources = index[0].reshape(-1).tolist()
            targets = index[1].reshape(-1).tolist()
        elif index.shape[1] == 2:
            sources = index[:, 0].reshape(-1).tolist()
            targets = index[:, 1].reshape(-1).tolist()
        else:
            raise ValueError("edge_index must have shape [2, edges] or [edges, 2]")
        if attributes.ndim == 1 and attributes.numel() == 3 and len(sources) == 1:
            attributes = attributes.reshape(1, 3)
        if attributes.ndim != 2 or attributes.shape[0] != len(sources):
            raise ValueError("edge_attr must have one row per directed edge")
        if attributes.shape[1] != len(dimensions):
            raise ValueError("edge_attr must contain the three OGB bond categories")
        raw_records = zip(sources, targets, attributes.tolist(), strict=True)
    else:
        try:
            raw_records = iter(edges)
        except TypeError as error:
            raise TypeError("edges must be PyG edge tensors or edge triples") from error

    records = []
    for source, target, category in raw_records:
        source = int(source)
        target = int(target)
        values = tuple(int(value) for value in category)
        if len(values) != len(dimensions):
            raise ValueError("each bond category must have three values")
        if any(value < 0 or value >= dimensions[column] for column, value in enumerate(values)):
            raise ValueError(f"bond category outside OGB dimensions: {values}")
        records.append((source, target, values))
    return records


def graph_paths(
    nodes: int,
    edges: Any,
    *,
    max_distance: int = MAX_DISTANCE,
    include_path_hist: bool = True,
) -> dict[str, torch.Tensor]:
    """Build deterministic directed shortest-path CSR for one 2D graph.

    ``edges`` may be a PyG ``(edge_index, edge_attr)`` pair, a ``Data`` object,
    a mapping with those two keys, or an iterable of ``(src, dst, category)``
    triples.  Directed categories are retained exactly as supplied.  For each
    source, neighbors are visited in ascending node order.  A pair is emitted
    only when its first discovered path has length 2 through ``max_distance``;
    direct edges, diagonal pairs, unreachable pairs, and over-cap pairs are
    omitted.

    The canonical fields are ``path_pair_src``, ``path_pair_dst``,
    ``path_pair_distance``, ``path_pair_ptr`` and ``path_edge_attr``.  The
    optional ``path_pair_hist`` is a flattened ``[pairs, 13]`` histogram with
    category blocks ordered as OGB bond type, stereo, and conjugation.
    """

    if isinstance(nodes, bool) or int(nodes) != nodes or int(nodes) < 1:
        raise ValueError("nodes must be a positive integer")
    node_count = int(nodes)
    if isinstance(max_distance, bool) or int(max_distance) != max_distance or int(max_distance) < 2:
        raise ValueError("max_distance must be an integer >= 2")
    max_distance = int(max_distance)
    dimensions = get_bond_feature_dims()
    offsets = _histogram_offsets(dimensions)
    adjacency: list[dict[int, tuple[int, int, int]]] = [dict() for _ in range(node_count)]
    for source, target, category in _edge_records(edges):
        if not (0 <= source < node_count and 0 <= target < node_count):
            raise ValueError("edge endpoint outside graph")
        if source == target:
            raise ValueError("self edges are not valid molecular bonds")
        previous = adjacency[source].get(target)
        if previous is not None and previous != category:
            raise ValueError("conflicting duplicate directed bond categories")
        adjacency[source][target] = category

    records: list[tuple[int, int, int, tuple[tuple[int, int, int], ...]]] = []
    for source in range(node_count):
        distances = [-1] * node_count
        paths: list[tuple[tuple[int, int, int], ...] | None] = [None] * node_count
        distances[source] = 0
        paths[source] = ()
        queue: deque[int] = deque([source])
        while queue:
            current = queue.popleft()
            current_distance = distances[current]
            if current_distance >= max_distance:
                continue
            current_path = paths[current]
            assert current_path is not None
            for target in sorted(adjacency[current]):
                if distances[target] != -1:
                    continue
                distances[target] = current_distance + 1
                paths[target] = current_path + (adjacency[current][target],)
                queue.append(target)
        for target, distance in enumerate(distances):
            if 2 <= distance <= max_distance:
                path = paths[target]
                assert path is not None
                records.append((source, target, distance, path))

    # Canonical storage order is source, distance, target.  BFS still decides
    # which tied path is retained; this order only makes the flattened CSR
    # stable for independent re-derivation and chunk comparisons.
    records.sort(key=lambda item: (item[0], item[2], item[1]))
    pair_sources = [item[0] for item in records]
    pair_targets = [item[1] for item in records]
    pair_distances = [item[2] for item in records]
    path_ptr = [0]
    path_categories: list[int] = []
    histograms: list[list[int]] = []
    for _, _, distance, path in records:
        if len(path) != distance:
            raise RuntimeError("BFS path length does not match distance")
        for category in path:
            path_categories.extend(category)
        path_ptr.append(path_ptr[-1] + len(path))
        histogram = [0] * offsets[-1]
        for category in path:
            for column, value in enumerate(category):
                histogram[offsets[column] + value] += 1
        histograms.append(histogram)

    result: dict[str, torch.Tensor] = {
        PATH_PAIR_SRC: torch.tensor(pair_sources, dtype=torch.long),
        PATH_PAIR_DST: torch.tensor(pair_targets, dtype=torch.long),
        PATH_PAIR_DISTANCE: torch.tensor(pair_distances, dtype=torch.long),
        PATH_PAIR_PTR: torch.tensor(path_ptr, dtype=torch.long),
        PATH_EDGE_ATTR: torch.tensor(path_categories, dtype=torch.long).reshape(-1, len(dimensions)),
    }
    if include_path_hist:
        result[PATH_PAIR_HIST] = torch.tensor(
            histograms, dtype=torch.int16
        ).reshape(-1, offsets[-1])

    # These aliases make the pure function convenient for small callers while
    # keeping the sidecar/runtime field names explicit and unambiguous.
    result["pair_src"] = result[PATH_PAIR_SRC]
    result["pair_dst"] = result[PATH_PAIR_DST]
    result["distance"] = result[PATH_PAIR_DISTANCE]
    result["path_ptr"] = result[PATH_PAIR_PTR]
    result["edge_attr"] = result[PATH_EDGE_ATTR]
    if include_path_hist:
        result["path_hist"] = result[PATH_PAIR_HIST]
    return result


def path_histogram(
    path_ptr: torch.Tensor,
    path_edge_attr: torch.Tensor,
    *,
    bond_feature_dims: Sequence[int] | None = None,
) -> torch.Tensor:
    """Derive flattened per-path category counts from CSR path sequences."""

    dimensions = tuple(bond_feature_dims or get_bond_feature_dims())
    offsets = _histogram_offsets(dimensions)
    pointer = torch.as_tensor(path_ptr, dtype=torch.long).reshape(-1)
    categories = torch.as_tensor(path_edge_attr, dtype=torch.long)
    if categories.numel() == 0:
        categories = categories.reshape(0, len(dimensions))
    if categories.ndim != 2 or categories.shape[1] != len(dimensions):
        raise ValueError("path_edge_attr must have shape [path edges, 3]")
    if pointer.numel() < 1 or int(pointer[0]) != 0:
        raise ValueError("path CSR must start at zero")
    if bool((pointer[1:] < pointer[:-1]).any()) or int(pointer[-1]) != categories.shape[0]:
        raise ValueError("path CSR pointer bounds are invalid")
    pairs = pointer.numel() - 1
    result = torch.zeros((pairs, offsets[-1]), dtype=torch.long)
    lengths = pointer[1:] - pointer[:-1]
    pair_index = torch.repeat_interleave(torch.arange(pairs), lengths)
    for column, dimension in enumerate(dimensions):
        values = categories[:, column]
        if values.numel() and bool((values < 0).any() or (values >= dimension).any()):
            raise ValueError("path category outside OGB bond dimensions")
        if values.numel():
            result.index_put_(
                (pair_index, values + offsets[column]),
                torch.ones(values.shape[0], dtype=torch.long),
                accumulate=True,
            )
    return result


def path_mean_from_bond_tables(
    path_lengths: torch.Tensor,
    path_counts: torch.Tensor | None,
    bond_tables: Sequence[torch.Tensor] | torch.Tensor | Any,
) -> torch.Tensor:
    """Compute the parameter-free G2 mean using existing bond tables.

    ``bond_tables`` may be the three embedding weight tensors, a stacked
    ``[3, categories, channels]`` tensor, or an OGB ``BondEncoder`` exposing
    ``bond_embedding_list``.  The sum of categorical table lookups is divided
    by the path length; no hop-position matrix or new parameter is involved.
    """

    if hasattr(bond_tables, "bond_embedding_list"):
        tables = [embedding.weight for embedding in bond_tables.bond_embedding_list]
    elif torch.is_tensor(bond_tables):
        if bond_tables.ndim != 3 or bond_tables.shape[0] != 3:
            raise ValueError("stacked bond tables must have shape [3, categories, channels]")
        tables = [bond_tables[index] for index in range(3)]
    else:
        tables = list(bond_tables)
    dimensions = get_bond_feature_dims()
    if len(tables) != len(dimensions):
        raise ValueError("expected one existing bond table per OGB category")
    channels = int(tables[0].shape[1])
    for table, dimension in zip(tables, dimensions, strict=True):
        if table.ndim != 2 or table.shape[0] < dimension or table.shape[1] != channels:
            raise ValueError("bond table shape does not match OGB feature dimensions")
    lengths = torch.as_tensor(path_lengths, dtype=torch.long).reshape(-1)
    if bool((lengths <= 0).any()):
        raise ValueError("path lengths must be positive")
    if path_counts is None:
        raise ValueError("path_counts or CSR-derived counts are required")
    counts = torch.as_tensor(path_counts)
    if counts.ndim != 2 or counts.shape != (lengths.numel(), sum(dimensions)):
        raise ValueError("path_counts has the wrong flattened OGB dimension")
    output = tables[0].new_zeros((lengths.numel(), channels))
    offsets = _histogram_offsets(dimensions)
    for table, left, right in zip(tables, offsets[:-1], offsets[1:], strict=True):
        output = output + counts[:, left:right].to(dtype=table.dtype) @ table[: right - left]
    return output / lengths.to(dtype=output.dtype).unsqueeze(1)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, map_location="cpu")


def prepare_degree_initial_state(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Prepare the G1 tensor-only initial state from the pinned frozen input."""

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if input_path == output_path:
        raise ValueError("G1 output must not overwrite the frozen input state")
    input_sha256 = digest(input_path)
    if input_sha256 != INITIAL_STATE_SHA256:
        raise RuntimeError(
            "frozen GPTrans initial-state artifact changed: "
            f"observed={input_sha256} expected={INITIAL_STATE_SHA256}"
        )
    payload = _torch_load(input_path)
    if not isinstance(payload, Mapping):
        raise ValueError("initial-state payload must be a mapping")
    if payload.get("format") != INITIAL_STATE_FORMAT:
        raise RuntimeError("frozen GPTrans initial-state format changed")
    raw_state = payload.get("model_state")
    if not isinstance(raw_state, Mapping):
        raise ValueError("initial-state payload lacks model_state")
    base_state_sha256 = state_digest(raw_state)
    if payload.get("state_sha256") != base_state_sha256:
        raise RuntimeError("frozen initial-state payload has an inconsistent state hash")
    if base_state_sha256 != INITIAL_MODEL_SHA256:
        raise RuntimeError(
            "frozen GPTrans model state changed: "
            f"observed={base_state_sha256} expected={INITIAL_MODEL_SHA256}"
        )
    missing = [name for name in _SCALED_KEYS if name not in raw_state]
    if missing:
        raise ValueError(f"initial state lacks degree encoder tensors: {missing}")

    scaled_state: dict[str, torch.Tensor] = {}
    for name, value in raw_state.items():
        if not torch.is_tensor(value):
            raise TypeError(f"initial state entry is not a tensor: {name}")
        scaled_state[name] = value.detach().cpu().clone()
    for name in _SCALED_KEYS:
        if not scaled_state[name].is_floating_point():
            raise TypeError(f"degree encoder tensor is not floating point: {name}")
        scaled_state[name].mul_(DEGREE_SCALE)

    scaled_state_sha256 = state_digest(scaled_state)
    scaled_payload = dict(payload)
    scaled_payload.update(
        {
            "format": INITIAL_SCALED_FORMAT,
            "base_format": INITIAL_STATE_FORMAT,
            "base_input_sha256": input_sha256,
            "base_state_sha256": base_state_sha256,
            "degree_scale": DEGREE_SCALE,
            "scaled_parameter_names": list(_SCALED_KEYS),
            "state_sha256": scaled_state_sha256,
            "model_state": scaled_state,
        }
    )
    atomic_torch_save(output_path, scaled_payload)
    return {
        "format": INITIAL_SCALED_FORMAT,
        "output_path": str(output_path),
        "output_sha256": digest(output_path),
        "base_input_sha256": input_sha256,
        "base_state_sha256": base_state_sha256,
        "state_sha256": scaled_state_sha256,
        "degree_scale": DEGREE_SCALE,
        "scaled_parameter_names": list(_SCALED_KEYS),
        "tensor_count": len(scaled_state),
    }


def _accepted_shard_specs(assets: Any) -> list[dict[str, Any]]:
    records = list(assets.manifest.get("geometry_shards", []))
    expected_roles = ("train", "train", "development")
    if len(records) != len(expected_roles):
        raise RuntimeError("accepted 100K sidecar requires two train and one development shard")
    for record, role in zip(records, expected_roles, strict=True):
        if record.get("role") != role or int(record.get("rows", -1)) != 50_000:
            raise RuntimeError("accepted sidecar shard layout changed")
    return records


def _source_index(graph: Any) -> int:
    for name in ("source_idx", "source_index", "row_index"):
        value = getattr(graph, name, None)
        if value is None:
            continue
        tensor = torch.as_tensor(value).reshape(-1)
        if tensor.numel() != 1:
            raise RuntimeError(f"graph {name} must contain one source index")
        return int(tensor.item())
    raise RuntimeError("graph lacks mandatory source_index identity")


def _graph_node_count(graph: Any) -> int:
    value = getattr(graph, "x", None)
    if value is None or torch.as_tensor(value).ndim < 1:
        raise RuntimeError("graph lacks node features")
    count = int(torch.as_tensor(value).shape[0])
    if count < 1:
        raise RuntimeError("empty molecular graph is not accepted")
    return count


def _build_shard_payload(
    shard: Any,
    record: Mapping[str, Any],
    *,
    include_path_hist: bool,
    start: int = 0,
    end: int | None = None,
) -> dict[str, Any]:
    dimensions = get_bond_feature_dims()
    histogram_width = sum(dimensions)
    shard_length = len(shard)
    if end is None:
        end = shard_length
    if not (0 <= int(start) < int(end) <= shard_length):
        raise ValueError("invalid bounded shard slice")
    start, end = int(start), int(end)
    expected_rows = end - start
    if int(record["rows"]) != expected_rows:
        raise RuntimeError("chunk row count changed")
    source_indices: list[int] = []
    node_counts: list[int] = []
    graph_node_ptr = [0]
    graph_pair_ptr = [0]
    pair_path_ptr = [0]
    pair_sources: list[int] = []
    pair_targets: list[int] = []
    pair_distances: list[int] = []
    path_categories: list[int] = []
    path_histograms: list[list[int]] = []
    node_total = pair_total = path_edge_total = 0

    for position in range(start, end):
        graph = shard[position]
        observed_source = _source_index(graph)
        expected_source = int(record["source_idx_min"]) + position - start
        if observed_source != expected_source:
            raise RuntimeError(
                f"source_index changed at shard row {position}: "
                f"observed={observed_source} expected={expected_source}"
            )
        node_count = _graph_node_count(graph)
        paths = graph_paths(
            node_count,
            (getattr(graph, "edge_index"), getattr(graph, "edge_attr")),
            max_distance=MAX_DISTANCE,
            include_path_hist=include_path_hist,
        )
        pair_count = int(paths[PATH_PAIR_SRC].numel())
        path_count = int(paths[PATH_EDGE_ATTR].shape[0])
        source_indices.append(observed_source)
        node_counts.append(node_count)
        pair_sources.extend(paths[PATH_PAIR_SRC].tolist())
        pair_targets.extend(paths[PATH_PAIR_DST].tolist())
        pair_distances.extend(paths[PATH_PAIR_DISTANCE].tolist())
        local_ptr = paths[PATH_PAIR_PTR].tolist()
        pair_path_ptr.extend(path_edge_total + int(value) for value in local_ptr[1:])
        path_categories.extend(paths[PATH_EDGE_ATTR].reshape(-1).tolist())
        if include_path_hist:
            path_histograms.extend(paths[PATH_PAIR_HIST].tolist())
        node_total += node_count
        pair_total += pair_count
        path_edge_total += path_count
        graph_node_ptr.append(node_total)
        graph_pair_ptr.append(pair_total)

    payload: dict[str, Any] = {
        "format": SHARD_FORMAT,
        "max_distance": MAX_DISTANCE,
        "directed": True,
        "neighbor_order": "ascending",
        "pair_order": "source-distance-target",
        "bond_feature_dims": list(dimensions),
        "source_index": torch.tensor(source_indices, dtype=torch.long),
        "graph_node_count": torch.tensor(node_counts, dtype=torch.long),
        "graph_node_ptr": torch.tensor(graph_node_ptr, dtype=torch.long),
        "graph_pair_ptr": torch.tensor(graph_pair_ptr, dtype=torch.long),
        PATH_PAIR_SRC: torch.tensor(pair_sources, dtype=torch.long),
        PATH_PAIR_DST: torch.tensor(pair_targets, dtype=torch.long),
        PATH_PAIR_DISTANCE: torch.tensor(pair_distances, dtype=torch.long),
        PATH_PAIR_PTR: torch.tensor(pair_path_ptr, dtype=torch.long),
        PATH_EDGE_ATTR: torch.tensor(path_categories, dtype=torch.int16).reshape(-1, len(dimensions)),
        "include_path_hist": include_path_hist,
    }
    if include_path_hist:
        payload[PATH_PAIR_HIST] = torch.tensor(
            path_histograms, dtype=torch.int16
        ).reshape(-1, histogram_width)
    return payload


def _load_one_shard(path: Path) -> Any:
    # Reuse the frozen loader's InMemoryDataset handling, but load one shard at
    # a time so a CPU preparation process does not retain all 150K rows.
    _, shards = _load_datasets((Path(path),))
    if len(shards) != 1:
        raise RuntimeError("frozen graph loader returned an unexpected shard count")
    return shards[0]


def build_sidecar(
    dataset_root: Path,
    output: Path,
    *,
    include_path_hist: bool = True,
) -> dict[str, Any]:
    """Build the complete label-free train/internal-dev G2 sidecar."""

    dataset_root = Path(dataset_root).resolve()
    output = Path(output).resolve()
    manifest_path = dataset_root / "manifest.json"
    assets = validate_fixed_assets(dataset_root, manifest_path, verify_content=True)
    records = _accepted_shard_specs(assets)
    if not isinstance(include_path_hist, bool):
        raise TypeError("include_path_hist must be bool")
    dimensions = get_bond_feature_dims()
    output.mkdir(parents=True, exist_ok=True)
    root_manifest: dict[str, Any] = {
        "format": SIDECAR_FORMAT,
        "status": "building",
        "dataset_manifest_sha256": digest(manifest_path),
        "dataset_identity": assets.manifest["identity"],
        "roles": {"train": 100_000, "development": 50_000},
        "rows": 150_000,
        "chunk_rows": CHUNK_ROWS,
        "input_shard_count": 3,
        "chunks_per_input_shard": 50,
        "chunk_count": 150,
        "max_distance": MAX_DISTANCE,
        "directed": True,
        "neighbor_order": "ascending",
        "pair_order": "source-distance-target",
        "bond_feature_dims": list(dimensions),
        "include_path_hist": include_path_hist,
        "labels_read": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
        "training_executed": False,
        "model_inference_executed": False,
        "shards": [],
    }
    atomic_json(output / "manifest.json", root_manifest)

    paths = [*assets.train_paths, *assets.development_paths]
    chunk_ordinal = 0
    for input_shard_ordinal, (record, path) in enumerate(zip(records, paths, strict=True)):
        shard = _load_one_shard(path)
        for start in range(0, int(record["rows"]), CHUNK_ROWS):
            end = min(start + CHUNK_ROWS, int(record["rows"]))
            chunk_record = dict(record)
            chunk_record.update(
                {
                    "rows": end - start,
                    "source_idx_min": int(record["source_idx_min"]) + start,
                    "source_idx_max": int(record["source_idx_min"]) + end - 1,
                    "input_shard_ordinal": input_shard_ordinal,
                    "input_shard_row_start": start,
                    "input_shard_row_end": end,
                }
            )
            payload = _build_shard_payload(
                shard,
                chunk_record,
                include_path_hist=include_path_hist,
                start=start,
                end=end,
            )
            relative_payload = Path("shards") / f"part-{chunk_ordinal:03d}.pt"
            relative_metadata = Path("shards") / f"part-{chunk_ordinal:03d}.json"
            payload_path = output / relative_payload
            metadata_path = output / relative_metadata
            atomic_torch_save(payload_path, payload)
            shard_record: dict[str, Any] = {
                "ordinal": chunk_ordinal,
                "input_shard_ordinal": input_shard_ordinal,
                "input_shard_row_start": start,
                "input_shard_row_end": end,
                "role": record["role"],
                "file": relative_payload.as_posix(),
                "metadata_file": relative_metadata.as_posix(),
                "rows": end - start,
                "source_idx_min": int(record["source_idx_min"]) + start,
                "source_idx_max": int(record["source_idx_min"]) + end - 1,
                "source_index_sha256": _tensor_digest(payload["source_index"]),
                "node_count": int(payload["graph_node_count"].sum()),
                "pairs": int(payload[PATH_PAIR_SRC].numel()),
                "path_edges": int(payload[PATH_EDGE_ATTR].shape[0]),
                "bytes": payload_path.stat().st_size,
                "sha256": digest(payload_path),
            }
            atomic_json(metadata_path, {"format": SHARD_FORMAT, **shard_record})
            shard_record["metadata_sha256"] = digest(metadata_path)
            root_manifest["shards"].append(shard_record)
            atomic_json(output / "manifest.json", root_manifest)
            chunk_ordinal += 1
        del shard

    root_manifest["status"] = "complete"
    root_manifest["complete"] = True
    atomic_json(output / "manifest.json", root_manifest)
    return root_manifest


def _safe_sidecar_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise RuntimeError(f"sidecar path escaped root: {relative}") from error
    return path


def _as_1d_tensor(payload: Mapping[str, Any], key: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    if key not in payload or not torch.is_tensor(payload[key]):
        raise RuntimeError(f"sidecar payload lacks tensor {key}")
    value = payload[key]
    if dtype is not None and value.dtype != dtype:
        raise RuntimeError(f"sidecar tensor {key} has dtype {value.dtype}, expected {dtype}")
    return value.reshape(-1)


def _validate_payload(
    payload: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    include_path_hist: bool,
) -> dict[str, int]:
    dimensions = get_bond_feature_dims()
    histogram_width = sum(dimensions)
    if payload.get("format") != SHARD_FORMAT:
        raise RuntimeError("sidecar shard format changed")
    for key, expected in (
        ("max_distance", MAX_DISTANCE),
        ("directed", True),
        ("neighbor_order", "ascending"),
        ("pair_order", "source-distance-target"),
        ("bond_feature_dims", list(dimensions)),
        ("include_path_hist", include_path_hist),
    ):
        if payload.get(key) != expected:
            raise RuntimeError(f"sidecar shard metadata changed: {key}")
    source_index = _as_1d_tensor(payload, "source_index", dtype=torch.long)
    node_count = _as_1d_tensor(payload, "graph_node_count", dtype=torch.long)
    node_ptr = _as_1d_tensor(payload, "graph_node_ptr", dtype=torch.long)
    graph_pair_ptr = _as_1d_tensor(payload, "graph_pair_ptr", dtype=torch.long)
    pair_src = _as_1d_tensor(payload, PATH_PAIR_SRC, dtype=torch.long)
    pair_dst = _as_1d_tensor(payload, PATH_PAIR_DST, dtype=torch.long)
    pair_distance = _as_1d_tensor(payload, PATH_PAIR_DISTANCE, dtype=torch.long)
    pair_path_ptr = _as_1d_tensor(payload, PATH_PAIR_PTR, dtype=torch.long)
    path_edge_attr = payload.get(PATH_EDGE_ATTR)
    if not torch.is_tensor(path_edge_attr):
        raise RuntimeError("sidecar payload lacks path edge categories")
    if path_edge_attr.ndim != 2 or path_edge_attr.shape[1] != len(dimensions):
        raise RuntimeError("path edge categories have the wrong shape")
    rows = int(source_index.numel())
    if node_count.numel() != rows or node_ptr.numel() != rows + 1 or graph_pair_ptr.numel() != rows + 1:
        raise RuntimeError("sidecar graph CSR lengths are inconsistent")
    if int(graph_pair_ptr[0]) != 0 or int(graph_pair_ptr[-1]) != pair_src.numel():
        raise RuntimeError("outer pair CSR bounds are inconsistent")
    if int(node_ptr[0]) != 0 or int(node_ptr[-1]) != int(node_count.sum()):
        raise RuntimeError("node CSR bounds are inconsistent")
    if bool((node_count < 1).any()) or bool((node_ptr[1:] < node_ptr[:-1]).any()):
        raise RuntimeError("invalid graph node counts")
    if bool((graph_pair_ptr[1:] < graph_pair_ptr[:-1]).any()):
        raise RuntimeError("invalid graph pair CSR")
    if pair_dst.numel() != pair_src.numel() or pair_distance.numel() != pair_src.numel():
        raise RuntimeError("pair field lengths are inconsistent")
    if pair_path_ptr.numel() != pair_src.numel() + 1:
        raise RuntimeError("path CSR must have one pointer per pair plus one")
    if int(pair_path_ptr[0]) != 0 or int(pair_path_ptr[-1]) != path_edge_attr.shape[0]:
        raise RuntimeError("path CSR bounds are inconsistent")
    if bool((pair_path_ptr[1:] < pair_path_ptr[:-1]).any()):
        raise RuntimeError("path CSR is not monotone")
    if pair_src.numel():
        if bool((pair_src < 0).any()) or bool((pair_dst < 0).any()):
            raise RuntimeError("negative path pair endpoint")
        if bool((pair_distance < 2).any()) or bool((pair_distance > MAX_DISTANCE).any()):
            raise RuntimeError("path sidecar contains a direct or over-cap pair")
        if bool((pair_path_ptr[1:] - pair_path_ptr[:-1] != pair_distance).any()):
            raise RuntimeError("path sequence length differs from distance")
    if path_edge_attr.numel():
        for column, dimension in enumerate(dimensions):
            values = path_edge_attr[:, column].long()
            if bool((values < 0).any()) or bool((values >= dimension).any()):
                raise RuntimeError("path category outside OGB dimensions")

    # Pair endpoints are local to each graph.  The explicit per-graph checks
    # also prevent an accidental global offset from being serialized into a
    # sidecar that the runtime expects to collate itself.
    for graph_index in range(rows):
        left = int(graph_pair_ptr[graph_index])
        right = int(graph_pair_ptr[graph_index + 1])
        if right > left:
            n = int(node_count[graph_index])
            if bool((pair_src[left:right] >= n).any()) or bool((pair_dst[left:right] >= n).any()):
                raise RuntimeError("path pair endpoint exceeds local graph nodes")
            if bool((pair_src[left:right] == pair_dst[left:right]).any()):
                raise RuntimeError("path sidecar contains a diagonal pair")
            source = pair_src[left:right]
            target = pair_dst[left:right]
            distance = pair_distance[left:right]
            if right - left > 1:
                previous_order = (source[:-1] > source[1:]) | (
                    (source[:-1] == source[1:])
                    & ((distance[:-1] > distance[1:]) | ((distance[:-1] == distance[1:]) & (target[:-1] > target[1:])))
                )
                if bool(previous_order.any()):
                    raise RuntimeError("path pairs are not in canonical order")
        if int(node_ptr[graph_index + 1] - node_ptr[graph_index]) != int(node_count[graph_index]):
            raise RuntimeError("node pointer does not match graph node count")

    if include_path_hist:
        path_hist = payload.get(PATH_PAIR_HIST)
        if not torch.is_tensor(path_hist) or path_hist.ndim != 2 or path_hist.shape != (pair_src.numel(), histogram_width):
            raise RuntimeError("path histogram has the wrong shape")
        derived = path_histogram(pair_path_ptr, path_edge_attr, bond_feature_dims=dimensions)
        if not torch.equal(path_hist.long(), derived):
            raise RuntimeError("path histogram does not match CSR categories")
    elif PATH_PAIR_HIST in payload:
        raise RuntimeError("path histogram is present despite disabled storage")

    expected_start = int(record["source_idx_min"])
    expected_source = torch.arange(expected_start, expected_start + rows, dtype=torch.long)
    if not torch.equal(source_index, expected_source):
        raise RuntimeError("sidecar source_index order is not fixed-data order")
    if int(record["source_idx_max"]) != expected_start + rows - 1:
        raise RuntimeError("sidecar source range is inconsistent")
    return {
        "rows": rows,
        "nodes": int(node_count.sum()),
        "pairs": int(pair_src.numel()),
        "path_edges": int(path_edge_attr.shape[0]),
    }


def _compare_tensor(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    if actual.dtype != expected.dtype or actual.shape != expected.shape or not torch.equal(actual, expected):
        raise RuntimeError(f"independent sidecar re-derivation mismatch: {label}")


def _rederive_shard(
    payload: Mapping[str, Any],
    shard: Any,
    record: Mapping[str, Any],
    *,
    include_path_hist: bool,
    start: int = 0,
    end: int | None = None,
) -> None:
    source_index = payload["source_index"]
    graph_node_count = payload["graph_node_count"]
    graph_node_ptr = payload["graph_node_ptr"]
    graph_pair_ptr = payload["graph_pair_ptr"]
    if end is None:
        end = len(shard)
    start, end = int(start), int(end)
    if not (0 <= start < end <= len(shard)) or end - start != source_index.numel():
        raise RuntimeError("independent shard chunk bounds changed")
    for local_position, position in enumerate(range(start, end)):
        graph = shard[position]
        expected_source = int(record["source_idx_min"]) + position
        if _source_index(graph) != expected_source:
            raise RuntimeError("independent source identity changed")
        node_count = _graph_node_count(graph)
        if int(graph_node_count[position]) != node_count:
            raise RuntimeError("independent node count changed")
        paths = graph_paths(
            node_count,
            (getattr(graph, "edge_index"), getattr(graph, "edge_attr")),
            max_distance=MAX_DISTANCE,
            include_path_hist=include_path_hist,
        )
        pair_left = int(graph_pair_ptr[local_position])
        pair_right = int(graph_pair_ptr[local_position + 1])
        path_left = int(payload[PATH_PAIR_PTR][pair_left])
        path_right = int(payload[PATH_PAIR_PTR][pair_right])
        expected_ptr = paths[PATH_PAIR_PTR] + path_left
        _compare_tensor(payload[PATH_PAIR_SRC][pair_left:pair_right], paths[PATH_PAIR_SRC], "pair source")
        _compare_tensor(payload[PATH_PAIR_DST][pair_left:pair_right], paths[PATH_PAIR_DST], "pair destination")
        _compare_tensor(payload[PATH_PAIR_DISTANCE][pair_left:pair_right], paths[PATH_PAIR_DISTANCE], "pair distance")
        _compare_tensor(payload[PATH_PAIR_PTR][pair_left:pair_right + 1], expected_ptr, "path pointer")
        _compare_tensor(payload[PATH_EDGE_ATTR][path_left:path_right], paths[PATH_EDGE_ATTR].to(dtype=payload[PATH_EDGE_ATTR].dtype), "path categories")
        if include_path_hist:
            _compare_tensor(payload[PATH_PAIR_HIST][pair_left:pair_right], paths[PATH_PAIR_HIST], "path histogram")
        if int(graph_node_ptr[local_position + 1] - graph_node_ptr[local_position]) != node_count:
            raise RuntimeError("independent node pointer changed")


def validate_sidecar(
    root: Path,
    dataset_root: Path | None = None,
    *,
    verify_content: bool = True,
    verify: bool = False,
    rederive: bool | None = None,
) -> dict[str, Any]:
    """Independently validate sidecar hashes/CSR and optionally rederive graphs.

    ``dataset_root`` binds the stable identity to the accepted fixed dataset.
    The default validation only checks the retrieved chunk hashes, fixed row
    coverage, and CSR invariants.  ``verify=True`` is the explicit CPU
    collector mode that performs a second pass over the source shards and
    rederives every path.  This function writes no acceptance record and never
    turns its own result into a training release.
    """

    if rederive is not None:
        # ``rederive`` was used by the first CPU-stage wrapper.  Keep it as a
        # compatibility spelling while making ``verify`` the documented
        # collector switch; GPU callers omit both and never rebuild paths.
        if verify and bool(rederive) != bool(verify):
            raise ValueError("verify and rederive disagree")
        verify = bool(rederive)
    if verify and dataset_root is None:
        raise ValueError("verify=True requires the accepted dataset_root")
    root = Path(root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != SIDECAR_FORMAT:
        raise RuntimeError("sidecar root format changed")
    if manifest.get("status") != "complete" or manifest.get("complete") is not True:
        raise RuntimeError("sidecar root is incomplete")
    dimensions = get_bond_feature_dims()
    if manifest.get("bond_feature_dims") != list(dimensions):
        raise RuntimeError("sidecar root bond dimensions changed")
    if manifest.get("max_distance") != MAX_DISTANCE or manifest.get("directed") is not True:
        raise RuntimeError("sidecar root path policy changed")
    if manifest.get("neighbor_order") != "ascending" or manifest.get("pair_order") != "source-distance-target":
        raise RuntimeError("sidecar root traversal policy changed")
    if manifest.get("rows") != 150_000 or manifest.get("roles") != {"train": 100_000, "development": 50_000}:
        raise RuntimeError("sidecar root role coverage is not train plus internal development")
    if (
        manifest.get("chunk_rows") != CHUNK_ROWS
        or manifest.get("input_shard_count") != 3
        or manifest.get("chunks_per_input_shard") != 50
        or manifest.get("chunk_count") != 150
    ):
        raise RuntimeError("sidecar root chunking policy changed")
    for flag in (
        "labels_read",
        "official_validation_role_read",
        "test_dev_role_read",
        "test_challenge_role_read",
        "training_executed",
        "model_inference_executed",
    ):
        if manifest.get(flag) is not False:
            raise RuntimeError(f"sidecar provenance flag changed: {flag}")
    include_path_hist = manifest.get("include_path_hist")
    if not isinstance(include_path_hist, bool):
        raise RuntimeError("sidecar root histogram policy is missing")
    records = manifest.get("shards")
    if not isinstance(records, list) or len(records) != 150:
        raise RuntimeError("sidecar root must contain 150 bounded chunks")

    assets = None
    source_records: list[dict[str, Any]] = []
    if dataset_root is not None:
        dataset_root = Path(dataset_root).resolve()
        assets = validate_fixed_assets(
            dataset_root, dataset_root / "manifest.json", verify_content=verify_content
        )
        source_records = _accepted_shard_specs(assets)
        if digest(dataset_root / "manifest.json") != manifest.get("dataset_manifest_sha256"):
            raise RuntimeError("sidecar fixed-dataset manifest hash changed")

    total_rows = total_nodes = total_pairs = total_path_edges = 0
    input_rows = [0, 0, 0]
    input_chunk_counts = [0, 0, 0]
    last_input_ordinal = -1
    for ordinal, record in enumerate(records):
        if not isinstance(record, dict) or int(record.get("ordinal", -1)) != ordinal:
            raise RuntimeError("sidecar shard ordinals are not contiguous")
        input_ordinal = int(record.get("input_shard_ordinal", -1))
        if input_ordinal not in (0, 1, 2) or input_ordinal < last_input_ordinal:
            raise RuntimeError("sidecar input shard ordinals are not ordered")
        last_input_ordinal = input_ordinal
        expected_role = ("train", "train", "development")[input_ordinal]
        if record.get("role") != expected_role:
            raise RuntimeError("sidecar shard role order changed")
        rows = int(record.get("rows", -1))
        row_start = int(record.get("input_shard_row_start", -1))
        row_end = int(record.get("input_shard_row_end", -1))
        expected_input_start = (0, 50_000, 100_000)[input_ordinal]
        if rows < 1 or rows > CHUNK_ROWS or row_end - row_start != rows:
            raise RuntimeError("sidecar chunk row bounds changed")
        if row_start != input_rows[input_ordinal] or row_end > 50_000:
            raise RuntimeError("sidecar chunks do not cover an input shard contiguously")
        expected_source_min = expected_input_start + row_start
        if int(record["source_idx_min"]) != expected_source_min:
            raise RuntimeError("sidecar chunk source lower bound changed")
        if int(record["source_idx_max"]) != expected_source_min + rows - 1:
            raise RuntimeError("sidecar chunk source upper bound changed")
        input_rows[input_ordinal] = row_end
        input_chunk_counts[input_ordinal] += 1
        payload_path = _safe_sidecar_path(root, str(record["file"]))
        metadata_path = _safe_sidecar_path(root, str(record["metadata_file"]))
        if not payload_path.is_file() or not metadata_path.is_file():
            raise RuntimeError("sidecar chunk or metadata is missing")
        if verify_content:
            if payload_path.stat().st_size != int(record["bytes"]) or digest(payload_path) != record["sha256"]:
                raise RuntimeError("sidecar chunk hash or size changed")
            if digest(metadata_path) != record.get("metadata_sha256"):
                raise RuntimeError("sidecar chunk metadata hash changed")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_metadata = {key: value for key, value in record.items() if key != "metadata_sha256"}
        if metadata != {"format": SHARD_FORMAT, **expected_metadata}:
            raise RuntimeError("sidecar chunk metadata does not match root")
        payload = _torch_load(payload_path)
        if not isinstance(payload, Mapping):
            raise RuntimeError("sidecar chunk is not a mapping payload")
        counts = _validate_payload(payload, record, include_path_hist=include_path_hist)
        if payload["source_index"].numel() != int(record["rows"]):
            raise RuntimeError("sidecar chunk source rows changed")
        if _tensor_digest(payload["source_index"]) != record["source_index_sha256"]:
            raise RuntimeError("sidecar source identity hash changed")
        if counts["nodes"] != int(record["node_count"]) or counts["pairs"] != int(record["pairs"]) or counts["path_edges"] != int(record["path_edges"]):
            raise RuntimeError("sidecar chunk count summary changed")
        total_rows += counts["rows"]
        total_nodes += counts["nodes"]
        total_pairs += counts["pairs"]
        total_path_edges += counts["path_edges"]

    if input_rows != [50_000, 50_000, 50_000] or input_chunk_counts != [50, 50, 50] or total_rows != 150_000:
        raise RuntimeError("sidecar does not cover the fixed 150K train/internal-dev rows")

    if assets is not None and verify:
        source_paths = [*assets.train_paths, *assets.development_paths]
        for input_ordinal, source_record in enumerate(source_records):
            shard = _load_one_shard(source_paths[input_ordinal])
            for record in records:
                if int(record["input_shard_ordinal"]) != input_ordinal:
                    continue
                payload_path = _safe_sidecar_path(root, str(record["file"]))
                payload = _torch_load(payload_path)
                _rederive_shard(
                    payload,
                    shard,
                    source_record,
                    include_path_hist=include_path_hist,
                    start=int(record["input_shard_row_start"]),
                    end=int(record["input_shard_row_end"]),
                )
            del shard
    runtime_compact_cache = _runtime_compact_cache_estimate(records)
    return {
        "format": "molgap-gptrans-author-path-sidecar-validation-v1",
        "valid": True,
        "sidecar_manifest_sha256": digest(manifest_path),
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "rows": total_rows,
        "nodes": total_nodes,
        "pairs": total_pairs,
        "path_edges": total_path_edges,
        "chunks_checked": len(records),
        "content_hashes_checked": bool(verify_content),
        "source_rederived": bool(assets is not None and verify),
        "include_path_hist": bool(include_path_hist),
        "runtime_compact_cache": runtime_compact_cache,
        "chunk_identities": [
            {
                "ordinal": int(record["ordinal"]),
                "input_shard_ordinal": int(record["input_shard_ordinal"]),
                "input_shard_row_start": int(record["input_shard_row_start"]),
                "input_shard_row_end": int(record["input_shard_row_end"]),
                "role": record["role"],
                "source_idx_min": int(record["source_idx_min"]),
                "source_idx_max": int(record["source_idx_max"]),
                "sha256": record["sha256"],
                "bytes": int(record["bytes"]),
                "metadata_sha256": record["metadata_sha256"],
            }
            for record in records
        ],
        "labels_read": False,
        "official_validation_role_read": False,
    }


class PathData(Data):
    """PyG data object with explicit local path-pair index collation."""

    def __inc__(self, key: str, value: Any, *args: Any, **kwargs: Any) -> int | torch.Tensor:
        if key == "path_pair_index":
            return int(self.num_nodes)
        return super().__inc__(key, value, *args, **kwargs)


class PathDataset(torch.utils.data.Dataset):
    """Source-index keyed view backed by one compact in-memory cache.

    The accepted shard files retain CSR path categories for independent CPU
    verification.  Runtime keeps only the six tensors needed by the model so
    shuffled access never re-reads full CSR chunks.
    """

    def __init__(
        self,
        graphs: Any,
        sidecar_root: Path,
        validation: Mapping[str, Any],
        *,
        max_cached_chunks: int = 8,
    ):
        # Keep the old keyword for callers that constructed this adapter
        # directly.  It no longer controls runtime I/O: only compact tensors
        # are loaded once below, so there is no full-payload LRU to tune.
        if int(max_cached_chunks) < 1:
            raise ValueError("max_cached_chunks must be positive")
        self.graphs = graphs
        self.sidecar_root = Path(sidecar_root).resolve()
        self.validation = dict(validation)
        self.max_cached_chunks = int(max_cached_chunks)
        manifest = json.loads((self.sidecar_root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("include_path_hist") is not True:
            raise RuntimeError("runtime path sidecar requires include_path_hist=True")
        self._records = tuple(manifest["shards"])
        self._starts = tuple(int(record["source_idx_min"]) for record in self._records)
        self.runtime_compact_cache = _runtime_compact_cache_estimate(self._records)
        if not self.runtime_compact_cache["within_cap"]:
            raise RuntimeError(
                "runtime compact sidecar exceeds the 4 GiB in-memory cap: "
                f'{self.runtime_compact_cache["estimated_bytes"]} bytes'
            )
        validated_cache = self.validation.get("runtime_compact_cache")
        if validated_cache is not None and validated_cache != self.runtime_compact_cache:
            raise RuntimeError("runtime compact cache estimate changed after validation")

        compact_payloads: list[dict[str, torch.Tensor]] = []
        histogram_width = sum(get_bond_feature_dims())
        for record in self._records:
            payload = _torch_load(_safe_sidecar_path(self.sidecar_root, record["file"]))
            if not isinstance(payload, Mapping) or payload.get("include_path_hist") is not True:
                raise RuntimeError("runtime sidecar chunk lacks accepted path histograms")
            compact: dict[str, torch.Tensor] = {}
            for key in RUNTIME_COMPACT_KEYS:
                value = payload.get(key)
                if not torch.is_tensor(value):
                    raise RuntimeError(f"runtime sidecar chunk lacks tensor {key}")
                compact[key] = value
            rows = int(compact["source_index"].numel())
            pairs = int(compact[PATH_PAIR_SRC].numel())
            if compact["graph_pair_ptr"].ndim != 1 or compact["graph_pair_ptr"].numel() != rows + 1:
                raise RuntimeError("runtime graph pair pointer shape changed")
            if compact["graph_pair_ptr"].numel() and int(compact["graph_pair_ptr"][-1]) != pairs:
                raise RuntimeError("runtime graph pair pointer bound changed")
            for key in (PATH_PAIR_DST, PATH_PAIR_DISTANCE):
                if compact[key].ndim != 1 or compact[key].numel() != pairs:
                    raise RuntimeError(f"runtime compact tensor shape changed: {key}")
            histogram = compact[PATH_PAIR_HIST]
            if histogram.ndim != 2 or histogram.shape != (pairs, histogram_width):
                raise RuntimeError("runtime path histogram shape changed")
            if histogram.dtype not in (torch.uint8, torch.int16):
                raise RuntimeError("runtime path histogram dtype changed")
            compact_payloads.append(compact)
            # Do not retain the CSR pointer/category tensors from payload.
            del payload
        self._compact_payloads = tuple(compact_payloads)

    def __len__(self) -> int:
        return len(self.graphs)

    def _compact_for(self, source_index: int) -> tuple[Mapping[str, torch.Tensor], int]:
        ordinal = bisect_right(self._starts, source_index) - 1
        if ordinal < 0 or ordinal >= len(self._records):
            raise RuntimeError(f"source_index is absent from sidecar: {source_index}")
        record = self._records[ordinal]
        if source_index > int(record["source_idx_max"]):
            raise RuntimeError(f"source_index is absent from sidecar: {source_index}")
        payload = self._compact_payloads[ordinal]
        position = source_index - int(record["source_idx_min"])
        observed = int(payload["source_index"][position])
        if observed != source_index:
            raise RuntimeError("runtime sidecar source identity changed")
        return payload, position

    def __getitem__(self, index: int) -> PathData:
        graph = self.graphs[index]
        source_index = _source_index(graph)
        payload, position = self._compact_for(source_index)
        path_left_pair = int(payload["graph_pair_ptr"][position])
        path_right_pair = int(payload["graph_pair_ptr"][position + 1])
        pair_src = payload[PATH_PAIR_SRC][path_left_pair:path_right_pair].clone()
        pair_dst = payload[PATH_PAIR_DST][path_left_pair:path_right_pair].clone()
        pair_distance = payload[PATH_PAIR_DISTANCE][path_left_pair:path_right_pair].clone()
        counts = payload[PATH_PAIR_HIST][path_left_pair:path_right_pair].clone()
        values = {key: value for key, value in graph}
        path_graph = PathData(**values)
        path_graph.path_pair_index = torch.stack((pair_src, pair_dst), dim=0)
        path_graph.path_counts = counts
        path_graph.path_lengths = pair_distance
        return path_graph


def attach_paths(graphs: Any, sidecar_root: Path) -> PathDataset:
    """Validate runtime chunks and return a lazy PyG path-augmented dataset."""

    validation = validate_sidecar(
        Path(sidecar_root),
        verify_content=True,
        verify=False,
    )
    return PathDataset(graphs, Path(sidecar_root), validation)


__all__ = [
    "CHUNK_ROWS",
    "DEGREE_SCALE",
    "MAX_DISTANCE",
    "RUNTIME_COMPACT_BYTES_CAP",
    "RUNTIME_COMPACT_KEYS",
    "RUNTIME_COMPACT_SCHEMA",
    "PathData",
    "PathDataset",
    "attach_paths",
    "build_sidecar",
    "digest",
    "get_bond_feature_dims",
    "graph_paths",
    "path_histogram",
    "path_mean_from_bond_tables",
    "prepare_degree_initial_state",
    "state_digest",
    "validate_sidecar",
]
