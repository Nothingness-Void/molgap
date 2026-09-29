"""Label-free GPTrans input-scale diagnostic on fixed PCQM train graphs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .gptrans_path_real_preflight import (
    MANIFEST_SHA256,
    SAMPLE_PER_SHARD,
    _sample_positions,
    atomic_json,
    sha256_file,
)


INITIAL_STATE_SHA256 = "9205fc0f0f97f1cc1cea84ab4bd24206274a00c7d84d26366497feee1710c20c"
INITIAL_MODEL_SHA256 = "8988db8659c6c7e2b27401312f43684215c34cd8d69309aed7ce946ee9cb1ec6"


def _state_digest(state: dict) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def component_energy(x, edge_index, state) -> dict[str, float | int]:
    """Measure initial atom and degree embedding energy on actual categories."""
    import torch

    if x.ndim != 2 or x.shape[1] != 9 or x.shape[0] < 1:
        raise ValueError("expected nonempty OGB nine-column atom graph")
    if edge_index.shape[0] != 2:
        raise ValueError("expected PyG edge_index")
    nodes = int(x.shape[0])
    atom = torch.zeros((nodes, 256), dtype=torch.float32)
    for column in range(9):
        table = state[f"atom_encoder.atom_embedding_list.{column}.weight"]
        category = x[:, column].long()
        if bool((category < 0).any()) or bool((category >= table.shape[0]).any()):
            raise ValueError("atom category outside frozen table")
        atom += table[category]
    adjacency = torch.zeros((nodes, nodes), dtype=torch.bool)
    if edge_index.numel():
        source, target = edge_index.long()
        if bool((source < 0).any() or (source >= nodes).any() or (target < 0).any() or (target >= nodes).any()):
            raise ValueError("bond endpoint outside molecule")
        adjacency[source, target] = True
    degree = adjacency.sum(-1).clamp_max(511)
    degree_vector = (
        state["in_degree_encoder.weight"][degree]
        + state["out_degree_encoder.weight"][degree]
    )
    return {
        "nodes": nodes,
        "atom_squared_sum": float(atom.double().square().sum()),
        "degree_squared_sum": float(degree_vector.double().square().sum()),
        "cross_sum": float((atom.double() * degree_vector.double()).sum()),
        "degree_max": int(degree.max()),
    }


def run(dataset_root: Path, initial_state_path: Path, output: Path) -> dict:
    import torch
    from torch_geometric.data import InMemoryDataset

    manifest_path = dataset_root / "manifest.json"
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise RuntimeError("fixed 100K manifest changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("identity", {}).get("name") != "ogb-train-100k":
        raise RuntimeError("wrong fixed dataset")
    records = [item for item in manifest["geometry_shards"] if item["role"] == "train"]
    if len(records) != 2:
        raise RuntimeError("expected two fixed train shards")
    if sha256_file(initial_state_path) != INITIAL_STATE_SHA256:
        raise RuntimeError("frozen initial state artifact changed")
    payload = torch.load(initial_state_path, map_location="cpu", weights_only=False)
    state = payload["model_state"]
    if payload.get("format") != "molgap-gptrans-t-seed42-initial-state-v1":
        raise RuntimeError("initial state format changed")
    if payload.get("state_sha256") != _state_digest(state) or payload["state_sha256"] != INITIAL_MODEL_SHA256:
        raise RuntimeError("frozen model state identity changed")

    class PackedGraphs(InMemoryDataset):
        def __init__(self, path: Path):
            super().__init__(root=None)
            self.data, self.slices = torch.load(path, map_location="cpu", weights_only=False)

    summary = {
        "format": "molgap-gptrans-initial-scale-preflight-v1",
        "status": "RUNNING", "manifest_sha256": MANIFEST_SHA256,
        "initial_state_sha256": INITIAL_STATE_SHA256,
        "initial_model_sha256": INITIAL_MODEL_SHA256,
        "sample_per_shard": SAMPLE_PER_SHARD,
        "labels_read": False, "training_executed": False,
        "official_evaluation_read": False, "shards": [],
    }
    atomic_json(output / "summary.json", summary)
    for ordinal, record in enumerate(records):
        path = (dataset_root / record["file"]).resolve()
        if not path.is_relative_to(dataset_root.resolve()):
            raise RuntimeError("shard escaped fixed dataset root")
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise RuntimeError("fixed train shard identity changed")
        graphs = PackedGraphs(path)
        positions = _sample_positions(len(graphs))
        totals = {key: 0.0 for key in ("atom_squared_sum", "degree_squared_sum", "cross_sum")}
        nodes = 0
        highest_degree = 0
        indices: list[int] = []
        for position in positions:
            graph = graphs[position]
            source_idx = getattr(graph, "source_idx", getattr(graph, "row_index", None))
            if source_idx is None or int(source_idx.reshape(-1)[0]) != int(record["source_idx_min"]) + position:
                raise RuntimeError("fixed graph source row changed")
            indices.append(int(source_idx.reshape(-1)[0]))
            energy = component_energy(graph.x, graph.edge_index, state)
            for key in totals:
                totals[key] += energy[key]
            nodes += energy["nodes"]
            highest_degree = max(highest_degree, energy["degree_max"])
        item = {
            "shard_ordinal": ordinal, "shard_sha256": record["sha256"],
            "sampled_rows": len(positions), "nodes": nodes,
            "source_indices_sha256": hashlib.sha256(
                json.dumps(indices, separators=(",", ":")).encode("ascii")
            ).hexdigest(),
            "source_idx_first": indices[0], "source_idx_last": indices[-1],
            "degree_max": highest_degree, "energy": totals,
        }
        atomic_json(output / f"train_shard_{ordinal}.json", item)
        summary["shards"].append(item)
        atomic_json(output / "summary.json", summary)
        del graphs
    totals = {key: sum(item["energy"][key] for item in summary["shards"])
              for key in ("atom_squared_sum", "degree_squared_sum", "cross_sum")}
    total_nodes = sum(item["nodes"] for item in summary["shards"])
    atom_energy = totals["atom_squared_sum"] / (total_nodes * 256)
    degree_energy = totals["degree_squared_sum"] / (total_nodes * 256)
    cross_energy = totals["cross_sum"] / (total_nodes * 256)
    summary["input_energy"] = {
        "atom_mean_square": atom_energy,
        "degree_mean_square": degree_energy,
        "atom_degree_cross_mean": cross_energy,
        "degree_fraction_without_cross": degree_energy / (atom_energy + degree_energy),
        "observed_total_mean_square": atom_energy + degree_energy + 2 * cross_energy,
        "sampled_nodes": total_nodes,
    }
    summary["status"] = "COMPLETE"
    atomic_json(output / "summary.json", summary)
    return summary
