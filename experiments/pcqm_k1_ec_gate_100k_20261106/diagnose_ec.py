"""Training-topology-only CPU viability check for the frozen EC question."""
import json
import statistics
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

from torch_geometric.data import Batch
from molgap.family_training_runner import read_recipe
from molgap.family_training_data import load_training_data
from molgap.k1_ec_gate import edge_statistics


def main():
    root = Path(__file__).resolve().parent
    started, cpu_started = time.perf_counter(), time.process_time()
    _, recipe = read_recipe(root / "control_recipe.json")
    import torch
    torch.set_num_threads(1)
    chain = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]])
    _, ec = edge_statistics(chain, 4)
    assert ec[0, 1] == 1 and ec[0, 0] == 0
    triangle = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    assert not edge_statistics(triangle, 3)[1].count_nonzero()
    data = load_training_data(Path("D:/molgap-data"), replace(recipe.dataset, development=False))
    sampled = [i * len(data.train) // 1024 for i in range(1024)]
    informative, conditional, totals, timing = 0, defaultdict(set), defaultdict(int), []
    nonzero, edges = [0, 0], 0
    for start in range(0, 1024, 128):
        batch = Batch.from_data_list([data.train[i] for i in sampled[start:start + 128]])
        began = time.perf_counter()
        _, expansion = edge_statistics(batch.edge_index, batch.num_nodes)
        timing.append(time.perf_counter() - began)
        source, target = batch.edge_index.tolist()
        degrees = torch.bincount(batch.edge_index[1], minlength=batch.num_nodes).tolist()
        membership = batch.batch.tolist()
        incoming = defaultdict(set)
        values = expansion.long().T.tolist()
        for u, v, value in zip(source, target, values):
            key = (degrees[u], degrees[v])
            conditional[key].add(tuple(value))
            totals[key] += 1
            incoming[v].add(tuple(value))
        informative += len({membership[v] for v, values in incoming.items() if len(values) > 1})
        for depth in range(2):
            nonzero[depth] += int(expansion[depth].count_nonzero())
        edges += len(source)
    diverse = [list(key) for key, values in conditional.items() if totals[key] >= 20 and len(values) > 1]
    median = statistics.median(timing)
    report = {"sampled_train_indices": sampled, "graphs": 1024, "edges": edges,
        "informative_graphs": informative, "informative_fraction": informative / 1024,
        "conditional_degree_groups_with_ec_variation": diverse,
        "ec2_ec3_nonzero_fractions": [n / edges for n in nonzero],
        "batch_topology_wall_seconds": timing, "median_batch_topology_wall_seconds": median,
        "wall_seconds": time.perf_counter() - started,
        "cpu_process_seconds": time.process_time() - cpu_started,
        "gpu_time": "not_applicable", "role": "training_membership/topology_only",
        "development_consumed": False, "labels_used": False,
        "release": informative >= 205 and len(diverse) >= 3 and median <= 0.05}
    print(json.dumps(report, sort_keys=True))
    return 0 if report["release"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
