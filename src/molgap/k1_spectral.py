"""One topology-only full-spectrum residual for the frozen K1 screen.

Complete eigensystems and a scalar smooth frequency filter preserve rotation
invariance within repeated eigenspaces. CPU cache construction precedes the
accelerator qualification; no molecular geometry is read or constructed.
"""
from pathlib import Path
import json
import time
import hashlib
import numpy as np
from .screen_policy import canonical_fingerprint
from .training_reproducibility import atomic_json, atomic_torch_save, sha256_file
from .v4_runtime import normalized_source_sha256


CONFIG = {"layer": 6, "latent_channels": 64, "frequency_basis": 8,
          "laplacian": "undirected-unweighted-symmetric-normalized",
          "eigensystem": "full-float64-cpu-to-float32",
          "filter": "per-channel-gaussian-eigenvalue", "gaussian_width": 0.25,
          "return_initialization": "zero", "seed": 42}


def configuration():
    return dict(CONFIG)


def eigensystem(graph):
    """CPU-only topology preprocessing; every eigenvector is retained."""
    n = int(graph.num_nodes)
    adjacency = np.zeros((n, n), dtype=np.float64)
    edges = graph.edge_index.detach().cpu().numpy()
    adjacency[edges[0], edges[1]] = 1.0
    adjacency = np.maximum(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, 0.0)
    degree = adjacency.sum(1)
    inv = np.zeros_like(degree)
    inv[degree > 0] = degree[degree > 0] ** -0.5
    laplacian = np.diag((degree > 0).astype(np.float64)) - inv[:, None] * adjacency * inv[None, :]
    values, vectors = np.linalg.eigh(laplacian)
    return values.astype(np.float32), vectors.astype(np.float32)


class SpectralDataset:
    def __init__(self, base, vectors, values, slices):
        self.base, self.vectors, self.values, self.slices = base, vectors, values, slices

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        graph = self.base[index].clone()
        begin, end = int(self.slices[index]), int(self.slices[index + 1])
        graph.spectral_vectors = self.vectors[begin:end]
        graph.spectral_values = self.values[begin:end, None]
        return graph


def spectral_roles(roles, directory, *, fixed_manifest_sha256, create=False):
    import torch
    directory = Path(directory)
    cache = directory / "spectral_cache.pt"
    metadata = directory / "spectral_cache_manifest.json"
    identity = {"configuration": configuration(), "fixed_manifest_sha256": fixed_manifest_sha256,
                "algorithm_source_sha256": normalized_source_sha256(Path(__file__)),
                "role_rows": {name: len(dataset) for name, dataset in roles.items()}}
    if not cache.exists():
        if not create:
            raise FileNotFoundError("Qualified spectral cache is required; do not rebuild during training")
        started = time.perf_counter()
        process_started = time.process_time()
        sizes = {name: [int(dataset[i].num_nodes) for i in range(len(dataset))]
                 for name, dataset in roles.items()}
        maximum = max(max(values) for values in sizes.values())
        expected_bytes = sum(sum(values) for values in sizes.values()) * (maximum + 1) * 4
        if maximum > 128 or expected_bytes > 768 * 1024 ** 2:
            raise RuntimeError("Full spectral cache exceeds declared CPU memory budget")
        payload = {}
        topology_hashes, source_row_hashes = {}, {}
        for name, dataset in roles.items():
            offsets = np.concatenate(([0], np.cumsum(sizes[name]))).astype(np.int64)
            vectors = np.zeros((int(offsets[-1]), maximum), dtype=np.float32)
            values = np.zeros(int(offsets[-1]), dtype=np.float32)
            topology_digest, row_digest = hashlib.sha256(), hashlib.sha256()
            source_rows = []
            for i in range(len(dataset)):
                graph = dataset[i]
                source_index = int(graph.source_idx.view(-1)[0])
                source_rows.append(source_index)
                row_digest.update(np.asarray([source_index], dtype="<i8").tobytes())
                topology_digest.update(np.asarray([source_index, int(graph.num_nodes)], dtype="<i8").tobytes())
                topology_digest.update(graph.edge_index.detach().cpu().contiguous().numpy().astype("<i8").tobytes())
                val, vec = eigensystem(graph)
                begin, end = offsets[i:i + 2]
                vectors[begin:end, :len(val)] = vec
                values[begin:end] = val
            payload[name] = {"vectors": torch.from_numpy(vectors), "values": torch.from_numpy(values),
                             "slices": torch.from_numpy(offsets), "source_rows": torch.tensor(source_rows)}
            topology_hashes[name] = topology_digest.hexdigest()
            source_row_hashes[name] = row_digest.hexdigest()
        atomic_torch_save(cache, payload)
        atomic_json(metadata, {"format": "molgap-k1-spectral-cache-v1", "identity": identity,
                              "cache_sha256": sha256_file(cache), "maximum_nodes": maximum,
                              "tensor_bytes": expected_bytes,
                              "ordered_topology_sha256": topology_hashes,
                              "ordered_source_idx_sha256": source_row_hashes,
                              "cpu_process_seconds": time.process_time() - process_started,
                              "wall_seconds": time.perf_counter() - started,
                              "geometry_model_input": False})
    manifest = json.loads(metadata.read_text(encoding="utf-8"))
    if manifest["identity"] != identity or sha256_file(cache) != manifest["cache_sha256"]:
        raise ValueError("Spectral cache identity or byte hash changed")
    payload = torch.load(cache, map_location="cpu", weights_only=True)
    result = {}
    for name, dataset in roles.items():
        item = payload[name]
        source_rows = item.pop("source_rows")
        expected_start = 0 if name == "train" else 100_000
        if not torch.equal(source_rows, torch.arange(expected_start, expected_start + len(dataset))):
            raise ValueError("Spectral cache source row order changed")
        result[name] = SpectralDataset(dataset, **item)
    return result


def attach_spectral(model):
    import torch
    import torch.nn as nn
    from torch_geometric.utils import to_dense_batch

    class SpectralResidual(nn.Module):
        def __init__(self):
            super().__init__()
            self.down = nn.Linear(192, 64, bias=False)
            self.frequency_coefficients = nn.Parameter(torch.zeros(CONFIG["frequency_basis"], 64))
            self.register_buffer("frequency_centers", torch.linspace(0.0, 2.0, CONFIG["frequency_basis"]))
            self.up = nn.Linear(64, 192, bias=False)
            nn.init.zeros_(self.up.weight)
            self.current_batch = None

        def forward(self, hidden):
            graph = self.current_batch
            if graph is None:
                raise RuntimeError("Spectral residual requires its topology cache batch")
            dense, _ = to_dense_batch(self.down(hidden), graph.batch)
            vectors, mask = to_dense_batch(graph.spectral_vectors, graph.batch)
            vectors = vectors[:, :, :dense.shape[1]]
            eigenvalues, _ = to_dense_batch(graph.spectral_values, graph.batch)
            basis = torch.exp(-0.5 * ((eigenvalues[..., 0, None] - self.frequency_centers) /
                                    CONFIG["gaussian_width"]) ** 2)
            transfer = basis @ self.frequency_coefficients
            # Identity frequency term permits a nonzero return gradient from step one.
            spectral = torch.bmm(vectors.transpose(1, 2), dense)
            returned = torch.bmm(vectors, spectral * (1.0 + transfer))
            return hidden + self.up(returned[mask])

    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(CONFIG["seed"])
        model.k1_spectral = SpectralResidual()

    def residual_hook(module, inputs, output):
        return model.k1_spectral(output)

    model.local_blocks[CONFIG["layer"] - 1].register_forward_hook(residual_hook)
    return model


def qualify_invariance(model, graph_batch, forward):
    """Assigned-device release qualification on a training-only batch."""
    import torch
    model.eval()
    with torch.no_grad():
        original = forward(model, graph_batch)
        sign_batch = graph_batch.clone()
        signs = torch.ones(sign_batch.spectral_vectors.shape[1], device=original.device)
        signs[::2] = -1
        sign_batch.spectral_vectors = sign_batch.spectral_vectors * signs
        sign_delta = float((forward(model, sign_batch) - original).abs().max())
        rotated = graph_batch.clone()
        rotations = 0
        for begin, end in zip(graph_batch.ptr[:-1].tolist(), graph_batch.ptr[1:].tolist()):
            values = graph_batch.spectral_values[begin:end, 0]
            for index in range(len(values) - 1):
                if abs(float(values[index + 1] - values[index])) <= 1e-6:
                    columns = rotated.spectral_vectors[begin:end, index:index + 2].clone()
                    rotation = columns.new_tensor([[0.6, -0.8], [0.8, 0.6]])
                    rotated.spectral_vectors[begin:end, index:index + 2] = columns @ rotation
                    rotations += 1
                    break
        rotation_delta = float((forward(model, rotated) - original).abs().max())
        permutation = torch.cat([torch.arange(end - 1, begin - 1, -1, device=original.device)
                                 for begin, end in zip(graph_batch.ptr[:-1].tolist(), graph_batch.ptr[1:].tolist())])
        inverse = torch.empty_like(permutation)
        inverse[permutation] = torch.arange(len(permutation), device=original.device)
        permuted = graph_batch.clone()
        for name in ("x", "random_walk_pe", "spectral_vectors"):
            setattr(permuted, name, getattr(permuted, name)[permutation])
        permuted.edge_index = inverse[permuted.edge_index]
        permutation_delta = float((forward(model, permuted) - original).abs().max())
    model.train()
    del model.k1_spectral.current_batch
    model.k1_spectral.current_batch = None
    report = {"sign_delta": sign_delta, "repeated_space_rotation_delta": rotation_delta,
              "repeated_space_rotations": rotations, "node_permutation_delta": permutation_delta,
              "maximum_absolute_delta": 1e-5}
    if max(sign_delta, rotation_delta, permutation_delta) > 1e-5 or rotations == 0:
        raise RuntimeError("Spectral basis/permutation qualification failed or no repeated-space fixture")
    return report
