"""Descriptive structure attribution on retained internal-development rows only.

No inference, fit, SMILES construction, geometry construction, or training.
Cycle rank is E-N+C, never an SSSR ring count. Overlapping bins are posthoc
descriptions, not causal attribution or cohort-selected training evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np
import torch

FROZEN_SOURCE = Path("D:/w/k1-width256/platforms/_records/kaggle/staging/k1_width256_v1/cpu_source/src")
CACHE = Path("D:/文档/molgap/data/pcqm_fixed_100k_v1")
TRAINING = Path("D:/w/k1-width256/platforms/_records/kaggle/training")
BASES = {
    "reference": TRAINING / "pcqm_k1_width256_reference_kaggle3_v1/k1_pair/reference",
    "candidate": TRAINING / "pcqm_k1_node_width256_kaggle3_v1/experiment/width256",
}
PREDICTION_SHA = {
    "reference": "857fe31475d6728c3c9b0612571a0397a67db93815714c20ac107e06e44d4e82",
    "candidate": "a43b162b691ac4e69da97d70f71a97f471fde44759c6321adf60c0aa9b52581f",
}
LOADER_SHA = "6551b45cb29f8354109967012d809a96db76218464e46c5858a4eda7b222eb82"
MANIFEST_SHA = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
SHARD_SHA = "f8c0d054d4794ce9a8887e6c1799806d89533f6ed3aeb93f3e838b7319ac842a"
OGB_FEATURES = Path("D:/文档/molgap/.venv/Lib/site-packages/ogb/utils/features.py")
OGB_FEATURES_SHA = "16cdbc628fbeef698e3a89309de7c18d8618524354a368b55fd32f24e9281958"
ROWS, START, STOP = 50000, 100000, 150000
BOOTSTRAP_REPLICATES, SEED, CUTOFF_SECONDS = 1000, 42, 600


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def require_hash(path, expected):
    actual = digest(path)
    if actual != expected:
        raise RuntimeError(f"Frozen input differs: {path}")
    return {"path": str(path), "sha256": actual}


def graph_features(graph):
    x, edges, attributes = graph.x.numpy(), graph.edge_index.numpy(), graph.edge_attr.numpy()
    n = len(x)
    if n < 1 or x.shape != (n, 9) or edges.shape[0] != 2 or attributes.shape != (edges.shape[1], 3):
        raise RuntimeError("OGB graph dimensions differ")
    if not np.isin(x[:, [7, 8]], [0, 1]).all() or not np.isin(attributes[:, 2], [0, 1]).all():
        raise RuntimeError("OGB boolean category differs")
    # Verify every real undirected bond has exactly two opposite directed edges.
    directed = {}
    for j, (u, v) in enumerate(edges.T):
        u, v = int(u), int(v)
        if not (0 <= u < n and 0 <= v < n) or u == v or (u, v) in directed:
            raise RuntimeError("Invalid, self, or duplicate directed bond")
        directed[u, v] = attributes[j]
    bonds = []
    for (u, v), attr in directed.items():
        if (v, u) not in directed or not np.array_equal(attr, directed[v, u]):
            raise RuntimeError("Missing or inconsistent reverse bond")
        if u < v:
            bonds.append((u, v, int(attr[2])))
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for u, v, _ in bonds:
        parent[find(u)] = find(v)
    components = len({find(i) for i in range(n)})
    e = len(bonds)
    return (n, e, components, e - n + components, float(x[:, 7].mean()),
            float(sum(b[2] for b in bonds) / e) if e else 0.0, float(x[:, 8].mean()))


def bootstrap_summary(mask, ref_ae, candidate_ae, rng, deadline):
    reference, candidate = ref_ae[mask], candidate_ae[mask]
    gain = reference - candidate
    if not len(gain):
        return {"n": 0, "reference_mean_AE_eV": None, "candidate_mean_AE_eV": None,
                "reference_minus_candidate_mean_AE_eV": None, "paired_row_bootstrap_95pct_eV": None}
    samples = np.empty(BOOTSTRAP_REPLICATES)
    # One replicate at a time keeps allocations bounded to a single cohort.
    for i in range(BOOTSTRAP_REPLICATES):
        deadline()
        samples[i] = gain[rng.integers(0, len(gain), len(gain))].mean()
    return {"n": len(gain), "reference_mean_AE_eV": float(reference.mean()),
            "candidate_mean_AE_eV": float(candidate.mean()),
            "reference_minus_candidate_mean_AE_eV": float(gain.mean()),
            "paired_row_bootstrap_95pct_eV": np.quantile(samples, [0.025, 0.975]).tolist()}


def run(trajectory, trajectory_sha, output):
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    output.mkdir(parents=True, exist_ok=False)
    # Import only the frozen package before importing its owning loader.
    if "molgap" in sys.modules:
        raise RuntimeError("Run in a fresh process; molgap was imported before frozen source binding")
    sys.path.insert(0, str(FROZEN_SOURCE))
    from molgap.training_reproducibility import atomic_json

    def costs():
        return {"wall_seconds": time.perf_counter() - wall_start,
                "process_cpu_seconds": time.process_time() - cpu_start,
                "CPU_threads": 1, "accelerator_used": False, "cutoff_seconds": CUTOFF_SECONDS}

    def timeout():
        atomic_json(output / "structure_failure.json", {"status": "STOP_FOR_COST", "costs": costs()})
        os._exit(124)

    watchdog = threading.Timer(CUTOFF_SECONDS, timeout)
    watchdog.daemon = True
    watchdog.start()

    def deadline():
        if time.perf_counter() - wall_start >= CUTOFF_SECONDS:
            raise TimeoutError("Structure diagnostic exceeded cutoff")

    try:
        trajectory_identity = require_hash(trajectory, trajectory_sha)
        if json.loads(trajectory.read_text(encoding="utf-8"))["record_mode"] != "prospective":
            raise RuntimeError("Prospective trajectory must be frozen before diagnostic")
        identities = {"trajectory": trajectory_identity,
                      "loader": require_hash(FROZEN_SOURCE / "molgap/k1_screen_training.py", LOADER_SHA),
                      "manifest": require_hash(CACHE / "manifest.json", MANIFEST_SHA),
                      "official_OGB_feature_encoder": require_hash(OGB_FEATURES, OGB_FEATURES_SHA)}
        from molgap.k1_screen_training import _PackedGraphDatasetFactory, FORBIDDEN_MODEL_FIELDS
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        records = {}
        for name, base in BASES.items():
            manifest_path = base / "output_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            declared = manifest["artifacts"]["predictions"]
            if declared != {"path": "development_predictions.pt", "sha256": PREDICTION_SHA[name]}:
                raise RuntimeError(f"{name} prediction declaration differs")
            identities[name] = {"prediction": require_hash(base / declared["path"], declared["sha256"]),
                                "output_manifest": {"path": str(manifest_path), "sha256": digest(manifest_path)}}
            record = torch.load(base / declared["path"], map_location="cpu", weights_only=False)
            for key in ("prediction_eV", "target_eV", "source_idx"):
                if record[key].shape != (ROWS,):
                    raise RuntimeError(f"{name} {key} shape differs")
                if not torch.isfinite(record[key]).all():
                    raise RuntimeError(f"{name} {key} is non-finite")
            if not torch.equal(record["source_idx"], torch.arange(START, STOP, dtype=torch.long)):
                raise RuntimeError(f"{name} source-index order differs")
            records[name] = record
        if not torch.equal(records["reference"]["target_eV"], records["candidate"]["target_eV"]):
            raise RuntimeError("Paired prediction targets differ")
        manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
        shards = [s for s in manifest["geometry_shards"] if s["role"] == "development"]
        if len(shards) != 1 or (shards[0]["rows"], shards[0]["source_idx_min"], shards[0]["source_idx_max"]) != (ROWS, START, STOP - 1):
            raise RuntimeError("Development shard identity differs")
        shard = shards[0]
        if shard["sha256"] != SHARD_SHA:
            raise RuntimeError("Development shard declaration differs")
        identities["development_shard"] = require_hash(CACHE / shard["file"], SHARD_SHA)
        graphs = _PackedGraphDatasetFactory.load(CACHE / shard["file"])
        if len(graphs) != ROWS or any(field in graphs._data for field in FORBIDDEN_MODEL_FIELDS):
            raise RuntimeError("Pure 2D loader contract differs")
        columns = np.empty((ROWS, 7), dtype=np.float64)
        target = records["reference"]["target_eV"].double().numpy()
        for i in range(ROWS):
            deadline()
            graph = graphs[i]
            if graph.source_idx.numel() != 1 or int(graph.source_idx.item()) != START + i:
                raise RuntimeError("Graph source-index order differs")
            if graph.y.numel() != 1 or float(graph.y.item()) != float(target[i]):
                raise RuntimeError("Graph target differs from paired prediction target")
            columns[i] = graph_features(graph)
        names = ("n_atoms", "n_undirected_bonds", "connected_components", "cycle_rank",
                 "aromatic_atom_fraction", "conjugated_bond_fraction", "cyclic_atom_fraction")
        arrays = {name: columns[:, i].astype(np.int64) if i < 4 else columns[:, i] for i, name in enumerate(names)}
        ref_ae = np.abs(records["reference"]["prediction_eV"].double().numpy() - target)
        candidate_ae = np.abs(records["candidate"]["prediction_eV"].double().numpy() - target)
        rng = np.random.default_rng(SEED)
        summarize = lambda mask: bootstrap_summary(mask, ref_ae, candidate_ae, rng, deadline)
        cohorts = {"n_atoms": [], "cycle_rank": []}
        size = arrays["n_atoms"]
        for label, mask in (("<=15", size <= 15), ("16-25", (size >= 16) & (size <= 25)),
                            ("26-35", (size >= 26) & (size <= 35)), (">=36", size >= 36)):
            cohorts["n_atoms"].append({"bin": label, **summarize(mask)})
        rank = arrays["cycle_rank"]
        for value in range(4):
            cohorts["cycle_rank"].append({"bin": str(value) if value < 3 else ">=3",
                                           **summarize(rank == value if value < 3 else rank >= 3)})
        quartiles = {}
        for name in ("aromatic_atom_fraction", "conjugated_bond_fraction"):
            cuts = np.quantile(arrays[name], [.25, .5, .75])
            # Value bins preserve ties; duplicate cuts can produce empty bins.
            quartiles[name] = {"boundaries": cuts.tolist(), "method": "numpy_linear_full50k_posthoc",
                               "tie_rule": "searchsorted_side_left; ties stay together; empty bins retained"}
            labels = np.searchsorted(cuts, arrays[name], side="left")
            arrays[name + "_quartile_bin"] = labels
            cohorts[name] = [{"bin": f"Q{i + 1}", **summarize(labels == i)} for i in range(4)]
        arrays.update(source_idx=np.arange(START, STOP), target_eV=target,
                      reference_AE_eV=ref_ae, candidate_AE_eV=candidate_ae,
                      paired_gain_eV=ref_ae - candidate_ae)
        npz_path = output / "structure_perrow.npz"
        temporary = npz_path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, npz_path)
        result = {"format": "molgap-k1-slot-structure-diagnostic-v1", "status": "complete",
                  "scope": "retained_internal_development_50k_pure2d_only", "identities": identities,
                  "script": {"path": str(Path(__file__).resolve()), "sha256": digest(Path(__file__))},
                  "alignment": {"rows": ROWS, "source_idx_start": START, "source_idx_stop_exclusive": STOP,
                                "exact_order": True, "identical_graph_and_prediction_targets": True,
                                "target_float32_sha256": hashlib.sha256(records["reference"]["target_eV"].float().numpy().tobytes()).hexdigest()},
                  "feature_mapping": {"aromatic_atom_fraction": "OGB x[:,7]: False=0 True=1",
                                      "cyclic_atom_fraction": "OGB x[:,8]: False=0 True=1",
                                      "conjugated_bond_fraction": "OGB edge_attr[:,2]: False=0 True=1; unique real bonds",
                                      "cycle_rank": "E-N+C, independent cycle count; not SSSR ring count",
                                      "zero_bond_conjugated_fraction": 0.0},
                  "overall": summarize(np.ones(ROWS, dtype=bool)), "cohorts": cohorts,
                  "quartile_boundaries": quartiles,
                  "bootstrap": {"replicates": BOOTSTRAP_REPLICATES, "seed": SEED, "unit": "paired_row",
                                "training_stochasticity": "unknown_single_seed", "multiple_comparison_correction": False},
                  "perrow": {"path": str(npz_path), "sha256": digest(npz_path)},
                  "costs": costs(),
                  "limitations": ["Descriptive posthoc bins; overlapping features are not causal proof",
                                  "No cohort-selected training or promotion claim",
                                  "Row bootstrap excludes training variability; intervals unadjusted for multiple comparisons",
                                  "Conjugation includes aromatic bonds under retained OGB features",
                                  "Retained geometry stripped by frozen loader; no 3D constructed or consumed"]}
        atomic_json(output / "structure_summary.json", result)
        return result
    except Exception as exc:
        atomic_json(output / "structure_failure.json", {"status": "STOP_FOR_COST" if isinstance(exc, TimeoutError) else "failed",
                                                         "error": str(exc), "costs": costs()})
        raise
    finally:
        watchdog.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--trajectory", type=Path, default=Path(__file__).parent / "trajectory.json")
    parser.add_argument("--trajectory-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.trajectory, args.trajectory_sha256, args.output)


if __name__ == "__main__":
    main()
