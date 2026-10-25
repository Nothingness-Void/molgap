"""Export exact TRAIN label bytes through the original frozen CPU loader."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time
import types

ARCHIVE = "ce4326952cc6b379cb577aac9bb090adfe8e290922839e397dd92a38f643a055"
PACKAGE = "09efe4c0b754b0c7b7f2d1ea8e7a679be3b0a7b2763eb30a8afc9fff131c1e32"
SOURCE = "873c3ac4e8c04f7658d5f1711dfe4e17f7d9eb12"
SPEC = "f5404cb801a792bf57f56dfde0b496e8f21a7ad67b9f22f93a8721cc1d04a994"
DATA = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"
RECIPE = "experiments/pcqm_k1_tail_regularization_500k/generated/d33412b0d70f352901d79c7ac46d5a38ef3dd7712a0b426b65b7f526ddc4ef3a/tail_wd1e3/recipe.json"
DEPENDENCIES = ["numpy<2", "torch-geometric==2.6.1", "ogb==1.3.6"]
OBSERVATION = {"format": "molgap-target-recovery-worker-observation-v1", "status": "RUNNING",
               "train_labels_state": "not_started", "decoded_train_membership": "unknown"}
OBSERVATION_PATH = None


def observe(stage, **fields):
    OBSERVATION.update(stage=stage, updated_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
    if OBSERVATION_PATH is not None:
        temporary = OBSERVATION_PATH.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(OBSERVATION, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, OBSERVATION_PATH)
    print("TARGET_RECOVERY_STAGE", stage, OBSERVATION["train_labels_state"], flush=True)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def main():
    global OBSERVATION_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    OBSERVATION_PATH = args.output / "worker_observation.json"
    observe("bootstrap_started", development="not_read", official_validation="not_read",
            test_dev="not_read", test_challenge="not_read")
    started, cpu_started = time.perf_counter(), time.process_time()

    def remaining():
        seconds = args.deadline - time.time()
        if seconds <= 0:
            raise TimeoutError("Frozen absolute CPU cutoff reached")
        return seconds

    remaining()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    mounts = list(args.input.rglob("source_payload.bin"))
    if len(mounts) != 1 or digest(mounts[0]) != ARCHIVE:
        raise ValueError("Expected exactly the original source_payload.bin mount")
    mounted_source = mounts[0].parent
    args.work.mkdir(parents=True, exist_ok=False)
    package, source = args.work / "package", args.work / "source"
    package.mkdir()
    source.mkdir()
    shutil.copyfile(mounts[0], package / "source.tar.gz")
    sidecars = ("experiment_spec.json", "package_manifest.json", "SOURCE_COMMIT.txt",
                "SOURCE_ARCHIVE_SHA256.txt", "SOURCE_FILES.json")
    for name in sidecars:
        shutil.copyfile(mounted_source / name, package / name)
    with tarfile.open(package / "source.tar.gz", "r:gz") as archive:
        stream = archive.extractfile("src/molgap/experiment_preflight.py")
        if stream is None:
            raise ValueError("Original safe extractor is absent")
        bootstrap = types.ModuleType("_frozen_source_bootstrap")
        bootstrap.__file__ = str(mounts[0]) + ":experiment_preflight.py"
        exec(compile(stream.read(), bootstrap.__file__, "exec"), bootstrap.__dict__)
    bootstrap._unpack(package, source)
    # Match the original platform bootstrap: install before package verification
    # can import training_reproducibility and eagerly cache NumPy in this process.
    loaded_before_install = {name: name in sys.modules for name in ("numpy", "torch", "torch_geometric")}
    if any(loaded_before_install.values()):
        raise ValueError("Scientific modules loaded before dependency installation")
    observe("dependency_install_started", loaded_before_install=loaded_before_install,
            bootstrap_dependencies=DEPENDENCIES)
    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *DEPENDENCIES],
                   check=True, timeout=remaining())
    sys.path.insert(0, str(source / "src"))
    from molgap.experiment_package import verify_experiment_source_package
    verified = verify_experiment_source_package(package)
    if (verified["package_identity"], verified["source_commit"], verified["spec_identity"]) != (PACKAGE, SOURCE, SPEC):
        raise ValueError("Original source/package/Spec identity differs")
    import numpy as np
    import numpy.random
    import torch
    from torch_geometric.data import InMemoryDataset
    numpy_disk_version = importlib.metadata.version("numpy")
    if np.__version__ != numpy_disk_version:
        raise ValueError("Loaded NumPy differs from installed distribution")
    bridge = torch.tensor([1.0], dtype=torch.float32).numpy()
    if bridge.dtype != np.dtype("float32") or bridge.tolist() != [1.0]:
        raise ValueError("Torch/NumPy array bridge failed")
    numpy_consistency = {"loaded_version": np.__version__, "disk_distribution_version": numpy_disk_version,
        "module_path": np.__file__, "random_module_path": numpy.random.__file__,
        "torch_numpy_bridge": "PASS", "pyg_data_import": "PASS",
        "python": sys.version, "torch": torch.__version__}
    observe("runtime_imports_verified", numpy_consistency=numpy_consistency)
    from molgap.family_training_data import load_training_data, resolve_training_manifest
    from molgap.family_training_runner import read_recipe, _assert_target_statistics
    from molgap.research_memory.trace import atomic_write
    from molgap.training_reproducibility import atomic_json
    from molgap.pcqm_fixed_records import graph_records
    from molgap.v4_runtime import normalized_source_sha256
    torch.set_num_threads(2)
    _, recipe = read_recipe(source / RECIPE)
    if recipe.dataset.manifest_sha256 != DATA:
        raise ValueError("Original recipe dataset pin differs")
    selection = replace(recipe.dataset, development=False)
    manifest_path = resolve_training_manifest(args.input, selection)
    manifest = json.loads(manifest_path.read_bytes())
    shards = [row for row in graph_records(manifest, DATA) if row["role"] == "train"]
    if len(shards) != 10 or sum(row["rows"] for row in shards) != 500000:
        raise ValueError("Expected all ten original TRAIN shards")
    remaining()
    observe("train_label_read_started", train_labels_state="reading_unknown_subset",
            decoded_train_membership="unknown")
    data = load_training_data(args.input, selection)
    observe("train_label_read_complete", train_labels_state="all500k_decoded",
            decoded_train_membership={"source_idx_start": 0, "source_idx_stop": 500000,
                                      "rows": 500000, "verified_by": "frozen load_training_data"})
    remaining()
    _assert_target_statistics(recipe, data)
    parts = data.train.datasets
    if any(part._data.y.dtype != torch.float32 for part in parts):
        raise ValueError("Exact target proof requires original float32 labels")
    labels = torch.cat([part._data.y.reshape(-1) for part in parts]).contiguous()
    indices = torch.cat([part._data.source_idx.reshape(-1) for part in parts]).contiguous()
    if (labels.numel() != 500000 or indices.dtype != torch.int64
            or not torch.equal(indices, torch.arange(500000))):
        raise ValueError("Complete ordered TRAIN identity differs")
    raw_labels = labels.numpy().astype("<f4", copy=False).tobytes()
    raw_indices = indices.numpy().astype("<i8", copy=False).tobytes()
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_write(args.output / "labels.bin", raw_labels)
    atomic_write(args.output / "source_idx.bin", raw_indices)
    atomic_write(args.output / "data_manifest.json", manifest_path.read_bytes())
    inventory = {row["path"]: row["sha256"] for row in json.loads((package / "SOURCE_FILES.json").read_bytes())["files"]}
    imported = {}
    for name, module in tuple(sys.modules.items()):
        if not name.startswith("molgap") or not getattr(module, "__file__", None):
            continue
        path = Path(module.__file__).resolve()
        if not path.is_relative_to(source.resolve()) or normalized_source_sha256(path) != inventory.get(path.relative_to(source.resolve()).as_posix()):
            raise ValueError("Executed MolGap import escaped the original source inventory: " + name)
        imported[name] = normalized_source_sha256(path)
    observe("export_complete", status="COMPLETE")
    receipt = {"format": "molgap-training-target-recovery-v1", "status": "COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "source_commit": SOURCE,
        "source_archive_sha256": ARCHIVE, "package_identity": PACKAGE, "spec_identity": SPEC,
        "dataset_manifest_sha256": DATA, "original_recipe_sha256": digest(source / RECIPE),
        "worker_sha256": digest(__file__), "training_rows": 500000, "shards": shards,
        "source_mount": str(mounted_source.relative_to(args.input)),
        "data_manifest_mount": str(manifest_path.relative_to(args.input)),
        "source_sidecars": {name: digest(package / name) for name in sidecars},
        "imported_source_sha256": imported,
        "imported_source_digest_semantics": "v4_runtime.normalized_source_sha256 (SOURCE_FILES inventory semantics)",
        "frozen_execution_transform": {"mean": recipe.target_mean_eV, "std": recipe.target_std_eV,
                                       "ddof": 1, "variance_convention": "sample"},
        "observed_train_statistics": {"mean": data.target_mean_eV, "std": data.target_std_eV},
        "files": {"labels.bin": {"sha256": digest(args.output / "labels.bin"), "bytes": len(raw_labels),
                                  "dtype": "little-endian-float32"},
                  "source_idx.bin": {"sha256": digest(args.output / "source_idx.bin"), "bytes": len(raw_indices),
                                      "dtype": "little-endian-int64"},
                  "data_manifest.json": {"sha256": digest(args.output / "data_manifest.json"),
                                         "bytes": (args.output / "data_manifest.json").stat().st_size},
                  "worker_observation.json": {"sha256": digest(OBSERVATION_PATH), "bytes": OBSERVATION_PATH.stat().st_size}},
        "roles": {"train": {"labels_read": True, "training_membership_verified": True,
                             "rows": 500000, "ordered_source_idx_sha256": hashlib.sha256(raw_indices).hexdigest()},
                  "development": "not_read", "official_validation": "not_read", "test_dev": "not_read", "test_challenge": "not_read"},
        "runtime": {"python": sys.version, "torch": torch.__version__,
                    "torch_geometric": importlib.metadata.version("torch-geometric"),
                    "numpy": importlib.metadata.version("numpy"), "bootstrap_dependencies": DEPENDENCIES, "numpy_consistency": numpy_consistency,
                    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "torch_threads": torch.get_num_threads()},
        "cost": {"worker_wall_seconds": time.perf_counter() - started,
                 "worker_process_cpu_seconds": time.process_time() - cpu_started,
                 "bootstrap_child_cpu": "entry_receipt includes reaped pip child usage",
                 "gpu": "not_applicable", "queue": "measurement_missing"},
        "scope": "Exact original TRAIN label recovery; no model, inference, optimizer or graph construction"}
    remaining()
    atomic_json(args.output / "manifest.json", receipt)
    print("TARGET_RECOVERY_COMPLETE", receipt["files"]["labels.bin"]["sha256"], flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        observe(OBSERVATION.get("stage", "startup"), status="FAILED",
                error={"type": type(exc).__name__, "message": str(exc)})
        raise
