"""One-time label-gated audit of the frozen PCQM K1 checkpoint."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path


SHADOW_ROWS = 10_000
BATCH_SIZE = 128
BOOTSTRAP_SEED = 2_026_091_142
BOOTSTRAP_REPLICATES = 20_000
MAX_INFERENCE_TIME_RATIO = 1.25
MIN_MEMORY_RESERVE = 0.15
EXPECTED_SHADOW_CACHE_SHA256 = (
    "4a9e4361247f497f5cd9911a69e69eddb1e0fecea6e33f318ada48aa594f6a44"
)
EXPECTED_SHADOW_INDEX_SHA256 = (
    "f68f0223dccb1c0f80e76035c79efe08b0959e4e14d5b1f7ebb5153d6cbc27bd"
)
EXPECTED_FULL_MODEL_SHA256 = (
    "467753c8caa26e3e8d537aa7933e693073fcc45851caf561174d604742d48e6a"
)
EXPECTED_K1_MODEL_SHA256 = (
    "9e9ac63a9887030dcfbdc795d843cc10e70f8d9dcec7421a13cff98970203784"
)
EXPECTED_PARAMETERS = {"full_gps": 4_771_073, "neural_atom_k1": 3_658_817}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, payload) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def find_manifest(format_name: str) -> tuple[Path, dict]:
    matches = []
    for path in Path("/kaggle/input").rglob("manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if manifest.get("format") == format_name:
            matches.append((path.parent, manifest))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {format_name}, found {len(matches)}")
    return matches[0]


def load_shadow_graphs() -> tuple[list, dict]:
    import torch

    root, manifest = find_manifest("molgap-pcqm-k1-shadow-cache-v1")
    checks = {
        "complete": manifest.get("complete") is True,
        "aggregate": manifest.get("aggregate_sha256") == EXPECTED_SHADOW_CACHE_SHA256,
        "index": manifest.get("effective_shadow_index_sha256")
        == EXPECTED_SHADOW_INDEX_SHA256,
        "count": manifest.get("shadow_graphs") == SHADOW_ROWS,
        "labels_sealed": manifest.get("shadow_labels_read") is False,
        "official_validation_sealed": manifest.get("official_validation_role_read")
        is False,
        "test_dev_sealed": manifest.get("test_dev_role_read") is False,
        "no_prior_inference": manifest.get("model_inference_executed") is False,
    }
    if not all(checks.values()):
        raise RuntimeError(f"Shadow cache contract failed: {checks}")
    graphs = []
    for shard in manifest["shards"]:
        path = root / shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise RuntimeError(f"Shadow shard changed: {path.name}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if len(payload) != int(shard["graph_count"]):
            raise RuntimeError(f"Shadow shard count changed: {path.name}")
        for graph in payload:
            # ``torch_geometric.data.Data.__getattr__`` returns ``None`` for a
            # missing store key, so Python's ``hasattr`` is not a valid
            # membership test here.
            if "y" in graph:
                raise RuntimeError("Shadow graph unexpectedly contains labels")
            # PyG increments attributes whose names contain ``index`` during
            # batching.  Preserve the immutable CSV identity under a neutral
            # key before a DataLoader ever sees the graph.
            graph.row_id = graph.row_index.view(-1).clone()
        graphs.extend(payload)
    if len(graphs) != SHADOW_ROWS:
        raise RuntimeError("Shadow graph count changed")
    row_values = [int(graph.row_id.item()) for graph in graphs]
    row_digest = hashlib.sha256(
        ",".join(str(row) for row in row_values).encode("ascii")
    ).hexdigest()
    if row_digest != EXPECTED_SHADOW_INDEX_SHA256:
        raise RuntimeError("Shadow graph row order changed")
    return graphs, manifest


def load_checkpoint_bundle() -> tuple[Path, dict]:
    root, manifest = find_manifest("molgap-pcqm-k1-shadow-checkpoints-v1")
    required = {
        "full_gps_best.pt": EXPECTED_FULL_MODEL_SHA256,
        "neural_atom_k1_best.pt": EXPECTED_K1_MODEL_SHA256,
    }
    if manifest.get("checkpoint_sha256") != required:
        raise RuntimeError("Frozen checkpoint identities changed")
    for name, expected in required.items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"Frozen checkpoint payload changed: {name}")
    stats_path = root / "target_stats.json"
    if sha256_file(stats_path) != manifest.get("target_stats_sha256"):
        raise RuntimeError("Frozen target normalization changed")
    return root, manifest


def _infer(model, graphs, mean: float, std: float) -> dict:
    import torch
    from torch_geometric.loader import DataLoader

    from .qm9_gape import forward_gap

    device = torch.device("cuda:0")
    model = model.to(device).eval()
    loader = DataLoader(graphs, batch_size=BATCH_SIZE, shuffle=False)
    prediction = []
    rows = []
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            row_values = batch.row_id.view(-1).clone()
            batch = batch.to(device)
            values = forward_gap(model, batch, augmented=False) * std + mean
            prediction.append(values.cpu())
            rows.append(row_values.cpu())
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    peak = int(torch.cuda.max_memory_allocated(device))
    total = int(torch.cuda.get_device_properties(device).total_memory)
    result = {
        "prediction": torch.cat(prediction),
        "row_index": torch.cat(rows).long(),
        "elapsed_s": elapsed,
        "graphs_per_s": SHADOW_ROWS / elapsed,
        "peak_memory_bytes": peak,
        "total_memory_bytes": total,
        "memory_reserve_fraction": 1.0 - peak / total,
    }
    del model
    torch.cuda.empty_cache()
    return result


def _open_csv_text(path: Path):
    if not zipfile.is_zipfile(path):
        return path.open("r", encoding="utf-8", newline="")
    archive = zipfile.ZipFile(path)
    members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
    if len(members) != 1:
        archive.close()
        raise RuntimeError(f"Expected one CSV member, found {members}")
    binary = archive.open(members[0], "r")
    text = io.TextIOWrapper(binary, encoding="utf-8", newline="")

    class ArchiveText:
        def __enter__(self):
            return text

        def __exit__(self, exc_type, exc, traceback):
            text.close()
            archive.close()

    return ArchiveText()


def read_shadow_labels_once(path: Path, row_indices) -> dict[int, float]:
    wanted = {int(value) for value in row_indices}
    if len(wanted) != SHADOW_ROWS:
        raise RuntimeError("Shadow inference row identities changed")
    labels = {}
    with _open_csv_text(path) as handle:
        reader = csv.reader(handle)
        header = next(reader)
        idx_column = header.index("idx")
        gap_column = header.index("homolumogap")
        for row in reader:
            index = int(row[idx_column])
            if index in wanted:
                labels[index] = float(row[gap_column])
                if len(labels) == SHADOW_ROWS:
                    break
    if len(labels) != SHADOW_ROWS:
        raise RuntimeError(f"Expected {SHADOW_ROWS} shadow labels, found {len(labels)}")
    return labels


def paired_bootstrap(delta, *, replicates: int = BOOTSTRAP_REPLICATES) -> tuple[float, float]:
    import numpy as np

    values = np.asarray(delta, dtype=np.float64)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    means = np.empty(replicates, dtype=np.float64)
    chunk = 200
    for start in range(0, replicates, chunk):
        stop = min(start + chunk, replicates)
        indices = generator.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def run_audit(source_csv: Path, output: Path, *, source_commit: str) -> dict:
    import numpy as np
    import torch

    from .qm9_neural_atom import make_encoder

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Frozen K1 shadow audit requires one visible GPU")
    graphs, cache_manifest = load_shadow_graphs()
    bundle_root, bundle = load_checkpoint_bundle()
    stats = json.loads((bundle_root / "target_stats.json").read_text(encoding="utf-8"))
    if stats.get("train_rows") != 100_000 or stats.get("model_inference_executed") is not False:
        raise RuntimeError("Frozen target-normalization contract changed")
    mean = float(stats["target_mean_eV"])
    std = float(stats["target_std_eV"])

    reports = {}
    for mode, filename in (
        ("full_gps", "full_gps_best.pt"),
        ("neural_atom_k1", "neural_atom_k1_best.pt"),
    ):
        model = make_encoder(mode)
        parameters = sum(parameter.numel() for parameter in model.parameters())
        if parameters != EXPECTED_PARAMETERS[mode]:
            raise RuntimeError(f"{mode} parameter count changed: {parameters}")
        state = torch.load(bundle_root / filename, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        reports[mode] = _infer(model, graphs, mean, std)
        reports[mode]["parameter_count"] = parameters

    full_rows = reports["full_gps"]["row_index"]
    k1_rows = reports["neural_atom_k1"]["row_index"]
    if not torch.equal(full_rows, k1_rows):
        raise RuntimeError("Paired shadow inference row order changed")
    unlabeled_path = output / "predictions_before_label_read.pt"
    atomic_torch_save(
        unlabeled_path,
        {
            "row_index": full_rows,
            "full_gps_prediction_eV": reports["full_gps"]["prediction"],
            "neural_atom_k1_prediction_eV": reports["neural_atom_k1"]["prediction"],
            "shadow_labels_read": False,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        },
    )

    # This is the sole target-label access in the audit, after both predictions
    # and their row identities have been durably frozen.
    labels = read_shadow_labels_once(source_csv, full_rows.tolist())
    target = torch.tensor([labels[int(row)] for row in full_rows], dtype=torch.float32)
    full_prediction = reports["full_gps"]["prediction"]
    k1_prediction = reports["neural_atom_k1"]["prediction"]
    full_error = (full_prediction - target).abs()
    k1_error = (k1_prediction - target).abs()
    delta = (k1_error - full_error).numpy().astype(np.float64)
    ci_low, ci_high = paired_bootstrap(delta)
    full_mae = float(full_error.mean())
    k1_mae = float(k1_error.mean())
    time_ratio = (
        reports["neural_atom_k1"]["elapsed_s"] / reports["full_gps"]["elapsed_s"]
    )
    passed = (
        k1_mae < full_mae
        and ci_high < 0.0
        and time_ratio <= MAX_INFERENCE_TIME_RATIO
        and reports["full_gps"]["memory_reserve_fraction"] >= MIN_MEMORY_RESERVE
        and reports["neural_atom_k1"]["memory_reserve_fraction"] >= MIN_MEMORY_RESERVE
    )
    payload_path = output / "audit_payload.pt"
    atomic_torch_save(
        payload_path,
        {
            "row_index": full_rows,
            "target_eV": target,
            "full_gps_prediction_eV": full_prediction,
            "neural_atom_k1_prediction_eV": k1_prediction,
            "shadow_labels_read": True,
            "shadow_label_values_accessed": SHADOW_ROWS,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
        },
    )
    for report in reports.values():
        report.pop("prediction")
        report.pop("row_index")
    metrics = {
        "format": "molgap-pcqm-k1-shadow-audit-v1",
        "complete": True,
        "source_commit": source_commit,
        "frozen_candidate": "neural_atom_k1",
        "cache_aggregate_sha256": cache_manifest["aggregate_sha256"],
        "checkpoint_manifest_sha256": sha256_file(bundle_root / "manifest.json"),
        "unlabeled_prediction_sha256": sha256_file(unlabeled_path),
        "audit_payload_sha256": sha256_file(payload_path),
        "target_mean_eV": mean,
        "target_std_eV": std,
        "shadow_rows": SHADOW_ROWS,
        "shadow_gap_mae_eV": {"full_gps": full_mae, "neural_atom_k1": k1_mae},
        "paired_delta_k1_minus_full_eV": k1_mae - full_mae,
        "paired_bootstrap_95_ci_eV": [ci_low, ci_high],
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "inference": reports,
        "k1_inference_time_ratio": time_ratio,
        "max_inference_time_ratio": MAX_INFERENCE_TIME_RATIO,
        "minimum_memory_reserve": MIN_MEMORY_RESERVE,
        "k1_shadow_passed": passed,
        "shadow_labels_read": True,
        "shadow_label_values_accessed": SHADOW_ROWS,
        "model_inference_executed": True,
        "training_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
    }
    atomic_json(output / "metrics.json", metrics)
    atomic_json(
        output / "completion_manifest.json",
        {
            **metrics,
            "artifact_sha256": {
                path.name: sha256_file(path)
                for path in sorted(output.iterdir())
                if path.is_file() and path.name != "completion_manifest.json"
            },
        },
    )
    return metrics
