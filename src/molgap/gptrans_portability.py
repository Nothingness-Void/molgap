"""Role-explicit frozen GPTrans inference; never a trainer or a model selector."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .training_reproducibility import atomic_json, atomic_torch_save, sha256_file

FORMAT = "molgap-gptrans-ema-portability-v1"
ARMS = {
    "ema9999": {
        "model_sha256": "5fceb2bc34d8d269dcc90cbe1b2697fe2ad3b6e5136e0262e59cf6165ea1152c",
        "payload_sha256": "67518aa94b471acdfa8294681cef8f5b45e6d5db2a701eb022fd6f5b7f412c60",
        "variant": "degree_scale",
    },
    "ema999": {
        "model_sha256": "e8d0b955f39526e8b517d44767749366bc641851310c0d2df40e443d1003e07b",
        "payload_sha256": "39a094e2ce9cc1d5aa01e637c37f5b90a5af728dc2136e8751797ae30fc38108",
        "variant": "degree_scale_ema999",
    },
}
CONFIG = dict(node_channels=256, pair_channels=32, num_layers=12, num_heads=8,
              shortest_path_cap=20, dropout=0.1, drop_path=0.1, layer_scale=1.0,
              n_targets=1)
ARCHITECTURE = "f156359acf2bcd121c04234c22195a12d4e605c17b1129c91c8a17a91c555896"
MODEL_SOURCE = "04edcb6f928617d1142c624d071b7d7accb15c92d2f66e3473ed666ca7e5b2b2"
TRANSFORM_ASSET = "9462cf73ea022b030675a7f18f2e7682eda6f90b7a42b47affc2f0f4071353be"
PARAMETERS = 5_246_817
BATCH = 128
CHUNK_ROWS = 5000
ORIGINAL_GAIN = 0.0066522625


def verify_file(path: Path, digest: str):
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"Missing or changed frozen input: {path.name}")


def check_rows(payload, start: int, count: int):
    import torch

    if set(payload) != {"source_idx", "target_eV", "prediction_eV"}:
        raise ValueError("Unexpected prediction fields")
    if any(value.ndim != 1 or value.numel() != count for value in payload.values()):
        raise ValueError("Prediction row count/shape changed")
    if payload["source_idx"].dtype != torch.int64 or not torch.equal(
        payload["source_idx"], torch.arange(start, start + count)
    ):
        raise ValueError("Prediction role/order changed")
    if not all(torch.isfinite(payload[key]).all() for key in ("target_eV", "prediction_eV")):
        raise ValueError("Nonfinite predictions or targets")


def check_reproduction(actual, saved):
    import torch

    check_rows(actual, 100000, 50000)
    saved = {key: saved[key].reshape(-1).cpu() for key in actual}
    check_rows(saved, 100000, 50000)
    if not torch.equal(actual["target_eV"], saved["target_eV"]):
        raise ValueError("Reproduction targets changed")
    maximum = float((actual["prediction_eV"] - saved["prediction_eV"]).abs().max())
    mae = float((actual["prediction_eV"].double() - actual["target_eV"].double()).abs().mean())
    saved_mae = float((saved["prediction_eV"].double() - saved["target_eV"].double()).abs().mean())
    if maximum > 0.001 or abs(mae - saved_mae) > 0.0001:
        raise ValueError(f"Frozen checkpoint reproduction failed: {maximum}, {mae - saved_mae}")
    return dict(accepted=True, max_abs_eV=maximum, mae_eV=mae,
                saved_mae_eV=saved_mae, mae_difference_eV=mae - saved_mae)


def check_barrier(output: Path, identity: dict):
    for arm in ARMS:
        path = output / arm / "reproduction.json"
        if not path.is_file():
            return False
        row = json.loads(path.read_text())
        if (row.get("identity") != identity and isinstance(row.get("identity"), dict)
                and {k: v for k, v in row["identity"].items() if k != "invocation_id"}
                    == {k: v for k, v in identity.items() if k != "invocation_id"}
                and row["identity"].get("invocation_id") != identity.get("invocation_id")):
            return False
        if row.get("identity") != identity or row.get("accepted") is not True:
            raise ValueError("Invalid shared reproduction gate")
    return True


def load_progress(directory: Path, identity: dict):
    path = directory / "progress.json"
    observed = {p.name for p in directory.glob("chunk_*.pt")}
    if not path.exists():
        if observed:
            raise ValueError("Unrecorded inference chunks")
        return []
    value = json.loads(path.read_text())
    chunks = value["chunks"]
    if value["identity"] != identity or observed != {row["file"] for row in chunks}:
        raise ValueError("Inference resume identity/inventory changed")
    for number, row in enumerate(chunks):
        if row["file"] != f"chunk_{number:02d}.pt" or row["rows"] != CHUNK_ROWS:
            raise ValueError("Inference chunk sequence changed")
        verify_file(directory / row["file"], row["sha256"])
    if len(chunks) > 10:
        raise ValueError("Unexpected chunk count")
    return chunks


def joined(directory: Path, chunks: list):
    import torch

    parts = []
    for row in chunks:
        verify_file(directory / row["file"], row["sha256"])
        parts.append(torch.load(directory / row["file"], map_location="cpu", weights_only=False))
    return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}


def _deadline(deadline: float):
    if time.time() >= deadline:
        raise TimeoutError("Allocation-inclusive 90-minute audit budget exhausted")


def _infer(graphs, model, directory: Path, *, role: str, identity: dict,
           mean: float, std: float, deadline: float, first_only: bool = False):
    import torch
    from torch.utils.data import Subset
    from torch_geometric.loader import DataLoader
    from .pcqm_k1_cross_scale_diagnostic import DEVELOPMENT

    directory.mkdir(parents=True, exist_ok=True)
    chunks = load_progress(directory, identity)
    start = DEVELOPMENT[role][0]
    for offset in range(len(chunks) * CHUNK_ROWS, min(len(graphs), CHUNK_ROWS) if first_only else len(graphs), CHUNK_ROWS):
        _deadline(deadline)
        began = time.monotonic()
        count = min(CHUNK_ROWS, len(graphs) - offset)
        loader = DataLoader(Subset(graphs, range(offset, offset + count)), batch_size=BATCH,
                            shuffle=False, drop_last=False, num_workers=0, pin_memory=True)
        values, targets, indices = [], [], []
        with torch.inference_mode():
            for batch in loader:
                _deadline(deadline)
                # Geometry never enters the encoder or moves to the accelerator.
                value = model(batch.x.to("cuda"), batch.edge_index.to("cuda"),
                              batch.edge_attr.to("cuda"), batch.batch.to("cuda")).reshape(-1)
                values.append((value.float() * std + mean).cpu())
                targets.append(batch.y.reshape(-1).float().cpu())
                indices.append(batch.source_idx.reshape(-1).long().cpu())
        payload = dict(source_idx=torch.cat(indices), target_eV=torch.cat(targets),
                       prediction_eV=torch.cat(values))
        check_rows(payload, start + offset, count)
        name = f"chunk_{offset // CHUNK_ROWS:02d}.pt"
        atomic_torch_save(directory / name, payload)
        chunks.append(dict(file=name, sha256=sha256_file(directory / name), rows=count,
                           elapsed_seconds=time.monotonic() - began))
        atomic_json(directory / "progress.json", dict(identity=identity, chunks=chunks))
        print(f"{identity['arm']} {role}: {offset + count}/{len(graphs)}", flush=True)
    return chunks


def worker(*, arm: str, inputs: Path, cache_100k: Path, cache_500k: Path,
           output: Path, release: dict, deadline: float, invocation_id: str):
    import torch
    from .gptrans import OGBGPTransTiny
    from .pcqm_k1_cross_scale_diagnostic import _accepted_development
    from .training_reproducibility import configure_fp32_determinism, build_runtime_manifest
    from .v4_runtime import state_dict_sha256

    began = time.monotonic()
    identity = {key: release[key] for key in ("source_commit", "archive_sha256", "contract_sha256")}
    barrier_identity = dict(identity, invocation_id=invocation_id)
    directory = output / arm
    directory.mkdir(parents=True, exist_ok=True)
    determinism = configure_fp32_determinism(42)
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Exactly one isolated T4 required per inference worker")
    runtime = build_runtime_manifest(determinism)
    atomic_json(directory / "runtime.json", runtime)
    spec = ARMS[arm]
    model_path, payload_path = inputs / f"{arm}_model.pt", inputs / f"{arm}_predictions.pt"
    verify_file(model_path, spec["model_sha256"])
    verify_file(payload_path, spec["payload_sha256"])
    transform = json.loads((inputs / "target_transform.json").read_text())
    if transform["asset_sha256"] != TRANSFORM_ASSET or transform["ddof"] != 1:
        raise ValueError("Portable training-only target transform changed")
    retained = torch.load(model_path, map_location="cpu", weights_only=False)
    config = dict(retained["model_config"])
    if config.pop("variant") != spec["variant"] or config != CONFIG or retained["architecture_sha256"] != MODEL_SOURCE:
        raise ValueError("Frozen model/config identity changed")
    stats = retained["target_stats"]
    if float(stats["mean_eV"]) != float(transform["mean"]) or float(stats["sample_std_eV"]) != float(transform["std"]):
        raise ValueError("Target statistics differ from portable asset")
    model = OGBGPTransTiny(**config)
    model.load_state_dict(retained["model"], strict=True)
    if sum(p.numel() for p in model.parameters()) != PARAMETERS:
        raise ValueError("Frozen parameter count changed")
    frozen_state = state_dict_sha256(model.state_dict())
    model = model.to("cuda").eval()
    role_events = []

    def graphs_for(role):
        _deadline(deadline)
        role_events.append(dict(role=role, event="prediction_input_and_labels_read", timestamp=time.time()))
        atomic_json(directory / "role_events.json", role_events)
        return _accepted_development(cache_100k if role == "original_100k" else cache_500k, role)

    graphs = graphs_for("original_100k")
    chunk_identity = dict(identity, arm=arm, model_sha256=spec["model_sha256"],
                          payload_sha256=spec["payload_sha256"], role="original_100k", batch=BATCH)
    arguments = dict(identity=chunk_identity, mean=transform["mean"], std=transform["std"], deadline=deadline)
    chunks = _infer(graphs, model, directory / "original_100k", role="original_100k", first_only=True, **arguments)
    # Both roles have ten chunks. The first measured chunk is kept, not repeated.
    projected = chunks[0]["elapsed_seconds"] * 22 + 180
    feasible = math.isfinite(projected) and projected < deadline - time.time()
    atomic_json(directory / "feasibility.json", dict(accepted=feasible, remaining_seconds=deadline-time.time(),
                                                    projected_seconds=projected, identity=identity))
    if not feasible:
        raise TimeoutError("Measured inference preflight exceeds the frozen allocation budget")
    chunks = _infer(graphs, model, directory / "original_100k", role="original_100k", **arguments)
    saved = torch.load(payload_path, map_location="cpu", weights_only=False)
    reproduction = check_reproduction(joined(directory / "original_100k", chunks), saved)
    role_events.append(dict(role="original_100k", event="metric_computed", timestamp=time.time()))
    atomic_json(directory / "role_events.json", role_events)
    atomic_json(directory / "reproduction.json", dict(identity=barrier_identity, **reproduction))
    del graphs, saved
    while not check_barrier(output, barrier_identity):
        _deadline(deadline)
        if list(output.glob(f"*/failure_{invocation_id}.json")):
            raise RuntimeError("Peer inference worker failed before role release")
        time.sleep(1)
    graphs = graphs_for("unseen_500k")
    arguments["identity"] = dict(chunk_identity, role="unseen_500k")
    chunks = _infer(graphs, model, directory / "unseen_500k", role="unseen_500k", **arguments)
    payload = joined(directory / "unseen_500k", chunks)
    check_rows(payload, 500000, 50000)
    mae = float((payload["prediction_eV"].double() - payload["target_eV"].double()).abs().mean())
    role_events.append(dict(role="unseen_500k", event="metric_computed", timestamp=time.time()))
    atomic_json(directory / "role_events.json", role_events)
    if state_dict_sha256(model.state_dict()) != frozen_state:
        raise ValueError("Inference mutated the frozen state")
    verify_file(model_path, spec["model_sha256"])
    atomic_json(directory / "terminal.json", dict(format=FORMAT, complete=True, arm=arm,
        identity=identity, model_sha256=spec["model_sha256"], state_sha256=frozen_state,
        parameter_count=PARAMETERS, model_inference_executed=True, training_executed=False,
        official_validation_role_read=False, test_dev_role_read=False, test_challenge_role_read=False,
        original=reproduction, unseen_mae_eV=mae, elapsed_seconds=time.monotonic()-began,
        peak_memory_bytes=int(torch.cuda.max_memory_allocated()), runtime_sha256=sha256_file(directory/"runtime.json")))


def analyze(output: Path, inputs: Path):
    """Retained tensors only; independent acceptance precedes scientific nomination."""
    from .k1_terminal_analysis import paired_saved_errors

    manifest = json.loads((output / "output_manifest.json").read_text())
    release = json.loads((inputs / "audit_release.json").read_text())
    if manifest["identity"] != release:
        raise ValueError("Remote output differs from locally frozen release")
    if manifest["status"] != "COMPLETE" or manifest["training_executed"] is not False:
        raise ValueError("Allocation terminal is not a complete NO_TRAIN run")
    actual_files = {p.relative_to(output).as_posix() for p in output.rglob("*")
                    if p.is_file() and p.name not in {"output_manifest.json", "analysis.json"}}
    if actual_files != set(manifest["files"]):
        raise ValueError("Output inventory differs from terminal manifest")
    for name, digest in manifest["files"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError("Unsafe terminal artifact pointer")
        verify_file(path, digest)
    cost = json.loads((output / "cost.json").read_text())
    if (cost["status"] != "COMPLETE" or cost["allocated_devices"] != 2
            or not 0 < cost["allocation_wall_seconds"] <= 5400
            or abs(cost["allocated_device_hours"]-cost["allocation_wall_seconds"]*2/3600) > 1e-9):
        raise ValueError("Allocation-inclusive native cost/budget invalid")
    identities = []
    roles = {}
    for arm in ARMS:
        terminal = json.loads((output / arm / "terminal.json").read_text())
        if (terminal["format"] != FORMAT or terminal["complete"] is not True
                or terminal["training_executed"] is not False
                or terminal["model_sha256"] != ARMS[arm]["model_sha256"]
                or terminal["parameter_count"] != PARAMETERS
                or any(terminal[k] is not False for k in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"))):
            raise ValueError("Audit terminal/role/model identity invalid")
        verify_file(output / arm / "runtime.json", terminal["runtime_sha256"])
        runtime = json.loads((output / arm / "runtime.json").read_text())
        if (runtime["accelerator"]["device_count_visible"] != 1
                or "T4" not in runtime["accelerator"]["name"]
                or runtime["determinism"]["precision"] != "fp32"
                or runtime["determinism"]["tf32_enabled"] is not False
                or runtime["determinism"]["deterministic_algorithms"] is not True):
            raise ValueError("Runtime isolation/precision qualification changed")
        events = json.loads((output / arm / "role_events.json").read_text())
        expected = [(role, event) for role in ("original_100k", "unseen_500k")
                    for event in ("prediction_input_and_labels_read", "metric_computed")]
        if [(row["role"], row["event"]) for row in events] != expected:
            raise ValueError("Observed role history incomplete or unexpected")
        identities.append(terminal["identity"])
        roles[arm] = {}
        for role, start in (("original_100k", 100000), ("unseen_500k", 500000)):
            directory = output / arm / role
            progress = json.loads((directory / "progress.json").read_text())
            chunks = load_progress(directory, progress["identity"])
            expected_identity = dict(terminal["identity"], arm=arm, role=role, batch=BATCH,
                model_sha256=ARMS[arm]["model_sha256"], payload_sha256=ARMS[arm]["payload_sha256"])
            if len(chunks) != 10 or progress["identity"] != expected_identity:
                raise ValueError("Incomplete/changed role chunks")
            roles[arm][role] = joined(directory, chunks)
            check_rows(roles[arm][role], start, 50000)
    allocation = json.loads((output / "allocation.json").read_text())
    if identities[0] != identities[1] or not check_barrier(output, dict(identities[0], invocation_id=allocation["invocation_id"])):
        raise ValueError("Cross-worker release identity/reproduction mismatch")
    expected_identity = {key: release[key] for key in ("source_commit", "archive_sha256", "contract_sha256")}
    if identities[0] != expected_identity:
        raise ValueError("Worker source/contract differs from frozen release")
    import torch
    for arm in ARMS:
        path = inputs / f"{arm}_predictions.pt"
        verify_file(path, ARMS[arm]["payload_sha256"])
        saved = torch.load(path, map_location="cpu", weights_only=False)
        check_reproduction(roles[arm]["original_100k"], saved)
    comparisons = {role: paired_saved_errors(roles["ema9999"][role], roles["ema999"][role])
                   for role in ("original_100k", "unseen_500k")}
    # The adapter returns candidate-minus-reference delta; convert only for the gate.
    diagnostic = comparisons["unseen_500k"]
    delta = diagnostic["candidate_minus_reference_eV"]
    gain = -delta
    ci = diagnostic["paired_row_bootstrap"]["ci95"]
    passed = gain >= ORIGINAL_GAIN * 0.5 and ci[1] < 0
    result = dict(format=FORMAT, accepted=True, experiment_purpose="NO_TRAIN",
        comparison_class="PAIRED_ENDPOINT", strict_ready=False, model_inference_executed_in_acceptance=False,
        comparisons=comparisons, native_cost=cost, gain_retained_fraction=gain/ORIGINAL_GAIN, nomination_passed=passed,
        threshold_eV=ORIGINAL_GAIN*0.5, decision="SOL_REQUIRED", identity=identities[0])
    atomic_json(output / "analysis.json", result)
    return result
