"""Thin triplet loader/release/acceptance over the accepted frozen audit owner."""
from pathlib import Path
import json

from .gptrans_portability import verify_file, check_reproduction, check_rows
from .gptrans_bottleneck import portability, portability_indices
from .training_reproducibility import atomic_json, sha256_file

BASE = "experiments/pcqm_gptrans_triplet_portability"
PROFILE = "gptrans-triplet-portability-v1"
KERNEL = "kaseichou/molgap-gptrans-triplet-portability-s42"
DATASET = "kaseichou/molgap-gptrans-triplet-portability-inputs"
MODE = "degree_local_triplet_aggregate_ema999"
MODEL_SHA = "60078a750db6ad551677d60020dc856c92230b3f2a2a7684f6b2de1227a36d47"
PAYLOAD_SHA = "216c48ea6944e83dca2b8dcdfcd64cb668bf9377783ab3cad6bb81656df58c88"
ORIGINAL_GAIN = .003049777398109436


def validate_release_contract(contract, release, inventory, metadata):
    expected = json.loads((Path(__file__).resolve().parents[2] / BASE / "contract.json").read_text())
    if contract != expected:
        raise ValueError("Triplet diagnostic differs from frozen repository authority")
    if (contract["workers"] != ["portability"] or contract["optimizer_steps"] != 0
            or contract["protected_roles_read"] is not False or contract["portability_rows"] != 10000
            or contract["model_assets"]["local_best.pt"]["source_checkpoint_sha256"] != MODEL_SHA
            or contract["reference_payloads"]["local_predictions.pt"]["sha256"] != PAYLOAD_SHA):
        raise ValueError("Triplet diagnostic scope/model changed")
    for name, spec in {**contract["model_assets"], **contract["reference_payloads"]}.items():
        if release["files"].get(name) != spec["sha256"]:
            raise ValueError("Triplet input differs from retained artifact")
    for name, digest in contract["source_identities"].items():
        if inventory.get(name, {}).get("sha256") != digest:
            raise ValueError("Frozen inference model source changed")
    if metadata["id"] != KERNEL or metadata["dataset_sources"] != [DATASET,
            "kaseichou/pcqm4mv2-ogb-fixed-100k-v1", "kaseichou/pcqm4mv2-ogb-fixed-500k-scnet-v1"]:
        raise ValueError("Frozen diagnostic owner/mounts changed")


def load_model(path, spec, transform):
    import torch
    from .gptrans import OGBGPTransTiny
    from .gptrans_triplet_communication import construct, PARAMETERS
    from .v4_runtime import state_dict_sha256
    verify_file(path, spec["sha256"])
    retained = torch.load(path, map_location="cpu", weights_only=False)
    if (retained["variant"] != MODE or retained["epoch"] != 38
            or retained["source_checkpoint_sha256"] != MODEL_SHA
            or retained["target_stats"] != {"mean_eV": transform["mean"], "sample_std_eV": transform["std"]}):
        raise ValueError("Frozen triplet checkpoint identity differs")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = construct(MODE, OGBGPTransTiny().state_dict())
    model.load_state_dict(retained["model"], strict=True)
    if (state_dict_sha256(model.state_dict()) != spec["state_sha256"]
            or sum(p.numel() for p in model.parameters()) != PARAMETERS[MODE]):
        raise ValueError("Frozen triplet state or parameter count differs")
    return model.to("cuda").eval()


def run_worker(inputs, cache100k, cache500k, output, deadline):
    import torch
    from .training_reproducibility import configure_fp32_determinism, build_runtime_manifest
    output.mkdir(parents=True, exist_ok=True)
    settings = configure_fp32_determinism(42)
    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise ValueError("One isolated T4 required")
    atomic_json(output / "runtime.json", build_runtime_manifest(settings))
    contract = json.loads((inputs / "contract.json").read_text())
    portability(inputs, cache100k, cache500k, output, contract, deadline, model_loader=load_model)


def accept(output, inputs):
    """Independent retained-tensor verification; never loads or executes models."""
    import torch
    from .k1_terminal_analysis import paired_saved_errors
    manifest = json.loads((output / "output_manifest.json").read_text())
    release = json.loads((inputs / "audit_release.json").read_text())
    if (manifest["release"] != release or manifest["status"] != "complete"
            or manifest["training_executed"] is not False or manifest["optimizer_steps"] != 0
            or any(manifest[k] is not False for k in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"))):
        raise ValueError("Diagnostic is not a complete NO_TRAIN run")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file() and p.name != "output_manifest.json"}
    if actual != set(manifest["files"]):
        raise ValueError("Output inventory differs")
    for name, digest in manifest["files"].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output.resolve()):
            raise ValueError("Unsafe output pointer")
        verify_file(path, digest)
    for name, digest in release["files"].items():
        verify_file(inputs / name, digest)
    ledger = json.loads((output / "allocation_ledger.json").read_text())
    if (ledger["status"] != "complete" or ledger["spec_identity"] != release["identity"]
            or ledger["allocation_device_count"] not in (1, 2) or not 0 < ledger["wall_seconds"] <= 1800
            or ledger["allocated_device_seconds"] != ledger["allocation_device_count"] * ledger["wall_seconds"]):
        raise ValueError("Allocation ledger differs from budget")
    worker = output / "portability"
    terminal = json.loads((worker / "terminal.json").read_text())
    runtime = json.loads((worker / "runtime.json").read_text())
    if (terminal["complete"] is not True or terminal["training_executed"] is not False
            or terminal["unchanged_state"] is not True or terminal["strict_ready"] is not False
            or runtime["accelerator"]["device_count_visible"] != 1
            or "T4" not in runtime["accelerator"]["name"]
            or runtime["determinism"]["precision"] != "fp32"
            or runtime["determinism"]["tf32_enabled"] is not False
            or runtime["determinism"]["deterministic_algorithms"] is not True):
        raise ValueError("Frozen worker identity/precision differs")
    events = json.loads((worker / "role_events.json").read_text())
    if [(r["role"], r["event"]) for r in events] != [(role, event)
            for role in ("original_100k", "unseen_500k") for event in ("prediction_input_and_labels_read", "metric_computed")]:
        raise ValueError("Role events differ")
    results = {}
    for role, count in (("original_100k", 10), ("unseen_500k", 2)):
        directory = worker / role
        progress = json.loads((directory / "progress.json").read_text())
        contract = json.loads((inputs / "contract.json").read_text())
        if progress["model_sha256"] != contract["model_assets"]["local_best.pt"]["sha256"] or len(progress["chunks"]) != count:
            raise ValueError("Incomplete frozen prediction chunks")
        parts = []
        for i, row in enumerate(progress["chunks"]):
            if row["file"] != f"chunk_{i:02d}.pt" or row["rows"] != 5000:
                raise ValueError("Prediction chunk scope changed")
            verify_file(directory / row["file"], row["sha256"])
            parts.append(torch.load(directory / row["file"], map_location="cpu", weights_only=False))
        values = {k: torch.cat([p[k] for p in parts]) for k in ("source_idx", "target_eV", "prediction_eV")}
        if role == "original_100k":
            check_rows(values, 100000, 50000)
            check_reproduction(values, torch.load(inputs / "local_predictions.pt", map_location="cpu", weights_only=False))
        elif not torch.equal(values["source_idx"], torch.tensor(portability_indices() + 500000)):
            raise ValueError("Frozen later cohort changed")
        reference = torch.load(inputs / ("reference_predictions.pt" if role == "original_100k" else "reference_later.pt"), map_location="cpu", weights_only=False)
        results[role] = paired_saved_errors(reference, values)
    gain = -results["unseen_500k"]["candidate_minus_reference_eV"]
    upper = results["unseen_500k"]["paired_row_bootstrap"]["ci95"][1]
    return dict(accepted=True, experiment_purpose="NO_TRAIN", comparison_class="PAIRED_ENDPOINT",
        strict_ready=False, training_replay_ready=False, local_model_inference_executed=False,
        comparisons=results, native_cost=ledger, retained_gain_fraction=gain / ORIGINAL_GAIN,
        consideration_gate_passed=gain >= ORIGINAL_GAIN / 2 and upper < 0,
        automatic_training_released=False)
