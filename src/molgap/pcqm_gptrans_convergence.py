"""Continue the accepted full GPTrans-T run under a validation stop rule."""
from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from .pcqm_gptrans_full_runner import (
    EMA_DECAY,
    EXPECTED_PARAMETERS,
    GPTRANS_SOURCE_SHA256,
    ExponentialMovingAverage,
    _forward,
    _gptrans_source_path,
    _make_model,
    _optimizer_step,
    _runtime_gate,
    _set_learning_rate,
)
from .pcqm_k1_full_runner import (
    BATCHES_PER_PASS,
    PHYSICAL_BATCH,
    _architecture_source_sha256,
    _load_graphs,
    _loader,
    _target_stats,
    validate_full_manifest,
)
from .pcqm_k1_gptrans_fusion import (
    EXPECTED_VALID_ROWS,
    _accepted_valid_shards,
    _validate_fusion_contract,
    mae,
)
from .training_reproducibility import (
    assert_finite_state_dict,
    atomic_json,
    atomic_torch_save,
    build_runtime_manifest,
    canonical_fingerprint,
    capture_rng_state,
    configure_fp32_determinism,
    restore_rng_state,
    sha256_file,
)


SEED = 42
BASE_CONTRACT_SHA256 = "732a9a73f341a2caad151fde20fea82cb3623e4e242337e95fb05b3fe33d2cad"
BASE_CHECKPOINT_SHA256 = "249a855908e8b27d20a67580a3772691aee446ed47b7711bc45e84f99308127b"
BASE_MODEL_BUNDLE_SHA256 = "0bc1c464f2e3cd64eb0926ae90e4f4f8c8094ce3a044405e3050f42518f1fb9e"
BASE_VALID_ACCEPTANCE_SHA256 = "09b1cf6679011e708d4b02f04b6a3202d4345f46312fc5a255301a705faee95a"
BASE_GLOBAL_STEP = 156_250
BASE_VALID_MAE_EV = 0.10901231320485097
MAX_ADDITIONAL_PASSES = 6
MAX_ADDITIONAL_STEPS = MAX_ADDITIONAL_PASSES * BATCHES_PER_PASS
START_LR = 1.0e-4
MIN_LR = 1.0e-6
PATIENCE = 3
MIN_IMPROVEMENT_EV = 1.0e-5
CHECKPOINT_EVERY_STEPS = 500
CHECKPOINT_FORMAT = "molgap-pcqm-gptrans-convergence-checkpoint-v1"
RESULT_FORMAT = "molgap-pcqm-gptrans-convergence-result-v1"


CONTRACT = {
    "format": "molgap-pcqm-gptrans-convergence-contract-v1",
    "question": "does the accepted full GPTrans-T improve under longer training",
    "source_training_contract_sha256": BASE_CONTRACT_SHA256,
    "source_checkpoint_sha256": BASE_CHECKPOINT_SHA256,
    "source_model_bundle_sha256": BASE_MODEL_BUNDLE_SHA256,
    "source_official_valid_acceptance_sha256": BASE_VALID_ACCEPTANCE_SHA256,
    "source_global_step": BASE_GLOBAL_STEP,
    "source_official_valid_mae_eV": BASE_VALID_MAE_EV,
    "architecture_source": "src/molgap/gptrans.py",
    "parameter_count": EXPECTED_PARAMETERS,
    "train_role": "official-train-full",
    "selection_role": "official-valid-consumed-for-convergence-selection",
    "test_dev_role": "sealed",
    "test_challenge_role": "sealed",
    "seed": SEED,
    "precision": "fp32",
    "tf32_enabled": False,
    "physical_batch_per_device": PHYSICAL_BATCH,
    "optimizer_state": "resume-source-adamw-moments",
    "ema_state": "resume-source-ema",
    "row_order": "continue-source-global-cursor",
    "schedule": "cosine-restart-per-continuation-step",
    "start_learning_rate": START_LR,
    "minimum_learning_rate": MIN_LR,
    "max_additional_passes": MAX_ADDITIONAL_PASSES,
    "max_additional_steps": MAX_ADDITIONAL_STEPS,
    "validation_interval_steps": BATCHES_PER_PASS,
    "patience_evaluations": PATIENCE,
    "minimum_improvement_eV": MIN_IMPROVEMENT_EV,
    "checkpoint_every_optimizer_steps": CHECKPOINT_EVERY_STEPS,
}
CONTRACT_SHA256 = canonical_fingerprint(CONTRACT)


class PackedGraphs(InMemoryDataset):
    def __init__(self, path: Path):
        super().__init__(root=None)
        self.data, self.slices = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )


def continuation_learning_rate(step: int) -> float:
    if not 1 <= step <= MAX_ADDITIONAL_STEPS:
        raise ValueError("Continuation step is outside the frozen schedule")
    progress = step / MAX_ADDITIONAL_STEPS
    return MIN_LR + 0.5 * (START_LR - MIN_LR) * (
        1.0 + math.cos(math.pi * progress)
    )


def _validate_source(source_root: Path, source_evaluation_path: Path) -> tuple[dict, dict]:
    acceptance = json.loads((source_root / "acceptance.json").read_text(encoding="utf-8"))
    checkpoint_path = source_root / "last_checkpoint.pt"
    bundle_path = source_root / "model_bundle.pt"
    if acceptance.get("accepted") is not True:
        raise RuntimeError("Accepted source GPTrans run required")
    expected = {
        "training_contract_sha256": BASE_CONTRACT_SHA256,
        "checkpoint_sha256": BASE_CHECKPOINT_SHA256,
        "model_bundle_sha256": BASE_MODEL_BUNDLE_SHA256,
        "global_step": BASE_GLOBAL_STEP,
    }
    if any(acceptance.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Source GPTrans acceptance identity changed")
    if sha256_file(checkpoint_path) != BASE_CHECKPOINT_SHA256:
        raise RuntimeError("Source GPTrans checkpoint bytes changed")
    if sha256_file(bundle_path) != BASE_MODEL_BUNDLE_SHA256:
        raise RuntimeError("Source GPTrans model bundle bytes changed")
    if sha256_file(source_evaluation_path) != BASE_VALID_ACCEPTANCE_SHA256:
        raise RuntimeError("Source official-valid acceptance bytes changed")
    evaluation = json.loads(source_evaluation_path.read_text(encoding="utf-8"))
    if (
        evaluation.get("accepted") is not True
        or evaluation.get("model_bundle_sha256") != BASE_MODEL_BUNDLE_SHA256
        or int(evaluation.get("official_valid_rows", -1)) != EXPECTED_VALID_ROWS
        or float(evaluation.get("official_valid_gap_mae_eV", float("nan")))
        != BASE_VALID_MAE_EV
    ):
        raise RuntimeError("Source official-valid result identity changed")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("training_contract_sha256") != BASE_CONTRACT_SHA256
        or int(checkpoint.get("global_step", -1)) != BASE_GLOBAL_STEP
    ):
        raise RuntimeError("Source checkpoint contract or cursor changed")
    return acceptance, checkpoint


def _valid_records(graph_root: Path) -> tuple[str, list[dict]]:
    acceptance, records = _accepted_valid_shards(graph_root)
    graph_sha = sha256_file(graph_root / "acceptance.json")
    _validate_fusion_contract(acceptance, graph_sha)
    return graph_sha, records


def _evaluate_ema(model, ema, records: list[dict], mean: float, std: float):
    evaluation_model = copy.deepcopy(model).eval()
    evaluation_model.load_state_dict(ema.state_dict(), strict=True)
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    source_indices: list[torch.Tensor] = []
    with torch.inference_mode():
        for record in records:
            dataset = PackedGraphs(Path(record["path"]))
            for batch in DataLoader(dataset, batch_size=PHYSICAL_BATCH, shuffle=False, num_workers=0):
                batch = batch.to("cuda", non_blocking=True)
                prediction = _forward(evaluation_model, batch).float() * std + mean
                predictions.append(prediction.cpu())
                targets.append(batch.y.view(-1).float().cpu())
                source_indices.append(batch.source_idx.view(-1).long().cpu())
    del evaluation_model
    torch.cuda.empty_cache()
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    source_idx = torch.cat(source_indices)
    if len(prediction) != EXPECTED_VALID_ROWS or len(torch.unique(source_idx)) != EXPECTED_VALID_ROWS:
        raise RuntimeError("Official-valid coverage or identity changed")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise RuntimeError("Official-valid prediction contains non-finite values")
    return {
        "source_idx": source_idx,
        "target_eV": target,
        "prediction_eV": prediction,
        "mae_eV": mae(prediction.numpy(), target.numpy()),
    }


def run_preflight(
    *, dataset_root: Path, manifest_path: Path, valid_graph_root: Path,
    source_root: Path, source_evaluation_path: Path, output: Path,
) -> dict:
    determinism = configure_fp32_determinism(SEED)
    accelerator = _runtime_gate()
    if _architecture_source_sha256(_gptrans_source_path()) != GPTRANS_SOURCE_SHA256:
        raise RuntimeError("Frozen GPTrans architecture source changed")
    acceptance, checkpoint = _validate_source(source_root, source_evaluation_path)
    manifest, paths = validate_full_manifest(dataset_root, manifest_path, verify_content=True)
    graphs, shards = _load_graphs(paths)
    mean_value, std_value = _target_stats(shards)
    graph_sha, valid_records = _valid_records(valid_graph_root)
    model = _make_model().to("cuda")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=START_LR, weight_decay=0.05)
    optimizer.load_state_dict(checkpoint["optimizer"])
    _set_learning_rate(optimizer, continuation_learning_rate(1))
    batch = next(iter(_loader(graphs, pass_index=checkpoint["pass_index"], start_batch=checkpoint["next_batch_in_pass"]))).to("cuda")
    _optimizer_step(
        model, optimizer, batch, torch.tensor(mean_value, device="cuda"),
        torch.tensor(std_value, device="cuda"), finite_check=True,
    )
    valid_dataset = PackedGraphs(Path(valid_records[0]["path"]))
    valid_batch = next(iter(DataLoader(valid_dataset, batch_size=PHYSICAL_BATCH, shuffle=False))).to("cuda")
    with torch.inference_mode():
        valid_prediction = _forward(model.eval(), valid_batch)
    if not torch.isfinite(valid_prediction).all():
        raise RuntimeError("Continuation preflight validation prediction is non-finite")
    runtime = build_runtime_manifest(determinism)
    result = {
        "format": "molgap-pcqm-gptrans-convergence-preflight-v1",
        "accepted": True,
        "contract_sha256": CONTRACT_SHA256,
        "source_checkpoint_sha256": acceptance["checkpoint_sha256"],
        "source_model_bundle_sha256": acceptance["model_bundle_sha256"],
        "source_official_valid_acceptance_sha256": sha256_file(source_evaluation_path),
        "train_cache_sha256": manifest["aggregates"]["topology"],
        "valid_graph_acceptance_sha256": graph_sha,
        "target_stats": {"mean_eV": mean_value, "sample_std_eV": std_value},
        "runtime_fingerprint": runtime["runtime_fingerprint"],
        "accelerator": accelerator,
        "physical_batch_per_device": int(batch.num_graphs),
        "official_validation_role_read": True,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "runtime_manifest.json", runtime)
    atomic_json(output / "preflight.json", result)
    return result


def _save_checkpoint(path: Path, *, model, ema, optimizer, source_global_step: int,
                     continuation_step: int, target_stats: dict, trace: list[dict],
                     best_mae: float, best_eval_index: int, stale: int,
                     identities: dict) -> None:
    global_step = source_global_step + continuation_step
    pass_index, next_batch = divmod(global_step, BATCHES_PER_PASS)
    atomic_torch_save(path, {
        "format": CHECKPOINT_FORMAT,
        "contract_sha256": CONTRACT_SHA256,
        **identities,
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "source_global_step": source_global_step,
        "continuation_step": continuation_step,
        "global_step": global_step,
        "pass_index": pass_index,
        "next_batch_in_pass": next_batch,
        "target_stats": target_stats,
        "trace": trace,
        "best_mae_eV": best_mae,
        "best_eval_index": best_eval_index,
        "stale_evaluations": stale,
        "rng_state": capture_rng_state(),
        "official_validation_role_read": True,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    })


def train_convergence(
    *, dataset_root: Path, manifest_path: Path, valid_graph_root: Path,
    source_root: Path, source_evaluation_path: Path, output: Path,
    preflight_path: Path, resume: bool,
    max_wall_seconds: int | None = None,
) -> dict:
    determinism = configure_fp32_determinism(SEED)
    accelerator = _runtime_gate()
    runtime = build_runtime_manifest(determinism)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("accepted") is not True
        or preflight.get("contract_sha256") != CONTRACT_SHA256
        or preflight.get("runtime_fingerprint") != runtime["runtime_fingerprint"]
        or preflight.get("accelerator") != accelerator
    ):
        raise RuntimeError("Matching accepted convergence preflight required")
    if _architecture_source_sha256(_gptrans_source_path()) != GPTRANS_SOURCE_SHA256:
        raise RuntimeError("Frozen GPTrans architecture source changed")
    acceptance, source = _validate_source(source_root, source_evaluation_path)
    manifest, paths = validate_full_manifest(dataset_root, manifest_path, verify_content=True)
    graphs, shards = _load_graphs(paths)
    mean_value, std_value = _target_stats(shards)
    target_stats = {"mean_eV": mean_value, "sample_std_eV": std_value}
    graph_sha, valid_records = _valid_records(valid_graph_root)
    identities = {
        "source_checkpoint_sha256": acceptance["checkpoint_sha256"],
        "source_model_bundle_sha256": acceptance["model_bundle_sha256"],
        "source_official_valid_acceptance_sha256": sha256_file(source_evaluation_path),
        "train_cache_sha256": manifest["aggregates"]["topology"],
        "valid_graph_acceptance_sha256": graph_sha,
        "runtime_fingerprint": runtime["runtime_fingerprint"],
    }
    if preflight.get("target_stats") != target_stats or any(preflight.get(k) != v for k, v in identities.items()):
        raise RuntimeError("Convergence inputs differ from preflight")

    configure_fp32_determinism(SEED)
    model = _make_model().to("cuda")
    ema = ExponentialMovingAverage(model, EMA_DECAY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=START_LR, weight_decay=0.05)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "last_checkpoint.pt"
    trace: list[dict] = []
    continuation_step = 0
    best_mae = BASE_VALID_MAE_EV
    best_eval_index = 0
    stale = 0
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("format") != CHECKPOINT_FORMAT or checkpoint.get("contract_sha256") != CONTRACT_SHA256 or any(checkpoint.get(k) != v for k, v in identities.items()):
            raise RuntimeError("Continuation checkpoint identity changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        continuation_step = int(checkpoint["continuation_step"])
        trace = list(checkpoint["trace"])
        best_mae = float(checkpoint["best_mae_eV"])
        best_eval_index = int(checkpoint["best_eval_index"])
        stale = int(checkpoint["stale_evaluations"])
        restore_rng_state(checkpoint["rng_state"])
    elif checkpoint_path.exists():
        raise FileExistsError("Existing continuation checkpoint requires --resume")
    else:
        model.load_state_dict(source["model"], strict=True)
        ema.load_state_dict(source["ema"])
        optimizer.load_state_dict(source["optimizer"])
        restore_rng_state(source["rng_state"])

    atomic_json(output / "contract.json", CONTRACT)
    atomic_json(output / "runtime_manifest.json", runtime)
    mean = torch.tensor(mean_value, device="cuda")
    std = torch.tensor(std_value, device="cuda")
    started = time.perf_counter()
    segment_started = started
    segment_loss = torch.zeros((), dtype=torch.float64, device="cuda")
    segment_rows = 0
    status = "running"
    while continuation_step < MAX_ADDITIONAL_STEPS and stale < PATIENCE:
        global_step = BASE_GLOBAL_STEP + continuation_step
        pass_index, start_batch = divmod(global_step, BATCHES_PER_PASS)
        for batch in _loader(graphs, pass_index=pass_index, start_batch=start_batch):
            batch = batch.to("cuda", non_blocking=True)
            next_step = continuation_step + 1
            _set_learning_rate(optimizer, continuation_learning_rate(next_step))
            loss = _optimizer_step(model, optimizer, batch, mean, std, finite_check=next_step % 50 == 0)
            ema.update(model)
            continuation_step = next_step
            segment_loss += loss.double() * PHYSICAL_BATCH
            segment_rows += PHYSICAL_BATCH
            evaluation_due = continuation_step % BATCHES_PER_PASS == 0
            wall_due = max_wall_seconds is not None and time.perf_counter() - started >= max_wall_seconds - 600
            complete = continuation_step == MAX_ADDITIONAL_STEPS
            checkpoint_due = continuation_step % CHECKPOINT_EVERY_STEPS == 0
            if evaluation_due:
                evaluated = _evaluate_ema(model, ema, valid_records, mean_value, std_value)
                eval_index = continuation_step // BATCHES_PER_PASS
                prediction_path = output / f"valid_predictions_pass_{eval_index:02d}.pt"
                atomic_torch_save(prediction_path, evaluated)
                improved = evaluated["mae_eV"] < best_mae - MIN_IMPROVEMENT_EV
                if improved:
                    best_mae = float(evaluated["mae_eV"])
                    best_eval_index = eval_index
                    stale = 0
                    assert_finite_state_dict(ema.state_dict(), label="GPTrans continuation EMA")
                    atomic_torch_save(output / "best_model_bundle.pt", {
                        "format": "molgap-pcqm-gptrans-convergence-best-v1",
                        "model_id": "gptrans_t_core",
                        "state_dict": ema.state_dict(),
                        "target_stats": target_stats,
                        "contract": CONTRACT,
                        "contract_sha256": CONTRACT_SHA256,
                        "source_model_bundle_sha256": BASE_MODEL_BUNDLE_SHA256,
                        "continuation_step": continuation_step,
                        "global_step": BASE_GLOBAL_STEP + continuation_step,
                        "official_valid_mae_eV": best_mae,
                    })
                else:
                    stale += 1
                trace.append({
                    "evaluation_index": eval_index,
                    "continuation_step": continuation_step,
                    "global_step": BASE_GLOBAL_STEP + continuation_step,
                    "additional_sample_presentations": continuation_step * PHYSICAL_BATCH,
                    "train_normalized_mae": float(segment_loss.cpu()) / segment_rows,
                    "official_valid_mae_eV": float(evaluated["mae_eV"]),
                    "best_official_valid_mae_eV": best_mae,
                    "stale_evaluations": stale,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "elapsed_seconds": time.perf_counter() - segment_started,
                    "prediction_sha256": sha256_file(prediction_path),
                    "improved": improved,
                })
                atomic_json(output / "trace.json", {"evaluations": trace})
                print(json.dumps(trace[-1], sort_keys=True), flush=True)
                segment_started = time.perf_counter()
                segment_loss = torch.zeros((), dtype=torch.float64, device="cuda")
                segment_rows = 0
            if checkpoint_due or evaluation_due or wall_due or complete or stale >= PATIENCE:
                _save_checkpoint(
                    checkpoint_path, model=model, ema=ema, optimizer=optimizer,
                    source_global_step=BASE_GLOBAL_STEP,
                    continuation_step=continuation_step, target_stats=target_stats,
                    trace=trace, best_mae=best_mae, best_eval_index=best_eval_index,
                    stale=stale, identities=identities,
                )
            if wall_due:
                status = "paused"
                break
            if complete or stale >= PATIENCE:
                status = "complete"
                break
        if status != "running":
            break

    if status == "running":
        status = "complete"
    result = {
        "format": RESULT_FORMAT,
        "complete": status == "complete",
        "status": status,
        "contract_sha256": CONTRACT_SHA256,
        **identities,
        "source_official_valid_mae_eV": BASE_VALID_MAE_EV,
        "best_official_valid_mae_eV": best_mae,
        "delta_vs_source_eV": best_mae - BASE_VALID_MAE_EV,
        "best_evaluation_index": best_eval_index,
        "continuation_step": continuation_step,
        "additional_sample_presentations": continuation_step * PHYSICAL_BATCH,
        "stale_evaluations": stale,
        "stop_reason": "patience" if stale >= PATIENCE else ("budget" if continuation_step >= MAX_ADDITIONAL_STEPS else "walltime"),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "best_model_bundle_sha256": sha256_file(output / "best_model_bundle.pt") if (output / "best_model_bundle.pt").is_file() else None,
        "official_validation_role_read": True,
        "test_dev_role_read": False,
        "test_challenge_role_read": False,
    }
    atomic_json(output / "run_summary.json", result)
    if result["complete"]:
        atomic_json(output / "completion_manifest.json", result)
    return result


def accept_convergence(*, output: Path) -> dict:
    manifest = json.loads((output / "completion_manifest.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(output / "last_checkpoint.pt", map_location="cpu", weights_only=False)
    if manifest.get("complete") is not True or manifest.get("contract_sha256") != CONTRACT_SHA256:
        raise RuntimeError("Completed matching convergence run required")
    if checkpoint.get("contract_sha256") != CONTRACT_SHA256:
        raise RuntimeError("Convergence checkpoint contract changed")
    if manifest.get("checkpoint_sha256") != sha256_file(output / "last_checkpoint.pt"):
        raise RuntimeError("Convergence checkpoint hash changed")
    trace = json.loads((output / "trace.json").read_text(encoding="utf-8"))["evaluations"]
    if len(trace) != manifest["continuation_step"] // BATCHES_PER_PASS:
        raise RuntimeError("Convergence evaluation count changed")
    observed_best = BASE_VALID_MAE_EV
    reference_source_idx = None
    reference_target = None
    for item in trace:
        candidate = float(item["official_valid_mae_eV"])
        if candidate < observed_best - MIN_IMPROVEMENT_EV:
            observed_best = candidate
    if abs(observed_best - float(manifest["best_official_valid_mae_eV"])) > 1e-12:
        raise RuntimeError("Convergence best metric differs from trace")
    for item in trace:
        path = output / f"valid_predictions_pass_{int(item['evaluation_index']):02d}.pt"
        if sha256_file(path) != item["prediction_sha256"]:
            raise RuntimeError("Convergence prediction hash changed")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if len(payload["prediction_eV"]) != EXPECTED_VALID_ROWS:
            raise RuntimeError("Convergence prediction coverage changed")
        source_idx = payload["source_idx"].view(-1).long()
        target = payload["target_eV"].view(-1).float()
        if len(torch.unique(source_idx)) != EXPECTED_VALID_ROWS:
            raise RuntimeError("Convergence prediction identity is not unique")
        if reference_source_idx is None:
            reference_source_idx = source_idx
            reference_target = target
        elif not torch.equal(source_idx, reference_source_idx) or not torch.equal(target, reference_target):
            raise RuntimeError("Convergence validation rows changed between evaluations")
        recomputed = mae(payload["prediction_eV"].numpy(), payload["target_eV"].numpy())
        if abs(recomputed - float(item["official_valid_mae_eV"])) > 1e-12:
            raise RuntimeError("Convergence metric differs from saved predictions")
    best_bundle = output / "best_model_bundle.pt"
    expected_bundle_sha = manifest.get("best_model_bundle_sha256")
    if manifest.get("best_evaluation_index", 0) > 0:
        if not best_bundle.is_file() or sha256_file(best_bundle) != expected_bundle_sha:
            raise RuntimeError("Convergence best model bundle hash changed")
        bundle = torch.load(best_bundle, map_location="cpu", weights_only=False)
        if (
            bundle.get("contract_sha256") != CONTRACT_SHA256
            or float(bundle.get("official_valid_mae_eV", float("nan"))) != observed_best
            or int(bundle.get("continuation_step", -1))
            != int(manifest["best_evaluation_index"]) * BATCHES_PER_PASS
        ):
            raise RuntimeError("Convergence best model bundle identity changed")
        assert_finite_state_dict(bundle["state_dict"], label="accepted GPTrans continuation")
    elif expected_bundle_sha is not None or best_bundle.exists():
        raise RuntimeError("Unexpected continuation bundle without an improvement")
    accepted = {**manifest, "accepted": True, "model_inference_executed": False}
    atomic_json(output / "acceptance.json", accepted)
    return accepted
