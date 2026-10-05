"""Matched 500K reference evidence with bounded, resumable Kaggle stages."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

from .training_reproducibility import (
    atomic_json, atomic_torch_save, build_runtime_manifest, capture_rng_state,
    configure_fp32_determinism, restore_rng_state, sha256_file,
)
from .screen_policy import canonical_fingerprint, validate_runtime_certificate
from .pcqm_k1_scale_runner import find_cache, load_roles, _targets
from .pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256
from .pcqm_gptrans_v4 import _state_sha256, _batch_sha256, _forward
from .futility_gate import MatchedPrefixGate, evaluate_matched_prefix_futility

PARAMETERS = {
    "full_gps": 4_771_073,
    "neural_atom_k1": 3_658_817,
    "gptrans": 5_246_817,
    "gptrans_pair_update_norm": 5_246_817,
    "edge_local_only": 3_433_601,
    "edge_sparse_global_369": 3_879_425,
    "gptrans_noisy_nodes": 5_277_400,
    "gptrans_noisy_pair_norm": 5_277_400,
    "k1_pretrained_consistency": 3_658_817,
    "gptrans_g1_bond_local_ema999": 5_871_201,
}
EPOCHS = 60
BS = 128
STEPS = 500000 // BS
COMPOSED_ARMS = frozenset({"k1_pretrained_consistency", "gptrans_g1_bond_local_ema999"})


def _composed_module(arm):
    if arm not in COMPOSED_ARMS:
        return None
    from . import pcqm_composed_500k
    return pcqm_composed_500k


def _evaluate_ema(model, ema, graphs, mean, std):
    live = {name: value.detach().clone() for name, value in model.state_dict().items()}
    try:
        model.load_state_dict(ema.state_dict(), strict=True)
        return evaluate(model, graphs, mean, std)
    finally:
        model.load_state_dict(live, strict=True)

# This screen's prefix gates were frozen against the accepted GPTrans trace.
GPTRANS_REFERENCE_TRACE_SHA256 = (
    "22cb2bea6ee531402b951334fb791f091ec64c68b3e6f539fe1c9851dcdce1d4"
)
PAIR_UPDATE_NORM_FUTILITY_GATES = (
    MatchedPrefixGate(30, 0.112521231174469, 0.006),
    MatchedPrefixGate(40, 0.10822822153568268, 0.003),
)


def schedule(epoch):
    return 1e-6 + (4e-4 - 1e-6) * (1 + math.cos(math.pi * epoch / (EPOCHS - 1))) / 2


def scientific_contract(arm="gptrans"):
    composed = _composed_module(arm)
    if composed is not None:
        return composed.scientific_contract(arm)
    loss_fingerprint = (
        "normalized-gap-l1-plus-noisy-nodes-ce-alpha0.1"
        if arm in {"gptrans_noisy_nodes", "gptrans_noisy_pair_norm"}
        else "normalized-gap-l1"
    )
    contract = {
        "benchmark_id": "pcqm-fixed500k-dev50k-matched60-v4",
        "data_role_fingerprint": FIXED_500K_MANIFEST_SHA256,
        "row_order_fingerprint": "global-randperm-seed42-plus-epoch-drop-last32",
        "feature_fingerprint": "ogb-node9-edge3-rwse16-no-geometry-input",
        "target_fingerprint": "pcqm4mv2-gap-eV",
        "seed": 42, "precision": "fp32",
        "optimizer_fingerprint": "adamw-unfused-foreachFalse-lr4e-4-wd1e-5-clip1",
        "schedule_fingerprint": "cosine60-epoch0-4e-4-epoch59-1e-6",
        "loss_fingerprint": loss_fingerprint,
        "target_transform_fingerprint": "all500k-train-only-mean-unbiased-std",
        "selection_fingerprint": "best-development-raw-model-60epochs",
        "role_access_fingerprint": "official-train-derived-train-and-internal-development-only",
        "sample_exposure": EPOCHS * STEPS * BS,
        "tail_batch_policy": "drop_last", "physical_batch_per_device": BS,
        "device_count": 1, "gradient_accumulation_steps": 1,
        "epochs": EPOCHS, "steps_per_epoch": STEPS,
    }
    if arm == "gptrans_pair_update_norm":
        contract.update({
            "selection_fingerprint": "best-development-raw-model-up-to60-matched-prefix-futility-v1",
            "futility_reference_trace_sha256": GPTRANS_REFERENCE_TRACE_SHA256,
            "futility_gates": [
                {
                    "completed_epochs": gate.completed_epochs,
                    "reference_best_mae_eV": gate.reference_best_mae_eV,
                    "maximum_deficit_eV": gate.maximum_deficit_eV,
                }
                for gate in PAIR_UPDATE_NORM_FUTILITY_GATES
            ],
        })
    return contract


def make_model(arm):
    if arm in COMPOSED_ARMS:
        raise ValueError("Composed models require the pinned initial-state factory")
    if arm in {"gptrans", "gptrans_pair_update_norm"}:
        from .pcqm_gptrans_v4 import _make_model
        variant = "pair_update_norm" if arm == "gptrans_pair_update_norm" else "reference"
        model = _make_model(variant=variant)
    elif arm in {"gptrans_noisy_nodes", "gptrans_noisy_pair_norm"}:
        from .noisy_nodes import GPTransNoisyNodes, make_noisy_nodes_pair_norm_model
        model = (
            make_noisy_nodes_pair_norm_model(noise_std=0.15, loss_weight=0.1)
            if arm == "gptrans_noisy_pair_norm"
            else GPTransNoisyNodes(noise_std=0.15, loss_weight=0.1)
        )
    elif arm in {"edge_local_only", "edge_sparse_global_369"}:
        from .pcqm_500k_v4_ablation import make_ablation_encoder
        model = make_ablation_encoder(arm)
    else:
        from .qm9_neural_atom import make_encoder
        model = make_encoder(arm)
    if sum(p.numel() for p in model.parameters()) != PARAMETERS[arm]:
        raise RuntimeError("Frozen architecture parameter count changed")
    return model


def optimizer_for(model):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=schedule(0), weight_decay=1e-5,
                            foreach=False, fused=False)


def loader(graphs, epoch=None):
    import torch
    from torch_geometric.loader import DataLoader
    workers = int(os.environ.get("MOLGAP_V4_LOADER_WORKERS", "2"))
    sampler = None if epoch is None else torch.randperm(
        len(graphs), generator=torch.Generator().manual_seed(42 + epoch)
    )[:STEPS * BS].tolist()
    return DataLoader(graphs, batch_size=BS, sampler=sampler, shuffle=False,
                      num_workers=workers, pin_memory=True,
                      persistent_workers=workers > 0,
                      generator=torch.Generator().manual_seed(9000 + (epoch or 0)))


def step(model, optimizer, batch, mean, std):
    import torch
    if batch.num_graphs != BS:
        raise RuntimeError("Non-128 optimizer batch")
    optimizer.zero_grad(set_to_none=True)
    if hasattr(model, "loss_weight") and model.training and model.loss_weight > 0.0:
        prediction, aux_loss = model(
            batch.x, batch.edge_index, batch.edge_attr, batch.batch, return_aux_loss=True
        )
        gap_loss = torch.nn.functional.l1_loss(prediction.view(-1), (batch.y.view(-1) - mean) / std)
        loss = gap_loss + model.loss_weight * aux_loss
    else:
        prediction = _forward(model, batch)
        loss = torch.nn.functional.l1_loss(prediction.view(-1), (batch.y.view(-1) - mean) / std)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.detach()


def evaluate(model, graphs, mean, std):
    import torch
    model.eval()
    rows = {"prediction": [], "target": [], "source_idx": []}
    with torch.no_grad():
        for batch in loader(graphs):
            batch = batch.to("cuda", non_blocking=True)
            rows["prediction"].append((_forward(model, batch) * std + mean).cpu())
            rows["target"].append(batch.y.view(-1).cpu())
            rows["source_idx"].append(batch.source_idx.view(-1).cpu().long())
    result = {key: torch.cat(values) for key, values in rows.items()}
    if not torch.equal(result["source_idx"], torch.arange(500000, 550000)):
        raise RuntimeError("Development row order changed")
    if not all(bool(torch.isfinite(value).all()) for value in result.values()):
        raise RuntimeError("Non-finite development output")
    result["mae_eV"] = float((result["prediction"] - result["target"]).abs().mean())
    return result


def run(arm, output, source_sha, stage_epochs=60, resume=None,
        resume_source_sha=None, max_stage_seconds=41_400,
        platform_id="kaggle1", preflight_only=False, *, initial_state=None,
        initial_state_sha256=None, target_transform=None, binding_identity=None):
    import shutil
    import torch
    if not isinstance(stage_epochs, int) or stage_epochs < 1:
        raise ValueError("stage_epochs must be a positive integer")
    if preflight_only and resume is not None and Path(resume).resolve() == output.resolve():
        raise ValueError("Resume preflight must use a separate output directory")
    composed = _composed_module(arm)
    parameters = composed.PARAMETERS[arm] if composed is not None else PARAMETERS[arm]
    if composed is not None:
        if initial_state is None or initial_state_sha256 is None:
            raise ValueError("Composed arms require pinned initial checkpoint bytes")
        initial_state = Path(initial_state)
        if sha256_file(initial_state) != initial_state_sha256:
            raise RuntimeError("Pinned initial checkpoint hash mismatch")
        if not isinstance(binding_identity, dict) or not binding_identity:
            raise ValueError("Composed arms require prospective binding identity")
    elif any(value is not None for value in (initial_state, initial_state_sha256,
                                             target_transform, binding_identity)):
        raise ValueError("Pinned composition inputs require a registered composed arm")
    make = (lambda: composed.make_model(arm, initial_state)) if composed else lambda: make_model(arm)
    optimize = (lambda model: composed.optimizer_for(arm, model)) if composed else optimizer_for
    learning_rate = (lambda epoch: composed.schedule(arm, epoch)) if composed else schedule
    train_step = (lambda model, optimizer, batch: composed.step(arm, model, optimizer, batch, mean, std)) if composed else lambda model, optimizer, batch: step(model, optimizer, batch, mean, std)
    output.mkdir(parents=True, exist_ok=True)
    if torch.cuda.device_count() != 1:
        raise RuntimeError("One visible accelerator per worker required")
    settings = configure_fp32_determinism(42)
    runtime = build_runtime_manifest(settings)
    root, manifest = find_cache(FIXED_500K_MANIFEST_SHA256)
    roles = load_roles(root, manifest)
    if composed is not None:
        composed.strip_geometry(roles)
        mean, std = composed.target_statistics(arm, roles["train"], target_transform)
    else:
        target = _targets(roles["train"]).double()
        mean, std = float(target.mean()), float(target.std(unbiased=True).clamp_min(1e-6))
    if not math.isfinite(mean) or not math.isfinite(std) or std <= 0:
        raise RuntimeError("Invalid target transformation")
    contract = scientific_contract(arm)
    contract["target_transform_fingerprint"] = canonical_fingerprint({"mean": mean, "std": std})
    if composed is not None:
        contract.update({"initial_state_sha256": initial_state_sha256,
                         "binding_identity": binding_identity,
                         "target_transform_artifact_sha256": sha256_file(Path(target_transform)) if target_transform else None})
    atomic_json(output / "runtime.json", runtime)
    atomic_json(output / "scientific_contract.json", contract)
    atomic_json(output / "data_manifest.json", manifest)
    batch = next(iter(loader(roles["train"], 0))).to("cuda")
    fixture_sha = _batch_sha256(batch)
    calibrations = []
    initial_sha = None
    torch.cuda.reset_peak_memory_stats()
    for repetition in range(2):
        configure_fp32_determinism(42)
        model = make().to("cuda").train()
        initial_sha = _state_sha256(model)
        optimizer = optimize(model)
        ema = composed.make_ema(arm, model) if composed is not None else None
        if composed is not None and repetition == 0:
            signal = composed.qualify_signal(arm, model, batch, mean, std)
            atomic_json(output / "objective_qualification.json", signal)
            # Qualification may consume dropout RNG; calibrations start identically.
            configure_fp32_determinism(42)
            del model, optimizer, ema
            model = make().to("cuda").train()
            optimizer = optimize(model)
            ema = composed.make_ema(arm, model)
        losses = []
        for _ in range(3):
            losses.append(float(train_step(model, optimizer, batch)))
            if ema is not None:
                ema.update(model)
        calibration = {"initial": initial_sha, "losses": losses, "state": _state_sha256(model)}
        if ema is not None:
            live = {name: value.detach().clone() for name, value in model.state_dict().items()}
            model.load_state_dict(ema.state_dict(), strict=True)
            calibration["ema_state"] = _state_sha256(model)
            model.load_state_dict(live, strict=True)
            del live
        if composed is not None:
            import io
            serialized = io.BytesIO()
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "rng": capture_rng_state(),
                        "ema": ema.state_dict() if ema is not None else None}, serialized)
            serialized.seek(0)
            replay_state = torch.load(serialized, map_location="cpu", weights_only=False)
            serialized.close()
            expected_loss = float(train_step(model, optimizer, batch))
            if ema is not None:
                ema.update(model)
            expected_state = _state_sha256(model)
            expected_ema = {key: value.detach().cpu().clone() for key, value in ema.state_dict().items()} if ema is not None else None
            del model, optimizer, ema
            torch.cuda.empty_cache()
            replay = make().to("cuda").train()
            replay_optimizer = optimize(replay)
            replay.load_state_dict(replay_state["model"], strict=True)
            replay_optimizer.load_state_dict(replay_state["optimizer"])
            replay_ema = composed.make_ema(arm, replay)
            if replay_ema is not None:
                replay_ema.load_state_dict(replay_state["ema"])
            restore_rng_state(replay_state["rng"])
            resumed_loss = float(train_step(replay, replay_optimizer, batch))
            if replay_ema is not None:
                replay_ema.update(replay)
            if expected_loss != resumed_loss or expected_state != _state_sha256(replay):
                raise RuntimeError("Deterministic resume optimizer-step calibration failed")
            if expected_ema is not None and any(not torch.equal(expected_ema[key], replay_ema.state_dict()[key].cpu()) for key in expected_ema):
                raise RuntimeError("Deterministic resume EMA calibration failed")
            calibration["resume_step_verified"] = True
            model, optimizer, ema = replay, replay_optimizer, replay_ema
            del replay, replay_optimizer, replay_ema, replay_state, expected_ema
        calibrations.append(calibration)
        del model, optimizer
        del ema
    atomic_json(output / "calibration.json", {"repeats": calibrations, "fixture": fixture_sha})
    if calibrations[0] != calibrations[1]:
        raise RuntimeError("Deterministic optimizer-step calibration failed")
    peak = torch.cuda.max_memory_reserved()
    if peak > 0.85 * torch.cuda.get_device_properties(0).total_memory:
        raise RuntimeError("BS128 optimizer-inclusive memory reserve below 15%")
    certificate = {
        "format": "molgap-runtime-certificate-v1", "status": "accepted",
        "platform_id": platform_id, "accelerator": torch.cuda.get_device_name(0),
        "precision": "fp32", "tf32_enabled": False, "deterministic_algorithms": True,
        "physical_batch_per_device": BS, "tail_batch_policy": "drop_last",
        "loader_workers": int(os.environ.get("MOLGAP_V4_LOADER_WORKERS", "2")),
        "software_fingerprint": runtime["installed_distributions_sha256"],
        "determinism_fingerprint": canonical_fingerprint(settings),
        "calibration_fixture_sha256": fixture_sha,
        "calibration_output_sha256": calibrations[0]["state"],
        "calibration_checks_passed": True, "runtime_fingerprint": runtime["runtime_fingerprint"],
    }
    certificate_id = canonical_fingerprint(certificate)
    validate_runtime_certificate(certificate, {"platform_id": platform_id,
        "accelerator": certificate["accelerator"], "runtime_certificate_id": certificate_id})
    atomic_json(output / "runtime_certificate.json", certificate)
    if preflight_only:
        if resume is not None:
            resume = Path(resume)
            prior_manifest = json.loads((resume / "stage_manifest.json").read_text(encoding="utf-8"))
            for name, checksum in prior_manifest["artifacts"].items():
                if sha256_file(resume / name) != checksum:
                    raise RuntimeError(f"Preflight resume artifact hash mismatch: {name}")
            prior_state = torch.load(resume / "last_checkpoint.pt", map_location="cpu", weights_only=False)
            if (prior_state["contract"] != contract or prior_state["arm"] != arm
                    or prior_state["source_sha256"] != (resume_source_sha or source_sha)
                    or prior_state["runtime_software"] != runtime["installed_distributions_sha256"]
                    or prior_state["accelerator"] != certificate["accelerator"]):
                raise RuntimeError("Preflight resume scientific/runtime identity mismatch")
            if composed is not None and (prior_state.get("next_batch_index") != 0
                    or prior_state.get("global_step") != prior_state["next_epoch"] * STEPS
                    or prior_state.get("next_schedule_epoch") != prior_state["next_epoch"]):
                raise RuntimeError("Preflight resume schedule/sampler cursor mismatch")
        artifacts = {
            path.name: sha256_file(path)
            for path in output.iterdir()
            if path.is_file() and path.name != "stage_manifest.json"
        }
        atomic_json(output / "stage_manifest.json", {
            "status": "PREFLIGHT_COMPLETE",
            "arm": arm,
            "next_epoch": prior_state["next_epoch"] if resume is not None else 0,
            "parameters": parameters,
            "contract": contract,
            "source_sha256": source_sha,
            "runtime_certificate_id": certificate_id,
            "artifacts": artifacts,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
            "test_challenge_role_read": False,
        })
        print(f"PREFLIGHT ONLY PASS {arm} parameters={parameters} peak={peak}", flush=True)
        return
    del batch
    torch.cuda.empty_cache()
    configure_fp32_determinism(42)
    model = make().to("cuda")
    optimizer = optimize(model)
    ema = composed.make_ema(arm, model) if composed is not None else None
    start_epoch, trace, best, best_epoch = 0, [], float("inf"), -1
    if resume is not None:
        resume = Path(resume)
        checksums = json.loads((resume / "stage_manifest.json").read_text())["artifacts"]
        for name in ("last_checkpoint.pt", "best_model.pt", "best_predictions.pt", "initial_state.pt"):
            if sha256_file(resume / name) != checksums[name]:
                raise RuntimeError(f"Resume hash mismatch: {name}")
        state = torch.load(resume / "last_checkpoint.pt", map_location="cpu", weights_only=False)
        expected_resume_source = resume_source_sha or source_sha
        if (state["contract"] != contract or state["arm"] != arm
                or state["source_sha256"] != expected_resume_source):
            raise RuntimeError("Resume scientific/source identity mismatch")
        if state["runtime_software"] != runtime["installed_distributions_sha256"] or state["accelerator"] != certificate["accelerator"]:
            raise RuntimeError("Resume runtime changed; requalification required")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        if composed is not None:
            if (state.get("next_batch_index") != 0
                    or state.get("global_step") != state["next_epoch"] * STEPS
                    or state.get("next_schedule_epoch") != state["next_epoch"]):
                raise RuntimeError("Resume schedule/sampler cursor mismatch")
            if (ema is None) != (state.get("ema") is None):
                raise RuntimeError("Resume EMA identity mismatch")
            if ema is not None:
                ema.load_state_dict(state["ema"])
        restore_rng_state(state["rng"])
        start_epoch, trace, best, best_epoch = state["next_epoch"], state["trace"], state["best"], state["best_epoch"]
        if not 0 <= start_epoch <= EPOCHS or len(trace) != start_epoch:
            raise RuntimeError("Resume epoch/trace mismatch")
        if resume.resolve() != output.resolve():
            for name in ("best_model.pt", "best_predictions.pt", "initial_state.pt"):
                shutil.copy2(resume / name, output / name)
            if ema is not None:
                if sha256_file(resume / "best_live_predictions.pt") != checksums["best_live_predictions.pt"]:
                    raise RuntimeError("Resume selected live predictions hash mismatch")
                shutil.copy2(resume / "best_live_predictions.pt", output / "best_live_predictions.pt")
    else:
        atomic_torch_save(output / "initial_state.pt", {"model": model.state_dict(), "state_sha256": initial_sha})
    print(f"PREFLIGHT PASS {arm} parameters={parameters} resume_epoch={start_epoch} peak={peak}", flush=True)
    stage_start = time.monotonic()
    futility_decisions = []
    if resume is not None and (resume / "futility_decisions.json").is_file():
        prior_futility = json.loads((resume / "futility_decisions.json").read_text(encoding="utf-8"))
        if prior_futility.get("arm") != arm:
            raise RuntimeError("Resume futility record belongs to another arm")
        futility_decisions = list(prior_futility.get("decisions", []))
        if output.resolve() != resume.resolve():
            shutil.copy2(resume / "futility_decisions.json", output / "futility_decisions.json")
    futility_stopped = False
    for epoch in range(start_epoch, min(EPOCHS, start_epoch + stage_epochs)):
        started = time.monotonic()
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(epoch)
        model.train()
        loss_sum = torch.zeros((), device="cuda")
        count = 0
        component_sums = {}
        for batch_index, batch in enumerate(loader(roles["train"], epoch)):
            loss_sum += train_step(model, optimizer, batch.to("cuda", non_blocking=True))
            if composed is not None:
                components = composed.step.last_components
                for name, value in components.items():
                    component_sums[name] = component_sums.get(name, 0.0) + value
            if ema is not None:
                ema.update(model)
            count += BS
            if (batch_index + 1) % 500 == 0:
                print(f"{arm} ep={epoch} batch={batch_index+1}/{STEPS}", flush=True)
        if count != STEPS * BS:
            raise RuntimeError("Sample exposure mismatch")
        live_evaluation = evaluate(model, roles["validation"], mean, std)
        evaluation = _evaluate_ema(model, ema, roles["validation"], mean, std) if ema is not None else live_evaluation
        mae = evaluation["mae_eV"]
        if mae < best:
            best, best_epoch = mae, epoch
            atomic_torch_save(output / "best_model.pt", {"arm": arm, "model": ema.state_dict() if ema is not None else model.state_dict(),
                "mean": mean, "std": std, "contract": contract, "epoch": epoch, "source_sha256": source_sha})
            atomic_torch_save(output / "best_predictions.pt", evaluation)
            if ema is not None:
                atomic_torch_save(output / "best_live_predictions.pt", live_evaluation)
        atomic_torch_save(output / f"predictions_epoch_{epoch:02d}.pt", evaluation)
        trace.append({"epoch": epoch, "train_mae_eV": float(loss_sum) / STEPS * std,
            "development_mae_eV": mae, "lr": learning_rate(epoch), "seconds": time.monotonic()-started,
            "global_step": (epoch+1)*STEPS, "sample_presentations": (epoch+1)*STEPS*BS,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        if composed is not None:
            trace[-1].update({"live_development_mae_eV": live_evaluation["mae_eV"],
                              "ema_development_mae_eV": mae if ema is not None else None,
                        "objective_components": {name: float(value / STEPS) for name, value in component_sums.items()}})
            atomic_json(output / "objective_components.json", {"epochs": [
                {"epoch": row["epoch"], "components": row["objective_components"]} for row in trace]})
        atomic_json(output / "trace.json", {"epochs": trace})
        checkpoint = {"arm": arm, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "rng": capture_rng_state(), "next_epoch": epoch+1,
            "trace": trace, "best": best, "best_epoch": best_epoch, "contract": contract,
            "source_sha256": source_sha, "runtime_software": runtime["installed_distributions_sha256"],
            "accelerator": certificate["accelerator"]}
        if composed is not None:
            checkpoint.update({"ema": ema.state_dict() if ema is not None else None,
                               "global_step": (epoch + 1) * STEPS, "next_batch_index": 0,
                               "next_schedule_epoch": epoch + 1, "mean": mean, "std": std})
        atomic_torch_save(output / "last_checkpoint.pt", checkpoint)
        atomic_json(output / "progress.json", {"status": "RUNNING", "next_epoch": epoch+1, "best": best})
        running_artifacts = {
            path.name: sha256_file(path)
            for path in output.iterdir()
            if path.is_file() and path.name != "stage_manifest.json"
        }
        atomic_json(output / "stage_manifest.json", {"status": "RUNNING", "arm": arm,
            "next_epoch": epoch+1, "best_development_mae_eV": best, "best_epoch": best_epoch,
            "parameters": parameters, "contract": contract, "source_sha256": source_sha,
            "resume_source_sha256": resume_source_sha,
            "runtime_certificate_id": certificate_id, "artifacts": running_artifacts,
            "official_validation_role_read": False, "test_dev_role_read": False,
            "test_challenge_role_read": False})
        print(f"{arm} ep{epoch:02d} dev={mae:.8f} best={best:.8f}@{best_epoch} {trace[-1]['seconds']:.1f}s", flush=True)
        if arm == "gptrans_pair_update_norm":
            decision = evaluate_matched_prefix_futility(
                completed_epochs=epoch + 1,
                candidate_best_mae_eV=best,
                gates=PAIR_UPDATE_NORM_FUTILITY_GATES,
            )
            if decision is not None:
                decision["reference_trace_sha256"] = GPTRANS_REFERENCE_TRACE_SHA256
                futility_decisions.append(decision)
                atomic_json(output / "futility_decisions.json", {
                    "format": "molgap-matched-prefix-futility-v1",
                    "arm": arm,
                    "decisions": futility_decisions,
                })
                if decision["futility_stopped"]:
                    futility_stopped = True
                    print(
                        "FUTILITY STOP "
                        f"epoch={epoch + 1} deficit={decision['candidate_minus_reference_eV']:.8f} "
                        f"threshold={decision['maximum_deficit_eV']:.8f}",
                        flush=True,
                    )
                    break
        # End at an epoch boundary well before Kaggle's session limit.
        if (max_stage_seconds is not None and
                time.monotonic() - stage_start + 1.5 * trace[-1]["seconds"] > max_stage_seconds):
            break
    if futility_stopped:
        status = "FUTILITY_STOPPED"
    else:
        status = "COMPLETE" if trace[-1]["epoch"] + 1 == EPOCHS else "STAGE_COMPLETE"
    artifacts = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != "stage_manifest.json"}
    atomic_json(output / "stage_manifest.json", {"status": status, "arm": arm,
        "next_epoch": trace[-1]["epoch"]+1, "best_development_mae_eV": best, "best_epoch": best_epoch,
        "parameters": parameters, "contract": contract, "source_sha256": source_sha,
        "resume_source_sha256": resume_source_sha,
        "runtime_certificate_id": certificate_id, "artifacts": artifacts,
        "official_validation_role_read": False, "test_dev_role_read": False, "test_challenge_role_read": False})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(set(PARAMETERS) | COMPOSED_ARMS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stage-epochs", type=int, default=4)
    parser.add_argument("--resume-source-sha")
    parser.add_argument("--max-stage-seconds", type=int, default=41_400)
    parser.add_argument("--platform-id", default="kaggle1")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--initial-state", type=Path)
    parser.add_argument("--initial-state-sha256")
    parser.add_argument("--target-transform", type=Path)
    parser.add_argument("--binding-identity", type=Path)
    args = parser.parse_args()
    try:
        run(args.arm, args.output, args.source_sha, args.stage_epochs, args.resume,
            args.resume_source_sha, args.max_stage_seconds, args.platform_id,
            args.preflight_only, initial_state=args.initial_state,
            initial_state_sha256=args.initial_state_sha256,
            target_transform=args.target_transform,
            binding_identity=json.loads(args.binding_identity.read_text(encoding="utf-8")) if args.binding_identity else None)
    except Exception as error:
        atomic_json(args.output / "failure.json", {"type": type(error).__name__, "error": str(error)})
        raise


if __name__ == "__main__":
    main()
