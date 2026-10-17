"""Shard-streamed repaired-2M SchNet training for accepted ETKDG views."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from .repaired_2m_3d_colab import (
    ROUTE_B_SCHNET_CONFIG,
    atomic_json,
    atomic_torch_save,
    export_embeddings,
)
from .schnet import SchNetWrapper


@dataclass(frozen=True)
class Repaired2MSchNetConfig:
    variant: str
    epochs: int = 20
    patience: int = 6
    batch_size: int = 128
    learning_rate: float = 2.1150972021685588e-4
    weight_decay: float = 1.4656553886225336e-5
    grad_clip: float = 10.0
    seed: int = 42
    source_rows: int = 2_000_000
    num_workers: int = 0
    amp_dtype: str = "float16"
    max_nonfinite_batches: int = 0
    recovery_validation_tolerance_eV: float = 0.005
    evaluation_replay_tolerance_eV: float = 0.0001
    max_split_mae_delta_eV: float = 0.05

    def __post_init__(self) -> None:
        if self.variant not in {"primary", "augmented"}:
            raise ValueError("variant must be primary or augmented")
        if self.amp_dtype not in {"float16", "bfloat16"}:
            raise ValueError("amp_dtype must be float16 or bfloat16")
        if self.max_nonfinite_batches < 0:
            raise ValueError("max_nonfinite_batches must be non-negative")
        if self.recovery_validation_tolerance_eV <= 0:
            raise ValueError("recovery validation tolerance must be positive")
        if self.evaluation_replay_tolerance_eV <= 0:
            raise ValueError("evaluation replay tolerance must be positive")
        if self.max_split_mae_delta_eV <= 0:
            raise ValueError("split MAE tolerance must be positive")


def stable_recovery_config(variant: str) -> Repaired2MSchNetConfig:
    """Numerically stable A100 recovery settings after an FP16 failure."""
    if variant == "primary":
        return Repaired2MSchNetConfig(
            variant=variant,
            epochs=6,
            patience=4,
            learning_rate=5.0e-5,
            grad_clip=1.0,
            amp_dtype="bfloat16",
            max_nonfinite_batches=0,
        )
    if variant == "augmented":
        return Repaired2MSchNetConfig(
            variant=variant,
            epochs=10,
            patience=4,
            learning_rate=1.0e-4,
            grad_clip=1.0,
            amp_dtype="bfloat16",
            max_nonfinite_batches=0,
        )
    raise ValueError("variant must be primary or augmented")


def _amp_dtype(config: Repaired2MSchNetConfig) -> torch.dtype:
    return (
        torch.bfloat16
        if config.amp_dtype == "bfloat16"
        else torch.float16
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    """Hash tensor identity independently of torch serialization details."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_roles(source_rows: int, seed: int) -> np.ndarray:
    permutation = np.random.default_rng(seed).permutation(source_rows)
    n_train = int(0.8 * source_rows)
    n_validation = int(0.1 * source_rows)
    roles = np.full(source_rows, 2, dtype=np.int8)
    roles[permutation[:n_train]] = 0
    roles[permutation[n_train : n_train + n_validation]] = 1
    return roles


def _source_idx(graph) -> int:
    value = int(graph.source_idx.view(-1)[0])
    return value


def _role_graphs(graphs: list, roles: np.ndarray, role: int) -> list:
    selected = []
    for graph in graphs:
        index = _source_idx(graph)
        if index < 0 or index >= len(roles):
            raise ValueError(f"source_idx {index} outside split contract")
        if int(roles[index]) == role:
            selected.append(graph)
    return selected


def _graph_paths(graph_dir: Path) -> list[Path]:
    paths = sorted(graph_dir.glob("graphs_*.pt"))
    if len(paths) != 100:
        raise RuntimeError(f"expected 100 graph shards in {graph_dir}, got {len(paths)}")
    return paths


def verify_accepted_graph_cache(
    graph_dir: Path,
    acceptance_path: Path,
) -> dict:
    """Rehash every immutable shard before spending accelerator time."""
    payload = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if payload.get("accepted") is not True:
        raise RuntimeError(f"view is not accepted: {acceptance_path}")
    paths = _graph_paths(graph_dir)
    if int(payload.get("expected_shards", -1)) != len(paths):
        raise RuntimeError("acceptance shard count differs from graph cache")
    records = payload.get("shards")
    if not isinstance(records, list) or len(records) != len(paths):
        raise RuntimeError("acceptance shard ledger differs from graph cache")
    by_name = {Path(record["path"]).name: record for record in records}
    if len(by_name) != len(records):
        raise RuntimeError("acceptance shard ledger contains duplicate names")
    accepted_rows = 0
    ledger = hashlib.sha256()
    for path in paths:
        record = by_name.get(path.name)
        if record is None:
            raise RuntimeError(f"acceptance ledger misses {path.name}")
        digest = sha256_file(path)
        if digest != record.get("sha256"):
            raise RuntimeError(f"graph shard hash differs: {path}")
        rows = int(record.get("rows", -1))
        if rows <= 0:
            raise RuntimeError(f"invalid accepted row count: {path}")
        accepted_rows += rows
        ledger.update(f"{path.name}\0{rows}\0{digest}\n".encode("ascii"))
    if accepted_rows != int(payload.get("accepted_rows", -1)):
        raise RuntimeError("accepted row total differs from shard ledger")
    source_rows = int(payload.get("source_rows", -1))
    if source_rows <= 0 or accepted_rows > source_rows:
        raise RuntimeError("accepted/source row contract is invalid")
    return {
        "acceptance_path": str(acceptance_path),
        "acceptance_sha256": sha256_file(acceptance_path),
        "accepted_rows": accepted_rows,
        "source_rows": source_rows,
        "shards": len(paths),
        "graph_ledger_sha256": ledger.hexdigest(),
    }


def _accept_view(graph_dir: Path, acceptance_path: Path) -> dict:
    return verify_accepted_graph_cache(graph_dir, acceptance_path)


def validate_evaluation_consistency(
    *,
    validation_mae: np.ndarray,
    test_mae: np.ndarray,
    replay_test_mae: np.ndarray,
    config: Repaired2MSchNetConfig,
    expected_validation_average_mae_eV: float | None = None,
) -> dict:
    """Reject backend, state, or split anomalies before artifact acceptance."""
    validation_mae = np.asarray(validation_mae, dtype=np.float64)
    test_mae = np.asarray(test_mae, dtype=np.float64)
    replay_test_mae = np.asarray(replay_test_mae, dtype=np.float64)
    for name, value in (
        ("validation", validation_mae),
        ("test", test_mae),
        ("test replay", replay_test_mae),
    ):
        if value.shape != (3,) or not np.isfinite(value).all():
            raise RuntimeError(f"{name} MAE is not a finite three-target vector")
    validation_average = float(validation_mae.mean())
    test_average = float(test_mae.mean())
    replay_delta = float(np.max(np.abs(test_mae - replay_test_mae)))
    split_delta = float(np.max(np.abs(validation_mae - test_mae)))
    if (
        expected_validation_average_mae_eV is not None
        and abs(
            validation_average - expected_validation_average_mae_eV
        )
        > config.recovery_validation_tolerance_eV
    ):
        raise RuntimeError(
            "recovery validation MAE differs from its source checkpoint"
        )
    if replay_delta > config.evaluation_replay_tolerance_eV:
        raise RuntimeError("test MAE replay differs on the same frozen state")
    if split_delta > config.max_split_mae_delta_eV:
        raise RuntimeError(
            "validation/test MAE divergence exceeds the random-split contract"
        )
    return {
        "validation_average_mae_eV": validation_average,
        "test_average_mae_eV": test_average,
        "test_replay_max_delta_eV": replay_delta,
        "validation_test_max_delta_eV": split_delta,
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _forward(model: nn.Module, batch) -> torch.Tensor:
    charges = batch.charges if hasattr(batch, "charges") else None
    return model(batch.z, batch.pos, batch.batch, charges=charges)


@torch.no_grad()
def _evaluate_role(
    *,
    model: nn.Module,
    graph_paths: list[Path],
    roles: np.ndarray,
    role: int,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    amp_dtype: torch.dtype = torch.float16,
) -> tuple[np.ndarray, int]:
    model.eval()
    absolute_error = torch.zeros(3, dtype=torch.float64)
    count = 0
    for path in graph_paths:
        graphs = torch.load(path, map_location="cpu", weights_only=False)
        selected = _role_graphs(graphs, roles, role)
        del graphs
        if not selected:
            continue
        loader = DataLoader(
            selected,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype):
                prediction = _forward(model, batch)
            target = batch.y.view(-1, 3)
            if not torch.isfinite(prediction).all():
                raise RuntimeError("non-finite repaired-2M validation prediction")
            absolute_error += (
                prediction.float().sub(target).abs().sum(dim=0).double().cpu()
            )
            count += int(batch.num_graphs)
        del loader, selected
    if count == 0:
        raise RuntimeError("evaluation role is empty")
    return (absolute_error / count).numpy(), count


def preflight_repaired_2m_schnet(
    *,
    primary_graph_dir: Path,
    primary_acceptance: Path,
    secondary_graph_dir: Path | None,
    secondary_acceptance: Path | None,
    variant: str,
    output_path: Path,
    batch_size: int = 16,
    config: Repaired2MSchNetConfig | None = None,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for repaired-2M SchNet preflight")
    if config is None:
        config = Repaired2MSchNetConfig(
            variant=variant, batch_size=batch_size
        )
    elif config.variant != variant:
        raise ValueError("preflight config variant differs")
    reports = {"primary": _accept_view(primary_graph_dir, primary_acceptance)}
    if variant == "augmented":
        if secondary_graph_dir is None or secondary_acceptance is None:
            raise ValueError("augmented preflight requires the secondary view")
        reports["secondary"] = _accept_view(
            secondary_graph_dir, secondary_acceptance
        )
    _set_seed(config.seed)
    device = torch.device("cuda")
    model = SchNetWrapper(
        **ROUTE_B_SCHNET_CONFIG, use_charges=True, n_targets=3
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.amp_dtype == "float16"
    )
    losses = []
    views = [primary_graph_dir]
    if variant == "augmented":
        views.append(secondary_graph_dir)
    torch.cuda.reset_peak_memory_stats()
    for graph_dir in views:
        graph_paths = _graph_paths(graph_dir)
        for path in (graph_paths[0], graph_paths[-1]):
            graphs = torch.load(
                path, map_location="cpu", weights_only=False
            )[:batch_size]
            batch = next(
                iter(DataLoader(graphs, batch_size=batch_size))
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=_amp_dtype(config)):
                loss = nn.functional.l1_loss(
                    _forward(model, batch), batch.y.view(-1, 3)
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.grad_clip
            )
            if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
                raise RuntimeError(
                    f"non-finite repaired-2M preflight on {path.name}"
                )
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach()))
    report = {
        "status": "complete",
        "variant": variant,
        "model_config": ROUTE_B_SCHNET_CONFIG,
        "input_reports": reports,
        "losses": losses,
        "finite": bool(np.isfinite(losses).all()),
        "device": torch.cuda.get_device_name(0),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    if not report["finite"]:
        raise RuntimeError("non-finite repaired-2M SchNet preflight")
    atomic_json(report, output_path)
    return report


def train_repaired_2m_schnet(
    *,
    primary_graph_dir: Path,
    primary_acceptance: Path,
    secondary_graph_dir: Path | None,
    secondary_acceptance: Path | None,
    output_dir: Path,
    config: Repaired2MSchNetConfig,
    recovery_checkpoint: Path | None = None,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for repaired-2M SchNet training")
    input_reports = {
        "primary": _accept_view(primary_graph_dir, primary_acceptance)
    }
    secondary_paths = []
    if config.variant == "augmented":
        if secondary_graph_dir is None or secondary_acceptance is None:
            raise ValueError("augmented training requires the secondary view")
        input_reports["secondary"] = _accept_view(
            secondary_graph_dir, secondary_acceptance
        )
        secondary_paths = _graph_paths(secondary_graph_dir)
    primary_paths = _graph_paths(primary_graph_dir)
    roles = source_roles(config.source_rows, config.seed)
    _set_seed(config.seed)
    device = torch.device("cuda")
    model = SchNetWrapper(
        **ROUTE_B_SCHNET_CONFIG, use_charges=True, n_targets=3
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=config.amp_dtype == "float16"
    )
    criterion = nn.L1Loss()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "model.pt"
    last_path = output_dir / "checkpoint.pt"
    contract = {
        "config": asdict(config),
        "model_config": ROUTE_B_SCHNET_CONFIG,
        "input_reports": input_reports,
        "recovery_checkpoint": (
            {
                "path": str(recovery_checkpoint),
                "sha256": sha256_file(recovery_checkpoint),
            }
            if recovery_checkpoint is not None
            else None
        ),
    }
    start_epoch = 0
    best_epoch = -1
    best_mae = float("inf")
    best_state = None
    best_origin = "training"
    wait = 0
    log = []
    recovery_baseline = None
    if last_path.exists():
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint.get("contract") != contract:
            raise RuntimeError("repaired-2M SchNet resume contract changed")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_state = checkpoint["best_state"]
        best_epoch = int(checkpoint["best_epoch"])
        best_mae = float(checkpoint["best_validation_average_mae_eV"])
        best_origin = str(checkpoint.get("best_origin", "training"))
        wait = int(checkpoint["wait"])
        log = list(checkpoint["log"])
        recovery_baseline = checkpoint.get("recovery_baseline")
        start_epoch = int(checkpoint["epoch"]) + 1
    elif recovery_checkpoint is not None:
        recovered = torch.load(
            recovery_checkpoint, map_location=device, weights_only=False
        )
        recovered_state = recovered.get("best_state")
        if recovered_state is None:
            raise RuntimeError("recovery checkpoint has no best_state")
        if not all(
            torch.isfinite(value).all() for value in recovered_state.values()
        ):
            raise RuntimeError("recovery best_state is non-finite")
        model.load_state_dict(recovered_state, strict=True)
        best_state = copy.deepcopy(recovered_state)
        best_epoch = -1
        best_mae = float(recovered["best_validation_average_mae_eV"])
        best_origin = "recovery_source"
        recovery_validation, recovery_validation_rows = _evaluate_role(
            model=model,
            graph_paths=primary_paths,
            roles=roles,
            role=1,
            device=device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            amp_dtype=_amp_dtype(config),
        )
        recovery_test, recovery_test_rows = _evaluate_role(
            model=model,
            graph_paths=primary_paths,
            roles=roles,
            role=2,
            device=device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            amp_dtype=_amp_dtype(config),
        )
        recovery_test_replay, replay_rows = _evaluate_role(
            model=model,
            graph_paths=primary_paths,
            roles=roles,
            role=2,
            device=device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            amp_dtype=_amp_dtype(config),
        )
        if recovery_test_rows != replay_rows:
            raise RuntimeError("recovery test replay row count differs")
        recovery_consistency = validate_evaluation_consistency(
            validation_mae=recovery_validation,
            test_mae=recovery_test,
            replay_test_mae=recovery_test_replay,
            config=config,
            expected_validation_average_mae_eV=best_mae,
        )
        recovery_baseline = {
            "state_dict_sha256": state_dict_sha256(recovered_state),
            "validation_rows": recovery_validation_rows,
            "test_rows": recovery_test_rows,
            "validation_mae_eV": recovery_validation.tolist(),
            "test_mae_eV": recovery_test.tolist(),
            "consistency": recovery_consistency,
        }

    training_paths = [("primary", path) for path in primary_paths]
    training_paths += [("secondary", path) for path in secondary_paths]
    for epoch in range(start_epoch, config.epochs):
        epoch_started = time.monotonic()
        model.train()
        total_loss = 0.0
        total_rows = 0
        nonfinite_batches = 0
        nonfinite_source_idx_sample = []
        paths = list(training_paths)
        random.Random(config.seed + epoch).shuffle(paths)
        for shard_index, (_, path) in enumerate(paths):
            graphs = torch.load(path, map_location="cpu", weights_only=False)
            selected = _role_graphs(graphs, roles, 0)
            del graphs
            if not selected:
                continue
            loader = DataLoader(
                selected,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=True,
                generator=torch.Generator().manual_seed(
                    config.seed * 1_000_000 + epoch * 1_000 + shard_index
                ),
            )
            for batch in loader:
                batch = batch.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    "cuda", dtype=_amp_dtype(config)
                ):
                    loss = criterion(_forward(model, batch), batch.y.view(-1, 3))
                if not torch.isfinite(loss):
                    nonfinite_batches += 1
                    nonfinite_source_idx_sample.extend(
                        int(value)
                        for value in batch.source_idx.view(-1)[:8].cpu()
                    )
                    if nonfinite_batches > config.max_nonfinite_batches:
                        raise RuntimeError(
                            "non-finite repaired-2M training loss exceeded "
                            f"limit at epoch {epoch}"
                        )
                    continue
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.grad_clip
                )
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    nonfinite_batches += 1
                    nonfinite_source_idx_sample.extend(
                        int(value)
                        for value in batch.source_idx.view(-1)[:8].cpu()
                    )
                    if nonfinite_batches > config.max_nonfinite_batches:
                        raise RuntimeError(
                            "non-finite repaired-2M gradient exceeded "
                            f"limit at epoch {epoch}"
                        )
                    scaler.update()
                    continue
                scaler.step(optimizer)
                scaler.update()
                total_loss += float(loss.detach()) * batch.num_graphs
                total_rows += int(batch.num_graphs)
            del loader, selected
        validation_mae, validation_rows = _evaluate_role(
            model=model,
            graph_paths=primary_paths,
            roles=roles,
            role=1,
            device=device,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            amp_dtype=_amp_dtype(config),
        )
        scheduler.step()
        average_mae = float(validation_mae.mean())
        improved = average_mae < best_mae
        if improved:
            best_mae = average_mae
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            best_origin = "training"
            wait = 0
        else:
            wait += 1
        row = {
            "epoch": epoch,
            "train_mae_eV": total_loss / max(total_rows, 1),
            "train_draw_rows": total_rows,
            "validation_rows": validation_rows,
            "validation_mae_eV": {
                name: float(validation_mae[index])
                for index, name in enumerate(("HOMO", "LUMO", "Gap"))
            },
            "validation_average_mae_eV": average_mae,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.monotonic() - epoch_started,
            "selected": improved,
            "nonfinite_batches": nonfinite_batches,
            "nonfinite_source_idx_sample": nonfinite_source_idx_sample[:64],
        }
        log.append(row)
        atomic_torch_save(
            {
                "contract": contract,
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_state": best_state,
                "best_epoch": best_epoch,
                "best_origin": best_origin,
                "best_validation_average_mae_eV": best_mae,
                "recovery_baseline": recovery_baseline,
                "wait": wait,
                "log": log,
            },
            last_path,
        )
        atomic_json(
            {
                "status": "training",
                "variant": config.variant,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_average_mae_eV": best_mae,
            },
            output_dir / "progress.json",
        )
        print(
            f"{config.variant} ep{epoch:02d} "
            f"train={row['train_mae_eV']:.6f} "
            f"val={average_mae:.6f}eV {row['elapsed_s']:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if wait >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("repaired-2M SchNet produced no finite checkpoint")
    atomic_torch_save(best_state, best_path)
    model.load_state_dict(best_state, strict=True)
    final_validation_mae, final_validation_rows = _evaluate_role(
        model=model,
        graph_paths=primary_paths,
        roles=roles,
        role=1,
        device=device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        amp_dtype=_amp_dtype(config),
    )
    test_mae, test_rows = _evaluate_role(
        model=model,
        graph_paths=primary_paths,
        roles=roles,
        role=2,
        device=device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        amp_dtype=_amp_dtype(config),
    )
    replay_test_mae, replay_test_rows = _evaluate_role(
        model=model,
        graph_paths=primary_paths,
        roles=roles,
        role=2,
        device=device,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        amp_dtype=_amp_dtype(config),
    )
    if replay_test_rows != test_rows:
        raise RuntimeError("final test replay row count differs")
    consistency = validate_evaluation_consistency(
        validation_mae=final_validation_mae,
        test_mae=test_mae,
        replay_test_mae=replay_test_mae,
        config=config,
        expected_validation_average_mae_eV=best_mae,
    )
    if best_origin == "recovery_source" and recovery_baseline is not None:
        if (
            state_dict_sha256(best_state)
            != recovery_baseline["state_dict_sha256"]
        ):
            raise RuntimeError("recovery best state identity changed")
        baseline_test = np.asarray(
            recovery_baseline["test_mae_eV"], dtype=np.float64
        )
        if (
            np.max(np.abs(test_mae - baseline_test))
            > config.evaluation_replay_tolerance_eV
        ):
            raise RuntimeError(
                "unchanged recovery best state produced a different final test MAE"
            )
    result = {
        "status": "complete",
        "variant": config.variant,
        "config": asdict(config),
        "model_config": ROUTE_B_SCHNET_CONFIG,
        "input_reports": input_reports,
        "best_epoch": best_epoch,
        "best_origin": best_origin,
        "best_state_dict_sha256": state_dict_sha256(best_state),
        "best_validation_average_mae_eV": best_mae,
        "final_validation_rows": final_validation_rows,
        "final_validation_mae_eV": {
            name: float(final_validation_mae[index])
            for index, name in enumerate(("HOMO", "LUMO", "Gap"))
        },
        "test_rows": test_rows,
        "test_mae_eV": {
            name: float(test_mae[index])
            for index, name in enumerate(("HOMO", "LUMO", "Gap"))
        },
        "test_average_mae_eV": float(test_mae.mean()),
        "evaluation_consistency": consistency,
        "recovery_baseline": recovery_baseline,
        "log": log,
        "artifacts": {
            "model": str(best_path),
            "model_sha256": sha256_file(best_path),
            "checkpoint": str(last_path),
            "checkpoint_sha256": sha256_file(last_path),
        },
    }
    atomic_json(result, output_dir / "metrics.json")
    atomic_json(
        {
            "status": "complete",
            "variant": config.variant,
            "model_sha256": result["artifacts"]["model_sha256"],
            "checkpoint_sha256": result["artifacts"]["checkpoint_sha256"],
        },
        output_dir / "completion_manifest.json",
    )
    return result


def export_repaired_2m_schnet_embeddings(
    *,
    root: Path,
    variant: str,
) -> dict:
    Repaired2MSchNetConfig(variant=variant)
    return export_embeddings(
        graph_dir=root / "primary" / "graph_shards",
        best_checkpoint=root / "outputs" / variant / "model.pt",
        output_dir=root / "embeddings" / variant,
        batch_size=128,
        num_workers=0,
        model_config=ROUTE_B_SCHNET_CONFIG,
    )


def accept_repaired_2m_schnet_embeddings(
    *,
    root: Path,
    output_path: Path,
    source_rows: int = 2_000_000,
) -> dict:
    variants = ("primary", "augmented")
    completions = {}
    for variant in variants:
        path = root / "embeddings" / variant / "completion.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete" or int(payload.get("parts", -1)) != 100:
            raise RuntimeError(f"incomplete embedding export: {variant}")
        completions[variant] = payload

    seen = np.zeros(source_rows, dtype=np.bool_)
    parts = []
    total_rows = 0
    for part_index in range(100):
        payloads = {}
        reports = {}
        for variant in variants:
            directory = root / "embeddings" / variant
            paths = sorted(directory.glob("embeddings_*.pt"))
            if len(paths) != 100:
                raise RuntimeError(f"{variant} embedding part count differs")
            path = paths[part_index]
            report_path = path.with_suffix(".json")
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("sha256") != sha256_file(path):
                raise RuntimeError(f"embedding hash differs: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            embeddings = payload["embeddings"]
            predictions = payload["predictions"]
            source_idx = payload["source_idx"].long()
            targets = payload["targets"].float()
            rows = len(source_idx)
            if (
                embeddings.shape != (rows, 176)
                or predictions.shape != (rows, 3)
                or targets.shape != (rows, 3)
            ):
                raise RuntimeError(f"embedding tensor shapes differ: {path}")
            if not all(
                torch.isfinite(tensor).all()
                for tensor in (embeddings, predictions, targets)
            ):
                raise RuntimeError(f"non-finite embedding payload: {path}")
            payloads[variant] = payload
            reports[variant] = {
                "path": path.name,
                "rows": rows,
                "sha256": report["sha256"],
            }
        primary = payloads["primary"]
        augmented = payloads["augmented"]
        if not torch.equal(primary["source_idx"], augmented["source_idx"]):
            raise RuntimeError(f"source_idx differs in embedding part {part_index}")
        if not torch.equal(primary["targets"], augmented["targets"]):
            raise RuntimeError(f"targets differ in embedding part {part_index}")
        indices = primary["source_idx"].numpy()
        if (indices < 0).any() or (indices >= source_rows).any():
            raise RuntimeError(f"source_idx outside contract in part {part_index}")
        if seen[indices].any():
            raise RuntimeError(f"duplicate source_idx in part {part_index}")
        seen[indices] = True
        rows = len(indices)
        total_rows += rows
        parts.append({"part": part_index + 1, "rows": rows, **reports})

    if total_rows != int(completions["primary"]["rows"]):
        raise RuntimeError("accepted embedding rows differ from primary completion")
    if total_rows != int(completions["augmented"]["rows"]):
        raise RuntimeError("accepted embedding rows differ from augmented completion")
    result = {
        "status": "accepted",
        "view": "primary_conformer_inference",
        "parts_per_variant": 100,
        "rows": total_rows,
        "source_rows": source_rows,
        "source_coverage": total_rows / source_rows,
        "embedding_dim": 176,
        "completions": completions,
        "parts": parts,
    }
    atomic_json(result, output_path)
    return result
