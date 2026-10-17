"""Development-only bounded fusion for accepted PCQM Route B embeddings."""

from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .pcqm_route_b_training import atomic_json, atomic_torch, sha256_file


ENCODER_NAMES = (
    "gps9",
    "gps11_160",
    "primary_schnet",
    "augmented_schnet",
)
EXPECTED_DIMS = {
    "gps9": 192,
    "gps11_160": 160,
    "primary_schnet": 176,
    "augmented_schnet": 176,
}
EXPECTED_ROWS = {"train": 915_012, "dev": 81_961, "official": 4_981}
EXPECTED_PARTS = {"train": 200, "dev": 200, "official": 1}
BASE_IDENTITIES = ("gps11_160", "augmented_schnet")


@dataclass(frozen=True)
class FusionConfig:
    hidden: int = 128
    correction_scale_eV: float = 0.10
    batch_size: int = 4096
    learning_rate: float = 3.0e-4
    weight_decay: float = 1.0e-5
    max_epochs: int = 40
    patience: int = 8


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()


def _part_map(output_dir: Path, role: str) -> tuple[dict[str, dict], dict]:
    manifest_path = output_dir / "embeddings" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    name = output_dir.name
    if (
        manifest.get("status") != "complete"
        or manifest.get("name") != name
        or manifest.get("embedding_dim") != EXPECTED_DIMS[name]
        or manifest.get("rows") != EXPECTED_ROWS
        or len(manifest.get("parts", [])) != sum(EXPECTED_PARTS.values())
        or manifest.get("official_valid_metric_read") is not False
    ):
        raise RuntimeError(f"Unaccepted embedding manifest: {manifest_path}")
    parts = {
        Path(item["source_shard"]).name: item
        for item in manifest["parts"]
        if item["role"] == role
    }
    expected_parts = EXPECTED_PARTS[role]
    if len(parts) != expected_parts:
        raise RuntimeError(f"{name}/{role} expected {expected_parts} parts")
    return parts, manifest


def load_aligned_roles(
    encoder_dirs: dict[str, Path],
    roles: tuple[str, ...] = ("train", "dev"),
) -> tuple[dict[str, dict[str, torch.Tensor]], dict]:
    """Load accepted parts and verify exact cross-encoder identity alignment."""
    if set(encoder_dirs) != set(ENCODER_NAMES):
        raise ValueError(f"Expected encoders {ENCODER_NAMES}")
    payloads = {name: {} for name in ENCODER_NAMES}
    report = {"roles": {}, "manifests": {}}
    for role in roles:
        part_maps = {}
        for name in ENCODER_NAMES:
            part_maps[name], manifest = _part_map(encoder_dirs[name], role)
            report["manifests"][name] = sha256_file(
                encoder_dirs[name] / "embeddings" / "manifest.json"
            )
        shard_names = sorted(part_maps[ENCODER_NAMES[0]])
        if any(set(part_maps[name]) != set(shard_names) for name in ENCODER_NAMES):
            raise RuntimeError(f"{role} embedding shard identities differ")

        embeddings = {name: [] for name in ENCODER_NAMES}
        indices, targets = [], []
        for shard_number, shard_name in enumerate(shard_names, start=1):
            reference_idx = None
            reference_target = None
            for name in ENCODER_NAMES:
                item = part_maps[name][shard_name]
                path = encoder_dirs[name] / "embeddings" / item["path"]
                if (
                    not path.is_file()
                    or path.stat().st_size != item["bytes"]
                    or sha256_file(path) != item["sha256"]
                ):
                    raise RuntimeError(f"Artifact differs: {path}")
                part = torch.load(path, map_location="cpu", weights_only=False)
                source_idx = part["source_idx"].view(-1).long()
                embedding = part["embeddings"].to(torch.float16)
                if (
                    part.get("name") != name
                    or part.get("role") != role
                    or len(source_idx) != item["rows"]
                    or embedding.shape
                    != (item["rows"], EXPECTED_DIMS[name])
                    or not torch.isfinite(embedding).all()
                ):
                    raise RuntimeError(f"Invalid embedding part: {path}")
                target = part.get("targets")
                if role == "official":
                    if target is not None:
                        raise RuntimeError("Official embedding part contains targets")
                else:
                    if target is None:
                        raise RuntimeError(f"Targets missing: {path}")
                    target = target.view(-1).float()
                    if len(target) != len(source_idx) or not torch.isfinite(target).all():
                        raise RuntimeError(f"Invalid targets: {path}")
                if reference_idx is None:
                    reference_idx = source_idx
                    reference_target = target
                elif not torch.equal(source_idx, reference_idx) or (
                    role != "official"
                    and not torch.equal(target, reference_target)
                ):
                    raise RuntimeError(f"Cross-encoder alignment differs: {shard_name}")
                embeddings[name].append(embedding)
            indices.append(reference_idx)
            if role != "official":
                targets.append(reference_target)
            if shard_number % 25 == 0 or shard_number == len(shard_names):
                print(
                    f"loaded {role} parts {shard_number}/{len(shard_names)}",
                    flush=True,
                )

        source_idx = torch.cat(indices)
        if len(source_idx.unique()) != len(source_idx):
            raise RuntimeError(f"{role} contains duplicate source_idx")
        rows = len(source_idx)
        if rows != EXPECTED_ROWS[role]:
            raise RuntimeError(f"{role} row count differs: {rows}")
        for name in ENCODER_NAMES:
            payloads[name][role] = torch.cat(embeddings[name])
        if role != "official":
            payloads[ENCODER_NAMES[0]][f"{role}_targets"] = torch.cat(targets)
        report["roles"][role] = {
            "rows": rows,
            "parts": len(shard_names),
            "source_idx_sha256": _tensor_sha256(source_idx),
            "target_sha256": (
                _tensor_sha256(torch.cat(targets))
                if role != "official"
                else None
            ),
        }
    return payloads, report


class FrozenGapReadout(nn.Module):
    """Apply an accepted encoder's frozen MLP head to cached embeddings."""

    def __init__(self, checkpoint: dict):
        super().__init__()
        state = checkpoint["model"]
        for key in (
            "head.0.weight",
            "head.0.bias",
            "head.3.weight",
            "head.3.bias",
            "head.5.weight",
            "head.5.bias",
        ):
            if key not in state:
                raise RuntimeError(f"Checkpoint lacks {key}")
        self.register_buffer("w0", state["head.0.weight"].float())
        self.register_buffer("b0", state["head.0.bias"].float())
        self.register_buffer("w1", state["head.3.weight"].float())
        self.register_buffer("b1", state["head.3.bias"].float())
        self.register_buffer("w2", state["head.5.weight"].float())
        self.register_buffer("b2", state["head.5.bias"].float())
        self.target_mean = float(checkpoint.get("target_mean_gap", 0.0))
        self.target_std = float(checkpoint.get("target_std_gap", 1.0))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(F.linear(embedding, self.w0, self.b0))
        hidden = F.silu(F.linear(hidden, self.w1, self.b1))
        prediction = F.linear(hidden, self.w2, self.b2)[:, 2]
        return prediction * self.target_std + self.target_mean


class PCQMBoundedFusion(nn.Module):
    """Preserve one frozen prediction and learn a bounded expert correction."""

    def __init__(
        self,
        base_checkpoint: dict,
        base_name: str,
        hidden: int,
        correction_scale_eV: float,
    ):
        super().__init__()
        if base_name not in BASE_IDENTITIES:
            raise ValueError(f"Unsupported base identity: {base_name}")
        if correction_scale_eV <= 0:
            raise ValueError("correction_scale_eV must be positive")
        self.base_name = base_name
        self.correction_scale_eV = correction_scale_eV
        self.base_readout = FrozenGapReadout(base_checkpoint)
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(EXPECTED_DIMS[name]),
                    nn.Linear(EXPECTED_DIMS[name], hidden),
                    nn.SiLU(),
                )
                for name in ENCODER_NAMES
            }
        )
        self.correction = nn.Sequential(
            nn.Linear(hidden * len(ENCODER_NAMES), hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, embeddings: dict[str, torch.Tensor]) -> torch.Tensor:
        base = self.base_readout(embeddings[self.base_name])
        context = torch.cat(
            [self.projections[name](embeddings[name]) for name in ENCODER_NAMES],
            dim=-1,
        )
        correction = self.correction(context).view(-1)
        return base + self.correction_scale_eV * torch.tanh(correction)


def preflight_fusion(
    *,
    encoder_dirs: dict[str, Path],
    output_path: Path,
    config: FusionConfig | None = None,
    batch_rows: int = 8,
) -> dict:
    """Run real CUDA forward/backward steps without opening official labels."""
    config = config or FusionConfig()
    if not torch.cuda.is_available():
        raise RuntimeError("PCQM Route B fusion preflight requires a GPU")
    part_maps = {
        name: _part_map(encoder_dirs[name], "train")[0]
        for name in ENCODER_NAMES
    }
    shared = set(part_maps[ENCODER_NAMES[0]])
    if any(set(part_maps[name]) != shared for name in ENCODER_NAMES):
        raise RuntimeError("Preflight train shard identities differ")
    shard_name = sorted(shared)[0]
    values, reference_idx, reference_target = {}, None, None
    artifacts = {}
    for name in ENCODER_NAMES:
        item = part_maps[name][shard_name]
        path = encoder_dirs[name] / "embeddings" / item["path"]
        if (
            path.stat().st_size != item["bytes"]
            or sha256_file(path) != item["sha256"]
        ):
            raise RuntimeError(f"Preflight artifact differs: {path}")
        part = torch.load(path, map_location="cpu", weights_only=False)
        source_idx = part["source_idx"].view(-1).long()
        target = part["targets"].view(-1).float()
        if reference_idx is None:
            reference_idx, reference_target = source_idx, target
        elif not torch.equal(source_idx, reference_idx) or not torch.equal(
            target, reference_target
        ):
            raise RuntimeError("Preflight source or target alignment differs")
        values[name] = part["embeddings"][:batch_rows].float().cuda()
        artifacts[name] = {
            "embedding_part": item["path"],
            "embedding_part_sha256": item["sha256"],
            "checkpoint_sha256": sha256_file(encoder_dirs[name] / "best.pt"),
        }

    losses = {}
    for base_name in BASE_IDENTITIES:
        checkpoint = torch.load(
            encoder_dirs[base_name] / "best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = PCQMBoundedFusion(
            checkpoint,
            base_name=base_name,
            hidden=config.hidden,
            correction_scale_eV=config.correction_scale_eV,
        ).cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        prediction = model(values)
        loss = F.l1_loss(prediction, reference_target[:batch_rows].cuda())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f"{base_name} preflight loss is not finite")
        losses[base_name] = float(loss.detach())
    report = {
        "format": "molgap-pcqm-route-b-fusion-preflight-v1",
        "status": "complete",
        "device": torch.cuda.get_device_name(0),
        "config": asdict(config),
        "batch_rows": batch_rows,
        "losses": losses,
        "artifacts": artifacts,
        "official_valid_metric_read": False,
        "official_test_used": False,
        "sealed_20k_used": False,
        "production_registry_changed": False,
    }
    atomic_json(output_path, report)
    return report


def preflight_fusion_from_payloads(
    *,
    payloads: dict[str, dict[str, torch.Tensor]],
    checkpoint_paths: dict[str, Path],
    output_path: Path,
    config: FusionConfig | None = None,
    batch_rows: int = 8,
) -> dict:
    """Run the same CUDA gate from an accepted consolidated payload."""
    config = config or FusionConfig()
    if not torch.cuda.is_available():
        raise RuntimeError("PCQM Route B fusion preflight requires a GPU")
    values = {
        name: payloads[name]["train"][:batch_rows].float().cuda()
        for name in ENCODER_NAMES
    }
    reference_target = payloads[ENCODER_NAMES[0]]["train_targets"][:batch_rows]
    losses = {}
    artifacts = {}
    for name in ENCODER_NAMES:
        artifacts[name] = {
            "checkpoint_sha256": sha256_file(checkpoint_paths[name]),
        }
    for base_name in BASE_IDENTITIES:
        checkpoint = torch.load(
            checkpoint_paths[base_name],
            map_location="cpu",
            weights_only=False,
        )
        model = PCQMBoundedFusion(
            checkpoint,
            base_name=base_name,
            hidden=config.hidden,
            correction_scale_eV=config.correction_scale_eV,
        ).cuda()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        prediction = model(values)
        loss = F.l1_loss(prediction, reference_target.cuda())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise RuntimeError(f"{base_name} preflight loss is not finite")
        losses[base_name] = float(loss.detach())
    report = {
        "format": "molgap-pcqm-route-b-consolidated-fusion-preflight-v1",
        "status": "complete",
        "device": torch.cuda.get_device_name(0),
        "config": asdict(config),
        "batch_rows": batch_rows,
        "losses": losses,
        "artifacts": artifacts,
        "official_valid_metric_read": False,
        "official_test_used": False,
        "sealed_20k_used": False,
        "production_registry_changed": False,
    }
    atomic_json(output_path, report)
    return report


def _contract_hash(
    base_name: str,
    seed: int,
    config: FusionConfig,
    checkpoint_paths: dict[str, Path],
    load_report: dict,
) -> str:
    contract = {
        "base_name": base_name,
        "seed": seed,
        "config": asdict(config),
        "checkpoint_sha256": {
            name: sha256_file(path) for name, path in checkpoint_paths.items()
        },
        "load_report": load_report,
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")
    ).hexdigest()


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    payloads: dict[str, dict[str, torch.Tensor]],
    role: str,
    target: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> float:
    model.eval()
    errors = []
    for begin in range(0, len(target), batch_size):
        index = slice(begin, begin + batch_size)
        values = {
            name: payloads[name][role][index].float().to(device)
            for name in ENCODER_NAMES
        }
        prediction = model(values)
        errors.append((prediction.cpu() - target[index]).abs())
    return float(torch.cat(errors).mean())


def train_fusion_seed(
    *,
    payloads: dict[str, dict[str, torch.Tensor]],
    load_report: dict,
    checkpoint_paths: dict[str, Path],
    base_name: str,
    seed: int,
    output_dir: Path,
    config: FusionConfig,
) -> dict:
    """Train or resume one development-selected bounded fusion seed."""
    _set_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("PCQM Route B fusion requires a GPU")
    device = torch.device("cuda")
    checkpoints = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in checkpoint_paths.items()
    }
    model = PCQMBoundedFusion(
        checkpoints[base_name],
        base_name=base_name,
        hidden=config.hidden,
        correction_scale_eV=config.correction_scale_eV,
    ).to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs, eta_min=1.0e-6
    )
    contract_hash = _contract_hash(
        base_name, seed, config, checkpoint_paths, load_report
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"
    start_epoch, best_epoch, best_dev, wait, log = 0, -1, float("inf"), 0, []
    if last_path.exists():
        last = torch.load(last_path, map_location=device, weights_only=False)
        if last.get("contract_hash") != contract_hash:
            raise RuntimeError(f"Resume contract differs: {last_path}")
        model.load_state_dict(last["model"], strict=True)
        optimizer.load_state_dict(last["optimizer"])
        scheduler.load_state_dict(last["scheduler"])
        start_epoch = int(last["next_epoch"])
        best_epoch = int(last["best_epoch"])
        best_dev = float(last["best_dev_gap_mae_eV"])
        wait = int(last["wait"])
        log = list(last["log"])

    target = payloads[ENCODER_NAMES[0]]["train_targets"]
    dev_target = payloads[ENCODER_NAMES[0]]["dev_targets"]
    # Evaluate the frozen readout directly; it is also an integrity check that
    # cached embeddings and the accepted checkpoint belong together.
    with torch.no_grad():
        base_errors = []
        for begin in range(0, len(dev_target), config.batch_size):
            index = slice(begin, begin + config.batch_size)
            prediction = model.base_readout(
                payloads[base_name]["dev"][index].float().to(device)
            )
            base_errors.append((prediction.cpu() - dev_target[index]).abs())
        base_dev = float(torch.cat(base_errors).mean())

    generator = torch.Generator().manual_seed(seed)
    for epoch in range(start_epoch, config.max_epochs):
        started = time.monotonic()
        model.train()
        order = torch.randperm(len(target), generator=generator)
        total_loss, total_rows = 0.0, 0
        for begin in range(0, len(order), config.batch_size):
            index = order[begin : begin + config.batch_size]
            values = {
                name: payloads[name]["train"][index].float().to(device)
                for name in ENCODER_NAMES
            }
            y = target[index].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(values)
            loss = F.l1_loss(prediction, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(index)
            total_rows += len(index)
        scheduler.step()
        dev_mae = _evaluate(
            model,
            payloads,
            "dev",
            dev_target,
            config.batch_size,
            device,
        )
        improved = dev_mae < best_dev
        if improved:
            best_dev, best_epoch, wait = dev_mae, epoch, 0
            atomic_torch(
                best_path,
                {
                    "format": "molgap-pcqm-route-b-fusion-best-v1",
                    "base_name": base_name,
                    "seed": seed,
                    "config": asdict(config),
                    "contract_hash": contract_hash,
                    "epoch": epoch,
                    "model": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "base_dev_gap_mae_eV": base_dev,
                    "best_dev_gap_mae_eV": best_dev,
                },
            )
        else:
            wait += 1
        row = {
            "epoch": epoch,
            "train_gap_mae_eV": total_loss / total_rows,
            "dev_gap_mae_eV": dev_mae,
            "selected": improved,
            "lr": scheduler.get_last_lr()[0],
            "elapsed_s": time.monotonic() - started,
        }
        log.append(row)
        atomic_torch(
            last_path,
            {
                "format": "molgap-pcqm-route-b-fusion-resume-v1",
                "base_name": base_name,
                "seed": seed,
                "config": asdict(config),
                "contract_hash": contract_hash,
                "next_epoch": epoch + 1,
                "best_epoch": best_epoch,
                "best_dev_gap_mae_eV": best_dev,
                "wait": wait,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "log": log,
            },
        )
        atomic_json(output_dir / "training_log.json", log)
        print(
            f"{base_name} seed={seed} ep{epoch:02d} "
            f"train={row['train_gap_mae_eV']:.6f} "
            f"dev={dev_mae:.6f}eV {row['elapsed_s']:.1f}s"
            f"{' *' if improved else ''}",
            flush=True,
        )
        if wait >= config.patience:
            break
    result = {
        "format": "molgap-pcqm-route-b-fusion-seed-v1",
        "status": "complete",
        "base_name": base_name,
        "seed": seed,
        "config": asdict(config),
        "base_dev_gap_mae_eV": base_dev,
        "best_epoch": best_epoch,
        "best_dev_gap_mae_eV": best_dev,
        "contract_hash": contract_hash,
        "official_valid_metric_read": False,
        "official_test_used": False,
        "sealed_20k_used": False,
        "production_registry_changed": False,
    }
    atomic_json(output_dir / "metrics.json", result)
    return result


def train_fusion_screen(
    *,
    encoder_dirs: dict[str, Path],
    output_dir: Path,
    seeds: tuple[int, ...] = (42, 43, 44),
    config: FusionConfig | None = None,
) -> dict:
    """Run the frozen two-identity, three-seed development-only screen."""
    config = config or FusionConfig()
    if config.correction_scale_eV != 0.10:
        raise ValueError("The transferred correction bound is frozen at 0.10 eV")
    payloads, load_report = load_aligned_roles(encoder_dirs, ("train", "dev"))
    checkpoint_paths = {
        name: encoder_dirs[name] / "best.pt" for name in ENCODER_NAMES
    }
    return train_fusion_screen_from_payloads(
        payloads=payloads,
        load_report=load_report,
        checkpoint_paths=checkpoint_paths,
        output_dir=output_dir,
        seeds=seeds,
        config=config,
    )


def train_fusion_screen_from_payloads(
    *,
    payloads: dict[str, dict[str, torch.Tensor]],
    load_report: dict,
    checkpoint_paths: dict[str, Path],
    output_dir: Path,
    seeds: tuple[int, ...] = (42, 43, 44),
    config: FusionConfig | None = None,
) -> dict:
    """Run the same frozen screen from a previously accepted merged payload."""
    config = config or FusionConfig()
    if config.correction_scale_eV != 0.10:
        raise ValueError("The transferred correction bound is frozen at 0.10 eV")
    for path in checkpoint_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    results = []
    for base_name in BASE_IDENTITIES:
        for seed in seeds:
            results.append(
                train_fusion_seed(
                    payloads=payloads,
                    load_report=load_report,
                    checkpoint_paths=checkpoint_paths,
                    base_name=base_name,
                    seed=seed,
                    output_dir=output_dir / base_name / f"seed_{seed}",
                    config=config,
                )
            )
    means = {
        base_name: float(
            np.mean(
                [
                    row["best_dev_gap_mae_eV"]
                    for row in results
                    if row["base_name"] == base_name
                ]
            )
        )
        for base_name in BASE_IDENTITIES
    }
    selected = min(means, key=means.get)
    summary = {
        "format": "molgap-pcqm-route-b-fusion-screen-v1",
        "status": "complete",
        "selection_scope": "scaffold-development-only",
        "config": asdict(config),
        "seeds": list(seeds),
        "results": results,
        "mean_dev_gap_mae_eV": means,
        "selected_base_identity": selected,
        "load_report": load_report,
        "official_valid_metric_read": False,
        "official_test_used": False,
        "sealed_20k_used": False,
        "production_registry_changed": False,
    }
    atomic_json(output_dir / "development_selection.json", summary)
    artifacts = {
        path.relative_to(output_dir).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path.name != "completion_manifest.json"
    }
    atomic_json(
        output_dir / "completion_manifest.json",
        {
            "status": "complete",
            "selected_base_identity": selected,
            "artifacts": artifacts,
            "official_valid_metric_read": False,
        },
    )
    return summary


def export_consolidated_fusion_payload(
    *,
    encoder_dirs: dict[str, Path],
    output_dir: Path,
) -> dict:
    """Merge accepted train/dev parts into one portable, immutable payload."""
    payloads, load_report = load_aligned_roles(encoder_dirs, ("train", "dev"))
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "fusion_payload.pt"
    atomic_torch(
        payload_path,
        {
            "format": "molgap-pcqm-route-b-consolidated-fusion-v1",
            "payloads": payloads,
            "load_report": load_report,
            "official_valid_metric_read": False,
            "official_test_used": False,
            "sealed_20k_used": False,
        },
    )

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {
        "fusion_payload.pt": {
            "bytes": payload_path.stat().st_size,
            "sha256": sha256_file(payload_path),
        }
    }
    for name in ENCODER_NAMES:
        source = encoder_dirs[name] / "best.pt"
        destination = checkpoint_dir / f"{name}_best.pt"
        temporary = destination.with_suffix(".tmp")
        if temporary.exists():
            raise FileExistsError(temporary)
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
        artifacts[destination.relative_to(output_dir).as_posix()] = {
            "bytes": destination.stat().st_size,
            "sha256": sha256_file(destination),
        }

    report = {
        "format": "molgap-pcqm-route-b-consolidated-fusion-manifest-v1",
        "status": "complete",
        "encoder_names": list(ENCODER_NAMES),
        "rows": {"train": EXPECTED_ROWS["train"], "dev": EXPECTED_ROWS["dev"]},
        "embedding_dims": EXPECTED_DIMS,
        "artifacts": artifacts,
        "load_report": load_report,
        "official_valid_metric_read": False,
        "official_test_used": False,
        "sealed_20k_used": False,
        "production_registry_changed": False,
    }
    atomic_json(output_dir / "manifest.json", report)
    return report


def load_consolidated_fusion_payload(
    input_dir: Path,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict, dict[str, Path]]:
    """Validate and load a consolidated payload without opening official labels."""
    manifest_path = input_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("format")
        != "molgap-pcqm-route-b-consolidated-fusion-manifest-v1"
        or manifest.get("status") != "complete"
        or manifest.get("encoder_names") != list(ENCODER_NAMES)
        or manifest.get("rows")
        != {"train": EXPECTED_ROWS["train"], "dev": EXPECTED_ROWS["dev"]}
        or manifest.get("embedding_dims") != EXPECTED_DIMS
        or manifest.get("official_valid_metric_read") is not False
    ):
        raise RuntimeError(f"invalid consolidated manifest: {manifest_path}")
    for relative, expected in manifest["artifacts"].items():
        path = input_dir / relative
        if (
            not path.is_file()
            or path.stat().st_size != expected["bytes"]
            or sha256_file(path) != expected["sha256"]
        ):
            raise RuntimeError(f"consolidated artifact differs: {path}")

    consolidated = torch.load(
        input_dir / "fusion_payload.pt",
        map_location="cpu",
        weights_only=False,
    )
    if (
        consolidated.get("format")
        != "molgap-pcqm-route-b-consolidated-fusion-v1"
        or consolidated.get("official_valid_metric_read") is not False
    ):
        raise RuntimeError("invalid consolidated payload contract")
    payloads = consolidated["payloads"]
    for name in ENCODER_NAMES:
        for role in ("train", "dev"):
            value = payloads[name][role]
            if (
                value.shape != (EXPECTED_ROWS[role], EXPECTED_DIMS[name])
                or value.dtype != torch.float16
                or not torch.isfinite(value).all()
            ):
                raise RuntimeError(f"invalid consolidated tensor: {name}/{role}")
    for role in ("train", "dev"):
        target = payloads[ENCODER_NAMES[0]][f"{role}_targets"]
        if target.shape != (EXPECTED_ROWS[role],) or not torch.isfinite(target).all():
            raise RuntimeError(f"invalid consolidated targets: {role}")
    checkpoint_paths = {
        name: input_dir / "checkpoints" / f"{name}_best.pt"
        for name in ENCODER_NAMES
    }
    return payloads, consolidated["load_report"], checkpoint_paths
