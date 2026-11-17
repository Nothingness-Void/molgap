"""Recoverable, identity-aligned fusion for Route B frozen embeddings."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn


CANDIDATES = {
    "minimal": ("gps9", "schnet_primary", "schnet_augmented"),
    "cost": ("gps9", "gps7", "schnet_primary", "schnet_augmented"),
    "precision": ("gps9", "gps11_160", "schnet_primary", "schnet_augmented"),
}
FUSION_MODES = ("gated", "concat", "bounded_residual")
ROLES = ("train", "validation", "test")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    for attempt in range(20):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(0.1)


def _atomic_torch(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _contract_hash(
    candidate: str,
    payloads: dict[str, dict],
    fusion_mode: str,
    hidden: int,
    correction_scale_eV: float,
) -> str:
    contract = {
        "candidate": candidate,
        "fusion_mode": fusion_mode,
        "hidden": hidden,
        "correction_scale_eV": correction_scale_eV,
        "inputs": {
            name: {
                role: {
                    "rows": len(payload[role]["source_idx"]),
                    "dim": int(payload[role]["embeddings"].shape[1]),
                }
                for role in ROLES
            }
            for name, payload in sorted(payloads.items())
        },
    }
    return hashlib.sha256(
        json.dumps(contract, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_aligned_payloads(paths: dict[str, Path], candidate: str) -> dict:
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown Route B candidate: {candidate}")
    required = CANDIDATES[candidate]
    missing = sorted(set(required) - set(paths))
    if missing:
        raise ValueError(f"Missing Route B payloads: {missing}")
    payloads = {
        name: torch.load(paths[name], map_location="cpu", weights_only=False)
        for name in required
    }
    reference = payloads[required[0]]
    for role in ROLES:
        expected_keys = {"source_idx", "cid", "embeddings", "targets"}
        for name, payload in payloads.items():
            missing_keys = expected_keys - set(payload[role])
            if missing_keys:
                raise ValueError(f"{name}/{role} missing {sorted(missing_keys)}")
            if not torch.equal(
                payload[role]["source_idx"], reference[role]["source_idx"]
            ):
                raise ValueError(f"{name}/{role} source_idx alignment differs")
            if list(payload[role]["cid"]) != list(reference[role]["cid"]):
                raise ValueError(f"{name}/{role} CID alignment differs")
            if not torch.equal(payload[role]["targets"], reference[role]["targets"]):
                raise ValueError(f"{name}/{role} targets differ")
            embeddings = payload[role]["embeddings"]
            if embeddings.ndim != 2 or not torch.isfinite(embeddings).all():
                raise ValueError(f"{name}/{role} embeddings are invalid")
        if len(set(reference[role]["source_idx"].tolist())) != len(
            reference[role]["source_idx"]
        ):
            raise ValueError(f"Duplicate source_idx in {role}")
    return payloads


class MultiExpertFusionHead(nn.Module):
    """Project variable-width experts before gated fusion."""

    def __init__(self, input_dims: dict[str, int], hidden: int = 128):
        super().__init__()
        self.names = tuple(input_dims)
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(dimension),
                    nn.Linear(dimension, hidden),
                    nn.SiLU(),
                )
                for name, dimension in input_dims.items()
            }
        )
        self.gate = nn.Linear(hidden * len(input_dims), len(input_dims))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        projected = [self.projections[name](values[name]) for name in self.names]
        joined = torch.cat(projected, dim=-1)
        weights = torch.softmax(self.gate(joined), dim=-1)
        fused = sum(
            weights[:, index : index + 1] * value
            for index, value in enumerate(projected)
        )
        return self.head(fused)


class ConcatFusionHead(nn.Module):
    """Keep every projected expert visible to the output head."""

    def __init__(self, input_dims: dict[str, int], hidden: int = 128):
        super().__init__()
        self.names = tuple(input_dims)
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(dimension),
                    nn.Linear(dimension, hidden),
                    nn.SiLU(),
                )
                for name, dimension in input_dims.items()
            }
        )
        self.head = nn.Sequential(
            nn.Linear(hidden * len(input_dims), hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(
            torch.cat(
                [self.projections[name](values[name]) for name in self.names],
                dim=-1,
            )
        )


class BoundedResidualFusionHead(nn.Module):
    """Preserve the strongest GPS path and add a bounded expert correction."""

    def __init__(
        self,
        input_dims: dict[str, int],
        hidden: int = 128,
        correction_scale_eV: float = 0.25,
    ):
        super().__init__()
        self.names = tuple(input_dims)
        self.base_name = (
            "gps11_160" if "gps11_160" in input_dims else self.names[0]
        )
        self.correction_scale_eV = correction_scale_eV
        self.base_head = nn.Sequential(
            nn.LayerNorm(input_dims[self.base_name]),
            nn.Linear(input_dims[self.base_name], hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )
        self.projections = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(dimension),
                    nn.Linear(dimension, hidden),
                    nn.SiLU(),
                )
                for name, dimension in input_dims.items()
            }
        )
        self.correction = nn.Sequential(
            nn.Linear(hidden * len(input_dims), hidden),
            nn.SiLU(),
            nn.Linear(hidden, 3),
        )

    def forward(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        base = self.base_head(values[self.base_name])
        context = torch.cat(
            [self.projections[name](values[name]) for name in self.names], dim=-1
        )
        return base + self.correction_scale_eV * torch.tanh(
            self.correction(context)
        )


def build_fusion_head(
    fusion_mode: str,
    input_dims: dict[str, int],
    hidden: int,
    correction_scale_eV: float,
) -> nn.Module:
    if fusion_mode == "gated":
        return MultiExpertFusionHead(input_dims, hidden=hidden)
    if fusion_mode == "concat":
        return ConcatFusionHead(input_dims, hidden=hidden)
    if fusion_mode == "bounded_residual":
        return BoundedResidualFusionHead(
            input_dims,
            hidden=hidden,
            correction_scale_eV=correction_scale_eV,
        )
    raise ValueError(f"Unknown fusion mode: {fusion_mode}")


def _mae(prediction: torch.Tensor, target: torch.Tensor) -> dict:
    values = (prediction - target).abs().mean(dim=0)
    return {
        "homo": float(values[0]),
        "lumo": float(values[1]),
        "gap": float(values[2]),
        "average": float(values.mean()),
    }


def train_route_b_fusion(
    paths: dict[str, Path],
    candidate: str,
    out_dir: Path,
    *,
    epochs: int = 100,
    batch_size: int = 512,
    hidden: int = 128,
    seed: int = 42,
    resume: bool = False,
    fusion_mode: str = "gated",
    correction_scale_eV: float = 0.25,
) -> dict:
    if fusion_mode not in FUSION_MODES:
        raise ValueError(f"Unknown fusion mode: {fusion_mode}")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payloads = load_aligned_payloads(paths, candidate)
    names = CANDIDATES[candidate]
    reference = payloads[names[0]]
    input_dims = {
        name: int(payloads[name]["train"]["embeddings"].shape[1])
        for name in names
    }
    if correction_scale_eV <= 0:
        raise ValueError("correction_scale_eV must be positive")
    contract_hash = _contract_hash(
        candidate,
        payloads,
        fusion_mode,
        hidden,
        correction_scale_eV,
    )
    model = build_fusion_head(
        fusion_mode,
        input_dims,
        hidden,
        correction_scale_eV,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    best_mae = float("inf")
    best_epoch = -1
    start_epoch = 0
    log = []
    last_path = out_dir / "last.pt"
    if resume:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if checkpoint["contract_hash"] != contract_hash:
            raise ValueError("Resume payload contract differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["next_epoch"])
        best_mae = float(checkpoint["best_mae"])
        best_epoch = int(checkpoint["best_epoch"])
        log = list(checkpoint["log"])

    train = reference["train"]
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train["source_idx"]), generator=generator)
        for begin in range(0, len(order), batch_size):
            index = order[begin : begin + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                {
                    name: payloads[name]["train"]["embeddings"][index]
                    .float()
                    .to(device)
                    for name in names
                }
            )
            loss = nn.functional.l1_loss(
                prediction, train["targets"][index].float().to(device)
            )
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_prediction = model(
                {
                    name: payloads[name]["validation"]["embeddings"]
                    .float()
                    .to(device)
                    for name in names
                }
            )
        validation = _mae(
            validation_prediction,
            reference["validation"]["targets"].float().to(device),
        )
        improved = validation["average"] < best_mae
        if improved:
            best_mae = validation["average"]
            best_epoch = epoch
            _atomic_torch(
                out_dir / "best.pt",
                {
                    "candidate": candidate,
                    "fusion_mode": fusion_mode,
                    "correction_scale_eV": correction_scale_eV,
                    "input_dims": input_dims,
                    "hidden": hidden,
                    "epoch": epoch,
                    "contract_hash": contract_hash,
                    "model": model.state_dict(),
                },
            )
        log.append(
            {
                "epoch": epoch,
                "validation": validation,
                "selected": improved,
                "elapsed_s": time.perf_counter() - started,
            }
        )
        _atomic_torch(
            last_path,
            {
                "candidate": candidate,
                "fusion_mode": fusion_mode,
                "correction_scale_eV": correction_scale_eV,
                "input_dims": input_dims,
                "hidden": hidden,
                "next_epoch": epoch + 1,
                "best_mae": best_mae,
                "best_epoch": best_epoch,
                "contract_hash": contract_hash,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "log": log,
            },
        )
        _atomic_json(out_dir / "training_log.json", log)

    best = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    model.eval()
    with torch.no_grad():
        test_prediction = model(
                {
                    name: payloads[name]["test"]["embeddings"].float().to(device)
                    for name in names
                }
            )
    metrics = {
        "experiment": "route_b_frozen_embedding_fusion",
        "candidate": candidate,
        "fusion_mode": fusion_mode,
        "correction_scale_eV": correction_scale_eV,
        "inputs": list(names),
        "contract_hash": contract_hash,
        "best_epoch": best_epoch,
        "best_validation_average_mae_eV": best_mae,
        "test": _mae(
            test_prediction, reference["test"]["targets"].float().to(device)
        ),
        "device": str(device),
        "checkpoints": {"best": "best.pt", "last": "last.pt"},
        "resume_supported": True,
        "production_registry_changed": False,
    }
    _atomic_json(out_dir / "metrics.json", metrics)
    return metrics
