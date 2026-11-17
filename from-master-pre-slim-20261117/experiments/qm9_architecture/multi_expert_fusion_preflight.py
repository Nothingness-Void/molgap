"""Leakage-free four-expert QM9 fusion preflight.

The encoders are frozen.  This screen preserves the expert identities used by
the historical precision candidate instead of averaging the two topology or
SchNet views before the fusion head sees them.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from molgap.qm9_screen import _metrics, set_seed
from molgap.route_b_fusion import ConcatFusionHead, MultiExpertFusionHead


NAMES = ("gps9", "gps11_160", "schnet_primary", "schnet_augmented")
ROLES = ("train", "validation", "test")


def _atomic_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class PredictionAwareFusionHead(nn.Module):
    """Four-expert gate whose router also sees frozen expert predictions."""

    def __init__(self, input_dims: dict[str, int], hidden: int, n_experts: int):
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
        self.gate = nn.Sequential(
            nn.Linear(hidden * n_experts + 3 * n_experts, hidden),
            nn.SiLU(),
            nn.Linear(hidden, n_experts),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 3 * n_experts, hidden),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(
        self,
        values: dict[str, torch.Tensor],
        predictions: torch.Tensor,
    ) -> torch.Tensor:
        projected = [self.projections[name](values[name]) for name in self.names]
        joined = torch.cat(projected + [predictions], dim=-1)
        weights = torch.softmax(self.gate(joined), dim=-1)
        fused = sum(
            weights[:, index : index + 1] * value
            for index, value in enumerate(projected)
        )
        return self.head(torch.cat((fused, predictions), dim=-1))


class PhysicsConsistentFusionHead(nn.Module):
    """Four-expert gate with Gap generated exactly as LUMO minus HOMO."""

    def __init__(
        self,
        input_dims: dict[str, int],
        hidden: int,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ):
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
        self.gate = nn.Linear(hidden * len(self.names), len(self.names))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 2),
        )
        self.register_buffer("target_mean", target_mean.clone())
        self.register_buffer("target_std", target_std.clone())

    def forward(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        projected = [self.projections[name](values[name]) for name in self.names]
        weights = torch.softmax(self.gate(torch.cat(projected, dim=-1)), dim=-1)
        fused = sum(
            weights[:, index : index + 1] * value
            for index, value in enumerate(projected)
        )
        homo_lumo = self.head(fused)
        physical_homo_lumo = homo_lumo * self.target_std[:2] + self.target_mean[:2]
        physical_gap = physical_homo_lumo[:, 1:2] - physical_homo_lumo[:, 0:1]
        gap = (physical_gap - self.target_mean[2]) / self.target_std[2]
        return torch.cat((homo_lumo, gap), dim=-1)


def _load_aligned(paths: dict[str, Path]) -> dict[str, dict]:
    raw = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    aligned: dict[str, dict] = {}
    for role in ROLES:
        positions = {
            name: {
                int(value): index
                for index, value in enumerate(payload[role]["source_idx"].tolist())
            }
            for name, payload in raw.items()
        }
        common = sorted(set.intersection(*(set(item) for item in positions.values())))
        if not common:
            raise ValueError(f"No common rows for {role}")
        item: dict[str, object] = {
            "source_idx": torch.tensor(common, dtype=torch.long),
            "embeddings": {},
            "predictions": {},
        }
        targets = None
        for name in NAMES:
            index = torch.tensor([positions[name][value] for value in common])
            source = raw[name][role]
            selected_targets = source["targets"][index]
            if targets is None:
                targets = selected_targets
            elif not torch.equal(targets, selected_targets):
                raise ValueError(f"Target mismatch for {name}/{role}")
            item["embeddings"][name] = source["embeddings"][index].float()
            item["predictions"][name] = source["predictions"][index].float()
        item["targets"] = targets.float()
        aligned[role] = item
    return aligned


def _values(item: dict, index: torch.Tensor, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: item["embeddings"][name][index].to(device)
        for name in NAMES
    }


def _all_predictions(item: dict, index: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> torch.Tensor:
    mean, std = stats
    return torch.cat(
        [
            (item["predictions"][name][index].to(device) - mean[name].to(device))
            / std[name].to(device)
            for name in NAMES
        ],
        dim=-1,
    )


def _predict(
    model: nn.Module,
    item: dict,
    variant: str,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    pred_stats: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    index = torch.arange(len(item["source_idx"]))
    values = _values(item, index, device)
    if variant == "prediction_aware_gate":
        output = model(
            values,
            _all_predictions(item, index, pred_stats, device),
        )
    else:
        output = model(values)
    return output.cpu() * target_std + target_mean


def _train_variant(
    *,
    variant: str,
    aligned: dict[str, dict],
    output_dir: Path,
    epochs: int,
    hidden: int,
    seed: int,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = aligned["train"]
    target_mean = train["targets"].mean(dim=0)
    target_std = train["targets"].std(dim=0).clamp_min(1e-6)
    pred_mean = {
        name: train["predictions"][name].mean(dim=0)
        for name in NAMES
    }
    pred_std = {
        name: train["predictions"][name].std(dim=0).clamp_min(1e-6)
        for name in NAMES
    }
    pred_stats = (pred_mean, pred_std)
    input_dims = {
        name: int(train["embeddings"][name].shape[1])
        for name in NAMES
    }
    if variant == "standard_gate":
        model: nn.Module = MultiExpertFusionHead(input_dims, hidden=hidden)
    elif variant == "concat":
        model = ConcatFusionHead(input_dims, hidden=hidden)
    elif variant == "prediction_aware_gate":
        model = PredictionAwareFusionHead(input_dims, hidden, len(NAMES))
    elif variant == "physics_exact_gate":
        model = PhysicsConsistentFusionHead(
            input_dims,
            hidden,
            target_mean,
            target_std,
        )
    else:
        raise ValueError(f"Unknown variant: {variant}")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed)
    best_mae = float("inf")
    best_epoch = -1
    best_state = None
    log = []
    wait = 0
    batch_size = 512
    normalized_targets = (train["targets"] - target_mean) / target_std
    for epoch in range(epochs):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train["source_idx"]), generator=generator)
        for begin in range(0, len(order), batch_size):
            index = order[begin : begin + batch_size]
            optimizer.zero_grad(set_to_none=True)
            values = _values(train, index, device)
            if variant == "prediction_aware_gate":
                prediction = model(values, _all_predictions(train, index, pred_stats, device))
            else:
                prediction = model(values)
            loss = nn.functional.l1_loss(
                prediction,
                normalized_targets[index].to(device),
            )
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation = _predict(
                model,
                aligned["validation"],
                variant,
                target_mean,
                target_std,
                pred_stats,
                device,
            )
        validation_metrics = _metrics(
            validation.numpy(), aligned["validation"]["targets"].numpy()
        )
        value = validation_metrics["average"]["mae"]
        improved = value < best_mae
        if improved:
            best_mae = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
        log.append({
            "epoch": epoch,
            "validation_average_mae_eV": value,
            "elapsed_s": time.perf_counter() - started,
            "selected": improved,
        })
        _atomic_save(
            output_dir / "checkpoint.pt",
            {
                "variant": variant,
                "input_dims": input_dims,
                "hidden": hidden,
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_validation_average_mae_eV": best_mae,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "log": log,
            },
        )
        _atomic_json(output_dir / "training_log.json", log)
        if wait >= 15:
            break
    if best_state is None:
        raise RuntimeError(f"No checkpoint for {variant}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_prediction = _predict(
            model,
            aligned["test"],
            variant,
            target_mean,
            target_std,
            pred_stats,
            device,
        )
    test_metrics = _metrics(
        test_prediction.numpy(), aligned["test"]["targets"].numpy()
    )
    _atomic_save(
        output_dir / "model.pt",
        {
            "variant": variant,
            "input_dims": input_dims,
            "hidden": hidden,
            "best_epoch": best_epoch,
            "model": model.state_dict(),
            "target_mean": target_mean,
            "target_std": target_std,
            "prediction_mean": pred_mean,
            "prediction_std": pred_std,
        },
    )
    return {
        "variant": variant,
        "input_dims": input_dims,
        "hidden": hidden,
        "n_params": sum(parameter.numel() for parameter in model.parameters()),
        "seed": seed,
        "device": str(device),
        "aligned_rows": {role: len(aligned[role]["source_idx"]) for role in ROLES},
        "best_epoch": best_epoch,
        "best_validation_average_mae_eV": best_mae,
        "test_metrics": test_metrics,
        "log": log,
        "artifacts": {
            "checkpoint": str(output_dir / "checkpoint.pt"),
            "model": str(output_dir / "model.pt"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gps9-payload", type=Path, required=True)
    parser.add_argument("--gps11-payload", type=Path, required=True)
    parser.add_argument("--schnet-primary-payload", type=Path, required=True)
    parser.add_argument("--schnet-augmented-payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["standard_gate", "concat", "prediction_aware_gate"],
        choices=[
            "standard_gate",
            "concat",
            "prediction_aware_gate",
            "physics_exact_gate",
        ],
    )
    args = parser.parse_args()
    paths = {
        "gps9": args.gps9_payload,
        "gps11_160": args.gps11_payload,
        "schnet_primary": args.schnet_primary_payload,
        "schnet_augmented": args.schnet_augmented_payload,
    }
    aligned = _load_aligned(paths)
    _atomic_save(args.output_dir / "aligned_payload.pt", aligned)
    summary = {
        "experiment": "qm9_four_expert_fusion_preflight",
        "seed": args.seed,
        "split": {role: len(aligned[role]["source_idx"]) for role in ROLES},
        "inputs": {name: str(path) for name, path in paths.items()},
        "variants": {},
    }
    for offset, variant in enumerate(args.variants):
        result = _train_variant(
            variant=variant,
            aligned=aligned,
            output_dir=args.output_dir / variant,
            epochs=args.epochs,
            hidden=args.hidden,
            seed=args.seed + offset,
        )
        summary["variants"][variant] = result
        _atomic_json(args.output_dir / variant / "metrics.json", result)
        print(json.dumps({variant: result["test_metrics"]}, indent=2), flush=True)
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
