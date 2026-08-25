"""Pure-2D GPS-family fusion preflight.

This screen keeps only the Route A/B-style topology experts: GPS9 and
GPS11-160.  It never loads a conformer or a geometry payload.  The fusion
heads are small and train on aligned, frozen encoder payloads so the result is
an isolated architecture probe rather than a second encoder training line.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from molgap.qm9_screen import _metrics, set_seed


NAMES = ("gps9", "gps11_160")
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


class _ProjectedExperts(nn.Module):
    def __init__(self, input_dims: dict[str, int], hidden: int):
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

    def project(self, values: dict[str, Tensor]) -> list[Tensor]:
        return [self.projections[name](values[name]) for name in self.names]


class StandardGate(_ProjectedExperts):
    def __init__(self, input_dims: dict[str, int], hidden: int):
        super().__init__(input_dims, hidden)
        self.gate = nn.Linear(hidden * len(self.names), len(self.names))
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, values: dict[str, Tensor], predictions: Tensor | None = None) -> Tensor:
        projected = self.project(values)
        weights = torch.softmax(self.gate(torch.cat(projected, dim=-1)), dim=-1)
        fused = sum(weights[:, i : i + 1] * value for i, value in enumerate(projected))
        return self.head(fused)


class TargetSpecificGate(_ProjectedExperts):
    """Shared expert gate with one small readout per orbital target."""

    def __init__(self, input_dims: dict[str, int], hidden: int):
        super().__init__(input_dims, hidden)
        self.gate = nn.Linear(hidden * len(self.names), len(self.names))
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, hidden // 2),
                    nn.SiLU(),
                    nn.Linear(hidden // 2, 1),
                )
                for _ in range(3)
            ]
        )

    def forward(self, values: dict[str, Tensor], predictions: Tensor | None = None) -> Tensor:
        projected = self.project(values)
        weights = torch.softmax(self.gate(torch.cat(projected, dim=-1)), dim=-1)
        fused = sum(weights[:, i : i + 1] * value for i, value in enumerate(projected))
        return torch.cat([head(fused) for head in self.heads], dim=-1)


class PredictionAwareGate(StandardGate):
    def __init__(self, input_dims: dict[str, int], hidden: int):
        _ProjectedExperts.__init__(self, input_dims, hidden)
        self.gate = nn.Sequential(
            nn.Linear(hidden * len(self.names) + 3 * len(self.names), hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(self.names)),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden + 3 * len(self.names), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, values: dict[str, Tensor], predictions: Tensor | None = None) -> Tensor:
        if predictions is None:
            raise ValueError("PredictionAwareGate requires expert predictions")
        projected = self.project(values)
        weights = torch.softmax(self.gate(torch.cat(projected + [predictions], dim=-1)), dim=-1)
        fused = sum(weights[:, i : i + 1] * value for i, value in enumerate(projected))
        return self.head(torch.cat((fused, predictions), dim=-1))


class ConcatHead(_ProjectedExperts):
    def __init__(self, input_dims: dict[str, int], hidden: int):
        super().__init__(input_dims, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * len(self.names), hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Linear(hidden // 2, 3),
        )

    def forward(self, values: dict[str, Tensor], predictions: Tensor | None = None) -> Tensor:
        return self.head(torch.cat(self.project(values), dim=-1))


def _load_aligned(paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    raw = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    aligned: dict[str, dict[str, object]] = {}
    for role in ROLES:
        positions = {
            name: {int(value): i for i, value in enumerate(payload[role]["source_idx"].tolist())}
            for name, payload in raw.items()
        }
        common = sorted(set.intersection(*(set(item) for item in positions.values())))
        if not common:
            raise ValueError(f"No common rows for {role}")
        source_idx = torch.tensor(common, dtype=torch.long)
        embeddings: dict[str, Tensor] = {}
        predictions: dict[str, Tensor] = {}
        targets: Tensor | None = None
        for name in NAMES:
            index = torch.tensor([positions[name][value] for value in common], dtype=torch.long)
            source = raw[name][role]
            selected_targets = source["targets"].index_select(0, index).float()
            if targets is None:
                targets = selected_targets
            elif not torch.equal(targets, selected_targets):
                raise ValueError(f"Target mismatch for {name}/{role}")
            embeddings[name] = source["embeddings"].index_select(0, index).float()
            predictions[name] = source["predictions"].index_select(0, index).float()
        assert targets is not None
        aligned[role] = {
            "source_idx": source_idx,
            "embeddings": embeddings,
            "predictions": predictions,
            "targets": targets,
        }
    return aligned


def _values(item: dict[str, object], index: Tensor, device: torch.device) -> dict[str, Tensor]:
    embeddings = item["embeddings"]
    return {name: embeddings[name].index_select(0, index).to(device) for name in NAMES}


def _prediction_features(
    item: dict[str, object],
    index: Tensor,
    means: dict[str, Tensor],
    stds: dict[str, Tensor],
    device: torch.device,
) -> Tensor:
    predictions = item["predictions"]
    return torch.cat(
        [
            (predictions[name].index_select(0, index).to(device) - means[name].to(device))
            / stds[name].to(device)
            for name in NAMES
        ],
        dim=-1,
    )


def _make_model(variant: str, input_dims: dict[str, int], hidden: int) -> nn.Module:
    if variant == "standard_gate":
        return StandardGate(input_dims, hidden)
    if variant == "target_specific_gate":
        return TargetSpecificGate(input_dims, hidden)
    if variant == "prediction_aware_gate":
        return PredictionAwareGate(input_dims, hidden)
    if variant == "concat":
        return ConcatHead(input_dims, hidden)
    raise ValueError(variant)


def _train_variant(
    *, variant: str, aligned: dict[str, dict[str, object]], output_dir: Path,
    epochs: int, hidden: int, seed: int,
) -> dict:
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = aligned["train"]
    target_mean = train["targets"].mean(dim=0)
    target_std = train["targets"].std(dim=0).clamp_min(1e-6)
    prediction_means = {name: train["predictions"][name].mean(dim=0) for name in NAMES}
    prediction_stds = {name: train["predictions"][name].std(dim=0).clamp_min(1e-6) for name in NAMES}
    input_dims = {name: int(train["embeddings"][name].shape[1]) for name in NAMES}
    model = _make_model(variant, input_dims, hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    normalized_targets = (train["targets"] - target_mean) / target_std
    generator = torch.Generator().manual_seed(seed)
    best_value = float("inf")
    best_epoch = -1
    best_state = None
    log = []
    batch_size = 512
    for epoch in range(epochs):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train["source_idx"]), generator=generator)
        for begin in range(0, len(order), batch_size):
            index = order[begin : begin + batch_size]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                _values(train, index, device),
                _prediction_features(train, index, prediction_means, prediction_stds, device)
                if variant == "prediction_aware_gate" else None,
            )
            loss = nn.functional.smooth_l1_loss(prediction, normalized_targets[index].to(device), beta=0.05)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            index = torch.arange(len(aligned["validation"]["source_idx"]))
            validation = model(
                _values(aligned["validation"], index, device),
                _prediction_features(aligned["validation"], index, prediction_means, prediction_stds, device)
                if variant == "prediction_aware_gate" else None,
            ).cpu() * target_std + target_mean
        metrics = _metrics(validation.numpy(), aligned["validation"]["targets"].numpy())
        value = metrics["average"]["mae"]
        improved = value < best_value
        if improved:
            best_value = value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
        log.append({"epoch": epoch, "validation_average_mae_eV": value, "selected": improved, "elapsed_s": time.perf_counter() - started})
        _atomic_save(output_dir / "checkpoint.pt", {"variant": variant, "epoch": epoch, "best_epoch": best_epoch, "best_validation_average_mae_eV": best_value, "model": model.state_dict(), "log": log})
        _atomic_json(output_dir / "training_log.json", log)
    if best_state is None:
        raise RuntimeError(f"No checkpoint for {variant}")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        index = torch.arange(len(aligned["test"]["source_idx"]))
        prediction = model(
            _values(aligned["test"], index, device),
            _prediction_features(aligned["test"], index, prediction_means, prediction_stds, device)
            if variant == "prediction_aware_gate" else None,
        ).cpu() * target_std + target_mean
    test_metrics = _metrics(prediction.numpy(), aligned["test"]["targets"].numpy())
    _atomic_save(output_dir / "model.pt", {"variant": variant, "input_dims": input_dims, "hidden": hidden, "best_epoch": best_epoch, "model": model.state_dict(), "target_mean": target_mean, "target_std": target_std})
    return {"variant": variant, "input_dims": input_dims, "hidden": hidden, "n_params": sum(p.numel() for p in model.parameters()), "seed": seed, "device": str(device), "aligned_rows": {role: len(aligned[role]["source_idx"]) for role in ROLES}, "best_epoch": best_epoch, "best_validation_average_mae_eV": best_value, "test_metrics": test_metrics, "log": log, "artifacts": {"checkpoint": str(output_dir / "checkpoint.pt"), "model": str(output_dir / "model.pt")}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gps9-payload", type=Path, required=True)
    parser.add_argument("--gps11-payload", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variants", nargs="+", default=["target_specific_gate", "standard_gate", "prediction_aware_gate"], choices=["target_specific_gate", "standard_gate", "prediction_aware_gate", "concat"])
    args = parser.parse_args()
    paths = {"gps9": args.gps9_payload, "gps11_160": args.gps11_payload}
    aligned = _load_aligned(paths)
    _atomic_save(args.output_dir / "aligned_payload.pt", aligned)
    summary = {"experiment": "qm9_pure2d_gps_fusion_preflight", "seed": args.seed, "geometry": "topology_only", "inputs": {name: str(path) for name, path in paths.items()}, "split": {role: len(aligned[role]["source_idx"]) for role in ROLES}, "variants": {}}
    for offset, variant in enumerate(args.variants):
        result = _train_variant(variant=variant, aligned=aligned, output_dir=args.output_dir / variant, epochs=args.epochs, hidden=args.hidden, seed=args.seed + offset)
        summary["variants"][variant] = result
        _atomic_json(args.output_dir / variant / "metrics.json", result)
        print(json.dumps({variant: result["test_metrics"]}, indent=2), flush=True)
    _atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
