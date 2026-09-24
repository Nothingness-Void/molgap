"""Frozen GPTrans feature extraction and train-role-only residual-head probe."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader

from molgap.pcqm_500k_v4_evidence import make_model
from molgap.training_reproducibility import atomic_json, atomic_torch_save, sha256_file


MANIFEST_SHA = "630d30046d6cdc1f91fb169cd1eb4720bd5b352dc1ebeb641a11ead1ebae9751"
SHARD_SHA = {
    "train": "4753e34c24d535b56f0da81ff92c5bd70348dee4d1acf216f2cc2c2bca669104",
    "development": "1a37dc5d458396b590044d978526822e6d1cb35df223de913a837dc7277d64c1",
}
CHECKPOINT_SHA = {
    "baseline": "e311f1972ba74ade32e1f16be717c44848235eeafbafd592758775553445c79e",
    "pair_norm": "d46b0bbb9709a5933396a262a1e42323fc45b23e92c7121ffb66468f099529a5",
}
PREDICTION_SHA = {
    "baseline": "0608084eb2a7a90923a652235572ee69578c4e138d5df3c457e5170f7da613bf",
    "pair_norm": "731d67452b31bed74909023031eb9bbc122e8b0e022ff6e2c35bd3599d7113f4",
}


class PackedGraphs(InMemoryDataset):
    def __init__(self, path: Path) -> None:
        super().__init__(root=None)
        self.data, self.slices = torch.load(path, map_location="cpu", weights_only=False)


def _identity(path: Path, expected: str, label: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: {observed}")


@torch.inference_mode()
def _features(model: nn.Module, batch) -> tuple[torch.Tensor, torch.Tensor]:
    node, pair, padding = model._dense_inputs(
        batch.x, batch.edge_index, batch.edge_attr, batch.batch
    )
    for block in model.blocks:
        node, pair = block(node, pair, padding)
    virtual = torch.cat((node[:, 0], pair[:, :, 0, 0]), dim=-1)
    valid = (~padding[:, 0, 0, 1:]).unsqueeze(-1)
    atom_mean = (node[:, 1:] * valid).sum(1) / valid.sum(1).clamp_min(1)
    return torch.cat((virtual, atom_mean), dim=-1), model.readout(virtual).view(-1)


def extract(
    *, arm: str, cache_root: Path, checkpoint: Path, predictions: Path,
    output: Path, train_limit: int, development_limit: int, batch_size: int,
) -> dict:
    start = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen extraction requires a local CUDA device")
    if arm not in CHECKPOINT_SHA:
        raise ValueError(f"Unsupported arm: {arm}")
    if not 0 < train_limit <= 50000 or not 0 < development_limit <= 50000:
        raise ValueError("Limits must remain within the fixed shards")
    _identity(cache_root / "manifest.json", MANIFEST_SHA, "graph manifest")
    _identity(checkpoint, CHECKPOINT_SHA[arm], "selected checkpoint")
    _identity(predictions, PREDICTION_SHA[arm], "selected predictions")
    prediction_payload = torch.load(predictions, map_location="cpu", weights_only=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_arm = "gptrans" if arm == "baseline" else "gptrans_pair_update_norm"
    if state["arm"] != expected_arm:
        raise RuntimeError("Selected checkpoint arm mismatch")
    model = make_model(expected_arm)
    model.load_state_dict(state["model"], strict=True)
    model.to("cuda").eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output.mkdir(parents=True, exist_ok=True)

    reports = {}
    for role, shard_number, limit, start_idx in (
        ("train", 9, train_limit, 450000),
        ("development", 10, development_limit, 500000),
    ):
        path = cache_root / f"train/train_shard_{shard_number:04d}.pt"
        _identity(path, SHARD_SHA[role], f"{role} graph shard")
        dataset = PackedGraphs(path)
        if len(dataset) != 50000:
            raise RuntimeError(f"Wrong {role} shard row count")
        rows = {key: [] for key in ("source_idx", "target", "features", "base_prediction")}
        peak = 0
        loader = DataLoader(Subset(dataset, range(limit)), batch_size=batch_size, shuffle=False, num_workers=0)
        role_start = time.perf_counter()
        for batch in loader:
            batch = batch.to("cuda")
            features, normalized = _features(model, batch)
            prediction = normalized * state["std"] + state["mean"]
            rows["source_idx"].append(batch.source_idx.view(-1).cpu().long())
            rows["target"].append(batch.y.view(-1).cpu().float())
            rows["features"].append(features.cpu().float())
            rows["base_prediction"].append(prediction.cpu().float())
            peak = max(peak, torch.cuda.max_memory_reserved())
        payload = {key: torch.cat(values) for key, values in rows.items()}
        if not torch.equal(payload["source_idx"], torch.arange(start_idx, start_idx + limit)):
            raise RuntimeError(f"{role} source order changed")
        if not all(torch.isfinite(payload[key]).all() for key in ("target", "features", "base_prediction")):
            raise RuntimeError(f"{role} contains nonfinite feature or prediction")
        replay_difference = None
        if role == "development":
            frozen = prediction_payload
            if not torch.equal(payload["source_idx"], frozen["source_idx"][:limit]):
                raise RuntimeError("Selected prediction source indices differ")
            if not torch.equal(payload["target"], frozen["target"][:limit]):
                raise RuntimeError("Selected prediction target differs")
            replay_difference = float((payload["base_prediction"] - frozen["prediction"][:limit]).abs().max())
            if replay_difference > 2e-5:
                raise RuntimeError(f"Selected prediction replay differs: {replay_difference}")
        out_path = output / f"{role}_features.pt"
        atomic_torch_save(out_path, payload)
        reports[role] = {
            "rows": limit,
            "source_idx_start": start_idx,
            "source_idx_stop": start_idx + limit,
            "features_sha256": sha256_file(out_path),
            "feature_dimensions": list(payload["features"].shape),
            "base_mae_eV": float((payload["base_prediction"] - payload["target"]).abs().mean()),
            "selected_prediction_max_abs_replay_delta_eV": replay_difference,
            "wall_seconds": time.perf_counter() - role_start,
            "peak_cuda_reserved_bytes": peak,
        }
        del dataset, loader, payload, rows
    result = {
        "format": "molgap-gptrans-500k-frozen-features-v1",
        "arm": arm,
        "scope": "500k_trained_encoder_frozen_train_role_head_probe",
        "graph_manifest_sha256": MANIFEST_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA[arm],
        "selected_predictions_sha256": PREDICTION_SHA[arm],
        "batch_size": batch_size,
        "device": torch.cuda.get_device_name(),
        "roles": reports,
        "wall_seconds": time.perf_counter() - start,
        "protected_roles_read": False,
    }
    atomic_json(output / "extraction.json", result)
    return result


def _new_head(dim: int, width: int) -> nn.Module:
    head = nn.Sequential(nn.Linear(dim, width), nn.GELU(), nn.Linear(width, 1))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


def _paired_gain(target: torch.Tensor, reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    gains = ((reference - target).abs() - (candidate - target).abs()).double().numpy()
    rng = np.random.default_rng(20260925)
    draws = np.empty(2000, dtype=np.float64)
    for start in range(0, len(draws), 100):
        indices = rng.integers(0, len(gains), size=(100, len(gains)))
        draws[start:start + 100] = gains[indices].mean(axis=1)
    return {
        "reference_minus_candidate_mae_eV": float(gains.mean()),
        "row_bootstrap_95pct_eV": np.quantile(draws, [0.025, 0.975]).tolist(),
        "row_win_fraction": float(np.mean(gains > 0)),
    }


def fit(*, baseline_dir: Path, pair_dir: Path, output: Path, epochs: int = 35) -> dict:
    start = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("Head fitting requires local CUDA")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(42)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    dev_outputs = {}
    for arm, folder in (("baseline", baseline_dir), ("pair_norm", pair_dir)):
        manifest = json.loads((folder / "extraction.json").read_text(encoding="utf-8"))
        roles = {}
        for role in ("train", "development"):
            path = folder / f"{role}_features.pt"
            if sha256_file(path) != manifest["roles"][role]["features_sha256"]:
                raise RuntimeError(f"Frozen feature hash changed: {arm}/{role}")
            roles[role] = torch.load(path, map_location="cpu", weights_only=True)
        train, dev = roles["train"], roles["development"]
        if train["features"].shape[0] != 50000 or dev["features"].shape[0] != 10000:
            raise RuntimeError("Head probe requires frozen 50K/10K extraction")
        fit_end = 40000
        output_predictions = {
            "source_idx": dev["source_idx"],
            "target": dev["target"],
            "original": dev["base_prediction"],
        }
        result = {
            "original_selected_checkpoint_mae_eV": float((dev["base_prediction"]-dev["target"]).abs().mean()),
            "heads": {},
        }
        delta = (train["target"] - train["base_prediction"]).float()
        bias = delta[:fit_end].mean()
        bias_dev = (dev["base_prediction"] + bias - dev["target"]).abs().mean()
        result["heads"]["bias_only"] = {
            "development_mae_eV": float(bias_dev),
            "train_holdout_mae_eV": float((train["base_prediction"][fit_end:] + bias - train["target"][fit_end:]).abs().mean()),
        }
        output_predictions["bias_only"] = dev["base_prediction"] + bias
        configurations = (
            ("virtual_256", 288, 256),
            ("virtual_512", 288, 512),
            ("virtual_atom_mean_256", 544, 256),
        )
        for name, dims, width in configurations:
            torch.manual_seed(42)
            features = train["features"][:, :dims]
            mu, sigma = features[:fit_end].mean(0), features[:fit_end].std(0).clamp_min(1e-4)
            fit_x = ((features[:fit_end] - mu) / sigma).to("cuda")
            hold_x = ((features[fit_end:] - mu) / sigma).to("cuda")
            fit_y = delta[:fit_end].to("cuda")
            hold_y = train["target"][fit_end:].to("cuda")
            hold_base = train["base_prediction"][fit_end:].to("cuda")
            head = _new_head(dims, width).to("cuda")
            optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
            best_epoch = -1
            best_error = float("inf")
            best_state = None
            for epoch in range(epochs):
                head.train()
                order = torch.randperm(fit_end, generator=torch.Generator().manual_seed(42 + epoch))
                for indices in order.split(1024):
                    optimizer.zero_grad(set_to_none=True)
                    correction = head(fit_x[indices.to("cuda")]).view(-1)
                    loss = (correction - fit_y[indices.to("cuda")]).abs().mean()
                    loss.backward()
                    optimizer.step()
                head.eval()
                with torch.inference_mode():
                    hold_error = float((hold_base + head(hold_x).view(-1) - hold_y).abs().mean())
                if hold_error < best_error:
                    best_epoch, best_error = epoch, hold_error
                    best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
            if best_state is None:
                raise RuntimeError("No finite selected head")
            head.load_state_dict(best_state)
            head.eval()
            dev_x = ((dev["features"][:, :dims] - mu) / sigma).to("cuda")
            with torch.inference_mode():
                dev_correction = head(dev_x).view(-1).cpu()
            dev_prediction = dev["base_prediction"] + dev_correction
            if not torch.isfinite(dev_prediction).all():
                raise RuntimeError("Nonfinite head prediction")
            state_path = output / f"{arm}_{name}.pt"
            atomic_torch_save(state_path, {"state": best_state, "mean": mu, "std": sigma, "dimension": dims, "width": width})
            result["heads"][name] = {
                "development_mae_eV": float((dev_prediction - dev["target"]).abs().mean()),
                "train_holdout_mae_eV": best_error,
                "selected_epoch": best_epoch,
                "parameters": sum(p.numel() for p in head.parameters()),
                "head_sha256": sha256_file(state_path),
            }
            output_predictions[name] = dev_prediction
            del fit_x, hold_x, fit_y, hold_y, hold_base, head, optimizer, dev_x
        predictions_path = output / f"{arm}_development_predictions.pt"
        atomic_torch_save(predictions_path, output_predictions)
        result["development_predictions_sha256"] = sha256_file(predictions_path)
        results[arm] = result
        dev_outputs[arm] = output_predictions
    baseline, candidate = dev_outputs["baseline"], dev_outputs["pair_norm"]
    if not torch.equal(baseline["source_idx"], candidate["source_idx"]) or not torch.equal(baseline["target"], candidate["target"]):
        raise RuntimeError("Frozen head development roles differ")
    paired = {}
    for arm, outputs in dev_outputs.items():
        paired[arm] = {
            "atom_mean_over_wide_virtual": _paired_gain(
                outputs["target"], outputs["virtual_512"], outputs["virtual_atom_mean_256"]
            ),
            "wide_virtual_over_narrow_virtual": _paired_gain(
                outputs["target"], outputs["virtual_256"], outputs["virtual_512"]
            ),
        }
    paired["pair_norm_vs_baseline"] = {
        variant: _paired_gain(baseline["target"], baseline[variant], candidate[variant])
        for variant in ("original", "virtual_256", "virtual_512", "virtual_atom_mean_256")
    }
    return {
        "format": "molgap-gptrans-500k-frozen-readout-head-probe-v1",
        "scope": "post_hoc_internal_development_subset_not_promotion",
        "fit_rows": 40000,
        "train_holdout_rows": 10000,
        "development_rows": 10000,
        "seed": 42,
        "max_epochs": epochs,
        "results": results,
        "paired_row_diagnostics": paired,
        "uncertainty_scope": "reused development rows only; not training-seed or model-selection uncertainty",
        "wall_seconds": time.perf_counter() - start,
        "protected_roles_read": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    extraction = commands.add_parser("extract")
    extraction.add_argument("--arm", choices=CHECKPOINT_SHA, required=True)
    extraction.add_argument("--cache-root", type=Path, required=True)
    extraction.add_argument("--checkpoint", type=Path, required=True)
    extraction.add_argument("--predictions", type=Path, required=True)
    extraction.add_argument("--output", type=Path, required=True)
    extraction.add_argument("--train-limit", type=int, default=50000)
    extraction.add_argument("--development-limit", type=int, default=10000)
    extraction.add_argument("--batch-size", type=int, default=64)
    fitting = commands.add_parser("fit")
    fitting.add_argument("--baseline-dir", type=Path, required=True)
    fitting.add_argument("--pair-dir", type=Path, required=True)
    fitting.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        result = extract(
            arm=args.arm, cache_root=args.cache_root, checkpoint=args.checkpoint,
            predictions=args.predictions, output=args.output,
            train_limit=args.train_limit, development_limit=args.development_limit,
            batch_size=args.batch_size,
        )
        print(json.dumps({"arm": result["arm"], "roles": result["roles"], "wall_seconds": result["wall_seconds"]}))
    else:
        result = fit(baseline_dir=args.baseline_dir, pair_dir=args.pair_dir, output=args.output)
        atomic_json(args.output / "head_probe.json", result)
        print(json.dumps(result["results"]))
