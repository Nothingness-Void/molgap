"""QM9 inference-time conformer-averaging scaling curve.

Question
--------
The QM9 screen measured single-conformer ETKDG and a two-conformer prediction
average, then stopped at two views. On the same fusion system, replacing ETKDG
geometry with QM9's own DFT geometry is worth about -0.024 eV average MAE, an
order of magnitude more than any architecture change screened. So the open
question is whether conformer averaging keeps paying past K=2.

This script answers it with one already-trained SchNet-ETKDG encoder over K
independent ETKDG views of the test split, so no encoder training is needed.

Views
-----
Each view is a distinct ETKDG seed. `build_etkdg_cache` derives a per-molecule
seed as `(seed * 1_000_003 + source_idx)`, so views are independent conformers
of the same molecules. Views are consumed from a prebuilt cache directory; the
Kaggle CPU run that produced them is the expensive part and is not repeated.

Metric
------
Equal-weight mean of per-view predictions, then MAE against the B3LYP eV
labels, on the exact all-view row intersection so every K uses identical
molecules. Deltas versus K=1 carry a paired bootstrap CI.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from molgap.qm9_screen import (
    ENCODER_CONFIGS,
    build_etkdg_cache,
    evaluate_encoder,
    fixed_split,
    load_qm9_records,
    make_encoder,
    target_stats,
)

DEFAULT_VIEW_SEEDS = (42, 43, 44, 45, 46, 47)
DEFAULT_K_VALUES = (1, 2, 3, 4, 6)
DEFAULT_CHECKPOINT = Path(
    "models/experiments/qm9_architecture_screen/n100000_10000_10000"
    "/schnet_etkdg/seed42/model.pt"
)
DEFAULT_CACHE = Path("results/kaggle/qm9_conformer_scaling/_qm9_cache")
DEFAULT_RESULTS = Path(
    "experiments/qm9_architecture/conformer_scaling"
)
# Local screen reference (older cache): compare curve shape, not absolute values.
REFERENCE = {
    "single_conformer_avg_mae": 0.09440,
    "two_conformer_average_avg_mae": 0.09146,
    "etkdg_to_dft_fusion_gain": -0.02428,
}


def _mae_per_target(pred: np.ndarray, true: np.ndarray) -> dict[str, float]:
    err = np.abs(pred - true)
    return {
        "HOMO": float(err[:, 0].mean()),
        "LUMO": float(err[:, 1].mean()),
        "Gap": float(err[:, 2].mean()),
        "average": float(err.mean()),
    }


def _paired_bootstrap(
    base_err: np.ndarray,
    other_err: np.ndarray,
    rng: np.random.Generator,
    draws: int,
) -> tuple[float, float, float]:
    """Paired CI on mean(other) - mean(base); negative means `other` is better."""
    delta = other_err - base_err
    rows = len(delta)
    samples = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        samples[i] = delta[rng.integers(0, rows, rows)].mean()
    return (
        float(delta.mean()),
        float(np.percentile(samples, 2.5)),
        float(np.percentile(samples, 97.5)),
    )


def _view_payload(
    *,
    records: list[dict],
    indices: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
    cache_dir: Path,
    seed: int,
    model,
    kind: str,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict]:
    graphs, report = build_etkdg_cache(
        records, indices, mean, std, cache_dir=cache_dir, seed=seed
    )
    ordered = [graphs[int(i)] for i in indices if int(i) in graphs]
    if not ordered:
        raise RuntimeError(f"View seed {seed} produced no usable graphs")
    payload = evaluate_encoder(kind, model, ordered, batch_size, device, mean, std)
    return (
        {
            "predictions": payload["predictions"].numpy(),
            "targets": payload["targets"].numpy(),
            "source_idx": payload["source_idx"].numpy(),
        },
        {
            "requested": report.get("requested"),
            "succeeded": report.get("succeeded"),
            "failed": report.get("failed"),
            "elapsed_s": round(float(report.get("elapsed_s", 0.0)), 1),
            "rows_scored": len(ordered),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--validation-size", type=int, default=10_000)
    parser.add_argument("--test-size", type=int, default=10_000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--view-seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_VIEW_SEEDS),
        help="ETKDG seeds, in the order views are added to the average",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    started = time.perf_counter()
    records = load_qm9_records(args.cache_dir)
    split = fixed_split(
        len(records),
        args.train_size,
        args.validation_size,
        args.test_size,
        args.split_seed,
    )
    mean, std = target_stats(records, split.train)
    print(
        f"records={len(records)} test_rows={len(split.test)} "
        f"fingerprint={split.fingerprint}",
        flush=True,
    )

    model, kind = make_encoder("schnet")
    model.load_state_dict(
        torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    )
    model = model.to(device).eval()
    batch_size = int(ENCODER_CONFIGS["schnet"]["batch_size"])

    views: dict[int, dict[str, np.ndarray]] = {}
    build_reports: dict[str, dict] = {}
    for seed in args.view_seeds:
        t0 = time.perf_counter()
        views[seed], build_reports[str(seed)] = _view_payload(
            records=records,
            indices=split.test,
            mean=mean,
            std=std,
            cache_dir=args.cache_dir,
            seed=seed,
            model=model,
            kind=kind,
            batch_size=batch_size,
            device=device,
        )
        single = _mae_per_target(views[seed]["predictions"], views[seed]["targets"])
        print(
            f"view seed={seed} rows={build_reports[str(seed)]['rows_scored']} "
            f"avgMAE={single['average']:.5f} GapMAE={single['Gap']:.5f} "
            f"{time.perf_counter() - t0:.0f}s",
            flush=True,
        )

    # Exact intersection so every K scores identical molecules.
    common = set(views[args.view_seeds[0]]["source_idx"].tolist())
    for seed in args.view_seeds[1:]:
        common &= set(views[seed]["source_idx"].tolist())
    common_sorted = sorted(common)
    print(f"all-view intersection rows={len(common_sorted)}", flush=True)

    aligned: dict[int, np.ndarray] = {}
    reference_targets: np.ndarray | None = None
    for seed in args.view_seeds:
        position = {
            int(value): i for i, value in enumerate(views[seed]["source_idx"].tolist())
        }
        index = np.array([position[value] for value in common_sorted])
        aligned[seed] = views[seed]["predictions"][index]
        targets = views[seed]["targets"][index]
        if reference_targets is None:
            reference_targets = targets
        elif not np.allclose(reference_targets, targets):
            raise RuntimeError(f"Target mismatch on intersection for seed {seed}")
    assert reference_targets is not None

    rng = np.random.default_rng(args.split_seed)
    base_err: np.ndarray | None = None
    curve: list[dict[str, Any]] = []
    k_values = [k for k in DEFAULT_K_VALUES if k <= len(args.view_seeds)]
    for k in k_values:
        seeds = list(args.view_seeds[:k])
        averaged = np.stack([aligned[s] for s in seeds]).mean(axis=0)
        metrics = _mae_per_target(averaged, reference_targets)
        row_err = np.abs(averaged - reference_targets).mean(axis=1)
        entry: dict[str, Any] = {"k": k, "seeds": seeds, **metrics}
        if base_err is None:
            base_err = row_err
        else:
            delta, low, high = _paired_bootstrap(
                base_err, row_err, rng, args.bootstrap
            )
            entry["delta_avg_vs_k1"] = round(delta, 6)
            entry["delta_ci95"] = [round(low, 6), round(high, 6)]
            entry["significant"] = bool(high < 0.0)
        curve.append(entry)
        tail = (
            f" delta={entry['delta_avg_vs_k1']:+.5f} "
            f"CI95=[{entry['delta_ci95'][0]:+.5f},{entry['delta_ci95'][1]:+.5f}]"
            f" sig={entry['significant']}"
            if "delta_avg_vs_k1" in entry
            else " (baseline)"
        )
        print(
            f"K={k} avgMAE={metrics['average']:.5f} "
            f"GapMAE={metrics['Gap']:.5f}{tail}",
            flush=True,
        )

    singles = [
        _mae_per_target(aligned[s], reference_targets)["average"]
        for s in args.view_seeds
    ]
    payload = {
        "experiment": "qm9_inference_conformer_scaling",
        "candidate": "schnet",
        "geometry": "etkdg",
        "checkpoint": str(args.checkpoint),
        "cache_dir": str(args.cache_dir),
        "split": {
            "train": args.train_size,
            "validation": args.validation_size,
            "test": args.test_size,
            "split_seed": args.split_seed,
            "fingerprint": split.fingerprint,
        },
        "view_seeds": list(args.view_seeds),
        "intersection_rows": len(common_sorted),
        "build_reports": build_reports,
        "single_view_avg_mae": {
            str(s): round(float(v), 6) for s, v in zip(args.view_seeds, singles)
        },
        "single_view_mean": round(float(np.mean(singles)), 6),
        "single_view_std": round(float(np.std(singles, ddof=1)), 6),
        "scaling_curve": curve,
        "bootstrap_draws": args.bootstrap,
        "reference_local_screen": REFERENCE,
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scaling.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        source_idx=np.array(common_sorted),
        targets=reference_targets,
        **{f"view_{s}": aligned[s] for s in args.view_seeds},
    )
    print(
        f"complete total={payload['total_seconds']}s -> {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
