"""Kaggle entrypoint: QM9 inference-time conformer-averaging scaling curve.

Question
--------
The QM9 screen measured single-conformer ETKDG (0.09440 eV average MAE) and a
two-conformer prediction average (0.09146 eV), a -0.00294 eV gain, then stopped
at two views. Meanwhile ETKDG->DFT geometry is worth -0.02428 eV on the same
fusion system, an order of magnitude more than any architecture change screened.

So the open question is whether conformer averaging keeps paying past K=2, and
how much of the geometry gap it can close. This kernel measures the curve for
K = 1..6 using ONE already-trained SchNet-ETKDG encoder, so no encoder training
is required and the only cost is conformer construction plus forward passes.

Protocol
--------
- Split: the frozen 100000/10000/10000 split, split seed 42, identical to the
  local screen, so the encoder never saw these rows.
- Rows: the official test role only. Equal averaging needs no validation-selected
  weight, so validation conformers are not built.
- Views: independent ETKDG seeds. Each seed produces a fully independent
  conformer per molecule via random_seed = (seed * 1_000_003 + source_idx).
- All views are built with the SAME code in ONE process, so the curve is
  internally consistent. Absolute values are NOT compared against the older
  local cache; see the reproducibility note below.
- Metric: equal-weight average of per-view predictions, then MAE against B3LYP
  eV labels. Paired bootstrap CIs versus K=1 on the exact all-view intersection.

Reproducibility note
--------------------
Subset-built conformers do not byte-match the older full-120K local cache for
the same nominal seed, although generation IS deterministic across processes for
a fixed index list. This kernel therefore rebuilds every view it uses instead of
mixing old and new caches, and reports its own K=1 baseline.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

VIEW_SEEDS = (42, 43, 44, 45, 46, 47)
K_VALUES = (1, 2, 3, 4, 6)
TRAIN_SIZE = 100_000
VALIDATION_SIZE = 10_000
TEST_SIZE = 10_000
SPLIT_SEED = 42
BOOTSTRAP = 2_000
PASCAL_COMPAT_RESTART = "MOLGAP_TORCH_COMPAT_RESTART"
OUT = Path("/kaggle/working/qm9_conformer_scaling")


def find_one(name: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name}, found {matches}")
    return matches[0]


def ensure_pascal_compatible_torch() -> None:
    """Replace Kaggle's stock torch when it cannot execute on a P100."""
    import os

    import torch as probe_torch

    if not probe_torch.cuda.is_available():
        return
    capability = probe_torch.cuda.get_device_capability(0)
    if capability != (6, 0) or "sm_60" in set(probe_torch.cuda.get_arch_list()):
        return
    if os.environ.get(PASCAL_COMPAT_RESTART) == "1":
        raise RuntimeError("cu126 install still lacks sm_60; refusing restart loop")
    print("P100 assigned and stock torch lacks sm_60; installing cu126 once.", flush=True)
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir",
        "--no-deps", "--force-reinstall", "torch==2.7.1",
        "nvidia-cusparselt-cu12==0.6.3",
        "--index-url", "https://download.pytorch.org/whl/cu126",
    ])
    os.environ[PASCAL_COMPAT_RESTART] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def install_dependencies() -> None:
    import torch

    version = torch.__version__.split("+")[0]
    cuda = (torch.version.cuda or "").replace(".", "")
    wheel_version = "2.7.0" if version.startswith("2.7.") else version
    index = f"https://data.pyg.org/whl/torch-{wheel_version}+cu{cuda}.html"
    # qm9_screen imports egnn, which needs torch_scatter; SchNet needs torch_cluster.
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "torch-geometric==2.6.1", "torch_cluster", "torch_scatter",
        "-f", index,
    ])
    try:
        import rdkit  # noqa: F401
    except ImportError:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "rdkit==2024.3.5",
        ])
    print(f"torch={torch.__version__} cuda={torch.version.cuda}", flush=True)


def stage_package() -> None:
    """Expose the uploaded modules as an importable `molgap` package."""
    root = Path("/kaggle/working/_molgap_runtime")
    package = root / "molgap"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    # Full import closure of qm9_screen, verified locally as a flat package.
    for name in (
        "qm9_screen.py", "qm9_payloads.py", "qm9_fusion.py", "graphs.py",
        "schnet.py", "gps.py", "gine.py", "egnn.py", "tensornet.py",
        "fusion.py", "constants.py", "utils.py",
    ):
        shutil.copy2(find_one(name), package)
    sys.path.insert(0, str(root))


def mae_per_target(pred, true):
    import numpy as np

    err = np.abs(pred - true)
    return {
        "HOMO": float(err[:, 0].mean()),
        "LUMO": float(err[:, 1].mean()),
        "Gap": float(err[:, 2].mean()),
        "average": float(err.mean()),
    }


def paired_bootstrap(base_err, other_err, rng, n=BOOTSTRAP):
    """Paired CI on mean(other) - mean(base); negative means `other` is better."""
    import numpy as np

    delta = other_err - base_err
    rows = len(delta)
    draws = np.empty(n, dtype=np.float64)
    for i in range(n):
        idx = rng.integers(0, rows, rows)
        draws[i] = delta[idx].mean()
    return float(delta.mean()), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ensure_pascal_compatible_torch()
    install_dependencies()
    stage_package()

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

    OUT.mkdir(parents=True, exist_ok=True)
    cache = Path("/kaggle/working/_qm9_cache")
    cache.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    started = time.perf_counter()
    records = load_qm9_records(cache)
    split = fixed_split(len(records), TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE, SPLIT_SEED)
    mean, std = target_stats(records, split.train)
    print(
        f"records={len(records)} test_rows={len(split.test)} "
        f"fingerprint={split.fingerprint} setup={time.perf_counter()-started:.0f}s",
        flush=True,
    )

    model, kind = make_encoder("schnet")
    state = torch.load(find_one("schnet_etkdg_seed42_model.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()
    batch_size = int(ENCODER_CONFIGS["schnet"]["batch_size"])

    views: dict[int, dict] = {}
    build_reports: dict[int, dict] = {}
    for seed in VIEW_SEEDS:
        t0 = time.perf_counter()
        graphs, report = build_etkdg_cache(
            records, split.test, mean, std, cache_dir=cache, seed=seed
        )
        ordered = [graphs[int(i)] for i in split.test if int(i) in graphs]
        payload = evaluate_encoder(kind, model, ordered, batch_size, device, mean, std)
        views[seed] = {
            "predictions": payload["predictions"].numpy(),
            "targets": payload["targets"].numpy(),
            "source_idx": payload["source_idx"].numpy(),
        }
        build_reports[seed] = {
            "requested": report.get("requested"),
            "succeeded": report.get("succeeded"),
            "failed": report.get("failed"),
            "elapsed_s": round(float(report.get("elapsed_s", 0.0)), 1),
        }
        single = mae_per_target(views[seed]["predictions"], views[seed]["targets"])
        print(
            f"view seed={seed} rows={len(ordered)} "
            f"avgMAE={single['average']:.5f} GapMAE={single['Gap']:.5f} "
            f"total={time.perf_counter()-t0:.0f}s",
            flush=True,
        )

    # Exact intersection so every K uses identical molecules.
    common = set(views[VIEW_SEEDS[0]]["source_idx"].tolist())
    for seed in VIEW_SEEDS[1:]:
        common &= set(views[seed]["source_idx"].tolist())
    common_sorted = sorted(common)
    print(f"all-view intersection rows={len(common_sorted)}", flush=True)

    aligned = {}
    reference_targets = None
    for seed in VIEW_SEEDS:
        pos = {int(v): i for i, v in enumerate(views[seed]["source_idx"].tolist())}
        index = np.array([pos[v] for v in common_sorted])
        aligned[seed] = views[seed]["predictions"][index]
        targets = views[seed]["targets"][index]
        if reference_targets is None:
            reference_targets = targets
        elif not np.allclose(reference_targets, targets):
            raise RuntimeError(f"Target mismatch on intersection for seed {seed}")

    rng = np.random.default_rng(SPLIT_SEED)
    base_err = None
    curve = []
    for k in K_VALUES:
        stack = np.stack([aligned[s] for s in VIEW_SEEDS[:k]])
        averaged = stack.mean(axis=0)
        metrics = mae_per_target(averaged, reference_targets)
        row_err = np.abs(averaged - reference_targets).mean(axis=1)
        entry = {"k": k, "seeds": list(VIEW_SEEDS[:k]), **metrics}
        if base_err is None:
            base_err = row_err
        else:
            delta, low, high = paired_bootstrap(base_err, row_err, rng)
            entry["delta_avg_vs_k1"] = round(delta, 6)
            entry["delta_ci95"] = [round(low, 6), round(high, 6)]
            entry["significant"] = bool(high < 0.0)
        curve.append(entry)
        print(
            f"K={k} avgMAE={metrics['average']:.5f} GapMAE={metrics['Gap']:.5f}"
            + (
                f" delta={entry['delta_avg_vs_k1']:+.5f} "
                f"CI95=[{entry['delta_ci95'][0]:+.5f},{entry['delta_ci95'][1]:+.5f}]"
                f" sig={entry['significant']}"
                if "delta_avg_vs_k1" in entry else " (baseline)"
            ),
            flush=True,
        )

    # Per-view spread isolates conformer noise from averaging benefit.
    singles = [mae_per_target(aligned[s], reference_targets)["average"] for s in VIEW_SEEDS]
    result = {
        "experiment": "qm9_inference_conformer_scaling",
        "candidate": "schnet",
        "geometry": "etkdg",
        "checkpoint": "schnet_etkdg_seed42 (local screen, trained on the same split)",
        "split": {
            "train": TRAIN_SIZE, "validation": VALIDATION_SIZE, "test": TEST_SIZE,
            "split_seed": SPLIT_SEED, "fingerprint": split.fingerprint,
        },
        "view_seeds": list(VIEW_SEEDS),
        "intersection_rows": len(common_sorted),
        "build_reports": build_reports,
        "single_view_avg_mae": {str(s): round(v, 6) for s, v in zip(VIEW_SEEDS, singles)},
        "single_view_mean": round(float(np.mean(singles)), 6),
        "single_view_std": round(float(np.std(singles, ddof=1)), 6),
        "scaling_curve": curve,
        "bootstrap_draws": BOOTSTRAP,
        "reference_local_screen": {
            "single_conformer_avg_mae": 0.09440,
            "two_conformer_average_avg_mae": 0.09146,
            "etkdg_to_dft_fusion_gain": -0.02428,
            "note": "Local screen numbers use an older cache; compare shape, not absolutes.",
        },
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    (OUT / "scaling.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    np.savez_compressed(
        OUT / "predictions.npz",
        source_idx=np.array(common_sorted),
        targets=reference_targets,
        **{f"view_{s}": aligned[s] for s in VIEW_SEEDS},
    )
    print(f"complete total={result['total_seconds']}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
