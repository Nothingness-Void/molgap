"""Kaggle entrypoint: QM9 2D-encoder seed repeats.

Question
--------
Every uncertainty band in the QM9 screen comes from frozen-head seeds; the README
states six times that encoder seeds were never repeated. Two decisions rest on
differences smaller than that unmeasured noise:

- GPS9-192 (0.08634 avg MAE) vs GPS11-160 (0.08642) standalone differ by 0.00008.
- As a second expert after GPS9, GPS11 beats GPS7 by 0.00115, against a
  head-seed spread of 0.0004-0.0005.

The Route B 1M architecture uses GPS11-160 as the identity path, so if that
0.00115 is encoder-initialisation noise the second expert was chosen on a
coin flip. This kernel trains GPS7 / GPS9 / GPS11-160 at seeds 43 and 44 under
the exact protocol of the existing seed-42 runs and reports the standalone
encoder-seed spread.

Scope
-----
Topology geometry only, so no ETKDG cache is needed and the run stays cheap
(~25 s/epoch, 30 epochs per encoder locally). This measures encoder-seed noise
for the pure-2D encoders. It does NOT rerun the fusion comparison: that needs
SchNet embeddings on the ETKDG cache and is a separate, heavier step justified
only if the spread here turns out to be large.

Everything else is held fixed: 100000/10000/10000 split, split seed 42, 30
epochs, and the learning rate / weight decay / patience defaults inside
`train_encoder`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

CANDIDATES = ("gps7", "gps9", "gps11_160")
SEEDS = (43, 44)
SEED42_REFERENCE = {
    # From results/phase8/experiments/qm9_architecture_screen (seed 42, 30 epochs).
    "gps7": {"average": 0.09160, "Gap": 0.10947},
    "gps9": {"average": 0.08634, "Gap": 0.10347},
    "gps11_160": {"average": 0.08642, "Gap": 0.10308},
}
TRAIN_SIZE = 100_000
VALIDATION_SIZE = 10_000
TEST_SIZE = 10_000
EPOCHS = 30
SPLIT_SEED = 42
PASCAL_COMPAT_RESTART = "MOLGAP_TORCH_COMPAT_RESTART"
OUT = Path("/kaggle/working/qm9_encoder_seeds")


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
    if probe_torch.cuda.get_device_capability(0) != (6, 0):
        return
    if "sm_60" in set(probe_torch.cuda.get_arch_list()):
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
    # qm9_screen imports egnn (torch_scatter); SchNet paths need torch_cluster.
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


def main() -> None:
    ensure_pascal_compatible_torch()
    install_dependencies()
    stage_package()

    import numpy as np

    from molgap.qm9_screen import train_encoder

    OUT.mkdir(parents=True, exist_ok=True)
    cache = Path("/kaggle/working/_qm9_cache")
    started = time.perf_counter()

    runs: list[dict] = []
    for candidate in CANDIDATES:
        for seed in SEEDS:
            t0 = time.perf_counter()
            result = train_encoder(
                candidate=candidate,
                geometry="topology",
                train_size=TRAIN_SIZE,
                validation_size=VALIDATION_SIZE,
                test_size=TEST_SIZE,
                epochs=EPOCHS,
                seed=seed,
                split_seed=SPLIT_SEED,
                cache_dir=cache,
                results_dir=OUT / "results",
                models_dir=OUT / "models",
            )
            test = result["metrics"]["test"]
            row = {
                "candidate": candidate,
                "seed": seed,
                "n_params": result["n_params"],
                "best_epoch": result["best_epoch"],
                "best_validation_average_mae_eV": result["best_validation_average_mae_eV"],
                "test_average_mae": test["average"]["mae"],
                "test_gap_mae": test["Gap"]["mae"],
                "test_homo_mae": test["HOMO"]["mae"],
                "test_lumo_mae": test["LUMO"]["mae"],
                "seconds": round(time.perf_counter() - t0, 1),
            }
            runs.append(row)
            print(
                f"{candidate} seed={seed} avgMAE={row['test_average_mae']:.5f} "
                f"GapMAE={row['test_gap_mae']:.5f} best_ep={row['best_epoch']} "
                f"{row['seconds']:.0f}s",
                flush=True,
            )
            (OUT / "runs.json").write_text(json.dumps(runs, indent=2), encoding="utf-8")

    # Per-candidate spread across seed 42 (reference) + the new seeds.
    summary = []
    for candidate in CANDIDATES:
        values = [SEED42_REFERENCE[candidate]["average"]]
        gaps = [SEED42_REFERENCE[candidate]["Gap"]]
        for row in runs:
            if row["candidate"] == candidate:
                values.append(row["test_average_mae"])
                gaps.append(row["test_gap_mae"])
        summary.append({
            "candidate": candidate,
            "seeds": [42, *SEEDS],
            "avg_mae_values": [round(float(v), 6) for v in values],
            "avg_mae_mean": round(float(np.mean(values)), 6),
            "avg_mae_std": round(float(np.std(values, ddof=1)), 6),
            "avg_mae_range": round(float(max(values) - min(values)), 6),
            "gap_mae_mean": round(float(np.mean(gaps)), 6),
            "gap_mae_std": round(float(np.std(gaps, ddof=1)), 6),
        })
        print(
            f"SUMMARY {candidate}: avg={summary[-1]['avg_mae_mean']:.5f} "
            f"+/-{summary[-1]['avg_mae_std']:.5f} range={summary[-1]['avg_mae_range']:.5f}",
            flush=True,
        )

    by_name = {row["candidate"]: row for row in summary}
    # The decisions under test, expressed as differences of three-seed means.
    contrasts = {
        "gps9_minus_gps11": round(
            by_name["gps9"]["avg_mae_mean"] - by_name["gps11_160"]["avg_mae_mean"], 6
        ),
        "gps7_minus_gps11": round(
            by_name["gps7"]["avg_mae_mean"] - by_name["gps11_160"]["avg_mae_mean"], 6
        ),
        "max_encoder_seed_std": round(
            max(row["avg_mae_std"] for row in summary), 6
        ),
        "head_seed_std_reference": 0.0005,
        "second_expert_gap_under_test": 0.00115,
    }
    payload = {
        "experiment": "qm9_encoder_seed_repeats",
        "geometry": "topology",
        "epochs": EPOCHS,
        "split": {
            "train": TRAIN_SIZE, "validation": VALIDATION_SIZE,
            "test": TEST_SIZE, "split_seed": SPLIT_SEED,
        },
        "seed42_reference": SEED42_REFERENCE,
        "runs": runs,
        "summary": summary,
        "contrasts": contrasts,
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    (OUT / "encoder_seeds.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"complete total={payload['total_seconds']}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
