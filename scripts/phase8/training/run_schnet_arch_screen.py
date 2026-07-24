"""Screen SchNet compute-shape combinations on the fixed 50K subset."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from molgap.utils import ensure_dirs


@dataclass(frozen=True)
class SchNetSpec:
    hidden: int
    filters: int
    interactions: int

    @property
    def name(self) -> str:
        return f"h{self.hidden}_f{self.filters}_i{self.interactions}"

    @classmethod
    def parse(cls, value: str) -> "SchNetSpec":
        try:
            hidden, filters, interactions = (int(part) for part in value.split(":"))
        except ValueError as error:
            raise argparse.ArgumentTypeError("spec must be HIDDEN:FILTERS:INTERACTIONS") from error
        if min(hidden, filters, interactions) <= 0:
            raise argparse.ArgumentTypeError("all spec values must be positive")
        return cls(hidden, filters, interactions)


def run_candidate(args: argparse.Namespace, spec: SchNetSpec) -> dict:
    arm = args.out_dir / spec.name
    ensure_dirs(arm)
    metrics_path = arm / "metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("complete", True):
            print(f"Reuse completed candidate: {spec.name}", flush=True)
            return metrics

    command = [
        sys.executable,
        str(Path(__file__).with_name("train_encoder.py")),
        "--kind", "schnet",
        "--graphs", str(args.graphs),
        "--hidden-channels", str(spec.hidden),
        "--num-filters", str(spec.filters),
        "--num-interactions", str(spec.interactions),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--batch-size", str(args.batch_size),
        "--eval-batch-size", str(args.batch_size),
        "--model-out", str(arm / "model.pt"),
        "--metrics-out", str(metrics_path),
        "--embeddings-out", str(arm / "embeddings.pt"),
        "--checkpoint-out", str(arm / "checkpoint.pt"),
        "--checkpoint-every", "1",
        "--seed", str(args.seed),
        "--split-seed", str(args.split_seed),
        "--no-embeddings",
    ]
    checkpoint_path = arm / "checkpoint.pt"
    if checkpoint_path.is_file():
        command.extend(["--resume-from", str(checkpoint_path)])
    print(f"Training candidate: {spec.name}", flush=True)
    subprocess.run(command, check=True)
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def compact(metrics: dict) -> dict:
    test = metrics["test_metrics"]
    return {
        "model_params": metrics["model_params"],
        "n_params": metrics["n_params"],
        "training_time_s": metrics["training_time_s"],
        "best_val_mae": metrics["best_val_mae"],
        "test_average_mae": test["average"]["mae"],
        "test_gap_mae": test["Gap"]["mae"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Screen SchNet compute shapes on fixed graphs")
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True,
                        help="dimension-screen root containing hidden_192 and hidden_160")
    parser.add_argument("--spec", type=SchNetSpec.parse, action="append", dest="specs")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    args.specs = args.specs or [
        SchNetSpec(160, 160, 6),
        SchNetSpec(160, 160, 5),
        SchNetSpec(176, 160, 6),
    ]
    ensure_dirs(args.out_dir)

    arms = {
        "h192_f192_i6": compact(json.loads(
            (args.reference_root / "hidden_192" / "metrics.json").read_text(encoding="utf-8")
        )),
        "h160_f192_i6": compact(json.loads(
            (args.reference_root / "hidden_160" / "metrics.json").read_text(encoding="utf-8")
        )),
    }
    for spec in args.specs:
        arms[spec.name] = compact(run_candidate(args, spec))

    baseline = arms["h192_f192_i6"]
    comparisons = {}
    for name, arm in arms.items():
        if name == "h192_f192_i6":
            continue
        delta_mae = arm["test_average_mae"] - baseline["test_average_mae"]
        comparisons[name] = {
            "test_average_mae_delta": delta_mae,
            "test_gap_mae_delta": arm["test_gap_mae"] - baseline["test_gap_mae"],
            "parameter_ratio": arm["n_params"] / baseline["n_params"],
            "training_time_ratio": arm["training_time_s"] / baseline["training_time_s"],
            "encoder_screen_pass": delta_mae <= 0.005,
        }
    summary = {"arms": arms, "comparisons_to_h192_f192_i6": comparisons}
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
