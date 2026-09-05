"""Single-DCU paired screens using the frozen PCQM scientific trainer."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import time
from pathlib import Path

BASELINE = "ogb_distance_angle_triangle_edge_state_graph_state9"
CANDIDATE = "ogb_distance_angle_vector_state_triangle_edge_state_graph_state9"
CANDIDATES = (BASELINE, CANDIDATE)
CACHE_SHA = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
CONTRACT = {
    "seed": 42, "batch_size": 48, "precision": "fp32", "max_epochs": 40,
    "patience": 8, "learning_rate": 1.6e-4, "weight_decay": 1e-6,
    "optimizer": "AdamW", "loss": "train_standardized_L1",
    "scheduler": "CosineAnnealingLR", "eta_min": 1e-6,
    "loader_workers": 0, "pin_memory": False, "target": "gap",
}


def configure(
    trainer,
    output: Path,
    source_commit: str,
    parameter_counts: dict,
    candidates=CANDIDATES,
    candidate=CANDIDATE,
):
    trainer.OUT = output
    trainer.SEED = 42
    trainer.EXPECTED_MODEL_SOURCE_COMMIT = source_commit
    trainer.CANDIDATES = candidates
    trainer.BASELINE = BASELINE
    trainer.EXPECTED_PARAMETER_COUNTS.update(parameter_counts)
    trainer.EXPECTED_GLOBAL_BLOCKS[candidate] = ()
    trainer.PARAMETER_BUDGET = 4_000_000
    trainer.SEARCH_BUDGET_S = 41_400
    trainer.PIN_MEMORY = False
    trainer.CANDIDATE_EXECUTION = "single_kunshan_dcu_sequential"
    for key, expected in (
        ("BATCH_SIZE", 48), ("LEARNING_RATE", 1.6e-4), ("WEIGHT_DECAY", 1e-6),
        ("MAX_EPOCHS", 40), ("PATIENCE", 8), ("LOADER_WORKERS", 0),
    ):
        if getattr(trainer, key) != expected:
            raise RuntimeError(f"Scientific trainer contract changed: {key}")
    if trainer.expected_input_cache_sha256() != CACHE_SHA:
        raise RuntimeError("Unexpected training cache role")


def export_csv(trainer, metrics: dict) -> None:
    import torch
    payload = torch.load(
        trainer.OUT / metrics["artifacts"]["validation_payload"],
        map_location="cpu", weights_only=False,
    )
    path = trainer.OUT / "results" / metrics["candidate"] / "validation.csv"
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["row_index", "target_eV", "prediction_eV"])
        for row, target, prediction in zip(
            payload["row_index"].reshape(-1).tolist(),
            payload["target_eV"].reshape(-1).tolist(),
            payload["prediction_eV"].reshape(-1).tolist(),
        ):
            if not math.isfinite(target) or not math.isfinite(prediction):
                raise RuntimeError("Nonfinite validation payload")
            writer.writerow([row, target, prediction])
    os.replace(temporary, path)
    metrics["artifacts"]["validation_csv"] = str(path.relative_to(trainer.OUT))
    metrics["artifacts"]["validation_csv_sha256"] = trainer.sha256_file(path)


def run(args) -> None:
    # Environment is set before importing the existing, configuration-bound runner.
    os.environ["MOLGAP_LOCAL_GLOBAL_RUN_MODE"] = "seed42_screen"
    os.environ["MOLGAP_LOCAL_GLOBAL_SEED"] = "42"
    from molgap import pcqm_local_global_runner as trainer
    from molgap.pcqm_gap_architecture import make_pcqm_gap_encoder
    import torch
    import torch_geometric

    if args.screen == "vector":
        from molgap.pcqm_vector_state import make_vector_state_encoder

        candidate = CANDIDATE
        candidates = CANDIDATES
        expected_counts = {BASELINE: 3_665_809, candidate: 3_696_209}
        format_name = "molgap-kunshan-vector-screen-v1"
        candidate_factory = make_vector_state_encoder
        baseline_delta = {"vector_state": "none"}
        candidate_delta = {
            "vector_state": "persistent_polar_order1_channels16",
            "vector_update_blocks": [2, 4, 6, 8],
            "relation": "directed_real_bond_displacement",
            "scalar_return": "norm_norm_dot_linear192_bias_free_zero_init",
        }
    else:
        from molgap.pcqm_moment_readout import (
            CANDIDATE_ID,
            MOMENT_PARAMETER_COUNT,
            make_moment_readout_encoder,
        )

        candidate = CANDIDATE_ID
        candidates = (BASELINE, candidate)
        expected_counts = {
            BASELINE: 3_665_809,
            candidate: MOMENT_PARAMETER_COUNT,
        }
        format_name = "molgap-kunshan-moment-readout-screen-v1"
        candidate_factory = make_moment_readout_encoder
        baseline_delta = {"moment_readout": "mean_only"}
        candidate_delta = {
            "moment_readout": (
                "mean_plus_nonlinear_projected_first_centered_second"
            ),
            "moment_channels": 32,
            "return": "linear64x192_bias_free_zero_init",
        }

    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    if (output / "completion.json").exists():
        raise RuntimeError("Completed paired screen must not run again")
    if len(args.source_commit) != 40 or any(c not in "0123456789abcdef" for c in args.source_commit):
        raise RuntimeError("A frozen Git source commit is required")
    preflight = json.loads(args.preflight.read_text(encoding="utf-8"))
    if preflight.get("accepted") is not True:
        raise RuntimeError("Remote model/symmetry preflight failed")
    counts = preflight["parameter_counts"]
    if counts != expected_counts:
        raise RuntimeError("Unexpected model parameter counts")
    configure(
        trainer,
        output,
        args.source_commit,
        counts,
        candidates=candidates,
        candidate=candidate,
    )
    torch.set_num_threads(1)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one visible DCU is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    trainer.atomic_json(output / "preflight.json", preflight)
    manifest = {
        "format": format_name,
        "source_commit": args.source_commit, "candidates": list(candidates),
        "contract": CONTRACT, "geometry_cache_aggregate_sha256": CACHE_SHA,
        "preflight_sha256": trainer.sha256_file(output / "preflight.json"),
        "platform": "SCNet Kunshan", "job_id": os.environ.get("SLURM_JOB_ID"),
        "train_graphs": 100_000, "validation_graphs": 10_000,
        "device": torch.cuda.get_device_name(0), "device_count": 1,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "pyg": torch_geometric.__version__, "hip": torch.version.hip},
        "official_validation_role_read": False, "test_dev_role_read": False,
    }
    identity = output / "run_manifest.json"
    if identity.exists():
        prior = json.loads(identity.read_text(encoding="utf-8"))
        for key in ("source_commit", "candidates", "contract", "geometry_cache_aggregate_sha256", "runtime", "device"):
            if prior.get(key) != manifest[key]:
                raise RuntimeError(f"Resume manifest mismatch: {key}")
        if not args.resume:
            raise RuntimeError("Partial output requires explicit --resume")
    else:
        trainer.atomic_json(identity, manifest)
    runs = []
    try:
        root, cache = trainer.find_geometry_cache(args.cache_root)
        graphs = trainer.load_graphs(root, cache)
        trainer.atomic_json(output / "progress.json", {**manifest, "state": "CACHE_VERIFIED", "complete": False})

        def factory(candidate):
            return (candidate_factory() if candidate == candidates[1]
                    else make_pcqm_gap_encoder(candidate))

        for candidate_name in candidates:
            metrics_path = output / "results" / candidate_name / "metrics.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                if metrics.get("source_commit") != args.source_commit or not metrics.get("complete"):
                    raise RuntimeError("Existing result identity mismatch")
                if "validation_csv" not in metrics["artifacts"]:
                    export_csv(trainer, metrics)
                    trainer.atomic_json(metrics_path, metrics)
                runs.append(metrics)
                continue
            trainer.atomic_json(output / "progress.json", {**manifest, "state": "TRAINING", "candidate": candidate_name, "completed_candidates": [r["candidate"] for r in runs], "complete": False})
            torch.cuda.reset_peak_memory_stats(0)
            metrics = trainer.train_one(graphs, candidate_name, started, 0, model_factory=factory)
            metrics["peak_memory_allocated_bytes"] = torch.cuda.max_memory_allocated(0)
            metrics["peak_memory_reserved_bytes"] = torch.cuda.max_memory_reserved(0)
            metrics["device_total_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
            metrics["platform_contract"] = CONTRACT
            metrics["architecture_delta"] = (
                candidate_delta if candidate_name == candidate else baseline_delta
            )
            export_csv(trainer, metrics)
            trainer.atomic_json(metrics_path, metrics)
            runs.append(metrics)
        trainer.atomic_json(output / "completion.json", {
            **manifest, "complete": True, "runs": runs,
            "elapsed_s": time.perf_counter() - started,
            "candidate_minus_control_eV": runs[1]["validation_gap_mae_eV"] - runs[0]["validation_gap_mae_eV"],
        })
        print("KUNSHAN_PAIRED_COMPLETE", flush=True)
    except Exception as error:
        trainer.atomic_json(output / "failure.json", {
            **manifest, "type": type(error).__name__, "message": str(error),
            "completed_candidates": [r["candidate"] for r in runs],
            "elapsed_s": time.perf_counter() - started,
        })
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--screen", choices=("vector", "moment_readout"), default="vector"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
