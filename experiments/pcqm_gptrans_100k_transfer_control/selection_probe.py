"""Compare raw and EMA states from the accepted local 100K pair without training."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from analyze_pair import paired_metrics, transfer_predictions


def check_terminal(arm: Path, *, reference: bool):
    from molgap import pcqm_gptrans_v4 as v4
    from molgap.v4_runtime import torch_load_compat

    manifest = json.loads((arm / "completion_manifest.json").read_text(encoding="utf-8"))
    checkpoint_path = arm / "last_checkpoint.pt"
    if not manifest.get("complete") or v4.sha256_file(checkpoint_path) != manifest["checkpoint_sha256"]:
        raise RuntimeError("Terminal checkpoint is incomplete or changed")
    checkpoint = torch_load_compat(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["epoch"] != 59 or manifest["best_epoch"] != 59:
        raise RuntimeError("Raw and selected EMA states are not from the same epoch")
    source = checkpoint if reference else checkpoint["run_identity"]
    for key in (("source_archive_sha256",) if reference else ("source_commit", "source_archive_sha256")):
        if source.get(key) != manifest[key]:
            raise RuntimeError(f"Checkpoint {key} differs from terminal manifest")
    if not reference and source["initial_state_sha256"] != "9205fc0f0f97f1cc1cea84ab4bd24206274a00c7d84d26366497feee1710c20c":
        raise RuntimeError("Candidate initial state changed")
    return manifest, checkpoint


def run_probe(args) -> dict:
    import torch
    from molgap import pcqm_gptrans_v4 as v4
    from molgap.noisy_nodes import make_noisy_nodes_pair_norm_model
    from molgap.v4_runtime import torch_load_compat

    started = time.perf_counter()
    reference_manifest, reference_checkpoint = check_terminal(args.reference, reference=True)
    candidate_manifest, candidate_checkpoint = check_terminal(args.candidate, reference=False)
    if reference_manifest["source_commit"] != candidate_manifest["source_commit"]:
        raise RuntimeError("Paired source commits differ")

    assets = v4.validate_fixed_assets(args.cache_100k, args.cache_100k / "manifest.json", verify_content=True)
    graphs, _ = v4._load_datasets(assets.development_paths)
    mean = float(reference_checkpoint["target_stats"]["mean_eV"])
    std = float(reference_checkpoint["target_stats"]["sample_std_eV"])
    raw = {}
    for name, checkpoint, builder in (
        ("reference", reference_checkpoint, v4._make_model),
        ("candidate", candidate_checkpoint, make_noisy_nodes_pair_norm_model),
    ):
        model = builder().to("cuda")
        state = checkpoint["model"] if name == "reference" else checkpoint["model_state"]
        model.load_state_dict(state, strict=True)
        raw[name] = {
            "development_100k": v4._evaluate(model, v4.ExponentialMovingAverage(model), graphs, mean, std),
            "frozen_transfer_500k_development": transfer_predictions(args.cache_500k, model, mean, std),
        }
        del model
        torch.cuda.empty_cache()
    del graphs

    ema = {}
    for name, arm in (("reference", args.reference), ("candidate", args.candidate)):
        path = arm / "development_predictions.pt"
        manifest = reference_manifest if name == "reference" else candidate_manifest
        if v4.sha256_file(path) != manifest["development_predictions_sha256"]:
            raise RuntimeError("Selected EMA predictions changed")
        ema[name] = torch_load_compat(path, map_location="cpu", weights_only=False)
        if not torch.equal(raw[name]["development_100k"]["target_eV"], ema[name]["target_eV"]):
            raise RuntimeError("Raw/EMA development targets differ")

    raw_100k = paired_metrics(raw["reference"]["development_100k"], raw["candidate"]["development_100k"], 100000, 150000)
    raw_500k = paired_metrics(raw["reference"]["frozen_transfer_500k_development"], raw["candidate"]["frozen_transfer_500k_development"], 500000, 550000)
    ema_100k = paired_metrics(ema["reference"], ema["candidate"], 100000, 150000)
    prior = json.loads(args.prior_analysis.read_text(encoding="utf-8"))
    ema_500k = prior["frozen_transfer_500k_development"]
    if abs(ema_100k["paired_gain_eV"] - prior["development_100k"]["paired_gain_eV"]) > 1e-9:
        raise RuntimeError("Published EMA result changed")

    return {
        "format": "molgap-local-gptrans-raw-vs-ema-selection-probe-v1",
        "status": "diagnostic_only",
        "source_commit": reference_manifest["source_commit"],
        "checkpoint_epoch": 59,
        "raw_development_100k": raw_100k,
        "ema_development_100k": ema_100k,
        "raw_frozen_transfer_500k_development": raw_500k,
        "ema_frozen_transfer_500k_development": ema_500k,
        "selection_effect_on_gain_100k_eV": ema_100k["paired_gain_eV"] - raw_100k["paired_gain_eV"],
        "selection_effect_on_gain_500k_eV": ema_500k["paired_gain_eV"] - raw_500k["paired_gain_eV"],
        "elapsed_script_seconds": time.perf_counter() - started,
        "artifact_sha256": {
            "reference_last_checkpoint": reference_manifest["checkpoint_sha256"],
            "candidate_last_checkpoint": candidate_manifest["checkpoint_sha256"],
            "reference_ema_development_predictions": reference_manifest["development_predictions_sha256"],
            "candidate_ema_development_predictions": candidate_manifest["development_predictions_sha256"],
            "prior_analysis": v4.sha256_file(args.prior_analysis),
        },
        "limitations": [
            "One training seed; this changes only evaluation weights at the same final epoch.",
            "The 500K development role was previously used for model selection.",
            "This does not test the 500K training optimizer or data-size effect.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--cache-100k", type=Path, required=True)
    parser.add_argument("--cache-500k", type=Path, required=True)
    parser.add_argument("--prior-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in ("raw_development_100k", "raw_frozen_transfer_500k_development", "selection_effect_on_gain_100k_eV", "selection_effect_on_gain_500k_eV")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
