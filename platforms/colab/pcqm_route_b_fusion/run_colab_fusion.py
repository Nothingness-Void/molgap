"""Run the accepted Route B Fusion protocol from one Drive payload."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from molgap.pcqm_route_b_fusion import (
    FusionConfig,
    load_consolidated_fusion_payload,
    preflight_fusion_from_payloads,
    train_fusion_screen_from_payloads,
)
from molgap.pcqm_route_b_training import atomic_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    payloads, load_report, checkpoint_paths = load_consolidated_fusion_payload(
        args.input_dir
    )
    preflight = preflight_fusion_from_payloads(
        payloads=payloads,
        checkpoint_paths=checkpoint_paths,
        output_path=args.result_dir / "preflight.json",
        config=FusionConfig(),
    )
    summary = train_fusion_screen_from_payloads(
        payloads=payloads,
        load_report=load_report,
        checkpoint_paths=checkpoint_paths,
        output_dir=args.checkpoint_dir,
        config=FusionConfig(),
    )
    for name in ("development_selection.json", "completion_manifest.json"):
        source = args.checkpoint_dir / name
        destination = args.result_dir / name
        temporary = destination.with_suffix(".tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(destination)
    report = {
        "format": "molgap-pcqm-route-b-colab-fusion-completion-v1",
        "status": "complete",
        "selected_base_identity": summary["selected_base_identity"],
        "mean_dev_gap_mae_eV": summary["mean_dev_gap_mae_eV"],
        "checkpoint_dir": str(args.checkpoint_dir),
        "input_manifest_sha256": sha256_file(args.input_dir / "manifest.json"),
        "development_selection_sha256": sha256_file(
            args.result_dir / "development_selection.json"
        ),
        "preflight": preflight,
        "official_valid_metric_read": False,
        "official_test_used": False,
        "sealed_20k_used": False,
        "production_registry_changed": False,
    }
    atomic_json(args.result_dir / "colab_completion.json", report)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
