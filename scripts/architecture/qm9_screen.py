"""Thin CLI for the reproducible QM9 architecture screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from molgap.qm9_conformer import train_schnet_conformer_augmented
from molgap.qm9_fusion import (
    evaluate_gap_threshold_route,
    train_bounded_residual_fusion,
    train_standard_fusion,
    validation_selected_prediction_blend,
)
from molgap.qm9_payloads import (
    align_embedding_payload_to_reference,
    combine_embedding_payloads_on_intersection,
    concatenate_embedding_payloads,
    static_blend_metrics,
)
from molgap.qm9_screen import (
    DEFAULT_CACHE,
    DEFAULT_RESULTS,
    prepare_qm9_files,
    evaluate_encoder_on_geometry,
    export_gps_multiscale_embeddings,
    train_encoder,
)


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("prepare")

    encoder = subparsers.add_parser("encoder")
    encoder.add_argument(
        "--candidate",
        choices=[
            "gine6",
            "gps7",
            "gps9",
            "gps9_160",
            "gps9_128",
            "gps9_meanmax",
            "gps11_160",
            "schnet",
            "tensornet",
            "egnn",
        ],
        required=True,
    )
    encoder.add_argument("--geometry", choices=["topology", "dft", "etkdg"], required=True)
    encoder.add_argument("--train-size", type=int, default=100_000)
    encoder.add_argument("--validation-size", type=int, default=10_000)
    encoder.add_argument("--test-size", type=int, default=10_000)
    encoder.add_argument("--epochs", type=int, default=30)
    encoder.add_argument("--seed", type=int, default=42)
    encoder.add_argument("--split-seed", type=int, default=42)
    encoder.add_argument("--resume", action="store_true")

    geometry_eval = subparsers.add_parser("evaluate-geometry")
    geometry_eval.add_argument("--candidate", choices=["schnet", "tensornet", "egnn"], required=True)
    geometry_eval.add_argument("--geometry", choices=["dft", "etkdg"], required=True)
    geometry_eval.add_argument("--checkpoint", type=Path, required=True)
    geometry_eval.add_argument("--output", type=Path, required=True)
    geometry_eval.add_argument("--embedding-output", type=Path, required=True)
    geometry_eval.add_argument("--train-size", type=int, default=100_000)
    geometry_eval.add_argument("--validation-size", type=int, default=10_000)
    geometry_eval.add_argument("--test-size", type=int, default=10_000)
    geometry_eval.add_argument("--split-seed", type=int, default=42)
    geometry_eval.add_argument("--geometry-seed", type=int, required=True)

    multiscale = subparsers.add_parser("gps-multiscale")
    multiscale.add_argument("--checkpoint", type=Path, required=True)
    multiscale.add_argument("--output", type=Path, required=True)
    multiscale.add_argument("--embedding-output", type=Path, required=True)
    multiscale.add_argument("--train-size", type=int, default=100_000)
    multiscale.add_argument("--validation-size", type=int, default=10_000)
    multiscale.add_argument("--test-size", type=int, default=10_000)
    multiscale.add_argument("--split-seed", type=int, default=42)
    multiscale.add_argument("--layers", type=int, nargs="+", default=[2, 4, -1])

    augmented = subparsers.add_parser("schnet-conformer-aug")
    augmented.add_argument("--train-size", type=int, default=100_000)
    augmented.add_argument("--validation-size", type=int, default=10_000)
    augmented.add_argument("--test-size", type=int, default=10_000)
    augmented.add_argument("--epochs", type=int, default=30)
    augmented.add_argument("--seed", type=int, default=42)
    augmented.add_argument("--split-seed", type=int, default=42)
    augmented.add_argument("--geometry-seeds", type=int, nargs=2, default=[42, 43])
    augmented.add_argument("--resume", action="store_true")

    blend = subparsers.add_parser("blend")
    blend.add_argument("--gine-payload", type=Path, required=True)
    blend.add_argument("--gps-payload", type=Path, required=True)
    blend.add_argument("--output", type=Path, required=True)

    concat = subparsers.add_parser("concat-payload")
    concat.add_argument("--primary-payload", type=Path, required=True)
    concat.add_argument("--secondary-payload", type=Path, required=True)
    concat.add_argument("--output", type=Path, required=True)

    align = subparsers.add_parser("align-payload")
    align.add_argument("--payload", type=Path, required=True)
    align.add_argument("--reference", type=Path, required=True)
    align.add_argument("--output", type=Path, required=True)

    intersect = subparsers.add_parser("intersect-payloads")
    intersect.add_argument("--primary-payload", type=Path, required=True)
    intersect.add_argument("--secondary-payload", type=Path, required=True)
    intersect.add_argument("--output-dir", type=Path, required=True)
    intersect.add_argument("--output", type=Path, required=True)

    fusion = subparsers.add_parser("fusion")
    fusion.add_argument("--gps-payload", type=Path, required=True)
    fusion.add_argument("--schnet-payload", type=Path, required=True)
    fusion.add_argument("--epochs", type=int, default=100)
    fusion.add_argument("--seed", type=int, default=42)
    fusion.add_argument("--hidden", type=int, default=192)
    fusion.add_argument("--fusion-type", choices=["gate", "concat"], default="gate")
    fusion.add_argument("--output", type=Path, required=True)

    residual = subparsers.add_parser("residual-fusion")
    residual.add_argument("--gps-payload", type=Path, required=True)
    residual.add_argument("--schnet-payload", type=Path, required=True)
    residual.add_argument("--epochs", type=int, default=100)
    residual.add_argument("--seed", type=int, default=42)
    residual.add_argument("--max-correction-eV", type=float, default=0.2)
    residual.add_argument("--output", type=Path, required=True)

    route = subparsers.add_parser("route")
    route.add_argument("--base-predictions", type=Path, required=True)
    route.add_argument("--dual-predictions", type=Path, required=True)
    route.add_argument("--target-payload", type=Path, required=True)
    route.add_argument("--output", type=Path, required=True)

    prediction_blend = subparsers.add_parser("prediction-blend")
    prediction_blend.add_argument("--first-predictions", type=Path, required=True)
    prediction_blend.add_argument("--second-predictions", type=Path, required=True)
    prediction_blend.add_argument("--target-payload", type=Path, required=True)
    prediction_blend.add_argument("--grid-points", type=int, default=101)
    prediction_blend.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: item for key, item in value.items()
        if not isinstance(item, torch.Tensor)
    }
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    if args.command == "prepare":
        print(prepare_qm9_files())
        return
    if args.command == "encoder":
        result = train_encoder(
            candidate=args.candidate,
            geometry=args.geometry,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            epochs=args.epochs,
            seed=args.seed,
            split_seed=args.split_seed,
            resume=args.resume,
        )
        print(json.dumps(result["metrics"]["test"], indent=2))
        return
    if args.command == "evaluate-geometry":
        result = evaluate_encoder_on_geometry(
            candidate=args.candidate,
            geometry=args.geometry,
            checkpoint=args.checkpoint,
            output=args.output,
            embedding_output=args.embedding_output,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            split_seed=args.split_seed,
            geometry_seed=args.geometry_seed,
        )
        print(json.dumps(result["metrics"]["test"], indent=2))
        return
    if args.command == "gps-multiscale":
        result = export_gps_multiscale_embeddings(
            checkpoint=args.checkpoint,
            output=args.output,
            embedding_output=args.embedding_output,
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            split_seed=args.split_seed,
            layers=tuple(args.layers),
        )
        print(json.dumps(result["metrics"]["test"], indent=2))
        return
    if args.command == "schnet-conformer-aug":
        result = train_schnet_conformer_augmented(
            train_size=args.train_size,
            validation_size=args.validation_size,
            test_size=args.test_size,
            epochs=args.epochs,
            seed=args.seed,
            split_seed=args.split_seed,
            geometry_seeds=tuple(args.geometry_seeds),
            resume=args.resume,
        )
        print(json.dumps(result["metrics"]["test"], indent=2))
        return
    if args.command == "blend":
        result = static_blend_metrics(args.gine_payload, args.gps_payload)
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    if args.command == "concat-payload":
        result = concatenate_embedding_payloads(
            args.primary_payload,
            args.secondary_payload,
            args.output,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "align-payload":
        result = align_embedding_payload_to_reference(
            args.payload,
            args.reference,
            args.output,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "intersect-payloads":
        result = combine_embedding_payloads_on_intersection(
            args.primary_payload,
            args.secondary_payload,
            args.output_dir,
        )
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    if args.command == "route":
        result = evaluate_gap_threshold_route(
            args.base_predictions,
            args.dual_predictions,
            args.target_payload,
        )
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    if args.command == "prediction-blend":
        result = validation_selected_prediction_blend(
            args.first_predictions,
            args.second_predictions,
            args.target_payload,
            args.grid_points,
        )
        write_json(args.output, result)
        print(json.dumps(result, indent=2))
        return
    if args.command == "fusion":
        result = train_standard_fusion(
            topology_payload=args.gps_payload,
            geometry_payload=args.schnet_payload,
            epochs=args.epochs,
            seed=args.seed,
            hidden=args.hidden,
            fusion_type=args.fusion_type,
        )
    else:
        result = train_bounded_residual_fusion(
            topology_payload=args.gps_payload,
            geometry_payload=args.schnet_payload,
            epochs=args.epochs,
            seed=args.seed,
            max_abs_correction_eV=args.max_correction_eV,
        )
    predictions = result.pop("predictions")
    source_idx = result.pop("source_idx")
    validation_predictions = result.pop("validation_predictions")
    validation_source_idx = result.pop("validation_source_idx")
    write_json(args.output, result)
    (DEFAULT_CACHE / "fusion").mkdir(parents=True, exist_ok=True)
    artifact_name = (
        f"{args.output.parent.parent.name}_{args.output.parent.name}_"
        f"{args.output.stem}.pt"
    )
    torch.save(
        {
            "validation_predictions": validation_predictions,
            "validation_source_idx": validation_source_idx,
            "test_predictions": predictions,
            "test_source_idx": source_idx,
        },
        DEFAULT_CACHE / "fusion" / artifact_name,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
