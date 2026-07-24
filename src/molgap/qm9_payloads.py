"""Alignment and combination utilities for cached QM9 embeddings."""
from __future__ import annotations

from pathlib import Path

import torch

from .qm9_screen import _atomic_torch_save, _metrics


def static_blend_metrics(payload_a: Path, payload_b: Path) -> dict:
    first = torch.load(payload_a, map_location="cpu", weights_only=False)
    second = torch.load(payload_b, map_location="cpu", weights_only=False)
    result = {"blend": "equal_prediction_average", "weight_a": 0.5, "weight_b": 0.5}
    metrics = {}
    for role in ("validation", "test"):
        a = first[role]
        b = second[role]
        if not torch.equal(a["source_idx"], b["source_idx"]):
            raise ValueError(f"Unaligned static-blend {role} rows")
        prediction = 0.5 * (a["predictions"] + b["predictions"])
        metrics[role] = _metrics(prediction.numpy(), a["targets"].numpy())
    result["metrics"] = metrics
    return result


def concatenate_embedding_payloads(
    primary_payload: Path,
    secondary_payload: Path,
    output: Path,
) -> dict:
    primary = torch.load(primary_payload, map_location="cpu", weights_only=False)
    secondary = torch.load(
        secondary_payload, map_location="cpu", weights_only=False
    )
    combined = {}
    for role in ("train", "validation", "test"):
        first = primary[role]
        second = secondary[role]
        if not torch.equal(first["source_idx"], second["source_idx"]):
            raise ValueError(f"Unaligned embedding-concat {role} rows")
        if not torch.equal(first["targets"], second["targets"]):
            raise ValueError(f"Mismatched embedding-concat {role} targets")
        combined[role] = {
            "embeddings": torch.cat(
                (first["embeddings"], second["embeddings"]), dim=-1
            ),
            "predictions": first["predictions"],
            "targets": first["targets"],
            "source_idx": first["source_idx"],
        }
    _atomic_torch_save(output, combined)
    return {
        "operation": "concatenate_embeddings",
        "primary_payload": str(primary_payload),
        "secondary_payload": str(secondary_payload),
        "output": str(output),
        "embedding_dim": int(combined["train"]["embeddings"].shape[1]),
        "rows": {role: len(value["source_idx"]) for role, value in combined.items()},
    }


def align_embedding_payload_to_reference(
    payload_path: Path,
    reference_path: Path,
    output: Path,
) -> dict:
    payload = torch.load(payload_path, map_location="cpu", weights_only=False)
    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    aligned = {}
    for role in ("train", "validation", "test"):
        positions = {
            int(value): i
            for i, value in enumerate(payload[role]["source_idx"].tolist())
        }
        reference_indices = reference[role]["source_idx"].tolist()
        missing = [
            int(value)
            for value in reference_indices
            if int(value) not in positions
        ]
        if missing:
            raise ValueError(
                f"Reference {role} contains {len(missing)} unavailable rows"
            )
        selected = torch.tensor(
            [positions[int(value)] for value in reference_indices]
        )
        aligned[role] = {
            key: value[selected]
            for key, value in payload[role].items()
            if isinstance(value, torch.Tensor)
        }
    _atomic_torch_save(output, aligned)
    return {
        "operation": "align_embedding_payload_to_reference",
        "payload": str(payload_path),
        "reference": str(reference_path),
        "output": str(output),
        "rows": {role: len(value["source_idx"]) for role, value in aligned.items()},
    }


def combine_embedding_payloads_on_intersection(
    primary_payload: Path,
    secondary_payload: Path,
    output_dir: Path,
) -> dict:
    """Build fair single- and dual-view payloads on their exact row intersection."""
    primary = torch.load(primary_payload, map_location="cpu", weights_only=False)
    secondary = torch.load(
        secondary_payload, map_location="cpu", weights_only=False
    )
    outputs = {
        "primary": {},
        "secondary": {},
        "average": {},
        "concat": {},
    }
    metrics = {"primary": {}, "secondary": {}, "average": {}}
    for role in ("train", "validation", "test"):
        first = primary[role]
        second = secondary[role]
        first_pos = {
            int(value): i for i, value in enumerate(first["source_idx"].tolist())
        }
        second_pos = {
            int(value): i for i, value in enumerate(second["source_idx"].tolist())
        }
        common = sorted(set(first_pos).intersection(second_pos))
        if not common:
            raise ValueError(f"No common {role} rows between embedding payloads")
        first_index = torch.tensor([first_pos[value] for value in common])
        second_index = torch.tensor([second_pos[value] for value in common])
        first_targets = first["targets"][first_index]
        second_targets = second["targets"][second_index]
        if not torch.equal(first_targets, second_targets):
            raise ValueError(f"Mismatched targets on common {role} rows")
        first_embeddings = first["embeddings"][first_index]
        second_embeddings = second["embeddings"][second_index]
        if first_embeddings.shape[1] != second_embeddings.shape[1]:
            raise ValueError(
                "Embedding average requires equal dimensions, got "
                f"{first_embeddings.shape[1]} and {second_embeddings.shape[1]}"
            )
        first_predictions = first["predictions"][first_index]
        second_predictions = second["predictions"][second_index]
        average_predictions = 0.5 * (first_predictions + second_predictions)
        source_idx = torch.tensor(common)
        shared = {"targets": first_targets, "source_idx": source_idx}
        outputs["primary"][role] = {
            **shared,
            "embeddings": first_embeddings,
            "predictions": first_predictions,
        }
        outputs["secondary"][role] = {
            **shared,
            "embeddings": second_embeddings,
            "predictions": second_predictions,
        }
        outputs["average"][role] = {
            **shared,
            "embeddings": 0.5 * (first_embeddings + second_embeddings),
            "predictions": average_predictions,
        }
        outputs["concat"][role] = {
            **shared,
            "embeddings": torch.cat(
                (first_embeddings, second_embeddings), dim=-1
            ),
            "predictions": average_predictions,
        }
        for name, predictions in (
            ("primary", first_predictions),
            ("secondary", second_predictions),
            ("average", average_predictions),
        ):
            metrics[name][role] = _metrics(
                predictions.numpy(), first_targets.numpy()
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, payload in outputs.items():
        path = output_dir / f"{name}.pt"
        _atomic_torch_save(path, payload)
        paths[name] = str(path)
    return {
        "operation": "combine_embedding_payloads_on_intersection",
        "primary_payload": str(primary_payload),
        "secondary_payload": str(secondary_payload),
        "outputs": paths,
        "rows": {
            role: len(outputs["primary"][role]["source_idx"])
            for role in ("train", "validation", "test")
        },
        "embedding_dims": {
            name: int(payload["train"]["embeddings"].shape[1])
            for name, payload in outputs.items()
        },
        "prediction_metrics": metrics,
    }
