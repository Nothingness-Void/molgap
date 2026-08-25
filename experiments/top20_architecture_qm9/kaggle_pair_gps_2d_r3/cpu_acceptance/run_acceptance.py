"""Kaggle CPU: replay R3 validation metrics and checkpoint identity."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath

import torch

OUT = Path("/kaggle/working/pure2d_r3_tensor_acceptance")
CANDIDATES = (
    "edge_state_structural_gps",
    "edge_state_structural_orbital",
    "pair_gps_2d_r3_orbital",
    "pair_gps_2d_r3_triplet",
    "pair_gps_2d_r3_combined",
)
EXPECTED_SOURCE_COMMIT = "b56205967f12f517c7eea4428c0dfe8571c54996"
EXPECTED_SPLIT_FINGERPRINT = "01656b1a538f89c8"
EXPECTED_RWSE_SHA256 = (
    "09943454c2e2aae7cf8eeaa828cca79c14be4b920c4c348e0de13a4263331ca5"
)
REFERENCE_AVERAGE = 0.11006919294595718
REFERENCE_GAP = 0.1318935602903366
PARAMETER_BUDGET = 4_800_000


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_one(name: str) -> Path:
    matches = list(Path("/kaggle/input").rglob(name))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name}, found {matches}")
    return matches[0]


def resolve_remote_artifact(remote_path: str) -> Path:
    relative = PurePosixPath(remote_path).relative_to("/kaggle/working")
    suffix = relative.as_posix()
    matches = [
        path
        for path in Path("/kaggle/input").rglob(relative.name)
        if path.as_posix().endswith(suffix)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected artifact suffix {suffix}: {matches}")
    return matches[0]


def load_candidate_metrics(candidate: str) -> tuple[Path, dict]:
    matches = []
    for path in Path("/kaggle/input").rglob("metrics.json"):
        value = json.loads(path.read_text())
        if value.get("candidate") == candidate:
            matches.append((path, value))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one metrics file for {candidate}: {matches}")
    return matches[0]


def recompute_metrics(payload: dict) -> dict:
    prediction = payload["predictions"].detach().cpu().float()
    target = payload["targets"].detach().cpu().float()
    if prediction.ndim != 2 or prediction.shape[1] != 3:
        raise RuntimeError(f"Unexpected prediction shape: {tuple(prediction.shape)}")
    if prediction.shape != target.shape:
        raise RuntimeError("Prediction and target shapes differ")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise RuntimeError("Predictions or targets contain non-finite values")
    values = (prediction - target).abs().mean(dim=0)
    return {
        "HOMO": {"mae": float(values[0])},
        "LUMO": {"mae": float(values[1])},
        "Gap": {"mae": float(values[2])},
        "average": {"mae": float(values.mean())},
    }


def compare_metrics(recorded: dict, replayed: dict, candidate: str, role: str) -> None:
    for target in ("HOMO", "LUMO", "Gap", "average"):
        delta = abs(float(recorded[target]["mae"]) - replayed[target]["mae"])
        if delta > 1e-7:
            raise RuntimeError(
                f"{candidate} {role} {target} replay mismatch: {delta}"
            )


def compare_state_dicts(model: dict, best_state: dict, candidate: str) -> None:
    if set(model) != set(best_state):
        raise RuntimeError(f"State keys differ for {candidate}")
    for key in model:
        left = model[key]
        right = best_state[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"State metadata differs for {candidate}:{key}")
        if not torch.equal(left, right):
            raise RuntimeError(f"Best model differs from checkpoint for {candidate}:{key}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        selection_path = find_one("selection.json")
        preflight_path = find_one("preflight.json")
        selection = json.loads(selection_path.read_text())
        preflight = json.loads(preflight_path.read_text())
        if selection.get("source_commit") != EXPECTED_SOURCE_COMMIT:
            raise RuntimeError("Selection source commit mismatch")
        if selection.get("test_role_read") is not False:
            raise RuntimeError("Selection reports a test-role read")
        if preflight.get("complete") is not True:
            raise RuntimeError("Preflight is incomplete")
        if preflight.get("source_commit") != EXPECTED_SOURCE_COMMIT:
            raise RuntimeError("Preflight source commit mismatch")
        if preflight.get("split_fingerprint") != EXPECTED_SPLIT_FINGERPRINT:
            raise RuntimeError("Preflight split mismatch")
        if preflight.get("rwse_output_sha256") != EXPECTED_RWSE_SHA256:
            raise RuntimeError("Preflight RWSE cache mismatch")
        if preflight.get("test_role_read") is not False:
            raise RuntimeError("Preflight reports a test-role read")
        preflight_rows = preflight.get("candidates", [])
        if tuple(row.get("candidate") for row in preflight_rows) != CANDIDATES:
            raise RuntimeError("Preflight candidate order mismatch")
        preflight_by_candidate = {
            row["candidate"]: row for row in preflight_rows
        }
        completed = selection.get("completed", [])
        if tuple(row.get("candidate") for row in completed) != CANDIDATES:
            raise RuntimeError("Candidate completion order mismatch")

        reports = []
        eligible = []
        train_indices = set()
        validation_indices = set()
        for selected_row in completed:
            candidate = selected_row["candidate"]
            metrics_path, metrics = load_candidate_metrics(candidate)
            preflight_row = preflight_by_candidate[candidate]
            for key in ("finite_prediction", "finite_loss", "finite_gradients"):
                if preflight_row.get(key) is not True:
                    raise RuntimeError(f"{candidate} failed {key} preflight")
            if metrics.get("test_role_evaluated") is not False:
                raise RuntimeError(f"{candidate} evaluated test")
            if set(metrics["metrics"]) != {"train", "validation"}:
                raise RuntimeError(f"{candidate} has unexpected roles")
            payload_path = resolve_remote_artifact(metrics["artifacts"]["embeddings"])
            model_path = resolve_remote_artifact(metrics["artifacts"]["model"])
            checkpoint_path = resolve_remote_artifact(metrics["artifacts"]["checkpoint"])
            payloads = torch.load(
                payload_path, map_location="cpu", weights_only=True
            )
            if set(payloads) != {"train", "validation"}:
                raise RuntimeError(f"{candidate} payload roles changed")
            replayed = {}
            role_indices = {}
            identity_errors = {}
            for role, expected_rows in (("train", 30_000), ("validation", 3_000)):
                role_payload = payloads[role]
                required = {"predictions", "targets", "embeddings", "source_idx"}
                if set(role_payload) != required:
                    raise RuntimeError(f"{candidate} {role} payload keys changed")
                if len(role_payload["source_idx"]) != expected_rows:
                    raise RuntimeError(f"{candidate} {role} row count changed")
                indices = role_payload["source_idx"].detach().cpu().long()
                if torch.unique(indices).numel() != expected_rows:
                    raise RuntimeError(f"{candidate} {role} indices are not unique")
                role_indices[role] = set(indices.tolist())
                replayed[role] = recompute_metrics(role_payload)
                compare_metrics(
                    metrics["metrics"][role], replayed[role], candidate, role
                )
                prediction = role_payload["predictions"].detach().cpu().float()
                target = role_payload["targets"].detach().cpu().float()
                target_identity = (
                    target[:, 1] - target[:, 0] - target[:, 2]
                ).abs().max()
                prediction_identity = (
                    prediction[:, 1] - prediction[:, 0] - prediction[:, 2]
                ).abs().max()
                if float(target_identity) > 1e-5:
                    raise RuntimeError(f"{candidate} target identity changed")
                if metrics.get("frontier_head") is not None and float(
                    prediction_identity
                ) > 1e-5:
                    raise RuntimeError(f"{candidate} frontier identity failed")
                identity_errors[role] = {
                    "target_max_eV": float(target_identity),
                    "prediction_max_eV": float(prediction_identity),
                }
            if role_indices["train"] & role_indices["validation"]:
                raise RuntimeError(f"{candidate} train/validation overlap")
            if not train_indices:
                train_indices = role_indices["train"]
                validation_indices = role_indices["validation"]
            elif (
                role_indices["train"] != train_indices
                or role_indices["validation"] != validation_indices
            ):
                raise RuntimeError(f"{candidate} split indices differ")

            model = torch.load(model_path, map_location="cpu", weights_only=True)
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            compare_state_dicts(model, checkpoint["best_state"], candidate)
            parameter_count = int(metrics["n_params"])
            if parameter_count != int(preflight_row["parameter_count"]):
                raise RuntimeError(f"{candidate} preflight parameter count differs")
            if parameter_count != int(selected_row["parameter_count"]):
                raise RuntimeError(f"{candidate} selection parameter count differs")
            if int(checkpoint["best_epoch"]) != int(metrics["best_epoch"]):
                raise RuntimeError(f"{candidate} checkpoint epoch differs")
            validation = replayed["validation"]
            row_eligible = (
                validation["average"]["mae"] < REFERENCE_AVERAGE
                and validation["Gap"]["mae"] < REFERENCE_GAP
                and parameter_count <= PARAMETER_BUDGET
            )
            if selected_row.get("eligible") is not row_eligible:
                raise RuntimeError(f"{candidate} eligibility differs")
            if row_eligible:
                eligible.append((validation["average"]["mae"], candidate))
            reports.append(
                {
                    "candidate": candidate,
                    "parameter_count": parameter_count,
                    "best_epoch": int(metrics["best_epoch"]),
                    "validation": validation,
                    "eligible": row_eligible,
                    "identity_errors": identity_errors,
                    "payload_sha256": sha256(payload_path),
                    "model_sha256": sha256(model_path),
                    "checkpoint_sha256": sha256(checkpoint_path),
                    "metrics_sha256": sha256(metrics_path),
                }
            )
        expected_winner = min(eligible)[1] if eligible else None
        if selection.get("selected_candidate") != expected_winner:
            raise RuntimeError("Selected candidate differs after tensor replay")
        report = {
            "format": "molgap-pure2d-r3-tensor-acceptance-v1",
            "accepted": True,
            "device": "cpu",
            "model_inference_executed": False,
            "source_commit": EXPECTED_SOURCE_COMMIT,
            "split_fingerprint": EXPECTED_SPLIT_FINGERPRINT,
            "test_role_read": False,
            "candidate_count": len(reports),
            "selected_candidate": expected_winner,
            "selection_sha256": sha256(selection_path),
            "preflight_sha256": sha256(preflight_path),
            "candidates": reports,
        }
        atomic_json(OUT / "tensor_acceptance.json", report)
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        atomic_json(
            OUT / "failure.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    main()
