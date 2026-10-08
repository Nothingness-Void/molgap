"""Focused checks for retained 500K resume artifacts."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from molgap.training_reproducibility import retained_resume_artifacts


ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_PATH = ROOT / "platforms" / "kaggle" / "run_legacy_500k_pair.py"
BOOTSTRAP_SPEC = importlib.util.spec_from_file_location(
    "legacy_500k_resume_bootstrap", BOOTSTRAP_PATH
)
bootstrap = importlib.util.module_from_spec(BOOTSTRAP_SPEC)
BOOTSTRAP_SPEC.loader.exec_module(bootstrap)

MINIMAL_RETENTION = "selected-and-resume-v1"
ARM_ID = "k1_pretrained_consistency"
SOURCE_SHA256 = "a" * 64
REQUIRED_ARTIFACTS = (
    "last_checkpoint.pt",
    "best_model.pt",
    "best_predictions.pt",
    "initial_state.pt",
    "runtime.json",
    "trace.json",
    "scientific_contract.json",
    "data_manifest.json",
    "runtime_certificate.json",
)
OPTIONAL_ARTIFACTS = (
    "best_live_predictions.pt",
    "objective_components.json",
    "supporting_optional.json",
)


def _artifact_map(names):
    return {name: hashlib.sha256(name.encode("utf-8")).hexdigest() for name in names}


def test_legacy_resume_retains_every_manifest_artifact():
    artifacts = _artifact_map(
        (*REQUIRED_ARTIFACTS, *OPTIONAL_ARTIFACTS, "predictions_epoch_0.pt")
    )
    manifest = {"artifacts": artifacts}

    assert retained_resume_artifacts(manifest) == artifacts


def test_minimal_resume_drops_only_numbered_epoch_predictions():
    epoch_predictions = (
        "predictions_epoch_0.pt",
        "predictions_epoch_007.pt",
        "predictions_epoch_42.pt",
    )
    retained_optional = (
        *OPTIONAL_ARTIFACTS,
        "predictions_epoch_final.pt",
        "predictions_epoch_4.backup.pt",
        "predictions_epoch_4.PT",
    )
    artifacts = _artifact_map(
        (*REQUIRED_ARTIFACTS, *retained_optional, *epoch_predictions)
    )
    manifest = {"artifacts": artifacts}

    retained = retained_resume_artifacts(manifest, MINIMAL_RETENTION)

    assert retained == {
        name: checksum
        for name, checksum in artifacts.items()
        if name not in epoch_predictions
    }
    assert set(OPTIONAL_ARTIFACTS).issubset(retained)


@pytest.mark.parametrize("missing", REQUIRED_ARTIFACTS)
def test_minimal_resume_rejects_each_missing_required_artifact(missing):
    artifacts = _artifact_map(
        name for name in (*REQUIRED_ARTIFACTS, *OPTIONAL_ARTIFACTS)
        if name != missing
    )

    with pytest.raises(ValueError):
        retained_resume_artifacts({"artifacts": artifacts}, MINIMAL_RETENTION)


def test_minimal_resume_rejects_unknown_retention_policy():
    manifest = {"artifacts": _artifact_map(REQUIRED_ARTIFACTS)}

    with pytest.raises(ValueError):
        retained_resume_artifacts(manifest, "selected-and-resume-v2")


@pytest.fixture
def loaded_bootstrap(monkeypatch):
    # Keep the bootstrap's file hashing portable across supported Python builds.
    monkeypatch.setattr(
        bootstrap,
        "digest",
        lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
    )
    return bootstrap


def _mounted_minimal_resume(tmp_path, *, status="STAGE_COMPLETE", next_epoch=24):
    mounted = tmp_path / "mounted"
    root = mounted / "checkpoints" / ARM_ID
    root.mkdir(parents=True)
    names = (*REQUIRED_ARTIFACTS, *OPTIONAL_ARTIFACTS, "predictions_epoch_24.pt")
    payloads = {name: ("payload:" + name).encode("utf-8") for name in names}
    artifacts = {
        name: hashlib.sha256(payloads[name]).hexdigest() for name in names
    }

    # A minimal mount retains selected/resume files and optional evidence. The
    # original stage manifest still lists the deliberately pruned epoch file.
    for name in (*REQUIRED_ARTIFACTS, *OPTIONAL_ARTIFACTS):
        (root / name).write_bytes(payloads[name])
    manifest = {
        "status": status,
        "arm": ARM_ID,
        "source_sha256": SOURCE_SHA256,
        "next_epoch": next_epoch,
        "artifacts": artifacts,
    }
    manifest_bytes = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    (root / "stage_manifest.json").write_bytes(manifest_bytes)
    arm = {
        "arm_id": ARM_ID,
        "resume": {
            "mount": "checkpoints",
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "source_sha256": SOURCE_SHA256,
            "next_epoch": next_epoch,
            "retention": MINIMAL_RETENTION,
        },
    }
    return mounted, root, arm, manifest_bytes


def test_bootstrap_resolves_minimal_resume_without_rewriting_manifest(
    tmp_path, loaded_bootstrap
):
    mounted, root, arm, manifest_bytes = _mounted_minimal_resume(tmp_path)

    resolved = loaded_bootstrap.resolve_resume(mounted, arm)

    assert resolved == root
    assert (root / "stage_manifest.json").read_bytes() == manifest_bytes


def test_bootstrap_rejects_missing_checkpoint_in_minimal_resume(
    tmp_path, loaded_bootstrap
):
    mounted, root, arm, manifest_bytes = _mounted_minimal_resume(tmp_path)
    (root / "last_checkpoint.pt").unlink()

    with pytest.raises((FileNotFoundError, RuntimeError)):
        loaded_bootstrap.resolve_resume(mounted, arm)

    assert (root / "stage_manifest.json").read_bytes() == manifest_bytes


def test_bootstrap_rejects_hash_damage_in_minimal_resume(
    tmp_path, loaded_bootstrap
):
    mounted, root, arm, manifest_bytes = _mounted_minimal_resume(tmp_path)
    (root / "runtime.json").write_bytes(b"tampered runtime")

    with pytest.raises(RuntimeError, match="artifact changed"):
        loaded_bootstrap.resolve_resume(mounted, arm)

    assert (root / "stage_manifest.json").read_bytes() == manifest_bytes


def test_bootstrap_retains_selected_and_optional_artifacts_without_epoch_files(
    tmp_path, loaded_bootstrap
):
    mounted, root, arm, manifest_bytes = _mounted_minimal_resume(
        tmp_path, status="COMPLETE", next_epoch=60
    )
    destination = tmp_path / "published" / ARM_ID

    assert loaded_bootstrap.retain_completed_arm(
        root, destination, arm, stage_epochs=60
    )

    assert {path.name for path in destination.iterdir()} == {
        *REQUIRED_ARTIFACTS,
        *OPTIONAL_ARTIFACTS,
        "stage_manifest.json",
    }
    assert (destination / "stage_manifest.json").read_bytes() == manifest_bytes
    assert (root / "stage_manifest.json").read_bytes() == manifest_bytes
