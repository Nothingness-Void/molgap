import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "platforms" / "kaggle" / "run_legacy_500k_pair.py"
SPEC = importlib.util.spec_from_file_location("legacy_500k_pair", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def file_digest_compat(monkeypatch):
    # The Kaggle runtime uses Python 3.11; local test environments may be older.
    if not hasattr(hashlib, "file_digest"):
        monkeypatch.setattr(
            hashlib,
            "file_digest",
            lambda stream, algorithm: hashlib.new(algorithm, stream.read()),
            raising=False,
        )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def completed_resume(tmp_path, *, changes=None, pinned_hash=None):
    arm_id = "k1_pretrained_consistency"
    source_sha = "a" * 64
    root = tmp_path / "retained" / arm_id
    root.mkdir(parents=True)
    payload = b"checkpoint bytes retained from the completed peer\n"
    (root / "last_checkpoint.pt").write_bytes(payload)
    manifest = {
        "status": "COMPLETE",
        "arm": arm_id,
        "source_sha256": source_sha,
        "next_epoch": 60,
        "artifacts": {"last_checkpoint.pt": hashlib.sha256(payload).hexdigest()},
    }
    manifest.update(changes or {})
    manifest_path = root / "stage_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    arm = {
        "arm_id": arm_id,
        "resume": {
            "source_sha256": source_sha,
            "next_epoch": 60,
            "manifest_sha256": pinned_hash or sha256(manifest_path),
        },
    }
    return root, arm


def test_completed_arm_is_retained_with_verified_artifacts(tmp_path):
    root, arm = completed_resume(tmp_path)
    destination = tmp_path / "published" / arm["arm_id"]

    assert runner.retain_completed_arm(root, destination, arm, stage_epochs=60)
    assert (destination / "last_checkpoint.pt").read_bytes() == (root / "last_checkpoint.pt").read_bytes()
    assert (destination / "stage_manifest.json").read_bytes() == (root / "stage_manifest.json").read_bytes()


@pytest.mark.parametrize(
    "changes",
    [
        {"status": "RUNNING"},
        {"next_epoch": 59},
        {"arm": "gptrans_g1_bond_local_ema999"},
        {"source_sha256": "b" * 64},
    ],
)
def test_completed_arm_rejects_inconsistent_manifest(tmp_path, changes):
    root, arm = completed_resume(tmp_path, changes=changes)
    destination = tmp_path / "published" / arm["arm_id"]

    with pytest.raises(RuntimeError, match="disagrees with the pinned resume"):
        runner.retain_completed_arm(root, destination, arm, stage_epochs=60)
    assert not destination.exists()


def test_completed_arm_rejects_changed_pinned_manifest_hash(tmp_path):
    root, arm = completed_resume(tmp_path, pinned_hash="0" * 64)
    destination = tmp_path / "published" / arm["arm_id"]

    with pytest.raises(RuntimeError, match="manifest changed"):
        runner.retain_completed_arm(root, destination, arm, stage_epochs=60)
    assert not destination.exists()


def test_incomplete_arm_is_left_for_continuation_without_creating_destination(tmp_path):
    destination = tmp_path / "published" / "k1_pretrained_consistency"
    arm = {"arm_id": "k1_pretrained_consistency", "resume": {"next_epoch": 53}}

    assert runner.retain_completed_arm(None, destination, arm, stage_epochs=60) is False
    assert not destination.exists()
