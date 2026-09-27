"""Synthetic receipt binding only; no API calls or models."""
import json
from pathlib import Path

import pytest

from molgap import k1_joint_study_ops as ops


def _setup(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, "REPO_ROOT", tmp_path)
    root = tmp_path / ops.REL
    source = tmp_path / "source"
    records = tmp_path / "records"
    (root / "kaggle").mkdir(parents=True)
    (records / "remote_metadata").mkdir(parents=True)
    source.mkdir()
    metadata = {
        "id": "kaseichou/molgap-k1-joint-atom-s42", "id_no": 123,
        "is_private": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["kaseichou/molgap-k1-joint-atom-source"],
        "code_file": "run.py",
    }
    for directory in (root / "kaggle", records / "remote_metadata"):
        (directory / "kernel-metadata.json").write_text(json.dumps(metadata))
        (directory / "run.py").write_text("# frozen entry\n", encoding="utf-8")
    (records / "push_response.json").write_text(json.dumps({
        "ref": metadata["id"], "version_number": 1, "kernel_id": 123,
        "error": None,
    }))
    (source / "SOURCE_COMMIT.txt").write_text("a" * 40)
    (source / "SOURCE_ARCHIVE_SHA256.txt").write_text("b" * 64)
    return root, source, records


def test_confirmed_receipt_reuses_existing_ab_binding(tmp_path, monkeypatch):
    root, source, records = _setup(tmp_path, monkeypatch)
    result = ops.bind(source, records)
    binding = json.loads((root / "monitor_binding.json").read_text())
    assert result["submission_confirmed"] is True
    assert result["actual_hardware"] == "pending_startup_log"
    assert binding["controller_thread_id"] == ops.A
    assert binding["monitor_thread_id"] == ops.B
    assert binding["healthy_action"] == "SILENT"
    assert binding["automatic_training_successor_authorized"] is False
    assert len(binding["jobs"]) == 1
    assert Path(binding["jobs"][0]["control_state"]).is_file()
    with pytest.raises(FileExistsError):
        ops.bind(source, records)


def test_changed_remote_entry_cannot_bind_as_released(tmp_path, monkeypatch):
    root, source, records = _setup(tmp_path, monkeypatch)
    (records / "remote_metadata/run.py").write_text("# not frozen entry\n")
    with pytest.raises(RuntimeError, match="Remote entry differs"):
        ops.bind(source, records)
    assert not (root / "submission_receipt_v1.json").exists()
