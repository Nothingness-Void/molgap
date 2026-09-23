"""Fail-closed source and checkpoint handling without running a model."""

import json

import pytest

from molgap.pcqm_k1_pair_token_scale_audit import _existing_chunks, _identity


def test_resumption_rejects_different_frozen_identity(tmp_path):
    (tmp_path / "progress.json").write_text(
        json.dumps({"identity": {**_identity(), "checkpoint_sha256": "wrong"},
                    "chunk_sha256": {}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="another input identity"):
        _existing_chunks(tmp_path)


def test_resumption_rejects_unrecorded_chunk(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "chunk_00.pt").write_bytes(b"interrupted-write")
    with pytest.raises(RuntimeError, match="Unrecorded audit chunks"):
        _existing_chunks(tmp_path)


def test_resumption_rejects_corrupt_recorded_chunk(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "chunk_00.pt").write_bytes(b"wrong")
    (tmp_path / "progress.json").write_text(
        json.dumps({"identity": _identity(), "chunk_sha256": {"chunk_00.pt": "0" * 64}}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Audit chunk changed"):
        _existing_chunks(tmp_path)
