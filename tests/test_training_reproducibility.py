"""JSON transport regression checks; no accelerators or scientific inputs."""
import json
import os
from collections import UserDict
from pathlib import Path

import pytest

from molgap import training_reproducibility as owner
from molgap import experiment_preflight as preflight


@pytest.mark.parametrize("value", [{"rows": []}, UserDict({"epoch": 1}), [],
                                 [{"epoch": 1, "train_seconds": 2.5}]])
def test_atomic_json_roundtrip(tmp_path, value):
    path = tmp_path / "nested" / "execution_times.json"
    owner.atomic_json(path, value)
    expected = dict(value) if isinstance(value, UserDict) else value
    assert json.loads(path.read_text()) == expected
    assert path.read_text() == json.dumps(expected, indent=2) + "\n"
    assert not path.with_name("." + path.name + ".tmp").exists()


@pytest.mark.parametrize("value", ["text", None, 1, (("epoch", 1),),
                                 [{"unserializable": object()}]])
def test_bad_json_preserves_previous_file(tmp_path, value):
    path = tmp_path / "timings.json"
    owner.atomic_json(path, {"previous": True})
    before = path.read_bytes()
    with pytest.raises(TypeError):
        owner.atomic_json(path, value)
    assert path.read_bytes() == before
    assert not path.with_name("." + path.name + ".tmp").exists()


@pytest.mark.parametrize("writer", [owner.atomic_json, preflight._atomic], ids=["training", "preflight"])
def test_replace_failure_preserves_previous_file_and_cleans_temp(tmp_path, monkeypatch, writer):
    path = tmp_path / "timings.json"
    writer(path, {"previous": True})
    before = path.read_bytes()
    primary = OSError("synthetic replace failure")

    def failed_replace(*args):
        raise primary

    with monkeypatch.context() as failed:
        failed.setattr(owner.os, "replace", failed_replace)
        with pytest.raises(OSError, match="synthetic replace failure"):
            writer(path, [{"epoch": 1}])
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]

    if writer is preflight._atomic:
        cleanup = PermissionError("synthetic cleanup denied")

        def failed_unlink(*args, **kwargs):
            raise cleanup

        with monkeypatch.context() as failed:
            failed.setattr(owner.os, "replace", failed_replace)
            failed.setattr(Path, "unlink", failed_unlink)
            with pytest.raises(OSError, match="synthetic replace failure") as caught:
                writer(path, [{"epoch": 1}])
        assert caught.value is primary
        assert caught.value.cleanup_error is cleanup
        assert path.read_bytes() == before
        for temporary in tmp_path.glob(".preflight-*"):
            temporary.unlink()

    if os.name == "nt":
        # An ordinary live reader reproduces Windows' actual rename denial.
        # No retry may turn an unsuccessful publication into a reported success.
        with path.open("rb"):
            with pytest.raises(PermissionError) as denied:
                writer(path, [{"epoch": 1}])
        assert denied.value.winerror == 5
        assert path.read_bytes() == before
        assert list(tmp_path.iterdir()) == [path]
