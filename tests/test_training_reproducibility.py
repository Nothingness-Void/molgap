"""JSON transport regression checks; no accelerators or scientific inputs."""
import json
from collections import UserDict

import pytest

from molgap import training_reproducibility as owner


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


def test_replace_failure_preserves_previous_file_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "timings.json"
    owner.atomic_json(path, {"previous": True})
    before = path.read_bytes()

    def failed_replace(*args):
        raise OSError("synthetic replace failure")

    monkeypatch.setattr(owner.os, "replace", failed_replace)
    with pytest.raises(OSError, match="synthetic replace failure"):
        owner.atomic_json(path, [{"epoch": 1}])
    assert path.read_bytes() == before
    assert not path.with_name("." + path.name + ".tmp").exists()
