"""Synthetic JSON writes only; no datasets, checkpoints or accelerator use."""
import copy
import json
import os
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import pytest

from molgap.training_reproducibility import atomic_json


def test_default_exact_bytes_and_explicit_false(tmp_path):
    value = {"z": [1, 1.25], "a": {"text": "\u00e9", "flag": True}}
    expected = (json.dumps(dict(value), indent=2) + "\n").replace("\n", os.linesep).encode("utf-8")
    default = tmp_path / "nested" / "default.json"
    explicit = tmp_path / "false.json"
    atomic_json(default, value)
    atomic_json(explicit, value, compact=False)
    assert default.read_bytes() == explicit.read_bytes() == expected
    assert not default.with_name(".default.json.tmp").exists()
    assert not explicit.with_name(".false.json.tmp").exists()


@pytest.mark.parametrize("value", [
    {},
    {"z": 1, "a": 2},
    {"unicode": "\u00e9\u4e2d\U0001f600", "escaped": "line\n\"\\"},
    {"floats": [0.0, -0.0, 1.2345678901234567, 1e-30, 1e30]},
    {"nested": [{"flag": True, "absent": None}, [], {}]},
])
def test_compact_exact_standard_encoder_and_logical_parity(tmp_path, value):
    path = tmp_path / "compact.json"
    before = copy.deepcopy(value)
    atomic_json(path, value, compact=True)
    encoded = json.dumps(dict(value), separators=(",", ":"), indent=None)
    assert path.read_bytes() == (encoded + os.linesep).encode("utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == json.loads(json.dumps(value, indent=2))
    assert value == before
    assert not path.with_name(".compact.json.tmp").exists()


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("value", [
    MappingProxyType({"z": (1, 2), "a": "value"}),
    [("z", 1), ("a", 2), ("z", 3)],
])
def test_dict_conversion_compatible_without_input_mutation(tmp_path, compact, value):
    before = dict(value)
    path = tmp_path / "mapping.json"
    atomic_json(path, value, compact=compact)
    expected = json.dumps(dict(value), separators=(",", ":"), indent=None) if compact else json.dumps(dict(value), indent=2)
    assert path.read_bytes() == (expected + "\n").replace("\n", os.linesep).encode("utf-8")
    assert dict(value) == before


@pytest.mark.parametrize("compact", [False, True])
def test_nan_and_infinity_encoder_defaults_unchanged(tmp_path, compact):
    value = {"nan": float("nan"), "positive": float("inf"), "negative": float("-inf")}
    path = tmp_path / "nonfinite.json"
    atomic_json(path, value, compact=compact)
    expected = json.dumps(value, separators=(",", ":"), indent=None) if compact else json.dumps(value, indent=2)
    assert path.read_bytes() == (expected + "\n").replace("\n", os.linesep).encode("utf-8")


@pytest.mark.parametrize("invalid", [0, 1, None, "true", "", 0.0, 1.0])
def test_compact_requires_exact_bool_before_filesystem_effects(tmp_path, invalid):
    path = tmp_path / "missing" / "output.json"
    with patch.object(Path, "mkdir") as mkdir, patch.object(Path, "write_text") as write, patch(
        "molgap.training_reproducibility.os.replace"
    ) as replace:
        with pytest.raises(ValueError, match="compact must be a bool"):
            atomic_json(path, {}, compact=invalid)
        mkdir.assert_not_called()
        write.assert_not_called()
        replace.assert_not_called()
    assert not path.parent.exists()


@pytest.mark.parametrize("compact", [False, True])
def test_compact_is_keyword_only(tmp_path, compact):
    path = tmp_path / "missing" / "output.json"
    with pytest.raises(TypeError):
        atomic_json(path, {}, compact)
    assert not path.parent.exists()


@pytest.mark.parametrize("compact", [False, True])
def test_atomic_replacement_and_rendering_error_retains_destination(tmp_path, compact):
    path = tmp_path / "output.json"
    temporary = path.with_name(".output.json.tmp")
    path.write_bytes(b"old destination\n")
    temporary.write_bytes(b"old temporary\n")
    with patch("molgap.training_reproducibility.os.replace", wraps=os.replace) as replace:
        with pytest.raises(TypeError):
            atomic_json(path, {"unencodable": object()}, compact=compact)
        replace.assert_not_called()
        assert path.read_bytes() == b"old destination\n"
        assert temporary.read_bytes() == b"old temporary\n"
        atomic_json(path, {"new": 1}, compact=compact)
        replace.assert_called_once_with(temporary, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"new": 1}
    assert not temporary.exists()
