"""Transport/dispatch tests only, without any model execution or platform call."""
import json
from pathlib import Path
import pytest
from molgap.gptrans_author_screen import mounted, restore_source_package
from molgap.training_reproducibility import sha256_file


def test_mount_requires_unique_actual_content(tmp_path):
    path = tmp_path / "first/manifest.json"
    path.parent.mkdir()
    path.write_text("synthetic")
    expected = sha256_file(path)
    assert mounted(tmp_path, "manifest.json", expected) == path
    other = tmp_path / "other/manifest.json"
    other.parent.mkdir()
    other.write_text("unrelated")
    assert mounted(tmp_path, "manifest.json", expected) == path
    other.write_text("synthetic")
    with pytest.raises(ValueError, match="found 2"):
        mounted(tmp_path, "manifest.json", expected)


def test_missing_mount_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="found 0"):
        mounted(tmp_path, "initial_state.pt", "a" * 64)


def test_source_restore_keeps_complete_package_and_checks_pin(tmp_path, monkeypatch):
    from molgap.experiment_package import SIDECARS
    source = tmp_path / "upload"
    source.mkdir()
    for name in SIDECARS:
        (source / ("source_payload.bin" if name == "source.tar.gz" else name)).write_text(name)
    def verify(directory):
        assert {p.name for p in directory.iterdir()} == SIDECARS
        assert (directory / "source.tar.gz").read_text() == "source.tar.gz"
        return {"archive_sha256": "a" * 64}
    monkeypatch.setattr("molgap.gptrans_author_screen.verify_experiment_source_package", verify)
    assert restore_source_package(source, tmp_path / "package", "a" * 64)["archive_sha256"] == "a" * 64
    with pytest.raises(ValueError, match="kernel pin"):
        restore_source_package(source, tmp_path / "wrong", "b" * 64)


def test_runtime_uses_native_v5_and_frozen_budget():
    import ast
    file = Path(__file__).parents[1] / "src/molgap/gptrans_author_screen.py"
    tree = ast.parse(file.read_text())
    assert not any(isinstance(node, ast.ClassDef) for node in tree.body)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(isinstance(node.func, ast.Name) and node.func.id == "gptrans_screen_arguments" for node in calls)
    assert any(isinstance(node.func, ast.Name) and node.func.id == "run_training" for node in calls)
    assert any(isinstance(node, ast.Attribute) and node.attr == "worker" for node in ast.walk(tree))
