"""Discovery parity and confinement checks with synthetic metadata only."""

from dataclasses import fields
from pathlib import Path

import pytest

from molgap.research_memory.discovery import DiscoveredRecords, discover_records
from molgap.research_memory.paths import repo_local_path


PATTERNS = {
    "evidence": "experiments/**/v5_evidence.json",
    "trajectories": "experiments/**/trajectory.json",
    "costs": "experiments/**/costs/*.json",
    "roles": "experiments/**/roles/*.json",
    "traces": "experiments/**/trace_manifest.json",
    "ready": "experiments/**/handoff/READY_FOR_DESKTOP.json",
    "comparison_prelaunch": "experiments/**/comparison_readiness_prelaunch.json",
    "comparison_readiness": "experiments/**/comparison_readiness.json",
    "reference_bundles": "experiments/**/reference_bundle.json",
    "target_transform_assets": "experiments/**/target_transform.json",
}


def _write(root, relative):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


def _legacy_paths(root):
    result = {}
    for kind, pattern in PATTERNS.items():
        paths = (repo_local_path(root, path) for path in root.glob(pattern))
        result[kind] = tuple(sorted({path for path in paths if path.is_file()}))
    return DiscoveredRecords(**result)


def _symlink(link, target, *, directory=False):
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError:
        pytest.skip("Host does not permit native symlink creation")


def test_discovery_matches_original_selectors_with_one_recursive_inventory(tmp_path, monkeypatch):
    for pattern in PATTERNS.values():
        suffix = pattern.split("/**/")[1].replace("*.json", "C-example.json")
        _write(tmp_path, "experiments/z/nested/" + suffix)
        _write(tmp_path, "experiments/" + suffix)
    for relative in (
        "experiments/a/costs/trajectory.json", "experiments/a/roles/v5_evidence.json",
        "experiments/a/costs/nested/excluded.json", "experiments/a/roles/ignored.txt",
        "experiments/a/READY_FOR_DESKTOP.json", "experiments/a/handoff/nested/READY_FOR_DESKTOP.json",
        "elsewhere/trajectory.json", "experiments/a/TRAJECTORY.JSON",
    ):
        _write(tmp_path, relative)
    (tmp_path / "experiments/directory/trajectory.json").mkdir(parents=True)
    expected = _legacy_paths(tmp_path)
    glob = Path.glob
    calls = []

    def counted(path, pattern):
        calls.append((path, pattern))
        return glob(path, pattern)

    monkeypatch.setattr(Path, "glob", counted)
    assert discover_records(tmp_path) == expected
    assert calls == [(tmp_path, "experiments/**/*")]


def test_discovery_without_experiments_is_empty(tmp_path):
    result = discover_records(tmp_path)
    assert all(getattr(result, field.name) == () for field in fields(result))


def test_discovery_preserves_linked_suffix_selectors(tmp_path):
    for name in ("costs", "roles", "handoff", "other"):
        target = tmp_path / "retained" / name
        target.mkdir(parents=True)
        _write(target, "trajectory.json")
        _write(target, "READY_FOR_DESKTOP.json")
        _symlink(tmp_path / "experiments/q" / name, target, directory=True)
    assert discover_records(tmp_path) == _legacy_paths(tmp_path)


@pytest.mark.parametrize("relative", ["trajectory.json", "costs/C.json", "handoff/READY_FOR_DESKTOP.json"])
def test_discovery_rejects_matching_escape_links(tmp_path, relative):
    root = tmp_path / "repo"
    outside = _write(tmp_path, "outside.json")
    _symlink(root / "experiments/q" / relative, outside)
    with pytest.raises(ValueError, match="escapes repository"):
        _legacy_paths(root)
    with pytest.raises(ValueError, match="escapes repository"):
        discover_records(root)


@pytest.mark.parametrize("relative", ["trajectory.json", "costs/C.json"])
def test_discovery_preserves_dangling_link_behavior(tmp_path, relative):
    root = tmp_path / "repo"
    _symlink(root / "experiments/q" / relative, tmp_path / "missing.json")
    if relative.startswith("costs/"):
        with pytest.raises(ValueError, match="escapes repository"):
            _legacy_paths(root)
        with pytest.raises(ValueError, match="escapes repository"):
            discover_records(root)
    else:
        assert discover_records(root) == _legacy_paths(root)


def test_discovery_applies_only_verified_finalized_replacements(tmp_path, monkeypatch):
    from importlib import import_module
    finalize_module = import_module("molgap.research_memory.finalize")
    original = _write(tmp_path, "experiments/q/trajectory.json")
    final = _write(tmp_path, "experiments/q/rml_finalized/trajectory.json")
    other = _write(tmp_path, "experiments/a/trajectory.json")
    calls = []

    def receipt(directory):
        calls.append(directory)
        return {"replacements": {"trajectory.json": "experiments/q/trajectory.json"}}

    monkeypatch.setattr(finalize_module, "verified_receipt", receipt)
    assert discover_records(tmp_path).trajectories == tuple(sorted((final, other)))
    assert calls == [final.parent]
    assert original.exists()

    def invalid(directory):
        raise ValueError("finalized package changed")

    monkeypatch.setattr(finalize_module, "verified_receipt", invalid)
    with pytest.raises(ValueError, match="finalized package changed"):
        discover_records(tmp_path)
