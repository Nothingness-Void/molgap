"""Declared allocation limits survive platform staging without scientific changes."""
from pathlib import Path
from types import SimpleNamespace
import pytest
from molgap import kaggle_workflow as owner


def plan():
    return {"account": "nothingnessvoid", "kernel": "nothingnessvoid/molgap-norm-test",
            "title": "MolGap Norm Test", "datasets": ["nothingnessvoid/norm-source"],
            "source_dataset": "nothingnessvoid/norm-source", "accelerator": "NvidiaTeslaT4",
            "allocation_wall_limit_seconds": 28800, "cleanup_reserve_seconds": 180}


def test_launch_freezes_explicit_eight_hour_budget(tmp_path, monkeypatch):
    monkeypatch.setattr("molgap.experiment_package.stage_source_inputs", lambda *a, **k: None)
    monkeypatch.setattr(owner, "read_packaged_text", lambda *a: "EXPECTED_LAUNCH_SHA256 = None\n")
    spec = SimpleNamespace(identity="a" * 64, to_dict=lambda: {"platform": {"device_count": 2}})
    staged = owner.stage_inputs(repo_root=tmp_path, output=tmp_path, package=tmp_path / "package",
        manifest={"package_identity": "b" * 64, "archive_sha256": "c" * 64}, spec=spec,
        platform_plan=plan(), metadata=owner._metadata(plan()), initial_states={}, jobs=[])
    assert staged["launch"]["allocation_wall_limit_seconds"] == 28800
    assert staged["launch"]["cleanup_reserve_seconds"] == 180


@pytest.mark.parametrize("value", [True, 0, 43201, 28800.0])
def test_invalid_budget_rejected_before_staging(value):
    value_plan = plan()
    value_plan["allocation_wall_limit_seconds"] = value
    with pytest.raises(ValueError, match="allocation wall ceiling"):
        owner._metadata(value_plan)
