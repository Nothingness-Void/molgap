"""Local Kaggle launch-budget wire checks; no platform submission is performed."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from molgap import kaggle_workflow


def _plan(**updates):
    value = {
        "account": "fixture-account",
        "kernel": "fixture-account/prenorm-review",
        "title": "Prenorm Review",
        "datasets": ["fixture-account/source"],
        "source_dataset": "fixture-account/source",
        "accelerator": "NvidiaTeslaT4",
    }
    value.update(updates)
    return value


def test_optional_budget_fields_leave_existing_kaggle_metadata_wire_unchanged():
    original = _plan()
    with_budget = _plan(allocation_wall_limit_seconds=14400, cleanup_reserve_seconds=240)

    metadata = kaggle_workflow._metadata(original)
    budget_metadata = kaggle_workflow._metadata(with_budget)

    assert metadata == budget_metadata
    assert "allocation_wall_limit_seconds" not in metadata
    assert "cleanup_reserve_seconds" not in metadata
    assert original == _plan()
    assert with_budget["allocation_wall_limit_seconds"] == 14400


def test_stage_inputs_transcribes_optional_budgets_to_frozen_launch(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    entry = repo / "platforms" / "kaggle" / "run_experiment.py"
    entry.parent.mkdir(parents=True)
    entry.write_text("EXPECTED_LAUNCH_SHA256 = None\n", encoding="utf-8")
    output = tmp_path / "staged"
    output.mkdir()
    monkeypatch.setattr(kaggle_workflow, "publish_immutable_bytes",
        lambda path, content: (Path(path).parent.mkdir(parents=True, exist_ok=True),
                               Path(path).write_bytes(content)))

    def stage_source_inputs(package, destination, initial_states):
        Path(destination).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr("molgap.experiment_package.stage_source_inputs", stage_source_inputs)
    plan = _plan(allocation_wall_limit_seconds=14400, cleanup_reserve_seconds=240)
    metadata = kaggle_workflow._metadata(plan)
    spec = SimpleNamespace(identity="spec-identity", to_dict=lambda: {"platform": {"device_count": 2}})

    staged = kaggle_workflow.stage_inputs(
        repo_root=repo,
        output=output,
        package=tmp_path / "package",
        manifest={"package_identity": "package-id", "archive_sha256": "a" * 64},
        spec=spec,
        platform_plan=plan,
        metadata=metadata,
        initial_states={},
        jobs=[],
    )

    assert staged["launch"]["allocation_wall_limit_seconds"] == 14400
    assert staged["launch"]["cleanup_reserve_seconds"] == 240
    assert "allocation_wall_limit_seconds" not in metadata
    assert "cleanup_reserve_seconds" not in metadata


@pytest.mark.parametrize("budget", [
    {"allocation_wall_limit_seconds": True},
    {"cleanup_reserve_seconds": False},
    {"allocation_wall_limit_seconds": 240, "cleanup_reserve_seconds": 240},
    {"allocation_wall_limit_seconds": 240, "cleanup_reserve_seconds": 241},
])
def test_metadata_rejects_boolean_budgets_and_nonpositive_remaining_runtime(budget):
    with pytest.raises(ValueError, match="budget|execution time"):
        kaggle_workflow._metadata(_plan(**budget))
