"""Accepted-reference binding faults using retained metadata only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from molgap import family_pair_acceptance as owner
from molgap.experiment_family_workflow import RunContext
from molgap.experiment_spec import ExperimentSpec
from molgap.family_training_runner import read_recipe
from molgap.gptrans_campaign_release import BASE
from molgap.screen_policy import canonical_fingerprint


@pytest.fixture
def reference(monkeypatch, tmp_path):
    import molgap.experiment_package as packages
    root = Path(__file__).resolve().parents[1]
    campaign = json.loads((root / BASE / "campaign.json").read_bytes())
    first = ExperimentSpec.from_json((root / campaign["slots"][0]["path"]).read_text())
    record, recipe = read_recipe(root / campaign["arms"]["reference"]["contract"]["path"])
    arm = first.to_dict()["arms"][0]
    package = {"spec_identity": first.identity, "source_commit": "c" * 40,
        "archive_sha256": "d" * 64, "package_identity": "e" * 64}
    context = RunContext(first.to_dict()["experiment_id"], first.to_dict()["logical_run_id"],
        "reference", canonical_fingerprint(arm), first.identity, "gptrans_t", "2",
        package["source_commit"], package["archive_sha256"], package["package_identity"],
        campaign["arms"]["reference"]["contract"]["sha256"], "kaggle", "fixture", "fixture/kernel", "3")
    previous = {"format": "molgap-family-pair-acceptance-v1", "accepted": True,
        "acceptance_order": ["reference", "cycle"], "spec_identity": first.identity,
        "source_commit": package["source_commit"], "source_archive_sha256": package["archive_sha256"],
        "package_identity": package["package_identity"], "full_allocation": {
            "wall_seconds": 100., "allocated_device_seconds": 200.}, "arms": {"reference": {
                "accepted": True, "role": "reference", "recipe_identity": recipe.identity,
                "context": context.to_dict(), "inspection": {"status": "MECHANICALLY_VERIFIED",
                    "observed": {"progress": {"status": "complete", "optimizer_steps": recipe.optimizer_steps,
                        "sample_presentations": record["acceptance_requirements"]["sample_presentations"],
                        "completed_passes": 60, "next_batch": 0}}}}}}
    monkeypatch.setattr(owner, "verify_experiment_source_package", lambda path: deepcopy(package))
    monkeypatch.setattr(packages, "read_packaged_text", lambda path, relative: (
        root / relative).read_text(encoding="utf-8"))
    return previous, {"source_commit": package["source_commit"]}, tmp_path


def validate(case):
    previous, package, path = case
    return owner._validate_campaign_reference(previous, package_dir=path, package=package,
        reference_package_dir=path)


def test_reference_binds_exact_frozen_slot1_spec_recipe_source_and_complete_endpoint(reference):
    arm, cost = validate(reference)
    assert arm["role"] == "reference" and cost == 200.


@pytest.mark.parametrize("field", ["spec_identity", "source_commit", "source_archive_sha256", "package_identity"])
def test_reference_rejects_wrong_source_or_spec(reference, field):
    reference[0][field] = "f" * (40 if field == "source_commit" else 64)
    with pytest.raises(ValueError):
        validate(reference)


@pytest.mark.parametrize("field", ["recipe_identity", "arm_identity", "training_recipe_sha256", "family_version", "logical_run_id", "platform"])
def test_reference_rejects_wrong_baseline_identity(reference, field):
    arm = reference[0]["arms"]["reference"]
    if field == "recipe_identity":
        arm[field] = "f" * 64
    else:
        arm["context"][field] = "wrong" if field in {"family_version", "logical_run_id", "platform"} else "f" * 64
    with pytest.raises(ValueError):
        validate(reference)


@pytest.mark.parametrize("cost", [None, True, -1., float("nan"), float("inf"), 16 * 3600 + 1])
def test_reference_cost_cannot_bypass_campaign_budget(reference, cost):
    reference[0]["full_allocation"]["allocated_device_seconds"] = cost
    with pytest.raises(ValueError, match="cost is invalid"):
        validate(reference)


def test_reference_rejects_partial_ten_pass_endpoint(reference):
    reference[0]["arms"]["reference"]["inspection"]["observed"]["progress"]["completed_passes"] = 10
    with pytest.raises(ValueError, match="complete frozen scientific endpoint"):
        validate(reference)


def test_reference_requires_actual_original_package(reference):
    previous, package, path = reference
    with pytest.raises(ValueError, match="slot1 source package"):
        owner._validate_campaign_reference(previous, package_dir=path, package=package,
            reference_package_dir=None)


def test_reference_package_cannot_use_a_different_frozen_recipe(reference, monkeypatch):
    import molgap.experiment_package as packages
    previous, package, path = reference
    read = packages.read_packaged_text
    original = path / "original"
    def changed(directory, relative):
        text = read(directory, relative)
        return text + " " if Path(directory) == original and relative.endswith("reference/contract.json") else text
    monkeypatch.setattr(packages, "read_packaged_text", changed)
    with pytest.raises(ValueError, match="Spec/recipe"):
        owner._validate_campaign_reference(previous, package_dir=path, package=package,
            reference_package_dir=original)
