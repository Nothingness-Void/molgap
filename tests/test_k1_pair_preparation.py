"""Shared draft bindings on fixed templates; no model execution or publication."""
from pathlib import Path

import pytest

from molgap import k1_pair_preparation as prep
from molgap.experiment_execution import training_adapter
from molgap.experiment_spec import ExperimentSpec


ROOT = Path(__file__).resolve().parents[1]
SEED43_SHA = "a" * 64  # Synthetic tensor pin, not scientific evidence.


@pytest.fixture
def draft_root(tmp_path, monkeypatch):
    for ref in (
        "experiments/pcqm_k1_slot_width96/reference_binding/original_reference_arm.json",
        "experiments/pcqm_k1_slot_width96/reference_binding/target_manifest.json",
        "experiments/pcqm_k1_fusion_distillation/training_recipe_distill_weak.json",
        "src/molgap/k1_screen_training.py", "src/molgap/k1_clean_second.py",
        "docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md",
    ):
        path = tmp_path / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / ref).read_bytes())
    for name in ("seed43", "clean_second"):
        path = tmp_path / "experiments" / name / "protocol.md"
        path.parent.mkdir(parents=True)
        path.write_text("Synthetic unpublished protocol", encoding="utf-8")
    monkeypatch.setattr(prep, "inspect_frozen_state_artifact", lambda path, **kw: {
        "state_sha256": kw["expected_state_sha256"], "file_sha256": "f" * 64, "device": "cpu"})
    return tmp_path


@pytest.mark.parametrize("seed,candidate,addon,family,mode", [
    (43, "single", None, "k1-t4-cost-quality", "reference"),
    (42, "clean_second", "k1_mean2_clean_second", "k1-clean-second-view", "mean2_clean_second"),
])
def test_fresh_pair_real_spec_and_all_record_bindings(draft_root, seed, candidate, addon, family, mode):
    slug = "molgap-fresh-seed43" if seed == 43 else "molgap-fresh-clean-second"
    rel = "experiments/seed43" if seed == 43 else "experiments/clean_second"
    commit = "1" * 40
    prior = ["prior-negative-cost-pair", "prior-frozen-bn-diagnostic"]
    parents = ["TB-prior-cost-pair", "TB-prior-bn-diagnostic"]
    records = prep.build_inputs(draft_root, source_commit=commit,
        initial_state=draft_root / "missing-parent-owned-state.pt",
        relative_dir=rel, logical_run_id=slug, initialization_seed=seed,
        initialization_sha256=SEED43_SHA if seed == 43 else prep.STATE_SHA,
        candidate_addon=addon, candidate_arm_id=candidate, family_id=family,
        candidate_addon_source="src/molgap/k1_clean_second.py" if addon else "src/molgap/k1_screen_training.py",
        allocation_wall_limit_seconds=12600, policy_id=slug + "-policy",
        question="Fresh pair question", supporting_evidence_ids=prior,
        parent_trajectory_ids=parents,
        hypothesis_overrides={"cheapest_falsifier": "Parent-owned fresh bounded pair"})
    # Validate the full Spec, including initialization/sampler/paired identity.
    spec = ExperimentSpec(records["experiment_spec.json"])
    reference, changed = spec.to_dict()["arms"]
    assert reference["arm_id"] == "mean2" and changed["arm_id"] == candidate
    assert training_adapter(reference).mode(reference) == "mean2"
    assert training_adapter(changed).mode(changed) == mode
    assert reference["initialization"] == changed["initialization"]
    assert reference["initialization"]["seed"] == seed
    assert reference["training"]["sampler"] == changed["training"]["sampler"]
    assert changed["training"]["sampler"]["name"] == f"seed{seed}-python-epoch-shuffle-v4"
    assert spec.to_dict()["prospective"]["same_run_replay"] == {
        "reference_arm_id": "mean2", "candidate_arm_ids": [candidate]}
    expected = prep.read(draft_root / "experiments/pcqm_k1_fusion_distillation/training_recipe_distill_weak.json")["acceptance_requirements"]
    for binding in spec.to_dict()["prospective"]["arms"]:
        arm_id = binding["arm_id"]
        recipe = records[f"training_recipe_{arm_id}.json"]
        assert recipe["allocation_wall_limit_seconds"] == 12600
        assert recipe["acceptance_requirements"] == expected
        assert recipe["training_recipe"]["seed"] == seed
        plan = records[f"training_plan_{arm_id}.json"]
        trajectory, cost = plan["trajectory"], plan["costs"][0]
        assert trajectory["trajectory_id"] == f"TB-{slug}-{arm_id}"
        assert trajectory["family_id"] == family
        assert trajectory["question"] == "Fresh pair question"
        assert trajectory["hypothesis"]["supporting_evidence_ids"] == prior
        assert trajectory["state_at_start"]["parent_trajectory_ids"] == parents
        assert trajectory["state_at_start"]["reference_ids"] == []
        assert trajectory["state_at_start"]["source_commit"] == commit
        assert plan["decision_state"]["source_commit"] == commit
        action = trajectory["actions"][0]
        assert action["source_commit"] == commit
        assert action["run_ids"] == [cost["run_id"]] == [f"{slug}:{arm_id}:downstream"]
        assert action["attempt_ids"] == [cost["attempt_id"]] == [slug + "-v1"]
        assert cost["measurement"]["device_hours"]["value"] == 3.5
        assert cost["measurement"]["wall_hours"]["value"] == 3.5
        assert binding["output"] == rel + "/rml/" + arm_id
        assert binding["plan_spec_ref"] == rel + f"/training_plan_{arm_id}.json"
        assert binding["plan_spec_sha256"] == prep.canonical_fingerprint(plan)
    assert records["workflow_plan.json"]["spec_identity"] == spec.identity
    assert records["family_acceptance_plan.json"]["spec_identity"] == spec.identity
    assert records["preparation_report.json"]["training_authorized"] is False
    assert records["policy.json"]["status"] == "draft"
    assert not list(draft_root.rglob("trajectory.json"))
    if addon:
        assert changed["addons"][0]["source_sha256"] == prep.normalized_source_sha256(draft_root / "src/molgap/k1_clean_second.py")
        assert changed["training"]["objective"]["sha256"] == prep.canonical_fingerprint({
            "base_loss": "normalized-gap-l1", "forward_passes": 2,
            "reduction": "arithmetic-mean-of-two-normalized-L1-losses",
            "optimizer_updates_per_batch": 1, "consistency_penalty": False,
            "dropout_enabled_per_forward": [True, False],
            "batchnorm_training_per_forward": [True, True], "batchnorm_updates_per_batch": 2})


@pytest.mark.parametrize("overrides,match", [
    ({"relative_dir": prep.REL}, "both a fresh"),
    ({"logical_run_id": prep.RUN}, "both a fresh"),
    ({"candidate_arm_id": "mean2"}, "distinct"),
    ({"relative_dir": "experiments/../escape"}, "confined"),
    ({"approved_release": {"binding": {}}}, "remain drafts"),
])
def test_fresh_identity_guards(draft_root, overrides, match):
    kwargs = {"relative_dir": "experiments/seed43", "logical_run_id": "molgap-fresh-seed43"}
    kwargs.update(overrides)
    with pytest.raises(ValueError, match=match):
        prep.build_inputs(draft_root, source_commit="1" * 40, initial_state=Path("missing.pt"), **kwargs)


def test_existing_candidate_rml_is_never_reused(draft_root):
    (draft_root / "experiments/clean_second/rml/clean_second").mkdir(parents=True)
    with pytest.raises(ValueError, match="cannot replace"):
        prep.build_inputs(draft_root, source_commit="1" * 40, initial_state=Path("missing.pt"),
            relative_dir="experiments/clean_second", logical_run_id="molgap-fresh-clean-second",
            candidate_arm_id="clean_second")
