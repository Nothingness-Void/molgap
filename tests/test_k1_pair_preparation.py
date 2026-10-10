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
    for name in ("seed43", "clean_second", "pcqm_k1_t4_cost_quality"):
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


@pytest.mark.parametrize("limit", [None, 14400])
def test_single_forward_reference_and_fused_candidate_bindings(draft_root, limit):
    run = "molgap-fused-layout-100k"
    rel = "experiments/seed43"
    state = draft_root / "parent-owned-initial-state.pt"
    inspections = []

    def inspect(path, **kwargs):
        inspections.append((path, kwargs))
        return {"state_sha256": kwargs["expected_state_sha256"], "device": "cpu"}

    objective = {"base_loss": "normalized-gap-l1", "forward_passes": 1,
                 "optimizer_updates_per_batch": 1, "consistency_penalty": False,
                 "batchnorm_updates_per_batch": 1}
    records = prep.build_inputs(draft_root, source_commit="1" * 40, initial_state=state,
        relative_dir=rel, logical_run_id=run, reference_arm_id="reference", reference_addon=None,
        candidate_arm_id="fused_layout", candidate_addon="k1_fused_layout",
        candidate_objective=objective, allocation_wall_limit_seconds=limit,
        account="nothingnessvoid", kernel="nothingnessvoid/fused-layout-kernel",
        title="K1 Fused Layout", source_dataset="nothingnessvoid/fused-source",
        dataset="nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1",
        policy_id="fused-policy", changed_mechanism="Fused layout only; one normalized L1 forward.",
        _state_inspector=inspect)
    assert inspections == [(state, {"expected_state_sha256": prep.STATE_SHA})]
    assert records["preparation_report.json"]["initial_state"]["device"] == "cpu"
    spec = ExperimentSpec(records["experiment_spec.json"])
    reference, candidate = spec.to_dict()["arms"]
    original = prep.read(draft_root / "experiments/pcqm_k1_slot_width96/reference_binding/original_reference_arm.json")
    assert reference["addons"] == []
    assert reference["addon_semantics"] == "baseline"
    assert reference["training"]["objective"] == original["training"]["objective"]
    assert candidate["training"]["objective"]["sha256"] == prep.canonical_fingerprint(objective)
    assert training_adapter(reference).mode(reference) == "reference"
    assert training_adapter(candidate).mode(candidate) == "fused_layout"
    assert candidate["addons"] == [{"name": "k1_fused_layout", "version": "1", "config": {},
        "source_sha256": prep.normalized_source_sha256(draft_root / "src/molgap/k1_screen_training.py")}]
    assert reference["initialization"] == candidate["initialization"]
    assert reference["training"]["sampler"] == candidate["training"]["sampler"]
    assert spec.to_dict()["prospective"]["same_run_replay"] == {
        "reference_arm_id": "reference", "candidate_arm_ids": ["fused_layout"]}
    for binding in spec.to_dict()["prospective"]["arms"]:
        arm_id = binding["arm_id"]
        recipe = records[f"training_recipe_{arm_id}.json"]
        if limit is None:
            assert "allocation_wall_limit_seconds" not in recipe
        else:
            assert recipe["allocation_wall_limit_seconds"] == limit
        assert recipe["acceptance_requirements"]["epochs"] == 40
        plan = records[f"training_plan_{arm_id}.json"]
        trajectory, cost = plan["trajectory"], plan["costs"][0]
        assert trajectory["record_mode"] == "prospective"
        assert trajectory["trajectory_id"] == binding["trajectory_id"] == f"TB-{run}-{arm_id}"
        assert trajectory["state_at_start"]["source_config_identity"] == prep.canonical_fingerprint(
            reference if arm_id == "reference" else candidate)
        assert trajectory["actions"][0]["run_ids"] == [cost["run_id"]] == [f"{run}:{arm_id}:downstream"]
        assert trajectory["actions"][0]["attempt_ids"] == [cost["attempt_id"]] == [run + "-v1"]
        assert trajectory["actions"][0]["cost_event_ids"] == [cost["cost_event_id"]]
        assert trajectory["hypothesis"]["expected_native_cost_ref"] == cost["cost_event_id"]
        for metric in ("device_hours", "wall_hours"):
            assert cost["measurement"][metric] == {
                "status": "measurement_missing" if limit is None else "estimated",
                "value": None if limit is None else 4.0}
        assert binding["plan_spec_sha256"] == prep.canonical_fingerprint(plan)
        assert binding["output"] == rel + "/rml/" + arm_id
        assert plan["decision_state"]["policy_id"] == "fused-policy"
        assert plan["decision_state"]["chosen_action"] == "PREPARE_ONLY"
    workflow = records["workflow_plan.json"]
    assert workflow["spec_identity"] == records["family_acceptance_plan.json"]["spec_identity"] == spec.identity
    assert workflow["kaggle"] == {"account": "nothingnessvoid", "kernel": "nothingnessvoid/fused-layout-kernel",
        "title": "K1 Fused Layout", "source_dataset": "nothingnessvoid/fused-source",
        "datasets": ["nothingnessvoid/fused-source", "nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1"],
        "accelerator": "NvidiaTeslaT4"}
    assert records["role_plan.json"]["dataset"] == workflow["kaggle"]["datasets"][1]
    assert records["role_plan.json"]["actual_consumption"] == "pending"
    assert all(arm["initial_state"] == state.as_posix() for arm in workflow["arms"])
    assert [arm["device"] for arm in workflow["arms"]] == [0, 1]
    assert records["policy.json"]["status"] == "draft"
    assert records["preparation_report.json"]["training_authorized"] is False
    assert "Nones" not in str(records["preparation_report.json"])


def test_account_derived_source_and_kernel_defaults(draft_root):
    run = "molgap-account-defaults"
    records = prep.build_inputs(draft_root, source_commit="1" * 40, initial_state=Path("missing.pt"),
        relative_dir="experiments/seed43", logical_run_id=run, account="nothingnessvoid",
        reference_arm_id="reference", reference_addon=None)
    kaggle = records["workflow_plan.json"]["kaggle"]
    assert kaggle["source_dataset"] == "nothingnessvoid/" + run + "-source"
    assert kaggle["kernel"] == "nothingnessvoid/" + run
    assert kaggle["title"] == run


@pytest.mark.parametrize("overrides", [
    {"reference_arm_id": "reference"}, {"reference_addon": None},
    {"account": "nothingnessvoid"}, {"kernel": "nothingnessvoid/new"},
    {"title": "New title"}, {"dataset": "nothingnessvoid/fixed"},
    {"allocation_wall_limit_seconds": None},
])
def test_new_options_cannot_rewrite_frozen_legacy_records(draft_root, overrides):
    with pytest.raises(ValueError, match="fresh identities"):
        prep.build_inputs(draft_root, source_commit="1" * 40, initial_state=Path("missing.pt"), **overrides)


@pytest.mark.parametrize("reference,candidate", [("same", "same"), ("../reference", "single"),
                                                 ("reference", "../candidate"), ("", "single")])
def test_both_arm_ids_are_confined(draft_root, reference, candidate):
    with pytest.raises(ValueError, match="distinct and path-safe"):
        prep.build_inputs(draft_root, source_commit="1" * 40, initial_state=Path("missing.pt"),
            relative_dir="experiments/seed43", logical_run_id="molgap-fresh-pair",
            reference_arm_id=reference, candidate_arm_id=candidate)


def test_fused_addon_cannot_inherit_two_forward_objective(draft_root):
    with pytest.raises(ValueError, match="explicit candidate_objective"):
        prep.build_inputs(draft_root, source_commit="1" * 40, initial_state=Path("missing.pt"),
            relative_dir="experiments/seed43", logical_run_id="molgap-fresh-pair",
            reference_arm_id="reference", reference_addon=None,
            candidate_arm_id="fused_layout", candidate_addon="k1_fused_layout")


def test_legacy_defaults_preserve_mean2_and_nvoid912(draft_root):
    records = prep.build_inputs(draft_root, source_commit="1" * 40, initial_state=Path("missing.pt"))
    spec = ExperimentSpec(records["experiment_spec.json"])
    reference, candidate = spec.to_dict()["arms"]
    assert [reference["arm_id"], candidate["arm_id"]] == ["mean2", "single"]
    assert training_adapter(reference).mode(reference) == "mean2"
    assert training_adapter(candidate).mode(candidate) == "reference"
    assert reference["training"]["objective"]["sha256"] == prep.canonical_fingerprint({
        "base_loss": "normalized-gap-l1", "forward_passes": 2,
        "reduction": "arithmetic-mean-of-two-dropout-bearing-L1-losses",
        "optimizer_updates_per_batch": 1, "consistency_penalty": False,
        "batchnorm_updates_per_batch": 2})
    original = prep.read(draft_root / "experiments/pcqm_k1_slot_width96/reference_binding/original_reference_arm.json")
    assert candidate["training"]["objective"] == original["training"]["objective"]
    assert records["workflow_plan.json"]["kaggle"] == {
        "account": "nvoid912", "kernel": "nvoid912/" + prep.RUN, "title": prep.RUN,
        "datasets": ["nvoid912/molgap-k1-t4-cost-quality-source-s42-v1", "nvoid912/pcqm4mv2-ogb-fixed-100k-v1"],
        "source_dataset": "nvoid912/molgap-k1-t4-cost-quality-source-s42-v1", "accelerator": "NvidiaTeslaT4"}
    for arm_id in ("mean2", "single"):
        assert records[f"training_recipe_{arm_id}.json"]["allocation_wall_limit_seconds"] == 14400
        plan = records[f"training_plan_{arm_id}.json"]
        assert plan["trajectory"]["trajectory_id"] == f"TB-k1-t4-cost-quality-{arm_id}-100k-s42-v1"
        assert plan["costs"][0]["run_id"] == prep.RUN + ":" + arm_id
        assert plan["costs"][0]["measurement"]["device_hours"] == {"status": "estimated", "value": 4.0}
