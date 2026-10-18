"""Declaration-only retry checks; execution belongs to the assigned Luna worker."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from molgap import modern2d_training_prepare as owner
from molgap import modern2d_training_release as release
from molgap.experiment_spec import ExperimentSpec
from molgap.research_memory.trace import file_digest
from test_modern2d_training_prepare import COMMIT, _freeze, _read, _write, case


RETRY_BASE = owner.BASE + "/retry_v2"
POLICY = "research_memory/policies/modern2d-formal-100k-reference-screen-retry-v2.v2.json"
REAL_RETRY_BUDGET = (Path(__file__).resolve().parents[1]
                     / "experiments/modern2d_100k_formal_screen/results/retry_budget.json")


def _read_real_retry_budget():
    return json.loads(REAL_RETRY_BUDGET.read_text(encoding="utf-8"))


def test_real_retry_budget_reconciles_stable_snapshot_and_rounds_once():
    record = _read_real_retry_budget()
    observations = record["quota_observations"]
    before = observations["before"]["quota"]["gpuQuota"]["timeUsed"]
    after = observations["after"]["quota"]["gpuQuota"]["timeUsed"]

    assert before["seconds"] + before["nanos"] / 1e9 == pytest.approx(459.068)
    assert after["seconds"] + after["nanos"] / 1e9 == pytest.approx(535.912)
    assert after["seconds"] + after["nanos"] / 1e9 - (
        before["seconds"] + before["nanos"] / 1e9
    ) == pytest.approx(76.844)
    assert record["prior_debit_wall_seconds"] == 77
    assert record["prior_debit_allocated_t4_hours"] == pytest.approx(2 * 77 / 3600)
    assert release.validate_retry_budget(record) == 77


@pytest.mark.parametrize("mutation", ["settled-snapshot", "quota-reset", "reserved"])
def test_real_retry_budget_rejects_tampered_reset_or_reserved_metadata(mutation):
    record = _read_real_retry_budget()
    if mutation == "settled-snapshot":
        record["quota_observations"]["after"]["quota"]["gpuQuota"]["timeUsed"]["seconds"] += 1
    elif mutation == "quota-reset":
        record["quota_observations"]["after"]["quota"]["quotaRefreshTime"] = "2026-10-25T00:00:00Z"
    else:
        record["quota_observations"]["confirmed"]["quota"]["gpuQuota"]["timeReserved"]["seconds"] = 1

    with pytest.raises(ValueError):
        release.validate_retry_budget(record)


@pytest.fixture
def retry_case(case, monkeypatch):
    _freeze(case)
    submission = case.root / owner.BASE / "results/submission_2026-10-18.json"
    _write(submission, {"kernel_id": 139364627, "platform_version": 1,
        "source_commit": "d67e30894089ffaa40569b93ca9c5130cfbb87ba"})
    budget = case.root / owner.BASE / "results/retry_budget.json"
    # Accounting validation is owned by release; this fixture verifies the
    # declaration seam forwards its pinned bytes without inventing a schema.
    _write(budget, {"format": "molgap-modern2d-retry-budget-v1",
        "aggregate_t4_hours_ceiling": 9, "prior_debit_wall_seconds": 76.844,
        "prior_debit_allocated_t4_hours": 2 * 76.844 / 3600,
        "prior_kernel_id": 139364627, "prior_kernel_version": 1,
        "retry_authorized": True, "accounting_scope": "all-account-quota-increments-times-two"})
    case.budget = {"path": budget.relative_to(case.root).as_posix(), "sha256": file_digest(budget)}
    case.validated = []

    def validate(record):
        case.validated.append(deepcopy(record))
        if record.get("retry_authorized") is not True:
            raise ValueError("retry not authorized")

    monkeypatch.setattr(release, "validate_retry_budget", validate, raising=False)
    case.snapshot_records = []

    def snapshot(root, records):
        case.snapshot_records.append(tuple(records))
        return deepcopy(case.snapshot)

    monkeypatch.setattr(owner, "_source_snapshot", snapshot)
    return case


def _retry(case, **overrides):
    options = dict(authorization="One retry after cost reconciliation within the same aggregate9 T4h",
        budget_reconciliation=case.budget)
    options.update(overrides)
    return owner.freeze_formal_retry(case.root, **options)


def _parent_bytes(case):
    return {p: p.read_bytes() for p in (case.root / owner.BASE).rglob("*") if p.is_file()}


def test_retry_keeps_exact_science_and_initial_paths_with_new_bindings(retry_case):
    c = retry_case
    original = _parent_bytes(c)
    result = _retry(c)
    root = c.root / RETRY_BASE
    old_spec = _read(c.root / owner.BASE / "experiment_spec.json")
    spec = ExperimentSpec(_read(root / "experiment_spec.json"))
    plan = _read(root / "acceptance_plan.json")
    authority = _read(root / "authority.json")
    workflow = _read(root / "workflow.json")
    old_workflow = _read(c.root / owner.BASE / "workflow.json")
    assert spec.to_dict()["logical_run_id"] == "modern2d-formal-100k-s42-v2"
    assert spec.to_dict()["arms"] == old_spec["arms"]
    assert spec.identity != old_workflow["spec_identity"]
    assert authority["authorization"].startswith("One retry after cost reconciliation")
    assert authority["source_commit"] == COMMIT
    assert authority["logical_run_id"] == "modern2d-formal-100k-s42-v2"
    assert authority["allocated_t4_hours_ceiling"] == authority["gpu"]["allocated_t4_hours_ceiling"] == 9
    assert authority["gpu"]["wall_ceiling_seconds"] == 16200
    assert authority["budget_reconciliation_source"] == c.budget
    assert authority["prior_attempt"]["logical_run_id"] == owner.RUN
    assert authority["prior_attempt"]["spec_identity"] == old_workflow["spec_identity"]
    for pin in (authority["prior_attempt"][k] for k in ("authority", "source_identity", "submission")):
        assert file_digest(c.root / pin["path"]) == pin["sha256"]
    assert workflow["kaggle"]["kernel"] == "kaseichou/molgap-modern2d-formal-100k-v2"
    assert workflow["kaggle"]["source_dataset"] == "kaseichou/molgap-modern2d-formal-source-v2"
    assert workflow["kaggle"]["datasets"] == ["kaseichou/molgap-modern2d-formal-source-v2", owner.DATASET]
    assert [j["initial_state"] for j in workflow["arms"]] == [j["initial_state"] for j in old_workflow["arms"]]
    assert _read(root / "experiment_input.json")["attempt_id"] == "v2"
    assert _read(root / "experiment_input.json")["machine_profile"] == owner.MACHINE_PROFILE
    assert set(plan) == {"format", "spec_identity", "authority", "inputs", "arms", "budget_reconciliation"}
    assert plan["format"] == owner.PLAN_FORMAT
    pin = plan["budget_reconciliation"]
    assert pin == {"path": RETRY_BASE + "/budget_reconciliation.json", "sha256": c.budget["sha256"]}
    assert (c.root / pin["path"]).read_bytes() == (c.root / c.budget["path"]).read_bytes()
    assert pin["path"] in workflow["source_records"]
    assert len(c.validated) == 1
    for arm in owner.ARMS:
        assert (root / arm / "contract.json").read_bytes() == (c.root / owner.BASE / arm / "contract.json").read_bytes()
    assert (root / "training_input.json").read_bytes() == (c.root / owner.BASE / "training_input.json").read_bytes()
    before = _read(c.root / owner.BASE / "training_inputs.json")
    after = _read(root / "training_inputs.json")
    assert after["models"] == before["models"]
    assert after["authority"] == plan["authority"] and after["authority"] != before["authority"]
    assert (root / "development_target_proof.json").read_bytes() == (c.root / owner.BASE / "development_target_proof.json").read_bytes()
    assert all(p.read_bytes() == raw for p, raw in original.items())
    assert result["comparison_scope"] == "CONTEXT_ONLY" and result["scientific_status"] == "INCONCLUSIVE"
    assert result["noReplay"] and not result["prepared"] and not result["submitted"] and not result["rml_published"]


def test_new_policy_trajectories_and_actions_pin_repaired_source(retry_case):
    c = retry_case
    _retry(c)
    policy = _read(c.root / POLICY)
    assert policy["version"] == "v2"
    for arm in owner.ARMS:
        proposal = _read(c.root / RETRY_BASE / arm / "plan_input.json")
        trajectory = proposal["trajectory"]
        state = trajectory["state_at_start"]
        assert trajectory["trajectory_id"] == "TC-modern2d-formal-100k-" + arm + "-s42-v2"
        assert state["reference_ids"] == proposal["decision_state"]["active_reference_ids"] == []
        assert state["prior_trajectory_ids"] == ["TC-modern2d-formal-100k-" + arm + "-s42-v1"]
        assert proposal["decision_state"]["policy_version"] == "v2"
        assert proposal["decision_state"]["policy_id"] == policy["policy_id"]
        assert POLICY in state["contract_refs"]
        assert state["source_commit"] == trajectory["actions"][0]["source_commit"] == COMMIT
        assert trajectory["actions"][0]["attempt_ids"] == [arm + "-v2"]


def test_no_new_role_event_paths_and_mutable_parent_status_not_pinned(retry_case):
    c = retry_case
    _retry(c)
    workflow = _read(c.root / RETRY_BASE / "workflow.json")
    role_plan = _read(c.root / RETRY_BASE / "role_plan.json")
    for role, pin in role_plan["role_manifests"].items():
        assert pin["path"] == RETRY_BASE + "/input_role_manifests/" + role + ".json"
        assert file_digest(c.root / pin["path"]) == pin["sha256"]
    assert not (c.root / RETRY_BASE / "roles").exists()
    assert all("/roles/" not in name for name in workflow["source_records"])
    assert owner.BASE + "/STATUS.md" not in workflow["source_records"]
    assert len(c.snapshot_records) == 2
    for records in c.snapshot_records:
        assert owner.BASE + "/STATUS.md" not in records
        assert c.budget["path"] in records
        assert owner.BASE + "/results/submission_2026-10-18.json" in records
        assert owner.BASE + "/ppgt_author_v2/contract.json" in records


def test_historical_v1_roles_directory_is_read_without_rewriting_it(retry_case):
    c = retry_case
    parent = c.root / owner.BASE
    # The original V1 freeze predates the non-RML manifest directory name.
    (parent / "input_role_manifests").rename(parent / "roles")
    role_plan = _read(parent / "role_plan.json")
    for pin in role_plan["role_manifests"].values():
        pin["path"] = pin["path"].replace("/input_role_manifests/", "/roles/")
    _write(parent / "role_plan.json", role_plan)
    workflow = _read(parent / "workflow.json")
    workflow["source_records"] = [name.replace("/input_role_manifests/", "/roles/")
                                  for name in workflow["source_records"]]
    _write(parent / "workflow.json", workflow)
    before = _parent_bytes(c)
    _retry(c)
    assert all(p.read_bytes() == raw for p, raw in before.items())
    assert not (parent / "retry_v2/roles").exists()
    assert (parent / "retry_v2/input_role_manifests/common_dev.json").is_file()


@pytest.mark.parametrize("change", ["bad-pin", "bad-shape", "unsafe-path", "empty-grant", "rejected-budget",
    "changed-config", "changed-transform", "changed-recipe", "changed-state", "dirty-source"])
def test_invalid_retry_does_not_publish_or_change_parent(retry_case, monkeypatch, change):
    c = retry_case
    options = {}
    if change == "bad-pin":
        options["budget_reconciliation"] = {**c.budget, "sha256": "0" * 64}
    elif change == "bad-shape":
        options["budget_reconciliation"] = {"path": c.budget["path"]}
    elif change == "unsafe-path":
        options["budget_reconciliation"] = {**c.budget, "path": "../outside.json"}
    elif change == "empty-grant":
        options["authorization"] = " "
    elif change == "rejected-budget":
        record = _read(c.root / c.budget["path"])
        record["retry_authorized"] = False
        _write(c.root / c.budget["path"], record)
        c.budget["sha256"] = file_digest(c.root / c.budget["path"])
    elif change in {"changed-config", "changed-transform"}:
        path = c.root / owner.BASE / "training_inputs.json"
        metadata = _read(path)
        model = metadata["models"]["ppgt_author_v2"]
        if change == "changed-config":
            model["model_config"]["hidden"] += 1
        else:
            model["feature_bindings"]["common_dev"]["transform_code_sha"] = "0" * 64
        _write(path, metadata)
    elif change == "changed-recipe":
        path = c.root / owner.BASE / "ppgt_author_v2/contract.json"
        record = _read(path)
        record["learning_rate"] = .002
        _write(path, record)
    elif change == "changed-state":
        (c.initial / "ppgt_author_v2.pt").write_bytes(b"changed")
    else:
        def dirty(*args):
            raise ValueError("uncommitted repaired source")
        monkeypatch.setattr(owner, "_source_snapshot", dirty)
    before = _parent_bytes(c)
    with pytest.raises(ValueError):
        _retry(c, **options)
    assert not (c.root / RETRY_BASE).exists()
    assert not (c.root / POLICY).exists()
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_existing_retry_or_policy_are_immutable(retry_case):
    c = retry_case
    _retry(c)
    before = _parent_bytes(c)
    with pytest.raises(ValueError, match="already exist"):
        _retry(c)
    assert all(p.read_bytes() == raw for p, raw in before.items())


def test_existing_policy_prevents_partial_retry(retry_case):
    c = retry_case
    _write(c.root / POLICY, {"preserve": True})
    with pytest.raises(ValueError, match="already exists"):
        _retry(c)
    assert not (c.root / RETRY_BASE).exists()
    assert _read(c.root / POLICY) == {"preserve": True}
