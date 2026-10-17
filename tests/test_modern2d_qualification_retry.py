"""Synthetic, declaration-only checks for versioned Modern2D retries.

The retained V1 fixture contains JSON/source declarations but no initial-state
payloads.  A retry must therefore prove its byte reuse without constructing a
model, loading a graph, or reading the external state root.
"""
import inspect
import json
from pathlib import Path
import subprocess

import pytest

from molgap import modern2d_qualification_prepare as owner
from molgap.experiment_spec import ExperimentSpec


ROOT = Path(__file__).resolve().parents[1]


def _json(path):
    return json.loads(Path(path).read_bytes())


def _git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()


def _snapshot(root, relative):
    base = Path(root) / relative
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in base.rglob("*") if path.is_file()}


@pytest.fixture
def retained_v1(tmp_path):
    source_root = ROOT
    root = tmp_path / "repo"
    root.mkdir()
    preparation = _json(source_root / owner.BASE / "preparation.json")
    records = [*preparation["source_records"],
               owner.BASE + "/experiment_spec.json", owner.BASE + "/preparation.json"]
    for relative in sorted(set(records)):
        source = source_root / relative
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    _git(root, "init")
    _git(root, "config", "user.email", "retry-fixture@example.invalid")
    _git(root, "config", "user.name", "Modern2D retry fixture")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", ".")
    _git(root, "-c", "commit.gpgsign=false", "commit", "-m", "Retain V1 declarations")
    return root


def _retry(root, version, *, previous_base=owner.BASE, previous_decision_ref=None):
    suffix = "v" + str(version)
    return owner.freeze_qualification_retry(
        root,
        base=owner.BASE + "/retry_" + suffix,
        kernel=owner.ACCOUNT + "/molgap-modern2d-native-retry-" + suffix,
        source_dataset=owner.ACCOUNT + "/molgap-modern2d-qualification-source-retry-" + suffix,
        authorization="Human reply: authorize bounded retry " + suffix,
        previous_base=previous_base,
        previous_decision_ref=previous_decision_ref,
    )


def test_retry_reuses_frozen_bytes_and_derives_v2_then_v3(retained_v1):
    root = retained_v1
    v1_snapshot = {**_snapshot(root, owner.BASE),
                   **_snapshot(root, "research_memory/policies")}
    v1_spec = ExperimentSpec(_json(root / owner.BASE / "experiment_spec.json"))
    v1_prep = _json(root / owner.BASE / "preparation.json")

    v2 = _retry(root, 2)
    v2_base = root / v2["base"]
    v2_spec = ExperimentSpec(_json(v2_base / "experiment_spec.json"))
    v2_prep = _json(v2_base / "preparation.json")
    v2_authority = _json(v2_base / "authority.json")
    v2_policy = _json(root / "research_memory/policies/modern2d-native-qualification-retry-v2.v1.json")

    assert v2["spec_identity"] == v2_spec.identity != v1_spec.identity
    assert v2_spec.to_dict()["logical_run_id"].endswith("-v2")
    assert v2_authority["run_reference"] == v2["kernel"]
    assert v2_authority["source_dataset"] == v2["source_dataset"]
    assert v2_authority["logical_run_id"] == v2_spec.to_dict()["logical_run_id"]
    assert v2_authority["previous_spec_identity"] == v1_spec.identity
    assert v2_authority["scratch_optimizer_updates_per_model"] == 6
    assert v2_policy["policy_id"] == "modern2d-native-qualification-retry-v2"
    assert v2_policy["approval"]["authority_ref"] == v2["base"] + "/authority.json"

    old_contracts = {
        job["arm_id"]: (root / job["recipe"]).read_bytes()
        for job in v1_prep["jobs"]
    }
    for job in v2_prep["jobs"]:
        assert job["recipe"].startswith(v2["base"] + "/")
        assert (root / job["recipe"]).read_bytes() == old_contracts[job["arm_id"]]
        assert job["initial_state"] == next(
            old["initial_state"] for old in v1_prep["jobs"] if old["arm_id"] == job["arm_id"])
    assert not list(root.rglob("*.pt"))
    assert all(record.startswith(v2["base"] + "/") or
               record.startswith("research_memory/") or
               record.startswith("platforms/") or
               record.startswith("experiments/modern2d_model_integration/integration_attempt_report")
               for record in v2_prep["source_records"])
    assert all(not (record.startswith(owner.BASE + "/") and
                    not record.startswith(v2["base"] + "/"))
               for record in v2_prep["source_records"])

    for item in v2_spec.to_dict()["prospective"]["arms"]:
        plan = _json(root / item["plan_spec_ref"])
        assert item["trajectory_id"].endswith("-v2")
        assert plan["trajectory"]["trajectory_id"] == item["trajectory_id"]
        assert plan["decision_state"]["policy_id"] == v2_policy["policy_id"]
        assert plan["trajectory"]["family_id"] == v2_policy["policy_id"]
        contract_refs = plan["trajectory"]["state_at_start"]["contract_refs"]
        assert contract_refs[0].startswith(v2["base"] + "/")
        assert all(not (value.startswith(owner.BASE + "/") and
                        not value.startswith(v2["base"] + "/")) for value in contract_refs)

    v2_policy_path = "research_memory/policies/modern2d-native-qualification-retry-v2.v1.json"
    previous_decision_ref = v2["base"] + "/results/inspection_2026-10-18.md"
    previous_decision = root / previous_decision_ref
    previous_decision.parent.mkdir(parents=True)
    previous_decision.write_bytes(b"retry-v2 actual inspection; infrastructure-only\n")
    previous_decision_bytes = previous_decision.read_bytes()
    v2_snapshot = {**_snapshot(root, v2["base"]),
                   v2_policy_path: (root / v2_policy_path).read_bytes()}
    v3 = _retry(root, 3, previous_base=v2["base"],
                previous_decision_ref=previous_decision_ref)
    v3_spec = ExperimentSpec(_json(root / v3["base"] / "experiment_spec.json"))
    v3_prep = _json(root / v3["base"] / "preparation.json")
    v3_authority = _json(root / v3["base"] / "authority.json")
    assert v3_spec.identity not in {v1_spec.identity, v2_spec.identity}
    assert v3_spec.to_dict()["logical_run_id"].endswith("-v3")
    assert v3["kernel"].endswith("-v3") and v3["source_dataset"].endswith("-v3")
    assert v3_authority["previous_attempt_refs"] == [
        v2["base"] + "/STATUS.md", v2["base"] + "/experiment_spec.json",
        previous_decision_ref]
    assert previous_decision_ref in v3_prep["source_records"]
    assert (root / previous_decision_ref).read_bytes() == previous_decision_bytes
    assert previous_decision.is_file()
    assert previous_decision.read_bytes() == previous_decision_bytes
    assert {**_snapshot(root, v2["base"]),
            v2_policy_path: (root / v2_policy_path).read_bytes()} == v2_snapshot
    for key, value in v1_snapshot.items():
        assert (root / key).read_bytes() == value


def test_retry_rejects_previous_decision_escape_and_missing_record(retained_v1):
    root = retained_v1
    v2 = _retry(root, 2)
    decision_ref = v2["base"] + "/results/inspection_2026-10-18.md"

    with pytest.raises(ValueError):
        _retry(root, 3, previous_base=v2["base"],
               previous_decision_ref=v2["base"] + "/../escape.md")

    with pytest.raises(ValueError, match="existing retained record"):
        _retry(root, 4, previous_base=v2["base"],
               previous_decision_ref=decision_ref)


def test_retry_rejects_reuse_overwrite_escape_mismatch_and_wrong_owner(retained_v1):
    root = retained_v1
    before = {**_snapshot(root, owner.BASE),
              **_snapshot(root, "research_memory/policies")}
    cases = [
        (dict(base=owner.BASE, kernel="kaseichou/retry-v2",
              source_dataset="kaseichou/source-v2"), "separate"),
        (dict(base=owner.BASE + "/retry_v2", kernel="kaseichou/retry-v2",
              source_dataset="kaseichou/source-v3"), "same version"),
        (dict(base=owner.BASE + "/../escape", kernel="kaseichou/retry-v2",
              source_dataset="kaseichou/source-v2"), "Comparison base"),
        (dict(base=owner.BASE + "/retry_v2", kernel="other/retry-v2",
              source_dataset="kaseichou/source-v2"), "safe kaseichou"),
    ]
    for values, message in cases:
        with pytest.raises(ValueError, match=message):
            owner.freeze_qualification_retry(
                root, authorization="Human reply: distinct retry authorization", **values)
    existing = root / owner.BASE / "retry_v4"
    existing.mkdir(parents=True)
    sentinel = existing / "preserve.txt"
    sentinel.write_bytes(b"do not overwrite")
    with pytest.raises(ValueError, match="already exists"):
        owner.freeze_qualification_retry(
            root, base=owner.BASE + "/retry_v4",
            kernel="kaseichou/retry-v4", source_dataset="kaseichou/source-v4",
            authorization="Human reply: v4 retry")
    assert sentinel.read_bytes() == b"do not overwrite"
    after = {**_snapshot(root, owner.BASE),
             **_snapshot(root, "research_memory/policies")}
    for key, value in before.items():
        assert (root / key).read_bytes() == value
    assert set(before) <= set(after)


def test_prepare_package_keeps_v1_defaults_and_accepts_versioned_bindings():
    parameters = inspect.signature(owner.prepare_package).parameters
    assert parameters["base"].default == owner.BASE
    assert parameters["kernel"].default == owner.KERNEL
    assert parameters["source_dataset"].default == owner.SOURCE_DATASET
    assert set(("base", "kernel", "source_dataset")) <= set(parameters)
