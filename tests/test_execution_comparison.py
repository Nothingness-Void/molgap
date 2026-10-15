"""Refactoring must reproduce the retained authored plans exactly."""
import json
from pathlib import Path

import pytest

from molgap.execution_comparison import ExecutionQuestion, execution_proposal
from molgap.experiment_spec import ExperimentSpec


@pytest.mark.parametrize("family", ["k1", "gptrans"])
def test_shared_proposal_matches_both_immutable_authored_records(family):
    root = Path(__file__).resolve().parents[1]
    base = root / ("experiments/" + family + "_infrastructure_t4_100k")
    spec = ExperimentSpec.from_json((base / "experiment_spec.json").read_text())
    for arm in spec.to_dict()["arms"]:
        expected = json.loads((base / arm["arm_id"] / "plan_input.json").read_bytes())
        t, d = expected["trajectory"], expected["decision_state"]
        h, state = t["hypothesis"], t["state_at_start"]
        q = ExecutionQuestion(t["family_id"], d["chosen_action"], t["question"], h["observed_deficiency"],
            h["changed_mechanism"], tuple(h["supporting_evidence_ids"]), tuple(h["alternative_explanations"]),
            tuple(h["related_closed_family_ids"]), h["decision_changed_if_positive"], h["decision_changed_if_negative"],
            tuple(state["reference_ids"]), tuple(state["prior_trajectory_ids"]))
        actual = execution_proposal(question=q, arm=arm, trajectory_id=t["trajectory_id"],
            source_commit=state["source_commit"], run=spec.to_dict()["logical_run_id"], version="v1",
            contract_ref=t["actions"][0]["evidence_refs"][0], contract_refs=state["contract_refs"],
            authority_ref=state["budget_snapshot_ref"], role_ref=state["role_snapshot_refs"][0],
            decision_ref=t["decision"]["decision_ref"], timestamp=d["state_timestamp"])
        assert actual == expected


def test_shared_envelope_writer_never_overwrites_an_existing_release(tmp_path):
    from molgap.execution_comparison import freeze_comparison_declarations
    base = "experiments/synthetic"
    existing = tmp_path / base / "experiment_spec.json"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"immutable")
    with pytest.raises(ValueError, match="already frozen"):
        freeze_comparison_declarations(tmp_path, base=base, experiment_id="synthetic", logical_run_id="synthetic",
            arms=[], bindings=[], jobs=[], entries=[], plan_format="synthetic", authority_ref="authority.json",
            source_extras=[], account="owner", kernel="owner/kernel", title="Synthetic", source_dataset="owner/source",
            data_datasets=["owner/fixed-data"])
    assert existing.read_bytes() == b"immutable"
