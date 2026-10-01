"""Accept saved results and delegate immutable metadata publication to RML."""
from __future__ import annotations

import json
from pathlib import Path

from molgap.constants import REPO_ROOT as ROOT
from molgap.experiment_family_workflow import RunContext, inspect_output, _check_context
from molgap.experiment_spec import ExperimentSpec
from molgap.k1_screen_training import validate_runtime_preflight, validate_screen_recipe
from molgap.research_memory.finalize import finalize, verified_receipt
from molgap.training_reproducibility import atomic_json, sha256_file
from molgap.screen_policy import canonical_fingerprint

HERE = ROOT / "experiments/pcqm_k1_node_width256"
RECORDS = HERE / "kaggle3_reconciliation_v1"
REMOTE = ROOT / "platforms/_records/kaggle/training/pcqm_k1_node_width256_kaggle3_v1/experiment/width256"
PROSPECTIVE = HERE / "kaggle3_v1/width256/trajectory.json"
PACKAGE = ROOT / "platforms/_records/kaggle/staging/k1_width256_v1/package"
PACKAGE_ID = "475390043d4c5600f55f1aaa888218a65ce83c74d85ae1cc1aae5b72a289bc95"
TID = "TB-k1-node-width256-kaggle3-100k-s42-v1"
RUN = "nvoid912/molgap-k1-node256-100k-s42-v1:width256:downstream"
EID = "pcqm-k1-node-width256-kaggle3-100k-s42-v1-terminal"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path):
    return path.relative_to(ROOT).as_posix()


def main():
    destination = PROSPECTIVE.parent / "rml_finalized"
    if destination.exists():
        print(json.dumps(verified_receipt(destination)))
        return
    spec = ExperimentSpec.from_json((HERE / "experiment_spec_kaggle3_v1.json").read_text())
    receipt = next((HERE / "kaggle3_submission_v1/launch_receipt").glob("*.json"))
    context = RunContext.from_launch(spec, receipt, PACKAGE,
        expected_package_identity=PACKAGE_ID, arm_id="width256")
    arm = spec.to_dict()["arms"][0]
    expected = read(HERE / "family_acceptance_plan.json")["arms"][0]["expected"]
    mechanical = inspect_output(REMOTE, context=context, expected=expected)
    if mechanical["status"] != "MECHANICALLY_VERIFIED":
        raise ValueError("Mechanical inspection failed")
    provenance = read(REMOTE / "runtime_provenance.json")
    _check_context(provenance["context"], context)
    certificate = validate_runtime_preflight(REMOTE, provenance)
    assert provenance["initialization_sha256"] == arm["initialization"]["state_sha256"]
    assert provenance["recipe_sha256"] == arm["training"]["recipe"]["sha256"]
    validate_screen_recipe(spec, "width256", read(REMOTE / "training_contract.json"))
    architecture = read(REMOTE / "architecture_preflight.json")
    assert certificate["architecture_sha256"] == canonical_fingerprint(architecture)
    assert architecture["mode"] == provenance["mode"] == "width256"
    assert architecture["parameter_count"] == 6035201
    assert architecture["reference_parameter_count"] == 3658817
    assert architecture["finite_nonzero_gradient_check"] is True
    uploaded = read(HERE / "kaggle3_submission_v1/source_dataset_acceptance.json")
    initialization = next(item for item in uploaded["files"] if item["name"] == "initial_states/width256.pt")
    assert provenance["initial_state_file_sha256"] == initialization["sha256"]
    scheduler = read(RECORDS / "scheduler_observation.json")
    identity = read(RECORDS / "remote_entry_identity.json")
    assert scheduler["platform_status"]["status"] == "complete"
    assert identity["metadata"]["id"] == scheduler["kernel_id"] == 136660752
    assert identity["metadata"]["currentVersionNumber"] == scheduler["version_number"] == 1
    assert identity["matches_frozen_entry_text"] is True
    metrics = read(RECORDS / "scientific_metrics.json")
    alignment, endpoint = metrics["alignment"], metrics["endpoint"]
    assert alignment["rows"] == 50000 and alignment["exact_order"] and alignment["identical_targets"]
    assert alignment["finite_predictions_and_targets"]
    assert endpoint["candidate_mae_eV"] == mechanical["observed"]["development_mae_eV"]
    assert endpoint["gate_eV"] == .003 and endpoint["paired_gain_eV"] < .003
    outcome = dict(execution_status="complete", artifact_status="hash_verified_retained",
        comparison_status="PAIRED_ENDPOINT_single_seed", scientific_status="NEGATIVE_UNDER_CONTRACT",
        transfer_status="not_evaluated", budget_decision="no_successor_or_scale_up",
        full_handoff_status="not_applicable")
    decision = dict(outcome="NEGATIVE_UNDER_CONTRACT", decision_ref=rel(HERE / "terminal_decision.md"),
        next_allowed_actions=[], reopen_conditions=["new_decision_relevant_explicitly_authorized_contract"])
    acceptance_path = RECORDS / "acceptance.json"
    role_use = dict(official_train_prefix_0_100000="used", fixed50k_development="selection_used",
        official_validation="untouched", test_dev="untouched", test_challenge="untouched")
    costs = []
    for category, name in [("training", "allocation_cost.json"), ("preflight", "diagnostic_cost.json")]:
        raw = read(REMOTE / name)
        measurements = {item["metric"]: item for item in raw["costs"]}
        costs.append(dict(schema="molgap-cost-event-v1", cost_event_id=f"cost-{TID}-observed-{category}",
            trajectory_id=TID, action_id="A001", run_id=RUN, attempt_id="kaggle3-width256-001",
            category=category, platform="kaggle3", hardware="Tesla T4; one assigned device; invocation lower bound",
            evidence_ref=rel(acceptance_path), measurement=dict(
                device_hours=dict(status="measured", value=measurements["device_seconds"]["value"] / 3600),
                wall_hours=dict(status="measured", value=measurements["wall_seconds"]["value"] / 3600),
                cpu_hours=dict(status="measurement_missing", value=None),
                queue_hours=dict(status="measurement_missing", value=None))))
    roles = []
    for role, kinds in [("official_train_prefix_0_100000", ["training_membership", "labels_read"]),
                        ("fixed50k_development", ["prediction_input", "labels_read", "metric_computed", "selection_used"])]:
        for kind in kinds:
            roles.append(dict(schema="molgap-role-event-v1", role_event_id=f"role-{TID}-{role}-{kind}",
                trajectory_id=TID, action_id="A001", run_id=RUN, dataset_identity="pcqm4mv2-ogb-fixed-100k-v1",
                row_manifest_hash=expected["source_idx_sha256"] if role == "fixed50k_development" else arm["data"]["split"]["sha256"],
                role_name=role, access_kind=kind, selection_used=role == "fixed50k_development",
                evidence_ref=rel(acceptance_path)))
    acceptance = dict(format="molgap-k1-width256-terminal-acceptance-v1", evidence_id=EID, run_id=RUN,
        trajectory_id=TID, outcome=outcome, trajectory_decision=decision, role_use=role_use,
        mechanical_acceptance_ref=rel(RECORDS / "mechanical_acceptance.json"),
        runtime_qualification_ref=rel(RECORDS / "runtime_qualification.json"),
        scientific_metrics_ref=rel(RECORDS / "scientific_metrics.json"), costs=costs, roles=roles,
        comparison_class="PAIRED_ENDPOINT", replay_eligible=False,
        replay_exclusions=["canonical_reference_evidence_id_not_frozen", "full_software_equivalence_not_established"],
        native_cost_scope="One assigned T4 diagnostic and training windows; bootstrap, queue, idle second device and full allocation unknown",
        role_observation_basis="Frozen trainer, contract, finite retained predictions and 40-epoch trace; no new protected-role access")
    atomic_json(acceptance_path, acceptance)
    paths = [HERE / "training_protocol_kaggle3_v1.md", HERE / "training_recipe.json", HERE / "terminal_decision.md",
        HERE / "reference_binding/reference_bundle.json", acceptance_path, Path(__file__), receipt]
    paths += [RECORDS / name for name in ["scientific_metrics.json", "analyze_saved_predictions.py", "attribution.md",
        "mechanical_acceptance.json", "runtime_qualification.json", "scheduler_observation.json", "remote_entry_identity.json",
        "retrieval_receipt.json", "additional_retrieval_receipt.json"]]
    paths += [REMOTE / name for name in ["output_manifest.json", "development_predictions.pt", "selected_model.pt",
        "last_checkpoint.pt", "canonical_trace.json", "training_contract.json", "runtime_certificate.json",
        "runtime_provenance.json", "runtime_manifest.json", "architecture_preflight.json", "allocation_cost.json", "diagnostic_cost.json"]]
    hashes = {rel(path): sha256_file(path) for path in paths}
    authority = [rel(HERE / "training_protocol_kaggle3_v1.md"), rel(HERE / "terminal_decision.md"),
        rel(acceptance_path), rel(RECORDS / "scientific_metrics.json")]
    retained = [RECORDS / "scientific_metrics.json", acceptance_path, HERE / "terminal_decision.md",
        REMOTE / "development_predictions.pt", REMOTE / "selected_model.pt", REMOTE / "last_checkpoint.pt", REMOTE / "canonical_trace.json"]
    evidence = dict(format="molgap-v5-evidence-envelope-v1", contract="MOLGAP-COMMON-V5-FINAL", evidence_id=EID,
        track="B", scope="desktop_k1_atom_width256_100k_saved_prediction_comparison", legacy_contract="k1-width256-kaggle3-single-candidate-v1",
        outcome=outcome, authority=dict(pointers=authority), role_use=role_use,
        migration=dict(migrated_at="2026-10-02", training_executed=False, inference_executed=False,
            scientific_reinterpretation=False, verification_scope="Metadata export of separately executed prospective training; no training or inference during acceptance"),
        observed_execution=dict(training_executed=True, inference_executed=False, execution_ref=rel(RECORDS / "scheduler_observation.json")),
        artifacts=[dict(name=path.name, locator=rel(path), sha256=hashes[rel(path)], availability="locally_retained_hash_verified") for path in retained])
    terminal = dict(format="molgap-rml-terminal-package-v1", trajectory_id=TID, run_id=RUN, action_id="A001",
        finalized_at=scheduler["observed_at_utc"], acceptance_ref=rel(acceptance_path), artifact_hashes=hashes,
        evidence=evidence, decision=decision, costs=costs, roles=roles)
    atomic_json(RECORDS / "terminal.json", terminal)
    # A retained trace is an evidence artifact. Missing frozen canonical reference
    # identity prevents indexed trace/replay admission, so no trace manifest is invented.
    result = finalize(ROOT, rel(PROSPECTIVE), rel(RECORDS / "terminal.json"))
    print(json.dumps({"status": result["status"], "finalization_id": result["finalization_id"]}))


if __name__ == "__main__":
    main()
