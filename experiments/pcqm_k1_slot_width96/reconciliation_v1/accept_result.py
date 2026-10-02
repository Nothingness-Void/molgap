"""Export slot96 acceptance using the retained width-screen caller and shared RML."""
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

HERE = ROOT / "experiments/pcqm_k1_slot_width96"
RECORDS = HERE / "reconciliation_v1"
REMOTE_BASE = ROOT / "platforms/_records/kaggle/training/pcqm_k1_slot_width96_kaggle3_v1/reconciliation_v1"
REMOTE = REMOTE_BASE / "family_outputs/slot96"
QUAL = REMOTE_BASE / "qualification/slot96"
PROSPECTIVE = HERE / "kaggle3_v1/slot96/trajectory.json"
PACKAGE = ROOT / "platforms/_records/kaggle/staging/slot96_workflow_v1b/package"
PACKAGE_ID = "567fe90b6d12ffe2a75e36feb8dd17f784029d427534381b4cb18d9b3ddc8840"
TID = "TB-k1-slot-width96-kaggle3-100k-s42-v1"
RUN = "molgap-k1-slot96-100k-s42-v1:slot96:downstream"
EID = "pcqm-k1-slot-width96-kaggle3-100k-s42-v1-terminal"


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
    receipt = next((HERE / "submission_v1/receipts").glob("*.json"))
    context = RunContext.from_launch(spec, receipt, PACKAGE,
        expected_package_identity=PACKAGE_ID, arm_id="slot96")
    arm = spec.to_dict()["arms"][0]
    expected = read(HERE / "family_acceptance_plan.json")["arms"][0]["expected"]
    mechanical = inspect_output(REMOTE, context=context, expected=expected)
    if mechanical["status"] != "MECHANICALLY_VERIFIED":
        raise ValueError("Mechanical inspection failed")
    provenance = read(QUAL / "runtime_provenance.json")
    _check_context(provenance["context"], context)
    certificate = validate_runtime_preflight(QUAL, provenance)
    assert provenance["initialization_sha256"] == arm["initialization"]["state_sha256"]
    assert provenance["recipe_sha256"] == arm["training"]["recipe"]["sha256"]
    validate_screen_recipe(spec, "slot96", read(REMOTE / "training_contract.json"))
    architecture = read(QUAL / "architecture_preflight.json")
    assert certificate["architecture_sha256"] == canonical_fingerprint(architecture)
    assert architecture["mode"] == provenance["mode"] == "slot96"
    assert architecture["parameter_count"] == 3853793
    assert architecture["reference_parameter_count"] == 3658817
    assert architecture["finite_nonzero_gradient_check"] is True
    uploaded = read(HERE / "source_verification.json")
    assert provenance["initial_state_file_sha256"] == uploaded["files"]["initial_states/slot96.pt"]
    scheduler = read(HERE / "remote_observation_20261002T032227Z.json")
    assert scheduler["status"]["status"] == "COMPLETE"
    assert scheduler["remote_metadata"]["id"] == 136686300
    assert scheduler["remote_metadata"]["currentVersionNumber"] == 1
    assert scheduler["remote_entry_sha256"] == scheduler["submitted_entry_sha256"]
    assert scheduler["remote_entry_normalized_match"] is True
    metrics = read(RECORDS / "scientific_metrics.json")
    alignment, endpoint = metrics["alignment"], metrics["endpoint"]
    assert alignment["rows"] == 50000 and alignment["exact_order"] and alignment["identical_targets"]
    assert alignment["finite_predictions"]
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
        raw = read(QUAL / name)
        measurements = {item["metric"]: item for item in raw["costs"]}
        costs.append(dict(schema="molgap-cost-event-v1", cost_event_id=f"cost-{TID}-observed-{category}",
            trajectory_id=TID, action_id="A001", run_id=RUN, attempt_id="kaggle3-slot96-001",
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
    acceptance = dict(format="molgap-k1-slot96-terminal-acceptance-v1", evidence_id=EID, run_id=RUN,
        trajectory_id=TID, outcome=outcome, trajectory_decision=decision, role_use=role_use,
        mechanical_acceptance_ref=rel(RECORDS / "mechanical_acceptance.json"),
        runtime_qualification_ref=rel(RECORDS / "runtime_qualification.json"),
        scientific_metrics_ref=rel(RECORDS / "scientific_metrics.json"), costs=costs, roles=roles,
        comparison_class="PAIRED_ENDPOINT", replay_eligible=False,
        replay_exclusions=["full_software_equivalence_not_established", "literal_development_role_identity_mismatch"],
        native_cost_scope="One assigned T4 diagnostic and training windows; bootstrap, queue, idle second device and full allocation unknown",
        role_observation_basis="Frozen trainer, contract, finite retained predictions and 40-epoch trace; no new protected-role access")
    atomic_json(RECORDS / "runtime_qualification.json", dict(
        format="molgap-k1-slot96-runtime-review-v1", candidate_qualified=True,
        certificate=certificate, architecture=architecture,
        candidate_runtime_fingerprint=provenance["runtime_fingerprint"],
        strict_reference_runtime_qualified=False, replay_eligible=False,
        exclusions=acceptance["replay_exclusions"],
        installed_distribution_differences=metrics["installed_distribution_differences"],
        development_role_labels_as_recorded=metrics["development_role_labels_as_recorded"],
        interpretation="Aligned finite rows allow descriptive pairing; frozen metadata unchanged"))
    atomic_json(acceptance_path, acceptance)
    paths = [HERE / "protocol.md", HERE / "training_recipe.json", HERE / "terminal_decision.md",
        HERE / "reference_binding/reference_bundle.json", acceptance_path, Path(__file__), receipt]
    paths += [RECORDS / name for name in ["scientific_metrics.json", "../analyze_saved_result.py", "attribution.md",
        "mechanical_acceptance.json", "runtime_qualification.json"]]
    paths += [HERE / "remote_observation_20261002T032227Z.json", HERE / "source_verification.json",
        REMOTE_BASE / "retrieval_receipt.json", REMOTE_BASE / "pinned_output_manifest.json"]
    paths += [REMOTE / name for name in ["output_manifest.json", "development_predictions.pt", "selected_model.pt",
        "last_checkpoint.pt", "canonical_trace.json", "training_contract.json"]]
    paths += [QUAL / name for name in ["runtime_certificate.json", "runtime_provenance.json", "runtime_manifest.json",
        "architecture_preflight.json", "allocation_cost.json", "diagnostic_cost.json"]]
    hashes = {rel(path.resolve()): sha256_file(path) for path in paths}
    authority = [rel(HERE / "protocol.md"), rel(HERE / "terminal_decision.md"),
        rel(acceptance_path), rel(RECORDS / "scientific_metrics.json")]
    retained = [RECORDS / "scientific_metrics.json", acceptance_path, HERE / "terminal_decision.md",
        REMOTE / "development_predictions.pt", REMOTE / "selected_model.pt", REMOTE / "last_checkpoint.pt", REMOTE / "canonical_trace.json"]
    evidence = dict(format="molgap-v5-evidence-envelope-v1", contract="MOLGAP-COMMON-V5-FINAL", evidence_id=EID,
        track="B", scope="desktop_k1_latent_slot96_100k_saved_prediction_comparison", legacy_contract="k1-slot96-kaggle3-single-candidate-v1",
        outcome=outcome, authority=dict(pointers=authority), role_use=role_use,
        migration=dict(migrated_at="2026-10-02", training_executed=False, inference_executed=False,
            scientific_reinterpretation=False, verification_scope="Metadata export of separately executed prospective training; no training or inference during acceptance"),
        observed_execution=dict(training_executed=True, inference_executed=False, execution_ref=rel(HERE / "remote_observation_20261002T032227Z.json")),
        artifacts=[dict(name=path.name, locator=rel(path), sha256=hashes[rel(path)], availability="locally_retained_hash_verified") for path in retained])
    terminal = dict(format="molgap-rml-terminal-package-v1", trajectory_id=TID, run_id=RUN, action_id="A001",
        finalized_at=scheduler["observed_at_utc"], acceptance_ref=rel(acceptance_path), artifact_hashes=hashes,
        evidence=evidence, decision=decision, costs=costs, roles=roles)
    atomic_json(RECORDS / "terminal.json", terminal)
    # A retained trace is an evidence artifact. Strict cross-job runtime and literal role
    # equivalence are unqualified, preventing replay admission, so no trace manifest is invented.
    result = finalize(ROOT, rel(PROSPECTIVE), rel(RECORDS / "terminal.json"))
    print(json.dumps({"status": result["status"], "finalization_id": result["finalization_id"]}))


if __name__ == "__main__":
    main()
