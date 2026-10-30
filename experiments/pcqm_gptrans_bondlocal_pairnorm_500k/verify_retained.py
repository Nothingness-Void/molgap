"""CPU-only verification of two retained prediction files, never model weights."""
from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from molgap.comparison_readiness import assess_comparison_readiness
from molgap.experiment_family_workflow import RunContext, TargetIdentityBinding, _inspect_predictions, tensor_digest
from molgap.research_memory.schemas import validate_trajectory
from molgap.research_memory.trace import atomic_write, file_digest, json_bytes
from molgap.screen_policy import canonical_fingerprint

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
SNAPSHOT = "6081909b399ec12529a0676ce337708c94e9bead"
OLD_BASE = "experiments/pcqm_clean_k1_gptrans_local_500k"
NEW_BASE = "experiments/pcqm_gptrans_bondlocal_pairnorm_lr_500k"
NEW = Path("D:/molgap-artifacts/reconciliation/gptrans-bondlocal-pairnorm-lr-500k/v1/manifest/experiment/pairnorm_lr1e3")
OLD_PRED = Path("D:/molgap-cleanup-retained-20261017/final-retirement/objects/e8cf179f9caeb143831cf440d17e1421f99c001cdb891823555d920c6cd686db")
OLD_SHA = OLD_PRED.name
NEW_SHA = "6909b9a00d322499853e6dc36a982571696061b2b2f1af74c670428ab4ed55b2"
OLD_CONTRACT_SHA = "6a6d020332a59a2887a1d1eaeb950b82f928aaa7270cd0c5cdf5401cea3d091d"
NEW_CONTRACT_SHA = "e0391c7b6317ca24801fd68708da13df513833c22a667b7ee9bec6869473f2d7"
RUNTIMES = ["6c704d88b7c3800979b1f20f9c2433092a25f38151cda00d4da5e08b7d823985", "60677044cd34a634c634235baae7da7f5c0c0c2163b1587a901e9b4405772b7f"]


def archived(path):
    return subprocess.check_output(["git", "show", f"{SNAPSHOT}:{path}"], cwd=ROOT)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    atomic_write(path, json_bytes(value))


def prepare():
    """Freeze the bounded question before decoding either prediction payload."""
    import hashlib

    path = HERE / "trajectory.json"
    if path.exists():
        validate_trajectory(read_json(path))
        return
    fixture = read_json(ROOT / "experiments/pcqm_k1_hop_fusion_qualification/plan_input.json")
    trajectory = fixture["trajectory"]
    prior_path = ROOT / NEW_BASE / "runs/pcqm-gptrans-pairnorm-lr-500k-s42/prospective/pairnorm_lr1e3/rml_finalized/trajectory.json"
    prior = read_json(prior_path)
    evidence = prior["result"]["evidence_ids"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    trajectory.update(trajectory_id="TB-gptrans-bondlocal-pairnorm-500k-reuse-20261028",
                      family_id="gptrans-retained-500k-reuse", question="Can the retained no-norm and pair-update-norm 500K outputs be reused without repeat training?")
    trajectory["hypothesis"].update(
        hypothesis_id="H-gptrans-retained-500k-reuse-20261028",
        observed_deficiency="Both endpoints exist but their runtime scopes and target encodings differ.",
        changed_mechanism="No model intervention: retained artifact and comparison eligibility audit only.",
        cheapest_falsifier="Verify pinned prediction bytes, original target digests, row alignment and runtime scopes on CPU.",
        alternative_explanations=["Targets differ only in storage encoding", "Runtime and execution changes confound the norm intervention", "Retired checkpoint bytes prevent resume reuse"],
        supporting_evidence_ids=evidence, related_closed_family_ids=[prior["family_id"]],
        expected_native_cost_ref="README.md; CPU-only audit, no accelerator allocation",
        decision_changed_if_positive="Reuse verified outputs for allowed analysis without retraining; strict eligibility remains separate.",
        decision_changed_if_negative="Retain exact blockers; do not automatically train a replacement reference.")
    role = prior["state_at_start"]["role_snapshot_refs"][0]
    trajectory["state_at_start"].update(source_commit=commit, source_config_identity=file_digest(Path(__file__)),
        contract_refs=[f"{REL}/README.md", f"{NEW_BASE}/reference_recipe.json"],
        reference_ids=evidence, parent_trajectory_ids=[prior["trajectory_id"]], prior_evidence_ids=evidence,
        role_snapshot_refs=[role], budget_snapshot_ref=f"{REL}/README.md")
    trajectory["actions"] = [{"action_id": "A001", "type": "local_saved_artifact_analysis", "run_ids": ["138717853-v1", "141037251-v1"], "attempt_ids": ["reuse-audit-001"], "source_commit": commit, "evidence_refs": [f"{REL}/README.md"], "cost_event_ids": []}]
    trajectory["result"] = {"evidence_ids": [], "evidence_refs": []}
    trajectory["decision"] = {"outcome": "ACTIVE", "decision_ref": f"{REL}/README.md", "next_allowed_actions": ["One CPU-only retained prediction audit"], "reopen_conditions": []}
    trajectory.update(comparison_class="CONTEXT_ONLY", comparison_readiness_ref=f"{REL}/README.md", comparison_blockers=["Different runtime certificate scopes", "No prospective norm-only comparison for the two historical runs"], reference_bundle_id=None)
    trajectory["audit_started_at"] = datetime.now(timezone.utc).isoformat()
    trajectory["source_script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    validate_trajectory(trajectory)
    save(path, trajectory)


def verify():
    import hashlib
    import torch

    started = time.perf_counter()
    old_bytes = archived(f"{OLD_BASE}/terminal_acceptance/outputs/experiment/gptrans_local/training_contract.json")
    if hashlib.sha256(old_bytes).hexdigest() != OLD_CONTRACT_SHA or file_digest(NEW / "training_contract.json") != NEW_CONTRACT_SHA:
        raise ValueError("Frozen contract hash mismatch")
    contracts = [json.loads(old_bytes), read_json(NEW / "training_contract.json")]
    manifests = [json.loads(archived(f"{OLD_BASE}/terminal_acceptance/outputs/experiment/gptrans_local/output_manifest.json")), read_json(NEW / "output_manifest.json")]
    provenance_bytes = [archived(f"{OLD_BASE}/terminal_acceptance/outputs/experiment/gptrans_local/runtime_provenance.json"), (NEW / "runtime_provenance.json").read_bytes()]
    certificates = [json.loads(archived(f"{OLD_BASE}/terminal_acceptance/outputs/experiment/gptrans_local/runtime_certificate.json")), read_json(NEW / "runtime_certificate.json")]
    for raw, certificate, scope in zip(provenance_bytes, certificates, RUNTIMES):
        if canonical_fingerprint(json.loads(raw)) != certificate["provenance_sha256"]:
            raise ValueError("Certificate/provenance binding mismatch")
        if json.loads(raw)["runtime_fingerprint"] != scope:
            raise ValueError("Unexpected runtime scope")
    plan_path = f"{NEW_BASE}/generated/68756a96f1b29444e610ba40b14dd6f7006cae9b127a06e36f2a598147af6044/acceptance_plan.json"
    target_binding = TargetIdentityBinding(ROOT, plan_path, file_digest(ROOT / plan_path))
    paths = [OLD_PRED, NEW / "development_predictions.pt"]
    pins = [OLD_SHA, NEW_SHA]
    observed, payloads = [], []
    for index, (path, pin, contract, manifest) in enumerate(zip(paths, pins, contracts, manifests)):
        if file_digest(path) != pin:
            raise ValueError(f"Prediction hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        context = RunContext(**manifest["context"])
        if manifest["artifacts"]["predictions"]["sha256"] != pin:
            raise ValueError("Output manifest prediction binding mismatch")
        result = _inspect_predictions(path, context=context, expected=contract["acceptance_requirements"], target_identity=target_binding if index else None)
        expected_mae = [0.10195137561321259, 0.09998737781524658][index]
        if abs(result["development_mae_eV"] - expected_mae) > 1e-10:
            raise ValueError("Previously accepted MAE not reproduced")
        if not torch.isfinite(payload["target_eV"]).all():
            raise ValueError("Nonfinite targets")
        if not torch.equal(payload["source_idx"], torch.arange(500000, 550000)):
            raise ValueError("Outside authorized ordered internal role")
        observed.append({"path": str(path), "sha256": pin, "target_dtype": str(payload["target_eV"].dtype),
                         "target_digest": tensor_digest(payload["target_eV"], role="target"), **result})
        payloads.append(payload)
    old, new = payloads
    if not torch.equal(old["source_idx"], new["source_idx"]):
        raise ValueError("Row order differs")
    target_delta = (old["target_eV"].double() - new["target_eV"].double()).abs()
    report = {"audit_completed_at": datetime.now(timezone.utc).isoformat(), "executed_script_sha256": file_digest(Path(__file__)), "prediction_file_integrity": "VERIFIED", "observed": observed,
        "row_count": 50000, "row_order_equal": True, "targets_equal_after_float32_cast": torch.equal(old["target_eV"].float(), new["target_eV"].float()),
        "target_max_abs_difference_eV": target_delta.max().item(), "target_mean_abs_difference_eV": target_delta.mean().item(),
        "mae_difference_eV": observed[1]["development_mae_eV"] - observed[0]["development_mae_eV"],
        "runtime_certificate_scopes": RUNTIMES, "provenance_certificate_bindings_verified": True,
        "prediction_analysis_reusable": target_delta.max().item() == 0.0, "strict_causal_reuse": False,
        "original_resume_checkpoint_reuse": "Old original resume bytes retired; prediction verification is not checkpoint verification",
        "scope": "Selection-used internal development only; no independent generalization or causal PairNorm attribution",
        "cpu_audit_wall_seconds": time.perf_counter() - started, "training_performed": False, "model_inference_performed": False}
    # Supply only observed identity fields; missing evidence remains missing.
    cand = {"comparison_identity": {"runtime_certificate_scope": RUNTIMES[1]}, "row_alignment_status": "aligned"}
    ref = {"comparison_identity": {"runtime_certificate_scope": RUNTIMES[0]}}
    report["comparison_readiness"] = assess_comparison_readiness(candidate_id="141037251-v1-pairnorm_lr1e3", candidate=cand,
        reference_id="138717853-v1-gptrans_local", reference=ref, declared_intervention_fields=["architecture_config_identity"])
    save(HERE / "reuse_report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in {"comparison_readiness", "observed"}}, indent=2))
    print(json.dumps(observed, indent=2))


if __name__ == "__main__":
    prepare()
    verify()
