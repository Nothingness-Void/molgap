"""Reuse the two-role NO_TRAIN terminal owner after controller interpretation."""
from datetime import datetime, timezone
import json

from .gptrans_triplet_portability import BASE, accept
from .training_reproducibility import atomic_json, sha256_file


def close(root, output, inputs):
    from .k1_relation_audit_records import prepare_no_train_terminal
    from .research_memory.finalize import finalize
    if (root / BASE / "rml_plan/rml_finalized").exists():
        return finalize(root, root / BASE / "rml_plan/trajectory.json", root / BASE / "results/terminal.json")
    decision = root / BASE / "decision.md"
    if not (root / BASE / "decision_terminal.md").is_file() or "decision_terminal.md" not in decision.read_text():
        raise ValueError("Dated controller interpretation required before terminal closure")
    accepted = accept(output, inputs)
    frozen = json.loads((root / BASE / "rml_plan/trajectory.json").read_text())
    results = root / BASE / "results"
    ledger = accepted["native_cost"]
    atomic_json(results / "acceptance_summary.json", accepted)
    atomic_json(results / "execution.json", dict(allocation=ledger, training_executed=False, local_model_inference_executed=False))
    atomic_json(results / "role_row_manifests.json", dict(
        original_100k=dict(rows=50000, source_idx_start=100000, source_idx_stop=150000,
            prediction_progress_sha256=sha256_file(output / "portability/original_100k/progress.json")),
        unseen_500k=dict(rows=10000, source_idx_start=500000, source_idx_stop=550000,
            row_selection="frozen-RandomState42-without-replacement-sorted", reused_development=True,
            prediction_progress_sha256=sha256_file(output / "portability/unseen_500k/progress.json"),
            reference_payload_sha256=sha256_file(inputs / "reference_later.pt"))))
    manifest = json.loads((output / "output_manifest.json").read_text())
    refs = [(output / name).relative_to(root).as_posix() for name in ("output_manifest.json", *manifest["files"]) if name.endswith(".json")]
    refs += [BASE + "/results/" + n + ".json" for n in ("acceptance_summary", "execution", "role_row_manifests", "role_history", "cost_records")]
    missing = dict(status="measurement_missing", value=None)
    paths = prepare_no_train_terminal(prefix=BASE, frozen=frozen, run_id=frozen["actions"][0]["run_ids"][0],
        evidence_id="pcqm-gptrans-triplet-portability-s42-v1", outcome=dict(execution_status="complete",
            artifact_status="accepted", comparison_status="paired_endpoint_diagnostic",
            scientific_status="no_train_frozen_cohort_observations", transfer_status="frozen_cohort_diagnostic_only",
            budget_decision="within_frozen_cap", full_handoff_status="not_authorized"),
        scope="frozen_triplet_NO_TRAIN_portability_not500K_optimization", finalized_at=datetime.now(timezone.utc).isoformat(),
        artifact_refs=refs, authority=[BASE + "/" + n for n in ("protocol.md", "contract.json", "role_plan.json", "budget.json", "release_binding.json", "submission_receipt.json", "decision.md", "decision_terminal.md")],
        acceptance_name="acceptance_summary", repo_root=root, cost_measurement=dict(
            device_hours=dict(status="measured", value=ledger["allocated_device_seconds"] / 3600),
            wall_hours=dict(status="measured", value=ledger["wall_seconds"] / 3600), cpu_hours=missing, queue_hours=missing), cost_semantics=ledger["scope"])
    return finalize(root, root / paths["trajectory"], root / paths["terminal"])
