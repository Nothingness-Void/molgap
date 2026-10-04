"""Publish reviewed terminal metadata without reexecuting frozen inference."""
import json
from pathlib import Path

from molgap.constants import REPO_ROOT as ROOT
from molgap.research_memory.finalize import finalize, verified_receipt
from molgap.training_reproducibility import atomic_json, sha256_file

HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()


def main():
    if (HERE / "rml/rml_finalized").exists():
        print(verified_receipt(HERE / "rml/rml_finalized")["finalization_id"])
        return
    terminal_path = HERE / "terminal.json"
    terminal = json.loads(terminal_path.read_text())
    before = HERE / "terminal_before_export_correction.json"
    if not before.exists():
        before.write_bytes(terminal_path.read_bytes())
    # V5 migration describes the exporter, not the separately executed workload.
    # This matches the accepted slot/readout and retained-scale-fit exports.
    terminal["evidence"]["migration"].update(
        inference_executed=False,
        verification_scope="Metadata-only export of separately executed prospective frozen K1 inference; no inference during publication.")
    terminal["evidence"]["observed_execution"] = {
        "execution_ref": f"{REL}/results/measurement.json",
        "inference_executed": True, "training_executed": False}
    audit = {"scope": "terminal-export metadata correction only; no source, input, inference or scientific result changes",
             "before_sha256": sha256_file(before),
             "reason": "Frozen close.py mapped observed execution into the migration exporter field; V5 rejects inference during migration.",
             "owning_validator": "src/molgap/v5_common.py:validate_v5_evidence_envelope",
             "accepted_mapping_example": "experiments/pcqm_k1_slot_readout_diagnostic/rml/rml_finalized/v5_evidence.json"}
    atomic_json(HERE / "export_correction.json", audit)
    for name in ("finalize_export.py", "export_correction.json", "terminal_before_export_correction.json"):
        pointer = f"{REL}/{name}"
        digest = sha256_file(ROOT / pointer)
        terminal["artifact_hashes"][pointer] = digest
        terminal["evidence"]["artifacts"].append({"name": name, "locator": pointer,
            "sha256": digest, "availability": "locally_retained_hash_verified"})
    atomic_json(terminal_path, terminal)
    print(json.dumps(finalize(ROOT, f"{REL}/rml", f"{REL}/terminal.json"), indent=2))


if __name__ == "__main__":
    main()
