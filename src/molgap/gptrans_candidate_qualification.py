"""Translate retained GPTrans author-arm metadata into shared qualification."""
from pathlib import Path

from .evidence_pointers import load_json_object
from .research_memory.candidate_qualification import assess_candidate_qualification, verify_candidate_qualification
from .research_memory.finalize import verified_receipt
from .research_memory.paths import resolve_repo_pointer
from .research_memory.trace import atomic_write, file_digest, json_bytes


def qualify_author_candidates(repo_root: Path) -> list[dict]:
    root = Path(repo_root).resolve()
    base = root / "experiments/pcqm_gptrans_author_alignment/gpu"
    reference = "experiments/pcqm_gptrans_v5_audit_reference/rml_plan/reference_qualification.json"
    rel = lambda p: p.resolve().relative_to(root).as_posix()
    results = []
    for mode in ("degree_scale", "path_bond_mean"):
        arm = base / mode
        finalized = arm / "rml_plan/rml_finalized"
        verified_receipt(finalized)
        marker = arm / "rml_plan/candidate_qualification.json"
        if marker.is_file():
            manifest, readiness = verify_candidate_qualification(root, rel(marker))
        else:
            terminal = load_json_object(finalized / "terminal_input.json")
            old = load_json_object(arm / "results/comparison_readiness.json")
            source = load_json_object(root / old["candidate_artifact_bindings"]["source_config"]["ref"])
            training = resolve_repo_pointer(root, old["candidate_artifact_bindings"]["checkpoint"]["ref"]).parent
            destination = arm / "results/qualification"
            descriptor = {"format": "molgap-candidate-qualification-v1",
                "trajectory_ref": rel(finalized / "trajectory.json"),
                "original_manifest_ref": rel(finalized / "trace_manifest.json"),
                "original_readiness_ref": rel(arm / "results/comparison_readiness.json"),
                "prelaunch_ref": rel(arm / "comparison_readiness_prelaunch.json"),
                "original_bundle_ref": source["reference_bundle_ref"],
                "reference_qualification_ref": reference,
                "acceptance_ref": rel(base / "results/acceptance.json"), "acceptance_arm": mode,
                "completion_ref": rel(training / "completion_manifest.json"),
                "qualified_manifest_ref": rel(destination / "qualified_trace_manifest.json"),
                "qualified_readiness_ref": rel(destination / "qualified_comparison_readiness.json"),
                "artifact_hashes": dict(terminal["artifact_hashes"]),
                "semantics": "Terminal binding reassessment of exact prospective reference; no new prospective information",
                "local_training_executed": False, "model_inference_executed": False,
                "successor_authorized": False}
            for key, pointer in descriptor.items():
                if key.endswith("_ref") and not key.startswith("qualified_"):
                    descriptor["artifact_hashes"][pointer] = file_digest(resolve_repo_pointer(root, pointer))
            prediction = load_json_object(resolve_repo_pointer(root,
                old["candidate_artifact_bindings"]["prediction_manifest"]["ref"]))
            descriptor["artifact_hashes"][prediction["artifact_locator"]] = file_digest(
                resolve_repo_pointer(root, prediction["artifact_locator"]))
            manifest, readiness = assess_candidate_qualification(root, descriptor)
            for key, value in (("qualified_manifest_ref", manifest), ("qualified_readiness_ref", readiness)):
                path = root / descriptor[key]
                if path.exists() and path.read_bytes() != json_bytes(value):
                    raise ValueError("qualification refuses to overwrite different evidence")
                atomic_write(path, json_bytes(value))
                descriptor["artifact_hashes"][descriptor[key]] = file_digest(path)
            atomic_write(marker, json_bytes(descriptor))
            verify_candidate_qualification(root, rel(marker))
        results.append({"trajectory_id": manifest["trajectory_id"], "comparison_class": readiness["comparison_class"],
                        "qualification_ref": rel(marker), "eligible": manifest["backtest_eligibility"]["eligible"]})
    return results
