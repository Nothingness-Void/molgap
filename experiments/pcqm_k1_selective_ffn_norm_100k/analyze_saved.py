"""Recover the retained float32 encoding for CPU artifact checks, not admission.

Bindings below are post-run target-only recovery inputs, not prelaunch plans or
strict reference evidence. Frozen source, recipes, predictions and hashes stay
unchanged. No graph, model forward or trainer is invoked.
"""
from pathlib import Path
import json
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["ROCR_VISIBLE_DEVICES"] = "-1"
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from molgap.common.paths import MachinePaths
from molgap.experiment_attribution import _load_arm_artifacts, _analyze_loaded_pair
from molgap.experiment_family_workflow import RunContext, TargetIdentityBinding, inspect_output
from molgap.experiment_package import read_packaged_text
from molgap.experiment_spec import ExperimentSpec
from molgap.family_training_runner import inspect_retained_native_qualification
from molgap.research_memory.trace import file_digest
from molgap.training_reproducibility import atomic_json


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    import torch
    if torch.cuda.is_available():
        raise RuntimeError("Saved-artifact acceptance is CPU-only")
    torch.set_num_threads(2)
    config = read(HERE / "input.json")
    paths = MachinePaths.from_file(config["machine_profile"]).outside_checkout(ROOT)
    outputs = paths.artifact_root / "research" / config["experiment_id"] / "attempt_2"
    checks = HERE / "terminal_checks"
    target_ref = "experiments/pcqm_k1_slot_width96/reference_binding/target_manifest.json"
    # This exact asset predates this question; never derive a new target pin from outputs.
    target_sha = "9c5b2f4a161cc3d6539cfa7bca89da37773276b201d775b1ab88300ac4c9de10"
    if file_digest(ROOT / target_ref) != target_sha:
        raise ValueError("Original target encoding asset changed")
    loaded, summary = {}, {"arms": {}, "comparisons": {}, "strict_comparison": "NOT_QUALIFIED",
        "encoding_recovery": "post-run binding to the pre-existing pinned float32 manifest",
        "terminal_rml": "PENDING", "training_or_inference_executed": False}
    for item in config["arms"]:
        arm = item["arm_id"]
        spec = ExperimentSpec(read(HERE / "attempt_2" / arm / "experiment_spec.json"))
        prepared = paths.run_directory(spec.to_dict()["logical_run_id"], "v2") / "prepared"
        package = prepared / "package"
        frozen = read(prepared / "prepared_training.json")
        job, = frozen["jobs"]
        execution = outputs / arm
        folder = execution / arm
        context = RunContext.from_launch(spec, execution / "launch_receipt.json", package,
            expected_package_identity=frozen["expected_package_identity"], arm_id=arm)
        contract = json.loads(read_packaged_text(package, job["recipe"]))
        expected = contract["acceptance_requirements"]
        binding_path = checks / (arm + "_target_encoding.json")
        # No reference or prelaunch comparison is invented by recovering an encoding.
        atomic_json(binding_path, {"format": "molgap-family-acceptance-plan-v1",
            "spec_identity": spec.identity, "arms": [{"arm_id": arm, "adapter": "k1-session-v1",
                "expected": expected, "contract": {"path": job["recipe"],
                    "sha256": context.training_recipe_sha256},
                "comparison_prelaunch": None, "reference_bundle": None,
                "reference_artifacts": {"target_manifest": {"path": target_ref, "sha256": target_sha}}}]})
        binding = TargetIdentityBinding(ROOT, binding_path.relative_to(ROOT).as_posix(), file_digest(binding_path))
        inspection = inspect_output(folder, context=context, expected=expected, target_identity=binding)
        atomic_json(checks / (arm + "_inspection.json"), inspection)
        if inspection["status"] != "MECHANICALLY_VERIFIED":
            raise ValueError(str(inspection["blockers"]))
        proofs = list((execution / "preflight").glob("*/runtime_provenance.json"))
        if len(proofs) != 1:
            raise ValueError("Expected exactly one retained native qualification")
        provenance = read(proofs[0])
        native = inspect_retained_native_qualification(proofs[0].parent,
            recipe_path=ROOT / job["recipe"], provenance=provenance, context=context)
        atomic_json(checks / (arm + "_native.json"), native)
        if native["status"] != "VERIFIED":
            raise ValueError(str(native["blockers"]))
        loaded[arm] = _load_arm_artifacts(folder, context=context, expected=expected,
            target_identity=binding, _inspection=inspection)
        summary["arms"][arm] = {"selected": loaded[arm]["selected_checkpoint"],
            "process_wall_seconds": loaded[arm]["wall_seconds"],
            "mechanical": inspection["status"], "native_proof_reinspection": native["status"],
            "output_manifest_sha256": loaded[arm]["hashes"]["output_manifest_sha256"]}
        print(arm, summary["arms"][arm], flush=True)
    for arm in ("ffn_rms", "local_residual_gain"):
        report = _analyze_loaded_pair(loaded["bn_reference"], loaded[arm],
            output_dir=checks / arm, n_bootstrap=2000, seed=42)
        comparison = report["primary_comparison"]
        summary["comparisons"][arm] = {**comparison,
            "contract_material_signal": comparison["mean_delta_eV"] <= -0.003 and comparison["upper_95_eV"] < 0,
            "row_win_loss": report["row_win_loss"]}
    atomic_json(checks / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
