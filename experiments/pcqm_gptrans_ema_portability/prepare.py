"""Local NO_TRAIN planning/source staging; does not publish or submit."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone

from molgap.gptrans_portability import ARMS, verify_file
from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file, canonical_fingerprint
from molgap.v4_bundle import build_v4_source_bundle
from molgap.frozen_inference_release import FORMAT, check_frozen_inference_release

BASE = Path("experiments/pcqm_gptrans_ema_portability")
LOCATORS = {
    "ema9999": "platforms/_records/kaggle/training/gptrans_author_inputs_dual_s42_v1/gptrans_author_screen/degree_scale/training",
    "ema999": "platforms/_records/kaggle/training/gptrans_g1_input_ema_v1/gptrans_input_ema_screen/degree_scale_ema999/training",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-plan", action="store_true", help="Preserve already frozen prospective bytes during an infrastructure-only packaging correction")
    parser.add_argument("--attempt", choices=("v1", "v2", "v3", "v4"), default="v1")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    destination = args.output.resolve()
    if destination.exists():
        raise ValueError("Never overwrite a staged release")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    snapshot = copy.deepcopy(json.loads((root / "experiments/pcqm_gptrans_readout_100k/gpu/degree_node_mean_readout_ema999/plan_input.json").read_text()))
    attempt_base = BASE if args.attempt == "v1" else BASE/f"attempt_{args.attempt}"
    trajectory_id = "TC-gptrans-g1-ema-portability-frozen-s42" + (f"-{args.attempt}" if args.attempt != "v1" else "")
    run_id = "gptrans-ema-portability-audit:" + args.attempt
    cost_id = "cost-" + trajectory_id
    state = snapshot["trajectory"]["state_at_start"]
    state.update(source_commit=commit, source_config_identity=sha256_file(root/BASE/"contract.json"),
        contract_refs=[(BASE/"contract.json").as_posix()],
        reference_ids=["pcqm-gptrans-author-degree-scale-100k-s42", "pcqm-gptrans-g1-degree-scale-ema999-100k-s42"],
        prior_evidence_ids=["pcqm-gptrans-author-degree-scale-100k-s42", "pcqm-gptrans-g1-degree-scale-ema999-100k-s42"],
        prior_trajectory_ids=[], parent_trajectory_ids=(["TC-gptrans-g1-ema-portability-frozen-s42" + (f"-v{int(args.attempt[1:])-1}" if args.attempt != "v2" else "")] if args.attempt != "v1" else []),
        role_snapshot_refs=[(BASE/"role_plan.json").as_posix()], budget_snapshot_ref=(BASE/"budget.json").as_posix())
    hypothesis = snapshot["trajectory"]["hypothesis"]
    hypothesis.update(hypothesis_id="H-"+trajectory_id,
        observed_deficiency="Corrected EMA improved the selection cohort; portability before scale-up is unknown.",
        supporting_evidence_ids=state["prior_evidence_ids"], alternative_explanations=["Selection cohort effects", "EMA averaging horizon"],
        changed_mechanism="Frozen selected weights only; NO_TRAIN paired inference",
        cheapest_falsifier="Reproduce both saved original payloads before opening fixed500K development",
        related_closed_family_ids=["gptrans-readout", "gptrans-path-ema-combination"], expected_native_cost_ref=cost_id,
        decision_changed_if_positive="Controller plans one matched500K training contrast after acceptance",
        decision_changed_if_negative="Close EMA portability; no unchanged training extension",
        historical_unknowns=["No500K optimization performed; cohort reused for research selection"])
    trajectory = snapshot["trajectory"]
    trajectory.update(trajectory_id=trajectory_id, family_id="gptrans-ema-frozen-portability",
        question="Does G1 corrected EMA retain at least half its original gain on fixed500K development without retraining?")
    trajectory["decision"].update(decision_ref=(BASE/"README.md").as_posix(), next_allowed_actions=["Bounded NO_TRAIN audit; controller-owned terminal analysis"], reopen_conditions=["Terminal evidence or actionable infrastructure fault"])
    trajectory["actions"] = [dict(action_id="A001", type="NO_TRAIN_frozen_inference", source_commit=commit,
        run_ids=[run_id], attempt_ids=[args.attempt], evidence_refs=[(BASE/"contract.json").as_posix()], cost_event_ids=[cost_id])]
    snapshot["decision_state"].update(available_actions=["RUN_FROZEN_EMA_AUDIT", "DEFER"], chosen_action="RUN_FROZEN_EMA_AUDIT",
        policy_id="gptrans-ema-portability-audit", policy_version="v1", source_commit=commit,
        state_timestamp=datetime.now(timezone.utc).isoformat())
    snapshot["costs"] = [dict(schema="molgap-cost-event-v1", cost_event_id=cost_id, trajectory_id=trajectory_id,
        action_id="A001", run_id=run_id, attempt_id=args.attempt, category="audit", platform="kaggle2", hardware="Tesla_T4x2",
        evidence_ref=(BASE/"budget.json").as_posix(), measurement={
            "device_hours": dict(status="estimated", value=2), "wall_hours": dict(status="estimated", value=1),
            "queue_hours": dict(status="measurement_missing", value=None), "cpu_hours": dict(status="measurement_missing", value=None)})]
    if args.reuse_plan:
        if (root/attempt_base/"rml_plan/rml_finalized").exists():
            raise ValueError("Cannot reuse a closed prospective attempt")
        prior = json.loads((root/attempt_base/"rml_plan/trajectory.json").read_text())
        if prior["trajectory_id"] != trajectory_id or prior["state_at_start"]["source_config_identity"] != sha256_file(root/BASE/"contract.json"):
            raise ValueError("Repackaging may not change the frozen scientific contract")
        prospective = dict(status="EXISTING_FROZEN_PLAN", trajectory_id=trajectory_id, path=(attempt_base/"rml_plan").as_posix())
    else:
        prospective = plan(root, snapshot, attempt_base/"rml_plan")
    destination.mkdir(parents=True)
    package = destination/"inputs"
    names = subprocess.check_output(["git", "ls-files", "src/molgap"], text=True).splitlines()
    names = [name for name in names if name.endswith(".py")]
    names += [(BASE/name).as_posix() for name in ("run.py", "contract.json", "role_plan.json", "budget.json")]
    bundle = build_v4_source_bundle(repo_root=root, relative_paths=names, output_dir=package, source_commit=commit, archive_name="source_payload.bin")
    for arm, location in LOCATORS.items():
        for kind, original in (("model", "best_model.pt"), ("predictions", "development_predictions.pt")):
            source = root/location/original
            verify_file(source, ARMS[arm]["model_sha256" if kind == "model" else "payload_sha256"])
            shutil.copyfile(source, package/f"{arm}_{kind}.pt")
    shutil.copyfile(root/"experiments/pcqm_gptrans_author_alignment/recovered_reference/target_transform.json", package/"target_transform.json")
    shutil.copyfile(root/BASE/"contract.json", package/"contract.json")
    entry, metadata = root/BASE/"run.py", root/BASE/"kernel-metadata.json"
    meta = json.loads(metadata.read_text())
    release = dict(format=FORMAT, experiment_purpose="NO_TRAIN", source_commit=commit,
        archive_sha256=bundle["archive_sha256"], contract_sha256=sha256_file(root/BASE/"contract.json"),
        entry_sha256=sha256_file(entry), metadata_sha256=sha256_file(metadata),
        kernel=meta["id"], dataset_sources=meta["dataset_sources"],
        prospective_trajectory_id=trajectory_id, prospective_sha256=sha256_file(root/attempt_base/"rml_plan/trajectory.json"),
        files={name: sha256_file(package/name) for name in (
            "source_payload.bin", "SOURCE_FILES.json", "contract.json", "target_transform.json",
            "ema9999_model.pt", "ema9999_predictions.pt", "ema999_model.pt", "ema999_predictions.pt")})
    release["identity"] = canonical_fingerprint(release)
    atomic_json(package/"audit_release.json", release)
    atomic_json(package/"dataset-metadata.json", dict(title="MolGap GPTrans EMA Portability Inputs",
        id="kaseichou/molgap-gptrans-ema-portability-inputs", licenses=[dict(name="CC0-1.0")]))
    report = check_frozen_inference_release(package, entry, metadata)
    atomic_json(destination/"release_report.json", report)
    atomic_json(root/attempt_base/"release_binding.json", dict(release=release, prospective=prospective,
        staged_input_root=str(package), release_report=str(destination/"release_report.json")))
    print(json.dumps(dict(prospective=prospective, source=bundle, release_status=report["status"]), indent=2))


if __name__ == "__main__":
    main()
