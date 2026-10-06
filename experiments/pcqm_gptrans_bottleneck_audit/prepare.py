"""Mechanical EMA asset retention and standard prospective packaging; no models run."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from molgap.gptrans_bottleneck import portability_indices, VARIANTS
from molgap.gptrans_portability import verify_file, check_rows
from molgap.training_reproducibility import atomic_json, atomic_torch_save, sha256_file, canonical_fingerprint
from molgap.v4_runtime import state_dict_sha256, normalized_source_sha256
from molgap.research_memory.plan import plan
from molgap.v4_bundle import build_v4_source_bundle
from molgap.frozen_inference_release import FORMAT, check_frozen_inference_release

BASE = "experiments/pcqm_gptrans_bottleneck_audit"
ASSETS = "platforms/_records/kaggle/bottleneck_assets_v1"
PATHS = {
    "reference": "platforms/_records/kaggle/training/gptrans_g1_input_ema_v1/gptrans_input_ema_screen/degree_scale_ema999/training",
    "local": "platforms/_records/kaggle/training/gptrans_capacity_relations_v1/gptrans_local_scale/degree_bond_local_ema999/training",
    "transition": "platforms/_records/kaggle/training/gptrans_pair_transition_v1/gptrans_pair_transition/degree_pair_transition_ema999/training",
}


def freeze_assets(root):
    import torch
    contract = json.loads((root/BASE/"contract.json").read_text())
    destination = root/ASSETS
    destination.mkdir(parents=True, exist_ok=False)
    authorities = {
        "local": BASE.replace("pcqm_gptrans_bottleneck_audit", "pcqm_gptrans_capacity_relations_100k")+"/gpu/retrieval_artifacts_v1.json",
        "transition": "experiments/pcqm_gptrans_pair_transition_100k/gpu/retrieval_artifacts_v1.json",
    }
    for name, kind, epoch in (("reference_19.pt", "reference", 19), ("reference_59.pt", "reference", 59),
            ("local_19.pt", "local", 19), ("local_59.pt", "local", 59),
            ("local_best.pt", "local", 41), ("transition_best.pt", "transition", 56)):
        original = root/PATHS[kind]/("best_model.pt" if "best" in name else f"checkpoint_epoch_{epoch:02d}.pt")
        digest = sha256_file(original)
        if kind in authorities:
            inventory = json.loads((root/authorities[kind]).read_text())["files"]
            bound = [row for row in inventory if row["path"].endswith(f"/{VARIANTS[kind]}/training/{original.name}")]
            if len(bound) != 1 or bound[0]["sha256"] != digest:
                raise ValueError("Source checkpoint differs from accepted retrieval")
        else:
            completion = json.loads((root/PATHS[kind]/"completion_manifest.json").read_text())
            if original.name == "checkpoint_epoch_59.pt" and digest != completion["checkpoint_sha256"]:
                raise ValueError("Reference terminal checkpoint differs")
            trace = json.loads((root/PATHS[kind]/"canonical_trace.json").read_text())
            if trace["observations"][epoch]["checkpoint_identity"] != "sha256:"+digest:
                raise ValueError("Reference checkpoint trace binding differs")
        saved = torch.load(original, map_location="cpu", weights_only=False)
        state = saved["model"] if "best" in name else saved["ema"]
        variant = saved["model_config"]["variant"] if "best" in name else saved["variant"]
        if variant != VARIANTS[kind] or saved["epoch"] != epoch or any(not torch.isfinite(t).all() for t in state.values()):
            raise ValueError("Frozen EMA metadata/state differs")
        state_sha = state_dict_sha256(state)
        payload = dict(format="molgap-frozen-diagnostic-EMA-v1", model=state, state_sha256=state_sha,
            source_checkpoint_sha256=digest, source_checkpoint=original.relative_to(root).as_posix(),
            epoch=epoch, variant=variant, target_stats=saved["target_stats"])
        atomic_torch_save(destination/name, payload)
        contract["model_assets"][name] = dict(kind=kind, epoch=epoch, sha256=sha256_file(destination/name),
            state_sha256=state_sha, source_checkpoint_sha256=digest, source_checkpoint=payload["source_checkpoint"])
    for kind, name in (("reference", "reference_predictions.pt"), ("local", "local_predictions.pt")):
        original = root/PATHS[kind]/"development_predictions.pt"
        if kind == "local":
            verify_file(original, "330e7e5407bad13a849559a47dba75717d63d5d62a8ec6a1e168c4a3377bd7e0")
        else:
            verify_file(original, "39a094e2ce9cc1d5aa01e637c37f5b90a5af728dc2136e8751797ae30fc38108")
        shutil.copyfile(original, destination/name)
        contract["reference_payloads"][name] = dict(sha256=sha256_file(destination/name), original=original.relative_to(root).as_posix())
    later = root/"platforms/_records/kaggle/training/gptrans_ema_portability_v4/gptrans_ema_portability/ema999/unseen_500k"
    progress = json.loads((later/"progress.json").read_text())
    parts = []
    for item in progress["chunks"]:
        verify_file(later/item["file"], item["sha256"])
        parts.append(torch.load(later/item["file"], map_location="cpu", weights_only=False))
    joined = {key:torch.cat([part[key] for part in parts]) for key in ("source_idx", "target_eV", "prediction_eV")}
    check_rows(joined, 500000, 50000)
    offsets = torch.tensor(portability_indices(), dtype=torch.long)
    atomic_torch_save(destination/"reference_later.pt", {k:v[offsets] for k,v in joined.items()})
    contract["reference_payloads"]["reference_later.pt"] = dict(sha256=sha256_file(destination/"reference_later.pt"),
        original_progress=later.relative_to(root).as_posix()+"/progress.json", original_progress_sha256=sha256_file(later/"progress.json"))
    for path in ("src/molgap/gptrans.py", "src/molgap/gptrans_capacity.py", "src/molgap/gptrans_pair_transition.py"):
        contract["source_identities"][path] = normalized_source_sha256(root/path)
    atomic_json(root/BASE/"contract.json", contract)
    shutil.copyfile(root/"experiments/pcqm_gptrans_author_alignment/recovered_reference/target_transform.json", destination/"target_transform.json")
    return contract


def prepare(root, output):
    if output.exists():
        raise ValueError("Never overwrite a frozen diagnostic package")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    snapshot = deepcopy(json.loads((root/"experiments/pcqm_gptrans_ema_portability/attempt_v4/rml_plan/trajectory.json").read_text()))
    trajectory = snapshot
    tid, run = "TC-gptrans-bottleneck-frozen-v1", "gptrans-bottleneck-audit:v1"
    cost_id = "cost-"+tid
    references = ["pcqm-gptrans-g1-degree-scale-ema999-100k-s42", "pcqm-gptrans-g1-degree-bond-local-ema999-100k-s42"]
    trajectory.update(trajectory_id=tid, family_id="gptrans-frozen-bottleneck-diagnostic",
        question="Is local-update strength, output reachability or cohort dependence limiting accepted GPTrans variants?")
    trajectory.pop("decision_state", None)
    trajectory["state_at_start"].update(source_commit=commit, source_config_identity=sha256_file(root/BASE/"contract.json"),
        contract_refs=[BASE+"/contract.json", BASE+"/protocol.md"], reference_ids=references,
        prior_evidence_ids=references, prior_trajectory_ids=["TC-gptrans-g1-pair-transition-100k-s42"],
        parent_trajectory_ids=[], role_snapshot_refs=[BASE+"/role_plan.json"], budget_snapshot_ref=BASE+"/budget.json")
    trajectory["hypothesis"].update(hypothesis_id="H-"+tid,
        observed_deficiency=trajectory["question"], supporting_evidence_ids=references,
        alternative_explanations=["ordinary finite-data fitting", "late output disconnection", "cohort-dependent gain"],
        changed_mechanism="none: frozen EMA eval-only derivatives and inference",
        cheapest_falsifier="same512 panel, accepted checkpoints,10000 frozen later rows",
        expected_native_cost_ref=cost_id, decision_changed_if_positive="separately freeze one targeted mechanism",
        decision_changed_if_negative="close unchanged local-addon scale-up",
        historical_unknowns=["no causal amplitude intervention; no500K optimization"])
    trajectory["actions"] = [dict(action_id="A001", type="NO_TRAIN_frozen_diagnostic", source_commit=commit,
        run_ids=[run], attempt_ids=["v1"], evidence_refs=[BASE+"/contract.json"], cost_event_ids=[cost_id])]
    trajectory["result"] = dict(evidence_ids=[], evidence_refs=[])
    trajectory["decision"] = dict(decision_ref=BASE+"/decision.md", outcome="ACTIVE",
        next_allowed_actions=["bounded frozen diagnostic and terminal interpretation"], reopen_conditions=["terminal or execution fault"])
    missing = dict(status="measurement_missing", value=None)
    spec = dict(trajectory=trajectory, decision_state=dict(known_trajectory_ids=[], known_evidence_ids=[],
        active_reference_ids=references, available_actions=["RUN_BOTTLENECK_AUDIT", "DEFER"], chosen_action="RUN_BOTTLENECK_AUDIT",
        policy_id="gptrans-bottleneck-audit", policy_version="v1", budget_snapshot_ref=BASE+"/budget.json",
        role_snapshot_refs=[BASE+"/role_plan.json"], state_timestamp=datetime.now(timezone.utc).isoformat(), source_commit=commit),
        costs=[dict(schema="molgap-cost-event-v1", cost_event_id=cost_id, trajectory_id=tid, action_id="A001", run_id=run,
            attempt_id="v1", category="audit", platform="kaggle2", hardware="Tesla_T4x2", evidence_ref=BASE+"/budget.json",
            measurement=dict(device_hours=dict(status="estimated", value=.67), wall_hours=dict(status="estimated", value=.335),
                cpu_hours=missing, queue_hours=missing))])
    prospective = plan(root, spec, root/BASE/"rml_plan")
    output.mkdir(parents=True)
    inputs = output/"inputs"
    names = [n for n in subprocess.check_output(["git", "ls-files", "src/molgap"], text=True).splitlines() if n.endswith(".py")]
    names += [BASE+"/"+name for name in ("run.py", "contract.json", "protocol.md")]
    bundle = build_v4_source_bundle(repo_root=root, relative_paths=names, output_dir=inputs,
        source_commit=commit, archive_name="source_payload.bin")
    contract = json.loads((root/BASE/"contract.json").read_text())
    for name in [*contract["model_assets"], *contract["reference_payloads"], "target_transform.json"]:
        shutil.copyfile(root/ASSETS/name, inputs/name)
    shutil.copyfile(root/BASE/"contract.json", inputs/"contract.json")
    release = dict(format=FORMAT, experiment_purpose="NO_TRAIN", source_commit=commit, archive_sha256=bundle["archive_sha256"],
        contract_sha256=sha256_file(inputs/"contract.json"), entry_sha256=sha256_file(root/BASE/"run.py"),
        metadata_sha256=sha256_file(root/BASE/"kernel-metadata.json"), kernel="kaseichou/molgap-gptrans-bottleneck-audit",
        dataset_sources=json.loads((root/BASE/"kernel-metadata.json").read_text())["dataset_sources"],
        prospective_trajectory_id=tid, prospective_sha256=sha256_file(root/BASE/"rml_plan/trajectory.json"),
        files={name:sha256_file(inputs/name) for name in ("source_payload.bin", "SOURCE_FILES.json", "contract.json",
            "target_transform.json", *contract["model_assets"], *contract["reference_payloads"])})
    release["identity"] = canonical_fingerprint(release)
    atomic_json(inputs/"audit_release.json", release)
    atomic_json(inputs/"dataset-metadata.json", dict(id="kaseichou/molgap-gptrans-bottleneck-inputs",
        title="MolGap GPTrans Bottleneck Inputs", licenses=[dict(name="CC0-1.0")]))
    report = check_frozen_inference_release(inputs, root/BASE/"run.py", root/BASE/"kernel-metadata.json")
    atomic_json(output/"release_report.json", report)
    atomic_json(root/BASE/"release_binding.json", dict(release=release, prospective=prospective, inputs=str(inputs),
        release_report=str(output/"release_report.json")))
    return report["status"]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-assets", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(freeze_assets(Path.cwd()) if args.freeze_assets else prepare(Path.cwd(), args.output.resolve()), indent=2))
