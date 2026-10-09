"""Mechanical asset retention plus existing prospective/package/release APIs."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

from .gptrans_triplet_portability import BASE, PROFILE, KERNEL, DATASET, MODE, MODEL_SHA, PAYLOAD_SHA
from .gptrans_portability import verify_file, check_rows
from .gptrans_bottleneck import portability_indices
from .training_reproducibility import atomic_json, atomic_torch_save, sha256_file, canonical_fingerprint
from .v4_runtime import state_dict_sha256, normalized_source_sha256

ASSETS = "platforms/_records/kaggle/triplet_portability_assets_v1"
CANDIDATE = "platforms/_records/kaggle/training/gptrans_triplet_communication_v1/gptrans_triplet_communication/" + MODE + "/training"
PARENT_BASE = "experiments/pcqm_gptrans_bottleneck_audit"


def freeze_assets(root):
    """Only inspect/copy retained CPU tensors; do not construct or execute a model."""
    import torch
    destination = root / ASSETS
    if (root / BASE / "contract.json").exists():
        raise ValueError("Never refreeze a completed asset contract")
    destination.mkdir(parents=True, exist_ok=True)
    original = root / CANDIDATE / "best_model.pt"
    verify_file(original, MODEL_SHA)
    predictions = root / CANDIDATE / "development_predictions.pt"
    verify_file(predictions, PAYLOAD_SHA)
    saved = torch.load(original, map_location="cpu", weights_only=False)
    if (saved["model_config"]["variant"] != MODE or saved["epoch"] != 38
            or sum(t.numel() for t in saved["model"].values()) != 5880961
            or any(not torch.isfinite(t).all() for t in saved["model"].values())):
        raise ValueError("Accepted triplet state/metadata changed")
    digest = state_dict_sha256(saved["model"])
    atomic_torch_save(destination / "local_best.pt", dict(format="molgap-frozen-diagnostic-EMA-v1",
        model=saved["model"], state_sha256=digest, source_checkpoint_sha256=MODEL_SHA,
        source_checkpoint=original.relative_to(root).as_posix(), epoch=38, variant=MODE,
        target_stats=saved["target_stats"]))
    shutil.copyfile(predictions, destination / "local_predictions.pt")
    parent = json.loads((root / PARENT_BASE / "contract.json").read_text())
    parent_predictions = root / parent["reference_payloads"]["local_predictions.pt"]["original"]
    verify_file(parent_predictions, parent["reference_payloads"]["local_predictions.pt"]["sha256"])
    shutil.copyfile(parent_predictions, destination / "reference_predictions.pt")
    accepted = json.loads((root / PARENT_BASE / "results/acceptance_summary.json").read_text())
    if accepted["accepted"] is not True:
        raise ValueError("Parent frozen-cohort authority not accepted")
    later = root / "platforms/_records/kaggle/training/gptrans_bottleneck_v1/gptrans_bottleneck/portability/unseen_500k"
    progress = json.loads((later / "progress.json").read_text())
    manifest_path = later.parent.parent / "output_manifest.json"
    evidence = json.loads((root / PARENT_BASE / "results/terminal_evidence.json").read_text())
    binding = next(row for row in evidence["artifacts"] if row["name"] == "output_manifest")
    verify_file(manifest_path, binding["sha256"])
    manifest = json.loads(manifest_path.read_text())
    verify_file(later / "progress.json", manifest["files"]["portability/unseen_500k/progress.json"])
    if progress["model_sha256"] != parent["model_assets"]["local_best.pt"]["sha256"]:
        raise ValueError("Later parent predictions belong to another checkpoint")
    parts = []
    for number, row in enumerate(progress["chunks"]):
        if row["file"] != f"chunk_{number:02d}.pt" or row["rows"] != 5000:
            raise ValueError("Parent later chunks differ")
        verify_file(later / row["file"], row["sha256"])
        verify_file(later / row["file"], manifest["files"]["portability/unseen_500k/" + row["file"]])
        parts.append(torch.load(later / row["file"], map_location="cpu", weights_only=False))
    joined = {k: torch.cat([p[k] for p in parts]) for k in ("source_idx", "target_eV", "prediction_eV")}
    if len(parts) != 2 or not torch.equal(joined["source_idx"], torch.tensor(portability_indices() + 500000)):
        raise ValueError("Parent later cohort differs")
    atomic_torch_save(destination / "reference_later.pt", joined)
    transform = root / "experiments/pcqm_gptrans_author_alignment/recovered_reference/target_transform.json"
    shutil.copyfile(transform, destination / "target_transform.json")
    payload = torch.load(predictions, map_location="cpu", weights_only=False)
    if any(payload[k] is not False for k in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read")):
        raise ValueError("Retained prediction role claims changed")
    check_rows({k: payload[k] for k in ("source_idx", "target_eV", "prediction_eV")}, 100000, 50000)
    source_names = ("gptrans.py", "gptrans_capacity.py", "gptrans_local_relation.py", "gptrans_triplet_communication.py")
    contract = dict(format="molgap-gptrans-triplet-portability-contract-v1", release_profile=PROFILE,
        experiment_purpose="NO_TRAIN", training_executed=False, optimizer_steps=0,
        physical_batch=128, precision="fp32", tf32_enabled=False, allocation_cap_seconds=1800,
        roles={"original_100k": [100000,150000], "unseen_500k": [500000,550000]},
        workers=["portability"], portability_rows=10000, protected_roles_read=False,
        portability_sampler="numpy-RandomState42-choice50000-without-replacement-sorted",
        target_transform_asset=json.loads(transform.read_text())["asset_sha256"],
        cohort_is_reused_development=True, weight_mode="eval-EMA-frozen-no-update",
        original_gain_eV=.003049777398109436, retained_fraction_gate=.5,
        automatic_training_released=False,
        model_assets={"local_best.pt": dict(kind="triplet_aggregate", epoch=38, parameter_count=5880961,
            sha256=sha256_file(destination / "local_best.pt"), state_sha256=digest,
            source_checkpoint_sha256=MODEL_SHA, source_checkpoint=original.relative_to(root).as_posix())},
        reference_payloads={name: dict(sha256=sha256_file(destination / name)) for name in
            ("local_predictions.pt", "reference_predictions.pt", "reference_later.pt")},
        parent_checkpoint=parent["model_assets"]["local_best.pt"],
        parent_later_manifest=dict(locator=manifest_path.relative_to(root).as_posix(), sha256=binding["sha256"]),
        source_identities={"src/molgap/" + name: normalized_source_sha256(root / "src/molgap" / name) for name in source_names})
    atomic_json(root / BASE / "contract.json", contract)
    return contract


def prepare(root, output):
    from .research_memory.plan import plan
    from .v4_bundle import build_v4_source_bundle
    from .frozen_inference_release import FORMAT, check_frozen_inference_release
    if output.exists():
        raise ValueError("Never overwrite a frozen release")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    trajectory = deepcopy(json.loads((root / PARENT_BASE / "rml_plan/trajectory.json").read_text()))
    trajectory.pop("decision_state", None)
    tid, run = "TC-gptrans-triplet-portability-s42-v1", "gptrans-triplet-portability:v1"
    refs = ["pcqm-gptrans-g1-degree-bond-local-ema999-100k-s42", "pcqm-gptrans-g1-degree-local-triplet-aggregate-ema999-100k-s42", "pcqm-gptrans-bottleneck-frozen-v1"]
    trajectory.update(trajectory_id=tid, family_id="gptrans-triplet-portability", question="Does frozen triplet aggregation retain its100K gain on the predeclared later500K development cohort?")
    trajectory["state_at_start"].update(source_commit=commit, source_config_identity=sha256_file(root / BASE / "contract.json"),
        contract_refs=[BASE + "/contract.json", BASE + "/protocol.md"], reference_ids=refs,
        prior_evidence_ids=refs, prior_trajectory_ids=["TC-gptrans-g1-local-triplet-aggregate-100k-s42", "TC-gptrans-bottleneck-frozen-v1"],
        parent_trajectory_ids=[], role_snapshot_refs=[BASE + "/role_plan.json"], budget_snapshot_ref=BASE + "/budget.json")
    trajectory["hypothesis"].update(hypothesis_id="H-" + tid, observed_deficiency=trajectory["question"],
        supporting_evidence_ids=refs, alternative_explanations=["cohort-sensitive100K fitting", "useful transferable pair communication"],
        changed_mechanism="none; frozen checkpoint inference only", cheapest_falsifier="reproduce50000 accepted rows then infer10000 predeclared later rows", expected_native_cost_ref="cost-" + tid,
        decision_changed_if_positive="consider one separately frozen500K training study", decision_changed_if_negative="close unchanged triplet scale-up",
        historical_unknowns=["no500K optimization; reused development, not sealed confirmation; one seed"])
    trajectory["actions"] = [dict(action_id="A001", type="NO_TRAIN_frozen_inference", source_commit=commit, run_ids=[run], attempt_ids=["v1"], evidence_refs=[BASE + "/contract.json"], cost_event_ids=["cost-" + tid])]
    trajectory["result"] = dict(evidence_ids=[], evidence_refs=[])
    trajectory["decision"] = dict(decision_ref=BASE + "/decision.md", outcome="ACTIVE", next_allowed_actions=["frozen inference and independent terminal acceptance"], reopen_conditions=["terminal state or actionable execution fault"])
    missing = dict(status="measurement_missing", value=None)
    prospective = plan(root, dict(trajectory=trajectory, decision_state=dict(known_trajectory_ids=[], known_evidence_ids=[], active_reference_ids=refs,
        available_actions=["RUN_BOTTLENECK_AUDIT", "DEFER"], chosen_action="RUN_BOTTLENECK_AUDIT", policy_id="gptrans-bottleneck-audit", policy_version="v1",
        budget_snapshot_ref=BASE + "/budget.json", role_snapshot_refs=[BASE + "/role_plan.json"], state_timestamp=datetime.now(timezone.utc).isoformat(), source_commit=commit),
        costs=[dict(schema="molgap-cost-event-v1", cost_event_id="cost-" + tid, trajectory_id=tid, action_id="A001", run_id=run, attempt_id="v1", category="audit", platform="kaggle2", hardware="Tesla_T4_up_to2", evidence_ref=BASE + "/budget.json",
            measurement=dict(device_hours=dict(status="estimated", value=.3), wall_hours=dict(status="estimated", value=.15), cpu_hours=missing, queue_hours=missing))]), root / BASE / "rml_plan")
    inputs = output / "inputs"
    names = [n for n in subprocess.check_output(["git", "ls-files", "src/molgap"], text=True).splitlines() if n.endswith(".py")]
    names += [BASE + "/" + n for n in ("run.py", "contract.json", "protocol.md")]
    bundle = build_v4_source_bundle(repo_root=root, relative_paths=names, output_dir=inputs, source_commit=commit, archive_name="source_payload.bin")
    contract = json.loads((root / BASE / "contract.json").read_text())
    for name in [*contract["model_assets"], *contract["reference_payloads"], "target_transform.json"]:
        shutil.copyfile(root / ASSETS / name, inputs / name)
    shutil.copyfile(root / BASE / "contract.json", inputs / "contract.json")
    release = dict(format=FORMAT, experiment_purpose="NO_TRAIN", source_commit=commit, archive_sha256=bundle["archive_sha256"],
        contract_sha256=sha256_file(inputs / "contract.json"), entry_sha256=sha256_file(root / BASE / "run.py"),
        metadata_sha256=sha256_file(root / BASE / "kernel-metadata.json"), kernel=KERNEL,
        dataset_sources=json.loads((root / BASE / "kernel-metadata.json").read_text())["dataset_sources"],
        prospective_trajectory_id=tid, prospective_sha256=sha256_file(root / BASE / "rml_plan/trajectory.json"),
        files={name: sha256_file(inputs / name) for name in ("source_payload.bin", "SOURCE_FILES.json", "contract.json", "target_transform.json", *contract["model_assets"], *contract["reference_payloads"])})
    release["identity"] = canonical_fingerprint(release)
    atomic_json(inputs / "audit_release.json", release)
    atomic_json(inputs / "dataset-metadata.json", dict(id=DATASET, title="MolGap GPTrans Triplet Portability Inputs", licenses=[dict(name="CC0-1.0")]))
    report = check_frozen_inference_release(inputs, root / BASE / "run.py", root / BASE / "kernel-metadata.json")
    atomic_json(output / "release_report.json", report)
    atomic_json(root / BASE / "release_binding.json", dict(release=release, prospective=prospective, inputs=str(inputs), release_report=str(output / "release_report.json")))
    return report["status"]
