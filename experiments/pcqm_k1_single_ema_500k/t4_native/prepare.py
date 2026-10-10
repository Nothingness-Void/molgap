"""Stage this platform control using existing RML/source/initialization owners."""
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
REL = HERE.relative_to(ROOT).as_posix()
RUN = "k1-single-ema-500k-t4-20261010"
TID = "TB-" + RUN
POLICY = "pcqm-k1-single-ema-500k-t4-native"
DATASET = "nothingnessvoid/molgap-k1-single-ema-500k-t4-20261010-source"
KERNEL = "nothingnessvoid/molgap-k1-single-ema-500k-t4-20261010"
BOOTSTRAP = "platforms/kaggle/training/k1_paired_500k.py"


def main():
    from molgap.research_memory.plan import plan
    from molgap.training_reproducibility import atomic_json, sha256_file
    from molgap.v4_bundle import build_v4_source_bundle

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    stage = ROOT / "platforms/_records/kaggle/staging" / RUN
    if stage.exists() or (HERE / "rml").exists():
        raise ValueError("Reconcile existing T4 preparation; no overwrite")
    parent_inputs = json.loads((PARENT / "inputs.json").read_text())
    initial = ROOT / "platforms/_records/colab/staging" / parent_inputs["run_id"] / "initial_state.pt"
    if sha256_file(initial) != parent_inputs["initial_file_sha256"]:
        raise ValueError("Parent frozen initialization changed")
    stage.mkdir(parents=True)
    source_dataset = stage / "source_dataset"
    source_dataset.mkdir()
    shutil.copyfile(initial, source_dataset / "initial_state.pt")
    inputs = dict(parent_inputs, source_commit=commit, run_id=RUN,
                  dataset="nothingnessvoid/pcqm4mv2-ogb-fixed-500k-scnet-v1",
                  allocation_wall_ceiling_seconds=32400, accelerator="T4")
    atomic_json(HERE / "inputs.json", inputs)
    shutil.copyfile(PARENT / "role_plan.json", HERE / "role_plan.json")
    specification = copy.deepcopy(json.loads((PARENT / "plan_input.json").read_text()))
    trajectory = specification["trajectory"]
    trajectory.update(trajectory_id=TID, family_id="k1-single-ema-500k-native-cost",
                      question="What is native T4 throughput for the frozen A100 single-forward/EMA500K recipe?")
    hypothesis = trajectory["hypothesis"]
    cost_id = "cost-k1-single-ema-500k-t4-native-planned"
    hypothesis.update(hypothesis_id="H-" + RUN,
        observed_deficiency="A100-only step timing cannot establish the paired recipe's T4 cost.",
        alternative_explanations=["GPU arithmetic limits", "Small-kernel overhead", "CPU or publication limits", "Cross-platform runtime differences"],
        changed_mechanism="No scientific mechanism change; explicitly qualified T4 execution control",
        cheapest_falsifier="One user-bounded nine-hour same-recipe T4 pair with retained step/epoch timing",
        decision_changed_if_positive="User may choose T4 for this workload; no automatic A100 stop",
        decision_changed_if_negative="Reserve the A100 decision for measured evidence; no automatic retry",
        expected_native_cost_ref=cost_id)
    trajectory["state_at_start"].update(source_commit=commit,
        source_config_identity=sha256_file(HERE / "inputs.json"),
        contract_refs=[f"{REL}/protocol.md", f"{REL}/inputs.json", f"{REL}/README.md"],
        role_snapshot_refs=[f"{REL}/role_plan.json"], budget_snapshot_ref=f"{REL}/protocol.md")
    trajectory["actions"] = [{"action_id": "A001", "type": "bounded_native_cost_paired_500k",
        "run_ids": [RUN + ":reference", RUN + ":ema999"], "attempt_ids": ["attempt-001"],
        "source_commit": commit, "evidence_refs": [f"{REL}/inputs.json"], "cost_event_ids": [cost_id]}]
    trajectory["decision"].update(decision_ref=f"{REL}/decision.md",
                                 next_allowed_actions=["One nine-hour Kaggle1 T4 pair"])
    trajectory.update(comparison_readiness_ref=f"{REL}/protocol.md", comparison_blockers=[
        "Native T4 qualification/timing pending", "Hardware/runtime control is not a strict scientific endpoint", "Consumed development"])
    cost = specification["costs"][0]
    cost.update(cost_event_id=cost_id, trajectory_id=TID, run_id=RUN, platform="kaggle",
                hardware="NVIDIA T4 physical allocation (one visible worker device)",
                evidence_ref=f"{REL}/protocol.md")
    # Budget the platform's possible two-card allocation, not just the used card.
    cost["measurement"]["wall_hours"] = {"value": 9.0, "status": "estimated"}
    cost["measurement"]["device_hours"] = {"value": 18.0, "status": "estimated"}
    specification["decision_state"].update(policy_id=POLICY,
        budget_snapshot_ref=f"{REL}/protocol.md", role_snapshot_refs=[f"{REL}/role_plan.json"],
        source_commit=commit, state_timestamp=datetime.now(timezone.utc).isoformat())
    policy = json.loads((ROOT / "research_memory/policies/pcqm-k1-single-ema-500k-a100.1.json").read_text())
    policy.update(policy_id=POLICY, comparability_selector={"scientific_contract": POLICY + "-v1"},
                  created_from_source_digest=sha256_file(HERE / "protocol.md"))
    atomic_json(ROOT / f"research_memory/policies/{POLICY}.1.json", policy)
    atomic_json(HERE / "plan_input.json", specification)
    atomic_json(HERE / "plan_receipt.json", plan(ROOT, specification, f"{REL}/rml"))
    allowlist = json.loads((PARENT / "source_allowlist.json").read_text())
    bundle = build_v4_source_bundle(repo_root=ROOT, relative_paths=allowlist + [BOOTSTRAP],
        output_dir=stage / "source", source_commit=commit)
    for name in ("source.tar.gz", "SOURCE_FILES.json", "SOURCE_COMMIT.txt", "SOURCE_ARCHIVE_SHA256.txt"):
        storage_name = "source_payload.bin" if name == "source.tar.gz" else name
        shutil.copyfile(stage / "source" / name, source_dataset / storage_name)
    for name in ("inputs.json", "protocol.md", "role_plan.json"):
        shutil.copyfile(HERE / name, source_dataset / name)
    for relative, name in (("trajectory.json", "prospective_trajectory.json"),
                           (f"costs/{cost_id}.json", "prospective_cost.json")):
        shutil.copyfile(HERE / "rml" / relative, source_dataset / name)
    runconfig = {"format": "molgap-colab-k1-paired500k-v2", "accelerator": "T4",
        "allocation_wall_limit_seconds": 32400, "source_commit": commit,
        "source_package_sha256": bundle["archive_sha256"],
        "job_id": "kaggle:" + KERNEL + ":attempt-001",
        "cpu_accepted_dataset_manifest_sha256": inputs["manifest_sha256"],
        "initial_format": "molgap-k1-single-ema-initial-v1",
        "initial_state_sha256": inputs["initial_tensor_sha256"]}
    files = {p.name: sha256_file(p) for p in source_dataset.iterdir() if p.is_file()}
    declaration = {"wall_limit_seconds": 32400, "files": files, "runconfig": runconfig,
        "source_archive_storage_name": "source_payload.bin",
        "graph_mount": "pcqm4mv2-ogb-fixed-500k-scnet-v1", "manifest_sha256": inputs["manifest_sha256"]}
    atomic_json(source_dataset / "k1_t4_payload.json", declaration)
    atomic_json(source_dataset / "dataset-metadata.json", {"id": DATASET,
        "title": "MolGap K1 Single EMA 500K T4 Source 20261010", "licenses": [{"name": "CC0-1.0"}]})
    kernel = stage / "kernel"
    kernel.mkdir()
    bootstrap = (ROOT / BOOTSTRAP).read_text().replace("__PAYLOAD_SHA256__", sha256_file(source_dataset / "k1_t4_payload.json"))
    with (kernel / "run.py").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(bootstrap)
    atomic_json(kernel / "kernel-metadata.json", {"id": KERNEL,
        "title": "MolGap K1 Single EMA 500K T4 20261010", "code_file": "run.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "dataset_sources": [inputs["dataset"], DATASET],
        "competition_sources": [], "kernel_sources": []})
    atomic_json(HERE / "submission/package_binding.json", {"run_id": RUN, "source": bundle,
        "payload_sha256": sha256_file(source_dataset / "k1_t4_payload.json"),
        "initial_file_sha256": inputs["initial_file_sha256"], "initial_tensor_sha256": inputs["initial_tensor_sha256"],
        "bootstrap_sha256": sha256_file(kernel / "run.py"),
        "kernel_metadata_sha256": sha256_file(kernel / "kernel-metadata.json"), "requested_kernel": KERNEL,
        "dataset": DATASET, "physical_job_identity": None})
    print(json.dumps({"prepared": True, "staging": str(stage), "source": bundle}))


if __name__ == "__main__":
    main()
