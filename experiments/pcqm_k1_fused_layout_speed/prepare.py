"""Freeze a local speed adapter with existing source and RML owners."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from molgap.research_memory.plan import plan_many
from molgap.training_reproducibility import atomic_json, canonical_fingerprint, sha256_file
from molgap.v4_bundle import build_v4_source_bundle
from molgap.v4_runtime import validate_standard_source_bundle

REL = "experiments/pcqm_k1_fused_layout_speed"
RUN = "k1-fused-layout-speed-100k-20261011"


def prepare(root):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    folder = root / REL
    inputs = {"format": "molgap-k1-local-speed-v1", "epochs": 10,
        "allocation_seconds": 3600, "schedule_epochs": 40, "batch_size": 128,
        "precision": "fp32-tf32-off", "development_access": False,
        "initial_format": "molgap-k1-single-ema-initial-v1",
        "initial_file_sha256": "0ce2cb3206a7a59a059b5efb09922a5d6e4478058ce2dbfec1434f3086fb5a80",
        "initial_tensor_sha256": "8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd",
        "source_commit": commit, "arms": ["reference", "fused_layout"]}
    atomic_json(folder / "inputs.json", inputs)
    timestamp = datetime.now(timezone.utc).isoformat()
    plans = []
    prior = "pcqm-k1-colab-execution-profile-20261007"
    for arm in inputs["arms"]:
        tid = f"TB-k1-local-speed-10ep-{arm}-20261011"
        cost = f"cost-k1-local-speed-{arm}-estimate-20261011"
        trajectory = {
            "trajectory_id": tid, "track": "B", "owner": "desktop",
            "family_id": "k1-local-fused-layout-speed", "question": "Does fused AdamW plus CPU layout reduce real100K TRAIN execution time?",
            "hypothesis": {"hypothesis_id": f"H-k1-local-speed-{arm}-20261011",
                "supporting_evidence_ids": [prior],
                "observed_deficiency": "Short joint scratch gain is not a real-stream end-to-end measurement",
                "changed_mechanism": "Explicit unfused control versus fused AdamW and CPU layout, no TF32",
                "cheapest_falsifier": "Ten TRAIN epochs per arm within shared3600s, no development reads",
                "alternative_explanations": ["Short-window gain is warmup-sensitive", "Shared loading and saves dominate", "RTX5060 does not transfer to A100/T4"],
                "decision_changed_if_positive": "Retain local measured gain for a separately authorized native/quality qualification",
                "decision_changed_if_negative": "Do not nominate this joint runtime from isolated microbenchmarks",
                "expected_native_cost_ref": cost,
                "related_closed_family_ids": ["k1-execution-profile-a100"]},
            "state_at_start": {"source_commit": commit,
                "source_config_identity": canonical_fingerprint(inputs),
                "contract_refs": [f"{REL}/protocol.md", f"{REL}/inputs.json"],
                "reference_ids": [], "parent_trajectory_ids": [], "prior_evidence_ids": [prior],
                "role_snapshot_refs": [f"{REL}/role_plan.json"], "budget_snapshot_ref": f"{REL}/protocol.md",
                "same_run_replay": {"schema": "molgap-same-run-replay-binding-v1",
                    "spec_identity": canonical_fingerprint(inputs), "logical_run_id": RUN,
                    "arm_id": arm, "comparison_role": "reference" if arm == "reference" else "candidate",
                    "reference_arm_id": "reference", "reference_trajectory_id": "TB-k1-local-speed-10ep-reference-20261011",
                    "reference_trajectory_ref": f"{REL}/reference/rml/trajectory.json"}},
            "actions": [{"action_id": "A001", "type": "execution_speed_training_prefix",
                "run_ids": [RUN], "attempt_ids": ["local-attempt001"], "source_commit": commit,
                "evidence_refs": [f"{REL}/inputs.json"], "cost_event_ids": [cost]}],
            "decision": {"outcome": "ACTIVE", "decision_ref": f"{REL}/decision.md",
                "next_allowed_actions": ["Authorized local speed prefix only"], "reopen_conditions": []},
            "comparison_class": "CONTEXT_ONLY", "comparison_readiness_ref": f"{REL}/protocol.md",
            "comparison_blockers": ["Speed-only prefix; no scientific endpoint", "RTX5060 only"], "reference_bundle_id": None}
        event = {"schema": "molgap-cost-event-v1", "cost_event_id": cost, "trajectory_id": tid,
            "action_id": "A001", "run_id": RUN, "attempt_id": "local-attempt001", "platform": "local",
            "hardware": "NVIDIA GeForce RTX5060", "category": "training", "evidence_ref": f"{REL}/protocol.md",
            "measurement": {"device_hours": {"value": 0.5, "status": "estimated"},
                "wall_hours": {"value": 0.5, "status": "estimated"},
                "cpu_hours": {"value": None, "status": "measurement_missing"},
                "queue_hours": {"value": None, "status": "not_applicable"}}}
        spec = {"trajectory": trajectory, "costs": [event], "decision_state": {
            "available_actions": ["RUN_DIAGNOSTIC", "NO_TRAIN"], "chosen_action": "RUN_DIAGNOSTIC",
            "policy_id": "pcqm-k1-local-speed", "policy_version": "1", "state_timestamp": timestamp}}
        atomic_json(folder / f"{arm}_plan.json", spec)
        plans.append({"spec": spec, "output": f"{REL}/{arm}/rml"})
    receipt = plan_many(root, plans)
    atomic_json(folder / "plan_receipt.json", receipt)
    package(root, inputs)


def package(root, inputs):
    folder = root / REL
    commit = inputs["source_commit"]
    for arm in inputs["arms"]:
        trajectory = json.loads((folder / arm / "rml/trajectory.json").read_text())
        if trajectory["state_at_start"]["source_commit"] != commit:
            raise ValueError("Published source identity changed")
        for pointer in (f"{REL}/protocol.md", f"{REL}/inputs.json", f"{REL}/role_plan.json", f"{REL}/decision.md"):
            if sha256_file(root / pointer) != trajectory["decision_state"]["source_hashes"][pointer]:
                raise ValueError(f"Published planning input changed: {pointer}")
    staging = root / "platforms/_records/local/staging" / (RUN + "-source2")
    # Explicit reviewed import closure, including the legacy shard pickle owner.
    names = ["__init__", "constants", "k1_local_speed", "k1_screen_training", "k1_execution_layout",
        "k1_loader_reuse", "packed_graph_dataset", "qm9_neural_atom", "pcqm_gap_architecture",
        "gps", "edge_state_gps", "screen_policy", "training_reproducibility", "v4_runtime",
        "pcqm_wedge"]
    source = build_v4_source_bundle(repo_root=root,
        relative_paths=[f"src/molgap/{name}.py" for name in names], output_dir=staging,
        source_commit=commit)
    extracted = staging / "frozen"
    extracted.mkdir(exist_ok=True)
    if any(extracted.iterdir()):
        raise ValueError("Recovery requires an empty, unpublished extracted directory")
    validate_standard_source_bundle(Path(source["archive"]), source["archive_sha256"], commit)
    with tarfile.open(source["archive"], "r:gz") as archive:
        # Validated regular-file inventory only; portable to the project Python.
        for member in archive.getmembers():
            target = extracted / member.name
            target.resolve().relative_to(extracted.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as handle, target.open("wb") as output:
                import shutil
                shutil.copyfileobj(handle, output)
    for name in ("SOURCE_COMMIT.txt", "SOURCE_FILES.json", "SOURCE_ARCHIVE_SHA256.txt"):
        import shutil
        shutil.copyfile(staging / name, extracted / name)
    config = copy.deepcopy(inputs)
    config.update(source_archive=source["archive"], source_package_sha256=source["archive_sha256"],
                  source_inventory_sha256=sha256_file(extracted / "SOURCE_FILES.json"))
    config.update(input_root="D:/文档/molgap/data/pcqm_fixed_100k_v1",
        initial_path="D:/w/k1-colab-500k-efficient/platforms/_records/colab/staging/k1-single-ema-500k-a100-20261010/initial_state.pt",
        prospective={arm: {"path": str(folder / arm / "rml/trajectory.json"),
            "sha256": sha256_file(folder / arm / "rml/trajectory.json")} for arm in inputs["arms"]})
    config_path = staging / "config.json"
    atomic_json(config_path, config)
    atomic_json(folder / "release.json", {"source": source, "source_root": str(extracted),
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "output": str(staging / "run"), "submitted_remote": False})
    print(json.dumps({"source_root": str(extracted), "config": str(config_path), "output": str(staging / "run")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--recover-package", action="store_true")
    args = parser.parse_args()
    root = args.repo_root.resolve()
    if args.recover_package:
        package(root, json.loads((root / REL / "inputs.json").read_text()))
    else:
        prepare(root)
