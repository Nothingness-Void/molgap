"""Freeze prospective first, then stage train-only CPU graphs and notebook payload."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-k1-colab-execution-profile-20261007"
RUN = "k1-profile-a100-20261007"
POLICY = "pcqm-k1-colab-execution-profile"


def main():
    import numpy as np
    from molgap.research_memory.plan import plan
    from molgap.training_reproducibility import atomic_json, sha256_file
    prior = json.loads((ROOT / "experiments/pcqm_k1_500k_bn_calibration/inputs.json").read_text())
    selected = prior["checkpoints"]["best_model.pt"]
    cache = Path(prior["cache_root"])
    if sha256_file(Path(selected["path"])) != selected["sha256"] or sha256_file(cache / "manifest.json") != prior["manifest_sha256"]:
        raise ValueError("Accepted input differs")
    names = ["__init__", "constants", "screen_policy", "qm9_neural_atom", "pcqm_gap_architecture",
             "gps", "k1_screen_training", "training_reproducibility", "v4_runtime", "k1_pretrained_combo"]
    files = {f"src/molgap/{name}.py": prior["frozen_source_files"][f"src/molgap/{name}.py"] for name in names if name != "__init__"}
    frozen = Path(prior["frozen_source_root"]).parent
    files["src/molgap/__init__.py"] = sha256_file(frozen / "src/molgap/__init__.py")
    for name,digest in files.items():
        if sha256_file(frozen / name) != digest:
            raise ValueError(f"Frozen source differs: {name}")
    sample = np.random.default_rng(20261007).choice(500000,4096,replace=False).tolist()
    inputs = {"source_commit": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "accepted_owner_commit": prior["owner_commit"], "selected": selected,
        "cache_root": prior["cache_root"], "manifest_sha256": prior["manifest_sha256"],
        "frozen_source_files": files, "sample_source_idx": sample, "worker_ceiling_seconds":1200}
    atomic_json(HERE / "inputs.json",inputs)
    atomic_json(HERE / "role_plan.json", {"train": {"range":[0,500000], "sample_rows":4096,
        "local_decoded_rows":500000,"selection_used":False},"development":"untouched",
        "official_validation":"untouched","test_dev":"untouched","test_challenge":"untouched"})
    refs = ["pcqm-k1-500k-bottleneck-diagnostic-20261007", "pcqm-k1-500k-bn-calibration-20261007"]
    policy = json.loads((ROOT / "research_memory/policies/pcqm-k1-500k-bn-calibration.1.json").read_text())
    policy.update(policy_id=POLICY,comparability_selector={"scientific_contract":POLICY+"-v1"},
        created_from_source_digest=sha256_file(HERE / "protocol.md"))
    atomic_json(ROOT / f"research_memory/policies/{POLICY}.1.json",policy)
    state = {"source_commit": inputs["source_commit"], "source_config_identity":sha256_file(HERE / "inputs.json"),
        "contract_refs":[f"{REL}/protocol.md",f"{REL}/inputs.json"],"reference_ids":[],
        "parent_trajectory_ids":[],"prior_evidence_ids":refs,
        "role_snapshot_refs":[f"{REL}/role_plan.json"],"budget_snapshot_ref":f"{REL}/protocol.md"}
    cost_id = "cost-k1-colab-profile-estimate"
    trajectory = {"schema":"molgap-trajectory-v1","trajectory_id":TID,"record_mode":"prospective",
        "track":"B","owner":"desktop","family_id":"k1-execution-profile-a100",
        "question":"Which K1 two-forward execution costs dominate and are selected-state consistency gradients aligned?",
        "hypothesis":{"hypothesis_id":"H-k1-colab-profile-20261007","supporting_evidence_ids":refs,
            "observed_deficiency":"Historical K1 epoch windows are43.8% slower than GPTrans; package accuracy flat.",
            "alternative_explanations":["CPU loader contention","local edge operators dominate","consistency gradients interfere","A100 behavior differs from T4"],
            "changed_mechanism":"Execution-only scratch forward/loader variants and selected-state gradient probe",
            "cheapest_falsifier":"4096 train members, four bounded cases and four gradient batches under1200s",
            "decision_changed_if_positive":"Nominate one measured bottleneck for separate runtime/MAE qualification",
            "decision_changed_if_negative":"Do not optimize an unmeasured assumed bottleneck",
            "expected_native_cost_ref":cost_id,"related_closed_family_ids":["k1-500k-bn-calibration"]},
        "state_at_start":state,"actions":[{"action_id":"A001","type":"execution_profile",
            "run_ids":[RUN],"attempt_ids":["attempt-001"],"source_commit":inputs["source_commit"],
            "evidence_refs":[f"{REL}/inputs.json"],"cost_event_ids":[cost_id]}],
        "result":{"evidence_ids":[],"evidence_refs":[]},"decision":{"outcome":"ACTIVE",
            "decision_ref":f"{REL}/decision.md","next_allowed_actions":["Bounded A100 execution-only analysis"],"reopen_conditions":[]},
        "comparison_class":"CONTEXT_ONLY","comparison_readiness_ref":f"{REL}/protocol.md",
        "comparison_blockers":["A100 not native historical T4","execution profile not accuracy comparison"],"reference_bundle_id":None}
    cost = {"schema":"molgap-cost-event-v1","cost_event_id":cost_id,"trajectory_id":TID,
        "action_id":"A001","run_id":RUN,"attempt_id":"attempt-001","platform":"colab",
        "hardware":"NVIDIA A100-SXM4-40GB","category":"preflight","evidence_ref":f"{REL}/protocol.md",
        "measurement":{key:{"value":1200/3600 if key in ("device_hours","wall_hours") else None,
            "status":"estimated" if key in ("device_hours","wall_hours") else "measurement_missing"}
            for key in ("device_hours","wall_hours","cpu_hours","queue_hours")}}
    spec = {"trajectory":trajectory,"costs":[cost],"decision_state":{"known_trajectory_ids":[],
        "known_evidence_ids":[],"active_reference_ids":[],"available_actions":["RUN_DIAGNOSTIC","NO_TRAIN"],
        "chosen_action":"RUN_DIAGNOSTIC","policy_id":POLICY,"policy_version":"1",
        "budget_snapshot_ref":f"{REL}/protocol.md","role_snapshot_refs":[f"{REL}/role_plan.json"],
        "source_commit":inputs["source_commit"],"state_timestamp":datetime.now(timezone.utc).isoformat()}}
    atomic_json(HERE / "plan_input.json",spec)
    atomic_json(HERE / "plan_receipt.json",plan(ROOT,spec,f"{REL}/rml"))
    # No graph decode/model execution occurs before the canonical prospective above.
    payload = ROOT / "platforms/_records/colab/staging/k1-profile-a100-20261007/payload"
    payload.mkdir(parents=True)
    for name in files:
        dest=payload/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(frozen/name,dest)
    for name in ("k1_frozen_inference", "k1_execution_profile"):
        shutil.copyfile(ROOT/f"src/molgap/{name}.py",payload/f"src/molgap/{name}.py")
    shutil.copytree(HERE / "rml",payload / "prospective")
    for name in ("protocol.md","role_plan.json","inputs.json"):
        shutil.copyfile(HERE/name,payload/name)
    shutil.copyfile(selected["path"],payload / "selected.pt")
    import torch
    from molgap.k1_screen_training import _PackedGraphDatasetFactory
    data_manifest=json.loads((cache/"manifest.json").read_text())
    graphs={}; shard_bindings=[]
    for shard in data_manifest["geometry_shards"]:
        if shard["role"] != "train":
            continue
        path=cache/shard["file"]
        if sha256_file(path) != shard["sha256"]:
            raise ValueError("Parent train shard differs")
        dataset=_PackedGraphDatasetFactory.load(path)
        ids=dataset._data.source_idx.view(-1).long()
        start=int(ids[0]); end=int(ids[-1])+1
        if not torch.equal(ids,torch.arange(start,end)) or not 0 <= start < end <= 500000:
            raise ValueError("Parent train row identity differs")
        for index in sample:
            if start <= index < end:
                graphs[index]=dataset[index-start].clone()
        shard_bindings.append(shard)
        del dataset
    ordered=[graphs[index] for index in sample]
    torch.save(ordered,payload/"train_probe.pt")
    payload_manifest={"format":"molgap-k1-profile-payload-v1","sample_source_idx":sample,
        "parent_manifest_sha256":prior["manifest_sha256"],"parent_training_shards":shard_bindings,
        "checkpoint":{"sha256":selected["sha256"],"source_sha256":"0d53f2ae5a46d51cff9730e00767fa16b80ffb27002fa50b52a67c12ac8ab65a"},
        "files":{p.relative_to(payload).as_posix():sha256_file(p) for p in payload.rglob('*') if p.is_file()}}
    atomic_json(payload/"payload_manifest.json",payload_manifest)
    atomic_json(HERE/"payload_manifest.json",payload_manifest)
    archive=payload.parent/"k1-profile-a100-20261007.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=3) as handle:
        for p in payload.rglob('*'):
            if p.is_file(): handle.write(p,p.relative_to(payload))
    atomic_json(HERE/"upload_binding.json",{"file":str(archive),"sha256":sha256_file(archive),"bytes":archive.stat().st_size})
    print(str(archive),archive.stat().st_size,flush=True)


if __name__ == "__main__":
    main()
