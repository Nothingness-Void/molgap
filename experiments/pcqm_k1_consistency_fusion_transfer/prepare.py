"""Bind existing artifact bytes and publish the standard RML prospective record."""
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from molgap.constants import REPO_ROOT as ROOT
from molgap.research_memory.plan import plan
from molgap.training_reproducibility import atomic_json, sha256_file

HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
TID = "TB-k1-consistency-fusion-transfer-20261004"
RUN = "local-k1-consistency-fusion-transfer-20261004"
PID = "pcqm-k1-consistency-fusion-transfer"


def main():
    origin = Path("D:/文档/molgap")
    cache = Path("D:/文档/molgap-exp/molgap-500k-v4-evidence/data/cache/pcqm4mv2_500k_v4")
    bindings = {}
    for arm, folder in (("mean2", "recovery_reconciliation_v3/dropout_mean2"),
                        ("consistency2", "failure_reconciliation_v1/dropout_consistency2")):
        acceptance = json.loads((ROOT / f"experiments/pcqm_k1_dropout_consistency/terminal_acceptance/dropout_{arm}/mechanical.json").read_text())
        for kind, filename, observed_key in (("model", "selected_model.pt", "selected_model"),
                                             ("predictions", "development_predictions.pt", "predictions")):
            relative = f"experiments/pcqm_k1_dropout_consistency/{folder}/{filename}"
            source = origin / relative
            digest = sha256_file(source)
            if digest != acceptance["observed"]["artifacts"][observed_key]["sha256"]:
                raise ValueError("Accepted artifact hash differs")
            target = ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                os.link(source, target)
            bindings[arm + "_" + kind] = {"path": relative, "sha256": digest}
    manifest = json.loads((cache / "manifest.json").read_text())
    if manifest["roles"]["development"] != {"source_idx_start": 500000, "source_idx_stop": 550000, "rows": 50000}:
        raise ValueError("Common development role differs")
    for name, filename in (("manifest", "manifest.json"), ("calibration_shard", "train/train_shard_0002.pt"),
                           ("development_shard", "train/train_shard_0010.pt")):
        source = cache / filename
        digest = sha256_file(source)
        if name != "manifest":
            expected = next(x["sha256"] for x in manifest["geometry_shards"] if x["file"] == filename)
            if digest != expected:
                raise ValueError("Graph shard differs from manifest")
        target = HERE / "inputs" / Path(filename).name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            os.link(source, target)
        bindings[name] = {"path": target.relative_to(ROOT).as_posix(), "sha256": digest}
    source_equivalence = {}
    for name in ("qm9_neural_atom.py", "pcqm_gap_architecture.py", "gps.py"):
        relative = "src/molgap/" + name
        prior = subprocess.check_output(["git", "show", "84288398:" + relative], cwd=ROOT).replace(b"\r\n", b"\n")
        if prior != (ROOT / relative).read_bytes().replace(b"\r\n", b"\n"):
            raise ValueError("Accepted model source changed")
        source_equivalence[relative] = "exact LF-normalized bytes equal original source commit 84288398"
    sources = [f"{REL}/run.py", f"{REL}/protocol.md", f"{REL}/prepare.py", f"{REL}/close.py",
               "src/molgap/k1_frozen_inference.py", "src/molgap/k1_screen_training.py",
               "src/molgap/router.py", "src/molgap/training_reproducibility.py", *source_equivalence]
    inputs = {"bindings": bindings, "mean": 5.3383002281188965, "std": 1.275090217590332,
              "hardware": "NVIDIA GeForce RTX 5060", "source_equivalence": source_equivalence,
              "source_hashes": {p: sha256_file(ROOT / p) for p in sources}}
    atomic_json(HERE / "inputs.json", inputs)
    protocol = f"{REL}/protocol.md"
    pidpath = ROOT / f"research_memory/policies/{PID}.1.json"
    policy = json.loads((ROOT / "research_memory/policies/pcqm-k1-consistency-residual-screen.1.json").read_text())
    policy.update(policy_id=PID, comparability_selector={"scientific_contract": PID + "-v1"},
                  required_observable_fields=["user_requested_frozen_inference_scope_reviewed"],
                  action_rule={"field": "user_requested_frozen_inference_scope_reviewed", "operator": "eq", "threshold": 1, "action": "RUN_DIAGNOSTIC"},
                  created_from_source_digest=sha256_file(HERE / "protocol.md"))
    atomic_json(pidpath, policy)
    spec = json.loads((ROOT / "experiments/pcqm_k1_consistency_residual_screen/plan_input.json").read_text())
    trajectory = spec["trajectory"]
    trajectory.update(trajectory_id=TID, family_id=PID,
        question="Does unfitted equal blending of the accepted K1 dropout arms retain at least 1 meV gain on a separate common cohort, and what is its inference cost?",
        hypothesis={"hypothesis_id": "H-k1-consistency-fusion-transfer-20261004",
          "observed_deficiency": "Equal blending gained 4.315135 meV on the prior posthoc development holdout, but separate-cohort prediction and native inference costs were missing.",
          "supporting_evidence_ids": ["pcqm-k1-consistency-residual-screen-20261004"],
          "alternative_explanations": ["Complementarity transfers to other internal-development rows.", "Prior gain is specific to the checkpoint-selection cohort.", "The gain transfers but doubles inference cost."],
          "changed_mechanism": "No model change: clean retained checkpoint inference with a frozen 50:50 average.",
          "cheapest_falsifier": "Reconstruct 2048 accepted predictions per model, then compare both retained models and their equal blend on 50K common development rows.",
          "related_closed_family_ids": ["k1-dropout-consistency", "k1-consistency-residual-screen"],
          "expected_native_cost_ref": "cost-" + TID + "-expected-inference",
          "decision_changed_if_positive": "Retain a common-cohort fusion nomination with measured inference cost; separate distillation or scale decisions remain unreleased.",
          "decision_changed_if_negative": "Close the transfer claim without retraining, adjusting weights, or consuming official roles."})
    prior_ids = ["pcqm-k1-dropout-mean2-kaggle3-100k-s42-v1-terminal", "pcqm-k1-dropout-consistency2-kaggle3-100k-s42-v1-terminal", "pcqm-k1-consistency-residual-screen-20261004"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    trajectory["state_at_start"] = {"source_commit": commit,
        "contract_refs": [protocol, f"{REL}/inputs.json", f"{REL}/plan_input.json", *sources],
        "reference_ids": prior_ids, "parent_trajectory_ids": ["TB-k1-consistency-residual-screen-20261004"],
        "prior_evidence_ids": prior_ids, "role_snapshot_refs": [protocol],
        "budget_snapshot_ref": protocol, "source_config_identity": sha256_file(HERE / "inputs.json")}
    trajectory["actions"] = [{"action_id": "A001", "type": "frozen_checkpoint_inference", "source_commit": commit,
                              "run_ids": [RUN], "attempt_ids": ["local-attempt-001"], "evidence_refs": [],
                              "cost_event_ids": ["cost-" + TID + "-expected-inference"]}]
    trajectory["decision"] = {"decision_ref": protocol, "outcome": "ACTIVE", "next_allowed_actions": ["RUN_DIAGNOSTIC"], "reopen_conditions": []}
    state = spec["decision_state"]
    state.update(active_reference_ids=prior_ids, available_actions=["RUN_DIAGNOSTIC", "NO_TRAIN"], chosen_action="RUN_DIAGNOSTIC",
                 policy_id=PID, role_snapshot_refs=[protocol], budget_snapshot_ref=protocol,
                 state_timestamp=datetime.now(timezone.utc).isoformat(), source_commit=commit)
    spec["costs"] = [{"schema": "molgap-cost-event-v1", "cost_event_id": "cost-" + TID + "-expected-inference",
        "trajectory_id": TID, "action_id": "A001", "run_id": RUN, "attempt_id": "local-attempt-001", "category": "inference",
        "platform": "local-windows", "hardware": inputs["hardware"], "evidence_ref": protocol,
        "measurement": {"device_hours": {"value": 600/3600, "status": "estimated"}, "cpu_hours": {"value": None, "status": "measurement_missing"},
                        "wall_hours": {"value": 600/3600, "status": "estimated"}, "queue_hours": {"value": None, "status": "not_applicable"}}}]
    atomic_json(HERE / "plan_input.json", spec)
    print(json.dumps(plan(ROOT, spec, f"{REL}/rml"), indent=2))


if __name__ == "__main__":
    main()
