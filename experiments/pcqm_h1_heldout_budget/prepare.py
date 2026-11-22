"""Build only the declared 94K view; package and train through the local family owner."""
import argparse
import json
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from molgap.common.wire import canonical_fingerprint as fingerprint
from molgap.research_memory.trace import json_bytes
from molgap.training_reproducibility import sha256_file

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REL = HERE.relative_to(ROOT).as_posix()
OUT = Path("D:/molgap-artifacts/diagnostics/pcqm-h1-heldout-budget")
ANCHOR = Path("D:/molgap-artifacts/runs/local-pcqm-gptrans-100k-20261119-s42/v1")
INIT = Path("D:/molgap-artifacts/initializations/structural-bias-norm-100k/gptrans.pt")
ORIGINAL = Path("D:/molgap-data/pcqm_fixed_100k_v1")
FULL = Path("D:/molgap-data/pcqm_full_train_v1")
DATA = OUT / "input" / "pcqm_h1_94k"
ARM = "pair_update_norm"
FAMILY = "pcqm-h1-heldout-budget"
RUN = "local-pcqm-h1-94k-s42"
P0 = "pcqm-pairnorm-residual-2d-p0-20261122"
PROTOCOL = REL + "/protocol.md"
P0_DIR = HERE.parent / "pcqm_pairnorm_residual_2d_p0"


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def pinned(path, digest):
    if sha256_file(path) != digest:
        raise ValueError(f"Hash mismatch: {path}")
    return path


def write(path, data):
    raw = json_bytes(data)
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError(path)
    else:
        with path.open("xb") as f:
            f.write(raw)


def plan_input(worker, source, identity):
    tid = "TB-" + FAMILY + ("-94k" if worker else "")
    run = RUN + ":" + ARM + ":downstream" if worker else FAMILY + "-score"
    attempt = ARM + "-v1" if worker else "score-v1"
    cost = "cost-" + tid + "-expected"
    action = "RUN_LOCAL_94K" if worker else "RUN_FIXED_BUDGET_STUDY"
    roles = [PROTOCOL]
    contracts = [PROTOCOL, REL + "/prepare.py", REL + ("/recipe.json" if worker else "/score.py")]
    hypothesis = dict(hypothesis_id="H-" + tid, observed_deficiency="P0 did not charge lost backbone data.",
        supporting_evidence_ids=[P0], alternative_explanations=["Recalibration alone", "Data loss outweighs correction",
        "The single-bond error excess is a cohort mix effect"], changed_mechanism="Reserve six percent of a fixed label budget for a residual stack.",
        cheapest_falsifier="Train one 94K model; reuse the same-init 100K model on the same out-of-fit rows.",
        decision_changed_if_positive="Consider a separately specified larger scale; no automatic launch.",
        decision_changed_if_negative="No scale-up under this pilot.", expected_native_cost_ref=cost,
        related_closed_family_ids=["pcqm-pairnorm-residual-2d-p0"])
    state = dict(source_commit=source, source_config_identity=identity, contract_refs=contracts, reference_ids=[],
        prior_evidence_ids=[P0], parent_trajectory_ids=["TB-pcqm-pairnorm-residual-2d-p0"] if not worker else ["TB-" + FAMILY],
        budget_snapshot_ref=PROTOCOL, role_snapshot_refs=roles)
    traj = dict(schema="molgap-trajectory-v1", trajectory_id=tid, owner="desktop", track="B", family_id=FAMILY,
        record_mode="prospective", question="Does 94K plus 6K correction beat the same-init 100K backbone on new rows?",
        hypothesis=hypothesis, state_at_start=state, actions=[dict(action_id="A001", type=action, run_ids=[run],
        attempt_ids=[attempt], source_commit=source, evidence_refs=[PROTOCOL], cost_event_ids=[cost])],
        result=dict(evidence_ids=[], evidence_refs=[]), decision=dict(outcome="ACTIVE", decision_ref=PROTOCOL,
        next_allowed_actions=[action], reopen_conditions=[]), comparison_class="CONTEXT_ONLY",
        comparison_readiness_ref=PROTOCOL, comparison_blockers=["Reused local anchor is mechanically verified, not a same-job pair."],
        reference_bundle_id=None)
    measures = {k: dict(value=None, status="measurement_missing") for k in ("cpu_hours", "device_hours")}
    measures.update(wall_hours=dict(value=1.5 if worker else .5, status="estimated"), queue_hours=dict(value=None, status="not_applicable"))
    return dict(trajectory=traj, costs=[dict(schema="molgap-cost-event-v1", trajectory_id=tid, cost_event_id=cost,
        action_id="A001", run_id=run, attempt_id=attempt, category="training" if worker else "audit",
        evidence_ref=PROTOCOL, platform="local-windows", hardware="RTX 5060 and desktop CPU", measurement=measures)],
        decision_state=dict(available_actions=[action, "DEFER"], chosen_action=action, policy_id=FAMILY, policy_version="1",
        budget_snapshot_ref=PROTOCOL, role_snapshot_refs=roles, source_commit=source, state_timestamp=datetime.now(timezone.utc).isoformat()))


def view():
    import torch
    from molgap.data.packed_graph import PackedGraphDataset
    from molgap.experiment_family_workflow import tensor_digest
    from molgap.family_training_data import DatasetSelection
    from molgap.family_training_runner import INTERFACES, build_screen_recipe, read_recipe
    from molgap.v4_runtime import sample_std_compat

    assert (HERE / "rml/trajectory.json").is_file() and not (HERE / "recipe.json").exists()
    base = ANCHOR / "output-native" / ARM
    manifest = read(pinned(base / "output_manifest.json", "bd0ebc8d6c15639abf7627b94dea89709c4c12a58ff206a51277b7d27b3ca007"))
    original, config = read_recipe(pinned(base / "training_contract.json", manifest["artifacts"]["contract"]["sha256"]))
    src = read(pinned(ORIGINAL / "manifest.json", config.dataset.manifest_sha256))
    data, labels = [], []
    DATA.mkdir(parents=True, exist_ok=False)
    for i, (r, n) in enumerate(zip(src["geometry_shards"], (50000, 44000, 50000), strict=True)):
        ds = PackedGraphDataset(pinned(ORIGINAL / r["file"], r["sha256"]), mmap=True).topology_only()
        assert torch.equal(ds._data.source_idx.ravel(), torch.arange(i*50000, (i+1)*50000))
        if n < len(ds):
            # PyG stores un-incremented edge indices. Re-collate the true prefix,
            # not a view whose tensor storages still contain withheld labels.
            packed, slices = ds.collate([ds.get(j) for j in range(n)])
        else:
            packed, slices = ds._data, ds.slices
        assert packed.y.numel() == n and torch.equal(packed.source_idx.ravel(), torch.arange(i*50000, i*50000+n))
        path = DATA / f"topology_{i}.pt"
        torch.save((packed, slices), path)
        data.append(dict(file=path.name, role=r["role"], rows=n, source_idx_min=i*50000,
                         source_idx_max=i*50000+n-1, bytes=path.stat().st_size, sha256=sha256_file(path)))
        labels.append(packed.y.ravel().double())
    roles = {**src["roles"], "train": dict(rows=94000, source_idx_start=0, source_idx_stop=94000)}
    fixed = dict(format="molgap-pcqm4mv2-fixed-subset-v1", status="complete", identity=dict(name="pcqm_h1_94k"),
                 source=src["source"], roles=roles, graph_contract=src["graph_contract"], assets={"topology": data})
    fixed.update({k: False for k in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read")})
    write(DATA / "manifest.json", fixed)
    y = torch.cat(labels[:2])
    dev = original["acceptance_requirements"]
    assert y.numel() == 94000 and tensor_digest(labels[2], role="target") == dev["target_sha256"]
    select = DatasetSelection("pcqm_h1_94k/manifest.json", sha256_file(DATA / "manifest.json"), True)
    new = replace(config, dataset=select, train_rows=94000, optimizer_steps=44040,
                  target_mean_eV=float(y.mean()), target_std_eV=float(sample_std_compat(y)))
    train_rows = tensor_digest(torch.arange(94000), role="source_idx")
    usage = fingerprint(dict(role="train", source_idx_sha256=train_rows, use="backbone_only", rows=94000))
    rec = build_screen_recipe(new.mode, configuration=new, family=INTERFACES[new.family],
        train_role_identity=usage, development_role_identity=original["development_role_identity"],
        source_idx_sha256=dev["source_idx_sha256"], target_sha256=dev["target_sha256"],
        logging=original["logging"], resume_keep_count=2)
    write(HERE / "recipe.json", rec)
    write(HERE / "data_identity.json", dict(parent=config.dataset.manifest_sha256, parent_shards=src["geometry_shards"],
        worker_manifest=select.manifest_sha256, membership=train_rows, role_bounds=roles, derived_topology=data))


def bind_worker(head):
    from molgap.experiment_spec import ExperimentSpec
    from molgap.family_training_runner import family_arm_bindings, read_recipe

    recipe, config = read_recipe(HERE / "recipe.json")
    original = read(pinned(ANCHOR / "prepared-native/package/experiment_spec.json",
                           "f316f05c916dc7b858daeccfd0b35f1299279aad9d0c9cd0166e3f28580d6ec7"))
    arm, = original["arms"]
    binds = family_arm_bindings(config)
    assert binds["base"] == arm["base"] and binds["objective"] == arm["training"]["objective"]
    arm["training"].update({k: binds[k] for k in ("transform", "sampler")})
    arm["training"]["recipe"]["sha256"] = sha256_file(HERE / "recipe.json")
    data = arm["data"]
    data["dataset"]["sha256"] = data["feature_sha256"] = config.dataset.manifest_sha256
    data["split"] = dict(name="h1-prefix94k-same-dev50k", version="1", sha256=fingerprint(read(HERE / "data_identity.json")["role_bounds"]))
    train = next(r for r in data["roles"] if r["role"] == "train")
    train["membership_sha256"] = train["row_order_sha256"] = read(HERE / "data_identity.json")["membership"]
    train["usage_sha256"] = recipe["train_role_identity"]
    arm["scientific_role"] = "candidate"
    original.update(experiment_id="pcqm_h1_heldout_budget", logical_run_id=RUN)
    plan = plan_input(True, head, fingerprint(arm))
    write(HERE / "worker_plan.json", plan)
    original["prospective"] = dict(arms=[dict(arm_id=ARM, output=REL + "/worker_rml",
        trajectory_id=plan["trajectory"]["trajectory_id"], plan_spec_ref=REL + "/worker_plan.json",
        plan_spec_sha256=sha256_file(HERE / "worker_plan.json"))])
    spec = ExperimentSpec(original)
    write(HERE / "spec.json", spec.to_dict())
    write(HERE / "preparation.json", dict(format="molgap-family-training-preparation-v1", spec_identity=spec.identity,
        arms=[dict(arm_id=ARM, device=0, recipe=REL + "/recipe.json", initial_state=str(pinned(INIT,
        "27590c1a928cafa8b633998e2291916b8f171d345ed5cb3b0234f0e9a37c3320")))],
        source_files=[PROTOCOL, "src/molgap/local_training.py", "platforms/local/train.py"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("study", "view", "worker"))
    stage = parser.parse_args().stage
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    if stage == "study":
        write(HERE / "study_plan.json", plan_input(False, head, sha256_file(HERE / "protocol.md")))
    elif stage == "view":
        view()
    else:
        bind_worker(head)
