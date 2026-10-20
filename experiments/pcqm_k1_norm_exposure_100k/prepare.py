"""Declare this question through existing family, prospective and envelope APIs."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import subprocess

from molgap.common.declarations import ExecutionProfile, ExecutionQuestion, build_execution_proposal
from molgap.execution_comparison import freeze_comparison_declarations
from molgap.experiment_execution import build_addon_declaration, build_family_recipe
from molgap.experiment_spec import FAMILIES
from molgap.family_training_data import DatasetSelection
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_runner import family_arm_bindings
from molgap.research_memory.trace import atomic_write
from molgap.training_reproducibility import canonical_fingerprint, sha256_file
from molgap.v4_runtime import inspect_frozen_state_artifact, normalized_source_sha256

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
PRIOR = ROOT / "experiments/pcqm_structural_bias_norm_100k/k1"
TARGET = "experiments/pcqm_k1_slot_width96/reference_binding/target_manifest.json"
DATA = "nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1"
MANIFEST = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
INITIAL = Path("D:/molgap-artifacts/initializations/structural-bias-norm-100k/k1.pt")
PROFILE = Path("D:/molgap-machine-profile.json")
FAMILY = ("neural_atom_k1", "3")
PAIRS = {
    "norm": (("bn_long", False, .0004, 0), ("ln_long", True, .0004, 0)),
    "optimization": (("ln_low_lr", True, .0002, 0), ("ln_low_lr_warmup", True, .0002, 8)),
}


def write(path, value):
    atomic_write(ROOT / path, (json.dumps(value, indent=2) + "\n").encode("utf-8"))


def run_id(pair):
    return "pcqm-k1-ln-" + pair + "-160ep-100k-s42-20261021-release1"


def contracts():
    prior = json.loads((PRIOR / "attempt_002/experiment_spec.json").read_bytes())["arms"][0]
    facts = json.loads((PRIOR / "training_recipe_post_bn.json").read_bytes())
    targets = json.loads((ROOT / TARGET).read_bytes())
    manifest_path = Path(json.loads(PROFILE.read_bytes())["data_root"]) / "pcqm_fixed_100k_v1/manifest.json"
    if sha256_file(manifest_path) != MANIFEST:
        raise ValueError("Fixed data manifest differs")
    inspect_frozen_state_artifact(INITIAL, expected_state_sha256=facts["initialization_sha256"])
    for pair, choices in PAIRS.items():
        dest, run = REL + "/" + pair + "_release1", run_id(pair)
        if (ROOT / dest / "authority.json").exists():
            raise ValueError("Contracts already declared; preserve frozen inputs")
        roles = deepcopy(prior["data"]["roles"])
        for role in roles:
            role["usage_sha256"] = canonical_fingerprint({"role": role["role"],
                "membership": role["membership_sha256"], "contract": run})
        write(dest + "/role_plan.json", {"roles": roles, "official_validation": "forbidden",
            "test_dev": "forbidden", "test_challenge": "forbidden", "source": "official-train prefix only"})
        write(dest + "/authority.json", {"format": "molgap-execution-authority-v1", "user_authorized": True,
            "authorized_at": datetime.now(timezone.utc).isoformat(), "account": "nothingnessvoid",
            "platform": "kaggle", "arms": [c[0] for c in choices], "seeds": [42],
            "allocated_t4_hours_ceiling": 24., "allocation_wall_limit_seconds": None,
            "scope": "One of two authorized Kaggle1 T4x2 notebooks; fixed100K,160 passes, no protected roles or automatic successor",
            "user_authorization": "More exposure with complete curves; Kaggle1 two slots/four arms; agent selects bounded routes",
            "quota_semantics": "Native ceiling24 T4-device hours per pair; platform quota reservation is a separate account unit"})
        atomic_write(ROOT / dest / "decision.md", b"# Prospective decision\n\nACTIVE: authorized160-pass recipe pair; no result or adoption.\n")
        policy = json.loads((ROOT / "research_memory/policies/pcqm-k1-post-ln-100k-s42-20261020.1.json").read_bytes())
        policy.update(policy_id=run, created_from_source_digest=sha256_file(HERE / "protocol.md"),
            comparability_selector={"scientific_contract": run})
        policy["action_rule"]["action"] = "RUN_PAIRED_100K"
        write("research_memory/policies/" + run + ".1.json", policy)
        for name, layernorm, lr, warmup in choices:
            addon = build_addon_declaration(FAMILY, "k1_post_layernorm", repo_root=ROOT, config={}) if layernorm else None
            recipe = FamilyRecipe("neural_atom_k1", "post_layernorm" if layernorm else "reference", 42,
                facts["initialization_sha256"], DatasetSelection("pcqm_fixed_100k_v1/manifest.json", MANIFEST, True),
                100000, 50000, 781 * 160, lr, .00001,
                ScheduleSpec("warmup_cosine_epoch" if warmup else "cosine_epoch", 160, warmup_units=warmup),
                "best-development-live", "python-epoch-shuffle", 128,
                facts["target_mean_eV"], facts["target_std_eV"], ema_decay=.999,
                torch_version="2.11.0", addons=(addon,) if addon else (), resume_chunks=True)
            record = build_family_recipe(FAMILY, addon="k1_post_layernorm" if layernorm else None,
                configuration=recipe, source_idx_sha256=roles[1]["row_order_sha256"],
                target_sha256=targets["development_target_sha256"], train_role_identity=roles[0]["usage_sha256"],
                development_role_identity=roles[1]["usage_sha256"], logging={"mode": "epoch", "interval": None})
            write(dest + "/training_recipe_" + name + ".json", record)
    print(json.dumps({"status": "CONTRACTS_DECLARED", "arms": 4, "passes_per_arm": 160}))


def plans():
    from molgap.family_training_runner import read_recipe
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    prior = json.loads((PRIOR / "attempt_002/experiment_spec.json").read_bytes())["arms"][0]
    evidence = json.loads((PRIOR / "attempt_002/rml/post_ln/rml_finalized/v5_evidence.json").read_bytes())["evidence_id"]
    timestamp = datetime.now(timezone.utc).isoformat()
    manifest_path = str(Path(json.loads(PROFILE.read_bytes())["data_root"]) / "pcqm_fixed_100k_v1/manifest.json")
    for pair, choices in PAIRS.items():
        dest, run = REL + "/" + pair + "_release1", run_id(pair)
        roles = json.loads((ROOT / dest / "role_plan.json").read_bytes())["roles"]
        question = ExecutionQuestion(run, "RUN_PAIRED_100K",
            "Does LN catch up with BN over160 passes under a matched long cosine schedule?" if pair == "norm" else
            "Does eight-pass warmup help a lower-LR LN recipe over160 passes?",
            "The40-pass LN recipe lost4.711meV; long-horizon behavior and LR/warmup interactions remain unmeasured.",
            "Only local BN/LN operator within the same long recipe." if pair == "norm" else
            "Only warmup/cosine schedule at the same reduced peak LR and long horizon.",
            (evidence,), ("Seed variation", "Annealing interaction", "Representation/statistical effects", "Job contention"),
            ("pcqm-k1-post-ln-100k-s42-20261020", "pcqm-k1-prenorm-norm-100k-s42"),
            "Retain late-behavior evidence and nominate independent review; no automatic scale-up or adoption.",
            "Close this exact long-horizon recipe, preserve full curves and attribution; no automatic successor.")
        arms, bindings, jobs, entries = [], [], [], []
        for device, (name, layernorm, lr, warmup) in enumerate(choices):
            recipe_ref = dest + "/training_recipe_" + name + ".json"
            record, recipe = read_recipe(ROOT / recipe_ref)
            b = family_arm_bindings(recipe)
            arm = {"arm_id": name, "scientific_role": "reference" if device == 0 else "candidate",
                "family": {"name": FAMILY[0], "version": FAMILY[1]}, "base": b["base"],
                "initialization": {"kind": "frozen_state", "seed": 42, "state_sha256": recipe.initialization_sha256},
                "data": {**deepcopy(prior["data"]), "roles": roles,
                    "split": {"name": "fixed100k-next50k", "version": "1", "sha256": canonical_fingerprint({"roles": roles})}},
                "training": {"recipe": {"name": "shared-family-training", "version": "1",
                    "sha256": normalized_source_sha256(ROOT / recipe_ref)}, "overrides": {},
                    **{k: b[k] for k in ("objective", "sampler", "transform")}},
                "addons": list(recipe.addons), "addon_semantics": "ordered" if recipe.addons else "baseline"}
            arms.append(arm)
            tid = "TB-" + run + "-" + name
            plan = build_execution_proposal(question=question, arm=arm, trajectory_id=tid, source_commit=commit,
                run=run, version="v1", contract_ref=recipe_ref,
                contract_refs=[REL + "/protocol.md", REL + "/evidence_review.md", recipe_ref],
                authority_ref=dest + "/authority.json", role_ref=dest + "/role_plan.json",
                decision_ref=dest + "/decision.md", timestamp=timestamp,
                profile=ExecutionProfile("desktop", "B", "kaggle", "Tesla T4",
                    "Synthetic wiring/atomic publication checks, native all-arm qualification, then the authorized160-pass comparison.",
                    6.75, 6.75), action_type="training")
            plan["decision_state"]["policy_version"] = "1"
            plan["trajectory"]["actions"][0]["attempt_ids"] = [run + "-v1"]
            plan["costs"][0]["attempt_id"] = run + "-v1"
            plan_ref = dest + "/plan_" + name + ".json";write(plan_ref, plan)
            bindings.append({"arm_id": name, "trajectory_id": tid, "plan_spec_ref": plan_ref,
                "plan_spec_sha256": sha256_file(ROOT / plan_ref), "output": dest + "/rml/" + name})
            jobs.append({"arm_id": name, "device": device, "recipe": recipe_ref, "initial_state": str(INITIAL)})
            entries.append({"arm_id": name, "adapter": "k1-session-v1", "expected": record["acceptance_requirements"],
                "contract": {"path": recipe_ref, "sha256": sha256_file(ROOT / recipe_ref)},
                "target_manifest": {"path": TARGET, "sha256": sha256_file(ROOT / TARGET)}})
        title = "MolGap K1 LN " + ("Norm" if pair == "norm" else "Optim") + " 160ep 100K S42"
        slug = title.lower().replace(" ", "-");source = "nothingnessvoid/k1-ln-" + pair + "-160ep-src-20261021"
        extras = [REL + "/protocol.md", REL + "/evidence_review.md", dest + "/role_plan.json", dest + "/decision.md", TARGET]
        result = freeze_comparison_declarations(ROOT, base=dest, experiment_id=run, logical_run_id=run,
            arms=arms, bindings=bindings, jobs=jobs, entries=entries,
            plan_format="molgap-family-same-run-acceptance-plan-v1", authority_ref=dest + "/authority.json",
            source_extras=extras, account="nothingnessvoid", kernel="nothingnessvoid/" + slug, title=title,
            source_dataset=source, data_datasets=[DATA], dataset_manifests={c[0]: manifest_path for c in choices},
            same_run_replay={"reference_arm_id": choices[0][0], "candidate_arm_ids": [choices[1][0]]})
        workflow = json.loads((ROOT / dest / "workflow_plan.json").read_bytes())
        workflow.pop("source_files")
        workflow.update(source_profile="runtime-v1", source_records=[dest + "/authority.json", *extras])
        write(dest + "/workflow_plan.json", workflow)
        write(dest + "/input.json", {"format": "molgap-experiment-input-v1", "spec_ref": dest + "/experiment_spec.json",
            "workflow_ref": dest + "/workflow_plan.json", "machine_profile": str(PROFILE),
            "authority_ref": dest + "/authority.json", "monitor_paths": [REL + "/STATUS.md"], "attempt_id": "v1"})
        print(json.dumps({"pair": pair, "source_commit": commit, **result}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("contracts", "plans"))
    args = parser.parse_args()
    contracts() if args.phase == "contracts" else plans()
