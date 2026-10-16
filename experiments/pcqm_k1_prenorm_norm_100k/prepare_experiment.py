"""Thin scientific declaration over existing initialization/workflow/RML owners."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess

import torch

from molgap.common.declarations import ExecutionQuestion, ExecutionProfile, build_execution_proposal
from molgap.execution_comparison import freeze_comparison_declarations
from molgap.experiment_catalog import FAMILIES
from molgap.experiment_execution import build_family_recipe
from molgap.family_training_data import DatasetSelection
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_runner import family_arm_bindings, read_recipe
from molgap.k1_prenorm import make_model
from molgap.screen_policy import canonical_fingerprint
from molgap.training_reproducibility import atomic_json, atomic_torch_save, sha256_file
from molgap.v4_runtime import read_frozen_state_artifact, state_dict_sha256, normalized_source_sha256


BASE = "experiments/pcqm_k1_prenorm_norm_100k"
RUN = "pcqm-k1-prenorm-norm-100k-s42"
ARMS = {"k1_pre_bn": "prenorm_bn", "k1_pre_ada": "prenorm_ada"}


def inputs(root, baseline, old):
    here = root / BASE
    storage = root / "platforms/_records/kaggle/staging/k1_prenorm_norm_100k"
    storage.mkdir(parents=True, exist_ok=True)
    state, observation = read_frozen_state_artifact(baseline,
        expected_state_sha256="8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd")
    roles = json.loads((old / "role_plan.json").read_text())
    atomic_json(here / "role_plan.json", roles)
    prior = json.loads((old / "config.json").read_text())
    recipe_template = json.loads((old / "training_recipe_k1_single_bn.json").read_text())
    refs = []
    for arm in ("k1_single_bn", "k1_ada_rmsnorm"):
        for name in ("canonical_trace.json", "development_predictions.pt", "training_contract.json"):
            src = old / "terminal_acceptance/outputs/experiment" / arm / name
            dst = here / "reference_binding/post_norm" / arm / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            refs.append({"path": dst.relative_to(root).as_posix(), "sha256": sha256_file(dst),
                         "source": "codex/exp/k1-norm-ablation-100k@d1438c87080c67a30dfeda775626de7ed57be2f9:" + src.relative_to(old).as_posix()})
    for name in ("terminal_acceptance/summary.json", "terminal_acceptance/interpretation_review_20261016.md"):
        src = old / name
        dst = here / "reference_binding/post_norm" / src.name
        shutil.copyfile(src, dst)
        refs.append({"path": dst.relative_to(root).as_posix(), "sha256": sha256_file(dst)})
    atomic_json(here / "reference_binding/post_norm/manifest.json", {
        "format": "molgap-retained-reference-files-v1", "comparison_scope": "context_only_cross_job",
        "source_commit": "1f627236c1f7780e35411e741352504092e145c8", "files": refs})
    config = {"run_id": RUN, "account": "kaseichou", "title": "MolGap K1 PreNorm Norm 100K S42",
        "kernel": "kaseichou/molgap-k1-prenorm-norm-100k-s42",
        "source_dataset": "kaseichou/molgap-k1-prenorm-norm-100k-s42-src",
        "dataset": "kaseichou/pcqm4mv2-ogb-fixed-100k-v1",
        "dataset_manifest_sha256": prior["dataset_manifest_sha256"], "seed": 42,
        "allocation_wall_limit_seconds": 14400, "cleanup_reserve_seconds": 180,
        "arms": {}}
    for arm, mode in ARMS.items():
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model = make_model(mode)
        initial = model.state_dict()
        for key in initial:
            if key in state:
                initial[key].copy_(state[key])
        digest = state_dict_sha256(initial)
        dest = storage / "initial_states" / (arm + ".pt")
        atomic_torch_save(dest, {"model_state": initial, "state_sha256": digest})
        recipe = FamilyRecipe("neural_atom_k1", mode, 42, digest,
            DatasetSelection("pcqm_fixed_100k_v1/manifest.json", config["dataset_manifest_sha256"], True),
            100000, 50000, 31240, .0004, .00001, ScheduleSpec("cosine_epoch", 40),
            "best-development-live", "python-epoch-shuffle", 128,
            5.3383002281188965, 1.275090217590332, ema_decay=.999, torch_version="2.11.0")
        record = build_family_recipe(("neural_atom_k1", "3"), addon="k1_" + mode,
            configuration=recipe,
            train_role_identity=roles["roles"][0]["usage_sha256"],
            development_role_identity=roles["roles"][1]["usage_sha256"],
            source_idx_sha256=recipe_template["acceptance_requirements"]["source_idx_sha256"],
            target_sha256=recipe_template["acceptance_requirements"]["target_sha256"],
            logging={"mode": "epoch", "interval": None})
        atomic_json(here / ("training_recipe_" + arm + ".json"), record)
        config["arms"][arm] = {"mode": mode, "addon": "k1_" + mode,
                              "initialization_sha256": digest, "initial_state": "initial_states/" + arm + ".pt"}
        del model, initial
    atomic_json(here / "config.json", config)
    atomic_json(here / "authority.json", {"format": "molgap-execution-authority-v1", "user_authorized": True,
        "platform": "kaggle", "account": "kaseichou", "allocated_t4_hours_ceiling": 8.,
        "arms": list(ARMS), "seeds": [42], "scope": "One100K PreNorm BN/Ada paired validation; no retry/scale-up",
        "authorized_at": datetime.now(timezone.utc).isoformat()})
    print(json.dumps({"status": "INPUTS_FROZEN", "initializations": config["arms"]}))


def declarations(root, manifest):
    here = root / BASE
    config = json.loads((here / "config.json").read_text())
    roles = json.loads((here / "role_plan.json").read_text())["roles"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    question = ExecutionQuestion(RUN, "RUN_PRENORM_COMPARISON",
        "Does AdaRMSNorm improve local pre-norm K1 versus a same-job pre-norm BN control at100K?",
        "Prior post-norm Ada replacement lost3.890meV; its causal attribution was corrected, leaving placement unresolved.",
        "Move local norms inside residual branches; both arms add the declared final node norm; compare BN versus adaptive RMS gain.",
        ("TB-pcqm-k1-norm-ablation-100k-s42-k1_ada_rmsnorm-terminal-20261016",),
        ("Operator-by-placement interaction", "Final normalization or source/job effects", "Recipe tuning and seed variance"),
        ("pcqm-k1-norm-ablation-100k-s42",),
        "Nominate a separately qualified pre-norm candidate; no automatic500K/adoption.",
        "Close this precise fixed-recipe replacement; do not generalize to all modern norms.")
    profile = ExecutionProfile("desktop", "B", "kaggle2", "Tesla_T4",
        "One fresh40-pass same-job seed42 pair with unchanged exposure and Live selection; no baseline repeat.", 2., 2.)
    arms, bindings, jobs, entries = [], [], [], []
    ref = lambda name, digest: {"name": name, "version": "1", "sha256": digest}
    for device, (arm, mode) in enumerate(ARMS.items()):
        recipe_ref = BASE + "/training_recipe_" + arm + ".json"
        record, recipe = read_recipe(root / recipe_ref)
        b = family_arm_bindings(recipe)
        declaration = {"arm_id": arm, "scientific_role": "reference" if device == 0 else "candidate",
            "family": {"name": "neural_atom_k1", "version": "3"}, "base": b["base"],
            "initialization": {"kind": "frozen_state", "seed": 42, "state_sha256": recipe.initialization_sha256},
            "data": {"dataset": ref("pcqm4mv2", config["dataset_manifest_sha256"]),
                "split": ref("fixed100k-next50k", canonical_fingerprint({"roles": roles})), "roles": roles,
                "feature_schema": FAMILIES[("neural_atom_k1", "3")].feature_schema,
                "feature_sha256": config["dataset_manifest_sha256"], "target": "pcqm4mv2-gap-eV-direct"},
            "training": {"recipe": ref("shared-family-training", normalized_source_sha256(root / recipe_ref)),
                "overrides": {}, **{key: b[key] for key in ("objective", "sampler", "transform")}},
            "addons": [{"name": "k1_" + mode, "version": "1", "config": {},
                        "source_sha256": normalized_source_sha256(root / "src/molgap/k1_prenorm.py")}],
            "addon_semantics": "ordered"}
        arms.append(declaration)
        tid = "TB-" + RUN + "-" + arm
        proposal = build_execution_proposal(question=question, arm=declaration, trajectory_id=tid,
            source_commit=commit, run=RUN, version="v1", contract_ref=BASE + "/config.json",
            contract_refs=[BASE + "/protocol.md", recipe_ref, BASE + "/reference_binding/post_norm/manifest.json"],
            authority_ref=BASE + "/authority.json", role_ref=BASE + "/role_plan.json", decision_ref=BASE + "/decision.md",
            timestamp=datetime.now(timezone.utc).isoformat(), profile=profile)
        proposal["trajectory"]["actions"][0]["attempt_ids"] = [RUN + "-v1"]
        proposal["costs"][0]["attempt_id"] = RUN + "-v1"
        proposal["trajectory"]["actions"][0]["type"] = "training"
        proposal["trajectory"]["decision"]["next_allowed_actions"] = ["one authorized100K training attempt"]
        plan_ref = BASE + "/plan_" + arm + ".json"
        atomic_json(root / plan_ref, proposal)
        bindings.append({"arm_id": arm, "trajectory_id": tid, "plan_spec_ref": plan_ref,
                         "plan_spec_sha256": sha256_file(root / plan_ref), "output": BASE + "/rml/" + arm})
        jobs.append({"arm_id": arm, "device": device, "recipe": recipe_ref,
                     "initial_state": str(root / "platforms/_records/kaggle/staging/k1_prenorm_norm_100k" /
                                          config["arms"][arm]["initial_state"])})
        target = "experiments/pcqm_k1_slot_width96/reference_binding/target_manifest.json"
        entries.append({"arm_id": arm, "adapter": "k1-session-v1",
            "contract": {"path": recipe_ref, "sha256": normalized_source_sha256(root / recipe_ref)},
            "expected": record["acceptance_requirements"],
            "target_manifest": {"path": target, "sha256": sha256_file(root / target)}})
    extras = [BASE + "/" + n for n in ("config.json", "authority.json", "protocol.md", "evidence_review.md", "role_plan.json",
              "reference_binding/post_norm/manifest.json", "reference_binding/post_norm/summary.json")]
    result = freeze_comparison_declarations(root, base=BASE, experiment_id=RUN, logical_run_id=RUN,
        arms=arms, bindings=bindings, jobs=jobs, entries=entries,
        plan_format="molgap-family-same-run-acceptance-plan-v1", authority_ref=BASE + "/authority.json",
        source_extras=extras, account=config["account"], kernel=config["kernel"], title=config["title"],
        source_dataset=config["source_dataset"], data_datasets=[config["dataset"]],
        same_run_replay={"reference_arm_id": "k1_pre_bn", "candidate_arm_ids": ["k1_pre_ada"]},
        dataset_manifests={arm: str(manifest) for arm in ARMS})
    workflow = json.loads((here / "workflow_plan.json").read_text())
    # Acceptance already has its own immutable staged sidecars. Keep the source
    # archive free of Spec/prospective-envelope self-reference to its Git commit.
    workflow["source_files"].remove(BASE + "/acceptance_plan.json")
    workflow["kaggle"].update(allocation_wall_limit_seconds=14400, cleanup_reserve_seconds=180)
    atomic_json(here / "workflow_plan.json", workflow)
    print(json.dumps({"status": "DECLARED", "source_commit": commit, **result}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("inputs", "declarations"), required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--old-experiment", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == "inputs":
        inputs(args.repo_root, args.baseline, args.old_experiment)
    else:
        declarations(args.repo_root, args.manifest)
