"""Question declarations only; shared owners prepare, train and submit."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import subprocess

from molgap.common.declarations import ExecutionProfile, ExecutionQuestion, build_execution_proposal
from molgap.experiment_execution import build_addon_declaration, build_family_recipe
from molgap.experiment_spec import ExperimentSpec, FAMILIES, SCHEMA_VERSION_V2, TERMINAL_PROTOCOL
from molgap.family_training_data import DatasetSelection
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.family_training_runner import family_arm_bindings, INTERFACES
from molgap.training_reproducibility import atomic_json, canonical_fingerprint, sha256_file
from molgap.v4_runtime import inspect_frozen_state_artifact, normalized_source_sha256


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REL = HERE.relative_to(ROOT).as_posix()
MANIFEST = "1b0e8fd579ab1cb86c02e833e7ad284b4af7582b059f912a77853fdccf3ede6d"
DATASET = "nothingnessvoid/pcqm4mv2-ogb-fixed-100k-v1"
PRIOR = ROOT / "experiments/family_topology_t4_100k"
TARGET = "experiments/pcqm_k1_slot_width96/reference_binding/target_manifest.json"


def generate(initials, profile):
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    timestamp = datetime.now(timezone.utc).isoformat()
    previous = json.loads((PRIOR / "experiment_spec.json").read_bytes())
    config = json.loads((PRIOR / "config.json").read_bytes())
    targets = json.loads((ROOT / TARGET).read_bytes())
    profile_value = json.loads(profile.read_bytes())
    def write(path, record):
        atomic_json(ROOT / path, record)
    def pin(path):
        return {"path": path, "sha256": sha256_file(ROOT / path)}
    def ref(name, digest):
        return {"name": name, "version": "1", "sha256": digest}
    manifest_path = Path(profile_value["data_root"]) / "pcqm_fixed_100k_v1/manifest.json"
    if sha256_file(manifest_path) != MANIFEST:
        raise ValueError("Fixed data manifest differs")
    for key, family, control, candidate, selected in (
            ("k1", "neural_atom_k1", "post_bn", "post_ln", "k1_mean2"),
            ("gptrans", "gptrans_t", "bond_local", "resistance", "gptrans_local")):
        dest = REL + "/" + key
        if (ROOT / dest / "rml").exists():
            raise ValueError("Prospective already published; reconcile original declarations")
        run = "pcqm-" + key + "-" + ("post-ln" if key == "k1" else "resistance") + "-100k-s42"
        names = [control, candidate]
        facts = config["arms"][selected]
        initial = initials[key]
        inspect_frozen_state_artifact(initial, expected_state_sha256=facts["initialization_sha256"])
        authority_ref, role_ref = dest + "/authority.json", dest + "/role_plan.json"
        roles = deepcopy(next(a for a in previous["arms"] if a["arm_id"] == selected)["data"]["roles"])
        for role in roles:
            role["usage_sha256"] = canonical_fingerprint({"role": role["role"],
                "membership": role["membership_sha256"], "contract": run})
        write(role_ref, {"roles": roles, "official_validation": "forbidden", "test_dev": "forbidden",
                        "test_challenge": "forbidden", "source": "official-train prefix only"})
        authority_path = ROOT / authority_ref
        authorized_at = json.loads(authority_path.read_bytes())["authorized_at"] if authority_path.exists() else timestamp
        write(authority_ref, {"format": "molgap-execution-authority-v1", "user_authorized": True,
            "authorized_at": authorized_at, "account": "nothingnessvoid", "platform": "kaggle",
            "arms": names, "seeds": [42], "allocated_t4_hours_ceiling": 12.0,
            "allocation_wall_limit_seconds": None, "ceiling_basis": "platform session maximum, not user timeout",
            "platform_limit_ref": "https://www.kaggle.com/docs/notebooks#technical-specifications",
            "scope": "One fixed100K same-job pair; no automatic retry, scale-up, protected roles or adoption",
            "user_authorization": "Prepare and submit Kaggle1; run100K without an added user wall-clock limit"})
        decision_ref = dest + "/decision.md"
        decision = ROOT / decision_ref
        if not decision.exists():
            from molgap.experiment_launch import publish_immutable_bytes
            publish_immutable_bytes(decision, b"# Decision\n\nACTIVE: prospective100K pair; no scientific result or promotion.\n")
        policy = json.loads((ROOT / "research_memory/policies/pcqm-k1-colab-execution-profile.1.json").read_bytes())
        policy.update(policy_id=run, created_from_source_digest=sha256_file(HERE / "protocol.md"),
                      comparability_selector={"scientific_contract": run})
        policy["action_rule"]["action"] = "RUN_PAIRED_100K"
        write("research_memory/policies/" + run + ".1.json", policy)
        prior_evidence_path = (ROOT / "experiments/pcqm_k1_prenorm_norm_100k/rml/k1_pre_ada/rml_finalized/v5_evidence.json"
            if key == "k1" else PRIOR / "rml/gptrans_local/rml_finalized/v5_evidence.json")
        prior_evidence_id = json.loads(prior_evidence_path.read_bytes())["evidence_id"]
        question = ExecutionQuestion(run, "RUN_PAIRED_100K",
            "Does only Post-LN improve K1 against its same-job Post-BN control?" if key == "k1" else
            "Does resistance topology improve BondLocalGPTrans enough to justify measured overhead?",
            "Prior normalization results do not establish a beneficial post-LN replacement." if key == "k1" else
            "No accepted resistance-pair endpoint exists; topology benefit and online pinv cost are unknown.",
            "Only18 post-residual local BN sites become node-wise LN." if key == "k1" else
            "Only a zero-initialized four-channel resistance projection before first attention.",
            (prior_evidence_id,), ("Seed variability", "Optimization interaction", "Graph-size cohorts", "Platform contention"),
            ("pcqm-k1-prenorm-norm-100k-s42",) if key == "k1" else ("family-topology-t4-execution",),
            "Nominate for independent review only; no automatic500K or production adoption.",
            "Close the precise frozen mechanism; retain attribution and no automatic retry.")
        arms, bindings, jobs, acceptance = [], [], [], []
        for device, name in enumerate(names):
            addon_name = (None if key == "k1" and device == 0 else "k1_post_layernorm" if key == "k1" else
                          "gptrans_degree_bond_local" if device == 0 else "gptrans_bond_local_resistance")
            addons = ([] if addon_name is None else [build_addon_declaration(INTERFACES[family], addon_name,
                       repo_root=ROOT, version="1", config={})])
            mode = ("reference" if addon_name is None else "post_layernorm" if key == "k1" else
                    "degree_bond_local_ema999" if device == 0 else "degree_bond_local_resistance")
            recipe_addons = tuple(addons) if mode in {"post_layernorm", "degree_bond_local_resistance"} else ()
            recipe = FamilyRecipe(family, mode, 42, facts["initialization_sha256"],
                DatasetSelection("pcqm_fixed_100k_v1/manifest.json", MANIFEST, True), 100000, 50000,
                781 * facts["epochs"], facts["learning_rate"], facts["weight_decay"],
                ScheduleSpec(facts["schedule"], facts["epochs"], warmup_units=facts["warmup_units"]),
                "best-development-live" if key == "k1" else "best-development-ema", facts["sampler"], 128,
                facts["target_mean_eV"], facts["target_std_eV"], ema_decay=.999,
                torch_version=config["torch_version"], addons=recipe_addons, resume_chunks=True)
            record = build_family_recipe(INTERFACES[family], addon=addon_name, configuration=recipe,
                source_idx_sha256="3270b30071692d712c468fa52fa247d37f0a8e4b5444dc7c913aadb8ca0832c4",
                target_sha256=targets["development_target_sha256"],
                train_role_identity=roles[0]["usage_sha256"], development_role_identity=roles[1]["usage_sha256"])
            record["profiling"] = config["profiling"] | {"module_paths":
                ["local_blocks.0", "local_blocks.8"] if key == "k1" else ["blocks.0", "blocks.11"]}
            recipe_ref = dest + "/training_recipe_" + name + ".json"
            write(recipe_ref, record)
            b = family_arm_bindings(recipe)
            arm = {"arm_id": name, "scientific_role": "reference" if device == 0 else "candidate",
                "family": {"name": family, "version": INTERFACES[family][1]}, "base": b["base"],
                "initialization": {"kind": "frozen_state", "seed": 42, "state_sha256": recipe.initialization_sha256},
                "data": {"dataset": ref("pcqm4mv2", MANIFEST),
                    "split": ref("fixed100k-next50k", canonical_fingerprint({"roles": roles})), "roles": roles,
                    "feature_schema": FAMILIES[INTERFACES[family]].feature_schema, "feature_sha256": MANIFEST,
                    "target": "pcqm4mv2-gap-eV-direct"},
                "training": {"recipe": ref("shared-family-training", normalized_source_sha256(ROOT / recipe_ref)),
                    "overrides": {}, **{k: b[k] for k in ("objective", "sampler", "transform")}},
                "addons": addons, "addon_semantics": "ordered" if addons else "baseline"}
            arms.append(arm)
            tid = "TB-" + run + "-" + name
            plan = build_execution_proposal(question=question, arm=arm, trajectory_id=tid, source_commit=commit,
                run=run, version="v1", contract_ref=recipe_ref,
                contract_refs=[REL + "/protocol.md", REL + "/evidence_review.md", recipe_ref],
                authority_ref=authority_ref, role_ref=role_ref, decision_ref=decision_ref, timestamp=timestamp,
                profile=ExecutionProfile("desktop", "B", "kaggle", "Tesla T4",
                    "Synthetic wiring/mask/gradient checks then one native preflight and authorized100K pair.",
                    None, None), action_type="training")
            plan["decision_state"]["policy_version"] = "1"
            # Same physical attempt is declared by Spec, not guessed from arm names.
            plan["trajectory"]["actions"][0]["attempt_ids"] = [run + "-v1"]
            plan["costs"][0]["attempt_id"] = run + "-v1"
            plan_ref = dest + "/plan_" + name + ".json"
            write(plan_ref, plan)
            bindings.append({"arm_id": name, "trajectory_id": tid, "plan_spec_ref": plan_ref,
                "plan_spec_sha256": sha256_file(ROOT / plan_ref), "output": dest + "/rml/" + name})
            jobs.append({"arm_id": name, "device": device, "recipe": recipe_ref, "initial_state": str(initial)})
            acceptance.append({"arm_id": name, "adapter": "k1-session-v1" if key == "k1" else "gptrans-session-v1",
                "expected": record["acceptance_requirements"], "contract": pin(recipe_ref), "target_manifest": pin(TARGET)})
        spec = ExperimentSpec({"schema_version": SCHEMA_VERSION_V2, "experiment_id": run, "logical_run_id": run,
            "arms": arms, "platform": {"name": "kaggle", "accelerator": "NvidiaTeslaT4", "device_count": 2,
                "cpu_cores": 4, "memory_gib": 29, "atomic_checkpoints": True, "retrievable_chunks": True},
            "prospective": {"arms": bindings, "same_run_replay": {"reference_arm_id": control, "candidate_arm_ids": [candidate]}},
            "evidence": {"policy": ref("molgap-v5", sha256_file(ROOT / "docs/operations/MOLGAP_COMMON_DIRECTION_V5_FINAL.md")),
                         "required_artifacts": ["v5_evidence", "costs", "roles", "trace_manifest", "terminal_artifact"]},
            "terminal_protocol": TERMINAL_PROTOCOL})
        write(dest + "/experiment_spec.json", spec.to_dict())
        write(dest + "/acceptance_plan.json", {"format": "molgap-family-same-run-acceptance-plan-v1",
                                               "spec_identity": spec.identity, "arms": acceptance})
        title = "MolGap " + ("K1 PostLN" if key == "k1" else "GPTrans Resistance") + " 100K S42"
        slug = title.lower().replace(" ", "-")
        source = "nothingnessvoid/" + ("k1-postln-100k-s42-src" if key == "k1" else "gptrans-resistance-100k-s42-src")
        # Prospective inputs and the Spec are metadata consumed by existing owners,
        # not executable source. Commit them after freezing their source HEAD.
        records = [REL + "/protocol.md", REL + "/evidence_review.md", authority_ref, role_ref, decision_ref, TARGET]
        write(dest + "/workflow_plan.json", {"format": "molgap-experiment-workflow-v1", "spec_identity": spec.identity,
            "source_profile": "runtime-v1", "source_records": records, "arms": jobs,
            "acceptance_plan": dest + "/acceptance_plan.json",
            "dataset_manifests": {name: str(manifest_path) for name in names},
            "kaggle": {"account": "nothingnessvoid", "kernel": "nothingnessvoid/" + slug, "title": title,
                       "datasets": [source, DATASET], "source_dataset": source, "accelerator": "NvidiaTeslaT4"}})
        write(dest + "/input.json", {"format": "molgap-experiment-input-v1", "spec_ref": dest + "/experiment_spec.json",
            "workflow_ref": dest + "/workflow_plan.json", "machine_profile": str(profile), "authority_ref": authority_ref,
            "monitor_paths": [REL + "/STATUS.md"], "attempt_id": "v1"})
        print(json.dumps({"status": "DRAFT_ONLY", "question": key, "spec_identity": spec.identity, "source_commit": commit}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k1-initial", type=Path, required=True)
    parser.add_argument("--gptrans-initial", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=Path("D:/molgap-machine-profile.json"))
    args = parser.parse_args()
    generate({"k1": args.k1_initial, "gptrans": args.gptrans_initial}, args.profile)
