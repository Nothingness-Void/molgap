"""One explicitly isolated GPU arm; uses the shared GPTrans preflight/trainer."""
import argparse
import json
from pathlib import Path

from molgap.chemical_aux_cache import ChemicalLabelCache
from molgap.gptrans_objective import GPTransObjectiveConfig
from molgap.gptrans_chemical_training import ChemicalTrainingAddon
from molgap.pcqm_gptrans_v4 import run_preflight, run_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "train"), required=True)
    for name in ("dataset-root", "manifest-path", "source-archive", "output", "initial-state-path", "objective-config"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-archive-sha256", "source-commit", "platform-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--preflight-path", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--cache-manifest-sha256")
    parser.add_argument("--cache-role-sha256")
    for name in ("spec", "package", "contract"):
        parser.add_argument("--" + name, type=Path)
    for name in ("expected-package-identity", "arm-id", "trajectory-id"):
        parser.add_argument("--" + name)
    args = vars(parser.parse_args())
    phase = args.pop("phase")
    config = GPTransObjectiveConfig.from_dict(json.loads(args.pop("objective_config").read_text()))
    root, manifest_hash, role_hash = (args.pop(k) for k in ("cache_root", "cache_manifest_sha256", "cache_role_sha256"))
    if config.enabled != bool(root and manifest_hash and role_hash):
        parser.error("Candidate requires all cache bindings; control must not supply them")
    if not config.enabled and any(v is not None for v in (root, manifest_hash, role_hash)):
        parser.error("Control must not supply cache arguments")
    cache = None if not config.enabled else ChemicalLabelCache(root, manifest_hash, expected_role_sha256=role_hash)
    args["training_addon"] = ChemicalTrainingAddon(config, cache)
    family = {k: args.pop(k) for k in ("spec", "package", "contract", "expected_package_identity", "arm_id", "trajectory_id")}
    if phase == "preflight":
        args.pop("preflight_path")
        result = run_preflight(**args)
        if result.get("accepted") is not True:
            raise RuntimeError("Chemical arm runtime preflight rejected")
        from molgap.gptrans_chemical_training import profile_training_overhead
        from molgap.training_reproducibility import atomic_json
        profile = profile_training_overhead(addon=args["training_addon"],
            dataset_root=args["dataset_root"], manifest_path=args["manifest_path"],
            initial_state_path=args["initial_state_path"])
        atomic_json(args["output"] / "objective_profile.json", profile)
        if not profile["accepted"]:
            raise RuntimeError("Chemical objective exceeds frozen 5% optimizer-step overhead gate")
    else:
        if args["preflight_path"] is None:
            parser.error("train requires --preflight-path")
        if not all(family.values()):
            parser.error("train requires the frozen Spec/package/arm/contract/trajectory bindings")
        from molgap.experiment_spec import ExperimentSpec
        from molgap.experiment_family_workflow import RunContext, FamilyOutputSession
        context = RunContext.for_training(
            ExperimentSpec.from_json(family["spec"].read_text()), family["package"],
            expected_package_identity=family["expected_package_identity"],
            arm_id=family["arm_id"], account="nothingnessvoid",
            run_reference="nothingnessvoid/molgap-gptrans-chemical-aux-100k-s42-v1")
        for field in ("source_commit", "source_archive_sha256"):
            if args[field] != getattr(context, field):
                raise ValueError("Trainer source differs from frozen family context: " + field)
        role = json.loads(family["contract"].read_text())["development_role_identity"]
        def metric(weights, identity):
            return {"metric": "MAE", "unit": "eV", "target": "Gap",
                    "role_identity": identity, "weights": weights, "direction": "minimize"}
        session = FamilyOutputSession(
            args["output"] / "family_outputs", context, adapter="gptrans-v1",
            contract=family["contract"], trajectory_id=family["trajectory_id"],
            metric_semantics={"live_train_metric": metric("live", "official_train_prefix_0_100000"),
                              "live_dev_metric": None, "ema_dev_metric": metric("ema", role)})
        args["output_session"] = session
        result = run_training(**args)
        import torch
        acceptance = session.complete(
            runtime={k: getattr(context, k) for k in ("platform", "account", "source_commit", "source_archive_sha256")}
                    | {"precision": "fp32"}, hardware=torch.cuda.get_device_name(0))
        from molgap.training_reproducibility import atomic_json
        atomic_json(args["output"] / "family_acceptance.json", acceptance)
        if acceptance["status"] != "MECHANICALLY_VERIFIED":
            raise RuntimeError("Family output acceptance failed: " + str(acceptance["blockers"]))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
