"""Fixed subprocess entrypoint; executable owners come only from the registry."""
from __future__ import annotations

import argparse
from pathlib import Path


def main(argv=None):
    from .experiment_execution import execute_training_phase, validate_execution_plan
    from .experiment_family_workflow import _json
    from .experiment_spec import ExperimentSpec
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("source-root", "package-dir", "input-root", "output", "launch"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--phase", required=True, choices=("preflight", "train"))
    args = parser.parse_args(argv)
    config = _json(args.launch)
    spec = ExperimentSpec.from_json((args.package_dir / "experiment_spec.json").read_text(encoding="utf-8"))
    jobs = validate_execution_plan(spec, config["jobs"])
    job = next(j for j in jobs if j["arm_id"] == args.arm)
    result = execute_training_phase(spec=spec, job=job, phase=args.phase,
        source_root=args.source_root, package_dir=args.package_dir,
        expected_package_identity=config["expected_package_identity"],
        input_root=args.input_root, output=args.output, account=config["account"],
        run_reference=config["run_reference"], staged_root=args.launch.parent,
        prospective_sha256=config["prospective_sha256"][job["arm_id"]])
    passed = (result.get("status") == "MECHANICALLY_VERIFIED" if args.phase == "train" else
              result.get("status") == "accepted" or result.get("accepted") is True)
    if not passed:
        raise RuntimeError("Owning family phase did not pass")


if __name__ == "__main__":
    main()
