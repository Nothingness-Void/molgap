"""Thin entry point; source bootstrap and prospective publication own release."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    from molgap.experiment_spec import ExperimentSpec
    from molgap.k1_screen_training import run_screen_arm, run_screen_preflight

    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("spec", "package-dir", "recipe", "initial-state", "input-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("expected-package-identity", "arm-id", "account", "run-reference", "trajectory-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--mode", choices=("reference", "ssma", "clean_fingerprint"), required=True)
    parser.add_argument("--phase", choices=("preflight", "train"), default="train")
    parser.add_argument("--preflight-dir", type=Path)
    parser.add_argument("--label-cache", type=Path)
    parser.add_argument("--label-cache-manifest-sha256")
    parser.add_argument("--label-role-sha256")
    arguments = parser.parse_args()
    cache = None
    cache_args = (arguments.label_cache, arguments.label_cache_manifest_sha256,
                  arguments.label_role_sha256)
    if any(value is not None for value in cache_args):
        if not all(value is not None for value in cache_args):
            parser.error("Label cache requires root, manifest SHA and role SHA")
        from molgap.chemical_aux_cache import ChemicalLabelCache
        cache = ChemicalLabelCache(arguments.label_cache,
            arguments.label_cache_manifest_sha256,
            expected_role_sha256=arguments.label_role_sha256)
    runner = run_screen_preflight if arguments.phase == "preflight" else run_screen_arm
    options = {} if arguments.phase == "preflight" else {"preflight_dir": arguments.preflight_dir}
    runner(spec=ExperimentSpec.from_json(arguments.spec.read_text(encoding="utf-8")),
        package_dir=arguments.package_dir,
        expected_package_identity=arguments.expected_package_identity,
        arm_id=arguments.arm_id, mode=arguments.mode, recipe_path=arguments.recipe,
        initial_state_path=arguments.initial_state, input_root=arguments.input_root,
        output=arguments.output, account=arguments.account, run_reference=arguments.run_reference,
        trajectory_id=arguments.trajectory_id, label_cache=cache, **options)


if __name__ == "__main__":
    main()
