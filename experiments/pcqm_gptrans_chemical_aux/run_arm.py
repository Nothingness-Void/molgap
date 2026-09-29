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
    if phase == "preflight":
        args.pop("preflight_path")
        result = run_preflight(**args)
    else:
        if args["preflight_path"] is None:
            parser.error("train requires --preflight-path")
        result = run_training(**args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
