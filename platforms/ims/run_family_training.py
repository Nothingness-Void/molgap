"""IMS entrypoint only: isolate a scheduler-assigned device, then reuse the worker."""
import argparse
import os
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--device", required=True, type=int)
    args, worker_args = parser.parse_known_args(argv)
    assigned = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if any(not value.strip() or value != value.strip() for value in assigned) or not 0 <= args.device < len(assigned):
        raise ValueError("IMS requires an explicit index within scheduler CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = assigned[args.device]
    if "--execution-policy" in worker_args:  # a named non-default policy (the serial driver passes it); the default path is below
        from fast_recipe_run import main as fast_worker
        return fast_worker(worker_args)
    if "--recipe" in worker_args:  # frozen-recipe route for standalone models; returns 2 when paused
        from family_recipe_run import main as recipe_worker
        return recipe_worker(worker_args)
    from molgap.experiment_training_worker import main as worker
    return worker(worker_args)


if __name__ == "__main__":
    sys.exit(main())
