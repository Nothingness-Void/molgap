"""IMS entrypoint only: isolate a scheduler-assigned device, then reuse the worker."""
import argparse
import os


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--device", required=True, type=int)
    args, worker_args = parser.parse_known_args(argv)
    assigned = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not assigned[0] or not 0 <= args.device < len(assigned):
        raise ValueError("IMS requires an explicit index within scheduler CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = assigned[args.device]
    from molgap.experiment_training_worker import main as worker
    return worker(worker_args)


if __name__ == "__main__":
    main()
