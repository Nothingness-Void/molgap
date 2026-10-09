"""Local package bootstrap only; parent owns Kaggle API and durable retrieval."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--setup", type=Path)
    parser.add_argument("--setup-sha256")
    parser.add_argument("--wall-budget-seconds", type=int, default=1200)
    args = parser.parse_args()
    started = time.perf_counter()
    if not 1 <= args.wall_budget_seconds <= 1200:
        raise ValueError("Bootstrap budget must be within1200seconds")
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path = args.root / "payload_manifest.json"
    if digest(manifest_path) != args.expected_manifest_sha256:
        raise ValueError("Pinned manifest differs")
    manifest = json.loads(manifest_path.read_text())
    for name in ("bootstrap.py", "run.sh"):
        if digest(args.root / name) != manifest["files"][name]:
            raise ValueError(f"Packaged bootstrap differs: {name}")
    if Path(__file__).resolve() != (args.root / "bootstrap.py").resolve():
        raise ValueError("Execute the pinned packaged bootstrap")
    if args.setup and (not args.setup_sha256 or digest(args.setup) != args.setup_sha256):
        raise ValueError("Explicit frozen setup digest required")
    if args.setup and (args.setup.resolve() != (args.root / "setup.sh").resolve()
                       or args.setup_sha256 != manifest["files"].get("setup.sh")):
        raise ValueError("Setup must match the exact packaged inventory")
    env = dict(os.environ, PYTHONPATH=str(args.root.resolve() / "src"),
               PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    # Shell timeout also terminates its process group. No remote action here.
    remaining = int(args.wall_budget_seconds-(time.perf_counter()-started))-2
    if remaining <= 0:
        raise TimeoutError("Package verification exhausted total ceiling")
    status = "incomplete"
    try:
        subprocess.run(["bash", str(args.root / "run.sh"), str(args.root.resolve()),
                        str(args.output.resolve()), args.expected_manifest_sha256,
                        str(args.setup.resolve()) if args.setup else "", str(remaining)],
                       env=env, check=True)
        status = "complete"
    finally:
        print(json.dumps({"package_wall_seconds": time.perf_counter()-started,
                          "status": status, "package_ceiling_seconds": args.wall_budget_seconds,
                          "setup_sha256": args.setup_sha256,
                          "payload_manifest_sha256": args.expected_manifest_sha256,
                          "gpu_busy_seconds": None}), flush=True)


if __name__ == "__main__":
    main()
