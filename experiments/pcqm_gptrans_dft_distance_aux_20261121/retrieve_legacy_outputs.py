"""Retrieve a finished IMS serial-chain run whose stage record has no worker digests (legacy launcher).

`submit_from_windows.py --retrieve-to` needs `outputs_sha256` in the stage record. This run's worker came from the
earlier launcher, which wrote none. The remote side is read-only here: `find` lists files, `sha256sum` hashes them, `scp`
copies them. Every local copy must equal the remote hash. The fixed-dataset mount copies under `dataset_mount/` are skipped
(fixed data, identical everywhere). Output lands in a new directory; an existing one is only re-verified.

Usage: python retrieve_legacy_outputs.py --key KEY --run-id gptdft-chain-20261121 --to DIR
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

BOUNDARY = "/lustre/home/users/sm2/chou"
RUNS = f"{BOUNDARY}/molgap-v4-submit/runs"
SAFE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def remote(ssh, command):
    return subprocess.run(ssh + [command], check=True, stdout=subprocess.PIPE, text=True).stdout


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--key", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--to", required=True, type=Path)
    parser.add_argument("--host", default="sm2@ccfep.ims.ac.jp")
    parser.add_argument("--port", default="22")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{2,63}", args.run_id):
        raise SystemExit("unsafe run id")
    root = f"{RUNS}/{args.run_id}/output/training"
    ssh = ["ssh", "-T", "-i", args.key, "-p", args.port, args.host]
    scp = ["scp", "-i", args.key, "-P", args.port]
    listing = remote(ssh, f"cd {root} && find . -type f ! -path './*/dataset_mount/*' | sort | xargs sha256sum")
    files = {}
    for line in listing.splitlines():
        digest, relative = line.split(None, 1)
        relative = relative.strip().removeprefix("./")
        parts = relative.split("/")
        if not re.fullmatch(r"[a-f0-9]{64}", digest) or not all(SAFE.match(p) for p in parts):
            raise SystemExit(f"unsafe remote entry: {line!r}")
        files[relative] = digest
    if not files:
        raise SystemExit("no remote files found")
    target = args.to
    target.mkdir(parents=True, exist_ok=True)
    for relative, digest in files.items():
        local = target / relative
        if local.exists() and sha256_file(local) == digest:
            continue
        if local.exists():
            raise SystemExit(f"existing file differs from remote hash: {relative}")
        local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(scp + [f"{args.host}:{root}/{relative}", str(local)], check=True)
        if sha256_file(local) != digest:
            local.unlink()
            raise SystemExit(f"copy differs from remote hash: {relative}")
    receipt = {"status": "RETRIEVED_LEGACY", "run_id": args.run_id, "remote_root": root,
               "skipped": "dataset_mount/ (fixed datasets)", "files_sha256": files}
    (target / "retrieval_legacy.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "output_root": str(target), "files": len(files)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
