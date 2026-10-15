"""Package a committed MolGap branch and submit one IMS V4 run."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path


RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
REMOTE_BOUNDARY = "/lustre/home/users/sm2/chou"
REMOTE_INFRA = f"{REMOTE_BOUNDARY}/molgap-v4-submit"


def run(command: list[str], *, cwd: Path | None = None, capture: bool = False) -> str:
    result = subprocess.run(command, cwd=cwd, check=True, text=True, capture_output=capture)
    return result.stdout.strip() if capture else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(repo: Path, template: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    status = run(["git", "status", "--porcelain"], cwd=repo, capture=True)
    if status:
        raise RuntimeError("Source repository must be clean before V4 packaging")
    commit = run(["git", "rev-parse", "HEAD"], cwd=repo, capture=True)
    spec = json.loads(template.read_text(encoding="utf-8"))
    run_id = str(spec.get("run_id", ""))
    if not RUN_ID.fullmatch(run_id):
        raise RuntimeError("Unsafe run_id")
    output.mkdir(parents=True)
    archive = output / "source.tar.gz"
    run(["git", "archive", "--format=tar.gz", "-o", str(archive), commit], cwd=repo)
    archive_sha256 = sha256_file(archive)
    spec["source"] = {"commit": commit, "archive_sha256": archive_sha256}
    inventory = []
    inventory_by_path = {}
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle.getmembers():
            if not member.isfile():
                continue
            stream = handle.extractfile(member)
            if stream is None:
                raise RuntimeError(f"Cannot read archive member: {member.name}")
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            value = digest.hexdigest()
            inventory.append({"path": member.name, "sha256": value})
            inventory_by_path[member.name] = value
    contract_path = spec.get("contract", {}).get("path")
    if contract_path not in inventory_by_path:
        raise RuntimeError(f"Contract is not present in committed source: {contract_path}")
    spec["contract"]["sha256"] = inventory_by_path[contract_path]
    (output / "run_spec.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "source_files.json").write_text(
        json.dumps({"format": "molgap-source-files-v1", "files": inventory}, indent=2) + "\n",
        encoding="utf-8",
    )
    return spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--host", default="ccfep.ims.ac.jp")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", default="sm2")
    args = parser.parse_args()
    spec = package(args.repo.resolve(), args.spec.resolve(), args.payload.resolve())
    run_id = spec["run_id"]
    destination = f"{args.user}@{args.host}"
    inbox = f"{REMOTE_INFRA}/inbox/{run_id}"
    ssh = ["ssh", "-T", "-i", str(args.key.resolve()), "-p", str(args.port), destination]
    scp = ["scp", "-i", str(args.key.resolve()), "-P", str(args.port)]
    run(ssh + [f"cd {REMOTE_BOUNDARY} && mkdir -p {inbox}"])
    run(scp + [str(args.payload / "source.tar.gz"), str(args.payload / "run_spec.json"), str(args.payload / "source_files.json"), f"{destination}:{inbox}/"])
    run(ssh + [f"cd {REMOTE_BOUNDARY} && {REMOTE_BOUNDARY}/molgap/.venv/bin/python {REMOTE_INFRA}/launcher.py submit --inbox {inbox}"])


if __name__ == "__main__":
    main()
