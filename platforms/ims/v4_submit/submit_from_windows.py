"""Package a committed MolGap branch and submit one IMS V4 run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tarfile
from pathlib import Path, PureWindowsPath
from tempfile import TemporaryDirectory
from uuid import uuid4


RUN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{2,63}$")
REMOTE_BOUNDARY = "/lustre/home/users/sm2/chou"
REMOTE_INFRA = f"{REMOTE_BOUNDARY}/molgap-v4-submit"
LAUNCHER = Path(__file__).with_name("launcher.py")  # runs from the inbox, so it needs no separate remote install


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
    spec["source"] = {"commit": commit, "archive_sha256": sha256_file(archive)}
    contract_path = spec["contract"]["path"]
    with tarfile.open(archive, "r:gz") as handle:
        member = handle.getmember(contract_path)
        if not member.isfile():
            raise RuntimeError(f"Contract is not a regular file: {contract_path}")
        with handle.extractfile(member) as stream:
            spec["contract"]["sha256"] = hashlib.sha256(stream.read()).hexdigest()
    (output / "run_spec.json").write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return spec


def reusable(template: Path | None, payload: Path) -> dict:
    path = payload / "run_spec.json"
    spec = json.loads(path.read_text(encoding="utf-8"))
    if template is not None:
        requested = json.loads(template.read_text(encoding="utf-8"))
        requested["source"] = spec["source"]
        requested["contract"]["sha256"] = spec["contract"]["sha256"]
        if requested != spec:
            raise ValueError("Requested IMS spec differs from the frozen payload")
    return spec


def retrieve_declared(spec: dict, observed: dict, scp: list[str], destination: str, target: Path) -> None:
    """Publish only complete, worker-hashed outputs from this staged run."""
    submission = observed.get("submission") or {}
    stages = observed.get("stages") or {}
    if submission.get("status") != "submitted" or set(stages) != set(spec["stages"]):
        raise ValueError("Retrieval requires the complete records of a bound IMS run")
    files = {}
    for name, stage in stages.items():
        declared = spec["stages"][name]["required_outputs"]
        digests = (stage or {}).get("outputs_sha256")
        if ((stage or {}).get("status") != "complete" or not isinstance(digests, dict)
                or set(digests) != set(declared)):
            raise ValueError(f"IMS stage {name} has no complete worker digest record")
        for relative, digest in digests.items():
            parts = relative.split("/")
            if (not all(re.fullmatch(r"[a-zA-Z0-9_][a-zA-Z0-9_.-]*", part)
                        and not part.endswith(".") and not PureWindowsPath(part).is_reserved() for part in parts)
                    or parts[0].casefold() == "retrieval.json" or not re.fullmatch(r"[a-f0-9]{64}", digest)):
                raise ValueError(f"Invalid IMS output binding: {relative}")
            if files.setdefault(relative, digest) != digest:
                raise ValueError(f"Conflicting IMS hashes: {relative}")
    if len({path.casefold() for path in files}) != len(files):
        raise ValueError("IMS outputs conflict on case-insensitive paths")
    receipt = {"status": "RETRIEVED", "observation": observed}
    if target.exists():
        retained = json.loads((target / "retrieval.json").read_text(encoding="utf-8"))
        if retained != receipt or any(sha256_file(target / path) != digest for path, digest in files.items()):
            raise ValueError("Existing IMS retrieval differs from the bound outputs")
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        remote = f"{destination}:{REMOTE_INFRA}/runs/{spec['run_id']}/output"
        with TemporaryDirectory(dir=target.parent, prefix=".ims-retrieve-") as tmp:
            staged = Path(tmp) / "verified"
            staged.mkdir()
            for relative, digest in files.items():
                local = staged / relative
                local.parent.mkdir(parents=True, exist_ok=True)
                run(scp + [f"{remote}/{relative}", str(local)])
                if sha256_file(local) != digest:
                    raise ValueError(f"IMS output differs from worker hash: {relative}")
            (staged / "retrieval.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            staged.rename(target)
    print(json.dumps({"status": "RETRIEVED", "output_root": str(target), "files": len(files)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--key", type=Path, default=os.getenv("MOLGAP_IMS_SSH_KEY"))
    operation = parser.add_mutually_exclusive_group()
    operation.add_argument("--status", action="store_true", help="Read durable IMS records without submitting")
    operation.add_argument("--retrieve-to", type=Path, help="Retrieve and verify the spec's completed outputs")
    operation.add_argument("--package-only", action="store_true", help="Freeze the payload (clean commit archive, spec hashes) and stop; no key, no ssh, nothing queued")
    operation.add_argument("--resume", action="store_true", help="Resume after reconciling the paused job with the scheduler")
    parser.add_argument("--host", default="ccfep.ims.ac.jp")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", default="sm2")
    args = parser.parse_args()
    if (args.key is None and not args.package_only) or (args.spec is None and args.payload is None):
        parser.error("Set --key or MOLGAP_IMS_SSH_KEY, and pass --spec or --payload")
    template = args.spec.resolve() if args.spec else None
    if template:
        run_id = json.loads(template.read_text(encoding="utf-8")).get("run_id", "")
        if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
            parser.error("Unsafe run_id")
    payload = (args.payload or Path(os.getenv("MOLGAP_IMS_PAYLOAD_ROOT",
               Path.home() / "molgap-ims-payloads")) / run_id).resolve()
    retained = payload.exists()
    if retained or args.status or args.resume or args.retrieve_to is not None or template is None:
        spec = reusable(template, payload)
    else:
        repo = args.repo or Path(run(["git", "rev-parse", "--show-toplevel"], cwd=template.parent, capture=True))
        spec = package(repo.resolve(), template, payload)
    run_id = spec["run_id"]
    if not RUN_ID.fullmatch(run_id):
        parser.error("Unsafe run_id in frozen payload")
    if args.package_only:
        print(json.dumps({"payload": str(payload), "source": spec["source"], "contract_sha256": spec["contract"]["sha256"]}, indent=2))
        return
    destination = f"{args.user}@{args.host}"
    ssh = ["ssh", "-T", "-i", str(args.key.resolve()), "-p", str(args.port), destination]
    scp = ["scp", "-i", str(args.key.resolve()), "-P", str(args.port)]
    python = f"{REMOTE_BOUNDARY}/molgap/.venv/bin/python"
    if args.status or args.retrieve_to is not None or (retained and not args.resume):
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        observation = subprocess.run(
            ssh + [f"cd {REMOTE_BOUNDARY} && {python} - status --run-id {run_id} --spec-sha256 {digest}"],
            input=LAUNCHER.read_bytes(), stdout=subprocess.PIPE, check=True)
        observed = json.loads(observation.stdout)
        if args.retrieve_to is not None:
            retrieve_declared(spec, observed, scp, destination, args.retrieve_to.resolve())
            return
        if args.status:
            print(json.dumps(observed, indent=2))
            return
        if observed.get("status") != "NOT_STAGED":
            submission = observed.get("submission")
            if not submission or submission.get("status") != "submitted":
                raise RuntimeError("IMS submission needs reconciliation; inspect --status and the scheduler")
            print(json.dumps(submission, indent=2))
            return
    inbox = f"{REMOTE_INFRA}/inbox/{run_id}-{uuid4().hex}"
    run(ssh + [f"cd {REMOTE_BOUNDARY} && mkdir -p {inbox}"])
    files = ["run_spec.json"] if args.resume else ["run_spec.json", "source.tar.gz"]
    run(scp + [*(str(payload / name) for name in files), str(LAUNCHER), f"{destination}:{inbox}/"])
    resume = " --resume" if args.resume else ""
    run(ssh + [f"cd {REMOTE_BOUNDARY} && {python} {inbox}/launcher.py submit --inbox {inbox}{resume}"])


if __name__ == "__main__":
    main()
