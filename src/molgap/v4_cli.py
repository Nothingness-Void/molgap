"""Local-only V4 packaging and submission preflight commands."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .v4_audit import audit_v4_paths
from .v4_bundle import build_v4_source_bundle
from .v4_runtime import validate_standard_source_bundle
from .v4_submission import (
    bind_runtime_certificate,
    execute_run_spec,
    read_run_spec,
    write_run_spec,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(prog="molgap-v4")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit")
    audit.add_argument("paths", nargs="+")

    bundle = commands.add_parser("bundle")
    bundle.add_argument("--output", type=Path, required=True)
    bundle.add_argument("--path", action="append", required=True)
    bundle.add_argument("--audit-path", action="append", required=True)

    check_bundle = commands.add_parser("check-bundle")
    check_bundle.add_argument("--archive", type=Path, required=True)
    check_bundle.add_argument("--sha256", required=True)
    check_bundle.add_argument("--commit", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--spec", type=Path, required=True)
    prepare.add_argument("--certificate", type=Path)
    prepare.add_argument("--output", type=Path, required=True)

    bind = commands.add_parser("bind-certificate")
    bind.add_argument("--spec", type=Path, required=True)
    bind.add_argument("--certificate", type=Path, required=True)
    bind.add_argument("--output", type=Path, required=True)
    bind.add_argument("--set", action="append", default=[])

    check_spec = commands.add_parser("check-spec")
    check_spec.add_argument("--spec", type=Path, required=True)
    check_spec.add_argument("--certificate", type=Path)

    execute = commands.add_parser("execute")
    execute.add_argument("--spec", type=Path, required=True)
    execute.add_argument("--mode", choices=("preflight", "train"), required=True)
    execute.add_argument("--certificate", type=Path)

    args = parser.parse_args()
    root = _repo_root()
    if args.command == "audit":
        result = audit_v4_paths([root / path for path in args.paths])
        if result["status"] != "accepted":
            print(json.dumps(result, indent=2, sort_keys=True))
            raise SystemExit(1)
    elif args.command == "bundle":
        audit_result = audit_v4_paths([root / path for path in args.audit_path])
        if audit_result["status"] != "accepted":
            print(json.dumps(audit_result, indent=2, sort_keys=True))
            raise SystemExit(1)
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
        result = build_v4_source_bundle(
            repo_root=root,
            relative_paths=args.path,
            output_dir=args.output,
            source_commit=commit,
        )
        result["static_audit"] = audit_result
    elif args.command == "check-bundle":
        result = {
            "accepted": True,
            "archive_sha256": validate_standard_source_bundle(
                args.archive, args.sha256, args.commit
            ),
        }
    elif args.command == "prepare":
        certificate = (
            json.loads(args.certificate.read_text(encoding="utf-8"))
            if args.certificate
            else None
        )
        result = write_run_spec(
            args.output, json.loads(args.spec.read_text(encoding="utf-8")), certificate
        )
    elif args.command == "check-spec":
        certificate = (
            json.loads(args.certificate.read_text(encoding="utf-8"))
            if args.certificate
            else None
        )
        result = read_run_spec(args.spec, certificate)
    elif args.command == "bind-certificate":
        certificate = json.loads(args.certificate.read_text(encoding="utf-8"))
        spec = json.loads(args.spec.read_text(encoding="utf-8"))
        updates = {}
        for item in args.set:
            if "=" not in item:
                raise ValueError("--set entries must use key=JSON-value")
            key, value = item.split("=", 1)
            updates[key] = json.loads(value)
        result = bind_runtime_certificate(
            spec, certificate, runner_parameter_updates=updates
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(result, handle, sort_keys=True, indent=2)
            handle.write("\n")
    else:
        certificate = (
            json.loads(args.certificate.read_text(encoding="utf-8"))
            if args.certificate
            else None
        )
        spec = read_run_spec(args.spec, certificate)
        result = execute_run_spec(
            spec, mode=args.mode, runtime_certificate=certificate
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
