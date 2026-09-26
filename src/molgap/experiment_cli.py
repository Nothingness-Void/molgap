"""Thin local ExperimentSpec CLI; no submission or scientific authority."""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from pathlib import Path
import sys

from .experiment_launch import (
    _safe_local, build_launch_receipt, canonical_json,
    reconcile_platform_response, write_launch_receipt,
)
from .experiment_package import build_experiment_source_package
from .experiment_preflight import LOADER_MODE, MODEL_MODE, run_experiment_preflight
from .experiment_prospective import plan_prospective
from .experiment_runner import run_experiment
from .experiment_spec import ExperimentSpec
from .experiment_terminal import (
    TerminalDescriptor, execute_terminal_descriptor, translate_terminal_descriptor,
)


class _Parser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **dict(kwargs, allow_abbrev=False))

    def error(self, message):
        raise ValueError(message)

    def print_help(self, file=None):
        print(canonical_json({"help": self.format_help()}), file=file or sys.stdout)


def _local(value: str) -> Path:
    path = Path(value).absolute()
    # Reuse the launch boundary: no network, traversal, link or reparse paths.
    _safe_local(path)
    return path


def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate-spec", "package", "preflight", "run-diagnostic",
                 "launch-receipt", "terminal", "plan-prospective"):
        command = commands.add_parser(name)
        command.add_argument("--spec", required=True, type=_local)
        if name in {"package", "terminal", "plan-prospective"}:
            command.add_argument("--repo-root", required=True, type=_local)
        if name in {"package", "preflight", "run-diagnostic"}:
            command.add_argument("--output", required=True, type=_local)
        if name in {"preflight", "launch-receipt"}:
            command.add_argument("--package", required=True, type=_local)
            command.add_argument("--expected-package-identity", required=True)
        if name == "package":
            command.add_argument("--allowlist", required=True, nargs="+", action="extend")
        elif name == "preflight":
            command.add_argument("--shard-root", required=True, type=_local)
            command.add_argument("--expected-shard-manifest-sha256", required=True)
            command.add_argument("--mode", choices=(LOADER_MODE, MODEL_MODE), default=LOADER_MODE)
        elif name == "run-diagnostic":
            command.add_argument("--device", required=True, nargs="+", action="extend")
            command.add_argument("--worker", choices=("adapter_probe", "construct"), default="adapter_probe")
        elif name == "launch-receipt":
            command.add_argument("--output-dir", required=True, type=_local)
            command.add_argument("--response", type=_local)
        elif name == "terminal":
            command.add_argument("--descriptor", required=True, type=_local)
            command.add_argument("--execute", action="store_true")
    return parser


def _dispatch(args) -> tuple[dict, int]:
    raw = _read(args.spec)
    spec = ExperimentSpec.from_json(raw)
    if raw != spec.to_json():
        raise ValueError("Expected canonical ExperimentSpec JSON bytes (no newline)")
    if args.command == "validate-spec":
        return {"spec_identity": spec.identity, "spec": spec.to_dict()}, 0
    if args.command == "plan-prospective":
        return plan_prospective(spec, args.repo_root)
    if args.command == "package":
        return build_experiment_source_package(spec, args.repo_root, args.allowlist, args.output), 0
    if args.command == "preflight":
        manifest = args.shard_root / "shard_manifest.json"
        _safe_local(manifest)
        result = run_experiment_preflight(
            spec, args.package, args.output,
            expected_package_identity=args.expected_package_identity,
            shard_manifest=manifest if manifest.exists() else None,
            shard_root=args.shard_root,
            expected_shard_manifest_sha256=args.expected_shard_manifest_sha256,
            mode=args.mode,
        )
        return result, 0 if result["status"] in {
            "LOADER_VERIFIED_ONLY", "MODEL_SMOKE_VERIFIED_ONLY",
        } else 1
    if args.command == "run-diagnostic":
        result = run_experiment(spec, args.output, args.device, worker=args.worker)
        return result, 0 if result["status"] == "SUCCEEDED" else 1
    if args.command == "launch-receipt":
        if args.response is None:
            result = build_launch_receipt(
                spec, args.package, expected_package_identity=args.expected_package_identity,
            )
        else:
            result = reconcile_platform_response(
                spec, args.package, _read(args.response),
                expected_package_identity=args.expected_package_identity,
            )
        write_launch_receipt(
            canonical_json(result), args.output_dir, spec, args.package,
            expected_package_identity=args.expected_package_identity,
        )
        return result, 1 if result["submission_state"] == "REJECTED" else 0
    raw = _read(args.descriptor)
    descriptor = TerminalDescriptor.from_json(spec, raw)
    if raw != descriptor.to_json():
        raise ValueError("Expected canonical TerminalDescriptor JSON bytes (no newline)")
    if args.execute:
        results = execute_terminal_descriptor(args.repo_root, spec, descriptor)
        return {"executed": True, "results": results}, 0 if all(
            result.get("pipeline_status") == "COMPLETE" for result in results
        ) else 1
    return {"executed": False, "arms": translate_terminal_descriptor(
        args.repo_root, spec, descriptor,
    )}, 0


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        # Keep any library diagnostics off the machine-readable output channel.
        with redirect_stdout(sys.stderr):
            result, code = _dispatch(args)
        print(canonical_json(result))
        return code
    except Exception as exc:
        print(canonical_json({"status": "ERROR", "error": {
            "type": type(exc).__name__, "message": str(exc),
        }}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
