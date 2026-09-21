"""Command-line interface for the MolGap Research Memory Layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molgap.constants import REPO_ROOT

from .backtest import build_screening_backtest
from .compiler import compile_research_memory, frozen_differences, rebuild_research_memory
from .ready import build_ready_package
from .validate import validate_repository_records
from .paths import repo_local_path


def _root(value: str | None) -> Path:
    return Path(value).resolve() if value else Path(REPO_ROOT).resolve()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m molgap.research_memory")
    parser.add_argument("--repo-root")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    commands.add_parser("rebuild")
    commands.add_parser("status")
    doctor = commands.add_parser("doctor")
    doctor.add_argument("--strict", action="store_true")
    check = commands.add_parser("check")
    check.add_argument("--frozen", action="store_true", required=True)
    ready = commands.add_parser("package-ready")
    ready.add_argument("--trajectory", required=True)
    commands.add_parser("backtest-screening")
    planner = commands.add_parser("plan")
    planner.add_argument("--spec", required=True)
    planner.add_argument("--output", required=True)
    finalize_cmd = commands.add_parser("finalize")
    finalize_cmd.add_argument("--trajectory", required=True)
    finalize_cmd.add_argument("--terminal", required=True)
    finalize_cmd.add_argument("--trace")
    terminal_cmd = commands.add_parser("terminal-pipeline")
    terminal_cmd.add_argument("--trajectory", required=True)
    terminal_cmd.add_argument("--terminal", required=True)
    terminal_cmd.add_argument("--trace")
    terminal_cmd.add_argument("--trace-source")
    terminal_cmd.add_argument("--arm")
    recovery = commands.add_parser("recover-trace")
    recovery.add_argument("--source", action="append", required=True)
    recovery.add_argument("--spec", required=True)
    recovery.add_argument("--output", required=True)
    commands.add_parser("replay-pool")
    policy = commands.add_parser("backtest-policy")
    policy.add_argument("--policy-id")
    policy.add_argument("--policy-version")
    args = parser.parse_args(argv)
    root = _root(args.repo_root)

    if args.command in {"plan", "finalize", "recover-trace", "terminal-pipeline"}:
        from molgap.evidence_pointers import load_json_object
        if args.command == "plan":
            from .plan import plan
            result = plan(root, load_json_object(repo_local_path(root, args.spec)), args.output)
        elif args.command == "recover-trace":
            from .recovery import recover_trace
            result = recover_trace(args.source, load_json_object(repo_local_path(root, args.spec)), args.output, repo_root=root)
        elif args.command == "finalize":
            from .finalize import finalize
            result = finalize(root, args.trajectory, args.terminal, args.trace)
        else:
            from .terminal_wiring import close_terminal_arm
            result = close_terminal_arm(
                repo_root=root,
                trajectory=args.trajectory,
                terminal=args.terminal,
                trace=args.trace,
                arm_identifier=args.arm,
                trace_source=args.trace_source,
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        if args.command == "terminal-pipeline" and result.get("pipeline_status") != "COMPLETE":
            raise SystemExit(1)
        return

    if args.command == "validate":
        validated = validate_repository_records(root)
        print(json.dumps({key: len(value) for key, value in validated["records"].items()}, indent=2))
        return
    if args.command == "rebuild":
        payloads = rebuild_research_memory(root)
        print(json.dumps({"status": "rebuilt", "files": sorted(payloads)}, indent=2))
        return
    if args.command == "check":
        differences = frozen_differences(root)
        if differences:
            raise SystemExit("stale derived outputs: " + ", ".join(differences))
        print("RML derived outputs are frozen and current")
        return
    if args.command == "status":
        path = root / "research_memory" / "derived" / "research_summary.md"
        if not path.is_file():
            raise SystemExit("research summary is missing; run rebuild")
        print(path.read_text(encoding="utf-8"), end="")
        return
    if args.command == "doctor":
        report = json.loads(
            (root / "research_memory" / "derived" / "completeness_report.json").read_text(
                encoding="utf-8"
            )
        )
        print(json.dumps(report, indent=2))
        failing = {"ERROR"} | ({"WARN"} if args.strict else set())
        if any(issue["severity"] in failing for issue in report["issues"]):
            raise SystemExit(1)
        return
    validated = validate_repository_records(root)
    records = validated["records"]
    if args.command in {"replay-pool", "backtest-policy"}:
        from .replay import build_replay_pool
        pool = build_replay_pool(root, records)
        if args.command == "replay-pool":
            result = pool
        else:
            from .policy import load_policy_registry
            from .backtest import build_policy_backtests
            policies = load_policy_registry(root)
            if args.policy_version and not args.policy_id:
                raise ValueError("--policy-version requires --policy-id")
            if args.policy_id:
                policies = [p for p in policies if p["policy_id"] == args.policy_id and
                            (not args.policy_version or p["version"] == args.policy_version)]
                if not policies:
                    raise ValueError("unknown policy/version")
            result = build_policy_backtests(policies, pool)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if args.command == "backtest-screening":
        result = build_screening_backtest([record for _, record in records["traces"]])
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    trajectories = {
        record["trajectory_id"]: (path, record) for path, record in records["trajectories"]
    }
    evidence = {record["evidence_id"]: record for _, record in records["evidence"]}
    trajectory_entry = trajectories.get(args.trajectory)
    if trajectory_entry is None:
        raise SystemExit(f"unknown trajectory: {args.trajectory}")
    trajectory_path, trajectory = trajectory_entry
    costs = {
        record["cost_event_id"]: record
        for _, record in records["costs"]
        if record["trajectory_id"] == args.trajectory
    }
    package = build_ready_package(trajectory, evidence, costs)
    output = trajectory_path.parent / "handoff" / "READY_FOR_DESKTOP.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(package, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)
