"""Command-line interface for the MolGap Research Memory Layer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess, sys

from molgap.constants import REPO_ROOT

from .backtest import build_screening_backtest
from .compiler import compile_research_memory, frozen_differences, rebuild_research_memory
from .ready import build_ready_package
from .validate import validate_repository_records
from .paths import repo_local_path
from .portability import committed_head_differences, missing_locally_claimed_artifacts
from .schemas import FAILURE_MODES, TRAJECTORY_OUTCOMES


def _root(value: str | None) -> Path:
    top = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True).stdout.strip()
    return Path(value or top or REPO_ROOT).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="platforms/local/run.py --module molgap.research_memory --",
        description="MolGap Research Memory Layer CLI. Run '<command> --help' for flags; "
                    "low-level commands are omitted below (see docs/operations/LIFECYCLE_MAP.md).",
    )
    parser.add_argument("--repo-root", help="default: the git root of the current directory")
    commands = parser.add_subparsers(dest="command", required=True, metavar="<command>")
    commands.add_parser("validate", description="Validate all canonical RML records without rewriting derived indexes.")
    commands.add_parser("rebuild", help="Rebuild derived RML after standalone canonical edits (not after normal closure)")
    commands.add_parser("status", help="Summarize current RML record counts and index status")
    term_help = "whole word, case-insensitive, anywhere in the card"
    cards = commands.add_parser("cards", help="List trajectory summary cards (--outcome / --term / --compact)")
    cards.add_argument("--outcome", choices=sorted(TRAJECTORY_OUTCOMES), help="Filter by exact trajectory outcome")
    cards.add_argument("--term", help=term_help)
    lookup_term_help = ("one or more whole words, case-insensitive; a card must contain every word "
                        "(e.g. --term ensemble seed)")
    lookup = commands.add_parser("lookup", help="Search prior trajectories by mechanism, family, failure mode or term (--compact recommended)")
    lookup.add_argument("--mechanism", nargs="+", metavar="WORD",
                        help="one or more case-insensitive substrings; each must occur in the card's "
                             "trajectory id, mechanism, family id or failure mode (e.g. --mechanism init noise)")
    lookup.add_argument("--term", nargs="+", metavar="WORD", help=lookup_term_help)
    lookup.add_argument("--failure-mode", choices=sorted(FAILURE_MODES), help="Filter by exact failure mode")
    lookup.add_argument("--family", help="case-insensitive substring of family_id")
    compact_help = "one line per card: id | outcome | family | mechanism slug | effect_meV | failure_mode"
    ref_help = "also read the cards only this git revision has committed (local wins); every card names its source"
    for command in (cards, lookup):
        command.add_argument("--compact", action="store_true", help=compact_help)
        command.add_argument("--ref", help=ref_help)
    levers = commands.add_parser("levers", help="Read-only table of tested levers and noise for one family")
    levers.add_argument("--family", required=True,
                        help="case-insensitive family id substring, or a glob with '*' (e.g. 'pcqm-k1-*-100k')")
    gaps = commands.add_parser("gaps", help="List sparse families and cross-scale contradictions (--compact)")
    gaps.add_argument("--ref", help=ref_help)
    gaps.add_argument("--compact", action="store_true",
                      help="one line per item: sparse | family | counts, or contradiction | arm | a vs b")
    doctor = commands.add_parser("doctor", help="Check RML completeness or clean unclosed staging (--clean-staging)")
    doctor.add_argument("--strict", action="store_true")
    doctor.add_argument("--clean-staging", action="store_true", help="list terminal_staging of unclosed arms; --apply removes them")
    doctor.add_argument("--apply", action="store_true")
    doctor.add_argument("--run-id", help="with --clean-staging --apply: the one run to clean")
    check = commands.add_parser("check", help="Verify frozen RML records and derived indexes (--frozen)")
    check.add_argument("--frozen", action="store_true", required=True)
    check.add_argument("--portable", action="store_true")
    ready = commands.add_parser("package-ready", description="Build a READY_FOR_DESKTOP handoff package for a trajectory.")
    ready.add_argument("--trajectory", required=True)
    commands.add_parser("backtest-screening", description="Run screening backtest over recorded traces.")
    planner = commands.add_parser("plan", description="Generate prospective RML records from a spec.")
    planner.add_argument("--spec", required=True)
    planner.add_argument("--output", required=True)
    finalize_cmd = commands.add_parser("finalize", description="Finalize one trajectory with a terminal record.")
    finalize_cmd.add_argument("--trajectory", required=True)
    finalize_cmd.add_argument("--terminal", required=True)
    finalize_cmd.add_argument("--trace")
    terminal_cmd = commands.add_parser("terminal-pipeline", description="Close one terminal arm and update derived RML.")
    terminal_cmd.add_argument("--trajectory", required=True)
    terminal_cmd.add_argument("--terminal", required=True)
    terminal_cmd.add_argument("--trace")
    terminal_cmd.add_argument("--trace-source")
    terminal_cmd.add_argument("--arm")
    terminal_rebuild = commands.add_parser("terminal-rebuild", help="Recovery only: rebuild derived RML after a failed closure update")
    terminal_rebuild.add_argument("--trajectory", action="append", required=True)
    recovery = commands.add_parser("recover-trace", description="Recover a canonical trace manifest from raw source files.")
    recovery.add_argument("--source", action="append", required=True)
    recovery.add_argument("--spec", required=True)
    recovery.add_argument("--output", required=True)
    analyze = commands.add_parser("analyze", help="Post-hoc analysis over saved predictions (--op / --predictions)")
    from .saved_analysis import OPS
    analyze.add_argument("--op", required=True, choices=OPS)
    analyze.add_argument("--predictions", nargs="+", required=True, help="PATH... or ARM=PATH...")
    analyze.add_argument("--pair", action="append", default=[], help="CAND=REF arm names (summary op)")
    analyze.add_argument("--role-id", required=True)
    analyze.add_argument("--out", required=True)
    commands.add_parser("replay-pool", description="Inspect the replayable trajectory pool.")
    policy = commands.add_parser("backtest-policy", description="Backtest an early-stop or screening policy.")
    policy.add_argument("--policy-id")
    policy.add_argument("--policy-version")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    getattr(sys.stdout, "reconfigure", lambda **_: None)(encoding="utf-8")
    root = _root(args.repo_root)

    if args.command == "terminal-rebuild":
        from .pipeline import recover_terminal_derived
        results = recover_terminal_derived(root, args.trajectory)
        print(json.dumps(results, indent=2, sort_keys=True))
        if any(result["pipeline_status"] != "COMPLETE" for result in results):
            raise SystemExit(1)
        return
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

    if args.command == "analyze":
        from .saved_analysis import run, write_output
        named = [p.partition("=")[::2] for p in args.predictions]
        paths = dict(named) if all(path for _, path in named) else args.predictions
        result = run(args.op, paths, role_id=args.role_id, pairs=[tuple(p.split("=", 1)) for p in args.pair] or None)
        write_output(result, args.out, root)
        print(json.dumps({"op": args.op, "rows": result["row_count"], "out": args.out, "posthoc": True}))
        return
    if args.command == "validate":
        validated = validate_repository_records(root)
        summary = {key: len(value) for key, value in validated["records"].items()}
        summary["known_gaps"] = validated["known_gaps"]
        print(json.dumps(summary, indent=2))
        return
    if args.command == "rebuild":
        payloads = rebuild_research_memory(root)
        print(json.dumps({"status": "rebuilt", "files": sorted(payloads)}, indent=2))
        return
    if args.command == "check":
        differences = frozen_differences(root)
        if differences:
            raise SystemExit("stale derived outputs: " + ", ".join(differences))
        if args.portable:
            records = validate_repository_records(root)["records"]
            try:
                head_differences = committed_head_differences(root, records)
                missing_artifacts = missing_locally_claimed_artifacts(root, records)
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            failures = []
            if missing_artifacts:
                failures.append(
                    "V5 evidence claims missing local artifacts:\n  "
                    + "\n  ".join(missing_artifacts)
                )
            missing_from_head = head_differences["missing_from_head"]
            if missing_from_head:
                failures.append(
                    "RML portable inputs are absent from Git HEAD:\n  "
                    + "\n  ".join(missing_from_head)
                )
            changed_from_head = head_differences["changed_from_head"]
            if changed_from_head:
                failures.append(
                    "RML portable inputs, runtime, or derived outputs differ from Git HEAD:\n  "
                    + "\n  ".join(changed_from_head)
                )
            if failures:
                raise SystemExit("\n".join(failures))
            print(
                "RML derived outputs are frozen and current; "
                "committed evidence closure, runtime, derived outputs, and local artifact claims are complete"
            )
        else:
            print("RML derived outputs are frozen and current")
        return
    if args.command == "lookup":
        # Several words travel as one space-separated string; matching splits it again.
        args.mechanism, args.term = (" ".join(v) if v else None for v in (args.mechanism, args.term))
    if args.command in {"lookup", "gaps", "cards", "levers"}:
        from . import cards as _cards
        built = None if getattr(args, "ref", None) else _cards.build_cards(root)
        try:
            built = built if built is not None else _cards.with_ref(root, args.ref)
        except ValueError as exc:  # an unreadable revision fails loudly, never as fewer cards
            raise SystemExit(f"--ref {args.ref}: {exc}") from exc
        if args.command == "levers":
            pointer = _cards.historical_pointer(root, built, family=args.family)
            print("\n".join([*_cards.levers(built, args.family), *filter(None, [pointer])]))
            return
        if args.command == "cards":
            selected = _cards.select(built, outcome=args.outcome, term=args.term)
            print("\n".join(_cards.compact_line(c) if args.compact else json.dumps(c, ensure_ascii=False)
                            for c in selected))
            return
        result = (_cards.lookup(built, **{k: getattr(args, k) for k in ("mechanism", "failure_mode", "family", "term")})
                  if args.command == "lookup" else _cards.gaps(built))
        if args.command == "gaps" and args.compact:
            print("\n".join(_cards.gaps_compact(result)))
            return
        pointer = args.command == "lookup" and _cards.historical_pointer(
            root, built, mechanism=args.mechanism, term=args.term, family=args.family)
        if pointer:
            result["historical_pointer"] = pointer
        if args.command == "lookup" and args.compact:
            print("\n".join([*map(_cards.compact_line, result["matches"]),
                             f"# families without a measured negative effect: {len(result['unfalsified'])} "
                             f"({sum(bool(u['negative_without_effect']) for u in result['unfalsified'])} with a "
                             "NEGATIVE_UNDER_CONTRACT arm, effect not recorded); "
                             f"search_terms: {' '.join(result['search_terms'])}", *filter(None, [pointer])]))
            return
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "status":
        path = root / "research_memory" / "derived" / "research_summary.md"
        path.is_file() or sys.exit("research summary is missing; run rebuild")
        print(path.read_text(encoding="utf-8"), end="")
        return
    if args.command == "doctor":
        if args.clean_staging:
            from molgap.experiment_terminal_staging import clean_unclosed_staging
            return print(json.dumps(clean_unclosed_staging(root, args.apply, args.run_id), indent=2))
        report = json.loads((root / "research_memory/derived/completeness_report.json").read_text(encoding="utf-8"))
        print(json.dumps({**report, "molgap_imported_from": __file__}, indent=2))
        failing = {"ERROR"} | ({"WARN"} if args.strict else set())
        sys.exit(any(issue["severity"] in failing for issue in report["issues"]))
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
        from .replay import replay_manifests
        result = build_screening_backtest(replay_manifests(root, records))
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
