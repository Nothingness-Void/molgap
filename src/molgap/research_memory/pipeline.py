"""Controller-neutral terminal handoff; never dispatches a successor."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .compiler import rebuild_research_memory
from .finalize import finalize, verified_receipt
from .paths import repo_local_path


def terminal_handoff(result: dict[str, Any]) -> dict[str, Any]:
    return {"trajectory_id": result["trajectory_id"], "finalization_status": result["status"],
            "finalization_id": result["finalization_id"], "outcome": result["outcome"],
            "validation_status": "NOT_RUN", "replay_pool_delta": None,
            "candidate_policy_deltas": None, "next_decision": "SOL_REQUIRED",
            "pipeline_status": "CANONICAL_FINALIZED"}


def rebuild_terminal_handoffs(repo_root: str | Path, handoffs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild once after independent canonical transactions; retry without training."""
    root = Path(repo_root).resolve()
    completed = [item for item in handoffs if item.get("finalization_id")]
    if not completed:
        return handoffs
    try:
        derived = repo_local_path(root, "research_memory/derived")
        previous_pool = _prior(derived / "replay_pool.json", {"entries": []})
        previous_reports = _prior(derived / "policy_backtest.json", {"reports": []})
        # Compilation owns repository validation. Do not validate the same catalog twice.
        payloads = rebuild_research_memory(root)
        pool = json.loads(payloads["replay_pool.json"])
        reports = json.loads(payloads["policy_backtest.json"])
        ids = lambda p: {(e["trajectory_id"], e["run_id"]) for e in p["entries"]}
        delta = {"added": sorted(ids(pool) - ids(previous_pool)),
                 "removed": sorted(ids(previous_pool) - ids(pool))}
        old = {(p["policy_id"], p["policy_version"]): p for p in previous_reports["reports"]}
        candidates = [p for p in reports["reports"] if p["policy_status"] == "candidate"
                      and old.get((p["policy_id"], p["policy_version"])) != p]
        for item in completed:
            item.update(validation_status="VALID", replay_pool_delta=delta,
                        candidate_policy_deltas=candidates, pipeline_status="COMPLETE")
    except (ValueError, OSError) as exc:
        for item in completed:
            item.update(pipeline_status="DERIVED_UPDATE_FAILED", validation_status="FAILED",
                        error=str(exc), recovery_action="terminal-rebuild",
                        canonical_finalization_preserved=True)
    return handoffs


def finalize_rebuild_backtest(repo_root: str | Path, trajectory: str | Path,
                             terminal: str | Path, trace: str | Path | None = None) -> dict[str, Any]:
    result = finalize(repo_root, trajectory, terminal, trace)
    return rebuild_terminal_handoffs(repo_root, [terminal_handoff(result)])[0]


def recover_terminal_derived(repo_root: str | Path, trajectories: list[str | Path]) -> list[dict[str, Any]]:
    """Verify immutable finalizations and refresh only disposable derived outputs."""
    if not trajectories:
        raise ValueError("at least one finalized trajectory is required")
    root = Path(repo_root).resolve()
    handoffs = []
    for trajectory in trajectories:
        path = repo_local_path(root, trajectory)
        if not path.is_dir():
            path = path.parent
        directory = path if path.name == "rml_finalized" else path / "rml_finalized"
        directory.relative_to(root / "experiments")
        receipt = verified_receipt(directory)
        handoffs.append(terminal_handoff({**receipt, "status": "ALREADY_FINALIZED"}))
    return rebuild_terminal_handoffs(root, handoffs)


def _prior(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default
