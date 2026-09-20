"""Controller-neutral terminal handoff; never dispatches a successor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .compiler import rebuild_research_memory
from .finalize import finalize
from .validate import validate_repository_records
from .paths import repo_local_path


def finalize_rebuild_backtest(repo_root: str | Path, trajectory: str | Path,
                             terminal: str | Path, trace: str | Path | None = None) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    derived = repo_local_path(root, "research_memory/derived")
    previous_pool = _prior(repo_local_path(root, derived / "replay_pool.json"), {"entries": []})
    previous_reports = _prior(repo_local_path(root, derived / "policy_backtest.json"), {"reports": []})
    result = finalize(root, trajectory, terminal, trace)
    handoff = {"trajectory_id": result["trajectory_id"], "finalization_status": result["status"],
               "finalization_id": result["finalization_id"], "outcome": result["outcome"],
               "validation_status": "NOT_RUN", "replay_pool_delta": None,
               "candidate_policy_deltas": None, "next_decision": "SOL_REQUIRED"}
    try:
        validate_repository_records(root)
        handoff["validation_status"] = "VALID"
        payloads = rebuild_research_memory(root)
        pool = json.loads(payloads["replay_pool.json"])
        reports = json.loads(payloads["policy_backtest.json"])
        ids = lambda p: {(e["trajectory_id"], e["run_id"]) for e in p["entries"]}
        handoff["replay_pool_delta"] = {"added": sorted(ids(pool) - ids(previous_pool)),
                                        "removed": sorted(ids(previous_pool) - ids(pool))}
        old = {(p["policy_id"], p["policy_version"]): p for p in previous_reports["reports"]}
        handoff["candidate_policy_deltas"] = [p for p in reports["reports"]
                                              if p["policy_status"] == "candidate" and old.get((p["policy_id"], p["policy_version"])) != p]
        handoff["pipeline_status"] = "COMPLETE"
    except (ValueError, OSError) as exc:
        handoff["pipeline_status"] = "DERIVED_UPDATE_FAILED"
        if handoff["validation_status"] == "NOT_RUN":
            handoff["validation_status"] = "INVALID"
        handoff["error"] = str(exc)
    return handoff


def _prior(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default
