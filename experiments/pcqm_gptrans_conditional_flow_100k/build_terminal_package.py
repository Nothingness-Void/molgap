"""Build the immutable RML terminal input from retained accepted metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPERIMENT = Path("experiments/pcqm_gptrans_conditional_flow_100k")
ACCEPTANCE = EXPERIMENT / "results/rml_acceptance_v2.json"
EVENTS = EXPERIMENT / "results/rml_observed_events_v2.json"
EVIDENCE = EXPERIMENT / "v5_evidence.json"
OUTPUT = EXPERIMENT / "results/terminal_v2.json"


def _load(root: Path, relative: Path | str) -> dict:
    return json.loads((root / relative).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(repo_root: Path, finalized_at: str) -> dict:
    root = repo_root.resolve()
    acceptance = _load(root, ACCEPTANCE)
    events = _load(root, EVENTS)
    evidence = _load(root, EVIDENCE)
    pointers = set(evidence["authority"]["pointers"])
    pointers.update(artifact["locator"] for artifact in evidence["artifacts"])
    pointers.update(
        {
            ACCEPTANCE.as_posix(),
            EVENTS.as_posix(),
            acceptance["trajectory_decision"]["decision_ref"],
        }
    )
    artifact_hashes = {}
    for pointer in sorted(pointers):
        path = (root / pointer).resolve()
        path.relative_to(root)
        if not path.is_file():
            raise FileNotFoundError(path)
        artifact_hashes[pointer] = _sha256(path)

    package = {
        "format": "molgap-rml-terminal-package-v1",
        "trajectory_id": "TC-gptrans-conditional-flow-100k-s42",
        "run_id": "nothingnessvoid/molgap-gptrans-conditional-flow-s42:v2",
        "action_id": "A001",
        "finalized_at": finalized_at,
        "acceptance_ref": ACCEPTANCE.as_posix(),
        "artifact_hashes": artifact_hashes,
        "evidence": evidence,
        "decision": acceptance["trajectory_decision"],
        "costs": events["costs"],
        "roles": events["roles"],
        "role_use": acceptance["role_use"],
    }
    output = root / OUTPUT
    if output.exists():
        existing = _load(root, OUTPUT)
        if existing != package:
            raise RuntimeError("terminal package already exists with different bytes")
        return package
    output.write_text(
        json.dumps(package, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--finalized-at", required=True)
    args = parser.parse_args()
    package = build(args.repo_root, args.finalized_at)
    print(json.dumps({
        "status": "built",
        "trajectory_id": package["trajectory_id"],
        "artifact_count": len(package["artifact_hashes"]),
        "output": OUTPUT.as_posix(),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
