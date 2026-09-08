"""No-model-inference acceptance for the QM9 local-hierarchy screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def accept(root: Path, *, source_commit: str) -> dict:
    metrics_path = root / "metrics.json"
    completion_path = root / "completion_manifest.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    required = {
        "format": "molgap-qm9-edgestate-local-hierarchy-screen-v2",
        "complete": True,
        "source_commit": source_commit,
        "architecture": "ogb_edge_state_structural_gps9",
        "seed": 42,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
    }
    for key, value in required.items():
        if metrics.get(key) != value or completion.get(key) != value:
            raise RuntimeError(f"Result contract changed for {key}")
    for relative, expected in completion["artifact_sha256"].items():
        path = root / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Artifact hash changed: {relative}")
    controls = [
        metrics["results"][name]["validation_gap_mae_eV"]
        for name in ("scratch_a", "scratch_b")
    ]
    candidate = metrics["results"]["local_hierarchy"]["finetune"][
        "validation_gap_mae_eV"
    ]
    values = controls + [
        candidate,
        metrics["control_mean_gap_mae_eV"],
        metrics["control_spread_eV"],
        metrics["candidate_gain_eV"],
        metrics["required_gain_eV"],
    ]
    if not all(math.isfinite(float(value)) for value in values):
        raise RuntimeError("Non-finite result value")
    mean = sum(controls) / 2
    spread = abs(controls[0] - controls[1])
    gain = mean - candidate
    required_gain = max(0.002, 2 * spread)
    nominated = candidate < min(controls) and gain >= required_gain
    checks = {
        "mean": abs(mean - metrics["control_mean_gap_mae_eV"]) < 1e-12,
        "spread": abs(spread - metrics["control_spread_eV"]) < 1e-12,
        "gain": abs(gain - metrics["candidate_gain_eV"]) < 1e-12,
        "required_gain": abs(required_gain - metrics["required_gain_eV"]) < 1e-12,
        "nomination": nominated == metrics["pcqm_transfer_nominated"],
    }
    if not all(checks.values()):
        raise RuntimeError(f"Result arithmetic changed: {checks}")
    return {
        "format": "molgap-qm9-edgestate-local-hierarchy-acceptance-v2",
        "accepted": True,
        "model_inference_executed": False,
        "source_commit": source_commit,
        "cache_aggregate_sha256": metrics["cache_aggregate_sha256"],
        "scratch_gap_mae_eV": controls,
        "candidate_gap_mae_eV": candidate,
        "candidate_gain_eV": gain,
        "required_gain_eV": required_gain,
        "pcqm_transfer_nominated": nominated,
        "official_pcqm_roles_read": False,
        "test_role_read": False,
        "checks": checks,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = accept(args.root, source_commit=args.source_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
