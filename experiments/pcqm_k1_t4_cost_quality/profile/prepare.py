"""Metadata-only plan generation and exact-byte accepted A100 sample reuse."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OVERLAYS = ("src/molgap/k1_execution_profile.py", "src/molgap/evidence_pointers.py")
HERE = Path(__file__).resolve().parent
PACKAGING_FILES = ("bootstrap.py", "run.sh", "setup.sh", "kaggle_entry.py", "protocol.md")


def default_plan(accepted_payload, trajectory, plan_receipt):
    """No pickle decode or execution. Parent reviews pins before preparation."""
    from molgap.training_reproducibility import sha256_file
    accepted_payload = Path(accepted_payload).resolve()
    manifest_path = accepted_payload / "payload_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    def binding(path):
        path = Path(path).resolve()
        return {"path": str(path), "sha256": sha256_file(path)}
    sources = {name: digest for name, digest in manifest["files"].items() if name.startswith("src/")}
    overrides = {name: binding(ROOT / name) for name in OVERLAYS}
    sources.update({name: item["sha256"] for name, item in overrides.items()})
    checkpoint = binding(accepted_payload / "selected.pt")
    checkpoint["source_sha256"] = manifest["checkpoint"]["source_sha256"]
    sample = binding(accepted_payload / "train_probe.pt")
    if checkpoint["sha256"] != manifest["checkpoint"]["sha256"] or sample["sha256"] != manifest["files"]["train_probe.pt"]:
        raise ValueError("Retained accepted payload differs; no replacement permitted")
    return {"source_input_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "trajectory": binding(trajectory), "plan_receipt": binding(plan_receipt),
            "accepted_profile_manifest": binding(manifest_path), "source_root": str(accepted_payload),
            "source_files": sources, "source_overrides": overrides,
            "packaging_files": {name: binding(HERE / name) for name in PACKAGING_FILES},
            "checkpoint": checkpoint, "train_probe": sample,
            "sample_source_idx": manifest["sample_source_idx"],
            "role": "official-train-derived-500k", "geometry_used": False}


def prepare(plan_path: Path, output: Path):
    from molgap.experiment_package import _allowlist, _source
    from molgap.experiment_launch import publish_immutable_bytes
    from molgap.training_reproducibility import atomic_json, sha256_file
    from molgap.k1_execution_profile import verify_t4_payload

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    def pinned(binding):
        path = Path(binding["path"])
        if sha256_file(path) != binding["sha256"]:
            raise ValueError(f"Pinned input differs: {path}")
        return path
    trajectory_path = pinned(plan["trajectory"])
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    if trajectory["record_mode"] != "prospective" or trajectory["decision"]["outcome"] != "ACTIVE":
        raise ValueError("Parent must publish active prospective before sample reuse")
    receipt_path = pinned(plan["plan_receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt["status"] != "PLANNED" or receipt["trajectory_id"] != trajectory["trajectory_id"]:
        raise ValueError("Matching RML plan receipt required")
    accepted = json.loads(pinned(plan["accepted_profile_manifest"]).read_text(encoding="utf-8"))
    rows = plan["sample_source_idx"]
    if len(rows) != 4096 or len(set(rows)) != 4096 or any(type(i) is not int or not 0 <= i < 500000 for i in rows):
        raise ValueError("Expected accepted fixed train4096 indices")
    if rows != accepted["sample_source_idx"] or any(plan["checkpoint"][key] != accepted["checkpoint"][key]
                                                   for key in ("sha256", "source_sha256")):
        raise ValueError("Accepted A100 sample/checkpoint identity differs")
    if plan["train_probe"]["sha256"] != accepted["files"]["train_probe.pt"]:
        raise ValueError("Accepted train_probe bytes differ")
    if plan["role"] != "official-train-derived-500k" or plan["geometry_used"] is not False:
        raise ValueError("Only accepted pure2D official-train-derived sample")
    sources = _allowlist(plan["source_files"])
    overrides = plan.get("source_overrides", {})
    if set(overrides) - set(OVERLAYS):
        raise ValueError("Only reviewed profiler/lightweight-pointer overlays permitted")
    for name, digest in accepted["files"].items():
        if name.startswith("src/") and name != "src/molgap/k1_execution_profile.py":
            if plan["source_files"].get(name) != digest:
                raise ValueError(f"Accepted frozen owner source differs: {name}")
    source_root = Path(plan["source_root"]).resolve()
    source_paths = {name: pinned(overrides[name]) if name in overrides else _source(source_root, name) for name in sources}
    for name, path in source_paths.items():
        if sha256_file(path) != plan["source_files"][name]:
            raise ValueError(f"Frozen source differs: {name}")
    selected, probe = pinned(plan["checkpoint"]), pinned(plan["train_probe"])
    packaging = plan["packaging_files"]
    if set(packaging) != set(PACKAGING_FILES):
        raise ValueError("Explicit bounded packaging inventory required")
    packaging_paths = {name: pinned(item) for name, item in packaging.items()}
    if output.exists():
        raise FileExistsError("Preparation requires a fresh output directory")
    output.mkdir(parents=True)
    for name, path in source_paths.items():
        destination = output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        publish_immutable_bytes(destination, path.read_bytes())
    for name, path in packaging_paths.items():
        publish_immutable_bytes(output / name, path.read_bytes())
    for name, path in (("prospective/trajectory.json", trajectory_path), ("prospective/plan_receipt.json", receipt_path),
                       ("selected.pt", selected), ("train_probe.pt", probe)):
        publish_immutable_bytes(output / name, path.read_bytes())
    manifest = {"format": "molgap-k1-native-t4-profile-payload-v1", "sample_source_idx": rows,
                "source_files": sources, "source_input_commit": plan["source_input_commit"],
                "accepted_profile_manifest_sha256": plan["accepted_profile_manifest"]["sha256"],
                "sample_preparation": "exact-byte reuse; no pickle decode",
                "parent_manifest_sha256": accepted["parent_manifest_sha256"],
                "parent_training_shards": accepted["parent_training_shards"],
                "checkpoint": {key: plan["checkpoint"][key] for key in ("sha256", "source_sha256")},
                "files": {p.relative_to(output).as_posix(): sha256_file(p) for p in output.rglob("*") if p.is_file()}}
    atomic_json(output / "payload_manifest.json", manifest)
    digest = sha256_file(output / "payload_manifest.json")
    verify_t4_payload(output, digest)
    return {"payload": str(output), "payload_manifest_sha256": digest}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--accepted-payload", type=Path)
    parser.add_argument("--trajectory", type=Path)
    parser.add_argument("--plan-receipt", type=Path)
    args = parser.parse_args()
    if args.plan:
        report = prepare(args.plan, args.output)
    else:
        if not all((args.accepted_payload, args.trajectory, args.plan_receipt)):
            parser.error("Plan generation requires --accepted-payload --trajectory --plan-receipt")
        from molgap.experiment_launch import publish_immutable_bytes
        plan = default_plan(args.accepted_payload, args.trajectory, args.plan_receipt)
        publish_immutable_bytes(args.output, json.dumps(plan, sort_keys=True, indent=2).encode("utf-8"))
        report = {"plan": str(args.output), "sample_decoded": False, "execution": False}
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
