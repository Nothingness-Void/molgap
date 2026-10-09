"""Validate or stage local prospective inputs. Never package, publish, or train."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import subprocess

from molgap.constants import REPO_ROOT as ROOT
from molgap.evidence_pointers import load_json_object, resolve_repo_pointer, verify_bound_artifact
from molgap.experiment_spec import _canonical
from molgap.research_memory.trace import atomic_write, file_digest
from molgap.screen_policy import canonical_fingerprint
from molgap.v4_runtime import inspect_frozen_state_artifact

REL = "experiments/pcqm_k1_t4_cost_quality"
RUN = "molgap-k1-t4-cost-quality-100k-s42-v1"
POLICY = "pcqm-k1-t4-cost-quality-100k-preparation"
STATE = Path("D:/w/k1-dropout-consistency/experiments/pcqm_k1_dropout_consistency/initial_state.pt")
STATE_SHA = "8ef6d4ba1abdca04d8740a11ec0e04587358117b1e9f9534b4fc19d2b6caedbd"
PRIOR = "pcqm-k1-consistency-fusion-transfer-20261004"
RELEASE_FORMAT = "molgap-k1-t4-parent-release-v1"
CLEAN = "experiments/pcqm_k1_clean_fit_generalization_500k"


def _bound(root: Path, binding: dict) -> Path:
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256"}
            or any(not isinstance(binding[key], str) or not binding[key] for key in ("path", "sha256"))):
        raise ValueError("Evidence requires a repository-local path and SHA pin")
    verify_bound_artifact(root, binding["path"], binding["sha256"])
    return resolve_repo_pointer(root, binding["path"])


def _validate_profile_release(root: Path, acceptance_path: Path) -> dict:
    """Read-only acceptance owner, never profiler completion or a copied validator."""
    path = root / REL / "profile/close.py"
    if not path.is_file():
        raise ValueError("Parent-coordinated profile.close acceptance callable is unavailable")
    module_spec = importlib.util.spec_from_file_location("k1_native_profile_close", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    validator = getattr(module, "accept", None)
    if not callable(validator):
        raise ValueError("profile.close.accept is unavailable")
    if acceptance_path != (root / REL / "profile/acceptance.json").resolve():
        raise ValueError("Profile pin must name the owning accepted record, not raw completion")
    report = validator(root)
    receipt = load_json_object(acceptance_path)
    if (not isinstance(report, dict) or not isinstance(report.get("analysis"), dict)
            or not isinstance(receipt.get("analysis"), dict)
            or not isinstance(receipt.get("outcome"), dict)
            or "single_step_saving_fraction" not in report["analysis"]):
        raise ValueError("Profile acceptance requires structured owner analysis/outcome, not booleans")
    if (receipt.get("analysis") != report["analysis"]
            or receipt.get("outcome", {}).get("artifact_status") != "local_hash_verified"
            or receipt.get("outcome", {}).get("scientific_status") != "NO_TRAIN"):
        raise ValueError("Pinned profile acceptance differs from actual validated artifacts")
    return {"status": "ACCEPTED",
            "native_step_reduction": report["analysis"]["single_step_saving_fraction"]}


def validate_parent_release(root: Path, release_file: Path, *, source_commit: str) -> dict:
    """Validate parent judgement and custody, not a new clean-fit scientific gate."""
    root = Path(root).resolve()
    path = Path(release_file)
    path = path if path.is_absolute() else root / path
    try:
        ref = path.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError("Parent release must be repository-local") from exc
    if not ref.startswith(REL + "/") or "/profile/" in ref:
        raise ValueError("Parent release must belong to the owning question outside profile")
    path = resolve_repo_pointer(root, ref)
    release = load_json_object(path)
    if release.get("format") != RELEASE_FORMAT:
        raise ValueError("Expected explicit parent release record, not raw completion")
    if (release.get("controller") != "human-controller"
            or not isinstance(release.get("approved_by"), str) or not release["approved_by"].strip()
            or not isinstance(release.get("approved_at"), str) or not release["approved_at"].strip()):
        raise ValueError("Explicit human-controller approval identity/date required")
    if (release.get("action") != "TRAIN_PAIR_100K"
            or release.get("allowed_actions") != ["TRAIN_PAIR_100K"]):
        raise ValueError("Parent release permits TRAIN_PAIR_100K only")
    if release.get("source_commit") != source_commit:
        raise ValueError("Parent release source commit mismatch")
    budget = release.get("budget")
    if (not isinstance(budget, dict) or set(budget) != {"allocated_t4_device_hours", "wall_seconds"}
            or type(budget.get("allocated_t4_device_hours")) not in (int, float)
            or type(budget.get("wall_seconds")) not in (int, float)
            or budget != {"allocated_t4_device_hours": 8, "wall_seconds": 14400}):
        raise ValueError("Parent release requires exactly 8 allocated T4 device-hours/14400 wall seconds")
    profile_path = _bound(root, release.get("profile_acceptance"))
    clean = release.get("clean_fit")
    if not isinstance(clean, dict):
        raise ValueError("Missing clean-fit pins and manual assessment")
    paths = {key: _bound(root, clean.get(key)) for key in ("terminal", "acceptance", "decision", "finalization")}
    for key, name in (("terminal", "rml/rml_finalized/terminal_input.json"), ("acceptance", "acceptance.json"),
                      ("decision", "terminal_decision.md"), ("finalization", "closure_receipt.json")):
        if paths[key] != (root / CLEAN / name).resolve():
            raise ValueError("Clean-fit pins must name the integrated canonical records")
    terminal = load_json_object(paths["terminal"])
    receipt = load_json_object(paths["finalization"])
    acceptance = load_json_object(paths["acceptance"])
    if (not isinstance(terminal.get("decision"), dict)
            or not isinstance(terminal.get("artifact_hashes"), dict)
            or not isinstance(receipt.get("input_artifact_hashes"), dict)
            or not isinstance(receipt.get("published_hashes"), dict)
            or not isinstance(acceptance.get("outcome"), dict)):
        raise ValueError("Clean-fit requires structured finalized records, not booleans")
    if (receipt.get("format") != "molgap-rml-finalization-v1"
            or not receipt.get("finalization_id") or not receipt.get("finalized_at")
            or receipt.get("outcome") != "NO_TRAIN"
            or terminal.get("format") != "molgap-rml-terminal-package-v1"
            or receipt.get("trajectory_id") != terminal.get("trajectory_id")
            or terminal.get("decision", {}).get("outcome") != "NO_TRAIN"
            or terminal.get("decision", {}).get("decision_ref") != clean["decision"]["path"]
            or terminal.get("acceptance_ref") != clean["acceptance"]["path"]
            or acceptance.get("format") != "molgap-local-frozen-diagnostic-acceptance-v1"
            or acceptance.get("outcome", {}).get("execution_status") != "complete_no_training"
            or acceptance.get("outcome", {}).get("artifact_status") != "local_hash_verified"
            or acceptance.get("outcome", {}).get("scientific_status") != "NO_TRAIN"):
        raise ValueError("Clean-fit requires an actual finalized NO_TRAIN accepted receipt")
    for key in ("acceptance", "decision"):
        pin = clean[key]
        if (receipt.get("input_artifact_hashes", {}).get(pin["path"]) != pin["sha256"]
                or terminal.get("artifact_hashes", {}).get(pin["path"]) != pin["sha256"]):
            raise ValueError("Clean-fit finalization/terminal does not pin accepted decision")
    if receipt.get("published_hashes", {}).get("terminal_input.json") != clean["terminal"]["sha256"]:
        raise ValueError("Clean-fit finalization does not bind terminal input")
    if (clean.get("assessment") != "NO_URGENT_FITTING_FAILURE"
            or not isinstance(clean.get("rationale"), str) or not clean["rationale"].strip()):
        raise ValueError("Manual assessment explaining no urgent fitting failure required")
    report = _validate_profile_release(root, profile_path)
    if not isinstance(report, dict) or report.get("status") != "ACCEPTED":
        raise ValueError("Native profile acceptance owner did not validate artifacts")
    reduction = report.get("native_step_reduction")
    if type(reduction) not in (int, float) or not math.isfinite(reduction) or not .25 <= reduction <= 1:
        raise ValueError("Accepted native profile reduction must be >=0.25")
    binding = {"path": ref, "sha256": file_digest(path)}
    verify_bound_artifact(root, ref, binding["sha256"])
    return {"binding": binding, "record": release, "profile_report": report}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_inputs(root: Path, *, source_commit: str, initial_state: Path,
                 state_timestamp: str = "2026-10-09", parent_release_file: Path | None = None,
                 **kwargs) -> dict:
    """Compatibility wrapper; only this old owner validates its legacy release."""
    from molgap.k1_pair_preparation import build_inputs as build_pair_inputs
    root = Path(root)
    release = None
    if parent_release_file is not None:
        # New questions bind their own explicit release after draft preparation.
        import inspect
        defaults = inspect.signature(build_pair_inputs).parameters
        if any(key not in defaults or value != defaults[key].default
               for key, value in kwargs.items()):
            raise ValueError("Fresh records remain drafts; parent must bind the new explicit release after building")
        for arm_id in ("mean2", "single"):
            if ((root / REL / f"training_plan_{arm_id}.json").exists()
                    or (root / REL / "kaggle3_v1" / arm_id).exists()):
                raise ValueError("Release plans require an unpublished fresh build")
        release = validate_parent_release(root, parent_release_file, source_commit=source_commit)
    return build_pair_inputs(root, source_commit=source_commit, initial_state=initial_state,
        state_timestamp=state_timestamp, approved_release=release,
        _state_inspector=inspect_frozen_state_artifact, **kwargs)


def stage_inputs(records: dict, output: Path) -> None:
    """Fresh draft directory only; no canonical trajectory publication or RML writes."""
    if output.exists():
        raise FileExistsError("Reconcile existing drafts; output must be fresh")
    output.mkdir(parents=True)
    for name, record in records.items():
        atomic_write(output / name, _canonical(record).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-local", type=Path, help="Fresh draft output; absent means validation only")
    parser.add_argument("--initial-state", type=Path, default=STATE)
    parser.add_argument("--parent-release-file", type=Path, help="Repository-local explicit human-controller release")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    records = build_inputs(ROOT, source_commit=commit, initial_state=args.initial_state,
                           parent_release_file=args.parent_release_file)
    if args.stage_local:
        output = args.stage_local.resolve()
        if not output.is_relative_to((ROOT / REL).resolve()) or "profile" in output.relative_to(ROOT / REL).parts:
            raise ValueError("Draft output must remain in the owning question, outside profile")
        stage_inputs(records, output)
    print(json.dumps(records["preparation_report.json"], sort_keys=True))


if __name__ == "__main__":
    main()
