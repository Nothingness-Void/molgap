"""Freeze an infrastructure retry of an unchanged scientific request.

The ppgt_capacity500k layout keeps one directory per retry (base/retry_vN) with
authority, experiment, platform, predecessor_disposition and predecessor_execution
JSON. Each is derived from the predecessor retry directory; only an explicit
allowlist of fields may differ, and everything else must stay byte-for-byte equal
after JSON parsing. Scope: the predecessor attempt was never submitted, so no
new debit or failure evidence exists and the disposition is carried forward.
A retry after a submitted failure needs a hand-authored disposition and is not
generated here.

Generating a retry grants no compute, platform or submission authority. This
module never submits, contacts a platform or reads credentials.
"""
import json
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from .common.declarations import run_version, safe_comparison_base
from .common.io import _safe_local, publish_immutable_bytes

FILES = ("authority", "experiment", "platform", "predecessor_disposition", "predecessor_execution")
_BUDGET = ("allocated_t4_hours_ceiling", "gpu.allocated_t4_hours_ceiling", "gpu.wall_ceiling_seconds",
           "kaggle_quota_hours_ceiling", "execution_repair.aggregate_wall_seconds",
           "execution_repair.prior_debit_wall_seconds", "execution_repair.prior_observed_wall_seconds",
           "execution_repair.repaired_current_invocation_wall_seconds")
# Dotted leaf paths that may differ from the predecessor, per file.
ALLOWED = {
    "authority": {"attempt_ids.0", "retry_of.decision_ref", "execution_repair.decision_ref",
                  "local_rejected_preparation_ref", *_BUDGET},
    "experiment": {"attempt_id", "authority_ref", "platform_ref", "retry_of.decision_ref", "timestamp",
                   "profile.estimated_device_hours", "profile.estimated_wall_hours"},
    "platform": set(),
    "predecessor_disposition": {"evidence_ref"},
    "predecessor_execution": set(),
}
_FALSE_FLAGS = ("extra_seed_authorized", "full_training_authorized", "automatic_successor_authorized",
                "protected_role_access_authorized")


def _flat(value, path=()):
    if isinstance(value, dict):
        items = [(path + (k,), v) for k, v in value.items()]
    elif isinstance(value, list):
        items = [(path + (str(i),), v) for i, v in enumerate(value)]
    else:
        return {".".join(path): value}
    return {k: v for key, item in items for k, v in _flat(item, key).items()}


def _set(tree, dotted, value):
    *parents, leaf = dotted.split(".")
    for part in parents:
        tree = tree[int(part)] if isinstance(tree, list) else tree[part]
    tree[int(leaf) if isinstance(tree, list) else leaf] = value


def _rewrite_refs(value, old, new):
    """Retarget refs into the predecessor directory only (other experiments' retries stay)."""
    if isinstance(value, str):
        return new + value[len(old):] if value.startswith(old) else value
    if isinstance(value, list):
        return [_rewrite_refs(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: _rewrite_refs(v, old, new) for k, v in value.items()}
    return value


def freeze_retry(repo_root, *, base, predecessor, attempt_id, retry_authorized, timestamp=None,
                 rejected_preparation_ref=None, validate=False):
    """Write base/retry_<attempt_id>/ from base/<predecessor>/ and return its Path.

    retry_authorized must be exactly True (the operator's decision, never inferred).
    rejected_preparation_ref optionally records a hand-authored local-rejection file
    that the operator writes into the new directory; it is only a pointer here.
    The existing resolver (which owns the retry_of checks) only accepts committed
    files, so validate defaults to False: commit the output, then call check_retry.
    With validate=True a failure leaves the directory in place for inspection.
    """
    if retry_authorized is not True:
        raise ValueError("retry_authorized must be exactly True, given by the operator")
    root = Path(repo_root).absolute()
    _safe_local(root)
    base = safe_comparison_base(base, base="experiments")
    new = run_version(attempt_id)
    if re.fullmatch(r"retry_v[1-9][0-9]*", predecessor or "") is None:
        raise ValueError("predecessor must be a retry directory such as retry_v2")
    old = predecessor.removeprefix("retry_")
    if int(new[1:]) <= int(old[1:]):
        raise ValueError("New attempt id must be greater than the predecessor's")
    source, target = root / base / predecessor, root / base / f"retry_{new}"
    _safe_local(source)
    _safe_local(target)
    if target.exists():
        raise ValueError("Retry directory already exists; preserve its bytes")
    before = {name: json.loads((source / f"{name}.json").read_bytes().decode("utf-8")) for name in FILES}
    authority, experiment, disposition = before["authority"], before["experiment"], before["predecessor_disposition"]
    if experiment.get("attempt_id") != old or authority.get("attempt_ids") != [old]:
        raise ValueError("Predecessor files do not declare the predecessor attempt id")
    if disposition.get("retry_authorized") is not True:
        raise ValueError("Predecessor disposition does not authorize a retry")
    if any(authority.get(flag) is not False for flag in _FALSE_FLAGS):
        raise ValueError("Predecessor authority must keep seeds, full training, successors and protected roles closed")
    identity = authority["retry_of"]["spec_identity"]
    if not (identity == experiment["retry_of"]["spec_identity"] == disposition["retry_of"]["spec_identity"]
            == before["predecessor_execution"]["spec_identity"]):
        raise ValueError("Predecessor spec identity differs between its files")

    aggregate, debit = disposition["aggregate_authorized_wall_seconds"], disposition["retry_budget_debit_wall_seconds"]
    remaining = aggregate - debit
    if remaining <= 0 or disposition["retry_remaining_wall_seconds"] != remaining:
        raise ValueError("Retry budget is exhausted or inconsistent with the recorded debit")
    devices = authority["gpu"]["allocated_devices"]
    wall_hours, device_hours = remaining / 3600, remaining * devices / 3600
    stamp = timestamp or datetime.now(timezone.utc).isoformat()
    decision = f"{base}/retry_{new}/predecessor_disposition.json"

    after = {name: _rewrite_refs(deepcopy(tree), f"{base}/retry_{old}/", f"{base}/retry_{new}/") for name, tree in before.items()}
    values = {"authority": {
        "attempt_ids.0": new, "allocated_t4_hours_ceiling": device_hours,
        "gpu.allocated_t4_hours_ceiling": device_hours, "gpu.wall_ceiling_seconds": remaining,
        "kaggle_quota_hours_ceiling": wall_hours, "execution_repair.aggregate_wall_seconds": aggregate,
        "execution_repair.prior_debit_wall_seconds": debit,
        "execution_repair.prior_observed_wall_seconds": disposition["observed_allocation_wall_seconds"],
        "execution_repair.repaired_current_invocation_wall_seconds": remaining},
        "experiment": {"attempt_id": new, "timestamp": stamp, "profile.estimated_device_hours": device_hours,
                       "profile.estimated_wall_hours": wall_hours}}
    for name, leaves in values.items():
        for dotted, value in leaves.items():
            _set(after[name], dotted, value)
    after["authority"].pop("local_rejected_preparation_ref", None)
    if rejected_preparation_ref is not None:
        after["authority"]["local_rejected_preparation_ref"] = rejected_preparation_ref
        after["authority"] = dict(sorted(after["authority"].items()))  # authority.json is key-sorted
    if after["authority"]["retry_of"]["decision_ref"] != decision != after["experiment"]["retry_of"]["decision_ref"]:
        raise ValueError("retry_of decision reference does not point at the new disposition")

    for name in FILES:
        old_flat, new_flat = _flat(before[name]), _flat(after[name])
        changed = {k for k in old_flat.keys() | new_flat.keys() if old_flat.get(k, KeyError) != new_flat.get(k, KeyError)}
        if changed - ALLOWED[name]:
            raise ValueError(f"{name}.json would change non-allowlisted fields: {sorted(changed - ALLOWED[name])}")
    for key in ("arms", "seeds", "question", "same_run", "logical_run_id", "experiment_id"):
        if after["experiment"].get(key) != experiment.get(key):
            raise ValueError("Scientific fields of the experiment would change: " + key)

    for name in FILES:
        publish_immutable_bytes(target / f"{name}.json", (json.dumps(after[name], indent=2) + "\n").encode("utf-8"))
    if validate:
        check_retry(root, target)
    return target


def check_retry(repo_root, target):
    """Run the existing retry_of validation of experiment_input on a generated retry.

    The resolver packages tracked sources, so the new files must be committed first.
    Nothing is deleted on failure.
    """
    from .experiment_input import resolve_experiment_input
    try:
        resolve_experiment_input(Path(target) / "experiment.json", repo_root)
    except Exception as error:
        raise ValueError(f"Generated retry failed existing retry_of validation; left in place: {error}") from error
