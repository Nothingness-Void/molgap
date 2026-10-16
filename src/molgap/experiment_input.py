"""Human-editable references to existing scientific declarations and workflow owners.

This resolver never synthesizes hypotheses, authorizations or accepted references.
It freezes normal JSON through existing canonical parsers; old --spec is intact.
"""
from pathlib import Path
from time import perf_counter
import json

from .common.io import _safe_local, publish_immutable_bytes
from .common.paths import MachinePaths, validate_relative_path
from .common.wire import canonical_fingerprint
from .common.spec import _object, _identifier, _unique_object
from .experiment_spec import ExperimentSpec
from .research_memory.paths import repo_local_path


def _read(path):
    path = Path(path).absolute()
    _safe_local(path)
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))


def resolve_experiment_input(path, repo_root):
    """Resolve references, not host defaults; no RML publication or network IO."""
    root = Path(repo_root).absolute()
    _safe_local(root)
    value = _read(path)
    _object(value,"format spec_ref workflow_ref machine_profile authority_ref monitor_paths attempt_id","experiment input")
    if value["format"] != "molgap-experiment-input-v1":
        raise ValueError("Unknown editable experiment input")
    _identifier(value["attempt_id"],"attempt_id")
    for name in ("spec_ref","workflow_ref","authority_ref"):
        validate_relative_path(value[name])
    spec = ExperimentSpec(_read(repo_local_path(root,value["spec_ref"])))
    plan = _read(repo_local_path(root,value["workflow_ref"]))
    if type(plan) is not dict or plan.get("spec_identity") != spec.identity:
        raise ValueError("Editable input workflow differs from its scientific declaration")
    if (type(value["monitor_paths"]) is not list or not value["monitor_paths"] or
            any(type(p) is not str for p in value["monitor_paths"]) or
            len(value["monitor_paths"]) != len(set(value["monitor_paths"]))):
        raise ValueError("Existing owner monitor pointers must be explicit and unique")
    for pointer in value["monitor_paths"]:
        validate_relative_path(pointer)
    paths = MachinePaths.from_file(value["machine_profile"]).outside_checkout(root)
    authority = repo_local_path(root,value["authority_ref"])
    _safe_local(authority)
    if not authority.is_file():
        raise ValueError("Existing authority reference is missing; no authorization is inferred")
    # New preparations use one tracked distribution boundary. Preserve explicit
    # scientific/authority records from the old declaration, never old per-model
    # Python lists. The old --prepared and --spec paths keep their old formats.
    if "source_files" in plan:
        if "source_profile" in plan or "source_records" in plan:
            raise ValueError("Ambiguous legacy and runtime source fields")
        if type(plan["source_files"]) is not list or any(type(p) is not str for p in plan["source_files"]):
            raise ValueError("Invalid legacy source list")
        records = [p for p in plan.pop("source_files") if not p.endswith(".py")]
        plan.update(source_profile="runtime-v1",source_records=records)
    if (plan.get("source_profile") != "runtime-v1" or type(plan.get("source_records")) is not list
            or any(type(p) is not str for p in plan["source_records"])):
        raise ValueError("New editable workflow requires the fixed runtime source profile")
    plan["source_records"] = sorted(set([*plan["source_records"],value["authority_ref"]]))
    return value,spec,plan,paths


def freeze_experiment_input(path, repo_root):
    """Idempotent local freeze; one run/attempt directory outside the worktree."""
    value,spec,plan,paths = resolve_experiment_input(path,repo_root)
    return _freeze(value,spec,plan,paths)


def _freeze(value,spec,plan,paths):
    from .experiment_launch import canonical_json
    run = paths.run_directory(spec.to_dict()["logical_run_id"],value["attempt_id"])
    binding = {"format":value["format"],"input_identity":canonical_fingerprint(value),
               "spec_identity":spec.identity,"workflow_identity":canonical_fingerprint(plan),
               "authority_ref":value["authority_ref"],"monitor_paths":value["monitor_paths"]}
    frozen = run / "inputs"
    # Binding first: a partial freeze can be retried with identical bytes only.
    publish_immutable_bytes(frozen/"input_binding.json",canonical_json(binding).encode())
    publish_immutable_bytes(frozen/"experiment_spec.json",spec.to_json().encode())
    publish_immutable_bytes(frozen/"workflow_plan.json",canonical_json(plan).encode())
    return {"status":"INPUTS_FROZEN","run_directory":str(run),"spec_identity":spec.identity,
            "submitted":False,"scientific_acceptance":"NOT_EVALUATED"}


def prepare_experiment_input(path, repo_root):
    """Use the original preparation owner once; never retry an incomplete stage."""
    from .experiment_package import verify_experiment_source_package
    from .experiment_workflow import prepare_workflow
    started=perf_counter()
    value,spec,plan,paths=resolve_experiment_input(path,repo_root)
    frozen=_freeze(value,spec,plan,paths)
    run=Path(frozen["run_directory"]);prepared=run/"prepared"
    if prepared.exists():
        _safe_local(prepared)
        report=_read(prepared/"workflow_report.json") if (prepared/"workflow_report.json").is_file() else {}
        if report.get("status") != "PREPARED_FOR_PLATFORM":
            return {"status":"RECONCILIATION_REQUIRED","prepared":str(prepared),"submitted":False,
                    "reason":"Existing incomplete preparation is retained; no automatic replacement"}
        manifest=verify_experiment_source_package(prepared/"package")
        if manifest["spec_identity"] != spec.identity or manifest["package_identity"] != report.get("package_identity"):
            raise ValueError("Existing prepared package/input binding changed")
        result={"status":"PREPARED_FOR_PLATFORM","reused_preparation":True}
    else:
        result=prepare_workflow(spec,Path(repo_root).absolute(),plan,prepared,
                                source_cache=paths.artifact_root/"source-packages/runtime-cache")
    return {**result,"prepared":str(prepared),"operations":str(run/"operations"),
            "authority_ref":value["authority_ref"],"monitor_paths":value["monitor_paths"],
            "local_call_seconds":perf_counter()-started,"submitted":False}
