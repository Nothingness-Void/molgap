"""Strict validation for the staged execution launch configuration.

The launch file is the small boundary between release preparation and a
platform runtime.  It contains no execution authority: it only binds the
already verified source package, per-arm inputs, and prospective records to
the exact jobs that a runtime may start.

Older server releases predate this validator and omit ``launch_config``
entirely.  They remain compatible at the outer release gate.  Once a launch
file is supplied, the entire ``molgap-execution-launch-v1`` schema is
required and validation is deliberately fail-closed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import tarfile
from typing import Any

from .experiment_package import verify_experiment_source_package
from .experiment_spec import ExperimentSpec


LAUNCH_FORMAT = "molgap-execution-launch-v1"
_LAUNCH_FIELDS = {
    "format", "spec_identity", "expected_package_identity",
    "expected_source_archive_sha256", "account", "run_reference",
    "dataset_sources", "accelerator", "device_count", "jobs",
    "prospective_sha256",
}
_OPTIONAL_FIELDS = {"resume"}
_RESUME_FIELDS = {"manifest", "sha256"}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTITY = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_./:-]{0,511}\Z")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest(value: Any, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return value


def _identity(value: Any, label: str) -> str:
    if type(value) is not str or _IDENTITY.fullmatch(value) is None:
        raise ValueError(f"{label} must be an opaque identity")
    if any(token in value for token in ("?", "@", "#")):
        raise ValueError(f"{label} must not contain credentials or query text")
    return value


def _relative(value: Any, label: str) -> str:
    if (type(value) is not str or not value or "\\" in value or ":" in value
            or value.startswith("/") or PurePosixPath(value).is_absolute()
            or PureWindowsPath(value).drive):
        raise ValueError(f"{label} must be a relative POSIX path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    if any(part.endswith((".", " ")) for part in parts):
        raise ValueError(f"{label} contains an unsafe path component")
    return value


def _load_launch(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Launch configuration contains duplicate JSON keys")
            result[key] = value
        return result

    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(
                               ValueError("Launch configuration contains nonfinite JSON")))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Launch configuration is not valid JSON") from exc
    if type(value) is not dict:
        raise ValueError("Launch configuration must be a JSON object")
    # The staged file is hashed into the entrypoint.  Canonical bytes also
    # make report replay deterministic and prevent equivalent alternate forms.
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True, allow_nan=False).encode("utf-8")
    if raw != canonical:
        raise ValueError("Launch configuration must use canonical JSON bytes")
    return value


def _verified_archive_member(package_dir: Path, relative: str, expected_sha256: str) -> None:
    """Check one source member against the package archive without extraction."""
    package_dir = Path(package_dir).absolute()
    archive_path = package_dir / "source.tar.gz"
    found = None
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive:
            if member.name == relative:
                if found is not None or not member.isfile() or member.linkname:
                    raise ValueError("Launch recipe archive member is unsafe or duplicated")
                found = member
            elif member.name.casefold() == relative.casefold():
                raise ValueError("Launch recipe path has a case-colliding archive member")
        if found is None:
            raise ValueError("Launch recipe is absent from the verified source archive")
        stream = archive.extractfile(found)
        if stream is None or _sha(stream.read()) != expected_sha256:
            raise ValueError("Launch recipe bytes differ from the Spec recipe identity")


def _relative_to(root: Path, path: Path, label: str) -> str:
    root = Path(root).absolute().resolve()
    path = Path(path).absolute()
    try:
        relative = path.resolve().relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} is outside the staged input root") from exc
    return _relative(relative.as_posix(), label)


def _validate_resume(config: dict, arm_ids: set[str], root: Path) -> dict[str, dict[str, str]] | None:
    if "resume" not in config:
        return None
    resume = config["resume"]
    if type(resume) is not dict:
        raise ValueError("Launch resume must be an arm-id mapping")
    if set(resume) != arm_ids:
        raise ValueError("Launch resume must bind every Spec arm exactly once")
    validated = {}
    for arm_id, declaration in resume.items():
        if type(arm_id) is not str or type(declaration) is not dict or set(declaration) != _RESUME_FIELDS:
            raise ValueError("Each launch resume entry requires manifest and sha256")
        path = _relative(declaration["manifest"], f"resume[{arm_id}].manifest")
        digest = _digest(declaration["sha256"], f"resume[{arm_id}].sha256")
        actual = root / path
        if not actual.is_file() or _file_sha(actual) != digest:
            raise ValueError(f"Resume manifest bytes differ for arm {arm_id}")
        validated[arm_id] = {"manifest": path, "sha256": digest}
    return validated


def validate_launch_config(
    spec: ExperimentSpec,
    launch_config: Path,
    *,
    manifest: dict | None = None,
    staged_root: Path | None = None,
    package_dir: Path | None = None,
    expected_package_identity: str | None = None,
    input_root: Path | None = None,
    recipe_files: dict[str, str] | None = None,
    initial_states: dict[str, str | Path] | None = None,
) -> dict:
    """Validate one immutable v1 launch against its actual release inputs.

    ``manifest`` must be the trusted result of
    ``verify_experiment_source_package`` and is the preferred package
    authority for callers that have already verified it.
    ``package_dir`` is accepted as a compatibility convenience and is verified
    here when ``manifest`` is not supplied.  The optional release
    maps let the local gate prove that the jobs refer to the exact recipe and
    staged initialization files that were checked into its report.  Runtime
    callers can omit those maps when the package and staged input root are the
    only available authorities; the package/spec/trajectory/device bindings
    remain mandatory.
    """
    if type(spec) is not ExperimentSpec:
        raise TypeError("Expected exactly ExperimentSpec")
    launch_path = Path(launch_config).absolute()
    if not launch_path.is_file():
        raise ValueError("Launch configuration is missing")
    config = _load_launch(launch_path)
    if config.get("format") != LAUNCH_FORMAT:
        raise ValueError("Launch configuration has an unsupported or missing format")
    unknown = set(config) - (_LAUNCH_FIELDS | _OPTIONAL_FIELDS)
    missing = _LAUNCH_FIELDS - set(config)
    if unknown or missing:
        raise ValueError("Launch v1 has missing or unknown fields")
    if any(type(config[field]) is not str for field in ("spec_identity", "expected_package_identity",
                                                        "expected_source_archive_sha256", "account",
                                                        "run_reference", "accelerator")):
        raise ValueError("Launch v1 identity fields have invalid types")
    if config["spec_identity"] != spec.identity:
        raise ValueError("Launch/Spec identity mismatch")
    package_identity = _digest(config["expected_package_identity"], "expected_package_identity")
    archive_sha256 = _digest(config["expected_source_archive_sha256"], "expected_source_archive_sha256")
    if expected_package_identity is not None and package_identity != expected_package_identity:
        raise ValueError("Launch package identity differs from the release binding")
    _identity(config["account"], "account")
    _identity(config["run_reference"], "run_reference")
    if re.fullmatch(r"[a-z0-9-]+", config["account"]) is None:
        raise ValueError("account must be a lowercase platform slug")
    if (config["run_reference"].split("/", 1)[0] != config["account"]
            or not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", config["run_reference"])):
        raise ValueError("Launch account/run reference mismatch")
    sources = config["dataset_sources"]
    if (type(sources) is not list or not sources or any(type(item) is not str or not item for item in sources)
            or len(set(sources)) != len(sources)):
        raise ValueError("Launch dataset_sources must be a nonempty unique list")
    if config["accelerator"] != "NvidiaTeslaT4":
        raise ValueError("Launch v1 requires the reviewed T4 allocation")
    declaration = spec.to_dict()
    if type(config["device_count"]) is not int or config["device_count"] != declaration["platform"]["device_count"]:
        raise ValueError("Launch device count differs from the Spec")
    if config["device_count"] <= 0:
        raise ValueError("Launch device count must be positive")

    # Import lazily to keep this small boundary free of module-import cycles.
    from .experiment_execution import validate_execution_plan, validate_staged_trajectory

    jobs = config["jobs"]
    validated_jobs = validate_execution_plan(spec, jobs)
    arms = {arm["arm_id"]: arm for arm in declaration["arms"]}
    arm_ids = set(arms)
    prospective = config["prospective_sha256"]
    if type(prospective) is not dict or set(prospective) != arm_ids:
        raise ValueError("Launch requires one prospective digest per Spec arm")
    for arm_id, digest in prospective.items():
        _digest(digest, f"prospective_sha256[{arm_id}]")

    if (staged_root is not None and input_root is not None
            and Path(staged_root).absolute() != Path(input_root).absolute()):
        raise ValueError("Conflicting staged input roots")
    if staged_root is not None:
        input_root = staged_root
    # ``manifest`` is a trusted result of verify_experiment_source_package
    # when supplied by a release gate.  Do not re-read the complete package in
    # that path; package_dir is still used below for the selected recipe bytes.
    # Direct callers that provide only package_dir retain full verification.
    if manifest is not None:
        if type(manifest) is not dict or type(manifest.get("package_identity")) is not str:
            raise ValueError("Launch requires a verified package manifest")
        if (type(manifest.get("source_commit")) is not str
                or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", manifest["source_commit"]) is None):
            raise ValueError("Launch requires a verified source commit in the package manifest")
        if manifest["package_identity"] != package_identity:
            raise ValueError("Launch package identity differs from the verified package manifest")
        if manifest.get("archive_sha256") != archive_sha256:
            raise ValueError("Launch source archive hash differs from the verified package manifest")
        if manifest.get("spec_identity") != spec.identity:
            raise ValueError("Verified package Spec identity differs from the launch")
    elif package_dir is not None:
        package = Path(package_dir).absolute()
        manifest = verify_experiment_source_package(package)
        if manifest["package_identity"] != package_identity:
            raise ValueError("Launch package identity differs from the verified package manifest")
        if manifest["archive_sha256"] != archive_sha256:
            raise ValueError("Launch source archive hash differs from the verified package manifest")
        if manifest["spec_identity"] != spec.identity:
            raise ValueError("Verified package Spec identity differs from the launch")
    else:
        raise ValueError("Launch requires a verified package manifest or package directory")

    root = Path(input_root or launch_path.parent).absolute()
    if not root.is_dir():
        raise ValueError("Launch staged input root is missing")
    if launch_path.parent.resolve() != root.resolve():
        raise ValueError("Launch configuration must live at the staged input root")

    # Job paths are checked against the release's maps when available and
    # always against the frozen source archive/staged bytes.
    expected_recipes = recipe_files or {}
    expected_initials = initial_states or {}
    for job in validated_jobs:
        arm_id = job["arm_id"]
        recipe = _relative(job["recipe"], f"jobs[{arm_id}].recipe")
        if package_dir is not None:
            _verified_archive_member(package_dir, recipe,
                                     arms[arm_id]["training"]["recipe"]["sha256"])
        if expected_recipes:
            if set(expected_recipes) != arm_ids or expected_recipes[arm_id] != recipe:
                raise ValueError(f"Launch recipe path is not the checked release input for {arm_id}")
        initial = _relative(job["initial_state"], f"jobs[{arm_id}].initial_state")
        initial_path = root / initial
        if not initial_path.is_file():
            raise ValueError(f"Launch initialization input is missing for {arm_id}")
        if expected_initials:
            if set(expected_initials) != arm_ids:
                raise ValueError("Launch initialization bindings do not cover every arm")
            expected_value = Path(expected_initials[arm_id])
            if not expected_value.is_absolute():
                expected_value = root / expected_value
            expected_path = _relative_to(root, expected_value,
                                         f"initial_states[{arm_id}]")
            if expected_path != initial:
                raise ValueError(f"Launch initialization path is not the checked release input for {arm_id}")
        # The owning state inspector is already used by check_release_inputs;
        # invoking it here also makes a direct runtime gate self-contained.
        if arms[arm_id]["initialization"]["state_sha256"] is not None:
            from .v4_runtime import inspect_frozen_state_artifact
            inspect_frozen_state_artifact(initial_path,
                expected_state_sha256=arms[arm_id]["initialization"]["state_sha256"])

        trajectory = root / "prospective" / arm_id / "trajectory.json"
        if _file_sha(trajectory) != prospective[arm_id]:
            raise ValueError(f"Published prospective record changed for {arm_id}")
        observed = validate_staged_trajectory(spec, arm_id, root,
                                              expected_sha256=prospective[arm_id])
        if manifest is not None and observed["state_at_start"].get("source_commit") != manifest["source_commit"]:
            raise ValueError(f"Prospective source identity differs for {arm_id}")

    resume = _validate_resume(config, arm_ids, root)
    if resume is not None:
        if manifest is None:
            raise ValueError("Resume launch requires a verified source manifest")
        from .experiment_family_workflow import RunContext
        from .experiment_resume import validate_resume_bundle
        from .screen_policy import canonical_fingerprint
        for arm_id, pointer in resume.items():
            arm = arms[arm_id]
            context = RunContext(declaration["experiment_id"], declaration["logical_run_id"],
                arm_id, canonical_fingerprint(arm), spec.identity, arm["family"]["name"],
                arm["family"]["version"], manifest["source_commit"], archive_sha256,
                package_identity, arm["training"]["recipe"]["sha256"],
                declaration["platform"]["name"], config["account"], config["run_reference"], None)
            validate_resume_bundle(root / pointer["manifest"], spec, arm_id=arm_id, context=context,
                                   trajectory=root / "prospective" / arm_id / "trajectory.json")
    result = {
        "format": LAUNCH_FORMAT,
        "launch_sha256": _file_sha(launch_path),
        "spec_identity": spec.identity,
        "expected_package_identity": package_identity,
        "expected_source_archive_sha256": archive_sha256,
        "jobs": validated_jobs,
        "prospective_sha256": prospective,
    }
    if resume is not None:
        result["resume"] = resume
    if manifest is not None:
        result["source_commit"] = manifest["source_commit"]
    return result


__all__ = ["LAUNCH_FORMAT", "validate_launch_config"]
