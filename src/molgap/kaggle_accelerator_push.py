from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

import requests
from requests.auth import HTTPBasicAuth


KAGGLE_KERNEL_PUSH_URL = "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel"
ALLOWED_ACCELERATORS = {"NvidiaTeslaT4", "CPU-only"}
UNSUPPORTED_BATCH_ACCELERATORS = {"TpuV5E8"}


def _observed_identity(result: dict, owner: str) -> dict:
    """Use platform response fields; the requested slug is never observed identity."""
    import re

    ref, url = result.get("ref") or None, result.get("url") or None
    if isinstance(ref, str) and re.fullmatch(r"/code/[\w-]+/[\w-]+/?", ref):
        ref = ref.removeprefix("/code/").rstrip("/")
    url_ref, script_version = None, None
    conflicts = []
    if url:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in {"www.kaggle.com", "kaggle.com"}:
            conflicts.append("unexpected_response_url")
        else:
            match = re.fullmatch(r"/code/([^/]+)/([^/]+)/?", parsed.path)
            if match:
                url_ref = "/".join(match.groups())
            script_version = parse_qs(parsed.query).get("scriptVersionId", [None])[0]
    for candidate in (ref, url_ref):
        if candidate and (not re.fullmatch(r"[\w-]+/[\w-]+", candidate) or candidate.split("/")[0] != owner):
            conflicts.append("response_owner_or_ref_conflict")
    if ref and url_ref and ref != url_ref:
        conflicts.append("response_ref_url_conflict")
    if script_version and not re.fullmatch(r"[1-9]\d*", script_version):
        script_version = None
        conflicts.append("invalid_script_version_id")
    version = result.get("versionNumber")
    if type(version) is not int or version <= 0:
        version = None
    return {"kernel": None if conflicts else ref or url_ref,
            "url": url, "version_number": version, "script_version_id": script_version,
            "kernel_id": result.get("kernelId") or None, "identity_conflicts": conflicts}


class ReleaseRecheckError(ValueError):
    """A local release check failed before the kernel POST could run."""

    @staticmethod
    def _failure(check, cause):
        failure = {"check": check, "error_type": type(cause).__name__}
        for name in ("errno", "winerror"):
            value = getattr(cause, name, None)
            if isinstance(value, int):
                failure[name] = value
        cleanup = getattr(cause, "cleanup_error", None)
        if isinstance(cleanup, OSError):
            failure["cleanup_error"] = {"error_type": type(cleanup).__name__, **{
                name: getattr(cleanup, name) for name in ("errno", "winerror")
                if isinstance(getattr(cleanup, name, None), int)}}
        return failure

    @classmethod
    def from_io(cls, cause, expected_report=None):
        failure = cls._failure("release_input_read", cause)
        error = cls("Release input checks failed during recheck: release_input_read ("
                    + type(cause).__name__ + ")")
        error.expected_report, error.observed_report = expected_report, None
        error.changed_checks, error.recheck_kind = [], "check_failed"
        error.release_recheck = {"category": "check_failed", "changed_checks": [], "changed_fields": [],
            "failures": [failure]}
        return error

    @classmethod
    def from_report(cls, expected_report, observed_report):
        report, observed = expected_report, observed_report
        if report is None:
            changed = sorted({item["check"] for item in observed["errors"]})
            changed_fields = []
        else:
            expected_checks = report.get("checks", {})
            observed_checks = observed.get("checks", {})
            changed = sorted(key for key in expected_checks.keys() | observed_checks.keys()
                             if key not in expected_checks or key not in observed_checks
                             or expected_checks[key] != observed_checks[key])
            known_fields = {"status", "spec_identity", "package_identity", "source_commit",
                            "source_archive_sha256", "inputs", "limitations", "errors"}
            changed_fields = sorted(key for key in known_fields
                if key not in report or key not in observed or report[key] != observed[key])
            if any(key not in report or key not in observed or report[key] != observed[key]
                   for key in (report.keys() | observed.keys()) - known_fields - {"checks"}):
                changed_fields.append("report_shape")
        errors = observed.get("errors", [])
        if errors:
            owners = sorted({item.get("check", "unknown") + " (" + item.get("error_type", "unknown") + ")"
                             for item in errors})
            error = cls("Release input checks failed during recheck: " + ", ".join(owners[:8]))
        else:
            error = cls("Release inputs changed after verification: " + ", ".join(changed[:8]))
        error.expected_report, error.observed_report = report, observed
        error.changed_checks = changed
        error.recheck_kind = "check_failed" if errors else "binding_mismatch"
        error.release_recheck = {
            "category": error.recheck_kind,
            "changed_checks": sorted({key.partition(":")[0] for key in changed}
                                     | {item.get("check", "unknown") for item in errors})[:8],
            "changed_fields": changed_fields,
            "failures": [{key: item[key] for key in ("check", "error_type", "errno", "winerror", "cleanup_error")
                          if key in item} for item in errors[:8]],
        }
        return error


def _verify_release_report(path: Path, code_path: Path, *, _content_verification=None,
                           _verification=None,
                           _entry_bytes=None, _metadata_bytes=None) -> dict:
    from .experiment_preflight import check_release_inputs
    from .experiment_spec import ExperimentSpec
    from .training_reproducibility import sha256_file

    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("format") == "molgap-frozen-inference-release-v1":
        from .frozen_inference_release import check_frozen_inference_release
        inputs = report["inputs"]
        observed = check_frozen_inference_release(Path(inputs["input_root"]),
            Path(inputs["entry_script"]), Path(inputs["metadata"]))
        if observed != report or Path(inputs["entry_script"]).resolve() != code_path.resolve():
            raise ValueError("Frozen inference release inputs changed")
        return {key: report["release"][key] for key in ("source_commit", "archive_sha256", "contract_sha256")}
    if report["status"] != "LOCAL_RELEASE_INPUTS_VERIFIED":
        raise ValueError("Release input checks did not pass")
    inputs = report["inputs"]
    package = Path(inputs["package"])
    spec = ExperimentSpec.from_json((package / "experiment_spec.json").read_text(encoding="utf-8"))
    try:
        observed = check_release_inputs(spec, package, expected_package_identity=report["package_identity"],
            recipe_files=inputs["recipe_files"], initial_states=inputs["initial_states"],
            required_modules=inputs["required_modules"], pickle_inputs=inputs["pickle_inputs"],
            entry_script=inputs["entry_script"], input_root=inputs["input_root"],
            launch_config=inputs.get("launch_config"), kernel_metadata=inputs.get("kernel_metadata"),
            dataset_manifests=inputs.get("dataset_manifests"), _content_verification=_content_verification,
            _verification=_verification)
    except OSError as cause:
        raise ReleaseRecheckError.from_io(cause, expected_report=report) from cause
    if observed != report or observed["errors"]:
        raise ReleaseRecheckError.from_report(report, observed)
    entry_sha = sha256_file(code_path) if _entry_bytes is None else hashlib.sha256(_entry_bytes).hexdigest()
    changed = []
    if report["checks"].get("entry_script:kernel") != entry_sha:
        if _entry_bytes is None:
            raise ValueError("Kernel entry script is not bound to the release check")
        changed.append("entry_script:kernel")
    metadata_sha = report["checks"].get("workflow:launch", {}).get("metadata_sha256")
    if (_metadata_bytes is not None and metadata_sha is not None
            and hashlib.sha256(_metadata_bytes).hexdigest() != metadata_sha):
        changed.append("workflow:launch")
    if changed:
        error = ReleaseRecheckError("Submission snapshot differs from the verified release")
        error.expected_report, error.observed_report = report, None
        error.changed_checks, error.recheck_kind = changed, "binding_mismatch"
        error.release_recheck = {"category": "binding_mismatch",
            "changed_checks": sorted({key.partition(":")[0] for key in changed}),
            "changed_fields": [], "failures": []}
        raise error
    return {k: report[k] for k in ("spec_identity", "package_identity", "source_commit", "source_archive_sha256")}


def push_kernel_with_accelerator(
    *,
    package_dir: Path,
    credential_path: Path,
    accelerator: str,
    timeout_seconds: int = 120,
    release_report_path: Path | None = None,
    api=None,
    _content_verification=None,
    _verification=None,
    _before_post=None,
) -> dict[str, Any]:
    if accelerator in UNSUPPORTED_BATCH_ACCELERATORS:
        raise ValueError(
            "Kaggle accepted TpuV5E8 metadata but executed both script and "
            "notebook batch probes on CPU; use an interactive allocation probe "
            "before enabling TPU workloads"
        )
    if accelerator not in ALLOWED_ACCELERATORS:
        raise ValueError(f"Unsupported accelerator: {accelerator}")
    package = package_dir.resolve()
    metadata_bytes = (package / "kernel-metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes.decode("utf-8"))
    if accelerator == "CPU-only" and str(metadata.get("enable_gpu", False)).lower() == "true":
        raise ValueError("CPU-only publication requires GPU-disabled metadata")
    code_path = package / metadata["code_file"]
    if not code_path.is_file():
        raise FileNotFoundError(code_path)
    if not code_path.resolve().is_relative_to(package):
        raise ValueError("Kernel entry script is outside the package")
    code_bytes = code_path.read_bytes()
    code_text = code_bytes.decode("utf-8")
    if "EXPECTED_LAUNCH_SHA256" in code_text and release_report_path is None:
        raise ValueError("Prepared workflow entry requires its verified release report")
    release_binding = (_verify_release_report(release_report_path, code_path,
        _content_verification=_content_verification, _verification=_verification,
        _entry_bytes=code_bytes, _metadata_bytes=metadata_bytes)
        if release_report_path else None)
    if release_report_path:
        report_inputs = json.loads(release_report_path.read_text(encoding="utf-8"))["inputs"]
        if report_inputs.get("metadata") and Path(report_inputs["metadata"]).resolve() != (package / "kernel-metadata.json").resolve():
            raise ValueError("Frozen inference submission metadata differs")
        if report_inputs.get("kernel_metadata") and Path(report_inputs["kernel_metadata"]).resolve() != (package / "kernel-metadata.json").resolve():
            raise ValueError("Submission metadata differs from the prepared workflow metadata")
        if report_inputs.get("launch_config"):
            launch = json.loads(Path(report_inputs["launch_config"]).read_text(encoding="utf-8"))
            if launch["accelerator"] != accelerator:
                raise ValueError("Submission accelerator differs from the prepared workflow")
    credentials = json.loads(credential_path.read_text(encoding="utf-8"))
    username = str(credentials.get("username", ""))
    key = str(credentials.get("key", ""))
    if not username or not key:
        raise RuntimeError("Kaggle credential file is incomplete")
    if not str(metadata["id"]).startswith(f"{username}/"):
        raise RuntimeError("Kernel owner does not match the credential owner")

    request = {
        "slug": metadata["id"],
        "newTitle": metadata["title"],
        "text": code_text,
        "language": metadata["language"],
        "kernelType": metadata["kernel_type"],
        "isPrivate": str(metadata.get("is_private", "true")).lower() == "true",
        "enableGpu": accelerator.startswith("Nvidia"),
        "enableTpu": accelerator.startswith("Tpu"),
        "enableInternet": str(metadata.get("enable_internet", "false")).lower()
        == "true",
        "datasetDataSources": list(metadata.get("dataset_sources", [])),
        "competitionDataSources": list(metadata.get("competition_sources", [])),
        "kernelDataSources": list(metadata.get("kernel_sources", [])),
        "modelDataSources": list(metadata.get("model_sources", [])),
        "categoryIds": list(metadata.get("keywords", [])),
    }
    if accelerator != "CPU-only":
        request["machineShape"] = accelerator
    if _before_post is not None:
        _before_post()
    try:
        if api is not None:
            from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
            if api.get_config_value(api.CONFIG_NAME_USER) != username:
                raise ValueError("SaveKernel SDK owner differs from credential owner")
            with api.build_kaggle_client() as client:
                result = client.kernels.kernels_api_client.save_kernel(ApiSaveKernelRequest.from_dict(request)).to_dict()
        else:
            response = requests.post(
                KAGGLE_KERNEL_PUSH_URL,
                auth=HTTPBasicAuth(username, key),
                json=request,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            result = response.json()
    except (requests.Timeout, requests.ConnectionError) as exc:
        # A lost response does not tell us whether the platform created a run.
        return {"status": "submission_unknown", "requested_kernel": metadata["id"],
                "kernel": None, "accelerator": accelerator, "release_binding": release_binding,
                "reconciliation_required": True, "error_type": type(exc).__name__,
                "next_action": "Reconcile authoritative account state before any retry"}
    if result.get("error"):
        raise RuntimeError(f"Kaggle kernel push failed: {result['error']}")
    identity = _observed_identity(result, username)
    invalid = {"invalid_dataset_sources": result.get("invalidDatasetSources", []),
               "invalid_kernel_sources": result.get("invalidKernelSources", []),
               "invalid_model_sources": result.get("invalidModelSources", []),
               "invalid_competition_sources": result.get("invalidCompetitionSources", [])}
    return {
        "status": "submitted",
        "requested_kernel": metadata["id"],
        **identity,
        "accelerator": accelerator,
        **invalid,
        "reconciliation_required": bool(identity["identity_conflicts"] or not identity["kernel"]
                                        or (identity["version_number"] is None and identity["script_version_id"] is None)
                                        or any(invalid.values())),
        "release_binding": release_binding,
        "platform_response": {k: result[k] for k in (
            "ref", "url", "versionNumber", "kernelId", "invalidDatasetSources", "invalidKernelSources",
            "invalidModelSources", "invalidCompetitionSources") if k in result},
    }
