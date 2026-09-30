from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, parse_qs

import requests
from requests.auth import HTTPBasicAuth


KAGGLE_KERNEL_PUSH_URL = "https://www.kaggle.com/api/v1/kernels/push"
ALLOWED_ACCELERATORS = {"NvidiaTeslaT4"}
UNSUPPORTED_BATCH_ACCELERATORS = {"TpuV5E8"}


def _observed_identity(result: dict, owner: str) -> dict:
    """Use platform response fields; the requested slug is never observed identity."""
    import re

    ref, url = result.get("ref") or None, result.get("url") or None
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


def _verify_release_report(path: Path, code_path: Path) -> dict:
    from .experiment_preflight import check_release_inputs
    from .experiment_spec import ExperimentSpec
    from .training_reproducibility import sha256_file

    report = json.loads(path.read_text(encoding="utf-8"))
    if report["status"] != "LOCAL_RELEASE_INPUTS_VERIFIED":
        raise ValueError("Release input checks did not pass")
    inputs = report["inputs"]
    package = Path(inputs["package"])
    spec = ExperimentSpec.from_json((package / "experiment_spec.json").read_text(encoding="utf-8"))
    observed = check_release_inputs(spec, package, expected_package_identity=report["package_identity"],
        recipe_files=inputs["recipe_files"], initial_states=inputs["initial_states"],
        required_modules=inputs["required_modules"], pickle_inputs=inputs["pickle_inputs"],
        entry_script=inputs["entry_script"], input_root=inputs["input_root"])
    if observed != report or observed["errors"]:
        raise ValueError("Release inputs changed after verification")
    if report["checks"].get("entry_script:kernel") != sha256_file(code_path):
        raise ValueError("Kernel entry script is not bound to the release check")
    return {k: report[k] for k in ("spec_identity", "package_identity", "source_commit", "source_archive_sha256")}


def push_kernel_with_accelerator(
    *,
    package_dir: Path,
    credential_path: Path,
    accelerator: str,
    timeout_seconds: int = 120,
    release_report_path: Path | None = None,
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
    metadata = json.loads((package / "kernel-metadata.json").read_text(encoding="utf-8"))
    code_path = package / metadata["code_file"]
    if not code_path.is_file():
        raise FileNotFoundError(code_path)
    if not code_path.resolve().is_relative_to(package):
        raise ValueError("Kernel entry script is outside the package")
    release_binding = _verify_release_report(release_report_path, code_path) if release_report_path else None
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
        "text": code_path.read_text(encoding="utf-8"),
        "language": metadata["language"],
        "kernelType": metadata["kernel_type"],
        "isPrivate": str(metadata.get("is_private", "true")).lower() == "true",
        "enableGpu": accelerator.startswith("Nvidia"),
        "enableTpu": accelerator.startswith("Tpu"),
        "enableInternet": str(metadata.get("enable_internet", "false")).lower()
        == "true",
        "machineShape": accelerator,
        "datasetDataSources": list(metadata.get("dataset_sources", [])),
        "competitionDataSources": list(metadata.get("competition_sources", [])),
        "kernelDataSources": list(metadata.get("kernel_sources", [])),
        "modelDataSources": list(metadata.get("model_sources", [])),
        "categoryIds": list(metadata.get("keywords", [])),
    }
    try:
        response = requests.post(
            KAGGLE_KERNEL_PUSH_URL,
            auth=HTTPBasicAuth(username, key),
            json=request,
            timeout=timeout_seconds,
        )
    except (requests.Timeout, requests.ConnectionError) as exc:
        # A lost response does not tell us whether the platform created a run.
        return {"status": "submission_unknown", "requested_kernel": metadata["id"],
                "kernel": None, "accelerator": accelerator, "release_binding": release_binding,
                "reconciliation_required": True, "error_type": type(exc).__name__,
                "next_action": "Reconcile authoritative account state before any retry"}
    response.raise_for_status()
    result = response.json()
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
