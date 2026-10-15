"""Idempotent platform operations for an already prepared, authorized release.

Training and local preparation remain with their existing owners. Each mutating
request is journaled before dispatch; uncertain outcomes are reconciled by read
operations and never retried automatically.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from urllib.parse import quote

from .experiment_family_workflow import _json
from .experiment_launch import (_safe_local, build_launch_receipt, canonical_json,
                                reconcile_platform_response)
from .experiment_package import read_packaged_text, verify_experiment_source_package
from .experiment_spec import ExperimentSpec
from .kaggle_accelerator_push import _verify_release_report, push_kernel_with_accelerator
from .research_memory.trace import atomic_write
from .training_reproducibility import atomic_json, sha256_file


def _credentials(path):
    path = Path(path).absolute()
    _safe_local(path)
    if any((parent / ".git").exists() for parent in path.parents):
        raise ValueError("Keep Kaggle credentials outside Git checkouts")
    value = _json(path)
    if not all(type(value.get(key)) is str and value[key] for key in ("username", "key")):
        raise ValueError("Kaggle credential file is incomplete")
    return value


def connect_kaggle(credentials):
    """Authenticate the installed SDK without exposing or persisting the key."""
    from .kaggle_credentials import api_for_credentials
    _credentials(credentials)
    return api_for_credentials(credentials)


def _not_found(error):
    return (getattr(error, "status", None) == 404 or getattr(error, "status_code", None) == 404
            or getattr(getattr(error, "response", None), "status_code", None) == 404)


def _dataset_status(api, ref):
    try:
        value = api.dataset_status(ref)
    except Exception as error:
        if _not_found(error):
            return None
        raise RuntimeError("Dataset status is unknown; no publication retry") from None
    if type(value) is not str or not value:
        raise ValueError("Malformed dataset status; no publication retry")
    return {"status": value.lower()}


def _dataset_details(api, ref):
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest
    request = ApiGetDatasetRequest()
    request.owner_slug, request.dataset_slug = ref.split("/")
    with api.build_kaggle_client() as client:
        details = client.datasets.dataset_api_client.get_dataset(request)
    if details.ref != ref or details.is_private is not True:
        raise ValueError("Remote source dataset identity/private binding differs")
    return details


def _kernel_source(api, ref):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = ref.split("/")
    try:
        with api.build_kaggle_client() as client:
            return client.kernels.kernels_api_client.get_kernel(request)
    except Exception as error:
        if _not_found(error):
            return None
        raise RuntimeError("Kernel identity is unknown; no submission retry") from None


def _quota(http, budget):
    response = http.get("https://www.kaggle.com/api/v1/kernels/quota", timeout=(10, 30))
    response.raise_for_status()
    raw = response.json()
    quota = raw["gpuQuota"]
    def seconds(item):
        value = item["seconds"] + item.get("nanos", 0) / 1e9
        if not math.isfinite(value) or value < 0:
            raise ValueError("Malformed Kaggle quota")
        return value
    remaining = seconds(quota["totalTimeAllowed"]) - seconds(quota["timeUsed"]) - seconds(quota["timeReserved"])
    if remaining < budget * 3600:
        raise ValueError("Available Kaggle GPU quota is below the frozen allocation ceiling")
    return {"observed_at": datetime.now(timezone.utc).isoformat(), "quota": raw,
            "available_gpu_hours": remaining / 3600, "authorized_ceiling_gpu_hours": budget}


@contextmanager
def _exclusive(output):
    lock = output / ".submission.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError("Submission lock exists; reconcile the owning process before continuing") from None
    try:
        os.close(descriptor)
        yield
    finally:
        lock.unlink()


def _roundtrip_source(api, http, staged, remote, ref):
    """Reuse the bounded hash-checking transport; never download graph datasets."""
    from .kaggle_output_retrieval import _download_selected
    expected = {}
    for path in staged.rglob("*"):
        _safe_local(path)
        if path.is_file() and path.name != "dataset-metadata.json":
            expected[path.relative_to(staged).as_posix()] = path
    status = _dataset_status(api, ref)
    details = _dataset_details(api, ref)
    if (not isinstance(status, dict) or status.get("status") != "ready"
            or type(details.current_version_number) is not int or details.current_version_number != 1):
        raise ValueError("Source dataset must be ready at immutable version1")
    listing = api.dataset_list_files(ref, page_size=100)
    files = listing.files
    if (getattr(listing, "next_page_token", None) or len(files) != len(expected)
            or {f.name for f in files} != set(expected)):
        raise ValueError("Remote source layout differs from the prepared allowlist")
    records = []
    for item in files:
        path = expected[item.name]
        if item.total_bytes != path.stat().st_size:
            raise ValueError("Remote source file size differs")
        records.append({"path": item.name, "sha256": sha256_file(path), "bytes": item.total_bytes})
    # Resolve a stable directory tree before parallel readers start. Windows
    # realpath can expose an extended-path alias if an ancestor appears midway
    # through resolving a missing file; the transport correctly rejects it.
    _safe_local(remote)
    remote.mkdir(parents=True, exist_ok=True)
    for item in records:
        destination = remote / item["path"]
        _safe_local(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
    def download(item):
        name = item["path"]
        url = "https://www.kaggle.com/api/v1/datasets/download/" + ref + "/" + quote(name, safe="") + "?datasetVersionNumber=1"
        _download_selected(urls={name: url}, required={name: (name, item["sha256"])},
                           destination=remote, http_session=http)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, records))
    return {"status": "REMOTE_SOURCE_BYTES_VERIFIED", "dataset": ref, "version": 1, "files": records}


def _verify_remote_kernel(remote, response, metadata, entry):
    if remote is None:
        raise ValueError("Submitted kernel is not yet observable")
    m = remote.metadata
    payload = remote.blob.source.encode("utf-8")
    if (response.get("kernel") != metadata["id"]
            or type(response.get("kernel_id")) is not int or response["kernel_id"] <= 0
            or type(response.get("version_number")) is not int or response["version_number"] <= 0
            or m.id != response["kernel_id"] or m.current_version_number != response["version_number"]
            or not m.is_private or not m.enable_gpu or m.machine_shape != "NvidiaTeslaT4"
            or sorted(m.dataset_data_sources) != sorted(metadata["dataset_sources"]) or payload != entry):
        raise ValueError("Remote physical version/source/private/T4/mount binding differs")
    return {"kernel": response["kernel"], "kernel_id": m.id, "version": m.current_version_number,
            "entry_sha256": hashlib.sha256(payload).hexdigest(), "entry_bytes_equal": True,
            "private": m.is_private, "machine_shape": m.machine_shape,
            "dataset_sources": list(m.dataset_data_sources), "observed_at": datetime.now(timezone.utc).isoformat()}


def _startup(http, ref):
    """Retain a bounded startup sample; transport uncertainty is not job failure."""
    events, began = [], perf_counter()
    result = {"kernel": ref, "scope": "startup observation; not native qualification"}
    try:
        with http.get("https://www.kaggle.com/api/v1/kernels/logs/stream/" + ref,
                      headers={"Accept": "text/event-stream, */*"}, stream=True, timeout=(5, 10)) as response:
            response.raise_for_status()
            for line in response.iter_lines(chunk_size=512, decode_unicode=True):
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if line and line.startswith("data:"):
                    payload = line[5:].strip()
                    if payload == "END_OF_LOG":
                        break
                    try:
                        events.append(json.loads(payload))
                    except ValueError:
                        events.append({"data": payload})
                if len(events) >= 150 or perf_counter() - began >= 15:
                    break
        result["transport"] = "bounded_observation"
    except Exception as error:
        result["transport"] = type(error).__name__
    return {**result, "events": events}


def submit_prepared_workflow(prepared, credentials, output, *, authority_ref,
                             monitor_paths, execute=False, api=None, http_session=None):
    """Dry-check or advance one journal; reruns reconcile rather than duplicate."""
    prepared, output = Path(prepared).absolute(), Path(output).absolute()
    _safe_local(prepared)
    _safe_local(output)
    values = _credentials(credentials)
    package, kernel, staged = prepared / "package", prepared / "kernel", prepared / "source_dataset"
    for path in (kernel / "kernel-metadata.json", kernel / "run.py", staged / "dataset-metadata.json",
                 staged / "experiment_launch.json", prepared / "workflow_report.json"):
        _safe_local(path)
    manifest = verify_experiment_source_package(package)
    spec = ExperimentSpec.from_json((package / "experiment_spec.json").read_text(encoding="utf-8"))
    report = _json(prepared / "workflow_report.json")
    if report.get("status") != "PREPARED_FOR_PLATFORM" or report.get("package_identity") != manifest["package_identity"]:
        raise ValueError("Use the completed prepared workflow and its exact package")
    metadata = _json(kernel / "kernel-metadata.json")
    source_metadata = _json(staged / "dataset-metadata.json")
    account, ref, dataset = values["username"], metadata["id"], source_metadata["id"]
    launch = _json(staged / "experiment_launch.json")
    if (ref.split("/")[0] != account or dataset.split("/")[0] != account
            or launch["account"] != account or launch["run_reference"] != ref
            or launch["accelerator"] != "NvidiaTeslaT4" or dataset not in metadata["dataset_sources"]
            or metadata.get("is_private") is not True):
        raise ValueError("Account/private/T4/source mount differs from prepared inputs")
    from .experiment_package import SIDECARS
    allowed = {"source_payload.bin" if name == "source.tar.gz" else name for name in SIDECARS}
    allowed.update({"dataset-metadata.json", "experiment_launch.json"})
    for arm in spec.to_dict()["arms"]:
        allowed.update({"initial_states/" + arm["arm_id"] + ".pt",
                        "prospective/" + arm["arm_id"] + "/trajectory.json"})
    if "target_identity" in launch:
        from .kaggle_workflow import acceptance_input_pins
        from .experiment_family_workflow import _artifact_path
        binding = launch["target_identity"]
        acceptance_root = staged / "acceptance"
        plan_path = _artifact_path(acceptance_root, binding["plan_path"])
        if sha256_file(plan_path) != binding["plan_sha256"]:
            raise ValueError("Staged acceptance plan differs from frozen launch")
        allowed.update("acceptance/" + path for path in
                       acceptance_input_pins(spec, acceptance_root, binding["plan_path"]))
    observed = set()
    for path in staged.rglob("*"):
        _safe_local(path)
        if path.is_file():
            observed.add(path.relative_to(staged).as_posix())
    if observed != allowed:
        raise ValueError("Source upload contains missing or unbound files")
    if output.is_relative_to(prepared) or prepared.is_relative_to(output):
        raise ValueError("Keep operation records separate from immutable prepared inputs")
    authority = json.loads(read_packaged_text(package, authority_ref))
    budget = authority.get("allocated_t4_hours_ceiling")
    if (authority.get("user_authorized") is not True or type(budget) not in {int, float}
            or not math.isfinite(budget) or budget <= 0):
        raise ValueError("Packaged authority must authorize a positive allocated T4-hour ceiling")
    arms = spec.to_dict()["arms"]
    if (authority.get("arms") != [arm["arm_id"] for arm in arms]
            or authority.get("seeds") != sorted({arm["initialization"]["seed"] for arm in arms})):
        raise ValueError("Packaged authority does not cover exactly these arms and seeds")
    from .experiment_terminal import _path
    if not monitor_paths or len(monitor_paths) != len(set(monitor_paths)):
        raise ValueError("Expected explicit unique monitor pointers")
    for pointer in monitor_paths:
        _path(pointer)
    binding = {"package_identity": manifest["package_identity"], "kernel": ref, "dataset": dataset,
               "authority_ref": authority_ref, "authority_sha256": hashlib.sha256(canonical_json(authority).encode()).hexdigest(),
               "monitor_paths": list(monitor_paths)}
    if not execute:
        _verify_release_report(prepared / "release_report.json", kernel / "run.py")
        return {"status": "READY_FOR_PLATFORM_OPERATIONS", "binding": binding, "submitted": False}
    output.mkdir(parents=True, exist_ok=True)
    with _exclusive(output):
        journal = output / "submission_state.json"
        _safe_local(journal)
        state = _json(journal) if journal.exists() else {"binding": binding, "phase": "NEW", "timings": {}}
        if state["binding"] != binding:
            raise ValueError("Preserve the journal; submission inputs or authority changed")
        phase_start = perf_counter()
        def save(phase, **fields):
            nonlocal phase_start
            state["timings"][state["phase"]] = state["timings"].get(state["phase"], 0) + perf_counter() - phase_start
            phase_start = perf_counter()
            state.update(phase=phase, observed_at=datetime.now(timezone.utc).isoformat(), **fields)
            atomic_json(journal, state)
        api = api or connect_kaggle(credentials)
        if api.get_config_value(api.CONFIG_NAME_USER) != account:
            raise ValueError("Authenticated SDK account differs from prepared owner")
        if http_session is None:
            import requests
            http_session = requests.Session()
        http_session.auth = (account, values["key"])
        # Recover a lost response only from the actual matching physical source.
        if state["phase"] in {"KERNEL_SUBMITTING", "SUBMIT_UNKNOWN"}:
            remote = _kernel_source(api, ref)
            if remote is None:
                return {"status": "RECONCILIATION_REQUIRED", "phase": state["phase"], "submitted": "UNKNOWN"}
            response = {"status": "submitted", "kernel": ref, "kernel_id": remote.metadata.id,
                        "version_number": remote.metadata.current_version_number, "reconciliation_required": False}
            if response["version_number"] != 1:
                raise ValueError("Lost-response reconciliation requires the first physical version")
            _verify_remote_kernel(remote, response, metadata, (kernel / "run.py").read_bytes())
            save("KERNEL_ACCEPTED", kernel_response=response)
        if "kernel_response" not in state:
            _verify_release_report(prepared / "release_report.json", kernel / "run.py")
            atomic_json(output / "quota_before_publication.json", _quota(http_session, budget))
            if _kernel_source(api, ref) is not None:
                raise ValueError("Target kernel already exists; reconcile its owning journal")
            source_status = _dataset_status(api, dataset)
            if source_status is None:
                if state["phase"] != "NEW":
                    return {"status": "RECONCILIATION_REQUIRED", "phase": state["phase"], "submitted": False}
                save("SOURCE_SUBMITTING")
                try:
                    response = api.dataset_create_new(str(staged), public=False, quiet=True,
                        convert_to_csv=False, dir_mode="zip")
                    actual = str(response.ref).removeprefix("/datasets/")
                    if actual != dataset or response.error:
                        raise ValueError("Source publication identity differs")
                except Exception as error:
                    save("SOURCE_UNKNOWN", error_type=type(error).__name__)
                    return {"status": "RECONCILIATION_REQUIRED", "phase": "SOURCE_UNKNOWN", "submitted": False}
                save("SOURCE_PUBLISHED")
                source_status = _dataset_status(api, dataset)
            elif state["phase"] == "NEW":
                raise ValueError("Source dataset already exists; use its owning publication journal")
            if not source_status or source_status.get("status") != "ready":
                save("SOURCE_PENDING")
                return {"status": "WAIT_SOURCE_READY", "submitted": False}
            proof = _roundtrip_source(api, http_session, staged, output / "source_roundtrip", dataset)
            atomic_json(output / "source_upload_verification.json", proof)
            from .experiment_preflight import check_release_inputs
            prior = _json(prepared / "release_report.json")
            remote_root = output / "source_roundtrip"
            release = check_release_inputs(spec, package, expected_package_identity=manifest["package_identity"],
                recipe_files=prior["inputs"]["recipe_files"], required_modules=prior["inputs"]["required_modules"],
                pickle_inputs=prior["inputs"].get("pickle_inputs", ()),
                dataset_manifests=prior["inputs"].get("dataset_manifests"),
                initial_states={a: remote_root / "initial_states" / (a + ".pt") for a in prior["inputs"]["initial_states"]},
                input_root=remote_root, entry_script=kernel / "run.py", launch_config=remote_root / "experiment_launch.json",
                kernel_metadata=kernel / "kernel-metadata.json")
            if release["errors"]:
                raise ValueError("Remote source failed existing release checks")
            atomic_json(output / "remote_release_report.json", release)
            atomic_json(output / "quota_before_submit.json", _quota(http_session, budget))
            if _kernel_source(api, ref) is not None:
                raise ValueError("Target kernel appeared before POST; reconcile")
            save("KERNEL_SUBMITTING")
            try:
                response = push_kernel_with_accelerator(package_dir=kernel, credential_path=Path(credentials),
                    accelerator="NvidiaTeslaT4", release_report_path=output / "remote_release_report.json")
            except Exception as error:
                save("SUBMIT_UNKNOWN", error_type=type(error).__name__)
                return {"status": "RECONCILIATION_REQUIRED", "phase": "SUBMIT_UNKNOWN", "submitted": "UNKNOWN"}
            atomic_json(output / "platform_response.json", response)
            if response["status"] != "submitted" or response.get("reconciliation_required"):
                save("SUBMIT_UNKNOWN")
                return {"status": "RECONCILIATION_REQUIRED", "phase": "SUBMIT_UNKNOWN", "submitted": "UNKNOWN"}
            save("KERNEL_ACCEPTED", kernel_response=response)
        response = state["kernel_response"]
        verified = _verify_remote_kernel(_kernel_source(api, response["kernel"]), response, metadata, (kernel / "run.py").read_bytes())
        atomic_json(output / "source_verification.json", verified)
        unsubmitted = build_launch_receipt(spec, package, expected_package_identity=manifest["package_identity"])
        fact = lambda v: {"value": v, "missing_reason": None}
        envelope = {"format": "molgap-platform-response", "version": 1, "mode": "observed", "outcome": "accepted",
            "binding": unsubmitted["binding"], "conflict_kind": None,
            "canonical_platform_reference": fact(response["kernel"]), "platform_version": fact(str(response["version_number"])),
            "physical_runs": fact([{"arm_ids": [a["arm_id"] for a in spec.to_dict()["arms"]],
                "canonical_reference": response["kernel"], "platform_version": fact(str(response["version_number"])),
                "run_identity": f"kernel{response['kernel_id']}/version{response['version_number']}"}]),
            "timestamp": fact(state["observed_at"]), "monitor_paths": fact(list(monitor_paths))}
        receipt = reconcile_platform_response(spec, package, canonical_json(envelope), expected_package_identity=manifest["package_identity"])
        if receipt["submission_state"] != "ACCEPTED":
            raise ValueError("Observed physical run was not accepted by the receipt owner")
        receipt_path = output / "launch_receipt.json"
        if not receipt_path.exists():
            atomic_write(receipt_path, canonical_json(receipt).encode())
        else:
            from .experiment_launch import read_launch_receipt
            retained = read_launch_receipt(receipt_path, spec, package, expected_package_identity=manifest["package_identity"])
            if (retained["canonical_platform_reference"] != fact(response["kernel"])
                    or retained["platform_version"] != fact(str(response["version_number"]))
                    or retained["physical_runs"] != envelope["physical_runs"]):
                raise ValueError("Retained launch receipt differs from observed physical run")
        status = api.kernels_status(response["kernel"] + "/" + str(response["version_number"]))
        atomic_json(output / "scheduler_observation.json", {"kernel": response["kernel"], "version": response["version_number"],
            "status": str(status.status), "failure_message": status.failure_message, "observed_at": datetime.now(timezone.utc).isoformat()})
        if not (output / "startup_observation.json").exists():
            atomic_json(output / "startup_observation.json", _startup(http_session, response["kernel"]))
        save("BOUND")
        return {"status": "SUBMISSION_BOUND", "kernel": response["kernel"], "kernel_id": response["kernel_id"],
                "version": response["version_number"], "scheduler_status": str(status.status), "submitted": True,
                "native_qualification": "NOT_EVALUATED", "timings": state["timings"]}
