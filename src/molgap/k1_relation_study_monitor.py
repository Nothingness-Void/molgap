"""Mechanical Kaggle polling for the existing Luna B, without decision logic."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from .server_control import LocalServerControlStore


def _latest_identity(api, kernel):
    from kaggle.api.kaggle_api_extended import ApiGetKernelRequest

    owner, slug = kernel.split("/", 1)
    request = ApiGetKernelRequest()
    request.user_name, request.kernel_slug = owner, slug
    with api.build_kaggle_client() as client:
        metadata = client.kernels.kernels_api_client.get_kernel(request).metadata
    return {"kernel": metadata.ref, "kernel_id": metadata.id,
            "version": metadata.current_version_number}


def tick(binding_path):
    binding = json.loads(Path(binding_path).read_text(encoding="utf-8"))
    if binding.get("owner") != "server" or binding.get("closed") is True:
        return {"events": [], "closed": True}
    credential = json.loads(Path(binding["credential_file"]).read_text(encoding="utf-8"))
    owner = binding.get("credential_owner", "kaseichou")
    if credential.get("username") != owner or any(
        not job["kernel"].startswith(owner + "/") for job in binding["jobs"] if not job.get("closed")):
        raise RuntimeError("Bound Kaggle credential owner mismatch")
    os.environ["KAGGLE_USERNAME"] = credential["username"]
    os.environ["KAGGLE_KEY"] = credential["key"]
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    events, observations = [], []
    for job in binding["jobs"]:
        if job.get("closed") is True:
            continue
        store = LocalServerControlStore(Path(job["control_state"]))
        state = store.load()
        run = state["binding"]
        if run["run_id"] != f"{job['kernel']}:v{job['version']}":
            raise RuntimeError("Monitor run binding differs from the release receipt")
        durable = [row for row in state["events"].values()
                   if row["event_status"] == "EVENT_DURABLE"]
        if durable:
            events.extend({"slot": job["slot"], "event_id": row["event_id"],
                           "event_type": row["event_type"]} for row in durable)
            continue
        if state["events"]:
            # Already delivered/claimed terminal state is controller custody.
            continue
        detail = {"kernel": job["kernel"], "version": job["version"]}
        try:
            # The SDK accepts /version but its status endpoint ignores it.
            # Qualify the latest physical version before trusting its status.
            identity = _latest_identity(api, job["kernel"])
            detail["resolved_latest_identity"] = identity
            if identity != {key: job[key] for key in ("kernel", "kernel_id", "version")}:
                raise RuntimeError("Latest remote version is not the bound job")
            response = api.kernels_status(f"{job['kernel']}/{job['version']}")
            status = getattr(response, "status", response)
            status = getattr(status, "name", str(status)).split(".")[-1].upper()
            detail["api_status"] = status
            # Kaggle's acknowledged cancellation is terminal, not an API outage.
            status = {"CANCEL_ACKNOWLEDGED": "CANCELLED", "CANCELED": "CANCELLED"}.get(status, status)
            if status not in {"QUEUED", "RUNNING", "COMPLETE", "ERROR", "FAILED", "CANCELLED"}:
                status = "UNKNOWN"
        except Exception as error:
            status = "UNKNOWN"
            detail["error_type"] = type(error).__name__
        result = store.observe(run_id=run["run_id"], attempt_id=run["attempt_id"],
            monitor_generation=run["monitor_generation"], remote_status=status, detail=detail)
        observations.append({"slot": job["slot"], **asdict(result)})
        if result.event_id:
            events.append({"slot": job["slot"], "event_id": result.event_id, "event_type": status})
    return {"events": events, "observations": observations, "model_inference_executed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("tick", "ack"))
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--slot")
    parser.add_argument("--event-id")
    args = parser.parse_args()
    if args.operation == "tick":
        result = tick(args.binding)
    else:
        binding = json.loads(args.binding.read_text())
        jobs = [job for job in binding["jobs"] if job["slot"] == args.slot]
        if len(jobs) != 1 or not args.event_id:
            raise ValueError("Acknowledgment requires one exact slot and event")
        result = LocalServerControlStore(Path(jobs[0]["control_state"])).deliver_to_a(args.event_id)
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
