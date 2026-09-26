"""Mechanical Kaggle polling for the existing Luna B, without decision logic."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from .server_control import LocalServerControlStore


def tick(binding_path):
    binding = json.loads(Path(binding_path).read_text(encoding="utf-8"))
    if binding.get("owner") != "server" or binding.get("closed") is True:
        return {"events": [], "closed": True}
    credential = json.loads(Path(binding["credential_file"]).read_text(encoding="utf-8"))
    if credential.get("username") != "kaseichou":
        raise RuntimeError("Kaggle2 credential owner mismatch")
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
            response = api.kernels_status(f"{job['kernel']}/{job['version']}")
            status = getattr(response, "status", response)
            status = getattr(status, "name", str(status)).split(".")[-1].upper()
            if status not in {"QUEUED", "RUNNING", "COMPLETE", "ERROR", "FAILED", "CANCELLED"}:
                detail["api_status"] = status
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
