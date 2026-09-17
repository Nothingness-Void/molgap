from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
from requests.auth import HTTPBasicAuth


KAGGLE_KERNEL_PUSH_URL = "https://www.kaggle.com/api/v1/kernels/push"
ALLOWED_ACCELERATORS = {"NvidiaTeslaT4"}
UNSUPPORTED_BATCH_ACCELERATORS = {"TpuV5E8"}


def push_kernel_with_accelerator(
    *,
    package_dir: Path,
    credential_path: Path,
    accelerator: str,
    timeout_seconds: int = 120,
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
    response = requests.post(
        KAGGLE_KERNEL_PUSH_URL,
        auth=HTTPBasicAuth(username, key),
        json=request,
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("error"):
        raise RuntimeError(f"Kaggle kernel push failed: {result['error']}")
    return {
        "status": "submitted",
        "kernel": metadata["id"],
        "accelerator": accelerator,
        "invalid_dataset_sources": result.get("invalidDatasetSources", []),
        "invalid_kernel_sources": result.get("invalidKernelSources", []),
        "invalid_model_sources": result.get("invalidModelSources", []),
    }
