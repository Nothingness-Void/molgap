"""Single-GPU fallback runner for one frozen GraphState width candidate."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


BASELINE = "ogb_distance_angle_triangle_edge_state_graph_state9"
CANDIDATE = "ogb_distance_angle_triangle_edge_state_graph_state9_w128"
ALLOWED = {BASELINE, CANDIDATE}
PASCAL_COMPAT_RESTART = "MOLGAP_GRAPHSTATE_WIDTH_PASCAL_RESTART"


def ensure_pascal_compatible_torch() -> None:
    """Install the verified CUDA 12.6 build when Kaggle assigns a P100."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Kaggle did not allocate a GPU")
    if torch.cuda.get_device_capability(0) != (6, 0):
        return
    if "sm_60" in set(torch.cuda.get_arch_list()):
        return
    if os.environ.get(PASCAL_COMPAT_RESTART) == "1":
        raise RuntimeError("P100 compatibility install still lacks sm_60")
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-cache-dir",
            "--no-deps",
            "--force-reinstall",
            "torch==2.7.1",
            "nvidia-cusparselt-cu12==0.6.3",
            "--index-url",
            "https://download.pytorch.org/whl/cu126",
        ]
    )
    os.environ[PASCAL_COMPAT_RESTART] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


def main() -> None:
    import molgap.pcqm_local_global_runner as base

    candidate = os.environ.get("MOLGAP_SINGLE_CANDIDATE", "")
    if candidate not in ALLOWED:
        raise RuntimeError(f"Unsupported single candidate: {candidate}")
    task_started = time.perf_counter()
    base.OUT.mkdir(parents=True, exist_ok=True)
    completed = False
    try:
        ensure_pascal_compatible_torch()
        base.install_dependencies()
        import torch

        if torch.cuda.device_count() != 1:
            raise RuntimeError(
                f"Expected one isolated GPU, found {torch.cuda.device_count()}"
            )
        gpu_name = torch.cuda.get_device_name(0)
        if not any(token in gpu_name for token in ("T4", "P100")):
            raise RuntimeError(f"Unsupported Kaggle GPU: {gpu_name}")
        if torch.cuda.get_device_capability(0) == (6, 0) and "sm_60" not in set(
            torch.cuda.get_arch_list()
        ):
            raise RuntimeError("P100-compatible torch did not provide sm_60")
        markers = list(Path("/kaggle/input").rglob("PCQM_GAP100K_SOURCE_COMMIT.txt"))
        if len(markers) != 1:
            raise RuntimeError(f"Expected one source marker, found {markers}")
        source_commit = markers[0].read_text(encoding="utf-8").strip()
        if source_commit != base.EXPECTED_MODEL_SOURCE_COMMIT:
            raise RuntimeError(f"GraphState width source changed: {source_commit}")

        cache_root, cache_manifest = base.find_input_cache()
        initialization = base.initialization_preflight()
        initialization_row = next(
            row for row in initialization if row["candidate"] == candidate
        )
        base.atomic_json(
            base.OUT / "initialization_preflight.json",
            {
                "format": "molgap-pcqm-graph-state-width-single-init-v1",
                "complete": True,
                "candidate": candidate,
                "source_commit": source_commit,
                "row": initialization_row,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        graphs = base.load_graphs(cache_root, cache_manifest)
        gpu_preflight = base.gpu_preflight(graphs, candidate, 0)
        metrics = base.train_one(graphs, candidate, task_started, 0)
        completed = True
        result = {
            "format": "molgap-pcqm-graph-state-width-single-result-v1",
            "complete": True,
            "candidate": candidate,
            "source_commit": source_commit,
            "geometry_cache_aggregate_sha256": base.EXPECTED_GEOMETRY_SHA256,
            "input_cache_aggregate_sha256": cache_manifest["aggregate_sha256"],
            "seed": base.SEED,
            "gpu": gpu_name,
            "execution": "isolated_single_gpu_parallel_kernel",
            "initialization_preflight": initialization_row,
            "gpu_preflight": gpu_preflight,
            "metrics": metrics,
            "elapsed_s": time.perf_counter() - task_started,
            "official_validation_role_read": False,
            "test_dev_role_read": False,
            "molecular_research_server_accessed": False,
            "full_data_authorized": False,
        }
        base.atomic_json(base.OUT / "single_result.json", result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        base.atomic_json(
            base.OUT / "single_failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "candidate": candidate,
                "completed": completed,
                "elapsed_s": time.perf_counter() - task_started,
                "official_validation_role_read": False,
                "test_dev_role_read": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
