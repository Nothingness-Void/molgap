"""Shared isolated T4x2 runtime for approved two-arm K1 screens.

The parent process owns dependency pinning, device reservation, bounded
waiting, and the durable job summary.  Each arm is then imported and trained
in a fresh one-GPU child process; the shared K1 runner owns preflight,
certificates, checkpoints, resume validation, and recovery chunks.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time


EDGE_MEMORY_MODES = (
    "neural_atom_k1_edge_read_norm",
    "neural_atom_k1_edge_context_read_norm",
)
EDGE_SLOT_MODES = (
    "neural_atom_k1_edge_context_no_slot_attention",
    "neural_atom_k1_edge_context_uniform_return",
)
GPSPP_LOCAL_MODES = (
    "neural_atom_k1_gpspp_sender",
    "neural_atom_k1_gpspp_bidirectional",
)
ALLOWED_MODE_PAIRS = tuple(
    frozenset(pair)
    for pair in (EDGE_MEMORY_MODES, EDGE_SLOT_MODES, GPSPP_LOCAL_MODES)
)
DEFAULT_MODES = EDGE_MEMORY_MODES
MODES = tuple(
    json.loads(os.environ.get("MOLGAP_SCREEN_MODES", json.dumps(DEFAULT_MODES)))
)
if (
    len(MODES) != 2
    or len(set(MODES)) != 2
    or frozenset(MODES) not in ALLOWED_MODE_PAIRS
):
    raise RuntimeError(f"Invalid approved two-arm K1 screen: {MODES}")

ROOT = Path(
    os.environ.get(
        "MOLGAP_K1_OUTPUT_ROOT", "/kaggle/working/pcqm_k1_edge_memory"
    )
)
MAX_WALL_SECONDS = 10 * 3600
TORCH_VERSION = "2.4.1+cu121"
TORCH_REQUIREMENT = "torch==2.4.1+cu121"
TORCH_CUDA_VERSION = "12.1"
PYG_VERSION = "2.6.1"
OGB_VERSION = "1.3.6"


def _atomic_json(path: Path, value: dict) -> None:
    """Publish a complete summary in one rename so partial jobs remain legible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _pin_runtime() -> None:
    """Install the accepted K1 software tuple without touching NumPy."""
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            TORCH_REQUIREMENT,
            "--index-url",
            "https://download.pytorch.org/whl/cu121",
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            f"torch-geometric=={PYG_VERSION}",
            f"ogb=={OGB_VERSION}",
        ]
    )
    # Keep the first torch import in a fresh process after installation.  The
    # workers repeat the import only after this ABI/device gate succeeds.
    subprocess.check_call(
        [
            sys.executable,
            "-c",
            (
                "import importlib.metadata as metadata; import torch; "
                f"assert torch.__version__ == {TORCH_VERSION!r}; "
                f"assert torch.version.cuda == {TORCH_CUDA_VERSION!r}; "
                f"assert metadata.version('torch-geometric') == {PYG_VERSION!r}; "
                f"assert metadata.version('ogb') == {OGB_VERSION!r}; "
                "assert torch.cuda.device_count() == 2; "
                "assert all('T4' in torch.cuda.get_device_name(i) for i in range(2))"
            ),
        ]
    )


def _resume_root() -> Path | None:
    """Return only the explicitly requested mode-parent resume directory."""
    value = os.environ.get("MOLGAP_K1_RESUME_ROOT", "").strip()
    if not value:
        return None
    root = Path(value)
    if not root.is_dir():
        raise FileNotFoundError(
            f"MOLGAP_K1_RESUME_ROOT is not a directory: {root}"
        )
    return root


def _resume_path(root: Path | None, mode: str) -> Path | None:
    if root is None:
        return None
    path = root / mode
    if not path.is_dir() or not (path / "last_checkpoint.pt").is_file():
        raise FileNotFoundError(
            f"Explicit resume arm is missing last_checkpoint.pt: {path}"
        )
    return path


def _unattended_restart_detected(output: Path) -> bool:
    # A log-only directory is harmless after a process dies before training;
    # any published training state requires an explicit resume source.
    return any(
        (output / name).is_file()
        for name in (
            "last_checkpoint.pt",
            "best_model.pt",
            "best_development_payload.pt",
            "preflight.json",
            "runtime_certificate.json",
            "trace.json",
            "arm_record.json",
            "completion_manifest.json",
        )
    )


def child(context: dict, mode: str) -> dict:
    """Run one arm after the parent has pinned the shared runtime."""
    if mode not in MODES:
        raise ValueError(mode)
    resume_root = _resume_root()
    resume_from = _resume_path(resume_root, mode)
    output = ROOT / mode
    if resume_from is None and _unattended_restart_detected(output):
        raise RuntimeError(
            f"Existing {output} requires explicit MOLGAP_K1_RESUME_ROOT; "
            "refusing an automatic restart"
        )

    # This import is intentionally inside the child: the parent has already
    # completed the ABI/device check and this process exposes one CUDA device.
    import torch

    if torch.cuda.device_count() != 1 or "T4" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Each K1 candidate worker requires exactly one T4")
    from molgap.pcqm_k1_variants_runner import train_arm

    record = train_arm(
        mode,
        output,
        source_commit=context["source_commit"],
        source_archive_sha256=context["source_archive_sha256"],
        resume_from=resume_from,
    )
    return {
        "mode": mode,
        "complete": True,
        "output_root": str(output),
        "resume_from": str(resume_from) if resume_from is not None else None,
        "parameter_count": record.get("training", {}).get("parameter_count"),
    }


def _tail(path: Path, limit: int = 14_000) -> str:
    try:
        return path.read_text(errors="replace")[-limit:]
    except OSError:
        return ""


def worker(mode: str, device: int, context: dict, deadline: float) -> dict:
    """Reserve one physical device and retain its complete parent-visible log."""
    ROOT.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / f"{mode}.log"
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    source_path = str(context["python_root"])
    environment.update(
        CUDA_VISIBLE_DEVICES=str(device),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONHASHSEED="42",
        MOLGAP_EDGE_MODE=mode,
        MOLGAP_EDGE_CONTEXT=json.dumps(context),
        PYTHONUNBUFFERED="1",
        PYTHONPATH=(
            source_path
            if not existing_pythonpath
            else source_path + os.pathsep + existing_pythonpath
        ),
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "molgap.k1_edge_kaggle_runtime"],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        last_line = None
        while process.poll() is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                process.wait()
                raise TimeoutError(
                    f"{mode}: ten-hour wall limit; retain checkpoints and logs"
                )
            try:
                process.wait(timeout=min(60, remaining))
            except subprocess.TimeoutExpired:
                pass
            lines = _tail(log_path, limit=1 << 20).splitlines()
            if lines and lines[-1] != last_line:
                last_line = lines[-1]
                print(f"{mode}: {last_line}", flush=True)
    if process.returncode:
        print(_tail(log_path), flush=True)
        raise RuntimeError(f"{mode}: child exit {process.returncode}")
    return {
        "mode": mode,
        "complete": True,
        "device": device,
        "log": str(log_path),
        "output_root": str(ROOT / mode),
    }


def _validate_context(context: dict) -> None:
    for key in ("python_root", "source_commit", "source_archive_sha256"):
        if not context.get(key):
            raise RuntimeError(f"Missing verified source context: {key}")
    if not Path(context["python_root"]).is_dir():
        raise FileNotFoundError(
            f"Verified source root is unavailable: {context['python_root']}"
        )


def main(context: dict | None = None) -> dict | None:
    # Child mode must not install packages, create a second pool, or select a
    # different arm.  It is reached only from worker()'s one-device process.
    if os.environ.get("MOLGAP_EDGE_MODE"):
        mode = os.environ["MOLGAP_EDGE_MODE"]
        if mode not in MODES:
            raise ValueError(mode)
        child(json.loads(os.environ["MOLGAP_EDGE_CONTEXT"]), mode)
        return None

    if context is None:
        raise RuntimeError("Use the verified source launcher with source context")
    _validate_context(context)
    resume_root = _resume_root()
    for mode in MODES:
        _resume_path(resume_root, mode)
    if resume_root is None:
        stale = [mode for mode in MODES if _unattended_restart_detected(ROOT / mode)]
        if stale:
            raise RuntimeError(
                "Existing training state requires explicit "
                f"MOLGAP_K1_RESUME_ROOT: {stale}"
            )

    started = time.monotonic()
    _pin_runtime()
    ROOT.mkdir(parents=True, exist_ok=True)
    deadline = started + MAX_WALL_SECONDS
    outcomes = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(worker, mode, device, context, deadline): mode
            for device, mode in enumerate(MODES)
        }
        for future in concurrent.futures.as_completed(futures):
            mode = futures[future]
            try:
                outcomes.append(future.result())
            except Exception as error:
                outcomes.append(
                    {
                        "mode": mode,
                        "complete": False,
                        "error": str(error),
                        "log": str(ROOT / f"{mode}.log"),
                        "output_root": str(ROOT / mode),
                    }
                )

    outcomes.sort(key=lambda item: item["mode"])
    summary = {
        "format": "molgap-k1-two-arm-job-summary-v1",
        "complete": all(item["complete"] for item in outcomes),
        "modes": list(MODES),
        "max_wall_seconds": MAX_WALL_SECONDS,
        "elapsed_seconds": time.monotonic() - started,
        "source_commit": context["source_commit"],
        "source_archive_sha256": context["source_archive_sha256"],
        "resume_root": str(resume_root) if resume_root is not None else None,
        "outcomes": outcomes,
    }
    _atomic_json(ROOT / "job_summary.json", summary)
    if not summary["complete"]:
        raise RuntimeError(
            f"K1 worker failure; retain independent logs/checkpoints: "
            f"{outcomes}"
        )
    return summary


if __name__ == "__main__":
    main()
