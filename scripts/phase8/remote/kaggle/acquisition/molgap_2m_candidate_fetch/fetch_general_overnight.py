"""Collect a long, untargeted PubChemQC pool in bounded 100K chunks."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import zipfile
from pathlib import Path


TOTAL_TARGET = 500_000
CHUNK_TARGET = 100_000
RUN_TAG = "phase8_2m_general_overnight_r01"
SEED_BASE = 91_000

EMBEDDED_FETCHER_B64 = "__FETCHER_PAYLOAD__"
EMBEDDED_SPEC_B64 = "__SPEC_PAYLOAD__"


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_input(filename: str) -> Path:
    matches = [
        Path(root) / filename
        for root, _, names in os.walk("/kaggle/input")
        if filename in names
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one mounted {filename}, found {matches}")
    return matches[0]


def mounted_checkpoint_csvs() -> list[Path]:
    return sorted(
        Path(root) / name
        for root, _, names in os.walk("/kaggle/input")
        for name in names
        if name.endswith(".csv") and (
            name.startswith("phase8_2m_")
            or name == "residual_target_round01_recovered.csv"
        )
    )


def install_dependencies() -> None:
    for module, package in (("ijson", "ijson"), ("rdkit", "rdkit")):
        try:
            __import__(module)
        except ImportError:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", package],
                check=True,
            )


def materialize_payloads() -> tuple[Path, Path]:
    runtime = Path("/kaggle/working/molgap_2m_general_runtime")
    runtime.mkdir(parents=True, exist_ok=True)
    fetcher = runtime / "fetch_repair_candidates.py"
    spec = runtime / "sampling_spec_2m.json"
    fetcher.write_bytes(base64.b64decode(EMBEDDED_FETCHER_B64))
    spec.write_bytes(base64.b64decode(EMBEDDED_SPEC_B64))
    return fetcher, spec


def package_chunk(out_dir: Path, chunk_index: int) -> Path:
    stem = f"{RUN_TAG}_chunk{chunk_index:02d}"
    archive = Path("/kaggle/working") / f"{stem}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in sorted(out_dir.glob(f"{stem}*")):
            if path.is_file():
                bundle.write(path, arcname=path.name)
    return archive


def main() -> None:
    install_dependencies()
    fetcher, spec = materialize_payloads()
    subprocess.run([sys.executable, str(fetcher), "--help"], check=True, stdout=subprocess.DEVNULL)

    out_dir = Path("/kaggle/working") / RUN_TAG
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path("/kaggle/working") / f"{RUN_TAG}_manifest.json"
    train_csv = find_input("phase8_repair_v3_1p5m.csv")
    fixed_exclusion = find_input("phase8_repair_v2_candidate_union_exclusion.csv")
    mounted = mounted_checkpoint_csvs()
    completed_csvs: list[Path] = []
    records: list[dict[str, object]] = []
    started = time.time()
    active_process: subprocess.Popen[str] | None = None
    stop_requested = False

    def write_manifest(state: str) -> None:
        atomic_json(
            manifest_path,
            {
                "tag": RUN_TAG,
                "state": state,
                "total_target_rows": TOTAL_TARGET,
                "chunk_target_rows": CHUNK_TARGET,
                "completed_target_rows": sum(int(record.get("rows", 0)) for record in records),
                "mounted_checkpoint_csvs": [str(path) for path in mounted],
                "chunks": records,
                "elapsed_s": time.time() - started,
            },
        )

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"received signal {signum}; packaging durable partial output", flush=True)
        if active_process and active_process.poll() is None:
            active_process.terminate()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    write_manifest("starting")

    chunk_count = (TOTAL_TARGET + CHUNK_TARGET - 1) // CHUNK_TARGET
    for chunk_index in range(1, chunk_count + 1):
        if stop_requested:
            break
        target = min(CHUNK_TARGET, TOTAL_TARGET - (chunk_index - 1) * CHUNK_TARGET)
        stem = f"{RUN_TAG}_chunk{chunk_index:02d}"
        csv_path = out_dir / f"{stem}.csv"
        report_path = out_dir / f"{stem}_report.json"
        progress_path = out_dir / f"{stem}_progress.json"
        log_path = out_dir / f"{stem}.log"
        command = [
            sys.executable,
            "-u",
            str(fetcher),
            "--spec",
            str(spec),
            "--train-csv",
            str(train_csv),
            "--general",
            "--max-kept",
            str(target),
            "--windows-per-file",
            "20",
            "--chunk-bytes",
            "16000000",
            "--download-workers",
            "4",
            "--seed",
            str(SEED_BASE + chunk_index),
            "--out-csv",
            str(csv_path),
            "--report-json",
            str(report_path),
            "--progress-json",
            str(progress_path),
            "--checkpoint-every",
            "500",
            "--overwrite",
            "--exclude-csv",
            str(fixed_exclusion),
        ]
        for prior in [*mounted, *completed_csvs]:
            command.extend(("--exclude-csv", str(prior)))

        record: dict[str, object] = {
            "chunk_index": chunk_index,
            "target_rows": target,
            "csv": csv_path.name,
            "report": report_path.name,
            "progress": progress_path.name,
            "state": "running",
        }
        records.append(record)
        write_manifest("running")
        print(f"chunk {chunk_index:02d}/{chunk_count}: started target={target:,}", flush=True)

        with log_path.open("w", encoding="utf-8") as log_handle:
            active_process = subprocess.Popen(
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            last_status = 0.0
            while active_process.poll() is None:
                time.sleep(5)
                now = time.time()
                if now - last_status >= 60:
                    if progress_path.exists():
                        progress = json.loads(progress_path.read_text(encoding="utf-8"))
                        record["latest_progress"] = progress
                        print(
                            f"chunk {chunk_index:02d}: rows={progress.get('total_rows', 0):,} "
                            f"files={progress.get('files_scanned', 0)}",
                            flush=True,
                        )
                    write_manifest("running")
                    last_status = now
            return_code = active_process.returncode
            active_process = None

        rows = 0
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            rows = int(report.get("total_rows", 0))
            record["report_summary"] = report
        elif progress_path.exists():
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            rows = int(progress.get("total_rows", 0))
        record["rows"] = rows
        record["return_code"] = return_code
        if csv_path.exists():
            record["sha256"] = sha256(csv_path)
        record["state"] = "complete" if return_code == 0 and rows == target else "partial"
        archive = package_chunk(out_dir, chunk_index)
        record["archive"] = archive.name
        write_manifest("running" if not stop_requested else "interrupted")
        print(
            f"chunk {chunk_index:02d}: rc={return_code} rows={rows:,} archive={archive.name}",
            flush=True,
        )
        if csv_path.exists() and rows > 0:
            completed_csvs.append(csv_path)
        if return_code != 0 or rows != target:
            break

    completed_rows = sum(int(record.get("rows", 0)) for record in records)
    final_state = "complete" if completed_rows == TOTAL_TARGET else "interrupted" if stop_requested else "partial"
    write_manifest(final_state)
    print(f"{RUN_TAG}: state={final_state} rows={completed_rows:,}", flush=True)


if __name__ == "__main__":
    main()
