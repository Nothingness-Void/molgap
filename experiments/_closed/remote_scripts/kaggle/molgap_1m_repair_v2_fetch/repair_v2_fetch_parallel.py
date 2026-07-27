"""Build one durable 60K round of the disjoint 1M-v2 candidate pool on Kaggle.

Each successful round is an independently downloadable Kaggle output.  The
next round mounts the prior checkpoint dataset, excludes its CIDs/SMILES, and
uses a new deterministic seed.  Do not turn this back into a single 600K job:
Kaggle only publishes task output after normal completion.
"""
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


ROUND_INDEX = 1
ROUND_TOTAL = 60_000
GROUPS = (
    ("rare", 1_000, ("very_low_gap", "low_gap_aromatic_edge")),
    ("aromatic_large", 8_000, ("large_aromatic_edge", "very_large_general", "aromatic_edge_general")),
    ("topology_elements", 12_000, ("s_or_cl_hard", "flexible_hard", "large_mw_500_700")),
    ("balanced", 39_000, ("balanced_general",)),
)

# Kaggle script kernels upload only `code_file`; companion files are injected
# by package_kernel.py immediately before submission.
EMBEDDED_FETCHER_B64 = "__FETCHER_PAYLOAD__"
EMBEDDED_SPEC_B64 = "__SPEC_PAYLOAD__"


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_input(filename: str) -> Path:
    matches = [Path(root) / filename for root, _, names in os.walk("/kaggle/input") if filename in names]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one mounted {filename}, found {matches}")
    return matches[0]


def prior_checkpoint_csvs() -> list[Path]:
    """Find only checkpoint CSVs, never arbitrary datasets mounted by Kaggle."""
    return sorted(Path(root) / name for root, _, names in os.walk("/kaggle/input") for name in names
                  if name.startswith("phase8_repair_v2_round") and name.endswith(".csv"))


def install_dependencies() -> None:
    for module, package in (("ijson", "ijson"), ("rdkit", "rdkit")):
        try:
            __import__(module)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", package], check=True)


def materialize_payloads() -> Path:
    runtime = Path("/kaggle/working/molgap_repair_v2_runtime")
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "fetch_repair_candidates.py").write_bytes(base64.b64decode(EMBEDDED_FETCHER_B64))
    (runtime / "repair_1m_v2_sampling_spec.json").write_bytes(base64.b64decode(EMBEDDED_SPEC_B64))
    return runtime


def write_manifest(out_dir: Path, records: dict[str, dict[str, object]], started: float, state: str,
                   prior_csvs: list[Path]) -> None:
    atomic_json(out_dir / "repair_1m_v2_round_manifest.json", {
        "tag": "phase8_repair_1m_v2_candidate_pool",
        "round_index": ROUND_INDEX,
        "round_target_rows": ROUND_TOTAL,
        "state": state,
        "prior_checkpoint_csvs": [str(path) for path in prior_csvs],
        "groups": records,
        "elapsed_s": time.time() - started,
    })


def main() -> None:
    install_dependencies()
    root = materialize_payloads()
    out_dir = Path("/kaggle/working") / f"phase8_repair_v2_round_{ROUND_INDEX:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = find_input("phase8_expansion_1m.csv")
    prior_csvs = prior_checkpoint_csvs()
    fetcher = root / "fetch_repair_candidates.py"
    spec = root / "repair_1m_v2_sampling_spec.json"
    subprocess.run([sys.executable, str(fetcher), "--help"], check=True, stdout=subprocess.DEVNULL)

    pending = list(GROUPS)
    max_parallel = min(len(GROUPS), max(1, os.cpu_count() or 1))
    running: list[tuple[str, subprocess.Popen[str]]] = []
    records: dict[str, dict[str, object]] = {}
    started = time.time()
    last_status_at = 0.0

    try:
        while pending or running:
            while pending and len(running) < max_parallel:
                group, target, buckets = pending.pop(0)
                stem = f"phase8_repair_v2_round{ROUND_INDEX:02d}_{group}"
                csv_path = out_dir / f"{stem}.csv"
                report_path = out_dir / f"{stem}_report.json"
                progress_path = out_dir / f"{stem}_progress.json"
                log_path = out_dir / f"{stem}.log"
                command = [
                    sys.executable, "-u", str(fetcher), "--spec", str(spec),
                    "--train-csv", str(train_csv), "--include-buckets", *buckets,
                    "--max-kept", str(target), "--windows-per-file", "16",
                    "--chunk-bytes", "16000000", "--download-workers", "2",
                    "--seed", str(42_000 + ROUND_INDEX * 100 + len(records)),
                    "--out-csv", str(csv_path), "--report-json", str(report_path),
                    "--progress-json", str(progress_path), "--checkpoint-every", "250", "--overwrite",
                ]
                for prior in prior_csvs:
                    command.extend(("--exclude-csv", str(prior)))
                log_handle = log_path.open("w", encoding="utf-8")
                process = subprocess.Popen(command, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
                process._molgap_log_handle = log_handle  # type: ignore[attr-defined]
                running.append((group, process))
                records[group] = {
                    "target_rows": target, "buckets": list(buckets), "pid": process.pid,
                    "csv": csv_path.name, "progress": progress_path.name, "log": log_path.name,
                }
                print(f"round {ROUND_INDEX:02d}: started {group}: pid={process.pid}", flush=True)
                write_manifest(out_dir, records, started, "running", prior_csvs)

            time.sleep(5)
            now = time.time()
            still_running: list[tuple[str, subprocess.Popen[str]]] = []
            for group, process in running:
                return_code = process.poll()
                if return_code is None:
                    still_running.append((group, process))
                    continue
                process._molgap_log_handle.close()  # type: ignore[attr-defined]
                records[group]["return_code"] = return_code
                print(f"round {ROUND_INDEX:02d}: finished {group}: rc={return_code}", flush=True)
            running = still_running

            if now - last_status_at >= 60:
                for group in records:
                    progress = out_dir / str(records[group]["progress"])
                    if progress.exists():
                        records[group]["latest_progress"] = json.loads(progress.read_text(encoding="utf-8"))
                write_manifest(out_dir, records, started, "running", prior_csvs)
                summary = {group: rec.get("latest_progress", {}).get("total_rows", 0)
                           if isinstance(rec.get("latest_progress"), dict) else 0
                           for group, rec in records.items()}
                print(f"round {ROUND_INDEX:02d}: durable rows {summary}", flush=True)
                last_status_at = now
    except KeyboardInterrupt:
        for _, process in running:
            process.send_signal(signal.SIGTERM)
        write_manifest(out_dir, records, started, "interrupted", prior_csvs)
        raise

    all_complete = True
    for group, target, _ in GROUPS:
        report_path = out_dir / f"phase8_repair_v2_round{ROUND_INDEX:02d}_{group}_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            records[group]["report"] = report
            records[group]["sha256"] = sha256(out_dir / str(records[group]["csv"]))
            all_complete = all_complete and records[group].get("return_code") == 0 and report.get("total_rows") == target
        else:
            all_complete = False
    write_manifest(out_dir, records, started, "complete" if all_complete else "failed", prior_csvs)

    archive = Path("/kaggle/working") / f"phase8_repair_v2_round_{ROUND_INDEX:02d}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in sorted(out_dir.iterdir()):
            bundle.write(path, arcname=path.name)
    print(f"round {ROUND_INDEX:02d}: state={'complete' if all_complete else 'failed'} archive={archive}", flush=True)
    if not all_complete:
        raise RuntimeError("Round did not reach all durable quota targets")


if __name__ == "__main__":
    main()
