"""Build one durable 60K round of the broad additive candidate pool.

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
SOURCE_SHARD_INDEX = 0
SOURCE_SHARD_COUNT = 1
ACQUISITION_PROFILE = "broad"
PROFILE_CONFIG = {
    "broad": {
        "windows_per_file": 16,
        "groups": (
            ("rare", 1_000, ("very_low_gap", "low_gap_aromatic_edge")),
            ("aromatic_large", 6_000, ("large_aromatic_edge", "very_large_general", "aromatic_edge_general")),
            ("topology_elements", 5_000, ("s_or_cl_hard", "flexible_hard", "large_mw_500_700")),
            ("balanced", 48_000, ("balanced_general",)),
        ),
    },
    "hard": {
        "windows_per_file": 24,
        "groups": (
            ("very_large", 35_000, ("high_sp3_very_large", "non_aromatic_very_large", "flexible_very_large")),
            ("macro_amide", 20_000, ("macrocycle_very_large", "multi_amide_very_large")),
            ("flexible_lowmid", 25_000, ("low_mid_gap_flexible",)),
            ("sp3_nonaromatic", 20_000, ("high_sp3_non_aromatic",)),
        ),
    },
    "complementary": {
        "windows_per_file": 24,
        "groups": (
            ("high_gap", 15_000, ("high_gap_hetero", "high_gap_rigid")),
            ("hetero_dense", 18_000, ("small_hetero_dense", "hetero_dense_midgap", "sulfur_rich", "halogen_rich")),
            ("bridged_rigid", 12_000, ("bridged_polycyclic", "fused_rigid")),
            ("conjugated_da", 15_000, ("donor_acceptor_conjugated", "conjugated_midgap")),
        ),
    },
}
PROFILE = PROFILE_CONFIG[ACQUISITION_PROFILE]
GROUPS = PROFILE["groups"]
ROUND_TOTAL = sum(target for _, target, _ in GROUPS)
WINDOWS_PER_FILE = int(PROFILE["windows_per_file"])

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
                  if name.endswith(".csv") and (
                      name.startswith("phase8_2m_")
                      or name == "residual_target_round01_recovered.csv"
                  ))


def fixed_exclusion_csvs() -> list[Path]:
    return [find_input("phase8_repair_v2_candidate_union_exclusion.csv")]


def install_dependencies() -> None:
    for module, package in (("ijson", "ijson"), ("rdkit", "rdkit")):
        try:
            __import__(module)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", package], check=True)


def materialize_payloads() -> Path:
    runtime = Path("/kaggle/working/molgap_2m_fetch_runtime")
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "fetch_repair_candidates.py").write_bytes(base64.b64decode(EMBEDDED_FETCHER_B64))
    (runtime / "sampling_spec_2m.json").write_bytes(base64.b64decode(EMBEDDED_SPEC_B64))
    return runtime


def write_manifest(out_dir: Path, records: dict[str, dict[str, object]], started: float, state: str,
                   prior_csvs: list[Path]) -> None:
    atomic_json(out_dir / "phase8_2m_round_manifest.json", {
        "tag": f"phase8_2m_{ACQUISITION_PROFILE}_candidate_pool",
        "acquisition_profile": ACQUISITION_PROFILE,
        "round_index": ROUND_INDEX,
        "round_target_rows": ROUND_TOTAL,
        "source_file_shard": {"index": SOURCE_SHARD_INDEX, "count": SOURCE_SHARD_COUNT},
        "state": state,
        "prior_checkpoint_csvs": [str(path) for path in prior_csvs],
        "groups": records,
        "elapsed_s": time.time() - started,
    })


def main() -> None:
    install_dependencies()
    root = materialize_payloads()
    out_dir = Path("/kaggle/working") / f"phase8_2m_round_{ROUND_INDEX:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = find_input("phase8_repair_v3_1p5m.csv")
    prior_csvs = fixed_exclusion_csvs() + prior_checkpoint_csvs()
    fetcher = root / "fetch_repair_candidates.py"
    spec = root / "sampling_spec_2m.json"
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
                stem = f"phase8_2m_round{ROUND_INDEX:02d}_{group}"
                csv_path = out_dir / f"{stem}.csv"
                report_path = out_dir / f"{stem}_report.json"
                progress_path = out_dir / f"{stem}_progress.json"
                log_path = out_dir / f"{stem}.log"
                command = [
                    sys.executable, "-u", str(fetcher), "--spec", str(spec),
                    "--train-csv", str(train_csv), "--include-buckets", *buckets,
                    "--max-kept", str(target), "--windows-per-file", str(WINDOWS_PER_FILE),
                    "--chunk-bytes", "16000000", "--download-workers", "2",
                    "--file-shard-index", str(SOURCE_SHARD_INDEX),
                    "--file-shard-count", str(SOURCE_SHARD_COUNT),
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

    all_processes_ok = True
    all_quotas_full = True
    for group, target, _ in GROUPS:
        report_path = out_dir / f"phase8_2m_round{ROUND_INDEX:02d}_{group}_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            records[group]["report"] = report
            records[group]["sha256"] = sha256(out_dir / str(records[group]["csv"]))
            all_processes_ok = all_processes_ok and records[group].get("return_code") == 0
            all_quotas_full = all_quotas_full and report.get("total_rows") == target
        else:
            all_processes_ok = False
            all_quotas_full = False
    state = "complete" if all_quotas_full else "complete_partial" if all_processes_ok else "failed"
    write_manifest(out_dir, records, started, state, prior_csvs)

    archive = Path("/kaggle/working") / f"phase8_2m_round_{ROUND_INDEX:02d}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for path in sorted(out_dir.iterdir()):
            bundle.write(path, arcname=path.name)
    print(f"round {ROUND_INDEX:02d}: state={state} archive={archive}", flush=True)
    if not all_processes_ok:
        raise RuntimeError("One or more fetch subprocesses failed")


if __name__ == "__main__":
    main()
