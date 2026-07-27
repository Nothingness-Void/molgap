"""Build a deterministic, distribution-stratified repaired-2M SchNet A/B subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST = (
    REPO / "results" / "phase8" / "repaired_2m" / "repaired_2m_manifest.parquet"
)
DEFAULT_CSV = REPO / "data" / "raw" / "phase8_repaired_2m.csv"
DEFAULT_OUT = REPO / "data" / "raw" / "phase8_repaired_2m_schnet_ab_30k.csv"
DEFAULT_REPORT = (
    REPO
    / "results"
    / "phase8"
    / "experiments"
    / "schnet_arch_repaired_2m_30k"
    / "subset_manifest.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def proportional_sample(
    manifest: pd.DataFrame,
    n_rows: int,
    seed: int,
) -> np.ndarray:
    """Allocate rows proportionally across source-group and joint-bucket strata."""
    if not 0 < n_rows <= len(manifest):
        raise ValueError(f"n_rows must be in [1, {len(manifest)}]")

    frame = manifest[["manifest_row", "source_group", "joint_bucket"]].copy()
    frame["stratum"] = (
        frame["source_group"].fillna("unknown").astype(str)
        + "||"
        + frame["joint_bucket"].fillna("unknown").astype(str)
    )
    counts = frame.groupby("stratum", sort=True).size()
    ideal = counts.astype(float) * n_rows / len(frame)
    allocation = np.floor(ideal).astype(int)
    remainder = n_rows - int(allocation.sum())
    if remainder:
        order = (ideal - allocation).sort_values(ascending=False, kind="stable").index
        allocation.loc[order[:remainder]] += 1

    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for stratum, count in allocation.items():
        if count == 0:
            continue
        rows = frame.loc[frame["stratum"] == stratum, "manifest_row"].to_numpy()
        selected.append(rng.choice(rows, size=int(count), replace=False))
    result = np.sort(np.concatenate(selected).astype(np.int64, copy=False))
    if len(result) != n_rows or len(np.unique(result)) != n_rows:
        raise RuntimeError("Stratified sample did not produce the requested unique rows")
    return result


def read_selected_csv(csv_path: Path, selected_rows: np.ndarray) -> pd.DataFrame:
    selected = set(selected_rows.tolist())
    parts: list[pd.DataFrame] = []
    offset = 0
    for chunk in pd.read_csv(csv_path, chunksize=100_000):
        stop = offset + len(chunk)
        local = [row - offset for row in selected if offset <= row < stop]
        if local:
            part = chunk.iloc[sorted(local)].copy()
            part["manifest_row"] = [offset + index for index in sorted(local)]
            parts.append(part)
        offset = stop
    if offset == 0 or not parts:
        raise RuntimeError(f"No selected rows found in {csv_path}")
    output = pd.concat(parts, ignore_index=True).sort_values("manifest_row")
    if len(output) != len(selected_rows):
        raise RuntimeError(
            f"Expected {len(selected_rows)} selected rows, found {len(output)}"
        )
    return output.drop(columns=["manifest_row"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--n-rows", type=int, default=30_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = pd.read_parquet(args.manifest)
    selected_rows = proportional_sample(manifest, args.n_rows, args.seed)
    output = read_selected_csv(args.csv, selected_rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    output.to_csv(temporary, index=False)
    temporary.replace(args.out)

    selected_manifest = manifest.set_index("manifest_row").loc[selected_rows]
    report = {
        "dataset": "phase8_repaired_2m_schnet_ab_subset",
        "rows": len(output),
        "seed": args.seed,
        "sampling": "proportional source_group x joint_bucket",
        "source_manifest": str(args.manifest),
        "source_csv": str(args.csv),
        "output_csv": str(args.out),
        "output_sha256": sha256(args.out),
        "source_group_counts": selected_manifest["source_group"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "joint_bucket_counts": selected_manifest["joint_bucket"]
        .value_counts()
        .sort_index()
        .to_dict(),
        "manifest_row_min": int(selected_rows.min()),
        "manifest_row_max": int(selected_rows.max()),
    }
    temporary_report = args.report.with_suffix(args.report.suffix + ".tmp")
    temporary_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary_report.replace(args.report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

