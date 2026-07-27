"""Create the single-file Kaggle upload bundle for one additive-2M fetch round."""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dir", type=Path, default=Path(__file__).parent)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--round-index", type=int, required=True)
    ap.add_argument("--source-shard-index", type=int, required=True)
    ap.add_argument("--source-shard-count", type=int, default=2)
    ap.add_argument("--profile", choices=("broad", "hard", "complementary"), default="broad")
    ap.add_argument("--spec-file", default="sampling_spec_2m.json")
    ap.add_argument("--kernel-id", required=True)
    ap.add_argument("--kernel-title", required=True)
    ap.add_argument("--checkpoint-dataset", action="append", default=[],
                    help="Optional Kaggle dataset slug containing completed prior-round CSVs")
    args = ap.parse_args()
    if not 1 <= args.round_index <= 99:
        raise ValueError("--round-index must be in [1, 99]")
    if not 0 <= args.source_shard_index < args.source_shard_count:
        raise ValueError("--source-shard-index must be in [0, --source-shard-count)")

    source = args.source_dir
    out = args.out_dir
    if out.exists():
        raise FileExistsError(f"Refusing to replace existing package directory: {out}")
    out.mkdir(parents=True)
    runner = (source / "fetch_2m_parallel.py").read_text(encoding="utf-8")
    replacements = {
        "__FETCHER_PAYLOAD__": base64.b64encode((source / "fetch_repair_candidates.py").read_bytes()).decode("ascii"),
        "__SPEC_PAYLOAD__": base64.b64encode((source / args.spec_file).read_bytes()).decode("ascii"),
    }
    for marker, payload in replacements.items():
        runner = runner.replace(marker, payload)
    runner, substitutions = re.subn(r"^ROUND_INDEX = \d+$", f"ROUND_INDEX = {args.round_index}", runner,
                                    count=1, flags=re.MULTILINE)
    runner, shard_substitutions = re.subn(
        r"^SOURCE_SHARD_INDEX = \d+$",
        f"SOURCE_SHARD_INDEX = {args.source_shard_index}",
        runner,
        count=1,
        flags=re.MULTILINE,
    )
    runner, count_substitutions = re.subn(
        r"^SOURCE_SHARD_COUNT = \d+$",
        f"SOURCE_SHARD_COUNT = {args.source_shard_count}",
        runner,
        count=1,
        flags=re.MULTILINE,
    )
    runner, profile_substitutions = re.subn(
        r'^ACQUISITION_PROFILE = "[a-z]+"$',
        f'ACQUISITION_PROFILE = "{args.profile}"',
        runner,
        count=1,
        flags=re.MULTILINE,
    )
    if (
        substitutions != 1
        or shard_substitutions != 1
        or count_substitutions != 1
        or profile_substitutions != 1
        or any(marker in runner for marker in replacements)
    ):
        raise RuntimeError("Kaggle payload injection failed")
    (out / "fetch_2m_parallel.py").write_text(runner, encoding="utf-8")

    metadata = json.loads((source / "kernel-metadata.json").read_text(encoding="utf-8"))
    metadata["id"] = args.kernel_id
    metadata["title"] = args.kernel_title
    datasets = metadata.setdefault("dataset_sources", [])
    for checkpoint in args.checkpoint_dataset:
        if checkpoint not in datasets:
            datasets.append(checkpoint)
    (out / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared Kaggle {args.profile} round {args.round_index:02d}: {out}")


if __name__ == "__main__":
    main()
