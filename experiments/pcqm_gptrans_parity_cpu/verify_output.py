"""Verify retrieved Kaggle CPU evidence; this is not scientific acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from kaggle_cpu.run import AUTHOR, LOCAL, SOURCES, TEST_NAMES


def verify(output: Path, pulled_source: Path | None = None) -> dict:
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    checks = {
        "schema": summary.get("schema") == "molgap-gptrans-parity-cpu-v1",
        "cpu_runtime": summary.get("mode") == "cpu-runtime" and summary.get("cpu_only") is True,
        "no_dataset_roles": summary.get("dataset_roles_read") == [],
        "no_checkpoint": summary.get("checkpoint_loaded") is False,
        "owner_and_slug": summary.get("kaggle_kernel_id") == "kaseichou/molgap-gptrans-cpu-parity-v1",
        "pinned_commits": summary.get("author_commit") == AUTHOR and summary.get("local_frozen_commit") == LOCAL,
        "exact_runner": summary.get("runner_sha256") == hashlib.sha256(
            (Path(__file__).parent / "kaggle_cpu" / "run.py").read_bytes()).hexdigest(),
        "runtime_versions": all(summary.get("runtime", {}).get(key) for key in ("python", "torch", "numpy")),
        "all_required_pass": summary.get("all_required_pass") is True,
        "test_set_exact": set(summary.get("tests", {})) == set(TEST_NAMES),
    }
    statuses = {}
    for name in TEST_NAMES:
        path = output / f"{name}.json"
        if not path.is_file():
            statuses[name] = "MISSING"
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        statuses[name] = item.get("status", "INVALID")
        checks[f"{name}_consistent"] = (
            item.get("status") == "PASS" and summary.get("tests", {}).get(name) == "PASS"
        )
    source = json.loads((output / "source_identity.json").read_text(encoding="utf-8"))
    details = source.get("detail", {})
    checks["all_source_hashes"] = (
        set(details) == set(SOURCES)
        and all(details[name].get("sha256") == expected
                and details[name].get("url") == url
                for name, (url, expected) in SOURCES.items())
    )
    if pulled_source is not None:
        pulled_metadata = json.loads(
            (pulled_source / "kernel-metadata.json").read_text(encoding="utf-8"))
        remote_code = pulled_source / pulled_metadata.get("code_file", "")
        checks["pulled_metadata"] = (
            pulled_metadata.get("id") == "kaseichou/molgap-gptrans-cpu-parity-v1"
            and pulled_metadata.get("enable_gpu") is False
            and pulled_metadata.get("dataset_sources") == []
            and pulled_metadata.get("kernel_type") == "script"
        )
        checks["pulled_source_normalized"] = (
            remote_code.is_file()
            and remote_code.read_text(encoding="utf-8") ==
            (Path(__file__).parent / "kaggle_cpu" / "run.py").read_text(encoding="utf-8")
        )
    return {"verified": all(checks.values()), "checks": checks,
            "statuses": statuses, "output": str(output.resolve())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="directory containing retrieved summary.json")
    parser.add_argument("--pulled-source", type=Path,
                        help="optional directory containing Kaggle-pulled source and metadata")
    args = parser.parse_args()
    result = verify(args.output, args.pulled_source)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
