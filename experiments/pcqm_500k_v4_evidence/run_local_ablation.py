"""Run or resume one matched-500K V4 ablation on the local RTX 5060."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

from molgap.pcqm_500k_v4_ablation import ABLATION_PARAMETERS
from molgap.pcqm_500k_v4_evidence import run
from molgap.pcqm_v4_local import local_source_sha256


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=sorted(ABLATION_PARAMETERS), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    acceptance = json.loads((args.data_root / "local_acceptance.json").read_text())
    if acceptance.get("status") != "accepted" or acceptance.get("deep_graph_check") is not True:
        raise RuntimeError("Local V4 cache has not passed deep acceptance")
    os.environ["MOLGAP_PCQM_500K_V4_ROOT"] = str(args.data_root.resolve())
    os.environ["MOLGAP_V4_LOADER_WORKERS"] = "0"
    suffix = f"{args.arm}_preflight" if args.preflight_only else args.arm
    output = (args.output_root / suffix).resolve()
    output.mkdir(parents=True, exist_ok=True)
    resume = None
    manifest_path = output / "stage_manifest.json"
    if manifest_path.exists() and not args.preflight_only:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") == "COMPLETE" and manifest.get("next_epoch") == 60:
            print(f"SKIP accepted terminal output: {output}")
            return
        resume = output
    with (output / "train.stdout.log").open("a", encoding="utf-8", buffering=1) as log:
        with contextlib.redirect_stdout(_Tee(sys.stdout, log)), contextlib.redirect_stderr(
            _Tee(sys.stderr, log)
        ):
            run(
                args.arm,
                output,
                local_source_sha256(args.repo_root),
                stage_epochs=60,
                resume=resume,
                max_stage_seconds=None,
                platform_id="local-rtx5060",
                preflight_only=args.preflight_only,
            )


if __name__ == "__main__":
    main()
