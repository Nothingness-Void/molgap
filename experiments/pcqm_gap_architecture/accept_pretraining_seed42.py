"""No-model-inference acceptance for paired GraphState9 pretraining screens."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


EXPECTED_CACHE = "3e4206fd239942ab79f9c4978cd4334f6025ba14e8f1cd1df78c20060a0d1d22"
EXPECTED_PARAMETERS = 3_665_809


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    scratch = json.loads((root / "scratch" / "metrics.json").read_text(encoding="utf-8"))
    pretrained = json.loads((root / "pretrained" / "metrics.json").read_text(encoding="utf-8"))
    errors = []
    if selection.get("complete") is not True:
        errors.append("selection is incomplete")
    if selection.get("mode") not in {"structure_source_tasks", "etkdg_geometry_denoising"}:
        errors.append("unknown mode")
    if selection.get("gpu_names") != ["Tesla T4", "Tesla T4"]:
        errors.append(f"unexpected GPUs: {selection.get('gpu_names')}")
    if selection.get("parameter_count") != EXPECTED_PARAMETERS:
        errors.append("downstream parameter count changed")
    if selection.get("geometry_cache_aggregate_sha256") != EXPECTED_CACHE:
        errors.append("geometry cache changed")
    if scratch.get("initial_encoder_sha256") != pretrained.get("initial_encoder_sha256"):
        errors.append("initial encoder hashes differ")
    if len(scratch.get("trace", [])) != 60:
        errors.append("scratch trace is not 60 epochs")
    if len(pretrained.get("trace", [])) != 40:
        errors.append("fine-tune trace is not 40 epochs")
    if len(pretrained.get("pretraining", {}).get("trace", [])) != 20:
        errors.append("pretraining trace is not 20 epochs")
    for payload in (selection, scratch, pretrained):
        if payload.get("official_validation_role_read") is not False:
            errors.append("official validation role was read")
        if payload.get("test_dev_role_read") is not False:
            errors.append("test-dev role was read")
        for key, value in payload.items():
            if isinstance(value, float) and not math.isfinite(value):
                errors.append(f"non-finite {key}")
    artifacts = {}
    for relative in (
        "selection.json", "progress.json", "scratch/metrics.json",
        "scratch/scratch_gap_checkpoint.pt", "pretrained/metrics.json",
        "pretrained/pretraining_checkpoint.pt", "pretrained/finetune_gap_checkpoint.pt",
    ):
        path = root / relative
        if not path.is_file():
            errors.append(f"missing {relative}")
        else:
            artifacts[relative] = digest(path)
    acceptance = {"accepted": not errors, "errors": errors,
                  "mode": selection.get("mode"), "model_inference_executed": False,
                  "official_validation_role_read": False, "test_dev_role_read": False,
                  "artifacts": artifacts}
    output = args.output or root / "acceptance.json"
    output.write_text(json.dumps(acceptance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(acceptance, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

