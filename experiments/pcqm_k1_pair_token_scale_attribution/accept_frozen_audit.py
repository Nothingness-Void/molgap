"""No-inference acceptance of the frozen 500K PairToken intervention output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from molgap.pcqm_k1_causal_audit import _atomic_json, _summarize
from molgap.pcqm_k1_explainability import sha256_file
from molgap.pcqm_k1_pair_token_scale_audit import (
    CHECKPOINT_SHA256, CHUNK_ROWS, DEVELOPMENT_ROWS, MODES, PAYLOAD_SHA256,
    _identity,
)
from molgap.pcqm_k1_scale import FIXED_500K_MANIFEST_SHA256


def accept(output_root: Path, payload_path: Path, checkpoint_path: Path,
           cache_root: Path, acceptance_path: Path) -> dict:
    if sha256_file(checkpoint_path) != CHECKPOINT_SHA256:
        raise RuntimeError("Accepted candidate checkpoint changed")
    if sha256_file(payload_path) != PAYLOAD_SHA256:
        raise RuntimeError("Accepted candidate predictions changed")
    if sha256_file(cache_root / "manifest.json") != FIXED_500K_MANIFEST_SHA256:
        raise RuntimeError("Fixed cache identity changed")
    completion = json.loads((output_root / "completion_manifest.json").read_text())
    audit_path = output_root / "causal_audit.json"
    audit = json.loads(audit_path.read_text())
    if completion.get("complete") is not True or audit.get("complete") is not True:
        raise RuntimeError("Audit has no terminal completion")
    if completion.get("identity") != _identity() or audit.get("identity") != _identity():
        raise RuntimeError("Audit input identity changed")
    if sha256_file(audit_path) != completion.get("causal_audit_sha256"):
        raise RuntimeError("Audit summary hash changed")
    if completion.get("chunk_sha256") != audit.get("chunk_sha256"):
        raise RuntimeError("Chunk manifests disagree")
    if audit.get("training_executed") is not False or audit.get("model_inference_executed") is not True:
        raise RuntimeError("Audit execution type changed")
    for role in ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read"):
        if audit.get(role) is not False:
            raise RuntimeError(f"Protected role flag changed: {role}")
    if audit.get("development_rows") != DEVELOPMENT_ROWS or audit.get("batch_size") != 128:
        raise RuntimeError("Audit rows or physical batch changed")
    rows = []
    hashes = audit["chunk_sha256"]
    if len(hashes) != DEVELOPMENT_ROWS // CHUNK_ROWS:
        raise RuntimeError("Missing audit chunks")
    for index in range(DEVELOPMENT_ROWS // CHUNK_ROWS):
        name = f"chunk_{index:02d}.pt"
        path = output_root / "chunks" / name
        if sha256_file(path) != hashes.get(name):
            raise RuntimeError(f"Audit chunk missing or corrupt: {name}")
        chunk = torch.load(path, map_location="cpu", weights_only=False)
        if set(chunk["predictions"]) != set(MODES) or len(chunk["target"]) != CHUNK_ROWS:
            raise RuntimeError(f"Audit chunk mode/row mismatch: {name}")
        rows.append(chunk)
    source_idx = torch.cat([part["source_idx"] for part in rows])
    target = torch.cat([part["target"] for part in rows])
    reference = torch.load(payload_path, map_location="cpu", weights_only=True)
    if not torch.equal(source_idx, reference["source_idx"].view(-1).long()) or not torch.equal(
        target, reference["target"].view(-1).float()
    ) or not torch.equal(source_idx, torch.arange(500_000, 550_000)):
        raise RuntimeError("Audit rows/targets do not match the frozen payload")
    y = target.numpy().astype(np.float64)
    predictions = {
        mode: torch.cat([part["predictions"][mode] for part in rows]).numpy().astype(np.float64)
        for mode in MODES
    }
    if not all(np.isfinite(values).all() for values in predictions.values()):
        raise RuntimeError("Non-finite audit prediction")
    descriptors = {
        key: torch.cat([part["descriptors"][key] for part in rows]).numpy().astype(np.float64)
        for key in rows[0]["descriptors"]
    }
    reproduced, _ = _summarize(y, predictions, descriptors, {})
    for stratum, summary in reproduced.items():
        reported = audit["interventions"][stratum]
        if reported["rows"] != summary["rows"]:
            raise RuntimeError(f"Stratum membership changed: {stratum}")
        if abs(reported["baseline_mae_eV"] - summary["baseline_mae_eV"]) > 1e-9:
            raise RuntimeError(f"Baseline summary changed: {stratum}")
        for mode, values in summary["counterfactuals"].items():
            original = reported["counterfactuals"][mode]
            for key in ("mae_eV", "gain_vs_baseline_eV", "row_fraction_improved"):
                if abs(original[key] - values[key]) > 1e-9:
                    raise RuntimeError(f"Counterfactual summary changed: {stratum}/{mode}/{key}")
    frozen_pred = reference["prediction"].view(-1).float().numpy().astype(np.float64)
    baseline_diff = np.abs(predictions["baseline"] - frozen_pred)
    if float(baseline_diff.max()) > 5e-4:
        raise RuntimeError("Audit baseline did not reproduce frozen predictions")
    report = {
        "format": "molgap-pcqm-k1-pair-token-scale-audit-acceptance-v1",
        "accepted": True,
        "model_inference_executed": False,
        "training_executed": False,
        "official_validation_role_read": False,
        "test_dev_role_read": False,
        "rows": DEVELOPMENT_ROWS,
        "identity": _identity(),
        "causal_audit_sha256": sha256_file(audit_path),
        "completion_manifest_sha256": sha256_file(output_root / "completion_manifest.json"),
        "audit_source_commit": audit["audit_source_commit"],
        "source_archive_sha256": audit["source_archive_sha256"],
        "baseline_mae_eV": reproduced["all"]["baseline_mae_eV"],
        "all_rows_interventions": reproduced["all"]["counterfactuals"],
    }
    _atomic_json(acceptance_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--reference-payload", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--acceptance-output", type=Path, required=True)
    args = parser.parse_args()
    report = accept(args.output_root, args.reference_payload, args.checkpoint,
                    args.cache_root, args.acceptance_output)
    print(json.dumps({"accepted": report["accepted"], "rows": report["rows"]}))


if __name__ == "__main__":
    main()
