"""Hash-bound, train-only auxiliary label cache and source-index join."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np

from .chemical_aux_labels import ChemicalLabelEncoder
from .research_memory.trace import atomic_write, json_bytes
from .training_reproducibility import sha256_file


def _pinned_json(path: Path, digest: str) -> dict:
    if len(digest) != 64 or sha256_file(path) != digest:
        raise ValueError("Pinned JSON hash mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def build_cache(rows_path: Path, rows_sha256: str, role_path: Path,
                role_sha256: str, output: Path) -> dict:
    """Rows are a preselected JSONL export, never the entire source dataset.

    The owning adapter supplies independently verified role/export hashes.
    This checks declared provenance; it cannot authenticate that declaration.
    """
    role = _pinned_json(role_path, role_sha256)
    if role.get("role") != "train" or role.get("rows_sha256") != rows_sha256:
        raise ValueError("Only a pinned training-role export is accepted")
    expected = role.get("source_indices")
    if (not isinstance(expected, list) or not expected or
        any(type(i) is not int or i < 0 for i in expected) or
        len(set(expected)) != len(expected)):
        raise ValueError("Role requires unique nonnegative source indices")
    if not role.get("dataset_identity") or not role.get("row_identity_semantics"):
        raise ValueError("Missing dataset or row identity semantics")
    if sha256_file(rows_path) != rows_sha256:
        raise ValueError("Training SMILES export hash mismatch")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    encoder = ChemicalLabelEncoder()
    descriptors, fingerprints, observed, failures = [], [], [], []
    with rows_path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if set(row) != {"source_index", "smiles"}:
                raise ValueError("Rows must contain exactly source_index and smiles")
            index = row["source_index"]
            position = len(observed)
            if position >= len(expected) or type(index) is not int or index != expected[position]:
                raise ValueError("Export row coverage/order differs from pinned role")
            observed.append(index)
            label = encoder.encode(index, row["smiles"])
            if label.status != "ok":
                failures.append({"source_index": index, "reason": label.status})
            else:
                descriptors.append(label.descriptors)
                fingerprints.append(label.fingerprint)
    if observed != expected:
        raise ValueError("Incomplete training SMILES export")
    report = {"schema": "molgap-chemical-cache-v1", "role": role,
              "role_sha256": role_sha256, "rows_sha256": rows_sha256,
              "encoder": encoder.identity(), "row_count": len(observed),
              "failures": failures, "accepted": not failures,
              "cpu_stage_wall_seconds": time.perf_counter() - started}
    if not failures:
        buffer = io.BytesIO()
        np.savez(buffer, source_indices=np.asarray(observed, dtype=np.int64),
                 descriptors=np.stack(descriptors), fingerprints=np.stack(fingerprints))
        atomic_write(output / "labels.npz", buffer.getvalue())
        report["labels_sha256"] = sha256_file(output / "labels.npz")
    atomic_write(output / "manifest.json", json_bytes(report))
    return report


class ChemicalLabelCache:
    def __init__(self, root: Path, manifest_sha256: str, *, expected_role_sha256: str):
        manifest = _pinned_json(root / "manifest.json", manifest_sha256)
        if (manifest.get("schema") != "molgap-chemical-cache-v1" or
            manifest.get("accepted") is not True or manifest.get("failures") != [] or
            manifest.get("role_sha256") != expected_role_sha256 or
            manifest.get("role", {}).get("role") != "train"):
            raise ValueError("Auxiliary cache is not accepted for the expected training role")
        if sha256_file(root / "labels.npz") != manifest["labels_sha256"]:
            raise ValueError("Auxiliary cache content changed")
        with np.load(root / "labels.npz", allow_pickle=False) as arrays:
            ids = arrays["source_indices"].copy()
            self.descriptors = arrays["descriptors"].copy()
            self.fingerprints = arrays["fingerprints"].copy()
        n = manifest["row_count"]
        if (ids.dtype != np.int64 or ids.tolist() != manifest["role"]["source_indices"] or
            len(set(ids.tolist())) != n or ids.shape != (n,) or
            self.descriptors.shape != (n, 200) or self.descriptors.dtype != np.float32 or
            self.fingerprints.shape != (n, 512) or self.fingerprints.dtype != np.uint8 or
            not np.isfinite(self.descriptors).all() or
            not np.isin(self.fingerprints, [0, 1]).all()):
            raise ValueError("Malformed auxiliary arrays or row alignment")
        self.positions = {int(index): position for position, index in enumerate(ids)}
        self.identity = {"manifest_sha256": manifest_sha256, "role_sha256": expected_role_sha256,
                         "rows_sha256": manifest["rows_sha256"]}
        self.manifest = manifest

    def attach(self, batch):
        """Join before device transfer, avoiding a CUDA-to-CPU sync per batch."""
        import torch
        ids = getattr(batch, "source_idx", getattr(batch, "row_index", None))
        if ids is None or ids.device.type != "cpu" or ids.dtype != torch.int64:
            raise ValueError("Batch needs CPU int64 source indices before transfer")
        indices = ids.view(-1).tolist()
        if len(indices) != int(batch.num_graphs) or len(set(indices)) != len(indices):
            raise ValueError("Batch source-index count/uniqueness mismatch")
        try:
            positions = [self.positions[index] for index in indices]
        except KeyError as error:
            raise ValueError("Batch contains a source index outside the frozen training cache") from error
        batch.chemical_descriptors = torch.from_numpy(self.descriptors[positions].copy())
        batch.chemical_fingerprint = torch.from_numpy(self.fingerprints[positions].copy())
        return batch
