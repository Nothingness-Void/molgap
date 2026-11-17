"""Hash-bound, train-only auxiliary label cache and source-index join."""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import numpy as np

from .chemical_aux_labels import ChemicalLabelEncoder, ChemicalLabels
from .research_memory.trace import atomic_write, json_bytes
from .training_reproducibility import sha256_file


def _pinned_json(path: Path, digest: str) -> dict:
    if len(digest) != 64 or sha256_file(path) != digest:
        raise ValueError("Pinned JSON hash mismatch")
    return json.loads(path.read_text(encoding="utf-8"))


def export_fixed_training_smiles(archive: Path, graph_root: Path, manifest_path: Path,
                                 output: Path) -> dict:
    """Bind a train-only export to the accepted V4 archive and graph row IDs."""
    import gzip
    import zipfile
    import pandas as pd
    from .pcqm_gptrans_v4 import validate_fixed_assets, TRAIN_ROWS
    from .pcqm_official_edge_state import load_official_splits, CSV_MEMBER

    assets = validate_fixed_assets(graph_root, manifest_path, verify_content=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(archive) != manifest["source"]["official_archive_sha256"]:
        raise ValueError("Official archive differs from accepted fixed graph provenance")
    selected = np.arange(TRAIN_ROWS, dtype=np.int64)
    train = load_official_splits(archive)["train"]
    if not np.isin(selected, train).all():
        raise ValueError("Fixed source indices are not exclusively official train rows")
    # The accepted builder retains official idx as graph source_idx. Only these
    # frozen train rows and SMILES are parsed; no evaluation targets are read.
    with zipfile.ZipFile(archive) as bundle, bundle.open(CSV_MEMBER) as member:
        with gzip.GzipFile(fileobj=member) as stream:
            frame = pd.read_csv(stream, nrows=TRAIN_ROWS, usecols=["idx", "smiles"])
    if len(frame) != TRAIN_ROWS or not np.array_equal(frame["idx"].to_numpy(), selected):
        raise ValueError("Official CSV train row mapping differs from frozen source indices")
    if frame["smiles"].isna().any():
        raise ValueError("Training SMILES export has missing rows")
    output.mkdir(parents=True, exist_ok=False)
    rows_path = output / "train_smiles.jsonl"
    payload = b"".join(json.dumps({"source_index": int(i), "smiles": str(s)},
                                 sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
                       for i, s in zip(frame["idx"], frame["smiles"], strict=True))
    atomic_write(rows_path, payload)
    role = {"role": "train", "source_indices": selected.tolist(),
            "dataset_identity": "pcqm-fixed100k-v4",
            "row_identity_semantics": "pcqm-fixed100k-v4-source_idx",
            "rows_sha256": sha256_file(rows_path),
            "official_archive_sha256": manifest["source"]["official_archive_sha256"],
            "fixed_manifest_sha256": sha256_file(manifest_path),
            "official_row_manifest_sha256": manifest["source"]["official_row_manifest_sha256"],
            "train_graph_sha256": [sha256_file(p) for p in assets.train_paths],
            "official_train_membership_verified": True,
            "protected_target_columns_read": False}
    role_path = output / "train_role.json"
    atomic_write(role_path, json_bytes(role))
    return {"rows": str(rows_path), "rows_sha256": role["rows_sha256"],
            "role": str(role_path), "role_sha256": sha256_file(role_path)}


def build_cache(rows_path: Path, rows_sha256: str, role_path: Path,
                role_sha256: str, output: Path, *,
                components=("descriptors", "fingerprints"), parse_policy="strict",
                descriptor_missing_policy="reject", fail_fast=False,
                priority_source_indices=()) -> dict:
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
    encoder = ChemicalLabelEncoder(components=components, parse_policy=parse_policy,
                                   descriptor_missing_policy=descriptor_missing_policy)
    priority = tuple(priority_source_indices)
    expected_set = set(expected)
    if (len(set(priority)) != len(priority) or
        any(type(i) is not int or i not in expected_set for i in priority)):
        raise ValueError("Priority rows must be unique members of the pinned training role")
    if type(fail_fast) is not bool:
        raise ValueError("fail_fast must be boolean")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    policy = {"components": list(encoder.components), "parse_policy": parse_policy,
              "descriptor_missing_policy": descriptor_missing_policy}
    legacy = policy == {"components": ["descriptors", "fingerprints"],
                        "parse_policy": "strict", "descriptor_missing_policy": "reject"}
    rows, labels, failures, missing = [], {}, [], []

    def report(*, accepted=False, export_complete=False, inventory_complete=False):
        record = {"schema": "molgap-chemical-cache-v1" if legacy else "molgap-chemical-cache-v2",
                  "role": role, "role_sha256": role_sha256, "rows_sha256": rows_sha256,
                  "encoder": encoder.identity(), "row_count": len(labels),
                  "expected_row_count": len(expected), "export_row_count": len(rows),
                  "export_coverage_complete": export_complete,
                  "failure_inventory_complete": inventory_complete,
                  "failures": failures, "accepted": accepted,
                  "cpu_stage_wall_seconds": time.perf_counter() - started}
        if not legacy:
            record.update(label_policy=policy, missing_descriptors=missing)
        return record

    # Check inexpensive structure/coverage before computing any descriptors.
    try:
        with rows_path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if type(row) is not dict or set(row) != {"source_index", "smiles"}:
                    raise ValueError("Rows must contain exactly source_index and smiles")
                position = len(rows)
                if (position >= len(expected) or type(row["source_index"]) is not int or
                    row["source_index"] != expected[position]):
                    raise ValueError("Export row coverage/order differs from pinned role")
                rows.append(row)
        if len(rows) != len(expected):
            raise ValueError("Incomplete training SMILES export")
    except (ValueError, TypeError) as error:
        failures.append({"source_index": None, "reason": "input_error:" + str(error)})
        atomic_write(output / "manifest.json", json_bytes(report()))
        raise
    by_index = {row["source_index"]: row for row in rows}
    priority_set = set(priority)
    order = [*priority, *(i for i in expected if i not in priority_set)]
    for index in order:
        try:
            label = encoder.encode(index, by_index[index]["smiles"])
        except Exception as error:
            label = ChemicalLabels(index, "label_error:" + type(error).__name__, None, None)
        labels[index] = label
        if label.status != "ok":
            failures.append({"source_index": index, "reason": label.status})
            # Retain the blocker immediately, even when collecting all failures.
            failure_record = report(export_complete=True)
            atomic_write(output / "manifest.json", json_bytes(failure_record))
            if fail_fast:
                return failure_record
        elif label.descriptor_valid_mask is not None and not label.descriptor_valid_mask.all():
            columns = np.flatnonzero(~label.descriptor_valid_mask).tolist()
            missing.append({"source_index": index, "column_indices": columns,
                            "column_names": [encoder.descriptor_names[c] for c in columns]})
    missing.sort(key=lambda row: row["source_index"])
    result = report(accepted=not failures, export_complete=True, inventory_complete=True)
    if not failures:
        arrays = {"source_indices": np.asarray(expected, dtype=np.int64)}
        for component in encoder.components:
            field = "fingerprint" if component == "fingerprints" else "descriptors"
            arrays[component] = np.stack([getattr(labels[i], field) for i in expected])
        if descriptor_missing_policy == "mask_nonfinite":
            arrays["descriptor_valid_mask"] = np.stack([labels[i].descriptor_valid_mask for i in expected])
        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        atomic_write(output / "labels.npz", buffer.getvalue())
        result["labels_sha256"] = sha256_file(output / "labels.npz")
    atomic_write(output / "manifest.json", json_bytes(result))
    return result


class ChemicalLabelCache:
    def __init__(self, root: Path, manifest_sha256: str, *, expected_role_sha256: str):
        manifest = _pinned_json(root / "manifest.json", manifest_sha256)
        if (manifest.get("schema") not in {"molgap-chemical-cache-v1", "molgap-chemical-cache-v2"} or
            manifest.get("accepted") is not True or manifest.get("failures") != [] or
            manifest.get("role_sha256") != expected_role_sha256 or
            manifest.get("role", {}).get("role") != "train"):
            raise ValueError("Auxiliary cache is not accepted for the expected training role")
        legacy = manifest["schema"] == "molgap-chemical-cache-v1"
        self.label_policy = ({"components": ["descriptors", "fingerprints"],
                              "parse_policy": "strict", "descriptor_missing_policy": "reject"}
                             if legacy else manifest.get("label_policy"))
        if not isinstance(self.label_policy, dict) or set(self.label_policy) != {
                "components", "parse_policy", "descriptor_missing_policy"}:
            raise ValueError("Malformed auxiliary label policy")
        if (type(self.label_policy["components"]) is not list or
            any(type(c) is not str for c in self.label_policy["components"])):
            raise ValueError("Malformed auxiliary components")
        self.components = tuple(self.label_policy["components"])
        if (not self.components or len(set(self.components)) != len(self.components) or
            self.components != tuple(c for c in ("descriptors", "fingerprints") if c in self.components) or
            self.label_policy["parse_policy"] not in {"strict", "pcqm_topology"} or
            self.label_policy["descriptor_missing_policy"] not in {"reject", "mask_nonfinite"} or
            ("descriptors" not in self.components and self.label_policy["descriptor_missing_policy"] != "reject")):
            raise ValueError("Unsupported auxiliary label policy")
        if not legacy and (manifest.get("export_coverage_complete") is not True or
                           manifest.get("failure_inventory_complete") is not True or
                           manifest.get("expected_row_count") != manifest["row_count"]):
            raise ValueError("Incomplete auxiliary cache coverage")
        masked = self.label_policy["descriptor_missing_policy"] == "mask_nonfinite"
        if sha256_file(root / "labels.npz") != manifest["labels_sha256"]:
            raise ValueError("Auxiliary cache content changed")
        with np.load(root / "labels.npz", allow_pickle=False) as arrays:
            keys = {"source_indices", *self.components} | ({"descriptor_valid_mask"} if masked else set())
            if set(arrays.files) != keys:
                raise ValueError("Cache arrays differ from declared components")
            ids = arrays["source_indices"].copy()
            self.descriptors = arrays["descriptors"].copy() if "descriptors" in self.components else None
            self.fingerprints = arrays["fingerprints"].copy() if "fingerprints" in self.components else None
            self.descriptor_valid_mask = arrays["descriptor_valid_mask"].copy() if masked else None
        n = manifest["row_count"]
        if (ids.dtype != np.int64 or ids.tolist() != manifest["role"]["source_indices"] or
            len(set(ids.tolist())) != n or ids.shape != (n,) or
            (self.descriptors is not None and (self.descriptors.shape != (n, 200) or
             self.descriptors.dtype != np.float32 or not np.isfinite(self.descriptors).all())) or
            (self.fingerprints is not None and (self.fingerprints.shape != (n, 512) or
             self.fingerprints.dtype != np.uint8 or not np.isin(self.fingerprints, [0, 1]).all()))):
            raise ValueError("Malformed auxiliary arrays or row alignment")
        if masked:
            mask = self.descriptor_valid_mask
            if (mask.shape != (n, 200) or mask.dtype != np.bool_ or
                not (self.descriptors[~mask] == 0).all()):
                raise ValueError("Malformed descriptor mask or missing-value storage")
            names = manifest["encoder"]["descriptor_names"]
            if len(names) != 200 or len(set(names)) != 200:
                raise ValueError("Malformed descriptor names")
            missing = [{"source_index": int(ids[row]), "column_indices": np.flatnonzero(~mask[row]).tolist(),
                        "column_names": [names[c] for c in np.flatnonzero(~mask[row])]} for row in range(n) if not mask[row].all()]
            missing.sort(key=lambda row: row["source_index"])
            if missing != manifest.get("missing_descriptors"):
                raise ValueError("Descriptor mask differs from recorded missingness")
        elif not legacy and manifest.get("missing_descriptors") != []:
            raise ValueError("Unexpected descriptor missingness without a mask policy")
        if not legacy:
            for key, value in self.label_policy.items():
                if manifest.get("encoder", {}).get(key) != value:
                    raise ValueError("Encoder and cache label policies disagree")
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
        if self.descriptors is not None:
            batch.chemical_descriptors = torch.from_numpy(self.descriptors[positions].copy())
        if self.fingerprints is not None:
            batch.chemical_fingerprint = torch.from_numpy(self.fingerprints[positions].copy())
        if self.descriptor_valid_mask is not None:
            batch.chemical_descriptor_valid_mask = torch.from_numpy(self.descriptor_valid_mask[positions].copy())
        return batch
