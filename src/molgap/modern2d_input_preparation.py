"""CPU metadata from externally pinned, accepted 100K/Common-Dev exports.

``prepare_qualification_inputs(request, execute=False)`` validates a request;
explicit execution opens the actual selected exports, never a future publication
layout. The caller owns CPU placement and its aggregate allocation budget.
No graph building, feature derivation, model construction or submission occurs.
"""
from dataclasses import asdict
import json
from pathlib import Path

from .common.io import _safe_local, publish_immutable_bytes
from .common.wire import canonical_fingerprint
from .cross_scale_plan import digest, exact
from .data.indexed_inputs import AcceptedIndexedInputPermit, IndexedRoleDataset, _local
from .pcqm_indexed_export import EXPORT_FORMAT, _chunk_paths, _inventory
from .research_memory.trace import file_digest, json_bytes, sync_directory


REQUEST_FORMAT = "molgap-modern2d-qualification-input-request-v1"
METADATA_FORMAT = "molgap-modern2d-qualification-input-metadata-v1"
ROLE_ROWS = {"train_100k": 100000, "common_dev": 50000}
REQUEST_FIELDS = {"format", "export_root", "export_acceptance_sha256",
                  "parent_acceptance_sha256", "output_root", "boundary", "roles", "inventory"}
MAX_METADATA_BYTES = 16 * 1024**2


def _read(path):
    _safe_local(path)
    if not path.is_file() or path.stat().st_size > MAX_METADATA_BYTES:
        raise ValueError("Missing or oversized preparation metadata")
    return json.loads(path.read_bytes())


def _request(request):
    exact(request, REQUEST_FIELDS, "qualification input request")
    if request["format"] != REQUEST_FORMAT:
        raise ValueError("Unknown qualification input request format")
    for key in ("export_acceptance_sha256", "parent_acceptance_sha256"):
        digest(request[key], key)
    exact(request["roles"], ROLE_ROWS, "exact 100K/Common-Dev roles")
    permits = {}
    for role, rows in ROLE_ROWS.items():
        record = exact(request["roles"][role], {"permit", "authority", "role_history", "rows",
                       "source_index_sha256", "logical_content_sha256"}, "external role pins")
        permit = AcceptedIndexedInputPermit(**record["permit"])
        if (permit.role != role or permit.acceptance_sha256 != request["export_acceptance_sha256"] or
                type(record["rows"]) is not int or record["rows"] != rows):
            raise ValueError("ROLE_BLOCKED: exact accepted export role/count required")
        permit.verify(record["authority"], record["role_history"])
        for key in ("source_index_sha256", "logical_content_sha256"):
            digest(record[key], key)
        permits[role] = permit
    if len({permit.version_identity for permit in permits.values()}) != 1:
        raise ValueError("Export role versions disagree")
    inventory = request["inventory"]
    if type(inventory) is not dict or not inventory or len(inventory) > 2 * sum(ROLE_ROWS.values()) + 6:
        raise ValueError("Explicit bounded accepted export inventory required")
    for name, item in inventory.items():
        if (type(name) is not str or not name or "\\" in name or ":" in name or
                name.startswith("/") or any(p in {"", ".", ".."} for p in name.split("/"))):
            raise ValueError("Invalid export inventory path")
        exact(item, {"sha256", "bytes"}, "export inventory item")
        digest(item["sha256"], "inventory bytes")
        if type(item["bytes"]) is not int or item["bytes"] < 1:
            raise ValueError("Explicit positive export byte count required")
    paths = {}
    for key in ("boundary", "export_root", "output_root"):
        if not isinstance(request[key], (str, Path)):
            raise ValueError("Explicit absolute local roots required")
        paths[key] = Path(request[key])
        if not paths[key].is_absolute():
            raise ValueError(key + " must be an explicit absolute local path")
        _safe_local(paths[key])
    boundary, root, output = (paths[k] for k in ("boundary", "export_root", "output_root"))
    if (not boundary.is_dir() or root == boundary or output == boundary or
            not root.is_relative_to(boundary) or not output.is_relative_to(boundary)):
        raise ValueError("Export and metadata roots must be below the caller boundary")
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Metadata must be separate from the immutable export")
    if output.exists():
        raise ValueError("Preparation requires a new metadata output directory")
    return root, output, permits


def _verify_export(root, request, permits):
    """Reuse the export inventory and accepted loader, without opening a parent mount."""
    acceptance_path = _local(root, "acceptance.json")
    if file_digest(acceptance_path) != request["export_acceptance_sha256"]:
        raise ValueError("Externally supplied export acceptance SHA differs")
    acceptance = _read(acceptance_path)
    if (acceptance.get("format") != EXPORT_FORMAT + "-acceptance" or
            acceptance.get("status") != "DATA_ROLE_ACCEPTED" or
            acceptance.get("scope") != "OFFICIAL_TRAIN_CPU_PREPARATION_ONLY" or
            acceptance.get("parent_acceptance_sha256") != request["parent_acceptance_sha256"] or
            acceptance.get("rule_identity") != permits["train_100k"].version_identity or
            acceptance.get("private_required") is not True or
            acceptance.get("inventory") != request["inventory"] or
            set(acceptance.get("roles", {})) != set(ROLE_ROWS) or
            any(acceptance.get(k) is not False for k in ("official_validation_read", "test_roles_read",
                                                       "graphs_rebuilt", "audit_label_statistics_exposed"))):
        raise ValueError("Export acceptance differs from external selected-role pins")
    # Check metadata pins before using any mounted manifest to determine paths.
    for name in ("export_request.json", "census.json", *(f"roles/{r}.json" for r in ROLE_ROWS)):
        pin = request["inventory"].get(name)
        path = _local(root, name)
        if (pin is None or pin["bytes"] > MAX_METADATA_BYTES or
                path.stat().st_size != pin["bytes"] or file_digest(path) != pin["sha256"]):
            raise ValueError("Export metadata differs from external inventory")
    exported_request = _read(_local(root, "export_request.json"))
    if (exported_request.get("format") != EXPORT_FORMAT or
            exported_request.get("roles") != list(ROLE_ROWS) or
            exported_request.get("parent_acceptance_sha256") != request["parent_acceptance_sha256"] or
            canonical_fingerprint(exported_request) != acceptance.get("request_identity")):
        raise ValueError("Export parent/request binding differs")
    allowed = {"export_request.json", "census.json"}
    for role, rows in ROLE_ROWS.items():
        record, permit = request["roles"][role], permits[role]
        expected = {"path": f"roles/{role}.json", "sha256": permit.manifest_sha256,
                    **{k: record[k] for k in ("rows", "source_index_sha256", "logical_content_sha256")}}
        if acceptance["roles"][role] != expected:
            raise ValueError("Export role differs from external metadata pins")
        manifest = _read(_local(root, expected["path"]))
        parent_permit = AcceptedIndexedInputPermit(**exported_request["permits"][role])
        if (parent_permit.role != role or parent_permit.acceptance_sha256 != request["parent_acceptance_sha256"] or
                parent_permit.version_identity != permit.version_identity or
                manifest.get("parent_acceptance_sha256") != request["parent_acceptance_sha256"] or
                manifest.get("parent_manifest_sha256") != parent_permit.manifest_sha256 or
                file_digest(_local(root, expected["path"])) != permit.manifest_sha256):
            raise ValueError("Export lost its original parent/physical manifest binding")
        chunks = manifest.get("graph_chunks")
        if type(chunks) is not list or not 1 <= len(chunks) <= rows:
            raise ValueError("Unbounded selected export chunks")
        allowed.update({expected["path"], f"members/{role}.i64le"})
        count = 0
        for index, chunk in enumerate(chunks):
            payload, receipt = _chunk_paths(role, index)
            pin = request["inventory"].get(payload)
            if (chunk.get("file") != payload or chunk.get("offsets") is not None or
                    type(chunk.get("rows")) is not int or not 1 <= chunk["rows"] <= rows or
                    pin != {"sha256": chunk.get("sha256"), "bytes": chunk.get("bytes")} or
                    not 1 <= pin["bytes"] <= permit.max_chunk_bytes):
                raise ValueError("Export selected physical chunk pin/budget differs")
            count += chunk["rows"]
            allowed.update({payload, receipt})
        if count != rows:
            raise ValueError("Export chunk rows differ from the exact selected role")
    if set(request["inventory"]) != allowed:
        raise ValueError("Inventory is not the exact selected export allowlist")
    _inventory(root, allowed | {"acceptance.json"}, complete=True)
    for name, pin in request["inventory"].items():
        path = _local(root, name)
        if path.stat().st_size != pin["bytes"] or file_digest(path) != pin["sha256"]:
            raise ValueError("Accepted export bytes changed: " + name)
    return acceptance


def prepare_qualification_inputs(request, *, execute=False):
    """Validate or publish CPU metadata, using independently supplied trust pins.

    Request fields are REQUEST_FIELDS. ``roles`` maps exactly train_100k and
    common_dev to {permit, authority, role_history, rows, source_index_sha256,
    logical_content_sha256}; permits bind the NEW exported physical manifests.
    ``inventory`` is the externally frozen acceptance.inventory mapping, excluding
    acceptance.json itself. All three roots are explicit absolute paths.

    Returns status plus path/SHA/byte records for the two immutable JSON outputs.
    The companion metadata contains shared train-only statistics, source_idx
    tensor digests, and both roles' feature bindings for the two typed defaults.
    No dataset is opened through the portable future ``indexed_assets`` layout.
    """
    if type(execute) is not bool:
        raise TypeError("execute must be an explicit bool")
    root, output, permits = _request(request)
    if not execute:
        return {"status": "PREPARE_ONLY_NOT_EXECUTED", "roles": dict(ROLE_ROWS),
                "output_root": str(output), "payloads_read": False, "remote_submission": False}
    _verify_export(root, request, permits)
    import torch
    from .esa_native import ESANativeConfig
    from .experiment_family_workflow import tensor_digest
    from .modern2d_feature_binding import build_indexed_feature_binding
    from .ppgt_author import PPGTAuthorConfig
    from .v4_runtime import sample_std_compat

    binding = {"format": "molgap-indexed-training-input-v1", "data_root": "indexed_assets",
               "scope": "ACCEPTED_INDEXED_INPUT_V1"}
    models = {"ppgt_author_v2": {"family": ["ppgt_small", "2"], "model_config": asdict(PPGTAuthorConfig()),
                                "feature_bindings": {}, "feature_binding_sha256": {}},
              "esa_native_v3": {"family": ["esa_small", "3"], "model_config": asdict(ESANativeConfig()),
                                "feature_bindings": {}, "feature_binding_sha256": {}}}
    roles, training_ids = {}, None
    for role, rows in ROLE_ROWS.items():
        record = request["roles"][role]
        dataset = IndexedRoleDataset(root, permits[role], authority=record["authority"],
                                     role_history=record["role_history"])
        if len(dataset) != rows:
            raise ValueError("Decoded selected role count differs")
        indices = dataset.source_indices
        roles[role] = {"rows": rows, "source_idx_tensor_sha256": tensor_digest(
            torch.tensor(indices, dtype=torch.int64), role="source_idx")}
        if role == "train_100k":
            training_ids = set(indices)
            targets = torch.cat(list(dataset.selected_targets()))
            if targets.device.type != "cpu" or len(targets) != rows or not bool(torch.isfinite(targets).all()):
                raise ValueError("Train normalization requires exact finite CPU selected labels")
            mean, std = float(targets.mean()), float(sample_std_compat(targets).clamp_min(1e-6))
            del targets
        elif training_ids.intersection(indices):
            raise ValueError("Training and Common-Dev memberships overlap")
        for model in models.values():
            feature = build_indexed_feature_binding(dataset, family=model["family"],
                                                    configuration=model["model_config"])
            model["feature_bindings"][role] = feature
            model["feature_binding_sha256"][role] = canonical_fingerprint(feature)
        binding["train" if role == "train_100k" else "development"] = {
            k: record[k] for k in ("permit", "authority", "role_history")}
        del dataset
    # Recheck immutable metadata after decoding; payload identity was verified by
    # the accepted indexed owner, which retains only one selected mapped chunk.
    if file_digest(_local(root, "acceptance.json")) != request["export_acceptance_sha256"]:
        raise ValueError("Export acceptance changed during metadata preparation")
    for role, permit in permits.items():
        if file_digest(_local(root, f"roles/{role}.json")) != permit.manifest_sha256:
            raise ValueError("Export role manifest changed during metadata preparation")
    _safe_local(output)
    output.mkdir(parents=True, exist_ok=False)

    def publish(name, value):
        path = publish_immutable_bytes(_local(output, name), json_bytes(value))
        sync_directory(output)
        return {"path": name, "sha256": file_digest(path), "bytes": path.stat().st_size}

    training_input = publish("training_input.json", binding)
    metadata = {"format": METADATA_FORMAT, "status": "INPUT_METADATA_PREPARED",
                "training_input": training_input, "data_root": "indexed_assets",
                "export_acceptance_sha256": request["export_acceptance_sha256"],
                "parent_acceptance_sha256": request["parent_acceptance_sha256"],
                "inventory_sha256": canonical_fingerprint(request["inventory"]),
                "target_mean_eV": mean, "target_std_eV": std,
                "normalization": {"role": "train_100k", "dtype": "float64", "correction": 1, "std_floor": 1e-6},
                "roles": roles, "models": models, "development_label_statistics_exposed": False,
                "scientific_training_authorized": False, "publication_status": "NOT_PUBLISHED"}
    qualification_inputs = publish("qualification_inputs.json", metadata)
    sync_directory(output.parent)
    return {"status": metadata["status"], "output_root": str(output),
            "training_input": training_input, "qualification_inputs": qualification_inputs,
            "remote_submission": False, "scientific_training_authorized": False}


__all__ = ["REQUEST_FORMAT", "METADATA_FORMAT", "prepare_qualification_inputs"]
