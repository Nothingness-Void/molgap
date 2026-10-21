"""Metadata-only CPU feature proof, separate from tensor-only model imports."""
from .common.wire import canonical_fingerprint
from .cross_scale_plan import digest, exact

BINDING_FIELDS = {"base_dataset_manifest_sha", "source_row_sha", "source_topology_sha",
                  "feature_schema_version", "transform_code_sha", "transform_config_sha",
                  "derived_feature_sha", "role_history_sha", "role", "scope"}


def validate_feature_binding(binding, expected):
    """An exact metadata proof, not a data loader or real-role release."""
    exact(binding, BINDING_FIELDS, "Modern2D feature binding")
    if binding != expected:
        raise ValueError("Feature binding differs from frozen dataset/source/order/transform")
    if binding["scope"] != "CPU_SYNTHETIC_ONLY" or binding["role"] != "synthetic":
        raise ValueError("Real/protected role is not released for this prototype")
    if binding["feature_schema_version"] not in {
            "ogb-atom9-bond3-rrwp-spe-v1", "ogb-atom9-bond3-directed-edge-mask-v1"}:
        raise ValueError("Unreviewed Modern2D feature schema")
    for k, v in binding.items():
        if k.endswith("_sha") and (k != "derived_feature_sha" or v is not None):
            digest(v, k)
    return canonical_fingerprint(binding)


def _indexed_capability(family):
    from .experiment_catalog import FAMILIES
    from .experiment_execution import (MODERN2D_PREVIEW_PROFILE,
        MODERN2D_SCIENTIFIC_INPUT_ABI)
    key = tuple(family)
    contract = FAMILIES.get(key)
    entry = contract.execution if contract is not None else None
    if entry is None or entry.scientific_input_abi != MODERN2D_SCIENTIFIC_INPUT_ABI:
        raise ValueError("Unreviewed indexed feature family/version")
    preview = tuple(entry.model_interface or key)
    preview_contract = FAMILIES.get(preview)
    preview_entry = preview_contract.execution if preview_contract is not None else None
    if (preview_entry is None or preview_entry.scientific_input_abi != MODERN2D_SCIENTIFIC_INPUT_ABI
            or preview_entry.artifact_adapter != MODERN2D_PREVIEW_PROFILE
            or preview_entry.feature_module is None
            or not preview_entry.feature_config_fields):
        raise ValueError("Modern2D family lacks a reviewed indexed input capability")
    return preview, preview_contract, preview_entry


def indexed_transform_identity(family, *, feature_only=False):
    """Pin the actual versioned transform closure, not just a schema string."""
    from pathlib import Path
    from .v4_runtime import normalized_source_sha256
    family, contract, entry = _indexed_capability(family)
    root = Path(__file__).resolve().parents[2]
    files = {"src/" + contract.source_module.replace(".", "/") + ".py",
             "src/molgap/ogb_features.py", *contract.execution.source_files}
    if feature_only:
        # Model weights/head code remain bound by the Spec, independently of
        # label-free topology derivation. Legacy bindings keep their old closure.
        files = {"src/molgap/modern2d_inputs.py", "src/molgap/ogb_features.py",
                 "src/" + entry.feature_module.replace(".", "/") + ".py",
                 *entry.feature_source_files}
    return canonical_fingerprint({name: normalized_source_sha256(root / name) for name in sorted(files)})


def transform_configuration(family, configuration):
    """Only values consumed by topology derivation, not encoder/readout width."""
    from dataclasses import asdict
    _, _, entry = _indexed_capability(family)
    values = dict(configuration) if type(configuration) is dict else asdict(configuration)
    return {key: values[key] for key in entry.feature_config_fields}


def build_indexed_feature_binding(dataset, *, family, configuration, feature_only=False):
    """Freeze on-the-fly derivation from an already verified indexed view.

    This records observed input/source/config identity; it does not authorize a
    role, materialize features, publish a dataset or certify GPU equivalence.
    """
    from dataclasses import asdict
    from importlib import import_module
    from .data.indexed_inputs import AcceptedIndexedInputPermit, IndexedInputPermit, IndexedRoleDataset
    family = tuple(family)
    preview, contract, _ = _indexed_capability(family)
    if (type(dataset) is not IndexedRoleDataset or
            type(dataset.permit) not in (IndexedInputPermit, AcceptedIndexedInputPermit)):
        raise ValueError("Verified indexed view and explicit new model version required")
    config_type = import_module(contract.source_module).CONFIG_TYPE
    values = dict(configuration) if type(configuration) is dict else asdict(configuration)
    if "layer_types" in values:
        values["layer_types"] = tuple(values["layer_types"])
    config = config_type(**values)
    permit, manifest = dataset.permit, dataset.manifest
    from .data.indexed_inputs import _local
    from .research_memory.trace import file_digest
    import json
    manifest_path = _local(dataset.root, "roles/" + permit.role + ".json")
    if (file_digest(manifest_path) != permit.manifest_sha256 or
            json.loads(manifest_path.read_bytes()) != manifest):
        raise ValueError("Indexed feature input manifest changed after role verification")
    binding = {
        "base_dataset_manifest_sha": permit.manifest_sha256,
        "source_row_sha": manifest["source_index_sha256"],
        "source_topology_sha": canonical_fingerprint({"chunks": [c["topology_sha256"] for c in manifest["graph_chunks"]]}),
        "feature_schema_version": contract.feature_schema,
        "transform_code_sha": indexed_transform_identity(family, feature_only=feature_only),
        "transform_config_sha": canonical_fingerprint(transform_configuration(family, config) if feature_only else asdict(config)),
        "derived_feature_sha": None,
        "role_history_sha": permit.role_history_sha256,
        "role": permit.role, "scope": permit.scope,
    }
    validate_indexed_feature_binding(binding, binding)
    return binding


def validate_indexed_feature_binding(binding, expected):
    """Verify a separately frozen derivation identity; no synthetic promotion."""
    from .experiment_catalog import FAMILIES
    from .experiment_execution import MODERN2D_SCIENTIFIC_INPUT_ABI, MODERN2D_PREVIEW_PROFILE
    exact(binding, BINDING_FIELDS, "indexed Modern2D feature binding")
    if binding != expected:
        raise ValueError("Indexed feature binding differs from frozen input/source/config/history")
    if (binding["role"] not in {"train_100k", "train_500k", "common_dev"} or
            binding["scope"] not in {"CPU_SYNTHETIC_ONLY", "ACCEPTED_INDEXED_INPUT_V1"} or
            binding["feature_schema_version"] not in {
                contract.feature_schema for contract in FAMILIES.values()
                if contract.execution is not None
                and contract.execution.scientific_input_abi == MODERN2D_SCIENTIFIC_INPUT_ABI
                and contract.execution.artifact_adapter == MODERN2D_PREVIEW_PROFILE} or
            binding["derived_feature_sha"] is not None):
        raise ValueError("Unreviewed indexed role/schema or unverified materialized feature cache")
    for key, value in binding.items():
        if key.endswith("_sha") and key != "derived_feature_sha":
            digest(value, key)
    return canonical_fingerprint(binding)
