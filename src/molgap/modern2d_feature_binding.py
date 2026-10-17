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


def indexed_transform_identity(family):
    """Pin the actual versioned transform closure, not just a schema string."""
    from pathlib import Path
    from .experiment_catalog import FAMILIES
    from .experiment_execution import MODERN2D_PREVIEW_INTERFACES
    from .v4_runtime import normalized_source_sha256
    family = tuple(family)
    if family not in MODERN2D_PREVIEW_INTERFACES:
        raise ValueError("Unreviewed indexed feature family/version")
    contract = FAMILIES[family]
    root = Path(__file__).resolve().parents[2]
    files = {"src/" + contract.source_module.replace(".", "/") + ".py",
             "src/molgap/ogb_features.py", *contract.execution.source_files}
    return canonical_fingerprint({name: normalized_source_sha256(root / name) for name in sorted(files)})


def build_indexed_feature_binding(dataset, *, family, configuration):
    """Freeze on-the-fly derivation from an already verified indexed view.

    This records observed input/source/config identity; it does not authorize a
    role, materialize features, publish a dataset or certify GPU equivalence.
    """
    from dataclasses import asdict
    from importlib import import_module
    from .data.indexed_inputs import AcceptedIndexedInputPermit, IndexedInputPermit, IndexedRoleDataset
    from .experiment_catalog import FAMILIES
    from .experiment_execution import MODERN2D_PREVIEW_INTERFACES
    family = tuple(family)
    if (family not in MODERN2D_PREVIEW_INTERFACES or type(dataset) is not IndexedRoleDataset or
            type(dataset.permit) not in (IndexedInputPermit, AcceptedIndexedInputPermit)):
        raise ValueError("Verified indexed view and explicit new model version required")
    config_type = import_module(FAMILIES[family].source_module).CONFIG_TYPE
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
        "source_topology_sha": canonical_fingerprint([c["topology_sha256"] for c in manifest["graph_chunks"]]),
        "feature_schema_version": FAMILIES[family].feature_schema,
        "transform_code_sha": indexed_transform_identity(family),
        "transform_config_sha": canonical_fingerprint(asdict(config)),
        "derived_feature_sha": None,
        "role_history_sha": permit.role_history_sha256,
        "role": permit.role, "scope": permit.scope,
    }
    validate_indexed_feature_binding(binding, binding)
    return binding


def validate_indexed_feature_binding(binding, expected):
    """Verify a separately frozen derivation identity; no synthetic promotion."""
    from .experiment_execution import MODERN2D_PREVIEW_INTERFACES
    exact(binding, BINDING_FIELDS, "indexed Modern2D feature binding")
    if binding != expected:
        raise ValueError("Indexed feature binding differs from frozen input/source/config/history")
    if (binding["role"] not in {"train_100k", "train_500k", "common_dev"} or
            binding["scope"] not in {"CPU_SYNTHETIC_ONLY", "ACCEPTED_INDEXED_INPUT_V1"} or
            binding["feature_schema_version"] not in {v[1] for v in MODERN2D_PREVIEW_INTERFACES.values()} or
            binding["derived_feature_sha"] is not None):
        raise ValueError("Unreviewed indexed role/schema or unverified materialized feature cache")
    for key, value in binding.items():
        if key.endswith("_sha") and key != "derived_feature_sha":
            digest(value, key)
    return canonical_fingerprint(binding)
