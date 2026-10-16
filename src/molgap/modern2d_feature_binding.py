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
