"""Read-only cohort and pure-2D sidecar guards, never graph preprocessing."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .common.wire import canonical_fingerprint
from .cross_scale_plan import CohortIdentity, FEATURE_FORMATS, digest, exact
from .research_memory.paths import repo_local_path
from .research_memory.trace import file_digest


def index_digest(values):
    import torch
    from .experiment_family_workflow import tensor_digest
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("Indices must be nonnegative integers")
    return tensor_digest(torch.tensor(values, dtype=torch.int64), role="source_idx")


def target_digest(values):
    import torch
    from .experiment_family_workflow import tensor_digest
    from math import isfinite
    if any(type(value) not in (int, float) or not isfinite(value) for value in values):
        raise ValueError("Targets must be finite eV values")
    return tensor_digest(torch.tensor(values, dtype=torch.float64), role="target")


def validate_members(cohort, *, row_indices, source_indices, targets):
    if type(cohort) is not CohortIdentity:
        raise TypeError("Typed cohort required")
    if not (len(row_indices) == len(source_indices) == len(targets) == cohort.rows):
        raise ValueError("Cohort member count mismatch")
    if source_indices != sorted(set(source_indices)) or row_indices != list(range(cohort.rows)):
        raise ValueError("Cohort reordering or duplicate members")
    if not source_indices or source_indices[-1] >= cohort.official_train_rows:
        raise ValueError("ROLE_BLOCKED: member outside official training boundary")
    if (index_digest(row_indices) != cohort.row_index_sha256 or
            index_digest(source_indices) != cohort.source_index_sha256 or
            target_digest(targets) != cohort.target_sha256):
        raise ValueError("Cohort row/source/target hash mismatch")
    return dict(zip(source_indices, targets))


def validate_nested_cohorts(small, large, development, *, memberships):
    records = [validate_members(c, **memberships[c.name]) for c in (small, large, development)]
    a, b, dev = records
    if not set(a) <= set(b):
        raise ValueError("100K is not a subset of 500K")
    if any(a[i] != b[i] for i in a):
        raise ValueError("Nested target encoding/values differ")
    if set(dev) & (set(a) | set(b)):
        raise ValueError("ROLE_BLOCKED: development overlaps training")
    if development.role != "development" or small.role != "train" or large.role != "train":
        raise ValueError("ROLE_BLOCKED: wrong cohort role")
    if len({(c.dataset, c.target, c.unit, c.target_encoding, c.official_train_rows) for c in (small, large, development)}) != 1:
        raise ValueError("Dataset/target identities differ")
    return {"status": "PAIRED_COHORTS_VERIFIED", "nested": True,
            "development_usage": "selection-used", "independent_holdout": False,
            "identity": canonical_fingerprint({c.name: asdict(c) for c in (small, large, development)})}


_COHORT_TOKEN = object()


class CohortProof:
    """In-process receipt of actual membership validation, not an asserted flag.

    Recreate through the validator after remounting; a JSON status/hash alone
    cannot manufacture a receipt. Authority/history pins remain part of its
    identity. Changing any compared identity makes the receipt stale.
    """
    __slots__ = ("_cohorts", "_identity")

    def __init__(self, cohorts, *, _token=None):
        if _token is not _COHORT_TOKEN:
            raise TypeError("CohortProof must come from verify_comparison_cohorts")
        object.__setattr__(self, "_cohorts", tuple(cohorts))
        object.__setattr__(self, "_identity", canonical_fingerprint({"cohorts": [asdict(c) for c in cohorts]}))

    def __setattr__(self, name, value):
        raise AttributeError("CohortProof is immutable")

    def summary(self):
        return {"status": "PAIRED_COHORTS_VERIFIED", "identity": self._identity,
                "cohorts": [asdict(c) for c in self._cohorts],
                "nested": True, "train_development_disjoint": True}

    def verify(self, left, right):
        small, large, dev = self._cohorts
        if canonical_fingerprint({"cohorts": [asdict(c) for c in self._cohorts]}) != self._identity:
            raise ValueError("Stale cohort proof")
        for identity, train in ((left, small), (right, large)):
            expected = cohort_comparison_identity(train, dev)
            if any(identity.get(k) != v or type(identity.get(k)) is not type(v)
                   for k, v in expected.items()):
                raise ValueError("Cohort proof does not bind the compared data/role/order/authority identity")


def cohort_comparison_identity(train, development):
    """Single translation of validated cohort pins into V5 comparison fields."""
    return {"dataset_identity": train.dataset, "data_role_identity": train.role_identity,
            "row_membership_identity": train.source_index_sha256,
            "row_order_identity": train.row_index_sha256,
            "train_manifest_sha256": train.manifest_sha256,
            "train_target_sha256": train.target_sha256,
            "train_topology_sha256": train.topology_sha256,
            "train_authority_sha256": train.authority_sha256,
            "train_role_history_sha256": train.role_history_sha256,
            "evaluation_role_identity": development.role_identity,
            "selection_role_identity": development.role_identity,
            "evaluation_row_sha256": development.source_index_sha256,
            "evaluation_target_sha256": development.target_sha256,
            "evaluation_row_count": development.rows,
            "evaluation_manifest_sha256": development.manifest_sha256,
            "evaluation_authority_sha256": development.authority_sha256,
            "evaluation_role_history_sha256": development.role_history_sha256}


def verify_comparison_cohorts(small, large, development, *, memberships):
    validate_nested_cohorts(small, large, development, memberships=memberships)
    return CohortProof((small, large, development), _token=_COHORT_TOKEN)


def audit_fixed_manifests(small, large):
    """Metadata-only historical audit: no inferred observed tensor digests."""
    from .pcqm_fixed_records import graph_records
    records = []
    for manifest, sha in (small, large):
        digest(sha)
        if any(manifest.get(k) is not False for k in
               ("official_validation_role_read", "test_dev_role_read", "test_challenge_role_read")):
            raise ValueError("Protected source role flag")
        if manifest.get("source", {}).get("external_data_used") is not False:
            raise ValueError("External source not allowed")
        records.append(graph_records(manifest, sha))
    small_train = [r for r in records[0] if r["role"] == "train"]
    large_train = [r for r in records[1] if r["role"] == "train"]
    keys = ("source_idx_min", "source_idx_max", "rows", "sha256")
    nested = [tuple(r[k] for k in keys) for r in small_train] == [tuple(r[k] for k in keys) for r in large_train[:len(small_train)]]
    sr, lr = small[0]["roles"], large[0]["roles"]
    same_dev = sr["development"] == lr["development"]
    a, b = sr["development"], lr["train"]
    overlap = max(a["source_idx_start"], b["source_idx_start"]) < min(a["source_idx_stop"], b["source_idx_stop"])
    return {"status": "ROLE_BLOCKED" if not nested or not same_dev or overlap else "METADATA_ONLY",
            "training_shards_nested_bytewise": nested, "development_identities_equal": same_dev,
            "small_development_overlaps_large_train": overlap,
            "source_index_tensor_sha256": None, "target_tensor_sha256": None,
            "graph_tensors_opened": False, "role_authorized": False}


def verify_sidecar(root, binding, *, expected_sources, expected_topology_sha256, max_bytes):
    """Verify existing raw bytes plus frozen metadata; do not deserialize/rebuild."""
    exact(binding, {"path", "sha256", "format", "approval_sha256", "source_sha256", "topology_sha256",
                    "fields", "pure_2d", "edge_convention", "source_indices"}, "sidecar")
    for key in ("sha256", "approval_sha256", "source_sha256", "topology_sha256"):
        digest(binding[key], key)
    allowed = FEATURE_FORMATS
    if (binding["format"] not in allowed or type(binding["fields"]) is not list or
            not binding["fields"] or not set(binding["fields"]) <= allowed[binding["format"]] or
            binding["pure_2d"] is not True or binding["edge_convention"] not in {"directed", "bidirected"}):
        raise ValueError("Unapproved feature version, 3D field or edge convention")
    if (binding["source_indices"] != expected_sources or
            binding["source_sha256"] != index_digest(expected_sources) or
            binding["topology_sha256"] != expected_topology_sha256):
        raise ValueError("Sidecar source order/topology mismatch")
    path = repo_local_path(Path(root), binding["path"])
    if type(max_bytes) is not int or max_bytes < 1 or path.stat().st_size > max_bytes:
        raise ValueError("Sidecar size guard")
    if file_digest(path) != binding["sha256"]:
        raise ValueError("Sidecar bytes differ")
    return {"status": "BYTES_VERIFIED_ONLY", "bytes": path.stat().st_size,
            "preprocessing_cost": None, "cache_read_cost": None, "cache_write_cost": None}


def validate_graph_fixture(edge_index, batch, valid, *, edge_convention="bidirected", attention_mask=None):
    """CPU fixture structural checks; no molecular model or scientific cache IO."""
    import torch
    if (edge_convention not in {"directed", "bidirected"} or edge_index.dtype != torch.int64 or
            edge_index.ndim != 2 or edge_index.shape[0] != 2 or batch.dtype != torch.int64 or
            batch.ndim != 1 or valid.dtype != torch.bool or valid.shape != batch.shape or
            any(t.device.type != "cpu" for t in (edge_index, batch, valid))):
        raise ValueError("Invalid CPU topology/mask schema")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= len(batch)):
        raise ValueError("Out-of-range edge")
    pairs = set(map(tuple, edge_index.T.tolist()))
    for i, j in pairs:
        if not valid[i] or not valid[j] or batch[i] != batch[j]:
            raise ValueError("Padding or cross-batch edge")
        if edge_convention == "bidirected" and (j, i) not in pairs:
            raise ValueError("Missing reverse edge")
    if attention_mask is not None:
        if attention_mask.device.type != "cpu" or attention_mask.dtype != torch.bool or attention_mask.shape != (len(batch), len(batch)):
            raise ValueError("Invalid attention mask")
        allowed = valid[:, None] & valid[None, :] & (batch[:, None] == batch[None, :])
        if bool((attention_mask & ~allowed).any()):
            raise ValueError("Attention mask connects padding or different graphs")
    return {"nodes": int(valid.sum()), "edges": len(pairs), "status": "CPU_FIXTURE_VERIFIED"}
