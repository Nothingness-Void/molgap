from copy import deepcopy
import pytest
from molgap.pcqm_fixed_records import graph_records, FIXED_KAGGLE_MANIFESTS


def test_legacy_packed_layout_requires_accepted_exact_identity():
    manifest = {"status": "complete", "format": "molgap-pcqm4mv2-kaggle-fixed-subset-v1",
        "geometry_shards": [{"role": "train", "file": "train/train_shard_0000.pt"}],
        "official_validation_role_read": False, "test_dev_role_read": False, "test_challenge_role_read": False}
    for digest in FIXED_KAGGLE_MANIFESTS:
        assert graph_records(manifest, digest) == manifest["geometry_shards"]
    with pytest.raises(ValueError, match="accepted exact"):
        graph_records(manifest, "0" * 64)
    changed = deepcopy(manifest)
    changed["test_dev_role_read"] = True
    with pytest.raises(ValueError, match="Protected"):
        graph_records(changed, next(iter(FIXED_KAGGLE_MANIFESTS)))


def test_topology_only_layout_is_preserved():
    manifest = {"status": "complete", "format": "molgap-pcqm4mv2-fixed-subset-v1",
                "assets": {"topology": [{"file": "topology/a.pt"}]}}
    assert graph_records(manifest, "0" * 64) == manifest["assets"]["topology"]
