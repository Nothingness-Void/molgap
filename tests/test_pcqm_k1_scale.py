from pathlib import Path

from molgap.pcqm_k1_scale import (
    FIXED_500K_DATASET,
    FIXED_500K_GEOMETRY_SHA256,
    FIXED_500K_MANIFEST_SHA256,
    ROLE_ROWS_READ,
    SCALE_TRAIN_ROWS,
    SCNET_REFERENCE_CACHE_SHA256,
    TRAIN_SHA256,
    UNSANITIZED_OGB_INDEX_SHA256,
    UNSANITIZED_OGB_SOURCE_INDICES,
    VALIDATION_ROWS,
    VALIDATION_SHA256,
    build_scale_split,
    index_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_scale_split_matches_scnet_roles_exactly():
    result = build_scale_split()

    assert result["train"] == list(range(SCALE_TRAIN_ROWS))
    assert result["validation"] == list(range(SCALE_TRAIN_ROWS, ROLE_ROWS_READ))
    assert len(result["train"]) == 500_000
    assert len(result["validation"]) == 50_000
    assert index_sha256(result["train"]) == TRAIN_SHA256
    assert index_sha256(result["validation"]) == VALIDATION_SHA256
    assert result["train_sha256"] == TRAIN_SHA256
    assert result["validation_sha256"] == VALIDATION_SHA256
    assert (
        result["scnet_reference_cache_aggregate_sha256"]
        == SCNET_REFERENCE_CACHE_SHA256
    )
    assert result["official_validation_role_read"] is False
    assert result["test_dev_role_read"] is False
    assert result["target_labels_read"] is False


def test_scale_split_is_deterministic_and_disjoint():
    first = build_scale_split()
    second = build_scale_split()

    assert first == second
    assert set(first["train"]).isdisjoint(first["validation"])
    assert len(first["train"]) == SCALE_TRAIN_ROWS
    assert len(first["validation"]) == VALIDATION_ROWS


def test_unsanitized_ogb_fallback_contract():
    from molgap.pcqm_k1_scale_cache import _unsanitized_ogb_payload

    payload = _unsanitized_ogb_payload("O[Si]123O[Si]3(O1)(O2)O")
    assert payload["num_nodes"] == 7
    assert payload["node_feat"].shape == (7, 9)
    assert payload["edge_index"].shape == (2, 18)
    assert payload["edge_feat"].shape == (18, 3)
    assert len(UNSANITIZED_OGB_SOURCE_INDICES) == 10
    assert index_sha256(UNSANITIZED_OGB_SOURCE_INDICES) == UNSANITIZED_OGB_INDEX_SHA256


def test_scale_runner_uses_accepted_cross_platform_fixed_dataset():
    import json

    acceptance = json.loads(
        (
            REPO_ROOT
            / "platforms/_records/kaggle/pcqm_fixed_datasets_kaggle2_v1/acceptance.json"
        ).read_text(encoding="utf-8")
    )["datasets"]["500k"]

    assert FIXED_500K_DATASET == acceptance["ref"]
    assert FIXED_500K_MANIFEST_SHA256 == acceptance["manifest_sha256"]
    assert FIXED_500K_GEOMETRY_SHA256 == (
        "30b57ac10ddcd1decb7729b299b9b92fbfdf0fe7de15700b40bd488cd3a9ac4d"
    )
    assert SCNET_REFERENCE_CACHE_SHA256 == acceptance["scnet_aggregate_sha256"]
    assert acceptance["bytewise_identical_to_scnet"] is True
