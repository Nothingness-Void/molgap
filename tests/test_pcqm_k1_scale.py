import numpy as np

from molgap.pcqm_gap_data import fixed_screen_split
from molgap.pcqm_k1_scale import (
    BASE_TRAIN_SHA256,
    SCALE_TRAIN_ROWS,
    VALIDATION_SHA256,
    build_scale_split,
    index_sha256,
)
from molgap.pcqm_shadow import SHADOW_SPLIT_SEED, frozen_shadow_split


def test_scale_split_preserves_roles_and_excludes_shadow():
    base = fixed_screen_split()
    base = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in base.items()}
    used = set(base["train"]) | set(base["validation"])
    shadow = frozen_shadow_split(used)
    shadow_payload = {
        "effective_shadow": shadow["shadow"].tolist(),
        "reserve": shadow["reserve"].tolist(),
        "split_seed": SHADOW_SPLIT_SEED,
    }

    result = build_scale_split(base, shadow_payload)
    train = np.asarray(result["train"], dtype=np.int64)
    validation = np.asarray(result["validation"], dtype=np.int64)
    effective_shadow = np.asarray(shadow_payload["effective_shadow"], dtype=np.int64)

    assert train.size == SCALE_TRAIN_ROWS
    assert set(base["train"]).issubset(set(train.tolist()))
    assert np.array_equal(validation, np.asarray(base["validation"], dtype=np.int64))
    assert np.intersect1d(train, validation).size == 0
    assert np.intersect1d(train, effective_shadow).size == 0
    assert result["base_train_sha256"] == BASE_TRAIN_SHA256
    assert result["validation_sha256"] == VALIDATION_SHA256
    assert index_sha256(train) == result["train_sha256"]
    assert result["official_validation_role_read"] is False
    assert result["test_dev_role_read"] is False
    assert result["target_labels_read"] is False


def test_scale_split_is_deterministic():
    base = fixed_screen_split()
    base = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in base.items()}
    used = set(base["train"]) | set(base["validation"])
    shadow = frozen_shadow_split(used)
    shadow_payload = {
        "effective_shadow": shadow["shadow"].tolist(),
        "reserve": shadow["reserve"].tolist(),
    }
    first = build_scale_split(base, shadow_payload)
    second = build_scale_split(base, shadow_payload)
    assert first["train_sha256"] == second["train_sha256"]
    assert first["added_train_sha256"] == second["added_train_sha256"]
