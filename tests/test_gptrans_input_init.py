"""Asset-only checks for the GPTrans input initialization candidate."""

import torch

from molgap import pcqm_gptrans_v4 as v4
from molgap.gptrans_input_init import (
    INPUT_EMBEDDING_KEYS,
    normal002_input_embedding_state,
)
from molgap.v4_runtime import state_dict_sha256


def _tiny_reference_state():
    return {
        **{name: torch.full((3, 4), float(index + 1))
           for index, name in enumerate(INPUT_EMBEDDING_KEYS)},
        "untouched.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "untouched.buffer": torch.tensor([1, 2], dtype=torch.long),
    }


def test_only_input_embedding_tables_change_without_advancing_global_rng():
    assert len(INPUT_EMBEDDING_KEYS) == len(set(INPUT_EMBEDDING_KEYS)) == 15
    reference = _tiny_reference_state()
    reference_hash = state_dict_sha256(reference)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(123)
        rng_before = torch.random.get_rng_state().clone()
        candidate = normal002_input_embedding_state(
            reference, expected_reference_sha256=reference_hash
        )
        assert torch.equal(torch.random.get_rng_state(), rng_before)
        torch.rand(7)
        repeated = normal002_input_embedding_state(
            reference, expected_reference_sha256=reference_hash
        )
    assert set(candidate) == set(reference)
    assert {name for name in reference if not torch.equal(reference[name], candidate[name])} == set(
        INPUT_EMBEDDING_KEYS
    )
    assert all(torch.equal(reference[name], _tiny_reference_state()[name]) for name in reference)
    assert state_dict_sha256(candidate) == state_dict_sha256(repeated)


def test_initialization_rejects_wrong_reference_hash_and_missing_embedding():
    reference = _tiny_reference_state()
    try:
        normal002_input_embedding_state(reference, expected_reference_sha256="0" * 64)
    except RuntimeError as error:
        assert "reference tensor state" in str(error)
    else:
        raise AssertionError("Changed reference state was accepted")

    reference.pop(INPUT_EMBEDDING_KEYS[-1])
    try:
        normal002_input_embedding_state(
            reference, expected_reference_sha256=state_dict_sha256(reference)
        )
    except RuntimeError as error:
        assert "Missing GPTrans input embedding" in str(error)
    else:
        raise AssertionError("Incomplete GPTrans embeddings were accepted")


def test_candidate_full_initial_state_hash_is_required(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    candidate_hash = "a" * 64
    monkeypatch.setattr(v4, "EXPECTED_PARAMETERS", 1)
    monkeypatch.setattr(v4, "EXPECTED_ARCHITECTURE_SHA256", "core")
    monkeypatch.setattr(v4, "_source_sha256", lambda _path: "core")
    monkeypatch.setattr(v4, "_state_sha256", lambda _model: candidate_hash)
    assert v4._verify_model_identity(
        model, variant=v4.INPUT_EMBEDDING_INIT_VARIANT,
        candidate_initial_sha256=candidate_hash,
    ) == (1, "core")
    for expected in (None, "b" * 64, v4.EXPECTED_INITIAL_MODEL_SHA256):
        try:
            v4._verify_model_identity(
                model, variant=v4.INPUT_EMBEDDING_INIT_VARIANT,
                candidate_initial_sha256=expected,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("Candidate state without its exact pinned hash was accepted")
    try:
        v4._verify_model_identity(model)
    except RuntimeError as error:
        assert "initialization changed" in str(error)
    else:
        raise AssertionError("Candidate state was accepted as the frozen reference")


def test_same_device_latency_gate_uses_block_medians_and_five_percent_cap():
    passing = v4._latency_summary({
        "reference": [1.0, 1.2, 1.0, 1.0],
        "candidate": [1.04, 1.04, 1.04, 2.0],
    })
    assert passing["reference_median_seconds"] == 1.0
    assert passing["candidate_median_seconds"] == 1.04
    assert passing["accepted"] is True
    failing = v4._latency_summary({
        "reference": [1.0, 1.0, 1.0, 1.0],
        "candidate": [1.06, 1.06, 1.06, 1.06],
    })
    assert failing["accepted"] is False
    try:
        v4._latency_summary({"reference": [0.0], "candidate": [0.0]})
    except RuntimeError as error:
        assert "non-finite or non-positive" in str(error)
    else:
        raise AssertionError("Zero-time latency block was accepted")
