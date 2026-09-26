"""Regression for exact producer/consumer evidence keys; no model execution."""
import ast
from pathlib import Path

import pytest

from molgap.k1_relation_study_records import validate_mechanism_evidence

RRWP = "neural_atom_k1_rrwp_pair"
COMMON = {
    "zero_return_projection", "zero_update", "valid_row_assignment_sum_one",
    "padding_assignment_zero", "finite", "different_receiver_features_within_graph",
    "graph0_perturbation_isolated", "receiver_specific_return_features",
    "permutation_equivariant_before_return", "resume_two_step_bitwise_equal",
}
RRWP_KEYS = {"rrwp_batched_graphs_match_independent_expected", "rrwp_isolate_self_transition"}


def evidence(extra=()):
    return {**dict.fromkeys(COMMON | set(extra), True), "largest_train_shape_probe": {"graphs": 128}}


def test_matches_remote_producer_field_names():
    source = Path(__file__).resolve().parents[1] / "src/molgap/k1_relation_resolution.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    check = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_remote_rrwp_checks")
    returned = next(n for n in ast.walk(check) if isinstance(n, ast.Return))
    assert {k.value for k in returned.value.keys} == RRWP_KEYS
    validate_mechanism_evidence(RRWP, evidence(RRWP_KEYS))


@pytest.mark.parametrize("field", sorted(COMMON | RRWP_KEYS))
@pytest.mark.parametrize("value", [False, None, 1, "true"])
def test_missing_or_nonboolean_success_is_rejected(field, value):
    record = evidence(RRWP_KEYS)
    if value is None:
        record.pop(field)
    else:
        record[field] = value
    with pytest.raises(ValueError):
        validate_mechanism_evidence(RRWP, record)


def test_invented_old_alias_is_not_accepted_as_evidence():
    with pytest.raises(ValueError):
        validate_mechanism_evidence(RRWP, evidence({"rrwp_powers_match_independent_expected"}))


def test_other_arms_require_only_their_actual_equations():
    validate_mechanism_evidence("neural_atom_k1_receiver_pair", evidence())
    validate_mechanism_evidence("neural_atom_k1_triplet_aggregate", evidence({"tgt_vector_loop_agreement"}))
    with pytest.raises(ValueError):
        validate_mechanism_evidence("neural_atom_k1_triplet_aggregate", evidence())
