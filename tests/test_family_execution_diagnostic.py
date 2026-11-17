"""Synthetic diagnostic bookkeeping; these fixtures grant no native release."""
from copy import deepcopy
import pytest
import torch

from molgap.family_execution_diagnostic import compare_trials, synthetic_graphs, tree_digest


def test_generated_graphs_are_finite_geometry_free_and_have_stable_identity():
    settings = {"rows": 4, "minimum_nodes": 2, "maximum_nodes": 4}
    left, right = synthetic_graphs(settings), synthetic_graphs(settings)
    assert [g.num_nodes for g in left] == [2, 3, 4, 2]
    for a, b in zip(left, right):
        assert "pos" not in a and a.edge_attr.shape[1] == 3
        assert a.x.dtype == torch.long and a.random_walk_pe.shape == (a.num_nodes, 16)
        assert tree_digest(a.to_dict()) == tree_digest(b.to_dict())


def test_tree_digest_binds_tensor_values_dtype_and_primitive_structure():
    state = {"model": torch.ones(2), "optimizer": {1: {"step": 1.}}, "rng": (1, 2)}
    original = tree_digest(state)
    assert tree_digest(deepcopy(state)) == original
    for changed in ({**state, "model": torch.zeros(2)}, {**state, "model": torch.ones(2).double()},
                    {**state, "rng": [1, 2]}, {**state, "optimizer": {1: {"step": 2.}}}):
        assert tree_digest(changed) != original


def fixture():
    config = {"families": ["neural_atom_k1"], "repeats": 2, "steps": 1,
        "sources": {"reference": {"commit": "old"}, "candidate": {"commit": "new"}}}
    rows = [{"family": "neural_atom_k1", "repeat": repeat, "implementation": name,
        "initialization_sha256": "same", "steps": 1, "samples": 128, "runtime": {"torch": "2.7.1"},
        "status": "COMPLETE", "source_commit": config["sources"][name]["commit"],
        "formal_training_executed": False, "molecular_roles_consumed": False,
        **{mode: {"seconds": value, "endpoint_sha256": "same"} for mode in ("compute", "session")}}
        for repeat in range(2) for name, value in (("reference", 2.), ("candidate", 1.))]
    return config, rows


def test_matched_repeats_keep_paired_ranges_without_claiming_scientific_quality():
    config, rows = fixture()
    result = compare_trials(rows, config)
    assert len(result["paired_summaries"]) == 2
    assert all(row["paired_time_reduction_range"] == [.5, .5] for row in result["paired_summaries"])


@pytest.mark.parametrize("bad", ["missing", "duplicate", "endpoint", "both_mode_mismatch", "source", "samples", "scope", "zero", "nan"])
def test_invalid_or_unpaired_measurements_fail_closed(bad):
    config, rows = fixture()
    if bad == "missing":
        rows.pop()
    elif bad == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif bad == "endpoint":
        rows[1]["session"]["endpoint_sha256"] = "different"
    elif bad == "both_mode_mismatch":
        for row in rows:
            row["session"]["endpoint_sha256"] = "different"
    elif bad == "source":
        rows[1]["source_commit"] = "unbound"
    elif bad == "samples":
        rows[1]["samples"] = 127
    elif bad == "scope":
        rows[1]["molecular_roles_consumed"] = True
    else:
        rows[1]["session"]["seconds"] = 0. if bad == "zero" else float("nan")
    with pytest.raises(ValueError):
        compare_trials(rows, config)
