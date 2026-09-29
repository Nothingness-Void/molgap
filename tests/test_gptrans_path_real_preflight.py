"""Cheap source-level checks; never load the local 100K graph cache."""
from molgap.gptrans_path_real_preflight import graph_path_counts, _sample_positions


def two_way(source, target, bond):
    return [(source, target, bond), (target, source, bond)]


def test_path_signatures_discriminate_same_distance_pairs():
    edges = (
        two_way(0, 1, (0, 0, 0))
        + two_way(1, 2, (0, 0, 0))
        + two_way(2, 3, (1, 0, 0))
    )
    result = graph_path_counts(4, edges)
    assert result["connected_nonbond_pairs"] == 3
    assert result["capped_nonbond_pairs"] == 3
    assert result["nontrivial_bond_pairs"] == 2
    assert result["same_distance_multiple_signature_pairs"] == 2
    assert result["distances_with_multiple_signatures"] == 1


def test_disconnected_and_identical_paths_do_not_fake_discrimination():
    edges = two_way(0, 1, (0, 0, 0)) + two_way(1, 2, (0, 0, 0))
    result = graph_path_counts(4, edges)
    assert result["connected_nonbond_pairs"] == 1
    assert result["same_distance_multiple_signature_pairs"] == 0


def test_invalid_asymmetric_graph_fails_closed():
    try:
        graph_path_counts(2, [(0, 1, (0, 0, 0))])
    except ValueError as error:
        assert "asymmetric" in str(error)
    else:
        raise AssertionError("asymmetric graph was accepted")


def test_sample_rows_are_deterministic_and_unique():
    values = _sample_positions(50_000)
    assert len(values) == 256 and len(set(values)) == 256
    assert values[0] == 0 and values[-1] == 49_804
