"""Synthetic pooling checks, without constructing a model or opening data."""
import pytest
import torch
import ast
from pathlib import Path

from molgap.gptrans_author_variants import pooled_path_bonds


def test_histogram_pool_matches_explicit_hop_mean_and_gradients():
    tables = [torch.nn.Embedding(n, 4) for n in (5, 6, 2)]
    categories = torch.tensor([[0, 1, 0], [2, 3, 1], [0, 3, 0]])
    counts = torch.cat([torch.bincount(categories[:, i], minlength=t.num_embeddings)
                        for i, t in enumerate(tables)]).unsqueeze(0)
    actual = pooled_path_bonds(counts, torch.tensor([3]), tables)
    expected = sum(table(categories[:, i]) for i, table in enumerate(tables)).mean(0, keepdim=True)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert all(table.weight.grad is not None for table in tables)


def test_path_histograms_fail_closed():
    tables = [torch.nn.Embedding(n, 4) for n in (5, 6, 2)]
    with pytest.raises(ValueError, match="sum to path length"):
        pooled_path_bonds(torch.zeros(1, 13), torch.tensor([2]), tables)
    with pytest.raises(ValueError, match="2..20"):
        pooled_path_bonds(torch.zeros(1, 13), torch.tensor([1]), tables)


def test_author_arms_have_separate_trace_identity_and_no_core_edit():
    from molgap.constants import REPO_ROOT
    from molgap.training_reproducibility import sha256_file
    from molgap.pcqm_gptrans_v4 import EXPECTED_ARCHITECTURE_SHA256, _source_sha256
    root = Path(REPO_ROOT)
    assert _source_sha256(root / "src/molgap/gptrans.py") == EXPECTED_ARCHITECTURE_SHA256
    source = (root / "src/molgap/pcqm_gptrans_v4.py").read_text()
    ast.parse(source)
    assert 'trajectory_id=trajectory_id, run_id=logical_run_id' in source
    assert 'if author_arm and (not v5_audit or not trajectory_id or not logical_run_id)' in source
