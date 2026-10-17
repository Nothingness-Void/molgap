"""Focused CPU-only topology/input-contract checks for Modern2D."""

import pytest
import torch
from torch_geometric.data import Batch, Data

from molgap.family_training_data import DatasetSelection, TrainingData
from molgap.family_training_loaders import FamilyLoaders
from molgap.family_training_recipe import FamilyRecipe, ScheduleSpec
from molgap.gptrans_execution_tuning import ExecutionTuning
from molgap.modern2d_inputs import topology_views, validated_topology_views
from molgap.ogb_features import ATOM_FEATURE_DIMS, BOND_FEATURE_DIMS
from modern2d_fixtures import batch as synthetic_batch
from modern2d_fixtures import model_config


def _graph(nodes, source, role, edges):
    x = torch.stack([
        (torch.arange(nodes, dtype=torch.long) + column) % dimension
        for column, dimension in enumerate(ATOM_FEATURE_DIMS)
    ], dim=1)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_count = edge_index.shape[1]
        edge_attr = torch.stack([
            (torch.arange(edge_count, dtype=torch.long) + column) % dimension
            for column, dimension in enumerate(BOND_FEATURE_DIMS)
        ], dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.long)
    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        # These fields deliberately contain no usable values. The topology
        # validator must not inspect labels or derived RWSE inputs.
        y=torch.tensor([float("nan")]),
        random_walk_pe=torch.full((nodes, 16), float("nan")),
        source_idx=torch.tensor([source], dtype=torch.long),
        role_name=role,
    )


def _validated_batch():
    return Batch.from_data_list([
        _graph(3, 41, "train_100k", [(0, 1), (1, 0), (1, 2)]),
        _graph(2, 109, "common_dev", []),
    ])


def test_legacy_topology_views_remains_synthetic_only_and_new_path_accepts_role_list():
    legacy = synthetic_batch()
    views = topology_views(legacy)
    assert len(views) == 2
    assert [view.source_idx for view in views] == [0, 1]

    role_bound = _validated_batch()
    assert [view.source_idx for view in validated_topology_views(role_bound)] == [41, 109]
    with pytest.raises(ValueError):
        topology_views(role_bound)


def test_validated_topology_views_preserves_exact_tensors_membership_and_ptr():
    data = _validated_batch()
    views = validated_topology_views(data)

    assert data.ptr.tolist() == [0, 3, 5]
    assert torch.equal(views[0].x, data.x[:3])
    assert torch.equal(views[0].edge_index, data.edge_index)
    assert torch.equal(views[0].edge_attr, data.edge_attr)
    assert views[0].source_idx == 41

    assert torch.equal(views[1].x, data.x[3:])
    assert views[1].edge_index.shape == (2, 0)
    assert views[1].edge_attr.shape == (0, 3)
    assert views[1].source_idx == 109


@pytest.mark.parametrize("kind", ["geometry", "cross_graph", "protected", "membership", "ptr", "category"])
def test_validated_topology_views_rejects_invalid_contract_fields(kind):
    data = _validated_batch()
    if kind == "geometry":
        data.pos = torch.zeros((data.num_nodes, 3))
    elif kind == "cross_graph":
        data.edge_index[1, 0] = 3
    elif kind == "protected":
        data.role_name[1] = "independent_audit"
    elif kind == "membership":
        data.batch[-1] = 2
    elif kind == "ptr":
        data.ptr[1] = 2
    else:
        data.x[0, 0] = ATOM_FEATURE_DIMS[0]
    with pytest.raises(ValueError):
        validated_topology_views(data)


@pytest.mark.parametrize("kwargs", [{"max_nodes": 2}, {"max_edges": 2}])
def test_oversized_topology_is_rejected_without_dropping_a_member(kwargs):
    with pytest.raises(ValueError, match="no member may be dropped"):
        validated_topology_views(_validated_batch(), **kwargs)


def test_modern2d_cpu_synthetic_loader_still_refuses_indexed_data():
    selection = DatasetSelection("synthetic/modern2d.json", "a" * 64, False)
    recipe = FamilyRecipe(
        "esa_small", "reference", 42, "b" * 64, selection,
        128, 0, 1, .001, .01, ScheduleSpec("cosine_step", 1),
        "final-live", "torch-pass-randperm", 1, 0., 1.,
        model_config=model_config("esa_small"),
    )
    data = TrainingData(
        train=[object()], development=None, manifest_sha256=selection.manifest_sha256,
        train_rows=128, development_rows=0, train_source_start=None,
        development_source_start=None, target_mean_eV=0., target_std_eV=1.,
        synthetic=True, train_source_indices=tuple(range(128)), indexed=True,
    )
    with pytest.raises(ValueError, match="Indexed data requires"):
        FamilyLoaders(recipe, data, ExecutionTuning(training_workers=0))
