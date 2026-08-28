from __future__ import annotations

import gzip
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Batch

from molgap.gps import (
    CategoricalConcatFeatureEncoder,
    CategoricalEdgeStateStructuralGPSWrapper,
    CategoricalRadicalContextEdgeStateStructuralGPSWrapper,
)
from molgap.ogb_features import (
    ATOM_FEATURE_DIMS,
    BOND_FEATURE_DIMS,
    atom_to_ogb_feature_vector,
    bond_to_ogb_feature_vector,
)
from molgap.pair_gps_2d import CategoricalPairGPS2DWrapper
from molgap.pcqm_feature_screen import (
    FeatureScreenConfig,
    _model,
    _warmup_cosine_factor,
    _graphs_from_screen_row,
    accept_feature_screen_graphs,
    build_feature_screen_graph_shard,
    prepare_feature_screen_rows,
)


def _archive(path: Path) -> None:
    smiles = [
        "C",
        "CC",
        "CCC",
        "CCCC",
        "CCCCC",
        "CO",
        "CN",
        "C=C",
        "C#N",
        "[CH2]CCC[CH2]",
        "O",
        "N",
        "F",
        "Cl",
    ]
    table = pd.DataFrame({
        "idx": np.arange(len(smiles)),
        "smiles": smiles,
        "homolumogap": np.linspace(1.0, 8.0, len(smiles)),
    })
    split_buffer = io.BytesIO()
    torch.save(
        {
            "train": torch.arange(10),
            "valid": torch.tensor([10, 11]),
            "test-dev": torch.tensor([12]),
            "test-challenge": torch.tensor([13]),
        },
        split_buffer,
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "pcqm4m-v2/raw/data.csv.gz",
            gzip.compress(table.to_csv(index=False).encode("utf-8")),
        )
        archive.writestr("pcqm4m-v2/split_dict.pt", split_buffer.getvalue())


def test_local_ogb_contract_matches_published_examples() -> None:
    molecule = Chem.MolFromSmiles("Cl[C@H](/C=C/C)Br")
    assert atom_to_ogb_feature_vector(molecule.GetAtomWithIdx(1)) == [
        5, 2, 4, 5, 1, 0, 2, 0, 0
    ]
    assert bond_to_ogb_feature_vector(molecule.GetBondWithIdx(2)) == [1, 2, 0]


def test_rich_features_remove_the_radical_closed_shell_collision() -> None:
    common = dict(gap=1.0, role_code=0, radical=1)
    radical = _graphs_from_screen_row(
        SimpleNamespace(source_idx=1, smiles="[CH2]CCC[CH2]", **common)
    )
    neutral = _graphs_from_screen_row(
        SimpleNamespace(
            source_idx=2,
            smiles="CCCCC",
            gap=1.0,
            role_code=0,
            radical=0,
        )
    )
    assert torch.equal(radical["legacy"].x, neutral["legacy"].x)
    assert torch.equal(radical["legacy"].edge_attr, neutral["legacy"].edge_attr)
    assert not torch.equal(radical["ogb"].x, neutral["ogb"].x)


def test_categorical_edge_state_forward_backward() -> None:
    left = _graphs_from_screen_row(
        SimpleNamespace(
            source_idx=1,
            smiles="[CH2]CCC[CH2]",
            gap=1.0,
            role_code=0,
            radical=1,
        )
    )["ogb"]
    right = _graphs_from_screen_row(
        SimpleNamespace(
            source_idx=2,
            smiles="c1ccccc1",
            gap=2.0,
            role_code=0,
            radical=0,
        )
    )["ogb"]
    batch = Batch.from_data_list([left, right])
    model = CategoricalEdgeStateStructuralGPSWrapper(
        atom_feature_dims=ATOM_FEATURE_DIMS,
        bond_feature_dims=BOND_FEATURE_DIMS,
        hidden_channels=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        n_targets=1,
        rwse_dim=16,
        edge_state_channels=8,
    )
    output = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    )
    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert model.node_emb.embeddings[5].weight.grad is not None


def test_categorical_pair_gps_forward_backward() -> None:
    graphs = [
        _graphs_from_screen_row(
            SimpleNamespace(
                source_idx=index,
                smiles=smiles,
                gap=float(index),
                role_code=0,
                radical=radical,
            )
        )["ogb"]
        for index, smiles, radical in (
            (1, "[CH2]CCC[CH2]", 1),
            (2, "c1ccccc1", 0),
        )
    ]
    batch = Batch.from_data_list(graphs)
    model = CategoricalPairGPS2DWrapper(
        atom_feature_dims=ATOM_FEATURE_DIMS,
        bond_feature_dims=BOND_FEATURE_DIMS,
        atom_input_channels=12,
        bond_input_channels=8,
        hidden_channels=16,
        pair_channels=8,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        n_targets=1,
        rwse_dim=16,
        path_steps=3,
        triplet_rank=4,
    )
    output = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    )
    assert output.shape == (2, 1)
    output.square().mean().backward()
    assert model.atom_encoder.embeddings[0].weight.grad is not None
    assert model.bond_encoder.embeddings[0].weight.grad is not None
    assert model.rwse_encoder[0].weight.grad is not None
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_feature_screen_pair_gps_model_family() -> None:
    model = _model(
        "ogb",
        FeatureScreenConfig(
            model_family="pair_gps",
            hidden_channels=16,
            num_layers=1,
            num_heads=4,
            pair_channels=8,
            triplet_rank=4,
            atom_input_channels=12,
            bond_input_channels=8,
        ),
    )
    assert isinstance(model, CategoricalPairGPS2DWrapper)


def test_concat_categorical_encoder_preserves_field_channels() -> None:
    encoder = CategoricalConcatFeatureEncoder(
        (4, 5, 6),
        embedding_dim=8,
        field_channels=3,
    )
    features = torch.tensor([[1, 2, 3], [2, 1, 4]])
    output = encoder(features)
    assert output.shape == (2, 8)
    output.square().mean().backward()
    assert all(embedding.weight.grad is not None for embedding in encoder.embeddings)
    assert encoder.projection.weight.grad is not None


def test_categorical_edge_state_concat_project_forward_backward() -> None:
    graph = _graphs_from_screen_row(
        SimpleNamespace(
            source_idx=1,
            smiles="[CH2]CCC[CH2]",
            gap=1.0,
            role_code=0,
            radical=1,
        )
    )["ogb"]
    batch = Batch.from_data_list([graph])
    model = CategoricalEdgeStateStructuralGPSWrapper(
        atom_feature_dims=ATOM_FEATURE_DIMS,
        bond_feature_dims=BOND_FEATURE_DIMS,
        hidden_channels=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        n_targets=1,
        rwse_dim=16,
        edge_state_channels=8,
        categorical_encoder="concat_project",
        categorical_field_channels=4,
    )
    output = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    )
    assert output.shape == (1, 1)
    output.square().mean().backward()
    assert model.node_emb.projection.weight.grad is not None


def test_radical_context_is_exact_identity_for_closed_shell() -> None:
    graph = _graphs_from_screen_row(
        SimpleNamespace(
            source_idx=1,
            smiles="CCCCC",
            gap=1.0,
            role_code=0,
            radical=0,
        )
    )["ogb"]
    batch = Batch.from_data_list([graph])
    common = dict(
        atom_feature_dims=ATOM_FEATURE_DIMS,
        bond_feature_dims=BOND_FEATURE_DIMS,
        hidden_channels=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        n_targets=1,
        rwse_dim=16,
        edge_state_channels=8,
    )
    torch.manual_seed(7)
    control = CategoricalEdgeStateStructuralGPSWrapper(**common)
    torch.manual_seed(7)
    candidate = CategoricalRadicalContextEdgeStateStructuralGPSWrapper(
        radical_context_channels=4,
        **common,
    )
    inputs = (
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    )
    assert torch.equal(control(*inputs), candidate(*inputs))


def test_radical_context_receives_gradient() -> None:
    graph = _graphs_from_screen_row(
        SimpleNamespace(
            source_idx=1,
            smiles="[CH2]CCC[CH2]",
            gap=1.0,
            role_code=0,
            radical=1,
        )
    )["ogb"]
    batch = Batch.from_data_list([graph])
    model = CategoricalRadicalContextEdgeStateStructuralGPSWrapper(
        atom_feature_dims=ATOM_FEATURE_DIMS,
        bond_feature_dims=BOND_FEATURE_DIMS,
        hidden_channels=16,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        n_targets=1,
        rwse_dim=16,
        edge_state_channels=8,
        radical_context_channels=4,
    )
    output = model(
        batch.x,
        batch.edge_index,
        batch.edge_attr,
        batch.batch,
        batch.random_walk_pe,
    )
    output.square().mean().backward()
    assert model.radical_context.weight.grad[1:].abs().sum() > 0


def test_warmup_cosine_schedule_has_bounded_endpoints() -> None:
    factors = [
        _warmup_cosine_factor(
            epoch,
            max_epochs=20,
            warmup_epochs=2,
            minimum_factor=0.0025,
        )
        for epoch in range(20)
    ]
    assert factors[0] == 0.5
    assert factors[1] == 1.0
    assert factors[2] == 1.0
    assert factors[-1] == 0.0025
    assert all(left >= right for left, right in zip(factors[2:], factors[3:]))


def test_screen_rows_and_graphs_use_official_train_only(tmp_path: Path) -> None:
    archive = tmp_path / "pcqm.zip"
    _archive(archive)
    rows_dir = tmp_path / "rows"
    graph_dir = tmp_path / "graphs"
    manifest = prepare_feature_screen_rows(
        archive,
        rows_dir,
        train_rows=6,
        development_rows=2,
        seed=7,
        shard_rows=8,
    )
    assert manifest["train_rows"] == 6
    assert manifest["development_rows"] == 2
    assert manifest["official_valid_used"] is False
    selected = pd.read_csv(rows_dir / manifest["shards"][0]["path"])
    assert set(selected.source_idx) <= set(range(10))

    build_feature_screen_graph_shard(rows_dir, graph_dir, shard_index=0)
    acceptance = accept_feature_screen_graphs(
        rows_dir, graph_dir, graph_dir / "acceptance.json"
    )
    assert acceptance["status"] == "accepted"
    assert acceptance["official_valid_used"] is False
    assert acceptance["schemas"]["legacy"]["counts"] == {
        "train": 6,
        "development": 2,
    }
    assert acceptance["schemas"]["ogb"]["counts"] == {
        "train": 6,
        "development": 2,
    }
