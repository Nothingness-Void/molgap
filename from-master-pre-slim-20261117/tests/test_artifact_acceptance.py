import json

import pytest
import torch
from torch_geometric.data import Data

from molgap.artifact_acceptance import (
    accept_repaired_3d_graphs,
    align_gps_embeddings_to_reference,
    accept_schnet_pair,
    sha256_file,
)


def test_schnet_pair_requires_both_variants(tmp_path):
    paths = []
    for variant in ("primary", "augmented"):
        path = tmp_path / f"{variant}.json"
        path.write_text(
            json.dumps(
                {
                    "accepted": True,
                    "variant": variant,
                    "roles": {
                        role: {
                            "rows": 2,
                            "identity_sha256": "identity",
                            "targets_sha256": "targets",
                        }
                        for role in ("train", "validation", "test")
                    },
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    result = accept_schnet_pair(*paths)
    assert result["fusion_unblocked"]


def test_schnet_pair_rejects_identity_mismatch(tmp_path):
    paths = []
    for variant in ("primary", "augmented"):
        path = tmp_path / f"{variant}.json"
        path.write_text(
            json.dumps(
                {
                    "accepted": True,
                    "variant": variant,
                    "roles": {
                        role: {
                            "rows": 2,
                            "identity_sha256": f"{variant}-{role}",
                            "targets_sha256": "targets",
                        }
                        for role in ("train", "validation", "test")
                    },
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    result = accept_schnet_pair(*paths)
    assert not result["fusion_unblocked"]
    assert not result["branch_alignment"]


def test_align_gps_embeddings_to_reference(tmp_path):
    raw_path = tmp_path / "raw.pt"
    reference_path = tmp_path / "reference.pt"
    accepted_path = tmp_path / "accepted.pt"
    report_path = tmp_path / "report.json"
    torch.save(
        {
            "source_idx": torch.tensor([3, 1, 2]),
            "embeddings": torch.tensor([[30.0], [10.0], [20.0]]),
        },
        raw_path,
    )
    roles = {}
    for role, index in zip(("train", "validation", "test"), (1, 2, 3)):
        roles[role] = {
            "source_idx": torch.tensor([index]),
            "cid": [str(index + 100)],
            "embeddings": torch.zeros(1, 2),
            "targets": torch.tensor([[1.0, 2.0, 3.0]]),
        }
    torch.save(roles, reference_path)
    result = align_gps_embeddings_to_reference(
        raw_path,
        reference_path,
        accepted_path,
        report_path,
        expected_dim=1,
        expected_sha256=sha256_file(raw_path),
        name="gps",
    )
    accepted = torch.load(accepted_path, weights_only=False)
    assert result["accepted"]
    assert accepted["train"]["embeddings"].item() == 10.0
    assert accepted["validation"]["embeddings"].item() == 20.0
    assert accepted["test"]["embeddings"].item() == 30.0


def _write_repaired_3d_fixture(tmp_path, graphs, *, nested_sidecar=False):
    source_csv = tmp_path / "source.csv"
    source_csv.write_text(
        "cid,canonical_smiles,homo,lumo,gap\n"
        "101,CC,-5.0,-1.0,4.0\n"
        "102,CO,-6.0,-2.0,4.0\n",
        encoding="utf-8",
    )
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    graph_path = shard_dir / "graphs_0000000_0000002.pt"
    torch.save(graphs, graph_path)
    sidecar = graph_path.with_suffix(".json")
    if nested_sidecar:
        sidecar = shard_dir / "reports" / f"{graph_path.stem}.json"
        sidecar.parent.mkdir()
    graph_sha256 = sha256_file(graph_path)
    sidecar.write_text(
        json.dumps(
            {
                "status": "complete",
                "start": 0,
                "stop": 2,
                "requested": 2,
                "reused": 0,
                "built": len(graphs),
                "failed": 2 - len(graphs),
                "failure_source_idx": list(range(len(graphs), 2)),
                "graphs": len(graphs),
                "path": graph_path.name,
                "bytes": graph_path.stat().st_size,
                "sha256": graph_sha256,
            }
        ),
        encoding="utf-8",
    )
    return source_csv, shard_dir


def test_repaired_3d_acceptance_uses_source_idx_when_graph_identity_is_absent(
    tmp_path,
):
    graphs = [
        Data(
            source_idx=torch.tensor([0]),
            pos=torch.zeros(2, 3),
            y=torch.tensor([[-5.0, -1.0, 4.0]]),
        ),
        Data(
            source_idx=torch.tensor([1]),
            pos=torch.ones(2, 3),
            y=torch.tensor([[-6.0, -2.0, 4.0]]),
        ),
    ]
    source_csv, shard_dir = _write_repaired_3d_fixture(tmp_path, graphs)
    result = accept_repaired_3d_graphs(
        shard_dir,
        source_csv,
        tmp_path / "acceptance.json",
        expected_shards=1,
    )
    assert result["accepted"]
    assert result["unique_cid"] == 2
    assert result["graphs_with_embedded_cid"] == 0
    assert result["graphs_with_embedded_smiles"] == 0
    assert result["target_alignment"]


def test_repaired_3d_acceptance_supports_reports_subdirectory(tmp_path):
    graphs = [
        Data(
            source_idx=torch.tensor([0]),
            pos=torch.zeros(2, 3),
            y=torch.tensor([[-5.0, -1.0, 4.0]]),
        ),
        Data(
            source_idx=torch.tensor([1]),
            pos=torch.ones(2, 3),
            y=torch.tensor([[-6.0, -2.0, 4.0]]),
        ),
    ]
    source_csv, shard_dir = _write_repaired_3d_fixture(
        tmp_path, graphs, nested_sidecar=True
    )
    result = accept_repaired_3d_graphs(
        shard_dir,
        source_csv,
        tmp_path / "acceptance.json",
        expected_shards=1,
    )
    assert result["shards"][0]["sidecar_path"] == (
        "reports/graphs_0000000_0000002.json"
    )


def test_repaired_3d_acceptance_rejects_embedded_identity_mismatch(tmp_path):
    graphs = [
        Data(
            source_idx=torch.tensor([0]),
            cid="999",
            canonical_smiles="CC",
            pos=torch.zeros(2, 3),
            y=torch.tensor([[-5.0, -1.0, 4.0]]),
        ),
        Data(
            source_idx=torch.tensor([1]),
            pos=torch.ones(2, 3),
            y=torch.tensor([[-6.0, -2.0, 4.0]]),
        ),
    ]
    source_csv, shard_dir = _write_repaired_3d_fixture(tmp_path, graphs)
    with pytest.raises(ValueError, match="CID/source_idx mismatch"):
        accept_repaired_3d_graphs(
            shard_dir,
            source_csv,
            tmp_path / "acceptance.json",
            expected_shards=1,
        )
