from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def test_embedding_subset_alignment(tmp_path: Path):
    from scripts.phase8.experiments.train_hierarchical_2d3d_fusion import (
        load_embedding_subset,
    )

    source_idx = np.asarray([2, 5, 9], dtype=np.int64)
    targets = np.arange(9, dtype=np.float32).reshape(3, 3)
    directory = tmp_path / "parts"
    directory.mkdir()
    torch.save(
        {
            "source_idx": torch.tensor([0, 2, 5, 7]),
            "embeddings": torch.arange(4 * 4, dtype=torch.float32).reshape(4, 4),
            "targets": torch.from_numpy(
                np.asarray(
                    [
                    [99.0, 99.0, 99.0],
                    targets[0],
                    targets[1],
                    [99.0, 99.0, 99.0],
                    ],
                    dtype=np.float32,
                )
            ),
        },
        directory / "embeddings_000.pt",
    )
    torch.save(
        {
            "source_idx": torch.tensor([9, 10]),
            "embeddings": torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4),
            "targets": torch.from_numpy(
                np.asarray(
                    [
                    targets[2],
                    [99.0, 99.0, 99.0],
                    ],
                    dtype=np.float32,
                )
            ),
        },
        directory / "embeddings_001.pt",
    )
    embeddings, present, reports = load_embedding_subset(
        directory,
        source_idx,
        targets,
        expected_dim=4,
    )
    assert embeddings.shape == (3, 4)
    assert present.tolist() == [True, True, True]
    assert sum(report["selected_rows"] for report in reports) == 3
