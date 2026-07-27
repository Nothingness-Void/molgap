from __future__ import annotations

from pathlib import Path

import torch

from cli.train_three_gps_embedding_residual import (
    load_chunked,
)


def test_load_chunked_embedding_contract(tmp_path: Path):
    directory = tmp_path / "gps7"
    directory.mkdir()
    artifacts = []
    for part, (start, end) in enumerate(((0, 3), (3, 5))):
        path = directory / f"part-{part:03d}.pt"
        payload = {
            "format": "molgap-gps-embedding-part-v1",
            "name": "gps7",
            "embeddings": torch.ones(end - start, 4, dtype=torch.float16) * part,
            "predictions": torch.ones(end - start, 3) * part,
            "targets": torch.ones(end - start, 3),
            "source_idx": torch.arange(start, end),
            "model_sha256": "model",
            "graph_sha256": "graph",
        }
        torch.save(payload, path)
        import hashlib

        artifacts.append(
            {
                "part": part,
                "start": start,
                "end": end,
                "rows": end - start,
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    import json

    (directory / "completion.json").write_text(
        json.dumps(
            {
                "format": "molgap-gps-embedding-parts-v1",
                "complete": True,
                "name": "gps7",
                "rows": 5,
                "hidden_channels": 4,
                "model_sha256": "model",
                "artifacts": artifacts,
            }
        )
    )
    result = load_chunked(
        directory,
        name="gps7",
        rows=5,
        hidden_channels=4,
    )
    assert result["embeddings"].shape == (5, 4)
    assert result["predictions"].shape == (5, 3)
    assert torch.equal(result["source_idx"], torch.arange(5))
