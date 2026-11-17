from pathlib import Path

import torch

from molgap.distillation import (
    atomic_json_write,
    atomic_torch_save,
    load_teacher_targets,
)


def test_atomic_writes_and_target_merge(tmp_path: Path) -> None:
    parts = tmp_path / "parts"
    records = []
    for number, (start, end) in enumerate(((0, 3), (3, 5))):
        path = parts / f"part-{number:03d}.pt"
        atomic_torch_save(
            {
                "source_idx": torch.arange(start, end),
                "targets": torch.full((end - start, 3), float(number + 1)),
            },
            path,
        )
        records.append({"path": str(path), "start": start, "end": end})
    manifest = parts / "manifest.json"
    atomic_json_write(
        {
            "format": "molgap-distillation-target-manifest-v1",
            "complete": True,
            "rows": 5,
            "parts": records,
        },
        manifest,
    )
    merged = load_teacher_targets(manifest)
    assert merged.shape == (5, 3)
    assert torch.equal(merged[:3], torch.ones(3, 3))
    assert torch.equal(merged[3:], torch.full((2, 3), 2.0))
