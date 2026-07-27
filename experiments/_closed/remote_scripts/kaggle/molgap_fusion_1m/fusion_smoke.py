"""Kaggle GPU and private-input mount gate for the MolGap 1M fusion run."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch

REQUIRED = (
    "pyg_3d_graphs_etkdg_expansion_1m.pt",
    "gps_expansion_1m_embeddings.pt",
    "gps_expansion_1m_depth9_embeddings.pt",
    "schnet_expansion_1m_embeddings.pt",
)


def resolve_input_root() -> Path:
    """Find the mounted private dataset without coupling to its Kaggle slug."""
    candidates = []
    for root, _, names in os.walk("/kaggle/input"):
        if all(name in names for name in REQUIRED):
            candidates.append(Path(root))
    assert len(candidates) == 1, f"Expected one complete MolGap input mount, found: {candidates}"
    return candidates[0]


def inspect_embedding(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    source = payload["source_idx"]
    embedding = payload["embeddings"]
    return {
        "shape": list(embedding.shape),
        "dtype": str(embedding.dtype),
        "source_rows": int(source.numel()),
        "source_bounds": [int(source[0]), int(source[-1])],
        "strictly_sorted": bool(torch.all(source[1:] > source[:-1])),
    }


def main() -> None:
    input_parent = Path("/kaggle/input")
    mounts = {
        str(directory): sorted(entry.name for entry in directory.iterdir())
        for directory in input_parent.iterdir()
        if directory.is_dir()
    }
    try:
        input_root = resolve_input_root()
        missing = []
    except AssertionError:
        input_root = None
        missing = list(REQUIRED)

    record = {
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__,
        "mounts": mounts,
        "input_root": str(input_root) if input_root else None,
        "missing": missing,
        "input_files": {name: (input_root / name).stat().st_size for name in REQUIRED} if input_root else {},
    }
    if input_root:
        record["embedding_checks"] = {
            name: inspect_embedding(input_root / name)
            for name in REQUIRED
            if name.endswith("embeddings.pt")
        }
    Path("/kaggle/working/fusion_smoke.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
